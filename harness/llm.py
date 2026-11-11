"""LLM 客户端。

- OpenAICompatClient：兼容 OpenAI Chat Completions 格式，可接 DeepSeek API 或本地 llama.cpp server。
- ScriptedLLM：按脚本返回固定回复，用于离线单元测试，不需要网络和 API Key。
"""
from __future__ import annotations

import email.utils
import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 模型返回的原始字符串，可能不是合法 JSON，由 harness 负责校验


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0  # 实际请求耗时（各次尝试之和），不含退避和限速等待
    retries: int = 0        # 这次调用重试了几次
    wait_s: float = 0.0     # 退避和限速等待的总时间


class LLMError(Exception):
    """模型调用失败（网络、限流、服务端错误）。带上失败前的重试次数和等待时间，方便统计。"""

    def __init__(self, message: str, retries: int = 0, wait_s: float = 0.0, latency_s: float = 0.0):
        super().__init__(message)
        self.retries = retries
        self.wait_s = wait_s
        self.latency_s = latency_s


MAX_RETRY_AFTER_S = 300.0         # Retry-After 最多等这么久，防止异常的头让评测挂住
RATE_LIMIT_BACKOFF_S = (10, 60)   # 429 没有 Retry-After 时：从 10 秒开始翻倍，最多 60 秒
NETWORK_BACKOFF_S = (1, 8)        # 网络抖动、5xx：从 1 秒开始翻倍，最多 8 秒


def parse_retry_after(value: Optional[str], now: Optional[float] = None) -> Optional[float]:
    """解析 Retry-After：秒数或 HTTP 日期。无法解析或为负数返回 None；日期已过返回 0；超过上限按上限。"""
    if not value or not value.strip():
        return None
    value = value.strip()
    try:
        seconds = float(value)
        if seconds < 0:
            return None
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        seconds = max(0.0, when.timestamp() - (time.time() if now is None else now))
    return min(seconds, MAX_RETRY_AFTER_S)


def backoff(attempt: int, base_cap: tuple[float, float]) -> float:
    base, cap = base_cap
    return min(base * 2 ** attempt, cap)


class RateLimiter:
    """保证相邻两次请求的开始时间至少间隔 min_interval_s 秒，主动放慢速度、减少 429。
    评测时每个 trial 新建一个客户端，所以同一个 RateLimiter 要在客户端之间共享。"""

    def __init__(self, min_interval_s: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.sleep = sleep
        self._last: Optional[float] = None

    def wait(self) -> float:
        """必要时等待，返回等了多少秒，并把当前时间记为最近一次请求的开始时间。"""
        waited = 0.0
        if self.min_interval_s > 0 and self._last is not None:
            gap = self.min_interval_s - (self.clock() - self._last)
            if gap > 0:
                self.sleep(gap)
                waited = gap
        self._last = self.clock()
        return waited


class OpenAICompatClient:
    """temperature / top_p 传 None 表示不发送该字段，由服务端用默认值（有的模型固定了取值，传了反而报错）。
    extra_body 是要原样合并进请求体的额外字段，比如 Kimi 关闭思考：{"thinking": {"type": "disabled"}}。
    rate_limiter 用来控制请求间隔；重试策略：429 按 Retry-After 等，没有就 10→60 秒退避；
    网络错误和 5xx 按 1→8 秒退避；其余 4xx 不重试。
    """

    # extra_body 不能覆盖的字段：核心字段由客户端管理，采样参数只能走专门的参数，避免两处设置不一致
    RESERVED_FIELDS = ("model", "messages", "tools", "temperature", "top_p")

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_s: float = 120.0,
                 max_retries: int = 2, temperature: Optional[float] = 0.0, top_p: Optional[float] = None,
                 extra_body: Optional[dict] = None, rate_limiter: Optional[RateLimiter] = None):
        conflicts = [k for k in (extra_body or {}) if k in self.RESERVED_FIELDS]
        if conflicts:
            raise ValueError(f"extra_body 不能包含 {', '.join(conflicts)}")
        if max_retries < 0:
            raise ValueError("max_retries 不能为负数")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.extra_body = json.loads(json.dumps(extra_body or {}))  # 深拷贝，外部修改不影响已创建的客户端
        self.rate_limiter = rate_limiter

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
        body = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if tools:
            body["tools"] = tools
        body.update(self.extra_body)
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        request_s = wait_s = 0.0
        attempt = 0
        while True:
            if self.rate_limiter:
                wait_s += self.rate_limiter.wait()
            start = time.monotonic()
            delay: Optional[float] = None  # None 表示不再重试
            try:
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data,
                                             headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                request_s += time.monotonic() - start
                result = self._parse(payload, request_s)
                result.retries, result.wait_s = attempt, wait_s
                return result
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 429:
                    delay = parse_retry_after(e.headers.get("Retry-After") if e.headers else None)
                    if delay is None:
                        delay = backoff(attempt, RATE_LIMIT_BACKOFF_S)
                elif not 400 <= e.code < 500:  # 5xx 是服务端临时问题；其余 4xx 是请求本身有问题，重试没有意义
                    delay = backoff(attempt, NETWORK_BACKOFF_S)
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
                # 服务端断开连接（RemoteDisconnected）、连接被重置这类错误 urlopen 不会包成 URLError，也按临时错误重试
                last_err = e
                delay = backoff(attempt, NETWORK_BACKOFF_S)
            request_s += time.monotonic() - start
            if delay is None or attempt >= self.max_retries:
                break
            time.sleep(delay)
            wait_s += delay
            attempt += 1
        raise LLMError(f"LLM 调用失败: {last_err}", retries=attempt, wait_s=wait_s, latency_s=request_s)

    @staticmethod
    def _parse(payload: dict, latency: float) -> LLMResponse:
        msg = payload["choices"][0]["message"]
        calls = [
            ToolCall(id=tc.get("id", f"call_{i}"), name=tc["function"]["name"],
                     arguments=tc["function"].get("arguments") or "{}")
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        usage = payload.get("usage") or {}
        return LLMResponse(content=msg.get("content") or "", tool_calls=calls,
                           prompt_tokens=usage.get("prompt_tokens", 0),
                           completion_tokens=usage.get("completion_tokens", 0),
                           latency_s=latency)


class ScriptedLLM:
    """按顺序返回预设回复；也可以传入函数，根据当前消息动态决定回复。"""

    def __init__(self, script: list[LLMResponse] | Callable[[list[dict]], LLMResponse]):
        self.script = script
        self.calls = 0

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
        self.calls += 1
        if callable(self.script):
            return self.script(messages)
        if not self.script:
            raise LLMError("脚本已用完")
        return self.script.pop(0)
