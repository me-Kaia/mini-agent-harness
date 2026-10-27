"""LLM 客户端。

- OpenAICompatClient：兼容 OpenAI Chat Completions 格式，可接 DeepSeek API 或本地 llama.cpp server。
- ScriptedLLM：按脚本返回固定回复，用于离线单元测试，不需要网络和 API Key。
"""
from __future__ import annotations

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
    latency_s: float = 0.0


class LLMError(Exception):
    """模型调用失败（网络、限流、服务端错误）。"""


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_s: float = 120.0,
                 max_retries: int = 2, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data,
                                             headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return self._parse(payload, time.monotonic() - start)
            except urllib.error.HTTPError as e:
                last_err = e
                # 4xx（除 429 限流外）是请求本身有问题，重试没有意义
                if 400 <= e.code < 500 and e.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
            time.sleep(min(2 ** attempt, 8))  # 指数退避
        raise LLMError(f"LLM 调用失败: {last_err}")

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
