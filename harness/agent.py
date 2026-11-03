"""Agent 主循环。

每一轮：检查终止条件 → 必要时压缩上下文 → 调模型 → 没有工具调用就结束；有就逐个执行并回灌结果 → 存档。

终止条件（任一触发即停）：
1. 模型不再调用工具，给出最终回答        -> completed
2. 轮数达到上限                          -> max_turns
3. 累计 token 超过预算                   -> token_budget
4. 累计耗时超过上限                      -> timeout
5. 同一个工具 + 同样参数被反复调用        -> repeated_call（死循环防护）
6. 模型连续给出非法的工具调用             -> format_errors
7. 模型服务本身不可用                     -> llm_error

为什么不只靠 max_turns：轮数上限只能兜底，等它触发时可能已经白白烧了很多 token；
重复调用检测能更早发现"卡住了"，token 和耗时预算则直接控制成本。
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from .context import ContextManager
from .llm import LLMError
from .state import Checkpointer, RunState, Tracer
from .tools import ToolRegistry


@dataclass
class AgentConfig:
    max_turns: int = 12
    token_budget: int = 50_000
    max_seconds: float = 300.0
    max_repeat: int = 2          # 同一指纹最多执行几次，第 max_repeat+1 次直接终止
    max_format_errors: int = 3   # 非法工具调用（未知工具 / 坏 JSON / 参数不合法）累计上限


SYSTEM_PROMPT = (
    "你是一个会使用工具的助手。需要信息时调用工具；工具返回错误时，根据错误信息修正参数或换一种方法。"
    "拿到足够信息后，直接用中文给出最终答案，不要再调用工具。"
)


def fingerprint(name: str, raw_args: str) -> str:
    """把参数解析后按 key 排序再哈希，这样 {"a":1,"b":2} 和 {"b":2,"a":1} 视为同一次调用。"""
    try:
        normalized = json.dumps(json.loads(raw_args or "{}"), sort_keys=True, ensure_ascii=False)
    except json.JSONDecodeError:
        normalized = raw_args
    return hashlib.sha1(f"{name}:{normalized}".encode("utf-8")).hexdigest()[:12]


class Agent:
    def __init__(self, llm, tools: ToolRegistry, config: Optional[AgentConfig] = None,
                 context: Optional[ContextManager] = None, checkpointer: Optional[Checkpointer] = None,
                 tracer: Optional[Tracer] = None, system_prompt: str = SYSTEM_PROMPT):
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()
        self.context = context or ContextManager()
        self.checkpointer = checkpointer
        self.tracer = tracer or Tracer(None)
        self.system_prompt = system_prompt

    # ---------- 对外接口 ----------
    def run(self, goal: str, run_id: Optional[str] = None) -> RunState:
        state = RunState(run_id=run_id or uuid.uuid4().hex[:8],
                         messages=[{"role": "system", "content": self.system_prompt},
                                   {"role": "user", "content": goal}])
        return self._loop(state)

    def resume(self, run_id: str) -> RunState:
        if not self.checkpointer:
            raise ValueError("没有配置 Checkpointer，无法恢复")
        state = self.checkpointer.load(run_id)
        if state.status != "running":
            return state  # 已经结束的任务不重复执行
        self.tracer.log(state.run_id, "resume", turn=state.turn)
        return self._loop(state)

    # ---------- 主循环 ----------
    def _loop(self, state: RunState) -> RunState:
        cfg = self.config
        start = time.monotonic() - state.elapsed_s  # 恢复时接着之前的耗时算
        while True:
            state.elapsed_s = time.monotonic() - start
            reason = self._limit_reached(state)
            if reason:
                return self._finish(state, reason)

            state.messages, compacted = self.context.maybe_compact(state.messages)
            if compacted:
                self.tracer.log(state.run_id, "compaction", messages=len(state.messages))

            try:
                resp = self.llm.chat(state.messages, tools=self.tools.schemas())
            except LLMError as e:
                return self._finish(state, "llm_error", str(e))
            state.turn += 1
            state.tokens_used += resp.prompt_tokens + resp.completion_tokens
            self.tracer.log(state.run_id, "llm_call", turn=state.turn, latency_s=round(resp.latency_s, 3),
                            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                            tool_calls=[tc.name for tc in resp.tool_calls])

            assistant = {"role": "assistant", "content": resp.content}
            if resp.tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in resp.tool_calls]
            state.messages.append(assistant)

            if not resp.tool_calls:
                state.final_answer = resp.content
                return self._finish(state, "completed")

            for i, tc in enumerate(resp.tool_calls):
                fp = fingerprint(tc.name, tc.arguments)
                state.call_counts[fp] = state.call_counts.get(fp, 0) + 1
                if state.call_counts[fp] > cfg.max_repeat:
                    # 给这次调用和同一条回复里剩下的调用都补上 tool 消息再退出，保证存档里的消息序列仍然合法
                    state.messages.append({"role": "tool", "tool_call_id": tc.id,
                                           "content": "已终止：重复调用同一工具和参数。"})
                    for rest in resp.tool_calls[i + 1:]:
                        state.messages.append({"role": "tool", "tool_call_id": rest.id,
                                               "content": "已终止：同一回复中更早的调用触发了重复调用上限，本调用未执行。"})
                    return self._finish(state, "repeated_call", f"{tc.name} 重复调用")

                result = self.tools.execute(tc.name, tc.arguments)
                if result.error_type in ("unknown_tool", "bad_json", "bad_args"):
                    state.format_errors += 1
                self.tracer.log(state.run_id, "tool_call", turn=state.turn, tool=tc.name, ok=result.ok,
                                error_type=result.error_type, attempts=result.attempts,
                                latency_s=round(result.latency_s, 3))
                state.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result.content})

            if self.checkpointer:
                self.checkpointer.save(state)

    def _limit_reached(self, state: RunState) -> str:
        cfg = self.config
        if state.turn >= cfg.max_turns:
            return "max_turns"
        if state.tokens_used >= cfg.token_budget:
            return "token_budget"
        if state.elapsed_s >= cfg.max_seconds:
            return "timeout"
        if state.format_errors >= cfg.max_format_errors:
            return "format_errors"
        return ""

    def _finish(self, state: RunState, status: str, detail: str = "") -> RunState:
        state.status = status
        self.tracer.log(state.run_id, "stop", status=status, detail=detail, turn=state.turn,
                        tokens=state.tokens_used, elapsed_s=round(state.elapsed_s, 3))
        if self.checkpointer:
            self.checkpointer.save(state)
        return state
