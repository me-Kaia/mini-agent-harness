"""上下文管理与压缩。

核心问题：上下文超长时要删掉一部分历史，但不能把"助手发起的工具调用"和"对应的工具结果"拆开——
拆开后，模型 API 会因为 tool 消息找不到对应的 tool_call 而报错，模型也会读到没头没尾的结果。

做法：先把消息切成"单元"，再以单元为最小粒度做取舍。
- system 消息、第一条用户消息（任务目标）：钉住，永远保留。
- 一条 assistant 消息 + 紧随其后的所有 tool 消息：合成一个单元，要么整体保留，要么整体被摘要。
- 最近 K 个单元：原样保留，保证模型知道"刚刚发生了什么"。
- 中间的单元：压成一条摘要消息。默认用确定性规则生成摘要（不额外调用模型）；
  也可以传入 summarizer 函数，改用模型生成摘要（效果更好，但多一次调用、有成本）。
"""
from __future__ import annotations

import json
from typing import Callable, Optional


def estimate_tokens(messages: list[dict]) -> int:
    """粗略估算：中英文混合大约 1 token ≈ 2~4 个字符，这里取 3。真实场景可换成模型的 tokenizer。"""
    total = 0
    for m in messages:
        total += 4 + len(m.get("content") or "") // 3
        for tc in m.get("tool_calls") or []:
            total += len(json.dumps(tc, ensure_ascii=False)) // 3
    return total


def split_units(messages: list[dict]) -> list[list[dict]]:
    units: list[list[dict]] = []
    for m in messages:
        if m["role"] == "tool" and units and units[-1][0]["role"] == "assistant":
            units[-1].append(m)
        else:
            units.append([m])
    return units


def default_summary(units: list[list[dict]], max_chars: int = 160) -> str:
    lines = []
    for unit in units:
        head = unit[0]
        if head["role"] == "assistant":
            for tc in head.get("tool_calls") or []:
                fn = tc["function"]
                lines.append(f"- 调用 {fn['name']}({fn['arguments'][:80]})")
            for t in unit[1:]:
                lines.append(f"  结果：{(t.get('content') or '')[:max_chars]}")
            if head.get("content"):
                lines.append(f"- 助手：{head['content'][:max_chars]}")
        else:
            lines.append(f"- {head['role']}：{(head.get('content') or '')[:max_chars]}")
    return "[以下是较早对话的摘要]\n" + "\n".join(lines)


class ContextManager:
    def __init__(self, limit_tokens: int = 6000, keep_recent_units: int = 4,
                 summarizer: Optional[Callable[[list[list[dict]]], str]] = None):
        self.limit_tokens = limit_tokens
        self.keep_recent_units = keep_recent_units
        self.summarizer = summarizer or default_summary

    def maybe_compact(self, messages: list[dict]) -> tuple[list[dict], bool]:
        if estimate_tokens(messages) <= self.limit_tokens:
            return messages, False
        units = split_units(messages)
        # 钉住开头的 system 和第一条 user（任务目标）
        pinned = 0
        while pinned < len(units) and units[pinned][0]["role"] in ("system",):
            pinned += 1
        if pinned < len(units) and units[pinned][0]["role"] == "user":
            pinned += 1
        middle_end = max(pinned, len(units) - self.keep_recent_units)
        middle = units[pinned:middle_end]
        if not middle:
            return messages, False  # 没有可压缩的部分（最近的单元本身就很长时会出现，见 README 局限）
        summary = {"role": "user", "content": self.summarizer(middle)}
        new_units = units[:pinned] + [[summary]] + units[middle_end:]
        return [m for u in new_units for m in u], True


def check_tool_pairs(messages: list[dict]) -> bool:
    """双向校验，用于测试和调试：
    - 每条 tool 消息都能找到紧邻的 assistant 发起的同 id 调用；
    - 每条 assistant 消息里的每个调用，在下一条非 tool 消息之前都有对应的 tool 消息。
    """
    open_ids: set[str] = set()  # 当前 assistant 发起、还没收到结果的调用
    for m in messages:
        if m["role"] == "tool":
            if m.get("tool_call_id") not in open_ids:
                return False
            open_ids.discard(m["tool_call_id"])
            continue
        if open_ids:  # 遇到下一条非 tool 消息时，上一批调用必须都已有结果
            return False
        if m["role"] == "assistant":
            open_ids = {tc["id"] for tc in m.get("tool_calls") or []}
    return not open_ids
