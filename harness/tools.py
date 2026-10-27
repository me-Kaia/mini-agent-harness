"""工具注册与执行。

设计取舍：
- 参数先校验再执行：模型给的参数不可信，非法参数直接把错误信息回灌给模型，让它自己改，而不是让工具崩溃。
- 只有幂等工具才自动重试：读文件可以重试，"发邮件""写数据库"这类有副作用的工具重试可能导致重复执行。
- 超时用线程实现：简单但有局限——Python 线程无法被强制终止，超时后线程仍在后台跑。
  生产环境应把工具放到子进程或沙箱里执行，超时后直接杀进程。
- 输出过长时保留头尾、截掉中间：日志、Dump 这类输出的关键信息往往在开头（摘要）和结尾（错误）。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable


class TransientToolError(Exception):
    """可重试的临时错误（如网络抖动）。"""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema 子集：type=object, properties, required
    func: Callable[..., Any]
    idempotent: bool = True
    timeout_s: float = 10.0

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


@dataclass
class ToolResult:
    ok: bool
    content: str
    error_type: str = ""  # unknown_tool / bad_json / bad_args / timeout / transient / exception
    attempts: int = 1
    latency_s: float = 0.0


_TYPE_MAP = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
             "array": list, "object": dict}


def validate_args(schema: dict, args: Any) -> list[str]:
    """最小化的参数校验：对象类型、必填字段、字段类型、枚举值。"""
    if not isinstance(args, dict):
        return ["参数必须是 JSON 对象"]
    errors = []
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            errors.append(f"缺少必填参数 '{key}'")
    for key, value in args.items():
        if key not in props:
            errors.append(f"未知参数 '{key}'")
            continue
        spec = props[key]
        expected = _TYPE_MAP.get(spec.get("type", ""))
        # bool 是 int 的子类，要单独排除
        if expected and (not isinstance(value, expected) or
                         (spec.get("type") in ("integer", "number") and isinstance(value, bool))):
            errors.append(f"参数 '{key}' 应为 {spec['type']}，实际是 {type(value).__name__}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"参数 '{key}' 只能取 {spec['enum']}")
    return errors


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    cut = len(text) - 2 * keep
    return f"{text[:keep]}\n...[已截断 {cut} 个字符]...\n{text[-keep:]}"


class ToolRegistry:
    def __init__(self, max_output_chars: int = 4000, max_retries: int = 2, backoff_s: float = 0.5):
        self.tools: dict[str, Tool] = {}
        self.max_output_chars = max_output_chars
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._pool = ThreadPoolExecutor(max_workers=4)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def execute(self, name: str, raw_args: str) -> ToolResult:
        start = time.monotonic()
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(False, f"错误：不存在工具 '{name}'，可用工具：{list(self.tools)}",
                              "unknown_tool")
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as e:
            return ToolResult(False, f"错误：参数不是合法 JSON（{e}），请重新生成参数。", "bad_json")
        errors = validate_args(tool.parameters, args)
        if errors:
            return ToolResult(False, "错误：参数校验失败：" + "；".join(errors), "bad_args")

        max_attempts = 1 + (self.max_retries if tool.idempotent else 0)
        last_type, last_msg = "", ""
        for attempt in range(1, max_attempts + 1):
            future = self._pool.submit(tool.func, **args)
            try:
                output = future.result(timeout=tool.timeout_s)
                return ToolResult(True, truncate(str(output), self.max_output_chars),
                                  attempts=attempt, latency_s=time.monotonic() - start)
            except FutureTimeout:
                last_type, last_msg = "timeout", f"工具执行超过 {tool.timeout_s}s"
            except TransientToolError as e:
                last_type, last_msg = "transient", str(e)
            except Exception as e:  # 工具内部的确定性错误，重试也没用
                return ToolResult(False, f"错误：工具执行失败：{type(e).__name__}: {e}", "exception",
                                  attempts=attempt, latency_s=time.monotonic() - start)
            if attempt < max_attempts:
                time.sleep(self.backoff_s * 2 ** (attempt - 1))
        return ToolResult(False, f"错误：{last_msg}（已尝试 {max_attempts} 次）", last_type,
                          attempts=max_attempts, latency_s=time.monotonic() - start)
