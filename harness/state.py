"""运行状态、断点存档与追踪日志。

- 每完成一轮（一次模型调用 + 对应工具执行）就存一次档，服务重启后可以从最近一轮继续。
- 写文件用"先写临时文件、再原子替换"，避免写到一半进程崩溃导致存档损坏。
- 存档点选在"一轮结束"而不是"工具执行中"：工具执行到一半的状态很难安全恢复，
  所以有副作用的工具需要自己保证幂等（例如带请求 ID 去重），见 README 局限。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field


@dataclass
class RunState:
    run_id: str
    messages: list[dict]
    turn: int = 0
    tokens_used: int = 0
    elapsed_s: float = 0.0       # 运行耗时，不含模型调用的退避和限速等待
    llm_retries: int = 0         # 模型调用累计重试次数
    wait_s: float = 0.0          # 模型调用累计退避和限速等待时间
    format_errors: int = 0
    call_counts: dict[str, int] = field(default_factory=dict)  # 工具调用指纹 -> 次数
    status: str = "running"  # running / completed / max_turns / token_budget / timeout / ...
    final_answer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunState":
        return cls(**d)


class Checkpointer:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def path(self, run_id: str) -> str:
        return os.path.join(self.directory, f"{run_id}.json")

    def save(self, state: RunState) -> None:
        path = self.path(state.run_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # 原子替换

    def load(self, run_id: str) -> RunState:
        with open(self.path(run_id), encoding="utf-8") as f:
            return RunState.from_dict(json.load(f))


class Tracer:
    """把每个事件追加写成一行 JSON，方便事后用 grep / jq / pandas 分析耗时和失败原因。"""

    def __init__(self, path: str | None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, run_id: str, event: str, **data) -> None:
        if not self.path:
            return
        record = {"ts": round(time.time(), 3), "run_id": run_id, "event": event, **data}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
