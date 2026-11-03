"""评测脚本：跑一组任务，统计成功率、轮数、token、耗时和终止原因。

用法：
  离线冒烟测试（不调用真实模型，只验证评测流程能跑通）：
    python evals/run_eval.py --provider mock
  接真实模型（DeepSeek API 或本地 llama.cpp server，见 README）：
    python evals/run_eval.py --provider openai --base-url https://api.deepseek.com --model deepseek-chat --api-key $KEY

判定规则：任务正常结束（status=completed）且最终回答包含所有期望关键词，才算成功。
关键词怎么算"包含"见 judge.py：日期统一格式、数字按完整的数比较。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import Agent, AgentConfig, LLMResponse, OpenAICompatClient, ScriptedLLM, ToolCall, ToolRegistry, Tracer
from harness.builtin_tools import calculator_tool, make_file_tools
from judge import judge


def mock_policy(messages):
    """一个很笨的规则"模型"：只会做纯算式题，其余一律回答不知道。仅用于验证评测流程。"""
    goal = messages[1]["content"]
    last = messages[-1]
    if last["role"] == "tool":
        return LLMResponse(content=f"结果是 {last['content']}", prompt_tokens=50, completion_tokens=10)
    m = re.search(r"([\d\.\s\+\-\*/\(\)]{3,})", goal)
    if m:
        return LLMResponse(tool_calls=[ToolCall("c1", "calculator", json.dumps({"expression": m.group(1).strip()}))],
                           prompt_tokens=50, completion_tokens=10)
    return LLMResponse(content="我不知道。", prompt_tokens=50, completion_tokens=10)


def build_agent(args, tracer):
    reg = ToolRegistry()
    reg.register(calculator_tool())
    for t in make_file_tools(os.path.join(ROOT, "sample_workspace")):
        reg.register(t)
    if args.provider == "mock":
        llm = ScriptedLLM(mock_policy)
    else:
        llm = OpenAICompatClient(args.base_url, args.model, args.api_key or os.getenv("LLM_API_KEY", ""))
    return Agent(llm, reg, AgentConfig(max_turns=args.max_turns, token_budget=args.token_budget), tracer=tracer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["mock", "openai"], default="mock")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--api-key", default="")
    p.add_argument("--tasks", default=os.path.join(ROOT, "evals", "tasks.jsonl"))
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--token-budget", type=int, default=30000)
    args = p.parse_args()

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tracer = Tracer(os.path.join(ROOT, "runs", f"trace-{stamp}.jsonl"))
    tasks = [json.loads(line) for line in open(args.tasks, encoding="utf-8") if line.strip()]

    rows, statuses = [], Counter()
    for task in tasks:
        agent = build_agent(args, tracer)
        start = time.monotonic()
        state = agent.run(task["goal"], run_id=task["id"])
        ok = state.status == "completed" and judge(state.final_answer, task["expect"])
        statuses[state.status] += 1
        rows.append({"id": task["id"], "ok": ok, "status": state.status, "turns": state.turn,
                     "tokens": state.tokens_used, "seconds": round(time.monotonic() - start, 2),
                     "answer": state.final_answer})
        print(f"[{'PASS' if ok else 'FAIL'}] {task['id']:<16} status={state.status:<14} "
              f"turns={state.turn} tokens={state.tokens_used}")

    n = len(rows)
    summary = {"provider": args.provider, "model": args.model if args.provider == "openai" else "mock",
               "tasks": n, "success_rate": round(sum(r["ok"] for r in rows) / n, 3),
               "avg_turns": round(sum(r["turns"] for r in rows) / n, 2),
               "avg_tokens": round(sum(r["tokens"] for r in rows) / n, 1),
               "status_counts": dict(statuses), "rows": rows}
    out = os.path.join(ROOT, "runs", f"eval-{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n成功率 {summary['success_rate']:.0%}  平均轮数 {summary['avg_turns']}  "
          f"平均 token {summary['avg_tokens']}  终止原因 {dict(statuses)}")
    print(f"详细结果：{out}")


if __name__ == "__main__":
    main()
