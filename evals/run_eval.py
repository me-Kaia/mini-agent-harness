"""评测脚本：跑一组任务，每题可跑多次，统计正确率、pass^N、格式遵从率、token、耗时和终止原因。

用法：
  离线冒烟测试（不调用真实模型，只验证评测流程能跑通）：
    python evals/run_eval.py --provider mock
    python evals/run_eval.py --provider mock --tasks evals/gsm8k_50.jsonl --trials 3
  接真实模型（任意 OpenAI 兼容接口，Key 从 --key-env 指定的环境变量读取，不会打印）：
    export KIMI_API_KEY=...
    python evals/run_eval.py --provider openai --base-url https://api.moonshot.cn/v1 --model kimi-k2 \
        --key-env KIMI_API_KEY --tasks evals/gsm8k_50.jsonl --trials 3 --temperature 0.7

判定规则：任务正常结束（status=completed）且最终回答包含所有期望关键词，才算成功。
关键词怎么算"包含"见 judge.py：日期统一格式、数字按完整的数比较。
任务里写 "judge": "final_answer"（如 GSM8K）时，只比较最后一个"答案："后面的数；
没写"答案："的记为格式错误（FMT），单独统计，不算答错，但也不计入成功。

多次运行（--trials N）的汇总口径：
- 平均正确率：所有运行里判为 PASS 的比例（N 次 × 题数）；
- pass^N：N 次全部 PASS 的题目比例，衡量稳定性，N=1 时等于平均正确率；
- 格式遵从率：final_answer 模式下正常结束的运行里，写了"答案：数字"的比例；没有这类运行时为 None；
- 平均 token / 耗时 / 轮数：按每次运行平均。
temperature 为 0 时多次运行的结果几乎相同，pass^N 意义不大，建议多次运行时调高。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import Agent, AgentConfig, LLMResponse, OpenAICompatClient, ScriptedLLM, ToolCall, ToolRegistry, Tracer
from harness.builtin_tools import calculator_tool, make_file_tools
from judge import FAIL, FORMAT_ERROR, PASS, grade


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


def build_agent(args, api_key, tracer):
    reg = ToolRegistry()
    reg.register(calculator_tool())
    for t in make_file_tools(os.path.join(ROOT, "sample_workspace")):
        reg.register(t)
    if args.provider == "mock":
        llm = ScriptedLLM(mock_policy)
    else:
        llm = OpenAICompatClient(args.base_url, args.model, api_key, temperature=args.temperature)
    return Agent(llm, reg, AgentConfig(max_turns=args.max_turns, token_budget=args.token_budget), tracer=tracer)


def summarize(rows: list[dict], trials: int) -> dict:
    """rows 是每次运行一行（同一题有 trials 行），返回汇总指标。"""
    n = len(rows)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["id"]].append(r["verdict"] == PASS)
    verdicts = Counter(r["verdict"] for r in rows)
    graded = [r for r in rows if r["judge"] == "final_answer" and r["status"] == "completed"]
    compliance = sum(r["verdict"] != FORMAT_ERROR for r in graded) / len(graded) if graded else None
    return {
        "tasks": len(by_task), "trials": trials, "runs": n,
        "mean_accuracy": round(verdicts[PASS] / n, 3),
        "pass_all": round(sum(all(v) for v in by_task.values()) / len(by_task), 3),
        "format_compliance": None if compliance is None else round(compliance, 3),
        "verdict_counts": {v: verdicts[v] for v in (PASS, FAIL, FORMAT_ERROR)},
        "avg_tokens": round(sum(r["tokens"] for r in rows) / n, 1),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
        "avg_turns": round(sum(r["turns"] for r in rows) / n, 2),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "passes_per_task": {tid: sum(v) for tid, v in by_task.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["mock", "openai"], default="mock")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--key-env", default="LLM_API_KEY", help="从哪个环境变量读取 API Key（Key 本身不会打印）")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tasks", default=os.path.join(ROOT, "evals", "tasks.jsonl"))
    p.add_argument("--trials", type=int, default=1, help="每题跑几次")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--token-budget", type=int, default=30000)
    args = p.parse_args()
    if args.trials < 1:
        p.error("--trials 至少为 1")

    api_key = ""
    if args.provider == "openai":
        api_key = os.getenv(args.key_env, "")
        if not api_key:
            p.error(f"环境变量 {args.key_env} 未设置或为空")

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tracer = Tracer(os.path.join(ROOT, "runs", f"trace-{stamp}.jsonl"))
    tasks = [json.loads(line) for line in open(args.tasks, encoding="utf-8") if line.strip()]

    rows = []
    for task in tasks:
        mode = task.get("judge", "keywords")
        for k in range(1, args.trials + 1):
            agent = build_agent(args, api_key, tracer)
            start = time.monotonic()
            state = agent.run(task["goal"], run_id=f"{task['id']}-t{k}")
            verdict = grade(state.final_answer, task["expect"], mode) if state.status == "completed" else FAIL
            rows.append({"id": task["id"], "trial": k, "ok": verdict == PASS, "verdict": verdict, "judge": mode,
                         "status": state.status, "turns": state.turn, "tokens": state.tokens_used,
                         "seconds": round(time.monotonic() - start, 2), "answer": state.final_answer})
            tag = {PASS: "PASS", FAIL: "FAIL", FORMAT_ERROR: "FMT "}[verdict]
            print(f"[{tag}] {task['id']:<16} trial={k}/{args.trials} status={state.status:<14} "
                  f"turns={state.turn} tokens={state.tokens_used}")

    stats = summarize(rows, args.trials)
    summary = {"provider": args.provider, "model": args.model if args.provider == "openai" else "mock",
               "base_url": args.base_url if args.provider == "openai" else None,
               "temperature": args.temperature, "tasks_file": os.path.relpath(args.tasks, ROOT),
               **stats, "rows": rows}
    out = os.path.join(ROOT, "runs", f"eval-{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    vc, fc = stats["verdict_counts"], stats["format_compliance"]
    print(f"\n{stats['tasks']} 题 × {args.trials} 次 = {stats['runs']} 次运行")
    print(f"平均正确率 {stats['mean_accuracy']:.1%}  pass^{args.trials} {stats['pass_all']:.1%}  "
          f"格式遵从率 {'-' if fc is None else f'{fc:.1%}'}")
    print(f"答对 {vc[PASS]}  答错 {vc[FAIL]}  格式错误 {vc[FORMAT_ERROR]}  "
          f"平均 token {stats['avg_tokens']}  平均耗时 {stats['avg_seconds']}s  平均轮数 {stats['avg_turns']}  "
          f"终止原因 {stats['status_counts']}")
    print(f"详细结果：{out}")


if __name__ == "__main__":
    main()
