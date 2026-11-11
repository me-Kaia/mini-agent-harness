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
- 平均 token / 耗时 / 轮数：按每次有效运行平均；
- 调用失败（llm_error，模型接口没返回）单独记为 error，列出次数和失败率，
  不进正确率、pass^N 和平均值的分母；某题所有 trial 都调用失败时，这题不计入 pass^N。
temperature 为 0 时多次运行的结果几乎相同，pass^N 意义不大，建议多次运行时调高。

标注修正：--label-issues（默认 evals/gsm8k_label_issues.jsonl）列出有问题的题目，
汇总同时给出原始口径（完全按原标准答案判）和修正后口径（relabel 的题按修正答案判，exclude 的题不计入），
并在结果文件的 label_corrections 里写明修正了哪几题、为什么。
已有结果可以不调用模型直接重判：python evals/run_eval.py --regrade runs/eval-xxx.json
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

from harness import Agent, AgentConfig, LLMResponse, OpenAICompatClient, RateLimiter, ScriptedLLM, ToolCall, ToolRegistry, Tracer
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


def build_agent(args, api_key, tracer, rate_limiter=None):
    reg = ToolRegistry()
    reg.register(calculator_tool())
    for t in make_file_tools(os.path.join(ROOT, "sample_workspace")):
        reg.register(t)
    if args.provider == "mock":
        llm = ScriptedLLM(mock_policy)
    else:
        llm = OpenAICompatClient(args.base_url, args.model, api_key, temperature=args.temperature,
                                 top_p=args.top_p, extra_body=args.extra_body, max_retries=args.max_retries,
                                 rate_limiter=rate_limiter)
    return Agent(llm, reg, AgentConfig(max_turns=args.max_turns, token_budget=args.token_budget), tracer=tracer)


def parse_extra_body(text: str | None) -> dict:
    """--extra-body 的值：JSON 对象字符串，为空时返回 {}。"""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"--extra-body 不是合法 JSON：{e}") from None
    if not isinstance(value, dict):
        raise ValueError("--extra-body 必须是 JSON 对象，比如 '{\"thinking\": {\"type\": \"disabled\"}}'")
    return value


def optional_float(text: str) -> float | None:
    """--temperature / --top-p 的值：数字，或 none 表示不发送这个字段。"""
    return None if text.lower() == "none" else float(text)


def tasks_file_label(path: str) -> str:
    """结果文件里记录的任务文件路径：项目内用相对路径；不在同一个盘上（Windows）时用绝对路径。"""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return os.path.abspath(path)


ERROR = "error"                 # 调用失败：模型接口没返回结果，与模型答得好坏无关
CALL_FAILURE_STATUSES = {"llm_error"}
DEFAULT_LABEL_ISSUES = os.path.join(ROOT, "evals", "gsm8k_label_issues.jsonl")


def load_label_issues(path: str | None) -> dict[str, dict]:
    """读取标注问题清单：action=relabel 时用 expect 重判，action=exclude 时修正后口径不计入。文件不存在返回空。"""
    if not path or not os.path.exists(path):
        return {}
    issues = {}
    with open(path, encoding="utf-8") as f:
        for line in filter(str.strip, f):
            issue = json.loads(line)
            if issue.get("action") not in ("relabel", "exclude"):
                raise ValueError(f"{issue.get('id')}：action 只能是 relabel 或 exclude")
            if issue["action"] == "relabel" and not issue.get("expect"):
                raise ValueError(f"{issue['id']}：relabel 需要给出修正后的 expect")
            issues[issue["id"]] = issue
    return issues


def judge_run(task: dict, status: str, answer: str, issue: dict | None = None) -> tuple[str, str | None]:
    """返回 (原始判定, 修正后判定)。修正后判定为 None 表示这题在修正后口径里不计入。"""
    mode = task.get("judge", "keywords")

    def verdict(expect):
        if status in CALL_FAILURE_STATUSES:
            return ERROR
        if status != "completed":
            return FAIL
        return grade(answer or "", expect, mode)

    raw = verdict(task["expect"])
    if issue is None:
        return raw, raw
    if issue["action"] == "exclude":
        return raw, None
    return raw, verdict(issue["expect"])


def summarize(rows: list[dict], trials: int, key: str = "verdict") -> dict:
    """rows 是每次运行一行（同一题有 trials 行），按 rows[key] 的判定汇总。

    key 为 None 的行（修正后口径里被排除的题）完全不计入；
    调用失败（ERROR）计入运行次数和失败率，但不进正确率、pass^N、平均 token/耗时的分母。
    """
    counted = [r for r in rows if r.get(key) is not None]
    valid = [r for r in counted if r[key] != ERROR]
    by_task = defaultdict(list)
    for r in valid:
        by_task[r["id"]].append(r[key] == PASS)
    verdicts = Counter(r[key] for r in counted)
    graded = [r for r in valid if r["judge"] == "final_answer" and r["status"] == "completed"]

    def ratio(a, b, digits=3):
        return round(a / b, digits) if b else None

    return {
        "tasks": len({r["id"] for r in counted}), "trials": trials, "runs": len(counted),
        "mean_accuracy": ratio(verdicts[PASS], len(valid)),
        "pass_all": ratio(sum(all(v) for v in by_task.values()), len(by_task)),
        "format_compliance": ratio(sum(r[key] != FORMAT_ERROR for r in graded), len(graded)),
        "error_count": verdicts[ERROR], "error_rate": ratio(verdicts[ERROR], len(counted)),
        "verdict_counts": {v: verdicts[v] for v in (PASS, FAIL, FORMAT_ERROR, ERROR)},
        "avg_tokens": ratio(sum(r["tokens"] for r in valid), len(valid), 1),
        "avg_seconds": ratio(sum(r["seconds"] for r in valid), len(valid), 2),
        "avg_turns": ratio(sum(r["turns"] for r in valid), len(valid), 2),
        # 重试和等待统计所有运行（含调用失败）；旧结果文件没有这两个字段，按 0 算
        "total_retries": sum(r.get("llm_retries", 0) for r in counted),
        "runs_with_retries": sum(1 for r in counted if r.get("llm_retries", 0) > 0),
        "total_wait_seconds": round(sum(r.get("wait_seconds", 0.0) for r in counted), 1),
        "status_counts": dict(Counter(r["status"] for r in counted)),
        "passes_per_task": {tid: sum(v) for tid, v in by_task.items()},
    }


def build_report(rows: list[dict], trials: int, issues: dict[str, dict]) -> dict:
    """原始口径 + 修正后口径（只在有题目被修正时给出），并写明修正了哪几题、为什么。"""
    report = summarize(rows, trials)
    applied = [issues[i] for i in dict.fromkeys(r["id"] for r in rows) if i in issues]
    if applied:
        report["corrected"] = summarize(rows, trials, key="corrected_verdict")
        report["label_corrections"] = [
            {k: issue[k] for k in ("id", "issue", "action", "original_expect", "expect", "reason") if k in issue}
            for issue in applied]
    return report


def pct(x):
    return "-" if x is None else f"{x:.1%}"


def print_report(stats: dict, trials: int) -> None:
    vc = stats["verdict_counts"]
    print(f"\n{stats['tasks']} 题 × {trials} 次 = {stats['runs']} 次运行")
    print(f"[原始口径] 平均正确率 {pct(stats['mean_accuracy'])}  pass^{trials} {pct(stats['pass_all'])}  "
          f"格式遵从率 {pct(stats['format_compliance'])}  调用失败 {stats['error_count']} 次（{pct(stats['error_rate'])}）")
    if "corrected" in stats:
        c = stats["corrected"]
        print(f"[修正后]   平均正确率 {pct(c['mean_accuracy'])}  pass^{trials} {pct(c['pass_all'])}  "
              f"（{c['tasks']} 题，{c['runs']} 次运行）")
        for issue in stats["label_corrections"]:
            how = f"按 {', '.join(issue['expect'])} 判" if issue["action"] == "relabel" else "不计入"
            print(f"  - {issue['id']}：{how}。{issue['reason']}")
    print(f"答对 {vc[PASS]}  答错 {vc[FAIL]}  格式错误 {vc[FORMAT_ERROR]}  调用失败 {vc[ERROR]}  "
          f"平均 token {stats['avg_tokens']}  平均耗时 {stats['avg_seconds']}s  平均轮数 {stats['avg_turns']}  "
          f"终止原因 {stats['status_counts']}")
    print(f"重试 {stats['total_retries']} 次（{stats['runs_with_retries']} 次运行有重试）  "
          f"退避和限速等待共 {stats['total_wait_seconds']}s（不计入耗时）")


def write_result(path: str, meta: dict, stats: dict, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**meta, **stats, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"详细结果：{path}")


def regrade(path: str, issues: dict[str, dict]) -> None:
    """不调用模型，用当前的判分规则和标注问题清单重判一份已有的结果文件。"""
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    tasks_file = os.path.join(ROOT, old.get("tasks_file", "evals/tasks.jsonl"))
    tasks = {t["id"]: t for t in map(json.loads, filter(str.strip, open(tasks_file, encoding="utf-8")))}
    rows = []
    for r in old["rows"]:
        raw, fixed = judge_run(tasks[r["id"]], r["status"], r["answer"], issues.get(r["id"]))
        rows.append({**r, "ok": raw == PASS, "verdict": raw, "corrected_verdict": fixed})
    trials = old.get("trials", 1)
    stats = build_report(rows, trials, issues)
    print_report(stats, trials)
    meta = {k: old.get(k) for k in ("provider", "model", "base_url", "temperature", "top_p", "extra_body",
                                     "tasks_file")}
    write_result(os.path.splitext(path)[0] + "-regraded.json", {**meta, "regraded_from": path}, stats, rows)


def select_error_runs(rows: list[dict]) -> list[tuple[str, int]]:
    """调用失败（模型接口没返回）的那几次运行：(题目 id, 第几次)。"""
    return [(r["id"], r["trial"]) for r in rows if r["verdict"] == ERROR]


def merge_reruns(rows: list[dict], new_rows: list[dict]) -> list[dict]:
    """把补跑结果按 (id, trial) 替换回原位置，标记 rerun=True 并记下原来的状态。不修改传入的列表。"""
    failed = {(r["id"], r["trial"]): r for r in rows if r["verdict"] == ERROR}
    replacement = {}
    for new in new_rows:
        key = (new["id"], new["trial"])
        if key not in failed:
            raise ValueError(f"{key} 在原结果里不是调用失败，不能用补跑结果替换")
        replacement[key] = {**new, "rerun": True, "rerun_of_status": failed[key]["status"]}
    return [replacement.get((r["id"], r["trial"]), r) for r in rows]


def rerun_errors(path: str, run_fn, issues: dict[str, dict], settings: dict) -> str | None:
    """读取结果文件，只重跑调用失败的那几次（同一题、同一次序号），合并后写到 <原文件名>-rerun.json。

    run_fn(task, trial) 返回一行结果；settings 是补跑时用的限速和重试设置，和这轮补跑了哪几次一起追加到 rerun_rounds。
    原结果的模型设置原样保留；没有调用失败时不做任何事，返回 None。
    """
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    todo = select_error_runs(old["rows"])
    if not todo:
        print("没有调用失败的运行，不需要补跑")
        return None
    tasks_file = os.path.join(ROOT, old["tasks_file"])  # 绝对路径时 join 返回它本身
    with open(tasks_file, encoding="utf-8") as f:
        tasks = {t["id"]: t for t in map(json.loads, filter(str.strip, f))}
    print(f"补跑 {len(todo)} 次调用失败的运行：{', '.join(f'{tid}#t{k}' for tid, k in todo)}")
    new_rows = [run_fn(tasks[tid], k) for tid, k in todo]
    rows = merge_reruns(old["rows"], new_rows)
    trials = old.get("trials", 1)
    stats = build_report(rows, trials, issues)
    print_report(stats, trials)
    meta = {k: v for k, v in old.items() if k != "rows" and k not in stats}
    # 每轮补跑追加一条记录，对补跑结果再补跑时，前几轮补跑了哪几次不会被覆盖
    meta["rerun_rounds"] = old.get("rerun_rounds", []) + [{
        "from": path, "settings": settings,
        "reruns": [{"id": r["id"], "trial": r["trial"], "verdict": r["verdict"]} for r in new_rows]}]
    out = os.path.splitext(path)[0] + "-rerun.json"
    write_result(out, meta, stats, rows)
    return out


def run_one(task: dict, trial: int, trials: int, args, api_key: str, tracer, rate_limiter,
            issues: dict[str, dict]) -> dict:
    """跑一题的第 trial 次，返回结果行。"""
    agent = build_agent(args, api_key, tracer, rate_limiter)
    state = agent.run(task["goal"], run_id=f"{task['id']}-t{trial}")
    raw, fixed = judge_run(task, state.status, state.final_answer, issues.get(task["id"]))
    tag = {PASS: "PASS", FAIL: "FAIL", FORMAT_ERROR: "FMT ", ERROR: "ERR "}[raw]
    print(f"[{tag}] {task['id']:<16} trial={trial}/{trials} status={state.status:<14} "
          f"turns={state.turn} tokens={state.tokens_used} retries={state.llm_retries}", flush=True)
    return {"id": task["id"], "trial": trial, "ok": raw == PASS, "verdict": raw, "corrected_verdict": fixed,
            "judge": task.get("judge", "keywords"), "status": state.status, "turns": state.turn,
            "tokens": state.tokens_used,
            # 耗时不含退避和限速等待，不同限流条件下的模型才能公平比较
            "seconds": round(state.elapsed_s, 2), "wait_seconds": round(state.wait_s, 2),
            "llm_retries": state.llm_retries, "answer": state.final_answer}


# 补跑时从原结果文件里沿用的设置：保证和原评测是同一个模型、同一组参数
RERUN_INHERITED = ("provider", "base_url", "model", "temperature", "top_p", "extra_body", "max_turns",
                   "token_budget")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["mock", "openai"], default="mock")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--key-env", default="LLM_API_KEY", help="从哪个环境变量读取 API Key（Key 本身不会打印）")
    p.add_argument("--temperature", type=optional_float, default=0.0, help="数字，或 none 表示不发送")
    p.add_argument("--top-p", type=optional_float, default=None, help="数字，或 none（默认）表示不发送")
    p.add_argument("--extra-body", default="",
                   help='合并进请求体的额外字段（JSON 对象），如 \'{"thinking": {"type": "disabled"}}\'')
    p.add_argument("--tasks", default=os.path.join(ROOT, "evals", "tasks.jsonl"))
    p.add_argument("--trials", type=int, default=1, help="每题跑几次")
    p.add_argument("--max-retries", type=int, default=2,
                   help="每次模型调用最多重试几次；429 按 Retry-After 等待，没有时 10→60 秒退避")
    p.add_argument("--min-interval", type=float, default=0.0,
                   help="相邻两次模型请求的最小间隔（秒），主动限速减少 429；0 表示不限")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--token-budget", type=int, default=30000)
    p.add_argument("--label-issues", default=DEFAULT_LABEL_ISSUES,
                   help="标注问题清单，用于计算修正后口径；只对 id 匹配的题生效")
    p.add_argument("--regrade", metavar="RESULT", help="不调用模型，重判已有的 eval-*.json")
    p.add_argument("--rerun-errors", metavar="RESULT",
                   help="只重跑结果文件里调用失败的那几次，合并后写到 <原文件名>-rerun.json；"
                        "模型和采样参数沿用原结果，--key-env、--min-interval、--max-retries 用本次命令行的值")
    args = p.parse_args()
    try:
        args.extra_body = parse_extra_body(args.extra_body)
    except ValueError as e:
        p.error(str(e))
    issues = load_label_issues(args.label_issues)
    if args.regrade:
        regrade(args.regrade, issues)
        return
    if args.rerun_errors:
        with open(args.rerun_errors, encoding="utf-8") as f:
            old = json.load(f)
        for k in RERUN_INHERITED:
            if k in old:  # 旧结果文件没记录的（如 max_turns）用命令行默认值，和当时一致
                setattr(args, k, old[k])  # temperature/top_p 为 None 表示原来就没发送，照样沿用
        args.extra_body = args.extra_body or {}
    if args.trials < 1:
        p.error("--trials 至少为 1")
    if args.max_retries < 0 or args.min_interval < 0:
        p.error("--max-retries 和 --min-interval 不能为负数")

    api_key = ""
    if args.provider == "openai":
        api_key = os.getenv(args.key_env, "")
        if not api_key:
            p.error(f"环境变量 {args.key_env} 未设置或为空")

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tracer = Tracer(os.path.join(ROOT, "runs", f"trace-{stamp}.jsonl"))
    rate_limiter = RateLimiter(args.min_interval)  # 所有 trial 共享，才能真正控制请求间隔

    if args.rerun_errors:
        trials = old.get("trials", 1)
        rerun_errors(args.rerun_errors,
                     lambda task, k: run_one(task, k, trials, args, api_key, tracer, rate_limiter, issues),
                     issues, settings={"min_interval": args.min_interval, "max_retries": args.max_retries,
                                       "trace": os.path.relpath(tracer.path, ROOT) if tracer.path else None})
        return

    tasks = [json.loads(line) for line in open(args.tasks, encoding="utf-8") if line.strip()]
    rows = [run_one(task, k, args.trials, args, api_key, tracer, rate_limiter, issues)
            for task in tasks for k in range(1, args.trials + 1)]

    stats = build_report(rows, args.trials, issues)
    print_report(stats, args.trials)
    meta = {"provider": args.provider, "model": args.model if args.provider == "openai" else "mock",
            "base_url": args.base_url if args.provider == "openai" else None,
            "temperature": args.temperature, "top_p": args.top_p, "extra_body": args.extra_body,
            "max_retries": args.max_retries, "min_interval": args.min_interval,
            "max_turns": args.max_turns, "token_budget": args.token_budget,
            "tasks_file": tasks_file_label(args.tasks)}
    write_result(os.path.join(ROOT, "runs", f"eval-{stamp}.json"), meta, stats, rows)


if __name__ == "__main__":
    main()
