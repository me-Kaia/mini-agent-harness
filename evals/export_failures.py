"""把一次评测结果里答错（fail）、格式错误（format_error）和调用失败（error）的运行导出成 CSV，方便逐题分析。

用法：
  python evals/export_failures.py runs/eval-20261110-151023.json
  python evals/export_failures.py runs/eval-xxx.json --out runs/failures.csv

CSV 用 UTF-8 带 BOM 编码，Excel 直接打开不会乱码；完整回答里的换行保留在单元格内。
每行一次运行；"本题通过次数" 是这道题在所有 trial 里 PASS 的次数，用来区分"偶尔错"和"每次都错"。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evals"))

from judge import FAIL, FORMAT_ERROR, extract_final_answer

ERROR = "error"  # 与 run_eval.ERROR 一致；不 import run_eval，避免导出时加载 harness
CORRECTED_LABEL = {None: "不计入"}

COLUMNS = ["id", "trial", "判定", "修正后判定", "标准答案", "模型最终答案", "本题通过次数", "终止状态",
           "轮数", "token", "耗时(s)", "题目", "完整回答"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result", help="run_eval.py 输出的 eval-*.json")
    p.add_argument("--out", help="默认与结果文件同名，后缀为 -failures.csv")
    args = p.parse_args()

    with open(args.result, encoding="utf-8") as f:
        result = json.load(f)
    tasks_file = os.path.join(ROOT, result.get("tasks_file", "evals/tasks.jsonl"))
    with open(tasks_file, encoding="utf-8") as f:
        tasks = {t["id"]: t for t in map(json.loads, filter(str.strip, f))}

    trials = result.get("trials", 1)
    passes = result.get("passes_per_task", {})
    bad = [r for r in result["rows"] if r["verdict"] in (FAIL, FORMAT_ERROR, ERROR)]
    bad.sort(key=lambda r: (passes.get(r["id"], 0), r["id"], r.get("trial", 1)))

    out = args.out or os.path.splitext(args.result)[0] + "-failures.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in bad:
            task = tasks.get(r["id"], {})
            value = extract_final_answer(r["answer"] or "")
            fixed = r.get("corrected_verdict", r["verdict"])
            w.writerow([r["id"], r.get("trial", 1), r["verdict"], CORRECTED_LABEL.get(fixed, fixed),
                        ", ".join(task.get("expect", [])),
                        "" if value is None else f"{value:g}", f"{passes.get(r['id'], 0)}/{trials}",
                        r["status"], r["turns"], r["tokens"], r["seconds"],
                        task.get("goal", ""), r["answer"] or ""])
    print(f"{len(bad)} 条（答错 {sum(r['verdict'] == FAIL for r in bad)}，"
          f"格式错误 {sum(r['verdict'] == FORMAT_ERROR for r in bad)}，"
          f"调用失败 {sum(r['verdict'] == ERROR for r in bad)}）已导出到 {out}")


if __name__ == "__main__":
    main()
