"""从 GSM8K 测试集里用固定随机种子抽题，转成 evals 的任务格式。

用法：
  python evals/make_gsm8k.py
  python evals/make_gsm8k.py --n 100 --seed 0 --out evals/gsm8k_100.jsonl

输入：evals/data/test.jsonl（每行 {"question", "answer"}，answer 最后一行是 "#### 数字"）
输出：每行 {"id": "gsm8k_<原始行号>", "goal": 题目 + 作答要求, "expect": [标准答案]}
"""
from __future__ import annotations

import argparse
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUCTION = "请逐步解答，最后一行只写：答案：数字"


def extract_answer(answer: str) -> str:
    if "####" not in answer:
        raise ValueError(f"answer 字段里没有 ####：{answer[-80:]!r}")
    return answer.split("####")[-1].strip().replace(",", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(ROOT, "evals", "data", "test.jsonl"))
    p.add_argument("--out", default=os.path.join(ROOT, "evals", "gsm8k_50.jsonl"))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.src, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    indices = sorted(random.Random(args.seed).sample(range(len(items)), args.n))

    with open(args.out, "w", encoding="utf-8") as f:
        for i in indices:
            item = items[i]
            task = {"id": f"gsm8k_{i}",
                    "goal": f"{item['question'].strip()}\n{INSTRUCTION}",
                    "expect": [extract_answer(item["answer"])]}
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"从 {len(items)} 题中抽取 {len(indices)} 题（seed={args.seed}），已写入 {args.out}")


if __name__ == "__main__":
    main()
