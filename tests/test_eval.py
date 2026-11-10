"""评测汇总指标的测试：多次运行的平均正确率、pass^N、格式遵从率。
运行：python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))

from judge import FAIL, FORMAT_ERROR, PASS
from run_eval import summarize


def row(tid, verdict, judge="final_answer", status="completed", tokens=100, seconds=1.0):
    return {"id": tid, "verdict": verdict, "judge": judge, "status": status,
            "tokens": tokens, "seconds": seconds, "turns": 1}


class TestSummarize(unittest.TestCase):
    def test_pass_all_differs_from_mean_accuracy(self):
        # a: 3/3 全对；b: 2/3；c: 0/3
        rows = [row("a", PASS)] * 3 + [row("b", PASS), row("b", PASS), row("b", FAIL)] + [row("c", FAIL)] * 3
        s = summarize(rows, trials=3)
        self.assertEqual((s["tasks"], s["runs"]), (3, 9))
        self.assertEqual(s["mean_accuracy"], round(5 / 9, 3))
        self.assertEqual(s["pass_all"], round(1 / 3, 3))
        self.assertEqual(s["passes_per_task"], {"a": 3, "b": 2, "c": 0})

    def test_format_compliance(self):
        rows = [row("a", PASS), row("a", FORMAT_ERROR), row("b", FAIL), row("b", FORMAT_ERROR)]
        s = summarize(rows, trials=2)
        self.assertEqual(s["format_compliance"], 0.5)
        self.assertEqual(s["verdict_counts"], {PASS: 1, FAIL: 1, FORMAT_ERROR: 2})

    def test_format_compliance_ignores_unfinished_and_keyword_runs(self):
        rows = [row("a", PASS), row("a", FAIL, status="max_turns"), row("k", FAIL, judge="keywords")]
        self.assertEqual(summarize(rows, trials=2)["format_compliance"], 1.0)
        self.assertIsNone(summarize([row("k", PASS, judge="keywords")], trials=1)["format_compliance"])

    def test_averages_are_per_run(self):
        rows = [row("a", PASS, tokens=100, seconds=1.0), row("a", PASS, tokens=300, seconds=3.0)]
        s = summarize(rows, trials=2)
        self.assertEqual((s["avg_tokens"], s["avg_seconds"]), (200.0, 2.0))


if __name__ == "__main__":
    unittest.main()
