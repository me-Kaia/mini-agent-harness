"""评测汇总指标的测试：多次运行的平均正确率、pass^N、格式遵从率、调用失败、标注修正。
运行：python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))

from judge import FAIL, FORMAT_ERROR, PASS
from run_eval import ERROR, judge_run, load_label_issues, merge_reruns, rerun_errors, select_error_runs, summarize


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
        self.assertEqual(s["verdict_counts"], {PASS: 1, FAIL: 1, FORMAT_ERROR: 2, ERROR: 0})

    def test_format_compliance_ignores_unfinished_and_keyword_runs(self):
        rows = [row("a", PASS), row("a", FAIL, status="max_turns"), row("k", FAIL, judge="keywords")]
        self.assertEqual(summarize(rows, trials=2)["format_compliance"], 1.0)
        self.assertIsNone(summarize([row("k", PASS, judge="keywords")], trials=1)["format_compliance"])

    def test_averages_are_per_run(self):
        rows = [row("a", PASS, tokens=100, seconds=1.0), row("a", PASS, tokens=300, seconds=3.0)]
        s = summarize(rows, trials=2)
        self.assertEqual((s["avg_tokens"], s["avg_seconds"]), (200.0, 2.0))


class TestRetryStats(unittest.TestCase):
    def test_retries_and_wait_totals_include_failed_runs(self):
        rows = [dict(row("a", PASS), llm_retries=2, wait_seconds=30.0),
                dict(row("a", ERROR, status="llm_error"), llm_retries=5, wait_seconds=150.0),
                row("b", PASS)]  # 旧结果文件没有这两个字段，按 0 算
        s = summarize(rows, trials=2)
        self.assertEqual((s["total_retries"], s["total_wait_seconds"]), (7, 180.0))
        self.assertEqual(s["runs_with_retries"], 2)


class TestCallFailures(unittest.TestCase):
    """调用失败（llm_error）单独统计，不进正确率分母。"""

    def test_errors_excluded_from_accuracy_denominator(self):
        rows = [row("a", PASS), row("a", PASS), row("a", ERROR, status="llm_error", tokens=0, seconds=0),
                row("b", PASS), row("b", FAIL), row("b", PASS)]
        s = summarize(rows, trials=3)
        self.assertEqual(s["runs"], 6)
        self.assertEqual(s["mean_accuracy"], round(4 / 5, 3))
        self.assertEqual((s["error_count"], s["error_rate"]), (1, round(1 / 6, 3)))
        self.assertEqual(s["verdict_counts"][ERROR], 1)
        # a 的两次有效运行都对，算"全对"；b 有一次错
        self.assertEqual(s["pass_all"], 0.5)
        # 平均 token 只算有效运行，不被调用失败的 0 拉低
        self.assertEqual(s["avg_tokens"], 100.0)

    def test_task_with_only_errors_is_not_in_pass_all(self):
        rows = [row("a", PASS), row("z", ERROR, status="llm_error")]
        s = summarize(rows, trials=1)
        self.assertEqual((s["mean_accuracy"], s["pass_all"], s["error_rate"]), (1.0, 1.0, 0.5))

    def test_judge_run_status_mapping(self):
        task = {"id": "t", "expect": ["18"], "judge": "final_answer"}
        self.assertEqual(judge_run(task, "llm_error", "")[0], ERROR)
        self.assertEqual(judge_run(task, "max_turns", "答案：18")[0], FAIL)
        self.assertEqual(judge_run(task, "completed", "答案：18")[0], PASS)


class TestLabelCorrections(unittest.TestCase):
    """有问题的题目：relabel 用修正后的答案重判，exclude 不计入修正后口径。"""

    TASK_RELABEL = {"id": "gsm8k_1309", "expect": ["2280"], "judge": "final_answer"}
    TASK_EXCLUDE = {"id": "gsm8k_255", "expect": ["192"], "judge": "final_answer"}
    ISSUES = {"gsm8k_1309": {"id": "gsm8k_1309", "action": "relabel", "expect": ["2180"], "reason": "r"},
              "gsm8k_255": {"id": "gsm8k_255", "action": "exclude", "reason": "r"}}

    def test_relabel(self):
        issue = self.ISSUES["gsm8k_1309"]
        self.assertEqual(judge_run(self.TASK_RELABEL, "completed", "答案：2180", issue), (FAIL, PASS))
        self.assertEqual(judge_run(self.TASK_RELABEL, "completed", "答案：2280", issue), (PASS, FAIL))

    def test_exclude(self):
        self.assertEqual(judge_run(self.TASK_EXCLUDE, "completed", "答案：176", self.ISSUES["gsm8k_255"]),
                         (FAIL, None))

    def test_no_issue_keeps_verdict(self):
        self.assertEqual(judge_run(self.TASK_RELABEL, "completed", "答案：2280"), (PASS, PASS))
        self.assertEqual(judge_run(self.TASK_RELABEL, "llm_error", ""), (ERROR, ERROR))

    def test_corrected_summary_drops_excluded_rows(self):
        rows = [dict(row("gsm8k_1309", FAIL), corrected_verdict=PASS),
                dict(row("gsm8k_255", FAIL), corrected_verdict=None),
                dict(row("ok", PASS), corrected_verdict=PASS)]
        raw = summarize(rows, trials=1)
        fixed = summarize(rows, trials=1, key="corrected_verdict")
        self.assertEqual((raw["tasks"], raw["mean_accuracy"]), (3, round(1 / 3, 3)))
        self.assertEqual((fixed["tasks"], fixed["mean_accuracy"], fixed["pass_all"]), (2, 1.0, 1.0))

    def test_load_label_issues(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "issues.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for issue in self.ISSUES.values():
                    f.write(json.dumps(issue, ensure_ascii=False) + "\n")
            self.assertEqual(load_label_issues(path), self.ISSUES)
            self.assertEqual(load_label_issues(os.path.join(d, "missing.jsonl")), {})
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "x", "action": "relabel", "reason": "缺 expect"}) + "\n")
            with self.assertRaises(ValueError):
                load_label_issues(path)


class TestRerunErrors(unittest.TestCase):
    """补跑：只重跑调用失败的那几次（同一题、同一次序号），结果合并回原结果并标明哪些是补跑的。"""

    def rows(self):
        return [dict(row("a", PASS), trial=1), dict(row("a", ERROR, status="llm_error"), trial=2),
                dict(row("b", FAIL), trial=1), dict(row("b", ERROR, status="llm_error"), trial=2)]

    def test_select_only_call_failures(self):
        self.assertEqual(select_error_runs(self.rows()), [("a", 2), ("b", 2)])

    def test_merge_replaces_in_place_and_marks_rerun(self):
        old = self.rows()
        new = [dict(row("b", PASS), trial=2), dict(row("a", ERROR, status="llm_error"), trial=2)]
        merged = merge_reruns(old, new)
        self.assertEqual([(r["id"], r["trial"]) for r in merged], [("a", 1), ("a", 2), ("b", 1), ("b", 2)])
        self.assertEqual([r.get("rerun", False) for r in merged], [False, True, False, True])
        self.assertEqual(merged[3]["verdict"], PASS)
        self.assertEqual(merged[3]["rerun_of_status"], "llm_error")
        self.assertEqual(merged[1]["verdict"], ERROR)  # 补跑仍失败的照实记录
        self.assertEqual(old[3]["verdict"], ERROR)     # 不修改传入的原结果

    def test_merge_rejects_runs_that_were_not_failures(self):
        with self.assertRaises(ValueError):
            merge_reruns(self.rows(), [dict(row("a", PASS), trial=1)])

    def test_rerun_errors_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            tasks_path = os.path.join(d, "tasks.jsonl")
            with open(tasks_path, "w", encoding="utf-8") as f:
                for tid in ("a", "b"):
                    f.write(json.dumps({"id": tid, "goal": tid, "expect": ["1"], "judge": "final_answer"}) + "\n")
            result_path = os.path.join(d, "eval-x.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({"model": "m", "temperature": 0.6, "tasks_file": tasks_path, "trials": 2,
                           "rows": self.rows()}, f)
            calls = []

            def run_fn(task, trial):
                calls.append((task["id"], trial))
                return dict(row(task["id"], PASS), trial=trial)

            out = rerun_errors(result_path, run_fn, issues={}, settings={"min_interval": 10, "max_retries": 5})
            self.assertEqual(calls, [("a", 2), ("b", 2)])
            with open(out, encoding="utf-8") as f:
                saved = json.load(f)
        self.assertTrue(out.endswith("eval-x-rerun.json"))
        self.assertEqual(saved["error_count"], 0)
        self.assertEqual(saved["verdict_counts"][PASS], 3)
        self.assertEqual(saved["model"], "m")  # 原结果的模型设置原样保留
        self.assertEqual(saved["rerun_rounds"], [{
            "from": result_path, "settings": {"min_interval": 10, "max_retries": 5},
            "reruns": [{"id": "a", "trial": 2, "verdict": PASS}, {"id": "b", "trial": 2, "verdict": PASS}]}])

    def test_second_round_keeps_first_round_history(self):
        # 对补跑结果再补跑：前一轮补跑了哪几次不能被覆盖掉
        with tempfile.TemporaryDirectory() as d:
            tasks_path = os.path.join(d, "tasks.jsonl")
            with open(tasks_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "a", "goal": "a", "expect": ["1"], "judge": "final_answer"}) + "\n")
            path = os.path.join(d, "eval-x.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"tasks_file": tasks_path, "trials": 2, "rows": [
                    dict(row("a", ERROR, status="llm_error"), trial=1),
                    dict(row("a", ERROR, status="llm_error"), trial=2)]}, f)
            first = rerun_errors(path, lambda t, k: dict(row("a", PASS if k == 1 else ERROR,
                                                             status="completed" if k == 1 else "llm_error"),
                                                         trial=k), issues={}, settings={"min_interval": 3})
            second = rerun_errors(first, lambda t, k: dict(row("a", PASS), trial=k), issues={},
                                  settings={"min_interval": 10})
            with open(second, encoding="utf-8") as f:
                saved = json.load(f)
        self.assertTrue(second.endswith("eval-x-rerun-rerun.json"))
        self.assertEqual([r["settings"]["min_interval"] for r in saved["rerun_rounds"]], [3, 10])
        self.assertEqual([[(x["id"], x["trial"], x["verdict"]) for x in r["reruns"]] for r in saved["rerun_rounds"]],
                         [[("a", 1, PASS), ("a", 2, ERROR)], [("a", 2, PASS)]])
        self.assertEqual([r.get("rerun") for r in saved["rows"]], [True, True])

    def test_nothing_to_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            result_path = os.path.join(d, "eval-x.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({"tasks_file": "unused", "trials": 1, "rows": [dict(row("a", PASS), trial=1)]}, f)
            self.assertIsNone(rerun_errors(result_path, lambda t, k: self.fail("不应调用"), issues={}, settings={}))



if __name__ == "__main__":
    unittest.main()
