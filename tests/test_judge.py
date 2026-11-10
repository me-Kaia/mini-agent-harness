"""评测判定规则的测试：数字按完整的数比较，日期统一成 YYYY-MM-DD 再比较。
运行：python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))

from judge import FAIL, FORMAT_ERROR, PASS, grade, judge, matches


class TestMatches(unittest.TestCase):
    def test_chinese_date_matches_iso(self):
        self.assertTrue(matches("截止日期是 **2026 年 12 月 15 日**。", "2026-12-15"))

    def test_date_without_padding(self):
        self.assertTrue(matches("2026年1月5日", "2026-01-05"))
        self.assertTrue(matches("2026/1/5", "2026-01-05"))

    def test_different_date_does_not_match(self):
        self.assertFalse(matches("2026 年 12 月 16 日", "2026-12-15"))

    def test_thousands_separator(self):
        self.assertTrue(matches("预算是 12,000 元", "12000"))

    def test_number_is_not_substring(self):
        self.assertFalse(matches("电容 C1：数量 250", "50"))
        self.assertFalse(matches("单价 0.05 元", "50"))

    def test_whole_number_matches(self):
        self.assertTrue(matches("总价 50 元", "50"))
        self.assertTrue(matches("**总价：50.0 元**", "50"))

    def test_text_keyword_is_substring(self):
        self.assertTrue(matches("**secret.txt 不存在。**", "不存在"))
        self.assertTrue(matches("- `notes.txt`", "notes.txt"))


class TestJudge(unittest.TestCase):
    def test_all_keywords_required(self):
        self.assertTrue(judge("有 notes.txt 和 parts.txt", ["notes.txt", "parts.txt"]))
        self.assertFalse(judge("只有 notes.txt", ["notes.txt", "parts.txt"]))


class TestGradeKeywords(unittest.TestCase):
    """默认模式（tasks.jsonl 的任务）沿用关键词逻辑。"""

    def test_default_mode_is_keywords(self):
        self.assertEqual(grade("总价 50 元", ["50"]), PASS)
        self.assertEqual(grade("数量 250", ["50"]), FAIL)
        self.assertEqual(grade("总价 50 元", ["50"], mode="keywords"), PASS)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            grade("答案：1", ["1"], mode="nope")


class TestGradeFinalAnswer(unittest.TestCase):
    """final_answer 模式：只看最后一个"答案："后面的数。"""

    def grade(self, answer, expected):
        return grade(answer, [expected], mode="final_answer")

    def test_correct_number_in_reasoning_but_wrong_final_answer(self):
        answer = "16 - 3 - 4 = 9 个蛋，9 * 2 = 18 美元。\n答案：20"
        self.assertEqual(self.grade(answer, "18"), FAIL)

    def test_thousands_separator(self):
        self.assertEqual(self.grade("答案：1,234", "1234"), PASS)

    def test_unit_after_number(self):
        self.assertEqual(self.grade("答案：18 美元", "18"), PASS)

    def test_last_answer_wins(self):
        self.assertEqual(self.grade("答案：20\n再检查一遍，前面算错了。\n答案：18", "18"), PASS)
        self.assertEqual(self.grade("答案：18\n再检查一遍。\n答案：20", "18"), FAIL)

    def test_missing_answer_marker_is_format_error(self):
        self.assertEqual(self.grade("她每天赚 18 美元。", "18"), FORMAT_ERROR)


if __name__ == "__main__":
    unittest.main()
