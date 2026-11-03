"""评测判定规则的测试：数字按完整的数比较，日期统一成 YYYY-MM-DD 再比较。
运行：python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))

from judge import judge, matches


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


if __name__ == "__main__":
    unittest.main()
