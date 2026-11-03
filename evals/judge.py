"""评测判定：最终回答是否包含所有期望关键词。

纯子串匹配有两类误判：
- 写法不同就判错："2026 年 12 月 15 日" 不包含 "2026-12-15"，"12,000" 不包含 "12000"。
- 数字被当成子串就误判为对："数量 250" 包含 "50"。

做法：先把回答规范化（日期统一成 YYYY-MM-DD，去掉数字的千分位逗号），再按关键词类型比较：
- 日期关键词：在规范化后的回答里找完整的日期，前后不能紧挨着数字；
- 数字关键词：把回答里的数一个个提取出来，按数值比较（"50.0" 等于 "50"），不做子串匹配；
- 其他关键词：普通子串匹配。
"1.2 万"、"一万二" 这类写法仍然识别不了，见 README 局限。
"""
from __future__ import annotations

import re

_DATE_PATTERNS = [
    re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?"),
    re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)"),
]
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def normalize(text: str) -> str:
    for pattern in _DATE_PATTERNS:
        text = pattern.sub(lambda m: f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}", text)
    return _THOUSANDS.sub("", text)


def matches(answer: str, keyword: str) -> bool:
    text = normalize(answer)
    keyword = normalize(keyword.strip())
    if _ISO_DATE.fullmatch(keyword):
        return re.search(rf"(?<!\d){keyword}(?!\d)", text) is not None
    if _NUMBER.fullmatch(keyword):
        # 日期已经是 YYYY-MM-DD，里面的年、月、日也会被提取成数，这对数字关键词没有影响
        return any(float(n) == float(keyword) for n in _NUMBER.findall(text))
    return keyword in text


def judge(answer: str, expected: list[str]) -> bool:
    return all(matches(answer, k) for k in expected)
