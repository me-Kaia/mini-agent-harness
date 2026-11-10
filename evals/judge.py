"""评测判定：最终回答是否包含所有期望关键词。

纯子串匹配有两类误判：
- 写法不同就判错："2026 年 12 月 15 日" 不包含 "2026-12-15"，"12,000" 不包含 "12000"。
- 数字被当成子串就误判为对："数量 250" 包含 "50"。

做法：先把回答规范化（日期统一成 YYYY-MM-DD，去掉数字的千分位逗号），再按关键词类型比较：
- 日期关键词：在规范化后的回答里找完整的日期，前后不能紧挨着数字；
- 数字关键词：把回答里的数一个个提取出来，按数值比较（"50.0" 等于 "50"），不做子串匹配；
- 其他关键词：普通子串匹配。
"1.2 万"、"一万二" 这类写法仍然识别不了，见 README 局限。

数学题（GSM8K）在任务里写 "judge": "final_answer"，改用最终答案模式：
推理过程里会出现大量中间数，按上面的规则只要正确答案出现过一次就会误判为对，
所以只取最后一个"答案："后面的数比较；没写"答案："的单独算格式错误，不算答错。
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


PASS, FAIL, FORMAT_ERROR = "pass", "fail", "format_error"

# "答案：" 或 "答案:"，后面到行尾的内容是最终答案
_FINAL_ANSWER = re.compile(r"答案\s*[：:]\s*([^\n]*)")
_SIGNED_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_answer(answer: str) -> float | None:
    """取最后一个"答案："后面的第一个数；去掉逗号，忽略单位和 Markdown 符号。找不到返回 None。"""
    found = _FINAL_ANSWER.findall(answer)
    if not found:
        return None
    m = _SIGNED_NUMBER.search(re.sub(r"(?<=\d),(?=\d)", "", found[-1]))
    return float(m[0]) if m else None


def judge_final_answer(answer: str, expected: list[str]) -> str:
    """数学题判分：只看最终答案，不看推理过程里出现过的数。"""
    value = extract_final_answer(answer)
    if value is None:
        return FORMAT_ERROR
    return PASS if value == float(expected[0].replace(",", "")) else FAIL


def grade(answer: str, expected: list[str], mode: str = "keywords") -> str:
    """按任务指定的模式判分，返回 PASS / FAIL / FORMAT_ERROR。

    - keywords（默认）：回答包含所有期望关键词，见 judge()；
    - final_answer：只比较最后一个"答案："后面的数；没有"答案："或后面没有数，判为格式错误。
    """
    if mode == "keywords":
        return PASS if judge(answer, expected) else FAIL
    if mode == "final_answer":
        return judge_final_answer(answer, expected)
    raise ValueError(f"未知的判分模式：{mode}")
