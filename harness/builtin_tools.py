"""示例工具：安全计算器、限定目录内的文件读取与列目录。"""
from __future__ import annotations

import ast
import operator
import os

from .tools import Tool

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> str:
    """只允许数字和四则运算，不用 eval，避免模型借计算器执行任意代码。"""
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
                raise ValueError("指数过大")
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"不支持的表达式成分：{type(node).__name__}")
    result = _eval(ast.parse(expression, mode="eval"))
    return str(round(result, 10) if isinstance(result, float) else result)


def make_file_tools(root: str) -> list[Tool]:
    root = os.path.realpath(root)

    def _safe(path: str) -> str:
        full = os.path.realpath(os.path.join(root, path))
        # 防目录穿越：解析符号链接和 ../ 之后，路径必须仍在 root 内
        if os.path.commonpath([full, root]) != root:
            raise PermissionError("只能访问工作目录内的文件")
        return full

    def read_file(path: str) -> str:
        with open(_safe(path), encoding="utf-8") as f:
            return f.read()

    def list_dir(path: str = ".") -> str:
        return "\n".join(sorted(os.listdir(_safe(path))))

    return [
        Tool("read_file", "读取工作目录内的文本文件", {
            "type": "object", "properties": {"path": {"type": "string", "description": "相对路径"}},
            "required": ["path"]}, read_file),
        Tool("list_dir", "列出工作目录内某个文件夹的文件", {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": []}, list_dir),
    ]


def calculator_tool() -> Tool:
    return Tool("calculator", "计算数学表达式，只支持 + - * / ** % 和括号", {
        "type": "object", "properties": {"expression": {"type": "string"}},
        "required": ["expression"]}, calculate)
