"""用 ScriptedLLM 离线测试 harness 的每个工程机制，不需要网络和 API Key。
运行：python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import (Agent, AgentConfig, Checkpointer, ContextManager, LLMResponse, ScriptedLLM,
                     Tool, ToolCall, ToolRegistry, Tracer, TransientToolError)
from harness.builtin_tools import calculate, calculator_tool, make_file_tools
from harness.context import check_tool_pairs, split_units


def call(name, args, cid="c1"):
    return LLMResponse(tool_calls=[ToolCall(cid, name, json.dumps(args) if isinstance(args, dict) else args)],
                       prompt_tokens=100, completion_tokens=20)


def answer(text):
    return LLMResponse(content=text, prompt_tokens=100, completion_tokens=20)


def registry(*tools, **kw):
    r = ToolRegistry(backoff_s=0.01, **kw)
    for t in tools:
        r.register(t)
    return r


class TestTermination(unittest.TestCase):
    def test_completes_after_tool_call(self):
        llm = ScriptedLLM([call("calculator", {"expression": "2*21"}), answer("答案是 42")])
        state = Agent(llm, registry(calculator_tool())).run("2*21=?")
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.final_answer, "答案是 42")
        self.assertEqual(state.messages[-2]["content"], "42")

    def test_max_turns(self):
        llm = ScriptedLLM(lambda m: call("calculator", {"expression": f"{len(m)}+1"}, cid=f"c{len(m)}"))
        state = Agent(llm, registry(calculator_tool()), AgentConfig(max_turns=3)).run("x")
        self.assertEqual(state.status, "max_turns")
        self.assertEqual(state.turn, 3)

    def test_token_budget(self):
        llm = ScriptedLLM(lambda m: call("calculator", {"expression": f"{len(m)}"}, cid=f"c{len(m)}"))
        state = Agent(llm, registry(calculator_tool()), AgentConfig(token_budget=250)).run("x")
        self.assertEqual(state.status, "token_budget")  # 每轮 120 token，第 3 轮前超预算

    def test_repeated_call_detected(self):
        # 模型反复用同样参数调同一个工具（卡住了），第 3 次应被拦下
        same = [call("calculator", '{"expression": "1+1"}', cid=f"c{i}") for i in range(5)]
        state = Agent(ScriptedLLM(same), registry(calculator_tool()), AgentConfig(max_repeat=2)).run("x")
        self.assertEqual(state.status, "repeated_call")
        self.assertEqual(state.turn, 3)
        self.assertTrue(check_tool_pairs(state.messages))

    def test_repeated_call_in_multi_tool_response(self):
        # 同一条回复里有两个调用，第一个触发重复上限，第二个也必须有对应的 tool 消息
        same = '{"expression": "1+1"}'
        multi = LLMResponse(tool_calls=[ToolCall("c2", "calculator", same),
                                        ToolCall("c3", "read_file", '{"path": "notes.txt"}')],
                            prompt_tokens=100, completion_tokens=20)
        llm = ScriptedLLM([call("calculator", same, "c0"), call("calculator", same, "c1"), multi])
        with tempfile.TemporaryDirectory() as d:
            reg = registry(calculator_tool(), *make_file_tools(d))
            state = Agent(llm, reg, AgentConfig(max_repeat=2)).run("x")
        self.assertEqual(state.status, "repeated_call")
        self.assertTrue(check_tool_pairs(state.messages))


class TestLLMWaitAccounting(unittest.TestCase):
    """模型调用的重试次数和退避/限速等待时间单独累计；耗时和超时都不算等待时间。"""

    def test_retries_and_wait_accumulate(self):
        r1 = call("calculator", {"expression": "2*21"})
        r1.retries, r1.wait_s = 2, 30.0
        r2 = answer("42")
        r2.retries, r2.wait_s = 1, 10.0
        state = Agent(ScriptedLLM([r1, r2]), registry(calculator_tool())).run("x")
        self.assertEqual(state.status, "completed")
        self.assertEqual((state.llm_retries, state.wait_s), (3, 40.0))

    def test_wait_not_counted_toward_timeout(self):
        # 限流等待不是模型或工具慢，不应触发超时、被判答错
        def slow(messages):
            time.sleep(0.05)  # 模拟真的等了一段时间
            resp = answer("done") if messages[-1]["role"] == "tool" else call("calculator", {"expression": "1+1"})
            resp.wait_s = 0.05
            return resp
        state = Agent(ScriptedLLM(slow), registry(calculator_tool()), AgentConfig(max_seconds=0.08)).run("x")
        self.assertEqual(state.status, "completed")
        self.assertLess(state.elapsed_s, 0.08)

    def test_llm_error_retries_are_recorded(self):
        from harness.llm import LLMError

        def fail(messages):
            raise LLMError("429", retries=3, wait_s=70.0)
        state = Agent(ScriptedLLM(fail), registry(calculator_tool())).run("x")
        self.assertEqual(state.status, "llm_error")
        self.assertEqual((state.llm_retries, state.wait_s), (3, 70.0))

    def test_old_checkpoint_without_new_fields_loads(self):
        from harness.state import RunState
        state = RunState.from_dict({"run_id": "r", "messages": [], "elapsed_s": 1.0})
        self.assertEqual((state.llm_retries, state.wait_s), (0, 0.0))


class TestToolErrors(unittest.TestCase):
    def test_bad_json_fed_back_then_model_fixes_it(self):
        llm = ScriptedLLM([call("calculator", "{expression: 1+1"), call("calculator", {"expression": "1+1"}, "c2"),
                           answer("2")])
        state = Agent(llm, registry(calculator_tool())).run("x")
        self.assertEqual(state.status, "completed")
        self.assertIn("不是合法 JSON", state.messages[3]["content"])

    def test_invalid_args_and_format_error_limit(self):
        bad = [call("calculator", {"expr": "1"}, cid=f"c{i}") for i in range(5)]
        state = Agent(ScriptedLLM(bad), registry(calculator_tool()),
                      AgentConfig(max_format_errors=2, max_repeat=10)).run("x")
        self.assertEqual(state.status, "format_errors")
        self.assertIn("缺少必填参数", state.messages[3]["content"])

    def test_format_errors_reset_after_valid_call(self):
        # 错、对交替出现：每次出错后模型都改对了，不算"连续"出错，不应触发 format_errors
        script = []
        for i in range(3):
            script.append(call("calculator", {"expr": "1"}, cid=f"bad{i}"))
            script.append(call("calculator", {"expression": "1+1"}, cid=f"ok{i}"))
        script.append(answer("2"))
        state = Agent(ScriptedLLM(script), registry(calculator_tool()),
                      AgentConfig(max_format_errors=3, max_repeat=10)).run("x")
        self.assertNotEqual(state.status, "format_errors")

    def test_unknown_tool(self):
        r = registry(calculator_tool()).execute("rm_rf", "{}")
        self.assertEqual(r.error_type, "unknown_tool")

    def test_timeout_then_retry(self):
        slow = Tool("slow", "", {"type": "object", "properties": {}}, lambda: time.sleep(1), timeout_s=0.05)
        r = registry(slow, max_retries=1).execute("slow", "{}")
        self.assertEqual((r.ok, r.error_type, r.attempts), (False, "timeout", 2))

    def test_transient_error_recovers(self):
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise TransientToolError("网络抖动")
            return "ok"
        r = registry(Tool("flaky", "", {"type": "object", "properties": {}}, flaky), max_retries=2).execute("flaky", "{}")
        self.assertEqual((r.ok, r.attempts), (True, 3))

    def test_non_idempotent_tool_not_retried(self):
        state = {"n": 0}

        def send():
            state["n"] += 1
            raise TransientToolError("网络抖动")
        t = Tool("send_email", "", {"type": "object", "properties": {}}, send, idempotent=False)
        r = registry(t, max_retries=3).execute("send_email", "{}")
        self.assertEqual((r.attempts, state["n"]), (1, 1))

    def test_large_output_truncated(self):
        big = Tool("big", "", {"type": "object", "properties": {}}, lambda: "A" * 500 + "B" * 9000 + "Z" * 500)
        r = registry(big, max_output_chars=1000).execute("big", "{}")
        self.assertTrue(r.content.startswith("A") and r.content.endswith("Z"))
        self.assertIn("已截断", r.content)


class TestFingerprint(unittest.TestCase):
    def test_key_order_does_not_matter(self):
        from harness.agent import fingerprint
        self.assertEqual(fingerprint("t", '{"a": 1, "b": 2}'), fingerprint("t", '{"b":2,"a":1}'))


class TestSafety(unittest.TestCase):
    def test_calculator_rejects_code(self):
        with self.assertRaises(ValueError):
            calculate("__import__('os').system('ls')")
        self.assertEqual(calculate("(0.05*400)+(0.12*250)"), "50.0")

    def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            reg = registry(*make_file_tools(d))
            r = reg.execute("read_file", json.dumps({"path": "../../etc/passwd"}))
            self.assertFalse(r.ok)
            self.assertIn("PermissionError", r.content)


class TestContext(unittest.TestCase):
    def _long_history(self, n=10):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "目标：算账"}]
        for i in range(n):
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"c{i}", "type": "function",
                                         "function": {"name": "read_file", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 900})
        return msgs

    def test_compaction_keeps_goal_and_tool_pairs(self):
        msgs = self._long_history()
        out, compacted = ContextManager(limit_tokens=1000, keep_recent_units=3).maybe_compact(msgs)
        self.assertTrue(compacted)
        self.assertEqual(out[1]["content"], "目标：算账")         # 任务目标被钉住
        self.assertIn("较早对话的摘要", out[2]["content"])
        self.assertEqual(len(split_units(out)), 2 + 1 + 3)
        self.assertTrue(check_tool_pairs(out))                    # 工具调用与结果没被拆开

    def test_no_compaction_under_limit(self):
        msgs = self._long_history(1)
        _, compacted = ContextManager(limit_tokens=100_000).maybe_compact(msgs)
        self.assertFalse(compacted)


class TestCheckpoint(unittest.TestCase):
    def test_resume_after_crash(self):
        with tempfile.TemporaryDirectory() as d:
            cp = Checkpointer(d)
            # 第一次运行：执行一轮工具后模型服务"崩溃"（脚本用完抛错）
            first = Agent(ScriptedLLM([call("calculator", {"expression": "6*7"})]),
                          registry(calculator_tool()), checkpointer=cp)
            s1 = first.run("6*7=?", run_id="job1")
            self.assertEqual(s1.status, "llm_error")
            # 模拟服务重启：把状态改回 running 后从存档继续
            saved = cp.load("job1")
            saved.status = "running"
            cp.save(saved)
            second = Agent(ScriptedLLM([answer("42")]), registry(calculator_tool()), checkpointer=cp)
            s2 = second.resume("job1")
            self.assertEqual((s2.status, s2.final_answer, s2.turn), ("completed", "42", 2))
            self.assertEqual(s2.messages[3]["content"], "42")  # 第一轮的工具结果被保留

    def test_trace_written(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            Agent(ScriptedLLM([answer("hi")]), registry(), tracer=Tracer(path)).run("x")
            with open(path, encoding="utf-8") as f:
                events = [json.loads(line)["event"] for line in f]
            self.assertEqual(events, ["llm_call", "stop"])


if __name__ == "__main__":
    unittest.main()
