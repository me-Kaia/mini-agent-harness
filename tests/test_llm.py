"""OpenAICompatClient 请求体的测试：temperature / top_p 可设置也可不发送，extra_body 合并额外字段。
不联网：用假的 urlopen 截获请求体。
运行：python -m unittest discover -s tests -v
"""
import io
import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evals"))

from harness import OpenAICompatClient
from run_eval import parse_extra_body

FAKE_RESPONSE = {"choices": [{"message": {"content": "答案：1"}}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def sent_body(client: OpenAICompatClient) -> dict:
    """调用一次 chat，返回实际发出去的请求体。"""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return io.BytesIO(json.dumps(FAKE_RESPONSE).encode("utf-8"))

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        resp = client.chat([{"role": "user", "content": "1"}])
    assert resp.content == "答案：1"
    return captured["body"]


def client(**kw) -> OpenAICompatClient:
    return OpenAICompatClient("http://example.test/v1", "m", "k", **kw)


class TestSamplingParams(unittest.TestCase):
    def test_default_keeps_old_behavior(self):
        # 默认和改动前一致：发 temperature=0.0，不发 top_p
        body = sent_body(client())
        self.assertEqual(body["temperature"], 0.0)
        self.assertNotIn("top_p", body)

    def test_set_temperature_and_top_p(self):
        body = sent_body(client(temperature=0.6, top_p=0.95))
        self.assertEqual(body["temperature"], 0.6)
        self.assertEqual(body["top_p"], 0.95)

    def test_none_means_not_sent(self):
        body = sent_body(client(temperature=None, top_p=None))
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)

    def test_zero_is_sent_not_omitted(self):
        # 0 是合法取值，不能被当成"不发送"
        body = sent_body(client(temperature=0, top_p=0))
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["top_p"], 0)


class TestExtraBody(unittest.TestCase):
    def test_extra_fields_are_merged(self):
        body = sent_body(client(extra_body={"thinking": {"type": "disabled"}}))
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["model"], "m")
        self.assertEqual(body["messages"], [{"role": "user", "content": "1"}])

    def test_cannot_override_core_fields(self):
        for key in ("model", "messages", "tools"):
            with self.assertRaises(ValueError, msg=key):
                client(extra_body={key: "x"})

    def test_cannot_duplicate_sampling_params(self):
        # temperature / top_p 只能走专门的参数，避免两处设置不一致
        for key in ("temperature", "top_p"):
            with self.assertRaises(ValueError, msg=key):
                client(extra_body={key: 0.5})

    def test_extra_body_not_shared_between_calls(self):
        extra = {"thinking": {"type": "disabled"}}
        c = client(extra_body=extra)
        sent_body(c)
        extra["thinking"] = "changed"  # 外部修改不应影响已创建的客户端
        self.assertEqual(sent_body(c)["thinking"], {"type": "disabled"})


class TestConnectionErrors(unittest.TestCase):
    def test_remote_disconnect_is_retried_then_llm_error(self):
        # 服务端直接断开连接时 urlopen 抛 http.client.RemoteDisconnected（不是 URLError），
        # 不能让它冒出去中断整个评测，应当按临时错误重试，最终转成 LLMError
        import http.client
        from harness.llm import LLMError
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(1)
            raise http.client.RemoteDisconnected("Remote end closed connection without response")

        with mock.patch("urllib.request.urlopen", fake_urlopen), mock.patch("time.sleep"):
            with self.assertRaises(LLMError):
                client(max_retries=2).chat([{"role": "user", "content": "1"}])
        self.assertEqual(len(calls), 3)

    def test_recovers_after_disconnect(self):
        import http.client
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                raise ConnectionResetError("reset")
            return io.BytesIO(json.dumps(FAKE_RESPONSE).encode("utf-8"))

        with mock.patch("urllib.request.urlopen", fake_urlopen), mock.patch("time.sleep"):
            resp = client().chat([{"role": "user", "content": "1"}])
        self.assertEqual(resp.content, "答案：1")


def http_429(retry_after=None):
    import email.message
    import urllib.error
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://example.test/v1/chat/completions", 429, "Too Many Requests",
                                  hdrs, io.BytesIO(b""))


def run_with_failures(c, failures, sleeps):
    """前 len(failures) 次请求依次抛出 failures 里的异常，之后成功；sleeps 记录每次 sleep 的秒数（不真睡）。"""
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return io.BytesIO(json.dumps(FAKE_RESPONSE).encode("utf-8"))

    with mock.patch("urllib.request.urlopen", fake_urlopen), mock.patch("harness.llm.time.sleep", sleeps.append):
        return c.chat([{"role": "user", "content": "1"}])


class TestRateLimitRetry(unittest.TestCase):
    def test_429_waits_retry_after(self):
        sleeps = []
        resp = run_with_failures(client(), [http_429("7")], sleeps)
        self.assertEqual(sleeps, [7.0])
        self.assertEqual(resp.retries, 1)
        self.assertEqual(resp.wait_s, 7.0)

    def test_429_without_header_backs_off_10_to_60(self):
        sleeps = []
        resp = run_with_failures(client(max_retries=5), [http_429()] * 5, sleeps)
        self.assertEqual(sleeps, [10, 20, 40, 60, 60])
        self.assertEqual(resp.retries, 5)

    def test_unparseable_retry_after_falls_back_to_backoff(self):
        sleeps = []
        run_with_failures(client(), [http_429("soon")], sleeps)
        self.assertEqual(sleeps, [10])

    def test_network_errors_keep_short_backoff(self):
        import http.client
        sleeps = []
        run_with_failures(client(), [http.client.RemoteDisconnected("x"), ConnectionResetError("y")], sleeps)
        self.assertEqual(sleeps, [1, 2])

    def test_max_retries_zero_means_no_retry(self):
        from harness.llm import LLMError
        sleeps = []
        with self.assertRaises(LLMError) as cm:
            run_with_failures(client(max_retries=0), [http_429("5")], sleeps)
        self.assertEqual(sleeps, [])
        self.assertEqual(cm.exception.retries, 0)

    def test_llm_error_carries_retries_and_wait(self):
        from harness.llm import LLMError
        sleeps = []
        with self.assertRaises(LLMError) as cm:
            run_with_failures(client(max_retries=2), [http_429("3")] * 3, sleeps)
        self.assertEqual(cm.exception.retries, 2)
        self.assertEqual(cm.exception.wait_s, 6.0)

    def test_latency_excludes_wait(self):
        # latency_s 只算实际请求的时间；退避等待单独记在 wait_s
        sleeps = []
        resp = run_with_failures(client(), [http_429("30")], sleeps)
        self.assertEqual(resp.wait_s, 30.0)
        self.assertLess(resp.latency_s, 5.0)


class TestParseRetryAfter(unittest.TestCase):
    def test_seconds_and_http_date(self):
        from harness.llm import parse_retry_after
        self.assertEqual(parse_retry_after("12"), 12.0)
        self.assertEqual(parse_retry_after(" 1.5 "), 1.5)
        now = 1_700_000_000.0  # 2023-11-14 22:13:20 GMT
        self.assertEqual(parse_retry_after("Tue, 14 Nov 2023 22:13:50 GMT", now=now), 30.0)

    def test_invalid_or_negative(self):
        from harness.llm import parse_retry_after
        for bad in (None, "", "soon", "-3"):
            self.assertIsNone(parse_retry_after(bad), bad)
        self.assertEqual(parse_retry_after("Tue, 14 Nov 2023 22:13:00 GMT", now=1_700_000_000.0), 0.0)

    def test_capped(self):
        from harness.llm import MAX_RETRY_AFTER_S, parse_retry_after
        self.assertEqual(parse_retry_after("99999"), MAX_RETRY_AFTER_S)


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


class TestRateLimiter(unittest.TestCase):
    def test_min_interval_between_requests(self):
        from harness.llm import RateLimiter
        clk = FakeClock()
        lim = RateLimiter(3.0, clock=clk.time, sleep=clk.sleep)
        self.assertEqual(lim.wait(), 0.0)      # 第一次不等
        clk.now += 1.0
        self.assertEqual(lim.wait(), 2.0)      # 距上次只过了 1 秒，补足到 3 秒
        clk.now += 5.0
        self.assertEqual(lim.wait(), 0.0)      # 已超过间隔，不等
        self.assertEqual(clk.sleeps, [2.0])

    def test_zero_interval_never_waits(self):
        from harness.llm import RateLimiter
        clk = FakeClock()
        lim = RateLimiter(0, clock=clk.time, sleep=clk.sleep)
        self.assertEqual([lim.wait(), lim.wait()], [0.0, 0.0])

    def test_shared_between_clients_and_counted_as_wait(self):
        # 评测时每个 trial 新建一个客户端，限速器必须共享才能真正控制请求间隔；限速等待计入 wait_s
        from harness.llm import RateLimiter
        clk = FakeClock()
        lim = RateLimiter(3.0, clock=clk.time, sleep=clk.sleep)
        sent_body(client(rate_limiter=lim))
        with mock.patch("urllib.request.urlopen",
                        lambda req, timeout=None: io.BytesIO(json.dumps(FAKE_RESPONSE).encode("utf-8"))):
            resp = client(rate_limiter=lim).chat([{"role": "user", "content": "1"}])
        self.assertEqual(clk.sleeps, [3.0])
        self.assertEqual(resp.wait_s, 3.0)
        self.assertEqual(resp.retries, 0)


class TestParseExtraBody(unittest.TestCase):
    def test_empty_means_none(self):
        self.assertEqual(parse_extra_body(None), {})
        self.assertEqual(parse_extra_body(""), {})

    def test_json_object(self):
        self.assertEqual(parse_extra_body('{"thinking": {"type": "disabled"}}'),
                         {"thinking": {"type": "disabled"}})

    def test_rejects_invalid_json_and_non_object(self):
        for bad in ("{thinking}", "[1, 2]", '"x"', "1"):
            with self.assertRaises(ValueError, msg=bad):
                parse_extra_body(bad)


if __name__ == "__main__":
    unittest.main()
