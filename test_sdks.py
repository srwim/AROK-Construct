#!/usr/bin/env python3
"""Real OpenAI and Anthropic SDKs through the proxy, against a stub upstream.
Skipped unless `pip install openai anthropic`. Run: cargo build && python -m unittest -v"""
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import construct as C

try:
    import anthropic
    import openai
except ImportError:
    openai = anthropic = None

SECRET = "sk-test-SECRET"


def sse(events):
    return b"".join((f"event: {e['type']}\n" if "type" in e else "").encode()
                    + b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)


class Stub(BaseHTTPRequestHandler):
    """Deterministic upstream speaking OpenAI chat completions and Anthropic messages."""
    seen = []  # request headers, for passthrough assertions

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append({k.lower(): v for k, v in self.headers.items()})
        m, n = req["model"], len(req["messages"])
        words = ["reply", f"{n}", f"from {m}"]
        if self.path.endswith("/chat/completions"):
            base = {"id": "c1", "created": 0, "model": m}
            if req.get("stream"):
                chunks = [{**base, "object": "chat.completion.chunk",
                           "choices": [{"index": 0, "delta": {"content": w + " "}, "finish_reason": None}]}
                          for w in words]
                chunks.append({**base, "object": "chat.completion.chunk",
                               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                body = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
                return self._out(body + b"data: [DONE]\n\n", "text/event-stream")
            return self._out(json.dumps({**base, "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": " ".join(words)},
                 "finish_reason": "stop"}]}).encode(), "application/json")
        msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": m,
               "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 3}}
        if req.get("stream"):
            events = [{"type": "message_start", "message": {**msg, "content": [], "stop_reason": None}},
                      {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}]
            events += [{"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": w + " "}} for w in words]
            events += [{"type": "content_block_stop", "index": 0},
                       {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 3}},
                       {"type": "message_stop"}]
            return self._out(sse(events), "text/event-stream")
        return self._out(json.dumps({**msg, "content": [{"type": "text", "text": " ".join(words)}],
                                     "stop_reason": "end_turn"}).encode(), "application/json")

    def _out(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@unittest.skipUnless(openai and anthropic, "pip install openai anthropic")
class TestSDKs(unittest.TestCase):
    def setUp(self):
        self.db = os.environ["CONSTRUCT_DB"] = os.path.join(tempfile.mkdtemp(prefix="construct-sdk-"), "t.db")
        self.up = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=self.up.serve_forever, daemon=True).start()
        self.upstream = f"http://127.0.0.1:{self.up.server_address[1]}"
        Stub.seen.clear()
        self.procs = []

    def tearDown(self):
        self.up.shutdown()
        self.up.server_close()
        for p in self.procs:
            p.terminate()
            p.wait()

    def serve(self, *a, **kw):
        p, url = C.serve(*a, **kw)
        self.procs.append(p)
        return url

    def no_secrets_stored(self):
        with sqlite3.connect(self.db) as db:
            hits = db.execute("SELECT count(*) FROM objects WHERE instr(data, ?)", (SECRET.encode(),)).fetchone()[0]
        self.assertEqual(hits, 0, "an API key leaked into the object store")

    def test_openai_record_then_offline_replay(self):
        def run(url):
            c = openai.OpenAI(api_key=SECRET, base_url=f"{url}/v1", organization="org-x")
            msgs = [{"role": "user", "content": "hi"}]
            plain = c.chat.completions.create(model="gpt-test", messages=msgs).choices[0].message.content
            streamed = "".join(ch.choices[0].delta.content or "" for ch in
                               c.chat.completions.create(model="gpt-test", messages=msgs, stream=True))
            return plain, streamed

        live = run(self.serve("record", "base", upstream=self.upstream))
        self.assertEqual(live, ("reply 1 from gpt-test", "reply 1 from gpt-test "))
        self.assertEqual(Stub.seen[0].get("openai-organization"), "org-x")
        self.assertEqual(Stub.seen[0].get("authorization"), f"Bearer {SECRET}")
        self.up.shutdown()
        self.assertEqual(run(self.serve("replay", "r1", parent="base")), live)
        self.no_secrets_stored()

    def test_anthropic_record_then_offline_replay(self):
        def run(url):
            c = anthropic.Anthropic(api_key=SECRET, base_url=url)
            kw = dict(model="claude-test", max_tokens=64, messages=[{"role": "user", "content": "hi"}],
                      extra_headers={"anthropic-beta": "test-beta-2026"})
            plain = c.messages.create(**kw).content[0].text
            with c.messages.stream(**kw) as s:
                streamed = s.get_final_message().content[0].text
            return plain, streamed

        live = run(self.serve("record", "base", upstream=self.upstream))
        self.assertEqual(live, ("reply 1 from claude-test", "reply 1 from claude-test "))
        self.assertEqual(Stub.seen[0].get("anthropic-beta"), "test-beta-2026")
        self.assertEqual(Stub.seen[0].get("x-api-key"), SECRET)
        self.up.shutdown()
        self.assertEqual(run(self.serve("replay", "r1", parent="base")), live)
        self.no_secrets_stored()

    def test_replay_miss_fails_fast_without_sdk_retries(self):
        self.serve("record", "empty", upstream=self.upstream)
        url = self.serve("replay", "r1", parent="empty")
        c = openai.OpenAI(api_key=SECRET, base_url=f"{url}/v1")  # default max_retries=2
        t = time.monotonic()
        with self.assertRaises(openai.ConflictError):
            c.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": "new"}])
        self.assertLess(time.monotonic() - t, 0.4, "SDK retried a deterministic replay miss")


if __name__ == "__main__":
    unittest.main()
