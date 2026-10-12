#!/usr/bin/env python3
"""SDK tests against the real binary. Run: cargo build && python -m unittest -v
(Proxy, store, SSE, and differ tests live in src/main.rs: cargo test.)"""
import os
import tempfile
import unittest
import urllib.error

import construct as C

NO_LLM = "http://127.0.0.1:9"  # effects never touch the upstream


class TestWrap(unittest.TestCase):
    def setUp(self):
        os.environ["CONSTRUCT_DB"] = os.path.join(tempfile.mkdtemp(prefix="construct-test-"), "t.db")
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            p.terminate()
            p.wait()

    def serve(self, *a, **kw):
        p, url = C.serve(*a, **kw)
        self.procs.append(p)
        return url

    def test_record_then_replay_skips_execution(self):
        runs = {"n": 0}

        def flaky(q):
            runs["n"] += 1
            return {"hits": [q, runs["n"]]}

        f = C.wrap(flaky, proxy=self.serve("record", "base", upstream=NO_LLM))
        self.assertEqual(f("x"), {"hits": ["x", 1]})
        self.assertEqual(f("x"), {"hits": ["x", 2]})  # same args, distinct call# -> distinct result
        self.assertEqual(runs["n"], 2)

        g = C.wrap(flaky, proxy=self.serve("replay", "r1", parent="base"))
        self.assertEqual(g("x"), {"hits": ["x", 1]})
        self.assertEqual(g("x"), {"hits": ["x", 2]})
        self.assertEqual(runs["n"], 2)  # never executed during replay

    def test_replay_miss_409(self):
        C.wrap(lambda: 1, proxy=self.serve("record", "base", upstream=NO_LLM), name="t")()
        h = C.wrap(lambda: 1, proxy=self.serve("replay", "r1", parent="base"), name="never_recorded")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            h()
        self.assertEqual(cm.exception.code, 409)

    def test_serve_reports_startup_failure(self):
        self.serve("record", "base", upstream=NO_LLM)
        with self.assertRaises(RuntimeError):
            C.serve("record", "base", upstream=NO_LLM)  # run name taken


if __name__ == "__main__":
    unittest.main()
