#!/usr/bin/env python3
"""End-to-end: record a 3-step agent run (wrapped nondeterministic tool, streamed
final step), replay it offline, fork it with a mutated prompt + substituted
model, diff the branches, and render the HTML timeline.

Run:  cargo build && python demo.py   (no network, no API keys)
"""
import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DB = os.environ["CONSTRUCT_DB"] = os.path.join(tempfile.mkdtemp(prefix="construct-demo-"), "demo.db")
import construct as C  # noqa: E402

SEARCH_EXECUTIONS = {"n": 0}  # proves replay never runs the real tool


def search(query, limit):
    SEARCH_EXECUTIONS["n"] += 1
    return {"results": [f"result {i + 1} about {query}" for i in range(limit)]}


# ---- stub "LLM" upstream: deterministic, OpenAI-shaped, supports streaming ----
class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs, model = req["messages"], req.get("model", "stub-1")
        topic = msgs[0]["content"].split()[-1]
        brief = any("brief" in m.get("content", "")
                    for m in msgs if isinstance(m.get("content"), str))
        n = sum(m["role"] == "assistant" for m in msgs)
        if n == 0:  # step 1: plan
            msg = {"role": "assistant",
                   "content": f"Plan: search for {topic}, then summarize the findings carefully."}
        elif n == 1:  # step 2: tool call
            args = {"query": topic, "limit": 1 if brief else 3}
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "search", "arguments": json.dumps(args)}}]}
        else:  # step 3: summary
            tool = next(m["content"] for m in msgs if m["role"] == "tool")
            style = "Briefly:" if brief else "In detail:"
            msg = {"role": "assistant",
                   "content": f"{style} {topic} summary from [{tool}] (model={model})"}
        if req.get("stream"):
            words = (msg["content"] or "").split(" ")
            half = len(words) // 2
            chunks = [{"model": model, "choices": [{"index": 0, "delta":
                       {"role": "assistant", "content": " ".join(words[:half]) + " "}}]},
                      {"model": model, "choices": [{"index": 0, "delta":
                       {"content": " ".join(words[half:])}}]}]
            body = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
            self._out(body + b"data: [DONE]\n\n", "text/event-stream")
        else:
            body = json.dumps({"id": "x", "object": "chat.completion", "model": model,
                               "choices": [{"index": 0, "message": msg,
                                            "finish_reason": "stop"}]}).encode()
            self._out(body, "application/json")

    def _out(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def sse_text(body):
    """Join OpenAI stream chunks into one assistant message (what an SDK does for you)."""
    text = "".join(json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                   for line in body.decode().splitlines()
                   if line.startswith("data: ") and line != "data: [DONE]")
    return {"role": "assistant", "content": text}


# ---- a 3-step agent: plan -> tool call (real wrapped tool) -> streamed summary ----
def agent(base, prompt, step2="Proceed."):
    msgs = [{"role": "user", "content": prompt}]
    do_search = C.wrap(search, proxy=base)

    def chat(stream=False):
        payload = {"model": "stub-1", "messages": msgs, **({"stream": True} if stream else {})}
        r = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(payload).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r) as resp:
            body = resp.read()
        return sse_text(body) if stream else json.loads(body)["choices"][0]["message"]

    out = [chat()]                                   # step 1: plan
    msgs += [out[-1], {"role": "user", "content": step2}]
    out.append(chat())                               # step 2: tool call
    msgs.append(out[-1])
    args = json.loads(out[-1]["tool_calls"][0]["function"]["arguments"])
    hits = do_search(args["query"], args["limit"])   # nondeterministic tool, recorded
    msgs.append({"role": "tool", "tool_call_id": "call_1",
                 "content": f"{len(hits['results'])} results about {args['query']}"})
    out.append(chat(stream=True))                    # step 3: streamed summary
    return out


def stub():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def stop(p):
    p.terminate()
    p.wait()


def timeline(run):
    with sqlite3.connect(DB) as db:
        return db.execute("SELECT type, src FROM events WHERE run = ? ORDER BY seq", (run,)).fetchall()


up, UP = stub()

print("1. RECORD: agent runs live through the proxy (tool + streamed step included)")
rec, url = C.serve("record", "base", upstream=UP)
a = agent(url, "Research and summarize: volcanoes")
stop(rec)
print(f"   {len(timeline('base'))} events recorded; searches executed: {SEARCH_EXECUTIONS['n']}")
print(f"   final answer: {a[2]['content']!r}")

print("2. REPLAY: upstream DOWN; whole run (LLM + tool + stream) replays from the log")
up.shutdown()
up.server_close()
rep, url = C.serve("replay", "replay-1", parent="base")
b = agent(url, "Research and summarize: volcanoes")
stop(rep)
assert a == b, "replay diverged from recording!"
assert SEARCH_EXECUTIONS["n"] == 1, "replay executed the real tool!"
print("   replay == recording, tool never re-executed, fully offline OK")

print("3. FORK: mutate the step-2 prompt; pre-fork cached, post-fork live as stub-2")
up, UP = stub()
frk, url = C.serve("fork", "brief-fork", upstream=UP, parent="base", model="stub-2")
c = agent(url, "Research and summarize: volcanoes", step2="Proceed, but be brief.")
stop(frk)
srcs = timeline("brief-fork")
expected = [("llm_call", "cache"), ("llm_call", "live"), ("effect", "live"), ("llm_call", "live")]
assert srcs == expected, f"unexpected timeline: {srcs}"
print(f"   timeline sources: {[s for _, s in srcs]}; forked answer: {c[2]['content']!r}")

print("4. DIFF: divergence report")
subprocess.run([C.binary(), "diff", "base", "brief-fork"], check=True)

print("5. VIEW: HTML timeline")
subprocess.run([C.binary(), "view", "base", "brief-fork", "-o", "construct-timeline.html"], check=True)
print(f"\nevent log (one SQLite file): {DB}")
