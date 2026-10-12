#!/usr/bin/env python3
"""Live check: record a real Claude agent run, replay it offline, render the timeline.

The agent reads the clock through a tool, so a naive re-run would differ;
Construct records the tool result and replay serves it back.

Needs ANTHROPIC_API_KEY and `pip install anthropic`. Costs well under a cent.
Run:  python live_check.py        (CONSTRUCT_MODEL picks the model)
"""
import datetime
import os
import subprocess
import tempfile

import anthropic

os.environ["CONSTRUCT_DB"] = os.path.join(tempfile.mkdtemp(prefix="construct-live-"), "live.db")
import construct as C  # noqa: E402

MODEL = os.environ.get("CONSTRUCT_MODEL", "claude-haiku-5-5")
UPSTREAM = os.environ.get("CONSTRUCT_UPSTREAM", "https://api.anthropic.com")
TOOLS = [{"name": "now", "description": "The current local date and time, ISO 8601.",
          "input_schema": {"type": "object", "properties": {}}}]
CLOCK_READS = {"n": 0}


def now():
    CLOCK_READS["n"] += 1
    return datetime.datetime.now().isoformat(timespec="seconds")


def agent(url):
    client = anthropic.Anthropic(base_url=url)
    clock = C.wrap(now, proxy=url)
    msgs = [{"role": "user", "content": "What time is it? Use the tool, then answer in one short sentence."}]
    for _ in range(4):
        with client.messages.stream(model=MODEL, max_tokens=200, tools=TOOLS, messages=msgs) as s:
            m = s.get_final_message()
        msgs.append({"role": "assistant", "content": m.content})
        if m.stop_reason != "tool_use":
            return "".join(b.text for b in m.content if b.type == "text")
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": b.id, "content": clock()}
                                                 for b in m.content if b.type == "tool_use"]})
    raise RuntimeError("agent did not finish in 4 turns")


def run(mode, name, **kw):
    proc, url = C.serve(mode, name, **kw)
    try:
        return agent(url)
    finally:
        proc.terminate()
        proc.wait()


live = run("record", "live", upstream=UPSTREAM)
reads = CLOCK_READS["n"]
print(f"1. recorded against {UPSTREAM} ({MODEL}): {live!r}; clock read {reads}x")

try:
    replayed = run("replay", "live-replay", parent="live")
except anthropic.ConflictError as e:
    raise SystemExit(f"2. replay missed: something in a request changed between runs.\n   {e}")
assert replayed == live, f"replay diverged: {replayed!r}"
assert CLOCK_READS["n"] == reads, "replay read the real clock"
print("2. replayed offline: identical answer, clock never read")

subprocess.run([C.binary(), "view", "live", "live-replay", "-o", "live-timeline.html"], check=True)
print(f"3. open live-timeline.html; event log in {os.environ['CONSTRUCT_DB']}")
