"""AROK Construct Python SDK (stdlib only).

    import construct
    search = construct.wrap(search)   # records live, replays without executing

    proc, url = construct.serve("record", "base", upstream="https://api.openai.com")
"""
import itertools
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

PROXY = os.environ.get("CONSTRUCT_PROXY", "http://127.0.0.1:8787")


def wrap(fn, proxy=None, name=None):
    """Route a nondeterministic function's results through the proxy so they
    record and replay. Args and result must be JSON-serializable. Each call is
    identified by (name, call#), so time()-style repeat calls replay in order."""
    label, count, base = name or fn.__name__, itertools.count(), proxy or PROXY

    def wrapped(*args, **kwargs):
        payload = {"name": label, "n": next(count), "args": [list(args), kwargs]}
        r = _post(base + "/construct/effect", payload)
        if r.get("cached"):
            return r["result"]
        result = fn(*args, **kwargs)
        _post(base + "/construct/effect", {**payload, "result": result})
        return result

    return wrapped


def _post(url, obj):
    r = urllib.request.Request(url, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read())


def binary():
    """Path to the construct binary: $CONSTRUCT_BIN, a cargo build, or a release
    download saved next to this file."""
    here = Path(__file__).parent
    for p in (os.environ.get("CONSTRUCT_BIN"), here / "target/release/construct",
              here / "target/debug/construct", here / "construct", here / "construct-windows-x64",
              here / "construct-linux-x64", here / "construct-macos-arm64"):
        for candidate in (p and Path(p), p and Path(f"{p}.exe")):
            if candidate and candidate.is_file():
                return str(candidate)
    raise FileNotFoundError("construct binary not found: put the release download next to construct.py, "
                            "run `cargo build`, or set CONSTRUCT_BIN")


def serve(mode, run, upstream=None, parent=None, model=None, port=0):
    """Start a proxy subprocess; returns (process, base_url). Uses $CONSTRUCT_DB."""
    args = [binary(), mode, "--run", run, "--port", str(port)]
    for flag, v in (("--upstream", upstream), ("--from", parent), ("--model", model)):
        if v:
            args += [flag, v]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
    m = re.search(r"http://127\.0\.0\.1:\d+", p.stdout.readline())
    p.stdout.close()  # the proxy prints only its startup line
    if not m:
        p.wait()
        raise RuntimeError(f"construct {mode} failed to start (exit {p.returncode}; see stderr)")
    return p, m.group(0)
