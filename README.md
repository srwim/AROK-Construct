# AROK Construct

Deterministic record, replay, fork, and diff for LLM agent runs.

A local reverse proxy records every LLM round-trip (streamed or not) plus any
nondeterministic tool results into one SQLite file: content-addressed blobs and
an append-only event log the database itself refuses to rewrite. Replay a run
fully offline, fork it at any step with a mutated prompt, substitute the model
from the fork point forward, and diff the two timelines.

Works with anything that speaks OpenAI/Anthropic-compatible HTTP (LangChain,
CrewAI, raw SDKs):

```sh
OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

## Build and try it

```sh
cargo build --release
cargo test               # proxy, store, streaming, differ
pip install openai anthropic   # optional: enables the real-SDK tests
python -m unittest       # Python SDK + real OpenAI/Anthropic SDKs through the binary
python demo.py           # record, replay offline, fork, diff, render timeline
python live_check.py     # real Claude agent: record, replay offline (needs ANTHROPIC_API_KEY)
```

No network, no API keys. The demo records a 3-step agent run (a real tool call
and a streamed response) against a stub LLM, replays it byte-identically with
the upstream killed and the tool never re-executed, forks it with a mutated
prompt and substituted model, prints the divergence report, and writes
`construct-timeline.html`.

## Usage

```sh
construct record --run base --upstream https://api.openai.com
construct replay --run r1 --from base                  # offline; miss = 409
construct fork   --run b1 --from base --upstream ... --model gpt-x
construct diff   base b1                               # divergence report
construct view   base b1 -o timeline.html              # visual timeline
construct runs                                         # list recorded runs
```

`CONSTRUCT_DB` picks the database (default `./construct.db`). Client headers
(auth, `anthropic-beta`, `OpenAI-Organization`, ...) pass through to the upstream
and are never stored. Replay misses carry `x-should-retry: false`, so the OpenAI
and Anthropic SDKs fail fast instead of retrying a deterministic 409.

Nondeterministic tools (search, time, random) record and replay through the
same proxy. Wrap them once and replay never re-executes them:

```python
import construct
search = construct.wrap(search)   # args/result must be JSON-serializable
```

`construct.py` is the whole Python SDK: stdlib only, also exposes
`construct.serve(...)` to start the proxy from Python.

## How it works

Replay is a `sha256(canonical request) -> response` cache, so fork isn't a
feature: replaying with a mutated prompt diverges automatically at the first
changed hash, and cache misses go live (recording the new branch). Streamed
responses are stored as raw SSE bytes and replayed byte-identically; they are
reassembled only for the differ and viewer. See [FORMAT.md](FORMAT.md).

Because the log is plain SQLite, audit queries need no tooling:

```sh
sqlite3 construct.db "SELECT run, count(*) FROM events WHERE src='live' GROUP BY run"
```

## The differ

Token-level word diff for response text, schema-aware diff for tool calls
(argument JSON is parsed before comparing, so formatting never diffs), and
structural diff for requests and effect args/results. The same engine drives
`diff` (terminal) and `view` (self-contained HTML, light and dark).

## Status

CI builds and tests on Windows, Linux, and macOS; tagging `v*` attaches
binaries to a GitHub release.

v0.1: single Rust binary, six dependencies (rusqlite, serde_json, sha2,
similar, tiny_http, ureq). Not yet: pass-through chunk pacing during record
(SSE is buffered whole), response-header capture, retention. Each is listed in
FORMAT.md with the condition that justifies building it.
