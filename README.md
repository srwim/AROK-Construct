# AROK Construct

Deterministic record, replay, fork, and diff for LLM agent runs.

A local reverse proxy records every LLM round-trip (streamed or not) plus any
nondeterministic tool results into one SQLite file: content-addressed blobs and
an append-only event log the database itself refuses to rewrite. Replay a run
fully offline, fork it at any step with a mutated prompt, substitute the model
from the fork point forward, and diff the two timelines.

Agent- and provider-agnostic: works with anything that speaks OpenAI-,
Anthropic- or Ollama-compatible HTTP (LangChain, CrewAI, raw SDKs), against
cloud APIs or local models:

```sh
OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

## Local LLMs

Construct has no provider built in: `--upstream` is any URL, so agents running
against local models are recorded the same way as cloud ones, and replay is
fully offline either way.

```sh
# Ollama: native API (/api/chat, /api/generate) or its OpenAI-compatible /v1
construct record --run local --upstream http://127.0.0.1:11434
# then point the agent's Ollama base URL at http://127.0.0.1:8787

# llama.cpp server, vLLM, LM Studio: OpenAI-compatible
construct record --run local --upstream http://127.0.0.1:8080
OPENAI_BASE_URL=http://127.0.0.1:8787/v1

# compare two local models on the same run
construct fork --run qwen --from local --upstream http://127.0.0.1:11434 --model qwen2.5
construct diff local qwen
```

Responses are read in OpenAI, Anthropic and Ollama native shapes, streamed
(SSE or Ollama's line-delimited JSON) or not, so search, diff, the timeline
and evals see the model's text and tool calls rather than raw JSON.

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
construct search refund policy --run b1               # full-text search over every step
construct eval   b1 --checks checks.json --baseline base  # score; exit 1 on regression
construct mcp                                          # MCP server on stdio (read-only)
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

## Ask an agent about its own runs (MCP)

`construct mcp` serves the log to any MCP client (Claude Desktop, Claude Code,
Cursor) over stdio. The connection is opened read-only at the SQLite level, so
an agent can investigate the evidence but never alter it.

| Tool | Returns |
|---|---|
| `list_runs` | every run: mode, parent, substituted model, live vs replayed steps |
| `get_step` | one step's exact request and response, assistant text, tool calls |
| `diff_runs` | first divergence between two runs and the full divergence report |
| `search_runs` | BM25-ranked full-text hits across every step, with highlighted snippets |
| `eval_run` | pass/fail per check; with a baseline, regressions and fixes |

Each tool declares an `outputSchema` and returns typed `structuredContent`
alongside readable text. Failures (unknown run, step out of range) come back
as tool errors the model can read and correct, not protocol errors.

```sh
claude mcp add construct -e CONSTRUCT_DB=/path/to/construct.db -- construct mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{ "mcpServers": { "construct": {
    "command": "C:\\tools\\construct.exe", "args": ["mcp"],
    "env": { "CONSTRUCT_DB": "C:\\agents\\construct.db" } } } }
```

Then ask: "Where did run b1 diverge from base, and what did the model do differently?"

## Retrieval

`construct search` (and the `search_runs` tool) ranks every recorded step with
SQLite FTS5 and BM25: the newest input message, the assistant's text and tool
calls, and effect arguments and results, with English stemming. The index is
built in SQLite's temp schema and caught up incrementally from the append-only
log, so it works on the read-only MCP connection and never writes to the
evidence file. Every word of the query must match; FTS operators in user text
are treated literally.

## Evals

A checks file states what a good run looks like:

```json
{ "checks": [
  { "name": "cites the policy", "includes": ["refund", "30 days"] },
  { "name": "never promises", "step": "any", "excludes": ["guarantee"] },
  { "name": "answer is structured", "schema": {
      "type": "object", "required": ["decision", "reason"],
      "properties": { "decision": { "enum": ["approve", "deny"] } } } },
  { "name": "lookup args valid", "tool": "lookup_order", "schema": {
      "type": "object", "required": ["order_id"], "additionalProperties": false,
      "properties": { "order_id": { "type": "string", "minLength": 6 } } } }
] }
```

Text checks read the last LLM call by default (`"step": "any"` reads them all,
a number reads one). `schema` validates the output as JSON (markdown fences
are stripped); `tool` + `schema` validates the arguments of every call to that
tool, and fails if the tool was never called. The validator covers the common
JSON Schema subset (`type`, `enum`, `const`, `required`, `properties`,
`additionalProperties: false`, `items`, min/max bounds and lengths); any other
keyword is an error, never a silent pass.

`construct eval` exits 1 when a check fails. With `--baseline`, it exits 1
only on regressions (passed in the baseline, fails now) and reports fixes, so
record a baseline, `fork` it with a new prompt or model, and gate the change
in CI:

```sh
construct fork --run candidate --from base --upstream https://api.openai.com --model gpt-x
construct eval candidate --checks checks.json --baseline base
```

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
