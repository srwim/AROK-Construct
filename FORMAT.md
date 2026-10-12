# AROK Construct storage format (v2)

One SQLite file (`$CONSTRUCT_DB`, default `./construct.db`), WAL mode,
`PRAGMA user_version = 2`. `src/main.rs` (`SCHEMA`) is the executable spec.

## Tables

```sql
objects (hash TEXT PRIMARY KEY, data BLOB)            -- content-addressed, write-once
runs    (name PRIMARY KEY, mode, parent -> runs, upstream, model_sub, ts)
events  (run -> runs, seq, type, ts, path, name, key, req -> objects, resp -> objects,
         status, src, ctype, PRIMARY KEY (run, seq))
```

- **objects**: `hash = sha256(data)`. A blob is the canonical-JSON bytes of a
  request, effect args, or effect result, or the raw bytes of a response
  (including raw SSE streams). `INSERT OR IGNORE`: identical blobs across
  thousands of runs are stored once.
- **runs**: one row per run/branch (the v1 `meta` line). `parent` is the run
  this one replays or forks from. Duplicate names are refused by the primary key.
- **events**: one row per LLM round-trip (`type = 'llm_call'`) or recorded
  nondeterministic function call (`type = 'effect'`), in `seq` order.

Append-only is enforced by the database, not the app: triggers abort any
`UPDATE` or `DELETE` on `events`, any `UPDATE` on `runs` or `objects`.
Foreign keys and `CHECK` constraints guard `mode`, `type`, and `src`.

Canonical JSON = sorted keys, no whitespace (`{"a":[1,2],"b":1}`).

## Event fields

- `key`: request identity. `llm_call`: sha256 of `canon([path, request])`
  **before** any mutation. `effect`: sha256 of `canon([name, call#, args])`;
  the call# makes repeated identical calls (e.g. `time()`) replay in order.
- `req`: object hash of the request actually sent (after model substitution)
  or the effect's args.
- `resp`: object hash of the raw response body or the effect's result.
- `src`: `live` (executed) or `cache` (served from the parent run).
- `path` (`llm_call`), `name` (`effect`).
- `ctype` (`llm_call`): response Content-Type; `text/event-stream` means `resp`
  is a raw SSE stream, reassembled on demand for diffing and viewing.

The branch point of a fork is the first event whose `key` is absent from the
parent: derivable, not stored.

## Replay semantics

Replay loads the parent's `key -> (status, resp, ctype)` map and serves matches
from `objects`. An agent fed byte-identical responses produces byte-identical
next requests, so every step hits the cache.

| mode   | cache hit | cache miss |
|--------|-----------|------------|
| record | n/a (always live) | live, recorded |
| replay | serve from store | **409**: entropy upstream of the LLM; wrap it as an effect |
| fork   | serve from store | go live (optionally rewrite `model`), record |

Fork at step N is replay with a mutated prompt. Model substitution is fork +
`--model`: cached steps stay cached, live steps are rewritten.

Upstream transport failures return **502** and are not recorded.

## Effects

`POST /construct/effect`:

1. query `{name, n, args}`: hit returns `{"cached": true, "result": ...}` and
   the function is **not** executed; miss in replay is 409; miss in
   record/fork returns `{"cached": false}`
2. the client executes the function and posts `{name, n, args, result}`
   (refused with 409 in replay)

Args and results must be JSON-serializable.

## Not yet: add when something needs it

Pass-through chunk pacing during record (SSE is buffered whole), response-header
capture, retention/compaction, migration of v1 JSONL logs (no v1 users yet).
