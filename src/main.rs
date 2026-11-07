//! AROK Construct: record, replay, fork, and diff LLM agent runs.
//!
//! One binary, one SQLite file. Point any OpenAI/Anthropic-compatible client at
//! the proxy (`OPENAI_BASE_URL=http://127.0.0.1:8787/v1`) and run your agent.
//! FORMAT.md is the storage contract; this file is its implementation.

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use similar::{capture_diff_slices, Algorithm, DiffTag, TextDiff};
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt::Write as _;
use std::io::{Read, Write as _};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type R<T> = Result<T, Box<dyn Error>>;

const USAGE: &str = "AROK Construct: record, replay, fork, and diff LLM agent runs

usage:
  construct record --run NAME --upstream URL [--port 8787]
  construct replay --run NAME --from RUN [--port 8787]       offline; miss = 409
  construct fork   --run NAME --from RUN --upstream URL [--model M] [--port 8787]
  construct diff   RUN_A RUN_B                              divergence report
  construct view   RUN [RUN...] [-o timeline.html]          HTML timeline
  construct runs                                            list recorded runs
  construct search QUERY [--run RUN] [--limit 10]           full-text search over every step
  construct eval   RUN --checks FILE [--baseline RUN]        score a run; exit 1 on failure/regression
  construct mcp                                             MCP server on stdio (read-only)

env: CONSTRUCT_DB (default ./construct.db), CONSTRUCT_UPSTREAM (default for --upstream)";

// ---------- store: one SQLite file; objects are content-addressed, events append-only ----------

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA user_version = 2;
CREATE TABLE IF NOT EXISTS objects (hash TEXT PRIMARY KEY, data BLOB NOT NULL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS runs (
  name TEXT PRIMARY KEY, mode TEXT NOT NULL CHECK (mode IN ('record','replay','fork')),
  parent TEXT REFERENCES runs(name), upstream TEXT, model_sub TEXT, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  run TEXT NOT NULL REFERENCES runs(name), seq INTEGER NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('llm_call','effect')), ts REAL NOT NULL,
  path TEXT, name TEXT, key TEXT NOT NULL,
  req TEXT NOT NULL REFERENCES objects(hash), resp TEXT NOT NULL REFERENCES objects(hash),
  status INTEGER NOT NULL, src TEXT NOT NULL CHECK (src IN ('live','cache')), ctype TEXT,
  PRIMARY KEY (run, seq));
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
  BEGIN SELECT RAISE(ABORT, 'construct: event log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
  BEGIN SELECT RAISE(ABORT, 'construct: event log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_update BEFORE UPDATE ON runs
  BEGIN SELECT RAISE(ABORT, 'construct: runs are append-only'); END;
CREATE TRIGGER IF NOT EXISTS objects_no_update BEFORE UPDATE ON objects
  BEGIN SELECT RAISE(ABORT, 'construct: objects are write-once'); END;
";

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

/// Canonical JSON: sorted keys (serde_json's map is a BTreeMap), no whitespace.
fn canon(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).expect("a Value always serializes")
}

fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).expect("a Value always serializes")
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

fn db_path() -> String {
    std::env::var("CONSTRUCT_DB").unwrap_or_else(|_| "construct.db".into())
}

fn open(path: &str) -> R<Connection> {
    let c = Connection::open(path)?;
    c.busy_timeout(Duration::from_secs(10))?;
    c.execute_batch(SCHEMA)?;
    Ok(c)
}

fn put(c: &Connection, data: &[u8]) -> R<String> {
    let h = sha(data);
    c.execute("INSERT OR IGNORE INTO objects (hash, data) VALUES (?1, ?2)", params![h, data])?;
    Ok(h)
}

fn get(c: &Connection, h: &str) -> R<Vec<u8>> {
    Ok(c.query_row("SELECT data FROM objects WHERE hash = ?1", [h], |r| r.get(0))?)
}

fn jget(c: &Connection, h: &str) -> R<Value> {
    Ok(serde_json::from_slice(&get(c, h)?)?)
}

#[derive(Clone, Debug)]
struct Ev {
    typ: String,
    path: Option<String>,
    name: Option<String>,
    key: String,
    req: String,
    resp: String,
    status: u16,
    src: String,
    ctype: Option<String>,
}

impl Ev {
    fn label(&self) -> &str {
        self.name.as_deref().or(self.path.as_deref()).unwrap_or("")
    }
}

/// llm_call and effect events of a run, in order.
fn timeline(c: &Connection, run: &str) -> R<Vec<Ev>> {
    let found: Option<i64> =
        c.query_row("SELECT 1 FROM runs WHERE name = ?1", [run], |r| r.get(0)).optional()?;
    if found.is_none() {
        return Err(format!("no such run '{run}' in {}", db_path()).into());
    }
    let mut st = c.prepare(&format!("SELECT {EV_COLS} FROM events WHERE run = ?1 ORDER BY seq"))?;
    let rows = st.query_map([run], |r| ev_at(r, 0))?;
    Ok(rows.collect::<Result<_, _>>()?)
}

const EV_COLS: &str = "type, path, name, key, req, resp, status, src, ctype";

fn ev_at(r: &rusqlite::Row, i: usize) -> rusqlite::Result<Ev> {
    Ok(Ev {
        typ: r.get(i)?,
        path: r.get(i + 1)?,
        name: r.get(i + 2)?,
        key: r.get(i + 3)?,
        req: r.get(i + 4)?,
        resp: r.get(i + 5)?,
        status: r.get(i + 6)?,
        src: r.get(i + 7)?,
        ctype: r.get(i + 8)?,
    })
}

// ---------- proxy ----------

struct State {
    db: Connection,
    seq: i64,
    cache: HashMap<String, (u16, String, Option<String>)>, // key -> (status, resp hash, ctype)
}

struct Proxy {
    mode: String,
    run: String,
    upstream: Option<String>,
    model: Option<String>,
    agent: ureq::Agent,
    // ponytail: one lock around the DB connection; upstream calls run outside it.
    // Per-run writer threads if a single agent ever saturates SQLite inserts.
    st: Mutex<State>,
}

type Reply = (u16, Vec<u8>, String);

fn reply(status: u16, v: Value) -> Reply {
    (status, canon(&v), "application/json".into())
}

fn fail(status: u16, msg: String) -> Reply {
    if status >= 500 {
        eprintln!("construct: {status} {msg}");
    }
    reply(status, json!({ "error": format!("construct: {msg}") }))
}

fn start(
    db: &str,
    mode: &str,
    run: &str,
    port: u16,
    upstream: Option<String>,
    parent: Option<String>,
    model: Option<String>,
) -> R<(tiny_http::Server, Arc<Proxy>)> {
    let c = open(db)?;
    let cache = match &parent {
        Some(p) => timeline(&c, p)?.into_iter().map(|e| (e.key, (e.status, e.resp, e.ctype))).collect(),
        None => HashMap::new(),
    };
    // bind before claiming the run name
    let server = tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| format!("bind 127.0.0.1:{port}: {e}"))?;
    c.execute(
        "INSERT INTO runs (name, mode, parent, upstream, model_sub, ts) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![run, mode, parent, upstream, model, now()],
    )
    .map_err(|e| match e {
        rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::ConstraintViolation => {
            format!("run '{run}' exists: logs are append-only, pick a new name").into()
        }
        e => Box::<dyn Error>::from(e),
    })?;
    let agent = ureq::AgentBuilder::new().timeout_read(Duration::from_secs(300)).build();
    let st = Mutex::new(State { db: c, seq: 0, cache });
    Ok((
        server,
        Arc::new(Proxy { mode: mode.into(), run: run.into(), upstream, model, agent, st }),
    ))
}

fn port_of(s: &tiny_http::Server) -> u16 {
    s.server_addr().to_ip().map_or(0, |a| a.port())
}

fn serve(server: tiny_http::Server, p: Arc<Proxy>) {
    // ponytail: thread per request; a pool if an agent ever fans out thousands of concurrent calls
    for rq in server.incoming_requests() {
        let p = p.clone();
        std::thread::spawn(move || handle(&p, rq));
    }
}

/// Hop-by-hop or proxy-owned; every other client header (auth, anthropic-beta,
/// OpenAI-Organization...) is forwarded upstream and never stored.
const SKIP_HEADERS: [&str; 11] = [
    "host", "content-length", "content-type", "connection", "transfer-encoding", "accept-encoding",
    "keep-alive", "te", "trailer", "upgrade", "proxy-authorization",
];

fn handle(p: &Proxy, mut rq: tiny_http::Request) {
    let path = rq.url().to_string();
    let headers: Vec<(String, String)> = rq
        .headers()
        .iter()
        .filter(|h| !SKIP_HEADERS.iter().any(|n| h.field.equiv(n)))
        .map(|h| (h.field.as_str().to_string(), h.value.as_str().to_string()))
        .collect();
    let mut raw = Vec::new();
    let (status, body, ctype) = if *rq.method() != tiny_http::Method::Post {
        fail(405, "only POST is proxied".into())
    } else if let Err(e) = rq.as_reader().read_to_end(&mut raw) {
        fail(400, e.to_string())
    } else {
        route(p, &path, &raw, &headers).unwrap_or_else(|e| fail(500, e.to_string()))
    };
    let header = tiny_http::Header::from_bytes("Content-Type", ctype.as_bytes())
        .or_else(|_| tiny_http::Header::from_bytes("Content-Type", "application/octet-stream"))
        .expect("static header is valid");
    let mut resp = tiny_http::Response::from_data(body).with_status_code(status).with_header(header);
    if status == 409 {
        // a replay miss is deterministic: stop the OpenAI/Anthropic SDKs from retrying it
        resp.add_header(tiny_http::Header::from_bytes("x-should-retry", "false").expect("static header is valid"));
    }
    let _ = rq.respond(resp);
}

fn log(st: &mut State, run: &str, e: &Ev) -> R<()> {
    st.db.execute(
        "INSERT INTO events (run, seq, type, ts, path, name, key, req, resp, status, src, ctype)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![run, st.seq, e.typ, now(), e.path, e.name, e.key, e.req, e.resp, e.status, e.src, e.ctype],
    )?;
    st.seq += 1;
    Ok(())
}

fn route(p: &Proxy, path: &str, raw: &[u8], headers: &[(String, String)]) -> R<Reply> {
    let Ok(req) = serde_json::from_slice::<Value>(raw) else {
        return Ok(fail(400, "request body is not JSON".into()));
    };
    if path == "/construct/effect" {
        return effect(p, &req);
    }
    // identity is the request *before* any model substitution
    let key = sha(&canon(&json!([path, req])));
    let hit = if p.mode == "record" { None } else { p.st.lock().unwrap().cache.get(&key).cloned() };
    let (sent, status, body, ctype, src) = match hit {
        Some((status, h, ctype)) => (req, status, get(&p.st.lock().unwrap().db, &h)?, ctype, "cache"),
        None if p.mode == "replay" => {
            let seq = p.st.lock().unwrap().seq;
            return Ok(fail(409, format!(
                "replay miss at seq {seq}: request never recorded; entropy upstream of the LLM \
                 (wrap it as an effect), or use 'fork' to go live"
            )));
        }
        None => {
            let mut req = req;
            if let (Some(m), Some(o)) = (&p.model, req.as_object_mut()) {
                if o.contains_key("model") {
                    o.insert("model".into(), json!(m)); // substitute-model: only live calls are rewritten
                }
            }
            match forward(p, path, &req, headers) {
                Ok((s, b, ct)) => (req, s, b, Some(ct), "live"),
                Err(e) => return Ok(fail(502, format!("upstream unreachable: {e}"))),
            }
        }
    };
    let mut st = p.st.lock().unwrap();
    let (rq, rs) = (put(&st.db, &canon(&sent))?, put(&st.db, &body)?);
    let ev = Ev {
        typ: "llm_call".into(),
        path: Some(path.into()),
        name: None,
        key,
        req: rq,
        resp: rs,
        status,
        src: src.into(),
        ctype: ctype.clone(),
    };
    log(&mut st, &p.run, &ev)?;
    Ok((status, body, ctype.unwrap_or_else(|| "application/json".into())))
}

/// Record/replay a nondeterministic client-side function (see construct.py `wrap`).
/// Query phase `{name, n, args}`; store phase adds `result`.
fn effect(p: &Proxy, req: &Value) -> R<Reply> {
    let name = req["name"].as_str().map(str::to_string);
    let key = sha(&canon(&json!([req["name"], req["n"], req["args"]])));
    let mut st = p.st.lock().unwrap();
    let args = put(&st.db, &canon(&req["args"]))?;
    let ev = |resp: String, src: &str| Ev {
        typ: "effect".into(),
        path: None,
        name: name.clone(),
        key: key.clone(),
        req: args.clone(),
        resp,
        status: 200,
        src: src.into(),
        ctype: None,
    };
    match req.get("result") {
        None => {
            if let Some((_, h, _)) = st.cache.get(&key).cloned() {
                log(&mut st, &p.run, &ev(h.clone(), "cache"))?;
                return Ok(reply(200, json!({ "cached": true, "result": jget(&st.db, &h)? })));
            }
            if p.mode == "replay" {
                return Ok(fail(409, format!(
                    "replay miss: effect '{}' (call #{}) was never recorded",
                    name.as_deref().unwrap_or("?"),
                    req["n"]
                )));
            }
            Ok(reply(200, json!({ "cached": false })))
        }
        Some(_) if p.mode == "replay" => Ok(fail(409, "replay mode never records effect results".into())),
        Some(result) => {
            let h = put(&st.db, &canon(result))?;
            st.cache.insert(key.clone(), (200, h.clone(), None));
            log(&mut st, &p.run, &ev(h, "live"))?;
            Ok(reply(200, json!({ "stored": true })))
        }
    }
}

fn forward(p: &Proxy, path: &str, req: &Value, headers: &[(String, String)]) -> R<(u16, Vec<u8>, String)> {
    let up = p.upstream.as_deref().ok_or("no upstream configured")?;
    let mut r = p.agent.post(&format!("{}{path}", up.trim_end_matches('/'))).set("Content-Type", "application/json");
    for (k, v) in headers {
        r = r.set(k, v);
    }
    // ponytail: SSE is buffered whole, then sent; replay stays byte-identical.
    // Pass-through chunking if record-mode time-to-first-token matters.
    let resp = match r.send_bytes(&canon(req)) {
        Ok(r) | Err(ureq::Error::Status(_, r)) => r,
        Err(e) => return Err(e.into()),
    };
    let (status, ctype) = (resp.status(), resp.header("Content-Type").unwrap_or("application/json").to_string());
    let mut body = Vec::new();
    resp.into_reader().read_to_end(&mut body)?;
    Ok((status, body, ctype))
}

// ---------- SSE reassembly ----------

/// Reassemble a recorded SSE stream into one response. OpenAI chunk deltas fully;
/// Anthropic text and tool-arg deltas; other Anthropic event types ignored.
fn sse_join(body: &[u8]) -> Value {
    let (mut text, mut tools, mut model) = (String::new(), BTreeMap::<i64, Value>::new(), Value::Null);
    let blank = || json!({ "id": null, "type": "function", "function": { "name": "", "arguments": "" } });
    let append = |v: &mut Value, s: &str| *v = json!(format!("{}{s}", v.as_str().unwrap_or("")));
    for line in String::from_utf8_lossy(body).lines() {
        let Some(data) = line.strip_prefix("data:") else { continue };
        let Ok(ev) = serde_json::from_str::<Value>(data.trim()) else { continue }; // skips "" and [DONE]
        if let Some(ch) = ev.get("choices") {
            if !ev["model"].is_null() {
                model = ev["model"].clone();
            }
            let d = &ch[0]["delta"];
            text.push_str(d["content"].as_str().unwrap_or(""));
            for tc in d["tool_calls"].as_array().into_iter().flatten() {
                let t = tools.entry(tc["index"].as_i64().unwrap_or(0)).or_insert_with(blank);
                if !tc["id"].is_null() {
                    t["id"] = tc["id"].clone();
                }
                if let Some(n) = tc["function"]["name"].as_str().filter(|n| !n.is_empty()) {
                    t["function"]["name"] = json!(n);
                }
                append(&mut t["function"]["arguments"], tc["function"]["arguments"].as_str().unwrap_or(""));
            }
        } else if ev["type"] == "content_block_start" && ev["content_block"]["type"] == "tool_use" {
            let b = &ev["content_block"];
            let name = b["name"].as_str().unwrap_or("");
            tools.insert(
                ev["index"].as_i64().unwrap_or(0),
                json!({ "id": b["id"], "type": "function", "function": { "name": name, "arguments": "" } }),
            );
        } else if ev["type"] == "content_block_delta" {
            let d = &ev["delta"];
            if d["type"] == "text_delta" {
                text.push_str(d["text"].as_str().unwrap_or(""));
            } else if d["type"] == "input_json_delta" {
                let t = tools.entry(ev["index"].as_i64().unwrap_or(0)).or_insert_with(blank);
                append(&mut t["function"]["arguments"], d["partial_json"].as_str().unwrap_or(""));
            }
        }
    }
    joined(text, tools.into_values().collect(), model)
}

/// Reassemble an Ollama native stream (`/api/chat` or `/api/generate`, one JSON object per line).
fn ndjson_join(body: &[u8]) -> Value {
    let (mut text, mut tools, mut model) = (String::new(), Vec::new(), Value::Null);
    for line in String::from_utf8_lossy(body).lines() {
        let Ok(ev) = serde_json::from_str::<Value>(line.trim()) else { continue };
        if !ev["model"].is_null() {
            model = ev["model"].clone();
        }
        text.push_str(ev["message"]["content"].as_str().or(ev["response"].as_str()).unwrap_or(""));
        tools.extend(ev["message"]["tool_calls"].as_array().into_iter().flatten().cloned());
    }
    joined(text, tools, model)
}

/// One OpenAI-shaped response from reassembled stream parts.
fn joined(text: String, tools: Vec<Value>, model: Value) -> Value {
    let mut msg = json!({ "role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) } });
    if !tools.is_empty() {
        msg["tool_calls"] = Value::Array(tools);
    }
    let mut out = json!({ "choices": [{ "index": 0, "message": msg, "finish_reason": "stop" }] });
    if !model.is_null() {
        out["model"] = model;
    }
    out
}

/// Streamed responses are stored as raw bytes and reassembled only for reading.
fn streamed(e: &Ev) -> Option<fn(&[u8]) -> Value> {
    let ct = e.ctype.as_deref().unwrap_or("");
    if ct.starts_with("text/event-stream") {
        Some(sse_join)
    } else if ct.starts_with("application/x-ndjson") {
        Some(ndjson_join)
    } else {
        None
    }
}

/// An event's response as JSON, reassembling SSE when needed.
fn resp_obj(c: &Connection, e: &Ev) -> R<Value> {
    let body = get(c, &e.resp)?;
    if let Some(join) = streamed(e) {
        return Ok(join(&body));
    }
    Ok(serde_json::from_slice(&body).unwrap_or_else(|_| json!(String::from_utf8_lossy(&body))))
}

// ---------- diff ----------

/// (text, normalized tool calls) from an OpenAI-, Anthropic- or Ollama-shaped response.
fn assistant_payload(r: &Value) -> (String, Value) {
    if let Some(ch) = r.get("choices") {
        let m = &ch[0]["message"];
        return (m["content"].as_str().unwrap_or("").into(), norm_tools(&m["tool_calls"]));
    }
    if let Some(blocks) = r["content"].as_array() {
        let text = blocks.iter().filter(|b| b["type"] == "text").filter_map(|b| b["text"].as_str()).collect();
        let tools = blocks
            .iter()
            .filter(|b| b["type"] == "tool_use")
            .map(|b| json!({ "name": b["name"], "args": b["input"] }))
            .collect();
        return (text, Value::Array(tools));
    }
    if let Some(m) = r.get("message").filter(|m| m.is_object()) {
        return (m["content"].as_str().unwrap_or("").into(), norm_tools(&m["tool_calls"])); // Ollama /api/chat
    }
    if let Some(t) = r["response"].as_str() {
        return (t.into(), json!([])); // Ollama /api/generate
    }
    (r.to_string(), json!([]))
}

/// Schema-aware: parse argument JSON so key order and whitespace never diff.
fn norm_tools(tool_calls: &Value) -> Value {
    Value::Array(
        tool_calls
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| {
                let f = &t["function"];
                let args = match &f["arguments"] {
                    Value::String(raw) => serde_json::from_str::<Value>(raw).unwrap_or_else(|_| json!(raw)),
                    v => v.clone(), // Ollama sends arguments as an object
                };
                json!({ "name": f["name"], "args": args })
            })
            .collect(),
    )
}

fn word_diff(a: &str, b: &str, esc: fn(&str) -> String, del: [&str; 2], ins: [&str; 2]) -> String {
    let (a, b): (Vec<&str>, Vec<&str>) = (a.split_whitespace().collect(), b.split_whitespace().collect());
    capture_diff_slices(Algorithm::Myers, &a, &b)
        .iter()
        .map(|op| {
            let (tag, ar, br) = op.as_tag_tuple();
            let (old, new) = (esc(&a[ar].join(" ")), esc(&b[br].join(" ")));
            let (d, i) = (format!("{}{old}{}", del[0], del[1]), format!("{}{new}{}", ins[0], ins[1]));
            match tag {
                DiffTag::Equal => old,
                DiffTag::Delete => d,
                DiffTag::Insert => i,
                DiffTag::Replace => format!("{d} {i}"),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn plain(s: &str) -> String {
    s.to_string()
}

fn udiff_raw(a: &Value, b: &Value) -> String {
    let (pa, pb) = (pretty(a) + "\n", pretty(b) + "\n");
    TextDiff::from_lines(&pa, &pb).unified_diff().context_radius(2).header("a", "b").to_string()
}

fn udiff(a: &Value, b: &Value) -> String {
    udiff_raw(a, b).lines().map(|l| format!("      {l}")).collect::<Vec<_>>().join("\n")
}

/// Divergence report: (text, first divergent step, steps in A, steps in B).
fn diff_report(c: &Connection, ra: &str, rb: &str) -> R<(String, Option<usize>, usize, usize)> {
    let (ca, cb) = (timeline(c, ra)?, timeline(c, rb)?);
    let mut out = String::new();
    macro_rules! println {
        ($($t:tt)*) => { writeln!(out, $($t)*).expect("writing to a String never fails") };
    }
    println!("diff {ra} ({} steps) <-> {rb} ({} steps)", ca.len(), cb.len());
    let mut diverged = None;
    for i in 0..ca.len().max(cb.len()) {
        let (ea, eb) = (ca.get(i), cb.get(i));
        if let (Some(x), Some(y)) = (ea, eb) {
            if x.key == y.key && x.resp == y.resp {
                println!("= step {i}  identical  {} {} ({})", x.typ, x.label(), &x.resp[..12]);
                continue;
            }
        }
        if diverged.is_none() {
            diverged = Some(i);
            println!("! first divergence at step {i}");
        }
        let (Some(x), Some(y)) = (ea, eb) else {
            let (e, run) = ea.map(|e| (e, ra)).or(eb.map(|e| (e, rb))).expect("i < longest timeline");
            let r = resp_obj(c, e)?;
            let (text, tools) = if e.typ == "llm_call" { assistant_payload(&r) } else { (r.to_string(), json!([])) };
            let shown: String = if text.is_empty() { tools.to_string() } else { text.chars().take(80).collect() };
            println!("+ step {i}  only in '{run}': {shown}");
            continue;
        };
        if x.typ != y.typ {
            println!("~ step {i}  timeline shape differs: {}/{} vs {}/{}", x.typ, x.label(), y.typ, y.label());
            continue;
        }
        let (qa, qb) = (jget(c, &x.req)?, jget(c, &y.req)?);
        let (pa, pb) = (resp_obj(c, x)?, resp_obj(c, y)?);
        if x.typ == "effect" {
            println!("~ step {i}  effect {}:", x.label());
            if qa != qb {
                println!("      args:\n{}", udiff(&qa, &qb));
            }
            if pa != pb {
                println!("      result:\n{}", udiff(&pa, &pb));
            }
            continue;
        }
        if qa != qb {
            println!("~ step {i}  request:\n{}", udiff(&qa, &qb));
        }
        let ((ta, tools_a), (tb, tools_b)) = (assistant_payload(&pa), assistant_payload(&pb));
        if ta != tb {
            println!("~ step {i}  response text (token diff):");
            println!("      {}", word_diff(&ta, &tb, plain, ["[-", "-]"], ["{+", "+}"]));
        }
        if tools_a != tools_b {
            println!("~ step {i}  tool calls (schema-aware):\n{}", udiff(&tools_a, &tools_b));
        }
    }
    if diverged.is_none() {
        println!("runs are identical");
    }
    Ok((out, diverged, ca.len(), cb.len()))
}

fn cmd_diff(c: &Connection, ra: &str, rb: &str) -> R<()> {
    print!("{}", diff_report(c, ra, rb)?.0);
    Ok(())
}

// ---------- HTML timeline viewer (see DESIGN.md) ----------

// ponytail: system font stacks only; the timeline is an audit artifact and must make no network requests
const CSS: &str = "
:root{color-scheme:light dark;
--ground:oklch(0.955 0.012 85);--raised:oklch(0.978 0.008 85);--ink:oklch(0.25 0.02 150);
--ink-2:oklch(0.47 0.015 150);--ink-3:oklch(0.53 0.012 150);--rule:oklch(0.86 0.014 85);
--accent:oklch(0.63 0.17 45);--accent-ink:oklch(0.5 0.15 42);
--del:oklch(0.45 0.13 30);--del-bg:oklch(0.91 0.045 30);--ins:oklch(0.38 0.08 150);--ins-bg:oklch(0.91 0.045 150);
--sans:system-ui,-apple-system,'Segoe UI',sans-serif;
--serif:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif;
--mono:ui-monospace,'Cascadia Mono','SF Mono',Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{
--ground:oklch(0.205 0.01 150);--raised:oklch(0.24 0.012 150);--ink:oklch(0.92 0.018 85);
--ink-2:oklch(0.74 0.014 85);--ink-3:oklch(0.62 0.012 85);--rule:oklch(0.33 0.012 150);
--accent:oklch(0.72 0.15 52);--accent-ink:oklch(0.8 0.12 55);
--del:oklch(0.86 0.07 30);--del-bg:oklch(0.33 0.06 30);--ins:oklch(0.88 0.07 150);--ins-bg:oklch(0.32 0.05 150)}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:14px/1.5 var(--sans);font-variant-numeric:tabular-nums}
.wrap{max-width:1200px;margin:0 auto;padding:40px 24px 64px}
.brand{margin:0 0 14px;font-size:12px;color:var(--ink-3)}
h1{margin:0 0 8px;font:400 28px/1.2 var(--serif);overflow-wrap:anywhere}
h1 .vs{color:var(--ink-3);font-size:18px;margin:0 .35em}
.lede{margin:0 0 20px;font-size:16px;max-width:72ch}
a{color:var(--accent-ink);text-underline-offset:3px}
.runs{margin:0 0 36px;display:grid;gap:4px;font-size:13px;color:var(--ink-2)}
.runs div{display:flex;flex-wrap:wrap;column-gap:16px}
.runs dt{color:var(--ink);font-weight:600;min-width:12ch}
.runs dd{margin:0;overflow-wrap:anywhere}
.colheads,.step{display:grid;grid-template-columns:var(--grid);column-gap:24px}
.colheads{position:sticky;top:0;z-index:1;background:var(--ground);border-bottom:1px solid var(--rule);padding:10px 0;font-size:13px;font-weight:600}
.step{border-bottom:1px solid var(--rule);padding:16px 0}
.gutter{position:relative;padding-left:18px;font-size:13px;color:var(--ink-3)}
.gutter::before{content:'';position:absolute;left:5px;top:-16px;bottom:-17px;width:1px;background:var(--rule)}
.after .gutter::before{background:var(--accent)}
.diverged .gutter::after{content:'';position:absolute;left:2px;top:.5em;width:7px;height:7px;border-radius:50%;background:var(--accent)}
.diverged .gutter .n{color:var(--accent-ink);font-weight:600}
.cell{min-width:0}
.span,.change{grid-column:2/-1}
.who{display:none}
.what{margin:0;font-weight:600}
.target{margin-left:.4em;font:12px var(--mono);font-weight:400;color:var(--ink-2)}
.facts{margin:2px 0 6px;font-size:12px;color:var(--ink-3)}
.bad{color:var(--accent-ink);font-weight:600}
.sum{margin:0;max-width:72ch;overflow-wrap:anywhere;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.same .what{font-weight:500;color:var(--ink-2)}
.same .sum{color:var(--ink-3)}
code{font:12px var(--mono)}
details{margin-top:8px}
summary{width:fit-content;cursor:pointer;font-size:12px;color:var(--ink-2)}
summary:hover{color:var(--ink)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.ev h4,.change h3{margin:14px 0 4px;font:600 12px/1.4 var(--sans);color:var(--ink-2)}
pre{margin:0;padding:10px 12px;max-height:28rem;overflow:auto;background:var(--raised);border-radius:2px;font:12px/1.5 var(--mono);white-space:pre-wrap;overflow-wrap:anywhere}
.hash{margin:6px 0 0;font:11px var(--mono);color:var(--ink-3)}
.change{margin-top:16px;padding-top:12px;border-top:1px solid var(--accent)}
.change-head{margin:0;font-size:12px;font-weight:600;color:var(--accent-ink)}
.udiff{max-height:none}
.udiff span{display:block;min-height:1.5em}
.udiff .d{background:var(--del-bg);color:var(--del)}
.udiff .i{background:var(--ins-bg);color:var(--ins)}
.udiff .h{color:var(--ink-3)}
.words{margin:0;max-width:80ch}
del{background:var(--del-bg);color:var(--del);text-decoration:line-through;text-decoration-thickness:1px}
ins{background:var(--ins-bg);color:var(--ins);text-decoration:none}
.missing{margin:0;color:var(--ink-3);font-style:italic}
footer{margin-top:40px;font-size:12px;color:var(--ink-3)}
@media (max-width:720px){
.wrap{padding:24px 16px 48px}
.colheads{display:none}
.step{display:block;position:relative;padding-left:2.25rem}
.gutter{position:absolute;left:0;top:16px;bottom:16px;width:2rem;padding-left:16px}
.cell+.cell,.cell+.change{margin-top:16px}
.who{display:block;margin:0 0 2px;font-size:12px;font-weight:600;color:var(--ink-2)}}
@media print{body{background:none}.colheads{position:static}pre{max-height:none}.step{break-inside:avoid}}
";

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#x27;")
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.into() } else { s.chars().take(n).collect::<String>() + "…" }
}

fn kind(e: &Ev) -> &'static str {
    if e.typ == "effect" { "Effect" } else { "LLM call" }
}

fn src_label(e: &Ev) -> &'static str {
    if e.src == "cache" { "replayed" } else { "live" }
}

/// `search(limit: 3, query: "volcanoes")` from normalized tool calls.
fn tool_line(tools: &Value) -> String {
    tools
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| {
            let args = match &t["args"] {
                Value::Object(m) => m.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", "),
                v => v.to_string(),
            };
            format!("{}({args})", t["name"].as_str().unwrap_or("?"))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn udiff_html(a: &Value, b: &Value) -> String {
    let lines: String = udiff_raw(a, b)
        .lines()
        .skip(2) // "--- a" / "+++ b"
        .map(|l| {
            let cls = match l.as_bytes().first() {
                Some(b'@') => "h",
                Some(b'+') => "i",
                Some(b'-') => "d",
                _ => "x",
            };
            format!(r#"<span class="{cls}">{}</span>"#, esc(l))
        })
        .collect();
    format!(r#"<pre class="udiff">{lines}</pre>"#)
}

/// One run's step: what it was, where it came from, a short summary, and the evidence.
fn cell(c: &Connection, e: &Ev, who: &str, sources: &str, span: bool) -> R<String> {
    let (req, resp) = (jget(c, &e.req)?, resp_obj(c, e)?);
    let mut facts = vec![esc(sources)];
    if let Some(m) = req["model"].as_str().filter(|_| e.typ == "llm_call") {
        facts.push(esc(m));
    }
    if e.status != 200 {
        facts.push(format!(r#"<span class="bad">HTTP {}</span>"#, e.status));
    }
    let streamed = streamed(e).is_some();
    let (summary, evidence) = if e.typ == "effect" {
        (
            format!("Returned <code>{}</code>", esc(&clip(&resp.to_string(), 200))),
            [("Arguments", &req), ("Result", &resp)],
        )
    } else {
        let (text, tools) = assistant_payload(&resp);
        let summary = if !text.is_empty() {
            esc(&clip(&text, 280))
        } else if tools.as_array().is_some_and(|t| !t.is_empty()) {
            format!("Calls <code>{}</code>", esc(&clip(&tool_line(&tools), 200)))
        } else {
            r#"<span class="missing">No text or tool calls.</span>"#.into()
        };
        let resp_head = if streamed { "Response (reassembled from a stream)" } else { "Response" };
        (summary, [("Request", &req), (resp_head, &resp)])
    };
    let ev: String = evidence.iter().map(|(h, v)| format!("<h4>{h}</h4><pre>{}</pre>", esc(&pretty(v)))).collect();
    Ok(format!(
        r#"<div class="cell{span}"><p class="who">{who}</p><p class="what">{kind}<span class="target">{target}</span></p><p class="facts">{facts}</p><p class="sum">{summary}</p><details><summary>Evidence</summary><div class="ev">{ev}<p class="hash">key {key} &middot; response {resp}</p></div></details></div>"#,
        span = if span { " span" } else { "" },
        who = esc(who),
        kind = kind(e),
        target = esc(e.label()),
        facts = facts.join(" &middot; "),
        key = &e.key[..12],
        resp = &e.resp[..12],
    ))
}

/// What differs between two runs at one step: request, response, tool calls (or effect args/result).
/// `carried`: the first divergence, when this step comes after it; its request diff then
/// repeats inherited changes, so it is folded instead of shown open.
fn change(c: &Connection, i: usize, x: &Ev, y: &Ev, ra: &str, rb: &str, carried: Option<usize>) -> R<String> {
    let mut parts = String::new();
    if x.typ != y.typ {
        parts = format!(
            "<p>Different kinds of step: {} {} in {}, {} {} in {}.</p>",
            kind(x), esc(x.label()), esc(ra), kind(y), esc(y.label()), esc(rb)
        );
    } else {
        let (qa, qb, pa, pb) = (jget(c, &x.req)?, jget(c, &y.req)?, resp_obj(c, x)?, resp_obj(c, y)?);
        let mut add = |h: &str, body: String| parts += &format!("<h3>{h}</h3>{body}");
        if x.typ == "effect" {
            if qa != qb {
                add("Arguments", udiff_html(&qa, &qb));
            }
            if pa != pb {
                add("Result", udiff_html(&pa, &pb));
            }
        } else {
            if qa != qb {
                let diff = udiff_html(&qa, &qb);
                add("Request", match carried {
                    Some(d) => format!("<details><summary>Show request diff (carries the change from step {d})</summary>{diff}</details>"),
                    None => diff,
                });
            }
            let ((ta, tools_a), (tb, tools_b)) = (assistant_payload(&pa), assistant_payload(&pb));
            if ta != tb {
                let words = word_diff(&ta, &tb, esc, ["<del>", "</del>"], ["<ins>", "</ins>"]);
                add("Response", format!(r#"<p class="words">{words}</p>"#));
            }
            if tools_a != tools_b {
                add("Tool calls", udiff_html(&tools_a, &tools_b));
            }
            if x.status != y.status {
                add("Status", format!("<p>HTTP {} in {}, HTTP {} in {}.</p>", x.status, esc(ra), y.status, esc(rb)));
            }
        }
    }
    Ok(format!(r#"<div class="change"><p class="change-head">What changed at step {i}</p>{parts}</div>"#))
}

fn render_view(c: &Connection, runs: &[String]) -> R<String> {
    let cols = runs.iter().map(|r| timeline(c, r)).collect::<R<Vec<_>>>()?;
    let n = cols.iter().map(Vec::len).max().unwrap_or(0);
    let same_at = |i: usize| {
        runs.len() > 1
            && cols[0].get(i).is_some_and(|f| cols.iter().all(|col| col.get(i).is_some_and(|e| e.key == f.key && e.resp == f.resp)))
    };
    let first_div = if runs.len() > 1 { (0..n).find(|&i| !same_at(i)) } else { None };

    let lede = match (runs.len(), first_div) {
        (_, _) if n == 0 => if runs.len() == 1 { "This run recorded no steps." } else { "These runs recorded no steps." }.to_string(),
        (1, _) => {
            let live = cols[0].iter().filter(|e| e.src == "live").count();
            format!("{n} steps: {live} live, {} replayed.", n - live)
        }
        (_, None) => format!("Identical across all {n} steps."),
        (_, Some(0)) => r##"Diverges at the first step. <a href="#step-0">Go to step 0</a>"##.to_string(),
        (_, Some(d)) => format!(r##"Identical through step {}; diverges at step {d}. <a href="#step-{d}">Go to step {d}</a>"##, d - 1),
    };

    let mut facts = String::new();
    for (r, col) in runs.iter().zip(&cols) {
        let (mode, parent, upstream, model, when): (String, Option<String>, Option<String>, Option<String>, String) = c.query_row(
            "SELECT mode, parent, upstream, model_sub, strftime('%Y-%m-%d %H:%M', ts, 'unixepoch', 'localtime') FROM runs WHERE name = ?1",
            [r],
            |x| Ok((x.get(0)?, x.get(1)?, x.get(2)?, x.get(3)?, x.get(4)?)),
        )?;
        let live = col.iter().filter(|e| e.src == "live").count();
        let mut f = vec![mode];
        f.extend(parent.map(|p| format!("from {p}")));
        f.push(when);
        f.push(format!("{} steps ({live} live, {} replayed)", col.len(), col.len() - live));
        f.extend(model.map(|m| format!("model swapped to {m}")));
        f.extend(upstream);
        facts += &format!("<div><dt>{}</dt><dd>{}</dd></div>", esc(r), esc(&f.join(" · ")));
    }

    let mut rows = String::new();
    for i in 0..n {
        let es: Vec<Option<&Ev>> = cols.iter().map(|col| col.get(i)).collect();
        let (cls, body) = if same_at(i) {
            let sources: Vec<String> = runs.iter().zip(&es).flat_map(|(r, e)| e.map(|e| format!("{} in {r}", src_label(e)))).collect();
            ("same", cell(c, es[0].expect("same_at implies present"), "", &sources.join(", "), true)?)
        } else {
            let mut body = String::new();
            for (r, e) in runs.iter().zip(&es) {
                body += &match e {
                    Some(e) => cell(c, e, r, src_label(e), false)?,
                    None => format!(r#"<div class="cell"><p class="who">{}</p><p class="missing">No step {i} in this run.</p></div>"#, esc(r)),
                };
            }
            if let [Some(x), Some(y)] = es.as_slice() {
                body += &change(c, i, x, y, &runs[0], &runs[1], first_div.filter(|&d| d < i))?;
            }
            (if runs.len() == 1 { "solo" } else { "diverged" }, body)
        };
        let after = if first_div.is_some_and(|d| i >= d) { " after" } else { "" };
        rows += &format!(r#"<section class="step {cls}{after}" id="step-{i}"><div class="gutter"><span class="n">{i}</span></div>{body}</section>"#);
    }
    if n == 0 {
        rows = r#"<p class="missing">No steps to show.</p>"#.into();
    }

    let names: Vec<String> = runs.iter().map(|r| esc(r)).collect();
    let colheads = if runs.len() > 1 {
        format!(r#"<div class="colheads"><span></span>{}</div>"#, names.iter().map(|n| format!("<span>{n}</span>")).collect::<String>())
    } else {
        String::new()
    };
    let generated: String = c.query_row("SELECT strftime('%Y-%m-%d %H:%M', 'now', 'localtime')", [], |r| r.get(0))?;
    Ok(format!(
        r#"<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · Construct</title><style>{CSS}</style><body><div class="wrap" style="--grid:3rem repeat({k},minmax(0,1fr))"><header><p class="brand">AROK Construct</p><h1>{h1}</h1><p class="lede">{lede}</p><dl class="runs">{facts}</dl></header><main>{colheads}{rows}</main><footer>{db} &middot; Construct {ver} &middot; generated {generated}</footer></div><script>addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(d=>d.open=true))</script></body></html>"#,
        title = names.join(" vs "),
        k = runs.len(),
        h1 = names.join(r#"<span class="vs">vs</span>"#),
        db = esc(&db_path()),
        ver = env!("CARGO_PKG_VERSION"),
    ))
}

fn cmd_view(c: &Connection, runs: &[String], out: &str) -> R<()> {
    std::fs::write(out, render_view(c, runs)?)?;
    println!("wrote {out}");
    Ok(())
}

fn runs_json(c: &Connection) -> R<Value> {
    let mut st = c.prepare(
        "SELECT name, mode, parent, model_sub, strftime('%Y-%m-%d %H:%M', ts, 'unixepoch', 'localtime'),
                (SELECT count(*) FROM events WHERE run = runs.name),
                (SELECT count(*) FROM events WHERE run = runs.name AND src = 'live')
         FROM runs ORDER BY name",
    )?;
    let rows = st.query_map([], |r| {
        Ok(json!({
            "name": r.get::<_, String>(0)?, "mode": r.get::<_, String>(1)?,
            "parent": r.get::<_, Option<String>>(2)?, "model_sub": r.get::<_, Option<String>>(3)?,
            "recorded": r.get::<_, String>(4)?, "steps": r.get::<_, i64>(5)?, "live": r.get::<_, i64>(6)?,
        }))
    })?;
    Ok(Value::Array(rows.collect::<Result<_, _>>()?))
}

fn cmd_runs(c: &Connection) -> R<()> {
    for r in runs_json(c)?.as_array().into_iter().flatten() {
        let parent = r["parent"].as_str().map(|p| format!(" <- {p}")).unwrap_or_default();
        let (name, mode, when) = (r["name"].as_str().unwrap_or(""), r["mode"].as_str().unwrap_or(""), r["recorded"].as_str().unwrap_or(""));
        println!("{name:24} {mode:7} {:3} steps  {when}{parent}", r["steps"]);
    }
    Ok(())
}

// ---------- search: full-text retrieval over every step (SQLite FTS5, BM25 ranking) ----------

/// Every string leaf of a JSON value, skipping structural keys.
fn strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|x| strings(x, out)),
        Value::Object(m) => m
            .iter()
            .filter(|(k, _)| !matches!(k.as_str(), "type" | "role" | "id" | "tool_use_id" | "tool_call_id" | "model"))
            .for_each(|(_, x)| strings(x, out)),
        _ => {}
    }
}

/// What a step is about: its newest input message and what came back (or an effect's args and result).
fn step_text(c: &Connection, e: &Ev) -> R<String> {
    let (req, resp) = (jget(c, &e.req)?, resp_obj(c, e)?);
    let mut out = vec![e.label().to_string()];
    if e.typ == "effect" {
        strings(&req, &mut out);
        strings(&resp, &mut out);
    } else {
        if let Some(last) = req["messages"].as_array().and_then(|m| m.last()) {
            strings(last, &mut out);
        }
        out.extend(req["prompt"].as_str().map(str::to_string)); // Ollama /api/generate
        let (text, tools) = assistant_payload(&resp);
        out.extend([text, tool_line(&tools)]);
    }
    Ok(out.join("\n"))
}

/// Bring the per-connection index up to date. It lives in the temp schema, so it works on the
/// read-only MCP connection and never touches the evidence file; events are append-only, so
/// rowid order is a complete change feed.
// ponytail: rebuilt per connection (one pass over the log); persist it in the db if logs reach millions of steps
fn index(c: &Connection) -> R<()> {
    c.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS temp.fts USING fts5(run UNINDEXED, step UNINDEXED, kind UNINDEXED, label UNINDEXED, body, tokenize = 'porter unicode61')",
    )?;
    let last: i64 = c.query_row("SELECT coalesce(max(rowid), 0) FROM temp.fts", [], |r| r.get(0))?;
    let mut st = c.prepare(&format!("SELECT rowid, run, seq, {EV_COLS} FROM events WHERE rowid > ?1 ORDER BY rowid"))?;
    let new = st
        .query_map([last], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?, ev_at(r, 3)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    for (rowid, run, seq, e) in new {
        c.execute(
            "INSERT INTO temp.fts (rowid, run, step, kind, label, body) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![rowid, run, seq, e.typ, e.label(), step_text(c, &e)?],
        )?;
    }
    Ok(())
}

/// Ranked steps matching every word of `query` (each word quoted, so FTS operators in user text are literal).
fn search(c: &Connection, query: &str, run: Option<&str>, limit: i64) -> R<Value> {
    let q: Vec<String> = query.split_whitespace().map(|t| format!("\"{}\"", t.replace('"', "\"\""))).collect();
    if q.is_empty() {
        return Err("query is empty".into());
    }
    index(c)?;
    let mut st = c.prepare(
        "SELECT run, step, kind, label, snippet(fts, 4, '[', ']', '…', 16), round(-bm25(fts), 3) FROM temp.fts
         WHERE fts MATCH ?1 AND (?2 IS NULL OR run = ?2) ORDER BY bm25(fts) LIMIT ?3",
    )?;
    let hits = st
        .query_map(params![q.join(" "), run, limit.clamp(1, 100)], |r| {
            Ok(json!({
                "run": r.get::<_, String>(0)?, "step": r.get::<_, i64>(1)?, "type": r.get::<_, String>(2)?,
                "label": r.get::<_, String>(3)?, "snippet": r.get::<_, String>(4)?, "score": r.get::<_, f64>(5)?,
            }))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({ "query": query, "hits": hits }))
}

fn cmd_search(c: &Connection, query: &str, run: Option<&str>, limit: i64) -> R<()> {
    let hits = search(c, query, run, limit)?;
    let hits = hits["hits"].as_array().expect("search returns hits");
    if hits.is_empty() {
        println!("no steps match '{query}'");
    }
    for h in hits {
        let (run, label, snip) = (h["run"].as_str().unwrap_or(""), h["label"].as_str().unwrap_or(""), h["snippet"].as_str().unwrap_or(""));
        println!("{run} step {}  {label}  ({})\n    {}", h["step"], h["score"], snip.replace('\n', " "));
    }
    Ok(())
}

// ---------- eval: score a run against checks; with a baseline, report regressions ----------

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_f64() => "number",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// A JSON Schema subset. An unsupported keyword is an error, never a silent pass.
fn validate(v: &Value, schema: &Value, at: &str, errs: &mut Vec<String>) -> R<()> {
    let Some(s) = schema.as_object() else { return Err(format!("schema at {at} is not an object").into()) };
    let num = |k: &str| s[k].as_f64().ok_or_else(|| Box::<dyn Error>::from(format!("{at}: '{k}' must be a number")));
    for (k, want) in s {
        match k.as_str() {
            "type" => {
                let types: Vec<&str> = match want {
                    Value::String(t) => vec![t.as_str()],
                    Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
                    _ => return Err(format!("{at}: 'type' must be a string or list").into()),
                };
                let got = type_name(v);
                let integral = v.as_f64().is_some_and(|f| f.fract() == 0.0);
                if !types.iter().any(|t| *t == got || (*t == "number" && got == "integer") || (*t == "integer" && integral)) {
                    errs.push(format!("{at}: expected {}, got {got}", types.join(" or ")));
                }
            }
            "enum" if !want.as_array().is_some_and(|a| a.contains(v)) => errs.push(format!("{at}: {v} is not one of {want}")),
            "const" if v != want => errs.push(format!("{at}: expected {want}, got {v}")),
            "required" => {
                for r in want.as_array().into_iter().flatten().filter_map(Value::as_str) {
                    if v.as_object().is_some_and(|o| !o.contains_key(r)) {
                        errs.push(format!("{at}: missing required '{r}'"));
                    }
                }
            }
            "properties" => {
                let props = want.as_object().ok_or_else(|| format!("{at}: 'properties' must be an object"))?;
                for (p, ps) in props {
                    if let Some(x) = v.get(p).filter(|_| v.is_object()) {
                        validate(x, ps, &format!("{at}.{p}"), errs)?;
                    }
                }
            }
            "additionalProperties" => match (want, v.as_object()) {
                (Value::Bool(true), _) | (Value::Bool(false), None) => {}
                (Value::Bool(false), Some(o)) => {
                    for key in o.keys().filter(|key| !s.get("properties").is_some_and(|p| p.get(key.as_str()).is_some())) {
                        errs.push(format!("{at}: unexpected property '{key}'"));
                    }
                }
                _ => return Err(format!("{at}: only true/false 'additionalProperties' is supported").into()),
            },
            "items" => {
                for (i, x) in v.as_array().into_iter().flatten().enumerate() {
                    validate(x, want, &format!("{at}[{i}]"), errs)?;
                }
            }
            "minimum" | "maximum" => {
                let (b, low) = (num(k)?, k == "minimum");
                if let Some(x) = v.as_f64().filter(|x| if low { *x < b } else { *x > b }) {
                    errs.push(format!("{at}: {x} is {} {b}", if low { "below" } else { "above" }));
                }
            }
            "minLength" | "maxLength" | "minItems" | "maxItems" => {
                let (b, low) = (num(k)? as usize, k.starts_with("min"));
                let n = v.as_str().map(|x| x.chars().count()).or(v.as_array().map(Vec::len)).filter(|_| k.ends_with("Length") == v.is_string());
                if let Some(n) = n.filter(|n| if low { *n < b } else { *n > b }) {
                    errs.push(format!("{at}: length {n} is {} {b}", if low { "below" } else { "above" }));
                }
            }
            "enum" | "const" | "title" | "description" | "$schema" | "default" | "examples" => {}
            other => return Err(format!("{at}: schema keyword '{other}' is not supported").into()),
        }
    }
    Ok(())
}

/// Models often fence JSON output in ```json ... ```.
fn unfence(s: &str) -> &str {
    let t = s.trim();
    match (t.strip_prefix("```"), t.strip_suffix("```")) {
        (Some(_), Some(_)) if t.len() > 6 => t[3..t.len() - 3].split_once('\n').map_or("", |(_, body)| body).trim(),
        _ => t,
    }
}

/// (passed, detail) for one check against one run.
fn check(c: &Connection, tl: &[Ev], ck: &Value) -> R<(bool, String)> {
    let llm: Vec<(usize, &Ev)> = tl.iter().enumerate().filter(|(_, e)| e.typ == "llm_call").collect();
    if let Some(tool) = ck["tool"].as_str() {
        let schema = ck.get("schema").ok_or("a tool check needs 'schema'")?;
        let (mut calls, mut errs) = (0, Vec::new());
        for (i, e) in &llm {
            for t in assistant_payload(&resp_obj(c, e)?).1.as_array().into_iter().flatten().filter(|t| t["name"] == tool) {
                calls += 1;
                validate(&t["args"], schema, &format!("step {i} {tool}"), &mut errs)?;
            }
        }
        return Ok(match (calls, errs.is_empty()) {
            (0, _) => (false, format!("{tool} was never called")),
            (n, true) => (true, format!("{n} call(s) to {tool} match the schema")),
            (_, false) => (false, errs.join("; ")),
        });
    }
    let text_at = |e: &Ev| resp_obj(c, e).map(|r| assistant_payload(&r).0);
    let target = match &ck["step"] {
        Value::Null => text_at(llm.last().ok_or("run has no LLM calls")?.1)?,
        Value::String(w) if w == "last" => text_at(llm.last().ok_or("run has no LLM calls")?.1)?,
        Value::String(w) if w == "any" => llm.iter().map(|(_, e)| text_at(e)).collect::<R<Vec<_>>>()?.join("\n"),
        Value::Number(n) => {
            let i = n.as_u64().ok_or("'step' must be a step number, \"last\" or \"any\"")? as usize;
            text_at(llm.iter().find(|(j, _)| *j == i).ok_or(format!("step {i} is not an LLM call"))?.1)?
        }
        _ => return Err("'step' must be a step number, \"last\" or \"any\"".into()),
    };
    let words = |k: &str| ck[k].as_array().into_iter().flatten().filter_map(Value::as_str).collect::<Vec<_>>();
    let (inc, exc, schema) = (words("includes"), words("excludes"), ck.get("schema"));
    if inc.is_empty() && exc.is_empty() && schema.is_none() {
        return Err("a check needs 'includes', 'excludes', 'schema', or 'tool' + 'schema'".into());
    }
    let low = target.to_lowercase();
    let mut errs: Vec<String> = inc.iter().filter(|w| !low.contains(&w.to_lowercase())).map(|w| format!("missing \"{w}\"")).collect();
    errs.extend(exc.iter().filter(|w| low.contains(&w.to_lowercase())).map(|w| format!("contains \"{w}\"")));
    if let Some(schema) = schema {
        match serde_json::from_str::<Value>(unfence(&target)) {
            Ok(v) => validate(&v, schema, "output", &mut errs)?,
            Err(e) => errs.push(format!("output is not JSON ({e})")),
        }
    }
    Ok((errs.is_empty(), if errs.is_empty() { "ok".into() } else { errs.join("; ") }))
}

fn eval_json(c: &Connection, run: &str, checks: &Value, baseline: Option<&str>) -> R<Value> {
    let list = checks.get("checks").unwrap_or(checks).as_array().ok_or("checks must be a list or {\"checks\": [...]}")?;
    if list.is_empty() {
        return Err("no checks to run".into());
    }
    let tl = timeline(c, run)?;
    let bl = baseline.map(|b| timeline(c, b)).transpose()?;
    let (mut results, mut regressions, mut fixes, mut passed) = (Vec::new(), Vec::new(), Vec::new(), 0);
    for (i, ck) in list.iter().enumerate() {
        let name = ck["name"].as_str().map_or_else(|| format!("check {i}"), str::to_string);
        let run_one = |t: &[Ev]| check(c, t, ck).map_err(|e| Box::<dyn Error>::from(format!("{name}: {e}")));
        let (pass, detail) = run_one(&tl)?;
        passed += usize::from(pass);
        let mut row = json!({ "name": name, "pass": pass, "detail": detail });
        if let Some(b) = &bl {
            let (bp, bd) = run_one(b)?;
            if bp && !pass {
                regressions.push(name.clone());
            } else if pass && !bp {
                fixes.push(name.clone());
            }
            row["baseline_pass"] = json!(bp);
            row["baseline_detail"] = json!(bd);
        }
        results.push(row);
    }
    Ok(json!({
        "run": run, "baseline": baseline, "passed": passed, "total": list.len(),
        "regressions": regressions, "fixes": fixes, "results": results,
    }))
}

/// Exit status is the gate: any failure (or, with a baseline, any regression) is an error.
fn cmd_eval(c: &Connection, run: &str, checks_file: &str, baseline: Option<&str>) -> R<()> {
    let checks: Value = serde_json::from_str(&std::fs::read_to_string(checks_file)?).map_err(|e| format!("{checks_file}: {e}"))?;
    let r = eval_json(c, run, &checks, baseline)?;
    println!("eval {run} against {checks_file}{}", baseline.map(|b| format!(" (baseline {b})")).unwrap_or_default());
    for row in r["results"].as_array().into_iter().flatten() {
        let mark = if row["pass"] == true { "PASS" } else { "FAIL" };
        let was = match row.get("baseline_pass") {
            Some(Value::Bool(true)) if row["pass"] == false => "  (regression: passed in baseline)",
            Some(Value::Bool(false)) if row["pass"] == true => "  (fixed: failed in baseline)",
            _ => "",
        };
        println!("{mark}  {}: {}{was}", row["name"].as_str().unwrap_or(""), row["detail"].as_str().unwrap_or(""));
    }
    let (passed, total, regr) = (r["passed"].as_u64().unwrap_or(0), r["total"].as_u64().unwrap_or(0), r["regressions"].as_array().map_or(0, Vec::len));
    println!("{passed}/{total} passed{}", if baseline.is_some() { format!("; {regr} regression(s)") } else { String::new() });
    match baseline {
        Some(_) if regr > 0 => Err(format!("{regr} regression(s) against baseline").into()),
        None if passed < total => Err(format!("{} of {total} checks failed", total - passed).into()),
        _ => Ok(()),
    }
}

// ---------- MCP server: read-only access to the log for any MCP client (stdio, JSON-RPC 2.0) ----------

const MCP_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// The MCP server never writes: the log is evidence, and an agent reading it must not be able to change it.
fn open_ro(path: &str) -> R<Connection> {
    let c = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
        .map_err(|e| format!("cannot open {path} read-only: {e}"))?;
    c.busy_timeout(Duration::from_secs(10))?;
    Ok(c)
}

fn mcp_tools() -> Value {
    let s = |d: &str| json!({ "type": "string", "description": d });
    let run = s("Run name, as listed by list_runs.");
    let run_desc = run.clone();
    let nullable = |t: &str| json!({ "type": [t, "null"] });
    let ro = json!({ "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false });
    json!([
        {
            "name": "list_runs",
            "title": "List recorded runs",
            "description": "Every recorded agent run: mode (record, replay, fork), parent run, substituted model, \
                            when it was recorded, and how many steps were live versus replayed.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false },
            "outputSchema": { "type": "object", "required": ["runs"], "properties": { "runs": { "type": "array", "items": {
                "type": "object",
                "required": ["name", "mode", "parent", "model_sub", "recorded", "steps", "live"],
                "properties": {
                    "name": { "type": "string" }, "mode": { "type": "string", "enum": ["record", "replay", "fork"] },
                    "parent": nullable("string"), "model_sub": nullable("string"), "recorded": { "type": "string" },
                    "steps": { "type": "integer" }, "live": { "type": "integer" },
                },
            } } } },
            "annotations": ro,
        },
        {
            "name": "get_step",
            "title": "Get one step's evidence",
            "description": "One step of a run: the exact request and response (streams reassembled), \
                            the assistant text and tool calls, model, HTTP status, and whether it was live or replayed.",
            "inputSchema": {
                "type": "object", "required": ["run", "step"], "additionalProperties": false,
                "properties": { "run": run, "step": { "type": "integer", "minimum": 0, "description": "Zero-based step number." } },
            },
            "outputSchema": {
                "type": "object",
                "required": ["run", "step", "steps", "type", "label", "src", "status", "model", "text", "tool_calls", "request", "response", "key"],
                "properties": {
                    "run": { "type": "string" }, "step": { "type": "integer" }, "steps": { "type": "integer" },
                    "type": { "type": "string", "enum": ["llm_call", "effect"] }, "label": { "type": "string" },
                    "src": { "type": "string", "enum": ["live", "replayed"] }, "status": { "type": "integer" },
                    "model": nullable("string"), "text": { "type": "string" },
                    "tool_calls": { "type": "array", "items": { "type": "object", "required": ["name", "args"] } },
                    "request": {}, "response": {}, "key": { "type": "string" },
                },
            },
            "annotations": ro,
        },
        {
            "name": "diff_runs",
            "title": "Diff two runs",
            "description": "Where two runs of an agent diverge: the first differing step, then request, response \
                            (word-level), and tool-call (schema-aware) differences for every step after it.",
            "inputSchema": {
                "type": "object", "required": ["run_a", "run_b"], "additionalProperties": false,
                "properties": { "run_a": run.clone(), "run_b": run },
            },
            "outputSchema": {
                "type": "object", "required": ["run_a", "run_b", "steps_a", "steps_b", "first_divergence", "report"],
                "properties": {
                    "run_a": { "type": "string" }, "run_b": { "type": "string" },
                    "steps_a": { "type": "integer" }, "steps_b": { "type": "integer" },
                    "first_divergence": nullable("integer"), "report": { "type": "string" },
                },
            },
            "annotations": ro,
        },
        {
            "name": "search_runs",
            "title": "Search every step",
            "description": "Full-text search (BM25-ranked) across every recorded step: the newest input message, \
                            the assistant text and tool calls, and effect arguments and results. Every word must match.",
            "inputSchema": {
                "type": "object", "required": ["query"], "additionalProperties": false,
                "properties": {
                    "query": s("Words to find, e.g. 'refund policy'."),
                    "run": s("Only search this run."),
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100, "description": "Most hits to return (default 10)." },
                },
            },
            "outputSchema": { "type": "object", "required": ["query", "hits"], "properties": {
                "query": { "type": "string" },
                "hits": { "type": "array", "items": { "type": "object",
                    "required": ["run", "step", "type", "label", "snippet", "score"],
                    "properties": {
                        "run": { "type": "string" }, "step": { "type": "integer" }, "type": { "type": "string" },
                        "label": { "type": "string" }, "snippet": { "type": "string" }, "score": { "type": "number" },
                    } } },
            } },
            "annotations": ro,
        },
        {
            "name": "eval_run",
            "title": "Evaluate a run",
            "description": "Score a run against checks; with a baseline run, report regressions and fixes. A check is \
                            {name, step?: \"last\"|\"any\"|N, includes?: [..], excludes?: [..], schema?: JSON Schema for the \
                            output} or {name, tool, schema} to validate every call to that tool's arguments.",
            "inputSchema": {
                "type": "object", "required": ["run", "checks"], "additionalProperties": false,
                "properties": {
                    "run": run_desc.clone(),
                    "baseline": s("Optional run to compare against, e.g. the parent of a fork."),
                    "checks": { "type": "array", "minItems": 1, "items": { "type": "object" } },
                },
            },
            "outputSchema": {
                "type": "object", "required": ["run", "baseline", "passed", "total", "regressions", "fixes", "results"],
                "properties": {
                    "run": { "type": "string" }, "baseline": nullable("string"),
                    "passed": { "type": "integer" }, "total": { "type": "integer" },
                    "regressions": { "type": "array", "items": { "type": "string" } },
                    "fixes": { "type": "array", "items": { "type": "string" } },
                    "results": { "type": "array", "items": { "type": "object", "required": ["name", "pass", "detail"] } },
                },
            },
            "annotations": ro,
        },
    ])
}

fn step_json(c: &Connection, run: &str, i: usize) -> R<Value> {
    let tl = timeline(c, run)?;
    let e = tl.get(i).ok_or_else(|| format!("run '{run}' has {} steps; step {i} does not exist", tl.len()))?;
    let (req, resp) = (jget(c, &e.req)?, resp_obj(c, e)?);
    let llm = e.typ == "llm_call";
    let (text, tools) = if llm { assistant_payload(&resp) } else { (String::new(), json!([])) };
    Ok(json!({
        "run": run, "step": i, "steps": tl.len(), "type": e.typ, "label": e.label(), "src": src_label(e),
        "status": e.status, "model": if llm { req["model"].clone() } else { Value::Null },
        "text": text, "tool_calls": tools, "request": req, "response": resp, "key": e.key,
    }))
}

/// None: no such tool. Some(Err): the tool ran and failed, reported to the model as `isError`.
fn mcp_call(c: &Connection, name: &str, a: &Value) -> Option<R<Value>> {
    let s = |k: &str| a[k].as_str().ok_or_else(|| Box::<dyn Error>::from(format!("'{k}' must be a string")));
    Some(match name {
        "list_runs" => runs_json(c).map(|r| json!({ "runs": r })),
        "get_step" => (|| step_json(c, s("run")?, a["step"].as_u64().ok_or("'step' must be a non-negative integer")? as usize))(),
        "diff_runs" => (|| {
            let (ra, rb) = (s("run_a")?, s("run_b")?);
            let (report, first, na, nb) = diff_report(c, ra, rb)?;
            Ok(json!({ "run_a": ra, "run_b": rb, "steps_a": na, "steps_b": nb, "first_divergence": first, "report": report }))
        })(),
        "search_runs" => (|| search(c, s("query")?, a["run"].as_str(), a["limit"].as_i64().unwrap_or(10)))(),
        "eval_run" => (|| eval_json(c, s("run")?, &a["checks"], a["baseline"].as_str()))(),
        _ => return None,
    })
}

/// One JSON-RPC message in, at most one reply out (notifications and stray responses get none).
fn mcp_handle(c: &Connection, msg: &Value) -> Option<Value> {
    let (id, method) = (msg.get("id")?.clone(), msg.get("method")?.as_str().unwrap_or(""));
    let ok = |result: Value| json!({ "jsonrpc": "2.0", "id": id, "result": result });
    let err = |code: i64, m: String| json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": m } });
    Some(match method {
        "initialize" => {
            let asked = msg["params"]["protocolVersion"].as_str().unwrap_or("");
            let version = MCP_VERSIONS.iter().find(|v| **v == asked).unwrap_or(&MCP_VERSIONS[0]);
            ok(json!({
                "protocolVersion": version,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "arok-construct", "title": "AROK Construct", "version": env!("CARGO_PKG_VERSION") },
                "instructions": "Read-only access to recorded LLM agent runs. Start with list_runs; \
                                 use diff_runs to find where two runs diverge and get_step for the exact evidence.",
            }))
        }
        "ping" => ok(json!({})),
        "tools/list" => ok(json!({ "tools": mcp_tools() })),
        "tools/call" => {
            let name = msg["params"]["name"].as_str().unwrap_or("");
            let args = msg["params"].get("arguments").cloned().unwrap_or_else(|| json!({}));
            match mcp_call(c, name, &args) {
                None => err(-32602, format!("unknown tool '{name}'")),
                Some(Ok(v)) => {
                    let text = v["report"].as_str().map_or_else(|| pretty(&v), str::to_string);
                    ok(json!({ "content": [{ "type": "text", "text": text }], "structuredContent": v }))
                }
                Some(Err(e)) => ok(json!({ "content": [{ "type": "text", "text": format!("construct: {e}") }], "isError": true })),
            }
        }
        m => err(-32601, format!("method not found: {m}")),
    })
}

// ponytail: one message per line, handled in order; JSON-RPC batches (dropped in MCP 2025-06-18) are not supported
fn cmd_mcp(c: &Connection) -> R<()> {
    let stdout = std::io::stdout();
    for line in std::io::stdin().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => mcp_handle(c, &msg),
            Err(e) => Some(json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": format!("parse error: {e}") } })),
        };
        if let Some(r) = reply {
            let mut out = stdout.lock();
            out.write_all(&canon(&r))?;
            out.write_all(b"\n")?;
            out.flush()?;
        }
    }
    Ok(())
}

// ---------- cli ----------

fn cli(args: &[String]) -> R<()> {
    let Some((cmd, rest)) = args.split_first() else { return Err(USAGE.into()) };
    let (mut flags, mut pos) = (HashMap::new(), Vec::new());
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        if a.starts_with('-') {
            flags.insert(a.as_str(), it.next().ok_or(format!("{a} needs a value"))?.clone());
        } else {
            pos.push(a.clone());
        }
    }
    let allowed: &[&str] = match cmd.as_str() {
        "record" => &["--run", "--upstream", "--port"],
        "replay" => &["--run", "--from", "--port"],
        "fork" => &["--run", "--from", "--upstream", "--model", "--port"],
        "view" => &["-o", "--out"],
        "diff" | "runs" | "mcp" => &[],
        "search" => &["--run", "--limit"],
        "eval" => &["--checks", "--baseline"],
        "help" | "-h" | "--help" => {
            println!("{USAGE}");
            return Ok(());
        }
        other => return Err(format!("unknown command '{other}'\n\n{USAGE}").into()),
    };
    if let Some(f) = flags.keys().find(|f| !allowed.contains(f)) {
        return Err(format!("{cmd}: unknown flag {f}").into());
    }
    match cmd.as_str() {
        "diff" => {
            let [a, b] = pos.as_slice() else { return Err("usage: construct diff RUN_A RUN_B".into()) };
            cmd_diff(&open(&db_path())?, a, b)
        }
        "view" if pos.is_empty() => Err("usage: construct view RUN [RUN...] [-o timeline.html]".into()),
        "view" => {
            let out = flags.get("-o").or(flags.get("--out")).map_or("timeline.html", String::as_str);
            cmd_view(&open(&db_path())?, &pos, out)
        }
        "runs" => cmd_runs(&open(&db_path())?),
        "mcp" => cmd_mcp(&open_ro(&db_path())?),
        "search" if pos.is_empty() => Err("usage: construct search QUERY [--run RUN] [--limit 10]".into()),
        "search" => {
            let limit = flags.get("--limit").map(|l| l.parse::<i64>()).transpose()?.unwrap_or(10);
            cmd_search(&open_ro(&db_path())?, &pos.join(" "), flags.get("--run").map(String::as_str), limit)
        }
        "eval" => {
            let ([run], Some(checks)) = (pos.as_slice(), flags.get("--checks")) else {
                return Err("usage: construct eval RUN --checks FILE [--baseline RUN]".into());
            };
            cmd_eval(&open_ro(&db_path())?, run, checks, flags.get("--baseline").map(String::as_str))
        }
        mode => {
            if !pos.is_empty() {
                return Err(format!("{mode}: unexpected argument '{}'", pos[0]).into());
            }
            let run = flags.get("--run").ok_or("--run is required")?;
            let parent = flags.get("--from").cloned();
            if mode != "record" && parent.is_none() {
                return Err("--from is required".into());
            }
            let upstream = flags.get("--upstream").cloned().or_else(|| std::env::var("CONSTRUCT_UPSTREAM").ok());
            let upstream = upstream.filter(|_| mode != "replay");
            if mode != "replay" && upstream.is_none() {
                return Err("--upstream (or $CONSTRUCT_UPSTREAM) is required".into());
            }
            let port = flags.get("--port").map(|p| p.parse::<u16>()).transpose()?.unwrap_or(8787);
            let (server, p) = start(&db_path(), mode, run, port, upstream.clone(), parent, flags.get("--model").cloned())?;
            println!(
                "construct {mode}: run '{run}' on http://127.0.0.1:{} -> {}",
                port_of(&server),
                upstream.as_deref().unwrap_or("offline")
            );
            serve(server, p);
            Ok(())
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = cli(&args) {
        eprintln!("construct: {e}");
        std::process::exit(1);
    }
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    fn tmpdb() -> String {
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir()
            .join(format!("construct-test-{}-{}.db", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
        let _ = std::fs::remove_file(&p);
        p.to_string_lossy().into()
    }

    fn proxy(db: &str, mode: &str, run: &str, up: Option<u16>, parent: Option<&str>, model: Option<&str>) -> u16 {
        let upstream = up.map(|u| format!("http://127.0.0.1:{u}"));
        let (s, p) = start(db, mode, run, 0, upstream, parent.map(Into::into), model.map(Into::into)).unwrap();
        let port = port_of(&s);
        thread::spawn(move || serve(s, p));
        port
    }

    fn post(port: u16, path: &str, body: &Value) -> (u16, Vec<u8>) {
        let r = match ureq::post(&format!("http://127.0.0.1:{port}{path}")).send_bytes(&canon(body)) {
            Ok(r) | Err(ureq::Error::Status(_, r)) => r,
            Err(e) => panic!("{e}"),
        };
        let status = r.status();
        let mut b = Vec::new();
        r.into_reader().read_to_end(&mut b).unwrap();
        (status, b)
    }

    fn chat(port: u16, msgs: Value, model: &str) -> (u16, Vec<u8>) {
        post(port, "/v1/chat/completions", &json!({ "model": model, "messages": msgs }))
    }

    fn content(body: &[u8]) -> String {
        serde_json::from_slice::<Value>(body).unwrap()["choices"][0]["message"]["content"].as_str().unwrap().into()
    }

    fn srcs(db: &str, run: &str) -> Vec<String> {
        timeline(&open(db).unwrap(), run).unwrap().into_iter().map(|e| e.src).collect()
    }

    /// Deterministic stub upstream.
    fn stub(f: fn(&Value) -> (Vec<u8>, &'static str)) -> u16 {
        let s = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = port_of(&s);
        thread::spawn(move || {
            for mut rq in s.incoming_requests() {
                let mut raw = Vec::new();
                rq.as_reader().read_to_end(&mut raw).unwrap();
                let (body, ct) = f(&serde_json::from_slice(&raw).unwrap());
                let h = tiny_http::Header::from_bytes("Content-Type", ct).unwrap();
                rq.respond(tiny_http::Response::from_data(body).with_header(h)).unwrap();
            }
        });
        port
    }

    fn echo(req: &Value) -> (Vec<u8>, &'static str) {
        let n = req["messages"].as_array().map_or(0, Vec::len);
        let msg = json!({ "role": "assistant", "content": format!("reply-{n} model={}", req["model"].as_str().unwrap_or("")) });
        (canon(&json!({ "choices": [{ "index": 0, "message": msg }] })), "application/json")
    }

    fn sse(_: &Value) -> (Vec<u8>, &'static str) {
        let chunks = [
            json!({"model": "s1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello "}}]}),
            json!({"model": "s1", "choices": [{"index": 0, "delta": {"content": "world"}}]}),
            json!({"model": "s1", "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{\"a\""}}]}}]}),
            json!({"model": "s1", "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ": 1}"}}]}}]}),
        ];
        let mut body: Vec<u8> = chunks.iter().flat_map(|c| [b"data: ".to_vec(), canon(c), b"\n\n".to_vec()].concat()).collect();
        body.extend_from_slice(b"data: [DONE]\n\n");
        (body, "text/event-stream")
    }

    // store

    #[test]
    fn store_roundtrip_and_dedup() {
        let c = open(&tmpdb()).unwrap();
        let (h1, h2) = (put(&c, b"hello").unwrap(), put(&c, b"hello").unwrap());
        assert_eq!(h1, h2);
        assert_eq!(get(&c, &h1).unwrap(), b"hello");
        assert_eq!(c.query_row("SELECT count(*) FROM objects", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
    }

    #[test]
    fn canon_ignores_key_order() {
        let a: Value = serde_json::from_str(r#"{"b": 1, "a": [1, 2]}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"a":[1,2],"b":1}"#).unwrap();
        assert_eq!(canon(&a), canon(&b));
        assert_eq!(canon(&a), br#"{"a":[1,2],"b":1}"#);
    }

    #[test]
    fn log_is_append_only() {
        let db = tmpdb();
        let up = stub(echo);
        let rec = proxy(&db, "record", "base", Some(up), None, None);
        chat(rec, json!([{"role": "user", "content": "hi"}]), "m1");
        let c = open(&db).unwrap();
        assert!(c.execute("UPDATE events SET status = 500", []).is_err());
        assert!(c.execute("DELETE FROM events", []).is_err());
        assert!(c.execute("UPDATE runs SET parent = 'x'", []).is_err());
        assert!(c.execute("UPDATE objects SET data = x'00'", []).is_err());
        assert_eq!(timeline(&c, "base").unwrap().len(), 1);
    }

    // proxy

    #[test]
    fn record_then_replay_identical() {
        let db = tmpdb();
        let up = stub(echo);
        let rec = proxy(&db, "record", "base", Some(up), None, None);
        let m1 = json!([{"role": "user", "content": "hi"}]);
        let m2 = json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]);
        let (a, b) = (chat(rec, m1.clone(), "m1"), chat(rec, m2.clone(), "m1"));
        let rep = proxy(&db, "replay", "r1", None, Some("base"), None);
        assert_eq!(a, chat(rep, m1, "m1"));
        assert_eq!(b, chat(rep, m2, "m1"));
        assert_eq!(srcs(&db, "r1"), ["cache", "cache"]);
    }

    #[test]
    fn replay_miss_409() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(stub(echo)), None, None);
        chat(rec, json!([{"role": "user", "content": "hi"}]), "m1");
        let rep = proxy(&db, "replay", "r1", None, Some("base"), None);
        let miss = json!({ "model": "m1", "messages": [{"role": "user", "content": "DIFFERENT"}] });
        match ureq::post(&format!("http://127.0.0.1:{rep}/v1/chat/completions")).send_bytes(&canon(&miss)) {
            Err(ureq::Error::Status(409, r)) => assert_eq!(r.header("x-should-retry"), Some("false")),
            other => panic!("expected 409, got {other:?}"),
        }
    }

    #[test]
    fn fork_sources_and_model_substitution() {
        let db = tmpdb();
        let up = stub(echo);
        let rec = proxy(&db, "record", "base", Some(up), None, None);
        let hi = json!([{"role": "user", "content": "hi"}]);
        chat(rec, hi.clone(), "m1");
        chat(rec, json!([{"role": "user", "content": "hi"}, {"role": "user", "content": "more"}]), "m1");
        let frk = proxy(&db, "fork", "b1", Some(up), Some("base"), Some("m2"));
        chat(frk, hi.clone(), "m1"); // hit
        let (_, body) = chat(frk, json!([{"role": "user", "content": "MUTATED"}]), "m1"); // miss: live, model swapped
        assert_eq!(srcs(&db, "b1"), ["cache", "live"]);
        assert!(content(&body).contains("model=m2"));
        // cached step must NOT be rewritten
        assert!(content(&chat(frk, hi, "m1").1).contains("model=m1"));
        // the recorded request is what was actually sent
        let c = open(&db).unwrap();
        assert_eq!(jget(&c, &timeline(&c, "b1").unwrap()[1].req).unwrap()["model"], "m2");
    }

    #[test]
    fn run_name_collision_refused() {
        let db = tmpdb();
        proxy(&db, "record", "base", Some(stub(echo)), None, None);
        let err = start(&db, "record", "base", 0, Some("http://x".into()), None, None).err().unwrap();
        assert!(err.to_string().contains("exists"), "{err}");
    }

    #[test]
    fn unknown_parent_refused() {
        let err = start(&tmpdb(), "replay", "r1", 0, None, Some("nope".into()), None).err().unwrap();
        assert!(err.to_string().contains("no such run"), "{err}");
    }

    #[test]
    fn upstream_down_is_502_and_not_recorded() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(1), None, None); // nothing listens on port 1
        assert_eq!(chat(rec, json!([{"role": "user", "content": "hi"}]), "m1").0, 502);
        assert!(srcs(&db, "base").is_empty());
    }

    #[test]
    fn non_json_body_is_400() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(stub(echo)), None, None);
        let r = ureq::post(&format!("http://127.0.0.1:{rec}/v1/chat/completions")).send_string("not json");
        assert!(matches!(r, Err(ureq::Error::Status(400, _))));
    }

    // effects (protocol level; construct.py wrap() is tested in test_construct.py)

    fn effect_call(port: u16, n: i64, f: &mut dyn FnMut() -> Value) -> Result<Value, u16> {
        let q = json!({ "name": "flaky", "n": n, "args": [["x"], {}] });
        let (s, b) = post(port, "/construct/effect", &q);
        if s != 200 {
            return Err(s);
        }
        let r: Value = serde_json::from_slice(&b).unwrap();
        if r["cached"] == true {
            return Ok(r["result"].clone());
        }
        let result = f();
        let mut store = q;
        store["result"] = result.clone();
        assert_eq!(post(port, "/construct/effect", &store).0, 200);
        Ok(result)
    }

    #[test]
    fn effects_record_then_replay_skip_execution() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(stub(echo)), None, None);
        let mut runs = 0;
        let mut flaky = || {
            runs += 1;
            json!({ "hits": ["x", runs] })
        };
        assert_eq!(effect_call(rec, 0, &mut flaky), Ok(json!({"hits": ["x", 1]})));
        assert_eq!(effect_call(rec, 1, &mut flaky), Ok(json!({"hits": ["x", 2]}))); // same args, distinct call#
        let rep = proxy(&db, "replay", "r1", None, Some("base"), None);
        let mut never = || -> Value { panic!("replay executed the real function") };
        assert_eq!(effect_call(rep, 0, &mut never), Ok(json!({"hits": ["x", 1]})));
        assert_eq!(effect_call(rep, 1, &mut never), Ok(json!({"hits": ["x", 2]})));
        assert_eq!(runs, 2);
        assert_eq!(srcs(&db, "r1"), ["cache", "cache"]);
    }

    #[test]
    fn effects_replay_miss_and_store_refused() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(stub(echo)), None, None);
        effect_call(rec, 0, &mut || json!(1)).unwrap();
        let rep = proxy(&db, "replay", "r1", None, Some("base"), None);
        assert_eq!(effect_call(rep, 7, &mut || json!(1)), Err(409));
        let store = json!({ "name": "flaky", "n": 7, "args": [["x"], {}], "result": 1 });
        assert_eq!(post(rep, "/construct/effect", &store).0, 409);
        assert!(srcs(&db, "r1").is_empty());
    }

    // streaming

    #[test]
    fn sse_record_replay_byte_identical_and_join() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(stub(sse)), None, None);
        let req = json!({ "model": "s1", "messages": [{"role": "user", "content": "hi"}], "stream": true });
        let (_, live) = post(rec, "/v1/chat/completions", &req);
        let rep = proxy(&db, "replay", "r1", None, Some("base"), None);
        let (_, cached) = post(rep, "/v1/chat/completions", &req);
        assert_eq!(live, cached);
        let c = open(&db).unwrap();
        let e = &timeline(&c, "base").unwrap()[0];
        assert!(e.ctype.as_deref().unwrap().starts_with("text/event-stream"));
        let msg = &resp_obj(&c, e).unwrap()["choices"][0]["message"];
        assert_eq!(msg["content"], "Hello world");
        assert_eq!(norm_tools(&msg["tool_calls"]), json!([{"name": "f", "args": {"a": 1}}]));
    }

    #[test]
    fn sse_join_anthropic() {
        let body = concat!(
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi \"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"there\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"f\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"2}\"}}\n\n",
        );
        let msg = &sse_join(body.as_bytes())["choices"][0]["message"];
        assert_eq!(msg["content"], "Hi there");
        assert_eq!(norm_tools(&msg["tool_calls"]), json!([{"name": "f", "args": {"a": 2}}]));
    }

    // viewer

    fn base_and_fork(db: &str) -> Vec<String> {
        let up = stub(echo);
        let rec = proxy(db, "record", "base", Some(up), None, None);
        chat(rec, json!([{"role": "user", "content": "hi"}]), "m1");
        chat(rec, json!([{"role": "user", "content": "hi"}, {"role": "user", "content": "more"}]), "m1");
        let frk = proxy(db, "fork", "b1", Some(up), Some("base"), Some("m2"));
        chat(frk, json!([{"role": "user", "content": "hi"}]), "m1");
        chat(frk, json!([{"role": "user", "content": "hi"}, {"role": "user", "content": "MUTATED"}]), "m1");
        for (port, turn) in [(rec, "more"), (frk, "MUTATED")] {
            chat(port, json!([{"role": "user", "content": "hi"}, {"role": "user", "content": turn}, {"role": "user", "content": "go"}]), "m1");
        }
        vec!["base".into(), "b1".into()]
    }

    #[test]
    fn view_merges_identical_and_marks_divergence() {
        let db = tmpdb();
        let runs = base_and_fork(&db);
        let html = render_view(&open(&db).unwrap(), &runs).unwrap();
        assert!(html.contains(r#"<section class="step same" id="step-0">"#), "step 0 should merge");
        assert!(html.contains(r#"<section class="step diverged after" id="step-1">"#), "step 1 should diverge");
        assert!(html.contains("Identical through step 0; diverges at step 1."));
        assert!(html.contains("live in base, replayed in b1"));
        assert!(html.contains("What changed at step 1"));
        assert!(html.contains("<del>model=m1</del> <ins>model=m2</ins>"), "response word diff");
        assert!(html.contains("model swapped to m2"));
        assert!(html.contains("carries the change from step 1"), "later request diffs fold");
        let first = &html[html.find(r#"id="step-1""#).unwrap()..html.find(r#"id="step-2""#).unwrap()];
        assert!(!first.contains("Show request diff"), "the first divergence shows its request diff open");
    }

    #[test]
    fn view_is_self_contained_and_escaped() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(stub(echo)), None, None);
        chat(rec, json!([{"role": "user", "content": "<script>alert(1)</script>"}]), "<b>m</b>");
        let html = render_view(&open(&db).unwrap(), &["base".into()]).unwrap();
        assert!(!html.contains("<script>alert") && !html.contains("<b>m</b>"), "user content must be escaped");
        for external in ["<link", " src=", "@import", "url(", "@font-face"] {
            assert!(!html.contains(external), "timeline must not load {external}");
        }
    }

    #[test]
    fn view_single_and_empty_runs() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(stub(echo)), None, None);
        chat(rec, json!([{"role": "user", "content": "hi"}]), "m1");
        proxy(&db, "record", "empty", Some(1), None, None);
        let c = open(&db).unwrap();
        assert!(render_view(&c, &["base".into()]).unwrap().contains("1 steps: 1 live, 0 replayed."));
        assert!(render_view(&c, &["empty".into()]).unwrap().contains("This run recorded no steps."));
        let both = render_view(&c, &["base".into(), "empty".into()]).unwrap();
        assert!(both.contains("Diverges at the first step.") && both.contains("No step 0 in this run."));
    }

    #[test]
    fn tool_line_reads_like_a_call() {
        let tools = norm_tools(&json!([{"function": {"name": "search", "arguments": "{\"query\": \"volcanoes\", \"limit\": 3}"}}]));
        assert_eq!(tool_line(&tools), r#"search(limit: 3, query: "volcanoes")"#);
    }

    // differ

    #[test]
    fn word_diff_marks_changes() {
        assert_eq!(word_diff("a b c", "a x c", plain, ["[-", "-]"], ["{+", "+}"]), "a [-b-] {+x+} c");
        assert_eq!(word_diff("same here", "same here", plain, ["[-", "-]"], ["{+", "+}"]), "same here");
        assert_eq!(word_diff("a <b>", "a", esc, ["<del>", "</del>"], ["<ins>", "</ins>"]), "a <del>&lt;b&gt;</del>");
    }

    #[test]
    fn norm_tools_is_schema_aware() {
        let a = json!([{"function": {"name": "f", "arguments": "{\"x\": 1, \"y\": 2}"}}]);
        let b = json!([{"function": {"name": "f", "arguments": "{\"y\":2,\"x\":1}"}}]);
        assert_eq!(norm_tools(&a), norm_tools(&b));
    }

    #[test]
    fn assistant_payload_anthropic() {
        let resp = json!({"content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "name": "f", "input": {"a": 1}}]});
        assert_eq!(assistant_payload(&resp), ("hi".into(), json!([{"name": "f", "args": {"a": 1}}])));
    }

    // mcp

    fn rpc(c: &Connection, method: &str, params: Value) -> Value {
        mcp_handle(c, &json!({ "jsonrpc": "2.0", "id": 7, "method": method, "params": params })).expect("requests get a reply")
    }

    fn call(c: &Connection, tool: &str, args: Value) -> Value {
        rpc(c, "tools/call", json!({ "name": tool, "arguments": args }))["result"].clone()
    }

    #[test]
    fn mcp_initialize_negotiates_version() {
        let c = open(&tmpdb()).unwrap();
        let r = rpc(&c, "initialize", json!({ "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }));
        assert_eq!(r["id"], 7);
        assert_eq!(r["result"]["protocolVersion"], "2025-03-26", "a supported version is echoed");
        assert_eq!(r["result"]["capabilities"], json!({ "tools": {} }));
        let r = rpc(&c, "initialize", json!({ "protocolVersion": "1999-01-01" }));
        assert_eq!(r["result"]["protocolVersion"], MCP_VERSIONS[0], "unknown versions get the latest");
        assert_eq!(rpc(&c, "ping", json!({}))["result"], json!({}));
    }

    #[test]
    fn mcp_notifications_get_no_reply_and_unknowns_are_errors() {
        let c = open(&tmpdb()).unwrap();
        assert!(mcp_handle(&c, &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })).is_none());
        assert!(mcp_handle(&c, &json!({ "jsonrpc": "2.0", "id": 1, "result": {} })).is_none(), "stray responses are ignored");
        assert_eq!(rpc(&c, "resources/list", json!({}))["error"]["code"], -32601);
        assert_eq!(rpc(&c, "tools/call", json!({ "name": "rm_rf" }))["error"]["code"], -32602);
    }

    #[test]
    fn mcp_tools_are_read_only_and_declare_schemas() {
        let c = open(&tmpdb()).unwrap();
        let tools = rpc(&c, "tools/list", json!({}))["result"]["tools"].clone();
        let names: Vec<&str> = tools.as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["list_runs", "get_step", "diff_runs", "search_runs", "eval_run"]);
        for t in tools.as_array().unwrap() {
            assert_eq!(t["annotations"]["readOnlyHint"], true);
            assert_eq!(t["inputSchema"]["type"], "object");
            assert_eq!(t["outputSchema"]["type"], "object");
        }
    }

    /// Every required field the tool's outputSchema names is present in its structuredContent.
    fn assert_matches_schema(tool: &str, v: &Value) {
        let tools = mcp_tools();
        let schema = &tools.as_array().unwrap().iter().find(|t| t["name"] == tool).unwrap()["outputSchema"];
        for k in schema["required"].as_array().unwrap() {
            assert!(v.get(k.as_str().unwrap()).is_some(), "{tool}: missing required field {k}");
        }
    }

    #[test]
    fn mcp_tools_answer_from_the_log() {
        let db = tmpdb();
        base_and_fork(&db);
        let c = open_ro(&db).unwrap();

        let r = call(&c, "list_runs", json!({}));
        assert_matches_schema("list_runs", &r["structuredContent"]);
        let runs = r["structuredContent"]["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 2);
        let b1 = runs.iter().find(|r| r["name"] == "b1").unwrap();
        assert_eq!(b1["parent"], "base");
        assert_eq!(b1["model_sub"], "m2");
        assert_eq!((b1["steps"].as_i64(), b1["live"].as_i64()), (Some(3), Some(2)));

        let r = call(&c, "get_step", json!({ "run": "b1", "step": 1 }));
        let s = &r["structuredContent"];
        assert_matches_schema("get_step", s);
        assert_eq!((s["src"].as_str(), s["model"].as_str(), s["text"].as_str()), (Some("live"), Some("m2"), Some("reply-2 model=m2")));
        assert!(r.get("isError").is_none());

        let r = call(&c, "diff_runs", json!({ "run_a": "base", "run_b": "b1" }));
        assert_matches_schema("diff_runs", &r["structuredContent"]);
        assert_eq!(r["structuredContent"]["first_divergence"], 1);
        assert!(r["content"][0]["text"].as_str().unwrap().contains("! first divergence at step 1"), "text is the report");
    }

    #[test]
    fn mcp_tool_failures_are_reported_to_the_model() {
        let db = tmpdb();
        base_and_fork(&db);
        let c = open_ro(&db).unwrap();
        for (tool, args, says) in [
            ("get_step", json!({ "run": "b1", "step": 99 }), "has 3 steps"),
            ("get_step", json!({ "run": "nope", "step": 0 }), "no such run"),
            ("get_step", json!({ "run": "b1", "step": -1 }), "non-negative integer"),
            ("diff_runs", json!({ "run_a": "base" }), "'run_b' must be a string"),
        ] {
            let r = call(&c, tool, args);
            assert_eq!(r["isError"], true, "{tool}");
            assert!(r["content"][0]["text"].as_str().unwrap().contains(says), "{tool}: {}", r["content"][0]["text"]);
        }
    }

    #[test]
    fn mcp_connection_cannot_write() {
        let db = tmpdb();
        base_and_fork(&db);
        assert!(open_ro(&db).unwrap().execute("INSERT INTO objects (hash, data) VALUES ('x', x'00')", []).is_err());
        assert!(open_ro(&tmpdb()).is_err(), "a missing database is an error, not a new empty file");
    }

    // search

    fn hits(v: &Value) -> Vec<(String, i64)> {
        v["hits"].as_array().unwrap().iter().map(|h| (h["run"].as_str().unwrap().into(), h["step"].as_i64().unwrap())).collect()
    }

    #[test]
    fn search_finds_steps_ranks_and_filters() {
        let db = tmpdb();
        base_and_fork(&db);
        let c = open_ro(&db).unwrap();
        assert_eq!(hits(&search(&c, "MUTATED", None, 10).unwrap()), [("b1".to_string(), 1)], "newest input message is indexed");
        let r = search(&c, "model=m2", None, 10).unwrap();
        assert!(hits(&r).iter().all(|(run, _)| run == "b1"), "response text is indexed: {r}");
        assert!(r["hits"][0]["snippet"].as_str().unwrap().contains("[model=m2]"), "snippet marks the matched phrase: {r}");
        assert_eq!(hits(&search(&c, "reply", Some("base"), 10).unwrap()).len(), 3, "run filter");
        assert_eq!(hits(&search(&c, "reply", None, 2).unwrap()).len(), 2, "limit");
        assert!(hits(&search(&c, "volcano", None, 10).unwrap()).is_empty());
        assert!(search(&c, "   ", None, 10).is_err());
        for q in [r#"a"b"#, "NOT -x OR", "col:val", "(", "*"] {
            assert!(search(&c, q, None, 10).is_ok(), "FTS syntax in user text is literal: {q}");
        }
    }

    #[test]
    fn search_index_catches_up_on_a_live_connection() {
        let db = tmpdb();
        let up = stub(echo);
        let rec = proxy(&db, "record", "base", Some(up), None, None);
        chat(rec, json!([{"role": "user", "content": "first question"}]), "m1");
        let c = open_ro(&db).unwrap();
        assert_eq!(hits(&search(&c, "question", None, 10).unwrap()).len(), 1);
        chat(rec, json!([{"role": "user", "content": "second question"}]), "m1");
        assert_eq!(hits(&search(&c, "question", None, 10).unwrap()).len(), 2, "new events are indexed incrementally");
    }

    #[test]
    fn search_indexes_tool_calls_and_effects() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(stub(sse)), None, None);
        post(rec, "/v1/chat/completions", &json!({ "model": "m", "stream": true, "messages": [] }));
        effect_call(rec, 0, &mut || json!({ "forecast": "sunny" })).unwrap();
        let c = open_ro(&db).unwrap();
        assert_eq!(hits(&search(&c, "f", None, 10).unwrap()), [("base".to_string(), 0)], "tool call names");
        assert_eq!(hits(&search(&c, "sunny", None, 10).unwrap()), [("base".to_string(), 1)], "effect results");
    }

    // eval

    fn errs_for(v: Value, schema: Value) -> Vec<String> {
        let mut errs = Vec::new();
        validate(&v, &schema, "x", &mut errs).unwrap();
        errs
    }

    #[test]
    fn validate_covers_the_common_subset() {
        let schema = json!({
            "type": "object", "required": ["id", "tags"], "additionalProperties": false,
            "properties": {
                "id": { "type": "integer", "minimum": 1 },
                "tags": { "type": "array", "maxItems": 2, "items": { "type": "string", "enum": ["a", "b"] } },
                "note": { "type": ["string", "null"], "maxLength": 3 },
            },
        });
        assert!(errs_for(json!({ "id": 2, "tags": ["a"], "note": null }), schema.clone()).is_empty());
        assert!(errs_for(json!({ "id": 2.0, "tags": [] }), schema.clone()).is_empty(), "2.0 is an integer");
        let e = errs_for(json!({ "id": 0, "tags": ["a", "c", "b"], "note": "long", "extra": 1 }), schema.clone());
        for want in ["x.id: 0 is below 1", "x.tags: length 3 is above 2", "x.tags[1]: \"c\" is not one of", "x.note: length 4", "unexpected property 'extra'"] {
            assert!(e.iter().any(|m| m.contains(want)), "{want} in {e:?}");
        }
        assert!(errs_for(json!({ "tags": [] }), schema.clone()).iter().any(|m| m.contains("missing required 'id'")));
        assert!(errs_for(json!("s"), schema).iter().any(|m| m == "x: expected object, got string"));
        let mut errs = Vec::new();
        assert!(validate(&json!("s"), &json!({ "pattern": "^a" }), "x", &mut errs).is_err(), "unsupported keywords never silently pass");
    }

    #[test]
    fn unfence_strips_markdown_code_fences() {
        assert_eq!(unfence("```json\n{\"a\": 1}\n```"), "{\"a\": 1}");
        assert_eq!(unfence("  {\"a\": 1} "), "{\"a\": 1}");
        assert_eq!(unfence("```"), "```");
    }

    #[test]
    fn eval_text_checks_and_baseline_regressions() {
        let db = tmpdb();
        base_and_fork(&db);
        let c = open_ro(&db).unwrap();
        let checks = json!({ "checks": [
            { "name": "names the model", "includes": ["MODEL=M1"] },
            { "name": "replies", "step": "any", "includes": ["reply-1", "reply-3"] },
            { "name": "no apology", "excludes": ["sorry"] },
            { "name": "step 0", "step": 0, "includes": ["reply-1"] },
        ] });
        let base = eval_json(&c, "base", &checks, None).unwrap();
        assert_eq!((base["passed"].as_u64(), base["total"].as_u64()), (Some(4), Some(4)), "{base}");
        let fork = eval_json(&c, "b1", &checks, Some("base")).unwrap();
        assert_eq!(fork["regressions"], json!(["names the model"]), "{fork}");
        assert_eq!(fork["results"][0]["detail"], "missing \"MODEL=M1\"");
        assert_eq!(fork["results"][0]["baseline_pass"], true);
        let fixed = eval_json(&c, "base", &json!([{ "name": "m2", "includes": ["m2"] }]), Some("b1")).unwrap();
        assert_eq!(fixed["regressions"], json!(["m2"]));
        let fixed = eval_json(&c, "b1", &json!([{ "name": "m2", "includes": ["m2"] }]), Some("base")).unwrap();
        assert_eq!(fixed["fixes"], json!(["m2"]));
    }

    #[test]
    fn eval_tool_and_output_schemas() {
        let db = tmpdb();
        let rec = proxy(&db, "record", "base", Some(stub(sse)), None, None);
        post(rec, "/v1/chat/completions", &json!({ "model": "m", "stream": true, "messages": [] }));
        let c = open_ro(&db).unwrap();
        let int_a = json!({ "type": "object", "required": ["a"], "properties": { "a": { "type": "integer" } } });
        let r = eval_json(&c, "base", &json!([
            { "name": "f args", "tool": "f", "schema": int_a },
            { "name": "f args strict", "tool": "f", "schema": { "type": "object", "properties": { "a": { "type": "string" } } } },
            { "name": "g called", "tool": "g", "schema": {} },
            { "name": "json out", "schema": { "type": "object" } },
        ]), None).unwrap();
        let pass: Vec<bool> = r["results"].as_array().unwrap().iter().map(|x| x["pass"] == true).collect();
        assert_eq!(pass, [true, false, false, false], "{r}");
        assert_eq!(r["results"][2]["detail"], "g was never called");
        assert!(r["results"][3]["detail"].as_str().unwrap().starts_with("output is not JSON"));
    }

    #[test]
    fn eval_rejects_malformed_checks() {
        let db = tmpdb();
        base_and_fork(&db);
        let c = open_ro(&db).unwrap();
        for (checks, says) in [
            (json!([]), "no checks"),
            (json!([{ "name": "empty" }]), "empty: a check needs"),
            (json!([{ "includes": ["x"], "step": 9 }]), "check 0: step 9 is not an LLM call"),
            (json!([{ "includes": ["x"], "step": "first" }]), "'step' must be"),
            (json!({ "nope": 1 }), "checks must be"),
        ] {
            let e = eval_json(&c, "base", &checks, None).unwrap_err().to_string();
            assert!(e.contains(says), "{says}: {e}");
        }
    }

    #[test]
    fn mcp_search_and_eval_tools() {
        let db = tmpdb();
        base_and_fork(&db);
        let c = open_ro(&db).unwrap();
        let r = call(&c, "search_runs", json!({ "query": "MUTATED" }));
        assert_matches_schema("search_runs", &r["structuredContent"]);
        assert_eq!(r["structuredContent"]["hits"][0]["run"], "b1");
        let r = call(&c, "eval_run", json!({ "run": "b1", "baseline": "base", "checks": [{ "name": "m1", "includes": ["m1"] }] }));
        assert_matches_schema("eval_run", &r["structuredContent"]);
        assert_eq!(r["structuredContent"]["regressions"], json!(["m1"]));
        assert_eq!(call(&c, "eval_run", json!({ "run": "b1", "checks": [{}] }))["isError"], true);
    }

    // local LLMs: Ollama's native API (OpenAI-compatible local servers are covered by the tests above)

    fn ollama(req: &Value) -> (Vec<u8>, &'static str) {
        let model = req["model"].as_str().unwrap_or("");
        let generate = req.get("prompt").is_some();
        let say = |s: &str| if generate { json!({ "model": model, "response": s, "done": false }) } else { json!({ "model": model, "message": { "role": "assistant", "content": s }, "done": false }) };
        let call = json!({ "model": model, "message": { "role": "assistant", "content": "", "tool_calls": [{ "function": { "name": "lookup", "arguments": { "city": "Ogden" } } }] }, "done": false });
        if req["stream"] == false {
            let mut one = say(&format!("Weather check with {model}"));
            one["done"] = json!(true);
            if !generate {
                one["message"]["tool_calls"] = call["message"]["tool_calls"].clone();
            }
            return (canon(&one), "application/json; charset=utf-8");
        }
        let mut lines = vec![say("Weather "), say(&format!("check with {model}"))];
        if !generate {
            lines.push(call);
        }
        lines.push(json!({ "model": model, "done": true, "eval_count": 12 }));
        (lines.iter().flat_map(|l| [canon(l), b"\n".to_vec()].concat()).collect(), "application/x-ndjson")
    }

    #[test]
    fn ollama_native_chat_and_generate_record_replay_and_read() {
        let db = tmpdb();
        let up = stub(ollama);
        let rec = proxy(&db, "record", "local", Some(up), None, None);
        let msgs = json!([{ "role": "user", "content": "weather in Ogden?" }]);
        let (_, streamed) = post(rec, "/api/chat", &json!({ "model": "llama3.2", "messages": msgs }));
        let (_, plain) = post(rec, "/api/chat", &json!({ "model": "llama3.2", "messages": msgs, "stream": false }));
        let (_, generated) = post(rec, "/api/generate", &json!({ "model": "llama3.2", "prompt": "say hi" }));

        let rep = proxy(&db, "replay", "again", None, Some("local"), None);
        assert_eq!(post(rep, "/api/chat", &json!({ "model": "llama3.2", "messages": msgs })).1, streamed, "NDJSON replays byte-identically");
        assert_eq!(post(rep, "/api/chat", &json!({ "model": "llama3.2", "messages": msgs, "stream": false })).1, plain);
        assert_eq!(post(rep, "/api/generate", &json!({ "model": "llama3.2", "prompt": "say hi" })).1, generated);

        let c = open_ro(&db).unwrap();
        for step in [0, 1] {
            let s = step_json(&c, "local", step).unwrap();
            assert_eq!(s["text"], "Weather check with llama3.2", "step {step}: {s}");
            assert_eq!(s["tool_calls"], json!([{ "name": "lookup", "args": { "city": "Ogden" } }]), "step {step}");
        }
        assert_eq!(step_json(&c, "local", 2).unwrap()["text"], "Weather check with llama3.2");
        assert_eq!(hits(&search(&c, "Ogden", None, 10).unwrap()).len(), 4, "prompt, args indexed: {}", search(&c, "Ogden", None, 10).unwrap());
        assert_eq!(hits(&search(&c, "say hi", Some("local"), 10).unwrap()), [("local".to_string(), 2)], "generate prompt indexed");
        let r = eval_json(&c, "local", &json!([
            { "name": "names the model", "step": 0, "includes": ["llama3.2"] },
            { "name": "lookup args", "tool": "lookup", "schema": { "type": "object", "required": ["city"], "properties": { "city": { "type": "string" } } } },
        ]), None).unwrap();
        assert_eq!(r["passed"], 2, "{r}");
        let html = render_view(&c, &["local".into()]).unwrap();
        assert!(html.contains("Response (reassembled from a stream)"));
        assert!(html.contains(r#"<p class="sum">Weather check with llama3.2</p>"#), "reassembled text is the step summary");
        assert!(html.contains("&quot;city&quot;: &quot;Ogden&quot;"), "tool call args appear in the evidence");
    }

    #[test]
    fn ollama_fork_swaps_the_local_model_and_diffs() {
        let db = tmpdb();
        let up = stub(ollama);
        let msgs = json!([{ "role": "user", "content": "weather in Ogden?" }]);
        let rec = proxy(&db, "record", "base", Some(up), None, None);
        post(rec, "/api/chat", &json!({ "model": "llama3.2", "messages": msgs }));
        let frk = proxy(&db, "fork", "qwen", Some(up), Some("base"), Some("qwen2.5"));
        post(frk, "/api/chat", &json!({ "model": "llama3.2", "messages": msgs, "options": { "temperature": 0 } }));
        let (report, first, _, _) = diff_report(&open_ro(&db).unwrap(), "base", "qwen").unwrap();
        assert_eq!(first, Some(0));
        assert!(report.contains("[-llama3.2-] {+qwen2.5+}"), "{report}");
    }
}
