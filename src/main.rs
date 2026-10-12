//! AROK Construct: record, replay, fork, and diff LLM agent runs.
//!
//! One binary, one SQLite file. Point any OpenAI/Anthropic-compatible client at
//! the proxy (`OPENAI_BASE_URL=http://127.0.0.1:8787/v1`) and run your agent.
//! FORMAT.md is the storage contract; this file is its implementation.

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use similar::{capture_diff_slices, Algorithm, DiffTag, TextDiff};
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::io::Read;
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
    let mut st = c.prepare(
        "SELECT type, path, name, key, req, resp, status, src, ctype FROM events WHERE run = ?1 ORDER BY seq",
    )?;
    let rows = st.query_map([run], |r| {
        Ok(Ev {
            typ: r.get(0)?,
            path: r.get(1)?,
            name: r.get(2)?,
            key: r.get(3)?,
            req: r.get(4)?,
            resp: r.get(5)?,
            status: r.get(6)?,
            src: r.get(7)?,
            ctype: r.get(8)?,
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
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
    let mut msg = json!({ "role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) } });
    if !tools.is_empty() {
        msg["tool_calls"] = Value::Array(tools.into_values().collect());
    }
    let mut out = json!({ "choices": [{ "index": 0, "message": msg, "finish_reason": "stop" }] });
    if !model.is_null() {
        out["model"] = model;
    }
    out
}

/// An event's response as JSON, reassembling SSE when needed.
fn resp_obj(c: &Connection, e: &Ev) -> R<Value> {
    let body = get(c, &e.resp)?;
    if e.ctype.as_deref().unwrap_or("").starts_with("text/event-stream") {
        return Ok(sse_join(&body));
    }
    Ok(serde_json::from_slice(&body).unwrap_or_else(|_| json!(String::from_utf8_lossy(&body))))
}

// ---------- diff ----------

/// (text, normalized tool calls) from an OpenAI- or Anthropic-shaped response.
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
                let raw = f["arguments"].as_str().unwrap_or("null");
                json!({ "name": f["name"], "args": serde_json::from_str::<Value>(raw).unwrap_or_else(|_| json!(raw)) })
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

fn cmd_diff(c: &Connection, ra: &str, rb: &str) -> R<()> {
    let (ca, cb) = (timeline(c, ra)?, timeline(c, rb)?);
    println!("diff {ra} ({} steps) <-> {rb} ({} steps)", ca.len(), cb.len());
    let mut diverged = false;
    for i in 0..ca.len().max(cb.len()) {
        let (ea, eb) = (ca.get(i), cb.get(i));
        if let (Some(x), Some(y)) = (ea, eb) {
            if x.key == y.key && x.resp == y.resp {
                println!("= step {i}  identical  {} {} ({})", x.typ, x.label(), &x.resp[..12]);
                continue;
            }
        }
        if !diverged {
            diverged = true;
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
    if !diverged {
        println!("runs are identical");
    }
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
    let streamed = e.ctype.as_deref().unwrap_or("").starts_with("text/event-stream");
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

fn cmd_runs(c: &Connection) -> R<()> {
    let mut st = c.prepare(
        "SELECT name, mode, (SELECT count(*) FROM events WHERE run = runs.name),
                strftime('%Y-%m-%d %H:%M', ts, 'unixepoch', 'localtime'), parent
         FROM runs ORDER BY name",
    )?;
    let rows = st.query_map([], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?, r.get::<_, Option<String>>(4)?))
    })?;
    for row in rows {
        let (name, mode, n, when, parent) = row?;
        let parent = parent.map(|p| format!(" <- {p}")).unwrap_or_default();
        println!("{name:24} {mode:7} {n:3} steps  {when}{parent}");
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
        "diff" | "runs" => &[],
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
}
