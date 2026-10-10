// AI backend for The Computational Learning Dynamics Atlas.
//
// A Cloudflare Worker (ES module syntax) with two endpoints that match the
// static site's client contract in app.js:
//
//   POST /api/explain  {edge, from}                 -> why a recorded connection holds
//   POST /api/chat     {messages, current, path}    -> grounded Q&A about the atlas
//
// Both answer with Server-Sent Events, one JSON object per event:
//   data: {"t":"text chunk"}   data: {"done":true}   data: {"error":"message"}
//
// The prompts are built here from atlas ids. Topic and connection text sent by
// the browser is never used: the worker loads atlas.json itself and looks the
// ids up, so the evidence label always comes from the atlas record.
//
// Configuration lives in wrangler.toml [vars]; the API key is a secret
// (`wrangler secret put ANTHROPIC_API_KEY`). See README.md.

const DEFAULTS = {
  MODEL: 'claude-haiku-5-5',            // cheapest current model; 'claude-sonnet-5-5' for higher quality
  EFFORT: 'low',                         // output_config.effort; '' to leave the model default
  THINKING: '',                          // '' = model default (adaptive); 'disabled' is accepted by Haiku 5.5 at effort <= high
  ATLAS_URL: 'https://apumutyala.github.io/computational-learning-dynamics-atlas/atlas.json',
  ALLOWED_ORIGINS: 'https://apumutyala.github.io',
  ALLOW_LOCALHOST: 'true',               // also allow http://localhost:* and http://127.0.0.1:*
  ANTHROPIC_BASE_URL: 'https://api.anthropic.com',
  CHAT_MAX_TOKENS: 600,
  EXPLAIN_MAX_TOKENS: 350,
  MAX_BODY_BYTES: 32768,
  MAX_MESSAGES: 12,
  MAX_USER_CHARS: 2000,                  // the chat textarea has maxlength=2000
  MAX_ASSISTANT_CHARS: 4000,
  RATE_PER_MINUTE: 6,                    // per-IP token bucket refill rate
  RATE_BURST: 10,                        // per-IP bucket size
  DAILY_REQUEST_CAP: 2000,               // all model calls per UTC day; 0 = AI off
  ATLAS_TTL_SECONDS: 600,
  EXPLAIN_CACHE_TTL_SECONDS: 604800,
};

const ANTHROPIC_VERSION = '2023-06-01';
const EVIDENCE = {
  exact: 'Exact, conditional: an exact identity that holds only under the stated assumptions',
  established: 'Established: a well-supported connection in the literature',
  empirical: 'Empirical: an observed association, not a guarantee',
  hypothesis: 'Hypothesis: an untested hypothesis',
};

function cfg(env, key) {
  const v = env && env[key];
  if (v === undefined || v === null || v === '') return DEFAULTS[key];
  return typeof DEFAULTS[key] === 'number' ? Number(v) : String(v);
}

/* ------------------------------------------------------------------ atlas */

let atlasCache = null; // { at, url, atlas, byId, edgeById, adj }

function indexAtlas(atlas) {
  const byId = new Map(atlas.nodes.map(n => [n.id, n]));
  const edgeById = new Map(atlas.edges.map(e => [e.id, e]));
  const adj = new Map(atlas.nodes.map(n => [n.id, []]));
  for (const e of atlas.edges) { adj.get(e.a)?.push(e); adj.get(e.b)?.push(e); }
  return { atlas, byId, edgeById, adj };
}

async function loadAtlas(env) {
  const url = cfg(env, 'ATLAS_URL');
  const ttl = cfg(env, 'ATLAS_TTL_SECONDS') * 1000;
  if (atlasCache && atlasCache.url === url && Date.now() - atlasCache.at < ttl) return atlasCache;
  try {
    const res = await fetch(url, { cf: { cacheTtl: cfg(env, 'ATLAS_TTL_SECONDS'), cacheEverything: true } });
    if (!res.ok) throw new Error(`atlas fetch ${res.status}`);
    const atlas = await res.json();
    if (!Array.isArray(atlas.nodes) || !Array.isArray(atlas.edges)) throw new Error('atlas shape');
    atlasCache = { at: Date.now(), url, ...indexAtlas(atlas) };
  } catch (err) {
    // Serve a stale copy rather than failing if the Pages site is briefly unreachable.
    if (!atlasCache || atlasCache.url !== url) throw err;
    console.warn('atlas refresh failed, serving stale copy:', err.message);
    atlasCache.at = Date.now() - ttl + 60_000; // retry in a minute
  }
  return atlasCache;
}

/* --------------------------------------------------------------- prompts */

const nameOf = (A, id) => A.byId.get(id)?.name ?? id;

function topicRecord(n) {
  return [
    `[[${n.id}]] ${n.name}`,
    `Cluster: ${n.group}`,
    n.authors ? `Key sources: ${n.authors}${n.year ? ` (${n.year})` : ''}` : null,
    `Summary: ${n.text}`,
    n.handle ? `Control handle: ${n.handle}` : null,
    n.boundary ? `Boundary: ${n.boundary}` : null,
    n.test ? `Suggested test: ${n.test}` : null,
    n.url ? `Reference: ${n.url}` : null,
  ].filter(Boolean).join('\n');
}

function edgeLine(A, e, fromId) {
  const to = e.a === fromId ? e.b : e.a;
  return `- ${nameOf(A, fromId)} -> [[${to}]] ${nameOf(A, to)} | evidence: ${e.type} | ${e.relation}: ${e.text}`;
}

function topicIndex(A) {
  const lines = [];
  for (const c of A.atlas.clusters || []) {
    const members = A.atlas.nodes.filter(n => n.group === c.name).map(n => `${n.id}=${n.name}`);
    lines.push(`${c.name}: ${members.join('; ')}`);
  }
  return lines.join('\n');
}

export function buildExplainPrompt(A, edge, fromId) {
  const from = A.byId.get(fromId), to = A.byId.get(edge.a === fromId ? edge.b : edge.a);
  const system = [
    'You explain one recorded connection in a curated research atlas about learning dynamics, control and alignment.',
    `The atlas labels this connection's evidence as "${EVIDENCE[edge.type] ?? edge.type}". That label is fixed by the atlas.`,
    'Never describe the connection as stronger, more general or better established than that label. If it is a hypothesis or empirical association, say so plainly.',
    'Use only the records given. If they do not support a claim, leave it out.',
    'Write 2 to 4 plain sentences for a curious newcomer, starting from the topic the reader is viewing. Plain text only: no markdown, no headings, no lists, no links.',
  ].join('\n');
  const user = [
    `The reader is viewing: ${from.name}.`,
    '',
    'Recorded connection:',
    `${from.name} <-> ${to.name}`,
    `Relation: ${edge.relation}`,
    `Evidence: ${edge.type}`,
    `Statement: ${edge.text}`,
    edge.url ? `Source: ${edge.url}` : null,
    '',
    'Topic records:',
    topicRecord(from),
    '',
    topicRecord(to),
    '',
    'Explain why this connection holds, within the stated evidence level.',
  ].filter(x => x !== null).join('\n');
  return { system, messages: [{ role: 'user', content: user }] };
}

export function buildChatSystem(A, current, path) {
  const parts = [
    `You are the guide inside "${A.atlas.title || 'the atlas'}", a curated map of topics in learning dynamics, control and alignment (The Computational Learning Dynamics Atlas).`,
    `Atlas scope: ${A.atlas.scope || ''}`,
    'Ground every answer in the atlas records below. When you mention an atlas topic, cite it as [[topic-id]] using the ids listed; the site turns those into links. Cite only ids that appear below.',
    'Each connection carries an evidence label (exact, established, empirical, hypothesis). Keep it: call hypotheses hypotheses and empirical links associations, and never upgrade a label. If the atlas does not cover something, say so and keep any outside knowledge clearly marked as general background.',
    'Be concise: at most about 180 words. Short paragraphs or a short list. No headings, no tables.',
    '',
    'All atlas topics by cluster (id=name):',
    topicIndex(A),
  ];
  const n = current && A.byId.get(current);
  if (n) {
    parts.push('', 'The reader is currently viewing this topic:', topicRecord(n));
    const es = A.adj.get(n.id) || [];
    if (es.length) parts.push('', `Its recorded connections (${es.length}):`, ...es.map(e => edgeLine(A, e, n.id)));
    const neighbourIds = [...new Set(es.map(e => (e.a === n.id ? e.b : e.a)))].slice(0, 8);
    if (neighbourIds.length) parts.push('', 'Neighbouring topic summaries:', ...neighbourIds.map(id => `- [[${id}]] ${nameOf(A, id)}: ${A.byId.get(id).text}`));
  } else {
    parts.push('', 'The reader has no topic selected.');
  }
  if (path.length) parts.push('', `The reader's recent trail (oldest first): ${path.map(id => `[[${id}]] ${nameOf(A, id)}`).join(' -> ')}`);
  return parts.join('\n');
}

/* ------------------------------------------------------------ validation */

class HttpError extends Error {
  constructor(status, message, headers = {}) { super(message); this.status = status; this.headers = headers; }
}

async function readJsonCapped(request, maxBytes) {
  const len = Number(request.headers.get('content-length') || 0);
  if (len > maxBytes) throw new HttpError(413, 'Request too large.');
  if (!request.body) throw new HttpError(400, 'Empty request.');
  const reader = request.body.getReader();
  const chunks = []; let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) { try { await reader.cancel(); } catch {} throw new HttpError(413, 'Request too large.'); }
    chunks.push(value);
  }
  const buf = new Uint8Array(total); let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
  try { return JSON.parse(new TextDecoder().decode(buf)); } catch { throw new HttpError(400, 'Invalid JSON.'); }
}

export function sanitizeMessages(raw, env) {
  if (!Array.isArray(raw)) throw new HttpError(400, 'messages must be an array.');
  const maxU = cfg(env, 'MAX_USER_CHARS'), maxA = cfg(env, 'MAX_ASSISTANT_CHARS');
  let msgs = raw.slice(-cfg(env, 'MAX_MESSAGES'))
    .filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim())
    .map(m => ({ role: m.role, content: m.content.slice(0, m.role === 'user' ? maxU : maxA) }));
  // The API wants alternating turns that start and end with the user.
  while (msgs.length && msgs[0].role !== 'user') msgs.shift();
  const merged = [];
  for (const m of msgs) {
    const last = merged[merged.length - 1];
    if (last && last.role === m.role) last.content += '\n\n' + m.content; else merged.push({ ...m });
  }
  while (merged.length && merged[merged.length - 1].role !== 'user') merged.pop();
  if (!merged.length) throw new HttpError(400, 'No question to answer.');
  return merged;
}

/* --------------------------------------------------- rate limits and caps */

// Per-IP token bucket. In-memory state lives in one Worker isolate, so a busy
// deployment spread over many isolates/colos enforces it loosely (each isolate
// has its own buckets). The daily cap below is the real spend bound.
const buckets = new Map();
export function takeToken(ip, env, now = Date.now()) {
  const rate = cfg(env, 'RATE_PER_MINUTE') / 60_000, burst = cfg(env, 'RATE_BURST');
  let b = buckets.get(ip);
  if (!b) { b = { tokens: burst, at: now }; buckets.set(ip, b); }
  b.tokens = Math.min(burst, b.tokens + (now - b.at) * rate); b.at = now;
  if (buckets.size > 5000) for (const [k, v] of buckets) if (v.tokens >= burst) buckets.delete(k);
  if (b.tokens < 1) return { ok: false, retryAfter: Math.ceil((1 - b.tokens) / rate / 1000) };
  b.tokens -= 1;
  return { ok: true };
}

// Global daily cap on model calls. With a KV namespace bound as ATLAS_KV the
// count is shared across isolates (KV is eventually consistent, so it can
// overshoot slightly under bursts; KV writes are also billed - see README).
// Without KV the count is per isolate, which bounds spend per isolate only.
const memDay = { day: '', count: 0 };
async function takeDaily(env) {
  const cap = cfg(env, 'DAILY_REQUEST_CAP');
  if (cap <= 0) return false;
  const day = new Date().toISOString().slice(0, 10);
  if (memDay.day !== day) { memDay.day = day; memDay.count = 0; }
  if (env.ATLAS_KV) {
    const key = `daily:${day}`;
    const n = Number(await env.ATLAS_KV.get(key)) || 0;
    if (n >= cap) return false;
    await env.ATLAS_KV.put(key, String(n + 1), { expirationTtl: 172800 });
    return true;
  }
  if (memDay.count >= cap) return false;
  memDay.count++;
  return true;
}

export function _resetState() { buckets.clear(); memDay.day = ''; memDay.count = 0; atlasCache = null; explainMem.clear(); }

/* ------------------------------------------------------------------- CORS */

function allowedOrigin(origin, env) {
  if (!origin) return null;
  const list = cfg(env, 'ALLOWED_ORIGINS').split(',').map(s => s.trim()).filter(Boolean);
  if (list.includes(origin)) return origin;
  if (cfg(env, 'ALLOW_LOCALHOST') === 'true' && /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) return origin;
  return null;
}

function corsHeaders(origin) {
  return origin ? {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  } : { 'Vary': 'Origin' };
}

const sse = obj => `data: ${JSON.stringify(obj)}\n\n`;

function errorResponse(status, message, cors, extra = {}) {
  // SSE-shaped body so an SSE reader could show the message; status stays honest.
  return new Response(sse({ error: message }), {
    status, headers: { ...cors, ...extra, 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-store' },
  });
}

/* --------------------------------------------------------------- upstream */

function upstreamBody(env, { system, messages, maxTokens }) {
  const body = { model: cfg(env, 'MODEL'), max_tokens: maxTokens, stream: true, system, messages };
  const effort = cfg(env, 'EFFORT');
  if (effort && effort !== 'default') body.output_config = { effort };
  const thinking = cfg(env, 'THINKING');
  if (thinking) body.thinking = { type: thinking };
  return body;
}

// Translate the Messages API event stream into the client's {"t"}/{"done"}/{"error"} events.
async function pipeAnthropic(upstream, writer, onText) {
  const enc = new TextEncoder();
  const write = s => writer.write(enc.encode(s));
  const reader = upstream.body.getReader(), dec = new TextDecoder();
  let buf = '', finished = false, usage = {};
  const handle = async (event, data) => {
    if (!data) return;
    let ev; try { ev = JSON.parse(data); } catch { return; }
    const type = ev.type || event;
    if (type === 'message_start') usage.input = ev.message?.usage?.input_tokens;
    else if (type === 'content_block_delta' && ev.delta?.type === 'text_delta' && ev.delta.text) {
      onText?.(ev.delta.text);
      await write(sse({ t: ev.delta.text }));
    } else if (type === 'message_delta') {
      if (ev.usage?.output_tokens != null) usage.output = ev.usage.output_tokens;
      if (ev.delta?.stop_reason === 'refusal') { finished = true; await write(sse({ error: 'The model declined to answer this one. Try rephrasing the question.' })); }
      else if (ev.delta?.stop_reason === 'max_tokens') await write(sse({ t: ' …' }));
    } else if (type === 'message_stop') {
      if (!finished) { finished = true; await write(sse({ done: true })); }
    } else if (type === 'error') {
      finished = true;
      const busy = /overloaded|rate_limit/.test(ev.error?.type || '');
      await write(sse({ error: busy ? 'The AI service is busy. Try again in a moment.' : 'The AI answer was interrupted.' }));
    }
    // ping, content_block_start/stop and thinking/signature deltas are ignored.
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        let event = '', data = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) data += line.slice(5).trim();
        }
        await handle(event, data);
        if (finished) break;
      }
      if (finished) break;
    }
    if (!finished) await write(sse({ error: 'The AI answer was interrupted.' }));
  } catch (err) {
    // Usually the browser closed the connection (navigation or abort).
    try { await reader.cancel(); } catch {}
    finished = false;
  } finally {
    try { await reader.cancel(); } catch {}
    try { await writer.close(); } catch {}
    console.log(JSON.stringify({ usage }));
  }
  return finished;
}

async function streamFromModel(env, ctx, cors, prompt, onComplete) {
  if (!env.ANTHROPIC_API_KEY) return errorResponse(503, 'The AI backend is not configured.', cors);
  const upstream = await fetch(`${cfg(env, 'ANTHROPIC_BASE_URL')}/v1/messages`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': env.ANTHROPIC_API_KEY,
      'anthropic-version': ANTHROPIC_VERSION,
    },
    body: JSON.stringify(upstreamBody(env, prompt)),
  }).catch(() => null);
  const headers = { ...cors, 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no' };
  if (!upstream || !upstream.ok || !upstream.body) {
    const status = upstream?.status;
    if (upstream) console.error('upstream error', status, (await upstream.text().catch(() => '')).slice(0, 500));
    const msg = status === 429 || status === 529 ? 'The AI service is busy. Try again in a moment.' : 'The AI service is unavailable right now.';
    // The client contract delivers failures as an SSE error event on a 200 stream.
    return new Response(sse({ error: msg }), { status: 200, headers });
  }
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  let text = '';
  const done = pipeAnthropic(upstream, writer, t => { text += t; }).then(ok => { if (ok && onComplete) return onComplete(text); });
  ctx?.waitUntil?.(done);
  return new Response(readable, { status: 200, headers });
}

/* --------------------------------------------------------------- handlers */

const explainMem = new Map();

async function handleExplain(body, env, ctx, cors, ip) {
  const A = await loadAtlas(env);
  const edge = typeof body.edge === 'string' && A.edgeById.get(body.edge);
  if (!edge) throw new HttpError(400, 'Unknown connection.');
  const fromId = body.from === edge.b ? edge.b : edge.a;
  const key = `explain:${cfg(env, 'MODEL')}:${edge.id}:${fromId}:${hashEdge(edge)}`;
  const cached = explainMem.get(key) ?? (env.ATLAS_KV ? await env.ATLAS_KV.get(key) : null);
  if (cached) {
    explainMem.set(key, cached);
    return new Response(sse({ t: cached }) + sse({ done: true }), { headers: { ...cors, 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-store', 'X-Cache': 'hit' } });
  }
  await guard(env, ip);
  const prompt = { ...buildExplainPrompt(A, edge, fromId), maxTokens: cfg(env, 'EXPLAIN_MAX_TOKENS') };
  return streamFromModel(env, ctx, cors, prompt, async text => {
    if (!text.trim()) return;
    if (explainMem.size > 2000) explainMem.clear();
    explainMem.set(key, text);
    if (env.ATLAS_KV) await env.ATLAS_KV.put(key, text, { expirationTtl: cfg(env, 'EXPLAIN_CACHE_TTL_SECONDS') });
  });
}

// Explanations are cached per edge text, so editing a connection in atlas.json invalidates its summary.
function hashEdge(e) {
  let h = 2166136261;
  for (const ch of `${e.type}|${e.relation}|${e.text}`) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(36);
}

async function handleChat(body, env, ctx, cors, ip) {
  const A = await loadAtlas(env);
  const messages = sanitizeMessages(body.messages, env);
  const current = typeof body.current === 'string' && A.byId.has(body.current) ? body.current : null;
  const path = (Array.isArray(body.path) ? body.path : []).filter(id => typeof id === 'string' && A.byId.has(id)).slice(-10);
  await guard(env, ip);
  return streamFromModel(env, ctx, cors, { system: buildChatSystem(A, current, path), messages, maxTokens: cfg(env, 'CHAT_MAX_TOKENS') });
}

async function guard(env, ip) {
  const t = takeToken(ip, env);
  if (!t.ok) throw new HttpError(429, 'Too many questions in a short time. Try again in a minute.', { 'Retry-After': String(t.retryAfter) });
  if (!(await takeDaily(env))) throw new HttpError(429, 'The AI features have reached their daily limit. They reset at 00:00 UTC.', { 'Retry-After': '3600' });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const origin = allowedOrigin(request.headers.get('Origin'), env);
    const cors = corsHeaders(origin);

    if (request.method === 'GET' && url.pathname === '/health') {
      return new Response(JSON.stringify({ ok: true, model: cfg(env, 'MODEL'), configured: !!env.ANTHROPIC_API_KEY }), { headers: { ...cors, 'Content-Type': 'application/json' } });
    }
    const route = url.pathname === '/api/explain' ? handleExplain : url.pathname === '/api/chat' ? handleChat : null;
    if (!route) return new Response('Not found', { status: 404, headers: cors });
    if (!origin) return new Response('Origin not allowed', { status: 403, headers: cors });
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });
    if (request.method !== 'POST') return new Response('Method not allowed', { status: 405, headers: { ...cors, Allow: 'POST, OPTIONS' } });

    try {
      const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
      const body = await readJsonCapped(request, cfg(env, 'MAX_BODY_BYTES'));
      if (!body || typeof body !== 'object') throw new HttpError(400, 'Invalid request.');
      return await route(body, env, ctx, cors, ip);
    } catch (err) {
      if (err instanceof HttpError) return errorResponse(err.status, err.message, cors, err.headers);
      console.error('handler error', err && err.stack || err);
      return errorResponse(500, 'The AI service is unavailable right now.', cors);
    }
  },
};
