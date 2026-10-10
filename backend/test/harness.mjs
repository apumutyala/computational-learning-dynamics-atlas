// Local, no-network harness: a mock Anthropic Messages API, a Node http adapter
// for the Worker's fetch handler, and a static file server for the site.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// Repository root (the static site), two levels up from backend/test/.
export const WORKTREE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
export const worker = (await import(pathToFileURL(path.join(WORKTREE, 'backend/worker.js')).href));

/* ---------- mock Anthropic ---------- */
export const mock = { mode: 'ok', requests: [], delayMs: 40 };
const wait = ms => new Promise(r => setTimeout(r, ms));
const ev = (type, obj) => `event: ${type}\ndata: ${JSON.stringify({ type, ...obj })}\n\n`;

export function startMock(port) {
  const srv = http.createServer(async (req, res) => {
    let raw = ''; for await (const c of req) raw += c;
    const body = JSON.parse(raw || '{}');
    mock.requests.push({ url: req.url, headers: req.headers, body });
    if (mock.mode === 'http529') { res.writeHead(529, { 'content-type': 'application/json' }); return res.end('{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'); }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    const isChat = !String(body.system).includes('You explain one recorded connection');
    const text = isChat
      ? ['Ridge regression ', 'is the posterior mean of a Gaussian model, ', 'which is why it links to ', '[[gp]].', '\n\nThe atlas marks that link as **exact, conditional**.']
      : ['Under Gaussian noise and a Gaussian prior, ', 'the ridge predictor equals the GP posterior mean. ', 'That identity holds only under those assumptions.'];
    res.write(ev('message_start', { message: { id: 'msg_mock', type: 'message', role: 'assistant', model: body.model, content: [], usage: { input_tokens: 1234, output_tokens: 1 } } }));
    res.write(ev('content_block_start', { index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } }));
    res.write(ev('content_block_delta', { index: 0, delta: { type: 'signature_delta', signature: 'sig' } }));
    res.write(ev('content_block_stop', { index: 0 }));
    res.write(ev('ping', {}));
    res.write(ev('content_block_start', { index: 1, content_block: { type: 'text', text: '' } }));
    if (mock.mode === 'refusal') {
      res.write(ev('message_delta', { delta: { stop_reason: 'refusal', stop_sequence: null }, usage: { output_tokens: 3 } }));
      res.write(ev('message_stop', {})); return res.end();
    }
    for (const t of text) {
      await wait(mock.delayMs);
      // split one event across two TCP writes to exercise buffering
      const s = ev('content_block_delta', { index: 1, delta: { type: 'text_delta', text: t } });
      res.write(s.slice(0, 10)); await wait(2); res.write(s.slice(10));
    }
    if (mock.mode === 'midstream_error') { res.write(ev('error', { error: { type: 'overloaded_error', message: 'Overloaded' } })); return res.end(); }
    if (mock.mode === 'truncate') return res.end(); // connection drops before message_stop
    res.write(ev('content_block_stop', { index: 1 }));
    res.write(ev('message_delta', { delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 42 } }));
    res.write(ev('message_stop', {}));
    res.end();
  });
  return new Promise(r => srv.listen(port, '127.0.0.1', () => r(srv)));
}

/* ---------- Worker adapter (Node http -> fetch handler) ---------- */
export function startWorker(port, env) {
  const srv = http.createServer(async (req, res) => {
    const chunks = []; for await (const c of req) chunks.push(c);
    const hasBody = !['GET', 'HEAD'].includes(req.method);
    const headers = new Headers();
    for (const [k, v] of Object.entries(req.headers)) headers.set(k, Array.isArray(v) ? v.join(', ') : v);
    if (!headers.has('cf-connecting-ip')) headers.set('cf-connecting-ip', req.socket.remoteAddress);
    const request = new Request(`http://127.0.0.1:${port}${req.url}`, { method: req.method, headers, body: hasBody ? Buffer.concat(chunks) : undefined, duplex: 'half' });
    const pending = [];
    const response = await worker.default.fetch(request, env, { waitUntil: p => pending.push(p), passThroughOnException() {} });
    res.writeHead(response.status, Object.fromEntries(response.headers));
    if (response.body) {
      const reader = response.body.getReader();
      req.on('close', () => reader.cancel().catch(() => {}));
      for (;;) { const { value, done } = await reader.read(); if (done) break; res.write(value); }
    }
    res.end();
    await Promise.allSettled(pending);
  });
  return new Promise(r => srv.listen(port, '127.0.0.1', () => r(srv)));
}

/* ---------- static site ---------- */
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml', '.md': 'text/markdown' };
export function startStatic(port, root) {
  const srv = http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (p.endsWith('/')) p += 'index.html';
    const f = path.join(root, p.replace(/^\/+/, ''));
    if (!f.startsWith(path.resolve(root)) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end('nf'); }
    res.writeHead(200, { 'content-type': TYPES[path.extname(f)] || 'application/octet-stream' });
    fs.createReadStream(f).pipe(res);
  });
  return new Promise(r => srv.listen(port, '127.0.0.1', () => r(srv)));
}
