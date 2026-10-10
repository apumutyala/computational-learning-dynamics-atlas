# Atlas AI backend

A small Cloudflare Worker that brings back the two AI features on the GitHub Pages site:

- **Connection summaries**: `POST /api/explain {edge, from}` explains why a recorded connection holds.
- **Ask the atlas**: `POST /api/chat {messages, current, path}` answers questions grounded in the topic being viewed, its neighbours and the reader's trail, and cites topics as `[[topic-id]]` (the site turns these into links).

Both stream Server-Sent Events in the format `app.js` already reads: `data: {"t":"..."}`, `data: {"done":true}`, `data: {"error":"..."}`.

Nothing here is deployed. Turning it on needs a Cloudflare account, an Anthropic API key and a spending decision, all of which belong to the site owner.

## How it works

- The worker downloads `atlas.json` from the Pages site (`ATLAS_URL`) and keeps it for 10 minutes per isolate. Atlas edits go live in the AI answers within that window, with no redeploy.
- Prompts are built on the server from ids. The browser sends only an edge id, topic ids and the chat history. Topic or connection text sent by the browser is ignored, and unknown ids are dropped. The evidence label in a connection summary comes from the atlas record, and the prompt tells the model not to upgrade it.
- The worker calls the Anthropic Messages API (`POST /v1/messages`, `stream: true`, `anthropic-version: 2023-06-01`) with plain `fetch`, with no SDK dependency, so it deploys as a single file. It turns `text_delta` events into `{"t"}` and skips thinking blocks. A refusal, an overload, a mid-stream `error` event or a dropped connection becomes one `{"error"}` event.
- Connection summaries are cached per (model, edge, direction, edge text). A repeat costs nothing. Editing an edge's text invalidates its cached summary.

## Guardrails

| Guard | Default | Setting |
|---|---|---|
| Allowed origins (CORS) | `https://apumutyala.github.io`, plus `http://localhost:*` / `http://127.0.0.1:*` | `ALLOWED_ORIGINS`, `ALLOW_LOCALHOST` |
| Request body cap | 32 KB (413 above it, with or without Content-Length) | `MAX_BODY_BYTES` |
| Chat history sent to the model | last 12 messages; user turns capped at 2,000 chars, assistant turns at 4,000 | `MAX_MESSAGES`, `MAX_USER_CHARS`, `MAX_ASSISTANT_CHARS` |
| Output cap | 600 tokens per chat answer, 350 per summary | `CHAT_MAX_TOKENS`, `EXPLAIN_MAX_TOKENS` |
| Per-IP rate limit | token bucket: burst 10, refill 6 per minute (429 + `Retry-After`) | `RATE_BURST`, `RATE_PER_MINUTE` |
| Global daily cap | 2,000 model calls per UTC day (429 after that). Cached summaries are still served | `DAILY_REQUEST_CAP` (`0` = AI off) |

Caveats:

- **The rate limit is in memory.** Each Worker isolate keeps its own buckets, and Cloudflare runs many isolates across many locations, so a determined client spread across isolates can get more than the nominal rate. The bound you can rely on is the daily cap plus the Anthropic spend limit (below). For stricter per-IP limits, add a Cloudflare rate-limiting rule or the Workers Rate Limiting binding (check Cloudflare's current docs for the syntax).
- **The daily cap is per isolate unless KV is bound.** Without KV it bounds spend per isolate only. With the `ATLAS_KV` binding it is shared across isolates. KV is eventually consistent, so under bursts it can overshoot by a few requests. KV writes are billed: the free plan includes about 1,000 writes a day, and this worker writes once per model call. Above roughly 900 calls a day, use the Workers paid plan or leave KV unbound. Check Cloudflare's current pricing.
- **CORS is enforced by browsers.** `curl` can set any `Origin` header. Requests without an allowed Origin are rejected, which stops casual scripts, but the real spend bound is the cap.

## Cost estimate (default model `claude-haiku-5-5`)

Pricing from the Claude API reference (cached 2026-10-06): Claude Haiku 5.5 is **$0.10 per million input tokens and $0.50 per million output tokens** for prompts up to 100K tokens. Every prompt here is far below that. Claude Sonnet 5.5 is $2 / $10.

Prompt sizes were measured with the worker's own prompt builders over every topic and edge in the current `atlas.json`. Tokens are estimated at 4 characters per token plus 30% for the newer tokenizer. Confirm with `count_tokens` once a key exists.

| Request | Characters | ≈ Input tokens |
|---|---|---|
| Chat system prompt, median topic | 9,618 | 3,100 |
| Chat system prompt, largest topic | 13,509 | 4,400 |
| Chat at the 12-message limit (6 max-length questions plus 5 full answers plus the largest system prompt) | ≈ 37,500 | 12,200 |
| Connection summary prompt (median) | 3,030 | 1,000 |

Arithmetic. Output is priced at the full cap, and thinking tokens count toward that cap, so these are upper bounds on output.

- **One typical chat request:** 3,200 in × $0.10/M = $0.00032, plus 600 out × $0.50/M = $0.00030, for **$0.00062**. So **$0.62 per 1,000 requests**.
- **One worst-case chat request** (12-message history at the size caps): 12,200 × $0.10/M = $0.00122, plus $0.00030, for **$0.00152**. So **$1.52 per 1,000 requests**.
- **One 5-question chat** (average about 4,500 input tokens per request as the history grows): 5 × (4,500 × $0.10/M + 600 × $0.50/M) = 5 × $0.00075 = **$0.00375**. So **about $3.75 per 1,000 chats**.
- **Connection summary:** 1,000 × $0.10/M + 350 × $0.50/M = $0.000275, so $0.28 per 1,000 uncached. There are at most 574 distinct (edge, direction) summaries, so filling the whole cache once costs about **$0.16**. After that, repeats are free while the cache holds.
- **Hard ceiling from the daily cap:** 2,000 calls × $0.00152 = **$3.04 per day** (about $91 per 30 days) if every call hit every cap. Realistic traffic costs far less.
- **With `MODEL = "claude-sonnet-5-5"`:** 20× the per-token price: about $12.40 per 1,000 typical requests and $30.40 per 1,000 worst-case requests. Lower `DAILY_REQUEST_CAP` when switching (for example 200, which caps spend at $6.08 a day).

Cloudflare: the Workers free plan covers 100,000 requests a day. Check their current limits.

## Turning it on (owner steps)

Prerequisites: Node 18 or later, a Cloudflare account, an Anthropic API key from the Claude Console.

1. **Set a hard spend limit at the source.** In the Claude Console, set a monthly spend limit on the workspace that owns the key. Use a dedicated key for this site. This is the backstop behind every other guardrail.
2. Log in and deploy from this folder:
   ```sh
   cd backend
   npx wrangler login
   npx wrangler secret put ANTHROPIC_API_KEY     # paste the key when prompted
   npx wrangler deploy                            # prints https://atlas-ai.<your-subdomain>.workers.dev
   ```
   Optional shared counters and cache: `npx wrangler kv namespace create ATLAS_KV`, paste the id into `wrangler.toml` (the commented block), then `npx wrangler deploy` again.
3. Smoke test: `curl https://atlas-ai.<sub>.workers.dev/health` should return `{"ok":true,"model":"claude-haiku-5-5","configured":true}`.
4. Point the site at it. In `app.js`, change one line:
   ```js
   const AI_API = 'https://atlas-ai.<your-subdomain>.workers.dev';
   ```
   Bump the cache-buster in `index.html` (`app.js?v=...`), commit and push to the Pages branch. The Ask button and the connection summaries appear. With `AI_API = ''`, all AI UI stays hidden and the page makes no AI calls.
5. Watch it: `npx wrangler tail` shows a `{"usage":{"input":N,"output":M}}` line per answer. The Claude Console shows spend.

## Switching it off

Use any one of these, from fastest to most thorough:

- **Instant, no site change:** set `DAILY_REQUEST_CAP = "0"` in `wrangler.toml` and run `npx wrangler deploy`. Every request gets a 429, and the site shows "unavailable" messages. Alternatively, `npx wrangler secret delete ANTHROPIC_API_KEY` makes every request return 503.
- **Hide the UI:** set `const AI_API = '';` in `app.js` and push. The site returns to static-only behaviour.
- **Remove it:** `npx wrangler delete` (and revoke the API key in the Claude Console).

## Local testing

No network, no key:

```sh
node backend/test/worker.test.mjs
```

This starts a mock Anthropic endpoint that streams canned SSE, including a thinking block, split TCP writes, a refusal, an overload, a mid-stream error and a dropped connection. It runs the worker's `fetch` handler in Node through a small http adapter (`backend/test/harness.mjs`) and checks:

- the SSE format, CORS headers and grounding
- that client text is ignored
- message trimming and the size caps
- the rate limit, the daily cap (in memory and through KV) and the explain cache
- the env-driven model and limits

`npx wrangler dev` also works for a local run against the real API once a key exists, but it spends money.
