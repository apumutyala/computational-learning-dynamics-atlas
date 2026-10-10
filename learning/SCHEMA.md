# Learning layer — data contracts

Everything here is static JSON loaded by the browser. Node ids and edge ids refer to `../atlas.json`.
`tools/validate_learning.py` checks every rule below; run it before committing.

## Shared skill vocabulary (use these exact tags)
`linear-algebra`, `probability`, `optimization`, `calculus-odes`, `stats-ml`, `deep-learning`,
`signals-estimation`, `control`, `dynamical-systems`, `generative-models`, `rl`, `theory`

Mastery levels are numbers in [0, 1] per skill (0 = never seen it, 1 = fluent).

## learning/nodes.json — per-topic learning metadata
```json
{ "version": 1,
  "nodes": { "<nodeId>": { "level": 1, "skills": ["probability"], "prereqs": ["<nodeId>"] } } }
```
- `level`: 1 intro · 2 intermediate · 3 advanced · 4 research frontier.
- `skills`: 1–3 tags the topic exercises. `prereqs`: node ids worth knowing first (may be empty; no cycles).
- Must cover every node in atlas.json.

## learning/paths.json — curated learning paths
```json
{ "version": 1,
  "paths": [ {
    "id": "regression-to-gps", "title": "…", "summary": "one or two sentences",
    "audience": "who it is for", "level": "intro|intermediate|advanced",
    "skills": ["stats-ml", "probability"], "estHours": 6,
    "steps": [ { "node": "<nodeId>", "objective": "what you can do after this step",
                 "why": "why this step follows the previous one", "via": "<edgeId>|null" } ] } ] }
```
- `via` is the atlas edge joining this step to the previous step (null for step 1, or when no direct edge exists).
- 4–9 steps per path. Every `node` exists. A `via` edge must actually connect the two step nodes.

## learning/background.json — onboarding questionnaire (self-report)
```json
{ "version": 1,
  "questions": [ { "id": "bg-math", "prompt": "…", "type": "single|multi",
    "options": [ { "id": "…", "label": "…", "sets": { "<skill>": 0.6 }, "goal": "<pathId>|null" } ] } ] }
```
- `sets` gives prior mastery per skill when chosen (max wins across answers). `goal` nudges path choice.

## learning/index.json — which question banks the site loads
```json
{ "version": 1, "questions": ["questions-1.json", "questions-2.json"] }
```
- Add a new bank's filename here or the site won't load it.

## learning/questions-*.json — diagnostic probe items
```json
{ "version": 1,
  "items": [ { "id": "q-ridge-1", "node": "<nodeId>", "skill": "<skill>", "difficulty": 1,
    "prompt": "…", "choices": ["…", "…", "…", "…"], "answer": 0,
    "explain": "why the answer is right (1–3 sentences)", "source": "https://…|null" } ] }
```
- `difficulty`: 1 recall · 2 apply · 3 reason/connect. 3–4 choices; `answer` is a 0-based index.
- Item ids are globally unique across all question files. Plain text only (no LaTeX); use Unicode (σ, ∇, Aᵀ).
