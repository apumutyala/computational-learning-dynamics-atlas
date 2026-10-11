# Handoff from the laptop session (2026-10-11)

## Deployment cluster fact-check
Applied by `tools/apply_factcheck_deploy.py` (idempotent; prints STALE if an edited field no longer matches).
- MCP nodes and edges now cite the 2026-07-28 revision (stateless requests, multi-round-trip requests, Roots/Sampling deprecated).
- EU AI Act dates follow the AI Omnibus (Reg. (EU) 2026/1744): Annex III from 2 Dec 2027, Annex I products from 2 Aug 2028.
- Wording fixes: Ganguli et al. model types; Gorilla citation in `toolcall`; FDA CDS four criteria and HIPAA scope in `clinical`; Chow's-rule sentence in rel-a37208e5 (stays `exact`); METR o3 caveat in `specgaming`; `structdecode` guarantee; rel-59643329; `legalhalluc` comparison; rel-638a5a9c; SR 21-8 in `modelrisk`; ISO 21448 scope; year orders; SEC order URL.
- The three `exact` edges (selective–conformal, selective–proper, safeinterface–barrier) were re-derived and hold.

## FDE track (owned here)
Applied by `tools/apply_fde_track.py` (idempotent) plus `learning/questions-6.json`.
Menu order (learn.js `TRACK_ORDER`):
1. `fde-harness-to-thesis`: mcp → agentsec → agentworm → multillm → agentbench → guardrail → defer → offswitch → alignmentthesis (all real vias)
2. `fde-agent-reliability`: toolcall → structdecode → agentbench → multillm → llmjudge → semanticentropy → selective → defer → guardrail
3. `fde-verticals`: tradingctl → modelrisk → aigovern → legalhalluc → clinical → defer → selective → assistive
4. `fde-foundations`: proper → conformal → onlinecal → guardrail → constrainedrl → barrier → safeinterface → rewardhack → specgaming
Background goal option `goal-fde` → `fde-agent-reliability`. Archetype wording only; no individual is named.
Not on any FDE path yet: promptinject, cyberagent, redteam, robotstd, sycophancy, auditlog, conformallm, mapeval (quiz items exist).
