"""Idempotent overlay: fact-check fixes for the "Deployed agents & high-stakes applications" cluster (2026-10-11).

Each fix replaces a substring in one field of a node or edge. Already-applied fixes are skipped;
fixes whose old text is missing (edited elsewhere) are reported as STALE and left alone.
Run from the repo root:  python tools/apply_factcheck_deploy.py
"""
import json
from pathlib import Path

ATLAS = Path(__file__).resolve().parent.parent / 'atlas.json'
MCP_OLD, MCP_NEW = 'https://modelcontextprotocol.io/specification/2025-06-18', 'https://modelcontextprotocol.io/specification/2026-07-28'
SEC_OLD, SEC_NEW = 'https://www.sec.gov/litigation/admin/2013/34-70694.pdf', 'https://www.sec.gov/files/litigation/admin/2013/34-70694.pdf'
OMNIBUS = 'https://eur-lex.europa.eu/eli/reg/2026/1744/oj'

FIXES = [
    # MCP: current revision is 2026-07-28 (stateless; Roots/Sampling deprecated; multi round-trip requests)
    ('mcp', 'year', '2025 (2025-06-18 revision)', '2024–2026 (2026-07-28 revision)'),
    ('mcp', 'handle', 'Standardizes capability negotiation and tool discovery so one server integration serves many hosts; clients may offer sampling, roots and elicitation back to servers.',
     'Standardizes tool, resource and prompt discovery so one server integration serves many hosts. Since the 2026-07-28 revision requests are stateless (each carries its protocol version and client capabilities), servers ask clients for input through multi-round-trip requests, and Roots and Sampling are deprecated.'),
    *[(i, f, MCP_OLD, MCP_NEW) for i, fs in [('mcp', ['url', 'handle', 'boundary']), ('rel-ee19df55', ['url']), ('rel-43fffe2c', ['url']), ('rel-44435781', ['url'])] for f in fs],
    # EU AI Act timing after the AI Omnibus, Regulation (EU) 2026/1744
    ('aigovern', 'handle', 'NIST organizes practice as Govern',
     f'After the AI Omnibus, high-risk obligations apply from 2 December 2027 (Annex III uses) and 2 August 2028 (Annex I products such as medical devices); general-purpose-model duties already apply ([Regulation (EU) 2026/1744]({OMNIBUS})). NIST organizes practice as Govern'),
    ('rel-37d596fa', 'text', 'as high-risk.', 'as high-risk (obligations apply from 2 December 2027 after the AI Omnibus).'),
    ('rel-c802a40e', 'text', 'over their lifetime.', 'over their lifetime (applicable from 2 December 2027 for Annex III systems after the AI Omnibus).'),
    ('rel-84a93951', 'text', 'or stopping the system.', 'or stopping the system (applicable from 2 December 2027 for Annex III systems after the AI Omnibus).'),
    ('rel-fde179f4', 'text', "AI that is, or is a safety component of, a product under EU medical-device legislation is classed as high-risk through the Act's product-legislation route.",
     "AI that is, or is a safety component of, a medical device that must undergo third-party conformity assessment under the EU device regulations is high-risk through the Act's product-legislation route (obligations apply from 2 August 2028 after the AI Omnibus)."),
    # Ganguli et al.: four model types = plain LM + three interventions
    ('redteam', 'handle', 'across 2.7B–52B models and four safety interventions', 'on 2.7B–52B models of four types: a plain LM and three safety interventions (HHH prompting, rejection sampling, RLHF)'),
    # Gorilla argues retrieval helps adapt to API changes
    ('toolcall', 'boundary', 'and API changes silently invalidate learned usage ([Patil et al.](https://arxiv.org/abs/2305.15334)).',
     'API changes invalidate usage learned in the weights; Gorilla-style retrieval mitigates this only when current documentation is retrieved ([Patil et al.](https://arxiv.org/abs/2305.15334)).'),
    # FDA CDS: all four criteria; HIPAA scope
    ('clinical', 'text', 'can be a regulated device unless a clinician can independently review its basis.',
     'is a regulated device unless it meets all four statutory non-device criteria, one of which is that a clinician can independently review its basis.'),
    ('clinical', 'handle', 'health data falls under the HIPAA Privacy Rule', 'protected health information held by covered entities and their business associates falls under the HIPAA Privacy Rule'),
    # Chow's rule: exact equality with eta is sufficient, not necessary
    ('rel-a37208e5', 'text', 'so thresholding a forecast is optimal only insofar as the forecast equals η.',
     "so a forecast trained with a proper score makes the plug-in Chow rule consistent. Exact equality is not required: the plug-in rule is optimal wherever the forecast's argmax and its side of the 1 − c threshold agree with η's."),
    # METR: single task, early o3
    ('specgaming', 'handle', "On RE-Bench, o3 reward hacked in 39 of 128 runs, for example by monkey-patching an evaluator or overwriting a timer; telling it not to cheat barely changed the rate",
     'On RE-Bench, an early version of o3 reward hacked in 39 of 128 runs, for example by monkey-patching an evaluator or overwriting a timer; on one task, instructions not to cheat left the rate at 70–95%'),
    # constrained decoding: guarantees are per supported grammar subset, and only for completed output
    ('structdecode', 'text', 'so output provably parses under a regular expression, JSON Schema or grammar,',
     'so every emitted prefix stays valid under a regular expression, a supported JSON Schema subset or a grammar, and completed output parses,'),
    ('rel-59643329', 'text', 'to programs whose outputs are exact given a correct call',
     'to programs whose outputs are exact (calculator, calendar) or at least authoritative relative to their source (search, QA) given a correct call'),
    ('legalhalluc', 'boundary', 'General chatbots did worse: 58–88% on verifiable questions about federal cases',
     'In the same audit GPT-4 hallucinated more than the legal tools; on verifiable questions about federal cases, general chatbots hallucinated 58–88%'),
    ('rel-638a5a9c', 'text', 'Proposed, untested: use semantic entropy as the nonconformity score for claim filtering. Validity comes from conformal theory for any score; whether this score retains more claims is the open question.',
     'Proposed: use semantic entropy as the claim-filter score. Validity holds for any score; Mohri and Hashimoto already test a sample-frequency score, so the open question is whether entailment-clustered entropy retains more claims.'),
    ('modelrisk', 'handle', 'across the model lifecycle.', 'across the model lifecycle. It also supersedes SR 21-8 and is aimed mainly at banking organizations with over $30 billion in assets.'),
    ('robotstd', 'handle', 'for automated driving functions', 'for road-vehicle functions that depend on situational awareness, including driver assistance and automated driving'),
    ('robotstd', 'year', '2022 / 2025', '2025 / 2022'),
    ('oversight', 'year', '2018 / 2022', '2022 / 2018'),
    ('safeinterface', 'year', '2021 / 2022', '2022 / 2021'),
    ('agentsec', 'year', '2024 / 2025', '2025'),
    ('tradingctl', 'text', 'millions of market orders', 'millions of child orders'),
    *[(i, f, SEC_OLD, SEC_NEW) for i, fs in [('tradingctl', ['url', 'handle', 'boundary']), ('rel-44934218', ['url']), ('rel-f4ff63ba', ['url'])] for f in fs],
]


def main():
    data = json.loads(ATLAS.read_text(encoding='utf-8'))
    items = {n['id']: n for n in data['nodes']} | {e['id']: e for e in data['edges']}
    applied = skipped = stale = 0
    for id_, field, old, new in FIXES:
        obj = items.get(id_)
        val = obj.get(field) if obj else None
        if not isinstance(val, str):
            print(f'STALE  {id_}.{field}: missing'); stale += 1
        elif new in val:
            skipped += 1
        elif old in val:
            obj[field] = val.replace(old, new); applied += 1
        else:
            print(f'STALE  {id_}.{field}: old text not found'); stale += 1
    ATLAS.write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False), encoding='utf-8', newline='')
    print(f'applied {applied}, already applied {skipped}, stale {stale}')


if __name__ == '__main__':
    main()
