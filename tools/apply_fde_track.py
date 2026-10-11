"""Idempotent overlay: flagship forward-deployed research engineer path + harness-lens titles for parts I-III.

Run from the repo root:  python tools/apply_fde_track.py
Keeps learning/paths.json formatting (indent 1, CRLF). Re-running is a no-op.
"""
import json
from pathlib import Path

PATHS = Path(__file__).resolve().parent.parent / 'learning' / 'paths.json'

FLAGSHIP = {
    "id": "fde-harness-to-thesis",
    "title": "From API contracts to the alignment thesis: agent harnesses, sandboxes, evals, monitoring and correction",
    "summary": "Follow an agent through the harness a forward-deployed team builds around it: the tool and API contract it calls through, the permissions and sandbox that bound what it can touch, the agent-to-agent channels attacks travel along, end-state evals over repeated trials, runtime monitoring, human escalation, and finally the correction interface the alignment thesis asks for. Parts I–III go deeper on reliability, verticals and the underlying math.",
    "audience": "Forward-deployed, product and platform engineers who build agent harnesses on top of APIs and tool protocols for customers, often with a security or API-platform background, and want to see how their day-to-day controls connect to evaluation, monitoring and alignment research. No statistics or control background assumed.",
    "level": "intermediate",
    "skills": ["deep-learning", "stats-ml", "probability", "rl"],
    "estHours": 18,
    "steps": [
        {"node": "mcp",
         "objective": "Sketch an agent harness as host, client and tool server; name what the protocol standardizes (tool, resource and prompt discovery and invocation) and what it explicitly leaves to the host (consent, access control, trust in tool descriptions).",
         "why": "Home ground: the API and tool contract. Everything an agent does in a customer system passes through this interface, so it is where the harness starts and where the later controls attach.",
         "via": None},
        {"node": "agentsec",
         "objective": "Scope an agent's credentials and tools to one task, run its actions in an isolated sandbox, and explain how separating control flow (from the trusted request) from data (from tools and documents) limits what a compromised step can do.",
         "why": "The atlas edge: the protocol cannot enforce its own security principles, so least privilege and consent must be built by whoever builds the harness. Permissions are the first product decision, before any model behaviour is measured.",
         "via": "rel-43fffe2c"},
        {"node": "agentworm",
         "objective": "Trace how a self-replicating injection moves from one agent's output into another agent's context through shared retrieval stores or messages, and say which harness boundary would stop it.",
         "why": "Untested link: applying control/data-flow separation to messages between agents has only been evaluated for a single agent. Once harnesses talk to each other, another agent's output is untrusted input like any web page.",
         "via": "rel-b5a8d503"},
        {"node": "multillm",
         "objective": "Classify failed multi-agent traces as system-design, inter-agent misalignment or task-verification failures, and run the matched-compute single-agent baseline before adding agents.",
         "why": "The atlas edge: injections propagate between agents even when they share only part of their communication. Multi-agent designs widen that attack surface and often add little accuracy, so they have to earn their place.",
         "via": "rel-43875211"},
        {"node": "agentbench",
         "objective": "Score an agent on the end state of the customer system (database, repository, ticket) rather than its transcript, and compute pass^k from repeated trials to show how reliability falls when every run has to succeed.",
         "why": "The atlas edge: multi-agent systems need end-state checks and single-agent baselines. Evals that check end states over repeated trials are the acceptance test a customer actually cares about.",
         "via": "rel-1ba46b31"},
        {"node": "guardrail",
         "objective": "Place input and output policy checks around the harness, and set up a sequential monitor on live traffic that alarms on a harmful rise in error rate while tolerating benign drift.",
         "why": "The atlas edge: an offline score covers a fixed task distribution, but production traffic shifts. Monitoring carries the eval into deployment.",
         "via": "rel-5d0619be"},
        {"node": "defer",
         "objective": "Design the escalation path: which cases go to a human reviewer, what the reviewer sees, and how to measure accuracy and cost of the agent plus reviewer against either alone.",
         "why": "The atlas edge: monitors exist to trigger intervention. In a product, intervention usually means handing the case, or the whole workflow, to a person.",
         "via": "rel-2d47942b"},
        {"node": "offswitch",
         "objective": "State the off-switch game's result (an agent uncertain about its objective, treating the human's choice as evidence, can prefer to stay interruptible) and the assumptions it needs (a rational human, a correct model).",
         "why": "The atlas edge: the off-switch game explains, in a formal model, why an agent would defer to a human, while learned deferral rules add an empirical model of the human's errors. Escalation in the harness is the engineering form of being correctable.",
         "via": "rel-096cfdd6"},
        {"node": "alignmentthesis",
         "objective": "State what the thesis asks of a correction interface and how its falsifier would show up in a deployed agent: better proxy scores alongside worse, independently measured user outcomes.",
         "why": "Untested link: the off-switch result supplies the incentive part of the thesis's correction interface, under strong assumptions. The thesis asks for the evidence a harness can actually collect: corrections that hold up and user outcomes measured independently of the agent's own metrics.",
         "via": "rel-43663d35"},
    ],
}

RETITLE = {
    "fde-agent-reliability": "Inside the agent harness: schema-checked tool calls, end-state evals, abstention and escalation",
    "fde-verticals": "Shipping agents into regulated verticals: trading controls, model risk, the AI Act, legal, clinical and accessibility",
    "fde-foundations": "The math under the harness: proper scores, conformal guarantees, drift monitoring and safety filters",
}
WHY = {  # (path, step node) -> new why
    ("fde-verticals", "tradingctl"): "Start where an automated actor in a customer's system lost the most money fastest. No learned model was involved, so the lesson is purely about the controls a harness puts around any automated actor: hard limits outside it, a kill switch and a careful release.",
    ("fde-foundations", "proper"): "Every threshold in the harness, whether to answer, escalate or block, reads a confidence score. Proper scoring rules are how you check that those scores mean what they say, so they come first.",
}


def main():
    raw = PATHS.read_text(encoding='utf-8')
    d = json.loads(raw)
    ps = d['paths']
    if not any(p['id'] == FLAGSHIP['id'] for p in ps):
        i = next((k for k, p in enumerate(ps) if p['id'].startswith('fde-')), len(ps))
        ps.insert(i, FLAGSHIP)
    for p in ps:
        if p['id'] in RETITLE:
            p['title'] = RETITLE[p['id']]
        for s in p['steps']:
            if (p['id'], s['node']) in WHY:
                s['why'] = WHY[(p['id'], s['node'])]
    out = json.dumps(d, indent=1, ensure_ascii=False).replace('\n', '\r\n')
    if raw.endswith('\n'):
        out += '\r\n'
    PATHS.write_text(out, encoding='utf-8', newline='')
    print('ok', len(ps), 'paths')


if __name__ == '__main__':
    main()
