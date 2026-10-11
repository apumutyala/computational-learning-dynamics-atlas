"""Remove the three project nodes whose only sources are private repositories.

Visitors cannot open those links, so the nodes, every edge touching them, their
learning metadata and quiz items, and doc passages that rely only on them are
removed. Safe to run more than once. Run tools/sync_counts.py afterwards.
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
GONE = {"typedcontrol", "stateproject", "structuredproject"}
PRIVATE = re.compile(r"github\.com/apumutyala/(chl-typed-control|State-Space-Decision-Model|chl-structured-models)")


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save_indented(path, data):
    (ROOT / path).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# atlas.json (minified, no trailing newline)
atlas = load("atlas.json")
atlas["nodes"] = [n for n in atlas["nodes"] if n["id"] not in GONE]
atlas["edges"] = [e for e in atlas["edges"] if e["a"] not in GONE and e["b"] not in GONE]
leftover = [e["id"] for e in atlas["edges"] if PRIVATE.search(json.dumps(e))]
leftover += [n["id"] for n in atlas["nodes"] if PRIVATE.search(json.dumps(n))]
assert not leftover, f"private links still cited by: {leftover}"
(ROOT / "atlas.json").write_text(json.dumps(atlas, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

# learning/nodes.json
meta = load("learning/nodes.json")
for k in GONE:
    meta["nodes"].pop(k, None)
for v in meta["nodes"].values():
    v["prereqs"] = [p for p in v["prereqs"] if p not in GONE]
save_indented("learning/nodes.json", meta)

# question banks and paths
for f in sorted((ROOT / "learning").glob("questions-*.json")):
    bank = json.loads(f.read_text(encoding="utf-8"))
    kept = [q for q in bank["items"] if q["node"] not in GONE and not PRIVATE.search(json.dumps(q))]
    if len(kept) != len(bank["items"]):
        bank["items"] = kept
        save_indented(f.relative_to(ROOT), bank)
paths = load("learning/paths.json")
for p in paths["paths"]:
    assert not any(s["node"] in GONE for s in p["steps"]), f"path {p['id']} uses a removed node"

# docs: drop passages whose only evidence is a private link
EDITS = {
    "docs/alignment-synthesis-thesis.pplx.md": [
        (re.compile(r", while the recorded typed-control experiment tests actual reweighting responses and finds gradient alignment useful in its tested setup \(\[Patterning\]\((https://arxiv\.org/html/2601\.13548v1)\), \[Item 4\]\([^)]*\)\)\."),
         r" ([Patterning](\1))."),
        (re.compile(r"The existing typed-control result establishes[^\n]*\n\nThe next step is to connect those tracks through a shared task and evaluation contract\. "),
         ""),
    ],
    "docs/learning-geometry-control-bridges.pplx.md": [
        (re.compile(r", structured recurrent models, and our typed-control experiments\."), ", and structured recurrent models."),
    ],
    "docs/evidence-ledger.pplx.md": [
        (re.compile(r"^\| \[(Typed-control Item 4|State-Space-Decision-Model PR 215|Structured-models project brief)\][^\n]*\n", re.M), ""),
    ],
    "docs/structure-informed-capability-learning.pplx.md": [
        (re.compile(r"## Connection to our existing work\n.*?(I recommend using HiPPO/LRU as explicit structural comparison points, rather than treating any attention-free network as inherently interpretable or controllable\.)[^\n]*\n\n[^\n]*\n\n", re.S),
         r"## Structural comparison points\n\n\1\n\n"),
        (re.compile(r"This recommendation follows from the combination of published structural results and the current typed-control baseline, rather than presuming a universal theory of intelligence or alignment \(\[LRU\]\(([^)]*)\), \[typed-control report\]\([^)]*\)\)\."),
         r"This recommendation follows from published structural results, rather than presuming a universal theory of intelligence or alignment ([LRU](\1))."),
    ],
}
for path, subs in EDITS.items():
    f = ROOT / path
    text = f.read_text(encoding="utf-8")
    for pattern, repl in subs:
        text = pattern.sub(repl, text)
    assert not PRIVATE.search(text), f"{path} still links a private repo"
    f.write_text(text, encoding="utf-8")

print(f"atlas: {len(atlas['nodes'])} topics, {len(atlas['edges'])} edges")
