"""Apply a connection-audit file (tools/review_*.json) to atlas.json and learning/nodes.json.

Safe to run more than once. A change applies only while the field still holds the text
the reviewers saw; anything edited since is reported as STALE and left alone.
New nodes get a deterministic free position near their cluster's centre.
Run tools/sync_counts.py afterwards.

usage: python3 tools/apply_review.py tools/review_2026-10-11.json
"""
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
review = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
atlas = json.loads((ROOT / "atlas.json").read_text(encoding="utf-8"))
nodes = {n["id"]: n for n in atlas["nodes"]}
edges = {e["id"]: e for e in atlas["edges"]}
stale = []

for d in review["edge_deletes"]:
    edges.pop(d["id"], None)

for c in review["edge_changes"]:
    e = edges.get(c["id"])
    if e is None:
        stale.append(f"{c['id']}: edge no longer exists")
    elif e["text"] == c["expect_text"]:
        e.update(c["set"])
    elif any(e.get(k) != v for k, v in c["set"].items()):
        stale.append(f"{c['id']}: text changed since review")

for c in review["node_changes"]:
    n = nodes.get(c["id"])
    if n is None:
        stale.append(f"{c['id']}: node no longer exists")
    elif n[c["field"]] == c["expect"]:
        n[c["field"]] = c["value"]
    elif n[c["field"]] != c["value"]:
        stale.append(f"{c['id']}.{c['field']}: text changed since review")


def free_spot(center, taken, r_min=36, r_max=70, gap=22):
    """First point on a golden-angle spiral of shells around center that is >= gap from every node."""
    golden = math.pi * (3 - math.sqrt(5))
    for r in range(r_min, r_max + 1, 4):
        for i in range(64):
            z = 1 - 2 * (i + 0.5) / 64
            rho, th = math.sqrt(1 - z * z), golden * i
            p = [round(center[0] + r * rho * math.cos(th), 1),
                 round(center[1] + r * z, 1),
                 round(center[2] + r * rho * math.sin(th), 1)]
            if all(math.dist(p, q) >= gap for q in taken):
                return p
    raise SystemExit(f"no free position near {center}")


centers = {c["name"]: c["center"] for c in atlas["clusters"]}
for n in review["new_nodes"]:
    if n["id"] in nodes:
        n = {**n, "p": nodes[n["id"]]["p"]}
    else:
        n = {**n, "p": free_spot(centers[n["group"]], [m["p"] for m in nodes.values()])}
    nodes[n["id"]] = n

for e in review["new_edges"]:
    if e["a"] in nodes and e["b"] in nodes:
        edges[e["id"]] = e
    else:
        stale.append(f"{e['id']}: endpoint missing ({e['a']}, {e['b']})")

atlas["nodes"], atlas["edges"] = list(nodes.values()), list(edges.values())
(ROOT / "atlas.json").write_text(json.dumps(atlas, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

meta_path = ROOT / "learning" / "nodes.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
for k, v in review["node_meta"].items():
    meta["nodes"].setdefault(k, v)
meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

print(f"applied: {len(atlas['nodes'])} topics, {len(atlas['edges'])} edges")
for s in stale:
    print("STALE", s)
