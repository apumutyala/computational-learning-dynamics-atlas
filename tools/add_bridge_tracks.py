#!/usr/bin/env python3
"""Add (or refresh) the bridge learning tracks and the plain-language track text. Idempotent.

Usage (from the repo root):
    python3 tools/add_bridge_tracks.py [--root DIR] [--no-sync]

Content lives in tools/bridge_tracks_data.json. The script:
  * upserts new topic nodes into existing clusters (a node without "p" is placed next to the nodes it
    is joined to, at least MIN_SEP from every other node; its position is then kept on re-runs);
  * upserts edges with deterministic ids "rel-" + sha1("a|b|relation")[:8]; an edge is skipped if a
    different edge already joins the same two nodes. Edges it does not own are never removed;
  * upserts learning/nodes.json entries for its nodes;
  * upserts its own paths (ids "bridge-") in learning/paths.json, resolving "via": "auto";
  * fills a path step's null "via" when an atlas edge now joins it to the previous step;
  * overlays plain-language text on other tracks' paths (summary, audience, step objective/why, keyed
    by path id and step node) and on edge texts (keyed by edge id). Missing targets are reported as
    STALE and skipped, so a later change elsewhere never breaks the run;
  * writes learning/questions-11.json (ids "q-br-") and lists it in learning/index.json;
  * adds or replaces goal options in learning/background.json (line edit, layout kept);
  * updates README.md path count and question-bank totals;
  * finally runs tools/sync_counts.py.
Running it twice gives byte-identical output. Python 3 stdlib only.

Apply order on the repo: remove_private_projects, apply_review, add_mechinterp, add_thesis_track,
apply_path_text, (FDE scripts), add_inference_track, add_intro_safety, add_bridge_tracks, rebrand,
sync_counts. It runs after add_intro_safety because that script rewrites its own paths wholesale.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import OrderedDict

QFILE = "questions-11.json"
QPREFIX = "q-br-"
PATH_PREFIX = "bridge-"
MIN_SEP = 22.0
TEXT_FIELDS = ("summary", "audience", "title")
STEP_FIELDS = ("objective", "why")


def edge_id(a, b, relation):
    return "rel-" + hashlib.sha1(("%s|%s|%s" % (a, b, relation)).encode("utf-8")).hexdigest()[:8]


def read(path):
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8")


def write_if_changed(path, text, changed):
    data = text.encode("utf-8")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            if fh.read() == data:
                return
    with open(path, "wb") as fh:
        fh.write(data)
    changed.append(path)


def dump_like(obj, raw):
    nl = "\r\n" if "\r\n" in raw else "\n"
    m = re.search(r"\n([ \t]+)\S", raw)
    indent = m.group(1) if m else " "
    if indent.strip(" ") == "":
        indent = len(indent)
    out = json.dumps(obj, indent=indent, ensure_ascii=False)
    if nl != "\n":
        out = out.replace("\n", nl)
    return out + (nl if raw.endswith(("\n", "\r\n")) else "")


def load_ordered(raw):
    return json.loads(raw, object_pairs_hook=OrderedDict)


def place(nid, anchors, pos):
    """Deterministic free spot near the anchors' centroid: walk a Fibonacci sphere of growing radius."""
    c = [sum(pos[a][k] for a in anchors) / len(anchors) for k in range(3)]
    golden = math.pi * (3 - math.sqrt(5))
    for ring in range(1, 40):
        r = MIN_SEP * 0.75 * ring
        for i in range(24):
            y = 1 - 2 * (i + 0.5) / 24
            rad = math.sqrt(1 - y * y)
            th = golden * i + int(hashlib.sha1(nid.encode()).hexdigest()[:4], 16)
            q = [round(c[0] + r * rad * math.cos(th), 1), round(c[1] + r * y, 1), round(c[2] + r * rad * math.sin(th), 1)]
            if all(math.dist(q, p) >= MIN_SEP for o, p in pos.items() if o != nid):
                return q
    raise SystemExit("no free position for %s" % nid)


# ---------------------------------------------------------------- atlas.json
def apply_atlas(root, data, report, changed):
    path = os.path.join(root, "atlas.json")
    atlas = load_ordered(read(path))
    nodes, edges = atlas["nodes"], atlas["edges"]
    groups = {c["name"] for c in atlas["clusters"]}

    idx = {n["id"]: i for i, n in enumerate(nodes)}
    mine = [e["node"]["id"] for e in data["nodes"]]
    for entry in data["nodes"]:
        node = OrderedDict(entry["node"])
        if node["group"] not in groups:
            raise SystemExit("node %s: unknown cluster %r" % (node["id"], node["group"]))
        if node["id"] in idx:
            old = nodes[idx[node["id"]]]
            if "p" not in node and "p" in old:
                node["p"] = old["p"]
            nodes[idx[node["id"]]] = node
        else:
            idx[node["id"]] = len(nodes)
            nodes.append(node)
    ids = set(idx)

    by_id = {e["id"]: i for i, e in enumerate(edges)}
    pairs = {}
    for e in edges:
        pairs.setdefault(frozenset((e["a"], e["b"])), set()).add(e["id"])
    added = 0
    for s in data["edges"]:
        a, b = s["a"], s["b"]
        if a not in ids or b not in ids:
            raise SystemExit("edge %s-%s refers to a missing node" % (a, b))
        eid = edge_id(a, b, s["relation"])
        edge = OrderedDict([("id", eid), ("a", a), ("b", b), ("type", s["type"]),
                            ("relation", s["relation"]), ("text", s["text"]), ("url", s["url"])])
        if eid in by_id:
            edges[by_id[eid]] = edge
        else:
            others = pairs.get(frozenset((a, b)), set()) - {eid}
            if others:
                report.append("edge %s-%s skipped: pair already joined by %s" % (a, b, ", ".join(sorted(others))))
                continue
            by_id[eid] = len(edges)
            edges.append(edge)
            pairs.setdefault(frozenset((a, b)), set()).add(eid)
        added += 1

    # Place new nodes next to what they connect to (same-cluster neighbours first).
    pos = {n["id"]: n["p"] for n in nodes if "p" in n}
    byn = {n["id"]: n for n in nodes}
    for nid in mine:
        n = byn[nid]
        if "p" in n:
            continue
        nb = [e["b"] if e["a"] == nid else e["a"] for e in edges if nid in (e["a"], e["b"])]
        nb = [x for x in nb if x in pos]
        same = [x for x in nb if byn[x]["group"] == n["group"]]
        anchors = same or nb or [x for x in pos if byn[x]["group"] == n["group"]]
        n["p"] = place(nid, anchors, pos)
        pos[nid] = n["p"]
        report.append("placed %s near %s" % (nid, ", ".join(anchors[:4])))

    stale = 0
    for eid, text in data.get("edge_text", {}).items():
        if eid in by_id:
            edges[by_id[eid]]["text"] = text
        else:
            report.append("STALE edge_text %s: edge missing" % eid); stale += 1

    counts = {}
    for n in nodes:
        counts[n["group"]] = counts.get(n["group"], 0) + 1
    for c in atlas["clusters"]:
        c["count"] = counts.get(c["name"], 0)
    write_if_changed(path, json.dumps(atlas, ensure_ascii=False, separators=(",", ":")), changed)
    report.append("bridge edges present: %d; edge texts overlaid: %d (stale %d)" % (
        added, len(data.get("edge_text", {})) - stale, stale))
    return atlas


def edge_between(atlas, a, b, prefer):
    hits = sorted(e["id"] for e in atlas["edges"] if {e["a"], e["b"]} == {a, b})
    pref = [h for h in hits if h in prefer]
    return (pref or hits or [None])[0]


def set_option_line(raw, bg):
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)
    line_txt = json.dumps(bg["option"], ensure_ascii=False)
    oid = '"id": "%s"' % bg["option"]["id"]
    cur = [i for i, l in enumerate(lines) if oid in l]
    if cur:
        i = cur[0]
        lead = re.match(r"\s*", lines[i]).group(0)
        lines[i] = lead + line_txt + ("," if lines[i].rstrip().endswith(",") else "")
    else:
        anchor = [i for i, l in enumerate(lines) if '"id": "%s"' % bg["after"] in l]
        if not anchor:
            raise SystemExit("background.json: anchor option %r not found" % bg["after"])
        i = anchor[0]
        lead = re.match(r"\s*", lines[i]).group(0)
        has_comma = lines[i].rstrip().endswith(",")
        if not has_comma:
            lines[i] = lines[i].rstrip() + ","
        lines.insert(i + 1, lead + line_txt + ("," if has_comma else ""))
    out = nl.join(lines)
    q = [x for x in json.loads(out)["questions"] if x["id"] == bg["question"]]
    if not q or not any(o["id"] == bg["option"]["id"] for o in q[0]["options"]):
        raise SystemExit("background.json: option landed outside question %r" % bg["question"])
    return out


# ---------------------------------------------------------------- learning/
def apply_learning(root, data, atlas, report, changed):
    ldir = os.path.join(root, "learning")

    path = os.path.join(ldir, "nodes.json")
    raw = read(path)
    doc = load_ordered(raw)
    for entry in data["nodes"]:
        doc["nodes"][entry["node"]["id"]] = OrderedDict(entry["learning"])
    write_if_changed(path, dump_like(doc, raw), changed)

    mine = {edge_id(e["a"], e["b"], e["relation"]) for e in data["edges"]}
    path = os.path.join(ldir, "paths.json")
    raw = read(path)
    doc = load_ordered(raw)
    for spec in data["paths"]:
        assert spec["id"].startswith(PATH_PREFIX)
        p = json.loads(json.dumps(spec), object_pairs_hook=OrderedDict)
        prev = None
        for s in p["steps"]:
            if s.get("via") == "auto":
                s["via"] = edge_between(atlas, prev, s["node"], mine) if prev else None
                if s["via"] is None and prev:
                    report.append("WARNING: path %s: no edge %s-%s" % (p["id"], prev, s["node"]))
            prev = s["node"]
        report.append("path %s via: %s" % (p["id"], " ".join(str(s["via"]) for s in p["steps"])))
        ids = [x["id"] for x in doc["paths"]]
        if p["id"] in ids:
            doc["paths"][ids.index(p["id"])] = p
        else:
            doc["paths"].append(p)

    # Plain-language overlays on other tracks' paths.
    by_path = {p["id"]: p for p in doc["paths"]}
    n_text = 0
    for ov in data.get("path_text", []):
        p = by_path.get(ov["path"])
        if p is None:
            report.append("STALE path_text %s: path missing" % ov["path"]); continue
        for f in TEXT_FIELDS:
            if f in ov:
                p[f] = ov[f]
        steps = {s["node"]: s for s in p["steps"]}
        for node, fields in ov.get("steps", {}).items():
            if node not in steps:
                report.append("STALE path_text %s/%s: step missing" % (ov["path"], node)); continue
            for f in STEP_FIELDS:
                if f in fields:
                    steps[node][f] = fields[f]
        n_text += 1
    report.append("path texts overlaid: %d" % n_text)

    # Any step left without a connecting edge gets one if an atlas edge now joins it to the previous step.
    filled = 0
    for p in doc["paths"]:
        for prev, s in zip(p["steps"], p["steps"][1:]):
            if s.get("via") is None:
                s["via"] = edge_between(atlas, prev["node"], s["node"], mine)
                filled += s["via"] is not None
    report.append("null via steps filled: %d" % filled)
    write_if_changed(path, dump_like(doc, raw), changed)
    n_paths = len(doc["paths"])

    qstyle = read(os.path.join(ldir, "questions-10.json"))
    items = [OrderedDict(i) for i in data["questions"]]
    assert all(i["id"].startswith(QPREFIX) for i in items)
    write_if_changed(os.path.join(ldir, QFILE), dump_like(OrderedDict([("version", 1), ("items", items)]), qstyle), changed)

    path = os.path.join(ldir, "index.json")
    raw = read(path)
    doc = load_ordered(raw)
    if QFILE not in doc["questions"]:
        doc["questions"].append(QFILE)
    write_if_changed(path, dump_like(doc, raw), changed)
    banks = list(doc["questions"])

    path = os.path.join(ldir, "background.json")
    out = read(path)
    for bg in data.get("background_options", []):
        out = set_option_line(out, bg)
    write_if_changed(path, out, changed)
    return n_paths, banks


# ---------------------------------------------------------------- README.md
def apply_readme(root, n_paths, banks, report, changed):
    path = os.path.join(root, "README.md")
    t = read(path)
    t, n = re.subn(r"There are \d+ curated paths", "There are %d curated paths" % n_paths, t)
    if n == 0:
        report.append("WARNING: README path count not found")
    sizes = []
    for b in banks:
        with open(os.path.join(root, "learning", b), encoding="utf-8") as fh:
            sizes.append(len(json.load(fh)["items"]))
    t, n = re.subn(r"The current banks hold \d+ items \([\d + ]+\)\.",
                   "The current banks hold %d items (%s)." % (sum(sizes), " + ".join(map(str, sizes))), t)
    if n == 0:
        report.append("WARNING: README bank totals not found")
    write_if_changed(path, t, changed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--no-sync", action="store_true", help="skip running tools/sync_counts.py at the end")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    with open(os.path.join(root, "tools", "bridge_tracks_data.json"), encoding="utf-8") as fh:
        data = json.load(fh, object_pairs_hook=OrderedDict)
    report, changed = [], []
    atlas = apply_atlas(root, data, report, changed)
    n_paths, banks = apply_learning(root, data, atlas, report, changed)
    apply_readme(root, n_paths, banks, report, changed)
    for line in report:
        print(line)
    print("changed: %s" % (", ".join(os.path.relpath(c, root) for c in changed) if changed else "nothing (already applied)"))
    if not args.no_sync:
        subprocess.run([sys.executable, os.path.join(root, "tools", "sync_counts.py")], check=True)


if __name__ == "__main__":
    sys.exit(main())
