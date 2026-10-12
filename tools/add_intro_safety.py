#!/usr/bin/env python3
"""Add (or refresh) the "AI safety foundations" cluster and the intro-to-AI-safety learning track. Idempotent.

Usage (from the repo root):
    python3 tools/add_intro_safety.py [--root DIR] [--no-sync]

The track is built around Stanford CS120, Introduction to AI Safety (Fall 2025,
https://web.stanford.edu/class/cs120/): one node for the course itself, topic nodes for course topics the
atlas did not yet cover, and paths that follow the course and then cross into the alignment thesis.
Content lives in tools/intro_safety_data.json; the study guide is docs/intro-ai-safety-cs120.md. The script:
  * upserts the cluster (name, color, center) and its nodes in atlas.json by id
    (minified, no trailing newline);
  * upserts its edges with deterministic ids "rel-" + sha1("a|b|relation")[:8]. An edge whose id
    exists is replaced; an edge is skipped if a different edge already joins the same two nodes.
    Edges it does not own are never removed;
  * upserts learning/nodes.json entries for its nodes;
  * upserts its paths in learning/paths.json, resolving "via": "auto" to the atlas edge between
    consecutive steps (preferring this track's own edges; null if none);
  * writes learning/questions-10.json (ids "q-ais-") and lists it in learning/index.json;
  * adds or replaces one goal option in learning/background.json (line edit, layout kept);
  * updates README.md: the cluster list sentence, the curated-path count, the question-bank totals and
    the docs/ list;
  * finally runs tools/sync_counts.py (cluster counts, topic/edge totals, atlas cache-bust hash).
Running it twice gives byte-identical output. Python 3 stdlib only.

Apply order on the repo: remove_private_projects, apply_review, add_mechinterp, add_thesis_track,
apply_path_text, (FDE scripts), add_inference_track, add_intro_safety, rebrand, sync_counts.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict

QFILE = "questions-10.json"
QPREFIX = "q-ais-"
PATH_PREFIX = "intro-ai-safety-"
MIN_SEP = 22.0


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


def text_style(raw):
    nl = "\r\n" if "\r\n" in raw else "\n"
    m = re.search(r"\n([ \t]+)\S", raw)
    indent = m.group(1) if m else " "
    if indent.strip(" ") == "":
        indent = len(indent)
    return nl, indent, raw.endswith(("\n", "\r\n"))


def dump_like(obj, raw):
    nl, indent, trail = text_style(raw)
    out = json.dumps(obj, indent=indent, ensure_ascii=False)
    if nl != "\n":
        out = out.replace("\n", nl)
    return out + (nl if trail else "")


def load_ordered(raw):
    return json.loads(raw, object_pairs_hook=OrderedDict)


# ---------------------------------------------------------------- atlas.json
def apply_atlas(root, data, report, changed):
    path = os.path.join(root, "atlas.json")
    atlas = load_ordered(read(path))
    nodes, edges, clusters = atlas["nodes"], atlas["edges"], atlas["clusters"]

    spec = data["cluster"]
    cl = OrderedDict([("name", spec["name"]), ("color", spec["color"]), ("count", 0), ("center", spec["center"])])
    names = [c["name"] for c in clusters]
    if spec["name"] in names:
        old = clusters[names.index(spec["name"])]
        cl["count"] = old.get("count", 0)
        clusters[names.index(spec["name"])] = cl
    else:
        clusters.append(cl)

    idx = {n["id"]: i for i, n in enumerate(nodes)}
    for entry in data["nodes"]:
        node = OrderedDict(entry["node"])
        if node["group"] != spec["name"]:
            raise SystemExit("node %s is not in the track cluster" % node["id"])
        if node["id"] in idx:
            nodes[idx[node["id"]]] = node
        else:
            idx[node["id"]] = len(nodes)
            nodes.append(node)
    ids = set(idx)

    by_id = {e["id"]: i for i, e in enumerate(edges)}
    pairs = {}
    for e in edges:
        pairs.setdefault(frozenset((e["a"], e["b"])), set()).add(e["id"])
    present = Counter()
    mine = set(n["node"]["id"] for n in data["nodes"])
    bridges = 0
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
        present[s["type"]] += 1
        if not (a in mine and b in mine):
            bridges += 1

    counts = Counter(n["group"] for n in nodes)
    for c in clusters:
        c["count"] = counts.get(c["name"], 0)

    pos = {n["id"]: n["p"] for n in nodes}
    for nid in sorted(mine):
        near = min((math.dist(pos[nid], q), o) for o, q in pos.items() if o != nid)
        if near[0] < MIN_SEP:
            report.append("WARNING: %s is %.1f from %s" % (nid, near[0], near[1]))

    write_if_changed(path, json.dumps(atlas, ensure_ascii=False, separators=(",", ":")), changed)
    report.append("track edges present: %d, %d bridging (%s)" % (
        sum(present.values()), bridges, ", ".join("%s %d" % kv for kv in sorted(present.items()))))
    return atlas


def edge_between(atlas, a, b, prefer):
    hits = sorted(e["id"] for e in atlas["edges"] if {e["a"], e["b"]} == {a, b})
    pref = [h for h in hits if h in prefer]
    return (pref or hits or [None])[0]


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
    write_if_changed(path, dump_like(doc, raw), changed)
    n_paths = len(doc["paths"])

    qstyle = read(os.path.join(ldir, "questions-9.json"))
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

    bg = data["background_option"]
    path = os.path.join(ldir, "background.json")
    raw = read(path)
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
    check = json.loads(out)
    q = [x for x in check["questions"] if x["id"] == bg["question"]]
    if not q or not any(o["id"] == bg["option"]["id"] for o in q[0]["options"]):
        raise SystemExit("background.json: option landed outside question %r" % bg["question"])
    write_if_changed(path, out, changed)
    return n_paths, banks


# ---------------------------------------------------------------- README.md
def apply_readme(root, data, n_paths, banks, report, changed):
    path = os.path.join(root, "README.md")
    t = read(path)
    phrase = data["cluster"]["readme_phrase"]
    if phrase not in t:
        new, n = re.subn(r"; and ([a-z][a-z ,&-]*?)\. Each topic window", r"; \1; and %s. Each topic window" % phrase, t, count=1)
        if n == 0:
            report.append("WARNING: README cluster sentence not found")
        t = new
    doc_line = data["readme_doc"]["line"]
    if doc_line not in t:
        lines = t.split("\n")
        anchor = [i for i, l in enumerate(lines) if l.startswith("- ") and data["readme_doc"]["after"] in l]
        if anchor:
            lines.insert(anchor[0] + 1, doc_line)
            t = "\n".join(lines)
        else:
            report.append("WARNING: README docs list anchor not found")
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
    with open(os.path.join(root, "tools", "intro_safety_data.json"), encoding="utf-8") as fh:
        data = json.load(fh, object_pairs_hook=OrderedDict)
    report, changed = [], []
    atlas = apply_atlas(root, data, report, changed)
    n_paths, banks = apply_learning(root, data, atlas, report, changed)
    apply_readme(root, data, n_paths, banks, report, changed)
    for line in report:
        print(line)
    print("changed: %s" % (", ".join(os.path.relpath(c, root) for c in changed) if changed else "nothing (already applied)"))
    if not args.no_sync:
        subprocess.run([sys.executable, os.path.join(root, "tools", "sync_counts.py")], check=True)


if __name__ == "__main__":
    sys.exit(main())
