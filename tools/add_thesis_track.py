#!/usr/bin/env python3
"""Add the "Alignment thesis" learning track to the atlas. Idempotent.

Usage (from the repo root):
    python3 tools/add_thesis_track.py [--root DIR]

Reads tools/thesis_track_data.json and:
  * upserts its topic nodes in atlas.json (minified, no trailing newline). Ids
    listed in `shared_ids` (also used by the mechanistic-interpretability
    effort) are left untouched if they already exist, so that effort's `group`,
    `p` and text win; only this track's edges are added to them;
  * adds its edges with deterministic ids "rel-" + sha1("a|b|relation")[:8].
    An edge whose id already exists is replaced. An edge is skipped if another
    edge already joins the same two nodes. Edges marked "optional" (to the
    deployment cluster) are added only when both endpoints exist;
  * recomputes every cluster `count` from node groups;
  * upserts learning/nodes.json entries (shared ids only if missing), keeping
    the file's indentation and line endings;
  * upserts the two paths in learning/paths.json, resolving each "via": "auto"
    to the atlas edge that joins the two steps (null if none);
  * writes learning/questions-8.json and lists it in learning/index.json;
  * adds or replaces one option in learning/background.json (text edit, so the
    one-option-per-line layout is kept);
  * recomputes topic, edge, cluster and path totals and the per-type table in
    README.md, and the totals in index.html.
Running it twice gives byte-identical output. Python 3 stdlib only.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict

QFILE = "questions-8.json"
QPREFIX = "q-th-"
ORIGIN = (0.0, -10.0, 0.0)
MIN_SEP, MIN_ORIGIN = 22.0, 100.0


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
    """Return (newline, indent, trailing_newline) of a pretty-printed JSON file."""
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


def apply_atlas(root, data, report, changed):
    path = os.path.join(root, "atlas.json")
    raw = read(path)
    atlas = load_ordered(raw)
    nodes, edges = atlas["nodes"], atlas["edges"]
    shared = set(data.get("shared_ids", []))
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    for entry in data["nodes"]:
        node = OrderedDict(entry["node"])
        nid = node["id"]
        if nid in idx:
            if nid in shared:
                report.append("node %s exists (shared id): left as is" % nid)
                continue
            nodes[idx[nid]] = node
        else:
            idx[nid] = len(nodes)
            nodes.append(node)
    ids = set(idx)

    by_id = {e["id"]: i for i, e in enumerate(edges)}
    pairs = {}
    for e in edges:
        pairs.setdefault(frozenset((e["a"], e["b"])), set()).add(e["id"])
    added = Counter()
    for spec in data["edges"]:
        a, b = spec["a"], spec["b"]
        if a not in ids or b not in ids:
            if spec.get("optional"):
                report.append("optional edge %s-%s skipped (node missing)" % (a, b))
                continue
            raise SystemExit("edge %s-%s refers to a missing node" % (a, b))
        eid = edge_id(a, b, spec["relation"])
        edge = OrderedDict([("id", eid), ("a", a), ("b", b), ("type", spec["type"]),
                            ("relation", spec["relation"]), ("text", spec["text"]), ("url", spec["url"])])
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
        added[spec["type"]] += 1

    counts = Counter(n["group"] for n in nodes)
    for c in atlas["clusters"]:
        c["count"] = counts.get(c["name"], 0)
    unknown = set(counts) - {c["name"] for c in atlas["clusters"]}
    if unknown:
        report.append("WARNING: groups without a cluster entry: %s" % ", ".join(sorted(unknown)))

    # Spacing check for this track's nodes (warn only; positions are stored data).
    mine = {e["node"]["id"] for e in data["nodes"]}
    pos = {n["id"]: n["p"] for n in nodes}
    for nid in sorted(mine):
        p = pos[nid]
        near = min((math.dist(p, q), o) for o, q in pos.items() if o != nid)
        if near[0] < MIN_SEP or math.dist(p, ORIGIN) < MIN_ORIGIN:
            report.append("WARNING: %s is %.1f from %s and %.1f from the origin" % (nid, near[0], near[1], math.dist(p, ORIGIN)))

    write_if_changed(path, json.dumps(atlas, ensure_ascii=False, separators=(",", ":")), changed)
    report.append("track edges present: %d (%s)" % (sum(added.values()), ", ".join("%s %d" % kv for kv in sorted(added.items()))))
    return atlas


def edge_between(atlas, a, b, prefer):
    hits = sorted(e["id"] for e in atlas["edges"] if {e["a"], e["b"]} == {a, b})
    pref = [h for h in hits if h in prefer]
    return (pref or hits or [None])[0]


def apply_learning(root, data, atlas, report, changed):
    ldir = os.path.join(root, "learning")
    shared = set(data.get("shared_ids", []))

    # nodes.json
    path = os.path.join(ldir, "nodes.json")
    raw = read(path)
    doc = load_ordered(raw)
    for entry in data["nodes"]:
        nid = entry["node"]["id"]
        if nid in shared and nid in doc["nodes"]:
            continue
        doc["nodes"][nid] = OrderedDict(entry["learning"])
    write_if_changed(path, dump_like(doc, raw), changed)

    # paths.json
    mine = {edge_id(e["a"], e["b"], e["relation"]) for e in data["edges"]}
    path = os.path.join(ldir, "paths.json")
    raw = read(path)
    doc = load_ordered(raw)
    for spec in data["paths"]:
        p = json.loads(json.dumps(spec), object_pairs_hook=OrderedDict)
        prev = None
        for s in p["steps"]:
            if s.get("via") == "auto":
                s["via"] = edge_between(atlas, prev, s["node"], mine) if prev else None
                if s["via"] is None:
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

    # questions-8.json (fully owned by this track)
    q5 = read(os.path.join(ldir, "questions-5.json"))
    items = [OrderedDict(i) for i in data["questions"]]
    assert all(i["id"].startswith(QPREFIX) for i in items)
    write_if_changed(os.path.join(ldir, QFILE), dump_like(OrderedDict([("version", 1), ("items", items)]), q5), changed)

    # index.json
    path = os.path.join(ldir, "index.json")
    raw = read(path)
    doc = load_ordered(raw)
    if QFILE not in doc["questions"]:
        doc["questions"].append(QFILE)
    write_if_changed(path, dump_like(doc, raw), changed)

    # background.json: line-level edit to keep its one-option-per-line layout.
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
    doc = json.loads(out)
    q = [x for x in doc["questions"] if x["id"] == bg["question"]]
    if not q or not any(o["id"] == bg["option"]["id"] for o in q[0]["options"]):
        raise SystemExit("background.json: option landed outside question %r" % bg["question"])
    write_if_changed(path, out, changed)
    return n_paths


def apply_counts(root, atlas, n_paths, report, changed):
    n_nodes, n_edges, n_clusters = len(atlas["nodes"]), len(atlas["edges"]), len(atlas["clusters"])
    types = Counter(e["type"] for e in atlas["edges"])
    label = OrderedDict([("Exact, conditional", "exact"), ("Established", "established"),
                         ("Empirical", "empirical"), ("Hypothesis", "hypothesis")])

    def sub(pattern, repl, text, name):
        new, n = re.subn(pattern, repl, text)
        if n == 0:
            report.append("WARNING: %s: pattern not found: %s" % (name, pattern))
        return new

    path = os.path.join(root, "README.md")
    t = read(path)
    t = sub(r"The atlas has \d+ topics in \d+ clusters, joined by \d+ recorded connections",
            "The atlas has %d topics in %d clusters, joined by %d recorded connections" % (n_nodes, n_clusters, n_edges), t, "README")
    for lab, key in label.items():
        t = sub(r"(\| %s \| )\d+( \|)" % re.escape(lab), r"\g<1>%d\g<2>" % types.get(key, 0), t, "README")
    t = sub(r"There are \d+ curated paths", "There are %d curated paths" % n_paths, t, "README")
    write_if_changed(path, t, changed)

    path = os.path.join(root, "index.html")
    t = read(path)
    t = sub(r"\d+ research topics and \d+ (typed|evidence-labelled) connections",
            lambda m: "%d research topics and %d %s connections" % (n_nodes, n_edges, m.group(1)), t, "index.html")
    t = sub(r"\d+ topics · \d+ typed connections", "%d topics · %d typed connections" % (n_nodes, n_edges), t, "index.html")
    write_if_changed(path, t, changed)
    report.append("totals: %d topics, %d clusters, %d edges (%s), %d paths" % (
        n_nodes, n_clusters, n_edges, ", ".join("%s %d" % (k, types.get(k, 0)) for k in label.values()), n_paths))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    with open(os.path.join(root, "tools", "thesis_track_data.json"), encoding="utf-8") as fh:
        data = json.load(fh, object_pairs_hook=OrderedDict)
    report, changed = [], []
    atlas = apply_atlas(root, data, report, changed)
    n_paths = apply_learning(root, data, atlas, report, changed)
    apply_counts(root, atlas, n_paths, report, changed)
    for line in report:
        print(line)
    print("changed: %s" % (", ".join(os.path.relpath(c, root) for c in changed) if changed else "nothing (already applied)"))


if __name__ == "__main__":
    sys.exit(main())
