#!/usr/bin/env python3
"""Validate the learning-data layer against learning/SCHEMA.md.

Usage (from the repo root):
    python tools/validate_learning.py [--root DIR]

Checks learning/nodes.json, learning/paths.json, learning/background.json and
learning/questions-*.json (each optional; missing files are noted and skipped)
against ../atlas.json. Prints a summary, then errors and warnings grouped by
file. Exit code 1 if any error was found. Python 3 stdlib only.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, OrderedDict

SKILLS = {
    "linear-algebra", "probability", "optimization", "calculus-odes", "stats-ml",
    "deep-learning", "signals-estimation", "control", "dynamical-systems",
    "generative-models", "rl", "theory",
}
PATH_LEVELS = {"intro", "intermediate", "advanced"}
BG_TYPES = {"single", "multi"}
LATEX_RE = re.compile(r"\\(frac|sum|int|sqrt|mathbf|mathrm|mathcal|alpha|beta|gamma|sigma|theta|lambda|nabla|partial|cdot|top|left|right|begin|end)\b|\$[^$\n]+\$|\\\(|\\\[")
AOTA_RE = re.compile(r"\b(all|none) of the above\b", re.IGNORECASE)


def setup_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


class Report:
    def __init__(self):
        self.errors = OrderedDict()
        self.warnings = OrderedDict()
        self.notes = []
        self.counts = OrderedDict()

    def err(self, f, msg):
        self.errors.setdefault(f, []).append(msg)

    def warn(self, f, msg):
        self.warnings.setdefault(f, []).append(msg)


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def is_num(v):
    return (isinstance(v, (int, float))) and not isinstance(v, bool)


def nonempty_str(v):
    return isinstance(v, str) and v.strip() != ""


def load_json(path, rep, fname):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except UnicodeDecodeError as e:
        rep.err(fname, "not valid UTF-8: %s" % e)
    except json.JSONDecodeError as e:
        rep.err(fname, "invalid JSON: %s" % e)
    except OSError as e:
        rep.err(fname, "cannot read: %s" % e)
    return None


def check_top(doc, key, kind, rep, fname):
    """Validate the {version, <key>} envelope; return the payload or None."""
    if not isinstance(doc, dict):
        rep.err(fname, "top level must be an object")
        return None
    if doc.get("version") != 1 or not is_int(doc.get("version")):
        rep.err(fname, "'version' must be the integer 1 (got %r)" % (doc.get("version"),))
    payload = doc.get(key)
    if not isinstance(payload, kind):
        rep.err(fname, "'%s' must be a%s %s" % (key, "n" if kind is list else "", "array" if kind is list else "object"))
        return None
    return payload


def req_str(obj, field, rep, fname, where):
    if not nonempty_str(obj.get(field)):
        rep.err(fname, "%s: '%s' must be a non-empty string" % (where, field))
        return False
    return True


def check_skill_list(val, rep, fname, where, lo, hi):
    if not isinstance(val, list):
        rep.err(fname, "%s: 'skills' must be an array" % where)
        return
    if not (lo <= len(val) <= hi):
        rep.err(fname, "%s: 'skills' must have %d-%d entries (has %d)" % (where, lo, hi, len(val)))
    for s in val:
        if s not in SKILLS:
            rep.err(fname, "%s: unknown skill %r" % (where, s))
    dups = [s for s, c in Counter(map(repr, val)).items() if c > 1]
    if dups:
        rep.err(fname, "%s: duplicate skills %s" % (where, ", ".join(dups)))


def find_cycle(graph):
    """Return one cycle as a list of node ids (first == last) or None."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    for start in graph:
        if color[start] != WHITE:
            continue
        stack = [(start, iter(graph[start]))]
        trail = [start]
        color[start] = GREY
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = BLACK
                stack.pop()
                trail.pop()
                continue
            if nxt not in color:
                continue
            if color[nxt] == GREY:
                i = trail.index(nxt)
                return trail[i:] + [nxt]
            if color[nxt] == WHITE:
                color[nxt] = GREY
                stack.append((nxt, iter(graph[nxt])))
                trail.append(nxt)
    return None


def validate_nodes(doc, atlas_nodes, rep, fname):
    nodes = check_top(doc, "nodes", dict, rep, fname)
    if nodes is None:
        return None
    graph = {}
    for nid, meta in nodes.items():
        where = "node %r" % nid
        if nid not in atlas_nodes:
            rep.err(fname, "%s: not an atlas node id" % where)
        if not isinstance(meta, dict):
            rep.err(fname, "%s: entry must be an object" % where)
            continue
        lvl = meta.get("level")
        if not is_int(lvl) or not (1 <= lvl <= 4):
            rep.err(fname, "%s: 'level' must be an integer 1-4 (got %r)" % (where, lvl))
        if "skills" not in meta:
            rep.err(fname, "%s: missing 'skills'" % where)
        else:
            check_skill_list(meta["skills"], rep, fname, where, 1, 3)
        pre = meta.get("prereqs")
        if not isinstance(pre, list):
            rep.err(fname, "%s: 'prereqs' must be an array (may be empty)" % where)
            pre = []
        good = []
        for p in pre:
            if not isinstance(p, str) or p not in atlas_nodes:
                rep.err(fname, "%s: prereq %r is not an atlas node id" % (where, p))
            elif p == nid:
                rep.err(fname, "%s: lists itself as a prereq" % where)
            else:
                good.append(p)
        if len(set(map(repr, pre))) != len(pre):
            rep.err(fname, "%s: duplicate prereqs" % where)
        graph[nid] = good
    for g in list(graph.values()):
        for p in g:
            graph.setdefault(p, [])
    cyc = find_cycle(graph)
    if cyc:
        rep.err(fname, "prereq cycle: %s" % " -> ".join(cyc))
    missing = sorted(set(atlas_nodes) - set(nodes))
    if missing:
        rep.err(fname, "missing %d atlas node(s): %s" % (len(missing), ", ".join(missing)))
    rep.counts[fname] = "%d nodes" % len(nodes)
    return nodes


def validate_paths(doc, atlas_nodes, atlas_edges, rep, fname):
    paths = check_top(doc, "paths", list, rep, fname)
    if paths is None:
        return None
    ids = Counter()
    on_path = set()
    for i, p in enumerate(paths):
        where = "path[%d]" % i
        if not isinstance(p, dict):
            rep.err(fname, "%s: must be an object" % where)
            continue
        if req_str(p, "id", rep, fname, where):
            where = "path %r" % p["id"]
            ids[p["id"]] += 1
        for f in ("title", "summary", "audience"):
            req_str(p, f, rep, fname, where)
        if p.get("level") not in PATH_LEVELS:
            rep.err(fname, "%s: 'level' must be one of intro|intermediate|advanced (got %r)" % (where, p.get("level")))
        if "skills" not in p:
            rep.err(fname, "%s: missing 'skills'" % where)
        else:
            check_skill_list(p["skills"], rep, fname, where, 1, len(SKILLS))
        eh = p.get("estHours")
        if not is_num(eh) or eh <= 0:
            rep.err(fname, "%s: 'estHours' must be a positive number (got %r)" % (where, eh))
        steps = p.get("steps")
        if not isinstance(steps, list):
            rep.err(fname, "%s: 'steps' must be an array" % where)
            continue
        if not (4 <= len(steps) <= 9):
            rep.err(fname, "%s: must have 4-9 steps (has %d)" % (where, len(steps)))
        prev = None
        for j, s in enumerate(steps):
            sw = "%s step %d" % (where, j + 1)
            if not isinstance(s, dict):
                rep.err(fname, "%s: must be an object" % sw)
                prev = None
                continue
            node = s.get("node")
            if not isinstance(node, str) or node not in atlas_nodes:
                rep.err(fname, "%s: node %r is not an atlas node id" % (sw, node))
                node = None
            else:
                on_path.add(node)
            req_str(s, "objective", rep, fname, sw)
            req_str(s, "why", rep, fname, sw)
            if "via" not in s:
                rep.err(fname, "%s: missing 'via' (use null when there is no edge)" % sw)
            via = s.get("via")
            if j == 0:
                if via is not None:
                    rep.err(fname, "%s: first step must have via=null (got %r)" % (sw, via))
            elif via is None:
                if "via" in s:
                    rep.warn(fname, "%s: no 'via' edge from the previous step" % sw)
            elif not isinstance(via, str) or via not in atlas_edges:
                rep.err(fname, "%s: via %r is not an atlas edge id" % (sw, via))
            elif node and prev:
                a, b = atlas_edges[via]
                if {a, b} != {prev, node}:
                    rep.err(fname, "%s: via %r joins %s<->%s, not %s<->%s" % (sw, via, a, b, prev, node))
            prev = node
    for pid, c in ids.items():
        if c > 1:
            rep.err(fname, "duplicate path id %r (%d times)" % (pid, c))
    rep.counts[fname] = "%d paths" % len(paths)
    return {"ids": set(ids), "on_path": on_path}


def validate_background(doc, path_ids, rep, fname):
    qs = check_top(doc, "questions", list, rep, fname)
    if qs is None:
        return
    ids = Counter()
    for i, q in enumerate(qs):
        where = "question[%d]" % i
        if not isinstance(q, dict):
            rep.err(fname, "%s: must be an object" % where)
            continue
        if req_str(q, "id", rep, fname, where):
            where = "question %r" % q["id"]
            ids[q["id"]] += 1
        req_str(q, "prompt", rep, fname, where)
        if q.get("type") not in BG_TYPES:
            rep.err(fname, "%s: 'type' must be single|multi (got %r)" % (where, q.get("type")))
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) < 2:
            rep.err(fname, "%s: 'options' must be an array with at least 2 entries" % where)
            continue
        oids = Counter()
        for k, o in enumerate(opts):
            ow = "%s option[%d]" % (where, k)
            if not isinstance(o, dict):
                rep.err(fname, "%s: must be an object" % ow)
                continue
            if req_str(o, "id", rep, fname, ow):
                oids[o["id"]] += 1
                ow = "%s option %r" % (where, o["id"])
            req_str(o, "label", rep, fname, ow)
            sets = o.get("sets", {})
            if not isinstance(sets, dict):
                rep.err(fname, "%s: 'sets' must be an object of skill -> [0,1]" % ow)
            else:
                for sk, v in sets.items():
                    if sk not in SKILLS:
                        rep.err(fname, "%s: 'sets' key %r is not a valid skill" % (ow, sk))
                    if not is_num(v) or not (0 <= v <= 1):
                        rep.err(fname, "%s: 'sets'[%r] must be a number in [0,1] (got %r)" % (ow, sk, v))
            goal = o.get("goal")
            if goal is not None:
                if path_ids is None:
                    rep.err(fname, "%s: goal %r but no learning/paths.json to resolve it" % (ow, goal))
                elif goal not in path_ids:
                    rep.err(fname, "%s: goal %r is not a path id" % (ow, goal))
        for oid, c in oids.items():
            if c > 1:
                rep.err(fname, "%s: duplicate option id %r" % (where, oid))
    for qid, c in ids.items():
        if c > 1:
            rep.err(fname, "duplicate question id %r" % qid)
    rep.counts[fname] = "%d questions" % len(qs)


def validate_questions(doc, atlas_nodes, seen_ids, node_hits, rep, fname):
    items = check_top(doc, "items", list, rep, fname)
    if items is None:
        return []
    answers = []
    for i, it in enumerate(items):
        where = "item[%d]" % i
        if not isinstance(it, dict):
            rep.err(fname, "%s: must be an object" % where)
            continue
        if req_str(it, "id", rep, fname, where):
            iid = it["id"]
            where = "item %r" % iid
            if iid in seen_ids:
                rep.err(fname, "%s: duplicate id (also in %s)" % (where, seen_ids[iid]))
            else:
                seen_ids[iid] = fname
        node = it.get("node")
        if not isinstance(node, str) or node not in atlas_nodes:
            rep.err(fname, "%s: node %r is not an atlas node id" % (where, node))
        else:
            node_hits[node] += 1
        if it.get("skill") not in SKILLS:
            rep.err(fname, "%s: skill %r is not in the vocabulary" % (where, it.get("skill")))
        d = it.get("difficulty")
        if not is_int(d) or not (1 <= d <= 3):
            rep.err(fname, "%s: 'difficulty' must be an integer 1-3 (got %r)" % (where, d))
        req_str(it, "prompt", rep, fname, where)
        req_str(it, "explain", rep, fname, where)
        if "source" not in it:
            rep.err(fname, "%s: missing 'source' (use null)" % where)
        else:
            src = it["source"]
            if src is not None and not (isinstance(src, str) and re.match(r"^https?://\S+$", src)):
                rep.err(fname, "%s: 'source' must be an http(s) URL or null (got %r)" % (where, src))
        ch = it.get("choices")
        ans = it.get("answer")
        if not isinstance(ch, list) or not (3 <= len(ch) <= 4):
            rep.err(fname, "%s: 'choices' must be an array of 3-4 entries (has %s)" % (
                where, len(ch) if isinstance(ch, list) else type(ch).__name__))
            ch = ch if isinstance(ch, list) else []
        else:
            norm = []
            for k, c in enumerate(ch):
                if not nonempty_str(c):
                    rep.err(fname, "%s: choice %d must be a non-empty string" % (where, k))
                else:
                    norm.append(" ".join(c.split()).casefold())
            dup = [c for c, n in Counter(norm).items() if n > 1]
            if dup:
                rep.err(fname, "%s: duplicate choices %s" % (where, ", ".join(repr(x) for x in dup)))
        if not is_int(ans) or not (0 <= ans < len(ch)):
            rep.err(fname, "%s: 'answer' must be an integer index into choices (got %r, %d choices)" % (where, ans, len(ch)))
        elif ch:
            answers.append(ans)
        text = " ".join(str(x) for x in [it.get("prompt"), it.get("explain")] + list(ch) if isinstance(x, str))
        if LATEX_RE.search(text):
            rep.warn(fname, "%s: LaTeX-looking text (%s); use plain Unicode" % (where, LATEX_RE.search(text).group(0)))
        if isinstance(it.get("prompt"), str) and AOTA_RE.search(it["prompt"]) or any(
                isinstance(c, str) and AOTA_RE.search(c) for c in ch):
            rep.warn(fname, "%s: uses 'all/none of the above'" % where)
    skew_warning(answers, rep, fname)
    rep.counts[fname] = "%d items" % len(items)
    return answers


def skew_warning(answers, rep, fname):
    if len(answers) < 5:
        return
    c = Counter(answers)
    idx, n = c.most_common(1)[0]
    if n / len(answers) > 0.40:
        dist = ", ".join("%d:%d" % (k, c[k]) for k in sorted(c))
        rep.warn(fname, "answer-index skew: %.0f%% of %d items use index %d (distribution %s)" % (
            100.0 * n / len(answers), len(answers), idx, dist))


def main(argv=None):
    setup_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repo root containing atlas.json and learning/ (default: cwd)")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    rep = Report()

    atlas = load_json(os.path.join(root, "atlas.json"), rep, "atlas.json")
    if not isinstance(atlas, dict) or not isinstance(atlas.get("nodes"), list) or not isinstance(atlas.get("edges"), list):
        rep.err("atlas.json", "missing or malformed (needs 'nodes' and 'edges' arrays); cannot validate")
        return finish(rep, root)
    atlas_nodes = {n["id"] for n in atlas["nodes"] if isinstance(n, dict) and "id" in n}
    atlas_edges = {e["id"]: (e.get("a"), e.get("b")) for e in atlas["edges"] if isinstance(e, dict) and "id" in e}

    ldir = os.path.join(root, "learning")

    def fpath(name):
        return os.path.join(ldir, name)

    nodes = paths_info = None
    if os.path.exists(fpath("nodes.json")):
        doc = load_json(fpath("nodes.json"), rep, "nodes.json")
        if doc is not None:
            nodes = validate_nodes(doc, atlas_nodes, rep, "nodes.json")
    else:
        rep.notes.append("learning/nodes.json not found - skipped")

    if os.path.exists(fpath("paths.json")):
        doc = load_json(fpath("paths.json"), rep, "paths.json")
        if doc is not None:
            paths_info = validate_paths(doc, atlas_nodes, atlas_edges, rep, "paths.json")
    else:
        rep.notes.append("learning/paths.json not found - skipped")

    if os.path.exists(fpath("background.json")):
        doc = load_json(fpath("background.json"), rep, "background.json")
        if doc is not None:
            validate_background(doc, paths_info["ids"] if paths_info else None, rep, "background.json")
    else:
        rep.notes.append("learning/background.json not found - skipped")

    qfiles = sorted(glob.glob(os.path.join(ldir, "questions-*.json")))
    seen_ids = {}
    node_hits = Counter()
    if not qfiles:
        rep.notes.append("no learning/questions-*.json found - skipped")
    for qf in qfiles:
        name = os.path.basename(qf)
        doc = load_json(qf, rep, name)
        if doc is not None:
            validate_questions(doc, atlas_nodes, seen_ids, node_hits, rep, name)

    # Cross-file coverage warnings.
    topic_nodes = sorted(atlas_nodes)
    if qfiles:
        zero = [n for n in topic_nodes if node_hits[n] == 0]
        if zero:
            rep.warn("questions-*", "%d node(s) with zero questions: %s" % (len(zero), ", ".join(zero)))
    if paths_info is not None:
        off = [n for n in topic_nodes if n not in paths_info["on_path"]]
        if off:
            rep.warn("paths.json", "%d node(s) on no path: %s" % (len(off), ", ".join(off)))
    return finish(rep, root)


def finish(rep, root):
    ne = sum(len(v) for v in rep.errors.values())
    nw = sum(len(v) for v in rep.warnings.values())
    print("validate_learning: root=%s" % root)
    for f, c in rep.counts.items():
        print("  checked %-24s %s" % (f, c))
    for n in rep.notes:
        print("  note: %s" % n)
    print("RESULT: %s - %d error(s), %d warning(s)" % ("FAIL" if ne else "OK", ne, nw))
    if ne:
        print("\nERRORS")
        for f, msgs in rep.errors.items():
            print("  [%s]" % f)
            for m in msgs:
                print("    - %s" % m)
    if nw:
        print("\nWARNINGS")
        for f, msgs in rep.warnings.items():
            print("  [%s]" % f)
            for m in msgs:
                print("    - %s" % m)
    return 1 if ne else 0


if __name__ == "__main__":
    sys.exit(main())
