"""Overlay reviewed wording onto learning paths (titles, summaries, audiences, step objectives and whys).

Runs after the scripts that build paths, because they rewrite those paths. A step is changed only while it
still points at the node it was reviewed for. Safe to run more than once.

usage: python3 tools/apply_path_text.py tools/path_text_2026-10-11.json
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
overlay = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
path_file = ROOT / "learning" / "paths.json"
doc = json.loads(path_file.read_text(encoding="utf-8"))
paths = {p["id"]: p for p in doc["paths"]}
for pid, change in overlay["paths"].items():
    p = paths.get(pid)
    if p is None:
        print(f"SKIP {pid}: path not found")
        continue
    for field in ("title", "summary", "audience", "estHours", "level"):
        if field in change:
            p[field] = change[field]
    for i, step_change in change.get("steps", {}).items():
        i = int(i)
        if i >= len(p["steps"]) or p["steps"][i]["node"] != step_change["node"]:
            print(f"SKIP {pid} step {i}: node changed since review")
            continue
        for field in ("objective", "why"):
            if field in step_change:
                p["steps"][i][field] = step_change[field]
path_file.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
