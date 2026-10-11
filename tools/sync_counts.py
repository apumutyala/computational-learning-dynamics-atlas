"""Recompute cluster counts and the topic/edge totals quoted in README.md and index.html."""
import collections
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
atlas = json.loads((ROOT / "atlas.json").read_text(encoding="utf-8"))
per_group = collections.Counter(n["group"] for n in atlas["nodes"])
atlas["clusters"] = [c for c in atlas["clusters"] if per_group[c["name"]]]
for c in atlas["clusters"]:
    c["count"] = per_group[c["name"]]
(ROOT / "atlas.json").write_text(json.dumps(atlas, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

T, E, C = len(atlas["nodes"]), len(atlas["edges"]), len(atlas["clusters"])
types = collections.Counter(e["type"] for e in atlas["edges"])

readme = (ROOT / "README.md").read_text(encoding="utf-8")
readme = re.sub(r"The atlas has \d+ topics in \d+ clusters, joined by \d+ recorded connections",
                f"The atlas has {T} topics in {C} clusters, joined by {E} recorded connections", readme)
for label, key in [("Exact, conditional", "exact"), ("Established", "established"),
                   ("Empirical", "empirical"), ("Hypothesis", "hypothesis")]:
    readme = re.sub(rf"^\| {re.escape(label)} \| \d+ \|", f"| {label} | {types[key]} |", readme, flags=re.M)
(ROOT / "README.md").write_text(readme, encoding="utf-8")

html = (ROOT / "index.html").read_text(encoding="utf-8")
html = re.sub(r"\b\d+ research topics and \d+ (typed|evidence-labelled) connections",
              lambda m: f"{T} research topics and {E} {m.group(1)} connections", html)
html = re.sub(r"\d+ topics · \d+ typed connections", f"{T} topics · {E} typed connections", html)
(ROOT / "index.html").write_text(html, encoding="utf-8")
print(f"{T} topics, {C} clusters, {E} edges, types {dict(types)}")

# Cache-bust the atlas fetch with a hash of the final atlas.json, so browsers pick up every data change.
digest = hashlib.sha1((ROOT / "atlas.json").read_bytes()).hexdigest()[:8]
app = (ROOT / "app.js").read_text(encoding="utf-8")
app = re.sub(r"fetch\('atlas\.json\?v=[^']*'\)", f"fetch('atlas.json?v={digest}')", app)
(ROOT / "app.js").write_text(app, encoding="utf-8")
