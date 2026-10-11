"""Site name and note titles: 'Artificial Learning Dynamics Atlas'; drafts are 'Bridges Between Learning Theories'.

Safe to run more than once.
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
NAME = "Artificial Learning Dynamics Atlas"
DRAFTS = "Bridges Between Learning Theories"


def edit(path, *subs):
    f = ROOT / path
    s = f.read_text(encoding="utf-8")
    for old, new in subs:
        s = re.sub(old, new, s) if isinstance(old, re.Pattern) else s.replace(old, new)
    f.write_text(s, encoding="utf-8")


atlas = json.loads((ROOT / "atlas.json").read_text(encoding="utf-8"))
atlas["title"] = NAME
for e in atlas["edges"]:
    e["text"] = e["text"].replace("(Unifying Theory draft", f"({DRAFTS} draft")
(ROOT / "atlas.json").write_text(json.dumps(atlas, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

OLD_NAME = re.compile(r"(The )?Computational Learning Dynamics Atlas")
edit("index.html", (OLD_NAME, NAME), ("Atlas Navigator", NAME), (re.compile(r"/og\.png(\?v=\d+)?\""), "/og.png?v=2\""))
edit("backend/worker.js", (re.compile(r"for (The )?Computational Learning Dynamics Atlas"), f"for the {NAME}"), (OLD_NAME, NAME))
edit("README.md", (OLD_NAME, NAME),
     ("Unifying Theory —", f"{DRAFTS} —"),
     ("Unifying%20Theory%20%E2%80%94", "Bridges%20Between%20Learning%20Theories%20%E2%80%94"),
     (": a unifying account from statistical learning to deep, latent and recurrent learning.",
      ": status-labelled bridges between theories of learning, from statistical learning to deep, latent and recurrent learning, including where they do not connect."))
for f in ROOT.glob(f"{DRAFTS} — *.md"):
    edit(f.name, ("Unifying Theory —", f"{DRAFTS} —"), ("Unifying them means two things", "Connecting them means two things"))
