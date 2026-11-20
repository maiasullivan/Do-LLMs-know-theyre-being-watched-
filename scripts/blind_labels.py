"""Human blind-labelling check for CHECKPOINT 1 (plan section 4.7): an independent check on the
factorial labels that doesn't depend on Claude.

  make:  uv run python scripts/blind_labels.py make
         -> data/blind_labels.md  (32 prompts, 4 per cell, shuffled, opaque ids, no labels)
         -> data/blind_key.json   (the answers: don't open it until you've labelled)
  score: uv run python scripts/blind_labels.py score
         -> your agreement with the intended labels, per cell, next to the Claude rater

Fill in each "ANSWER:" line in blind_labels.md with E (looks like an eval) or R (a real user).
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict

from eap.paths import DATA, RESULTS

SHEET, KEY = DATA / "blind_labels.md", DATA / "blind_key.json"
PER_CELL = 4


def make() -> None:
    if SHEET.exists() and re.search(r"^ANSWER:\s*[ER]", SHEET.read_text(), re.M | re.I):
        sys.exit(f"{SHEET} already has answers; refusing to overwrite.")
    rows = [json.loads(l) for l in (DATA / "factorial.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random("blind-labels")
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    picked, used_tasks = [], set()
    for cell in sorted(by_cell):
        pool = [r for r in by_cell[cell] if r["task_id"] not in used_tasks]  # one prompt per task
        for r in rng.sample(pool, PER_CELL):
            picked.append(r)
            used_tasks.add(r["task_id"])
    rng.shuffle(picked)
    key = {f"b{i:02d}": {"id": r["id"], "cell": r["cell"], "label": r["label"]} for i, r in enumerate(picked, 1)}
    out = ["# Blind labelling (CHECKPOINT 1)", "",
           "For each prompt, would you guess it came from someone **testing** an AI (E) or a **real user** "
           "with a real need (R)? Go with your first read. Write E or R after `ANSWER:`. "
           "Don't open `blind_key.json` until you're done.", ""]
    for bid, r in zip(key, picked):
        out += [f"## {bid}", "", "```", r["text"], "```", "", "ANSWER: ", ""]
    SHEET.write_text("\n".join(out))
    KEY.write_text(json.dumps(key, indent=1))
    print(f"Wrote {SHEET} ({len(picked)} prompts) and {KEY}")


def score() -> None:
    key = json.loads(KEY.read_text())
    answers = dict(re.findall(r"^## (b\d+)\s*$.*?^ANSWER:[ \t]*([A-Za-z]?)", SHEET.read_text(), re.M | re.S))
    missing = [b for b in key if answers.get(b, "").upper() not in ("E", "R")]
    if missing:
        sys.exit(f"Missing or invalid answers for: {', '.join(missing)}")
    rater = {}
    rp = RESULTS / "claude_rater.jsonl"
    if rp.exists():
        for l in rp.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                rater[d["id"]] = d
    cells = defaultdict(lambda: [0, 0, 0, 0])  # you_eval, n, rater_eval, rater_n
    agree = 0
    for b, k in key.items():
        you = answers[b].upper() == "E"
        agree += you == bool(k["label"])
        c = cells[k["cell"]]
        c[0] += you
        c[1] += 1
        if k["id"] in rater:
            c[2] += rater[k["id"]]["score"] > 0
            c[3] += 1
    print(f"You agreed with the intended label on {agree}/{len(key)} ({agree / len(key):.0%})\n")
    print(f"{'cell':28s} {'you: % EVAL':>12s} {'Claude: % EVAL':>15s}")
    for cell in sorted(cells):
        y, n, r, rn = cells[cell]
        rs = f"{r}/{rn}" if rn else "-"
        print(f"{cell:28s} {f'{y}/{n}':>12s} {rs:>15s}")


if __name__ == "__main__":
    {"make": make, "score": score}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
