"""Assemble the probe datasets offline from the generated files (no API calls).

Inputs:  data/tasks.jsonl, data/contexts.jsonl, data/benchmark_items.jsonl, data/handwritten_deploy.txt
Outputs (label 1 = eval, 0 = deploy):
  data/factorial.jsonl     every task in all 8 label x format x register cells (core dataset)
  data/explicit.jsonl      contrastive explicit cues, same tasks
  data/benchmark.jsonl     originals, paraphrases, and topic-matched deploy rewrites
  data/review_samples.md   10 prompts per factorial cell + 20 explicit + 20 benchmark (CHECKPOINT 1)
  data/handwrite_todo.md   the 40 benchmark items to rewrite by hand

Fails if any banned eval word appears anywhere in the factorial set.

Usage: uv run python scripts/build_datasets.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from eap.framings import BANNED, FORMATS, LABELS, REGISTERS, render_explicit, render_factorial

DATA = Path("data")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def build_factorial(tasks: dict[str, dict], contexts: list[dict]) -> list[dict]:
    rows = []
    for c in contexts:
        t = tasks[c["task_id"]]
        for label in (1, 0):
            for reg in REGISTERS:
                ctx = c["contexts"][f"{LABELS[label]}_{reg}"]
                for fmt in FORMATS:
                    r = render_factorial(t, c["split"], label, fmt, reg, ctx)
                    rows.append({"id": f"fac-{t['id']}-{r['cell']}", "task_id": t["id"], "split": c["split"],
                                 "domain": t["domain"], **r, "n_words": len(r["text"].split())})
    return rows


def build_explicit(tasks: dict[str, dict], splits: dict[str, str]) -> list[dict]:
    rows = []
    for tid, split in splits.items():
        for label in (1, 0):
            r = render_explicit(tasks[tid], split, label)
            rows.append({"id": f"exp-{tid}-{LABELS[label]}", "task_id": tid, "split": split,
                         "domain": tasks[tid]["domain"], **r, "n_words": len(r["text"].split())})
    return rows


def load_handwritten(path: Path) -> tuple[dict[str, str], list[str]]:
    """'[bid] text' lines -> {bid: text}; untagged lines returned separately."""
    tagged, untagged = {}, []
    if not path.exists():
        return tagged, untagged
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        text = line
        if line.startswith("[") and "]" in line:
            bid, text = line[1 : line.index("]")], line[line.index("]") + 1 :].strip()
            tagged[bid] = text.replace("\\n", "\n")
        else:
            untagged.append(text.replace("\\n", "\n"))
    return tagged, untagged


def build_benchmark(items: list[dict]) -> list[dict]:
    hand, untagged = load_handwritten(DATA / "handwritten_deploy.txt")
    unknown = set(hand) - {it["bid"] for it in items}
    if unknown:
        print(f"WARNING: handwritten ids not in benchmark_items: {sorted(unknown)}")
    rows = []
    for it in items:
        base = {"task_id": it["bid"], "source": it["source"]}
        rows.append({"id": f"bench-{it['bid']}-orig", **base, "label": 1, "kind": "original", "text": it["text"]})
        if it.get("paraphrase_text"):
            rows.append({"id": f"bench-{it['bid']}-para", **base, "label": 1, "kind": "paraphrase",
                         "text": it["paraphrase_text"]})
        if it["bid"] in hand:
            rows.append({"id": f"bench-{it['bid']}-hand", **base, "label": 0, "kind": "rewrite",
                         "author": "handwritten", "text": hand[it["bid"]]})
        elif it.get("rewrite"):
            rows.append({"id": f"bench-{it['bid']}-claude", **base, "label": 0, "kind": "rewrite",
                         "author": "claude", "text": it["rewrite"]})
    for k, text in enumerate(untagged):
        rows.append({"id": f"bench-hw{k:02d}", "task_id": f"hw{k:02d}", "source": "handwritten", "label": 0,
                     "kind": "unmatched", "author": "handwritten", "text": text})
    for r in rows:
        r["n_words"] = len(r["text"].split())
    return rows


def write_handwrite_todo(items: list[dict]) -> None:
    lines = ["# Benchmark items to rewrite by hand (40)\n\n",
             "Write one line per item in `data/handwritten_deploy.txt`, starting with its id in square brackets, e.g.\n\n",
             "`[mmlu1234] my nephew asked me ... and I honestly couldn't explain it`\n\n",
             "Keep the same underlying question (so the answer is the same), in your own voice as a real user would ",
             "write it. No multiple-choice options, and avoid the words test, exam, quiz, homework, evaluation, benchmark, score.\n"]
    for it in items:
        if it["handwrite"]:
            lines.append(f"\n**[{it['bid']}]** ({it['source']})\n\n> {it['question'].replace(chr(10), ' ')}\n")
    (DATA / "handwrite_todo.md").write_text("".join(lines))


def summarise(rows: list[dict], key: str) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    for k in sorted(groups, key=str):
        rs = groups[k]
        print(f"    {str(k):28s} n={len(rs):4d}  mean_words={sum(r['n_words'] for r in rs) / len(rs):5.1f}")


def write_review(fac: list[dict], exp: list[dict], bench: list[dict]) -> None:
    rng = random.Random(0)
    out = ["# Review samples (CHECKPOINT 1)\n\nLabel 1 = eval, 0 = deploy. ",
           "Pay most attention to **deploy-benchmark** and **eval-freeform** cells.\n"]
    by_cell = defaultdict(list)
    for r in fac:
        by_cell[r["cell"]].append(r)
    sections = [(f"factorial: {c}", by_cell[c], 10) for c in sorted(by_cell)]
    sections += [("explicit", exp, 20), ("benchmark", bench, 20)]
    for name, rows, k in sections:
        out.append(f"\n## {name}\n")
        for r in rng.sample(rows, min(k, len(rows))):
            tag = r.get("cell") or r.get("kind") or ""
            out.append(f"\n**{r['id']}** · label {r['label']} · {tag}\n\n```\n{r['text']}\n```\n")
    (DATA / "review_samples.md").write_text("".join(out))


def main() -> None:
    tasks = {t["id"]: t for t in read_jsonl(DATA / "tasks.jsonl")}
    contexts = read_jsonl(DATA / "contexts.jsonl")
    items = read_jsonl(DATA / "benchmark_items.jsonl")
    if not tasks or not contexts:
        sys.exit("Need data/tasks.jsonl and data/contexts.jsonl: run generate_tasks.py and generate_contexts.py first.")
    splits = {c["task_id"]: c["split"] for c in contexts}

    fac = build_factorial(tasks, contexts)
    leaks = [(r["id"], m.group(0)) for r in fac if (m := BANNED.search(r["text"]))]
    if leaks:
        sys.exit(f"BUILD FAILED: banned eval words in factorial set: {leaks[:20]}")
    exp = build_explicit(tasks, splits)
    bench = build_benchmark(items) if items else []

    write_jsonl(DATA / "factorial.jsonl", fac)
    write_jsonl(DATA / "explicit.jsonl", exp)
    if bench:
        write_jsonl(DATA / "benchmark.jsonl", bench)
        write_handwrite_todo(items)

    n_train = sum(s == "train" for s in splits.values())
    print(f"tasks with contexts: {len(splits)} (train {n_train}, test {len(splits) - n_train}); keyword grep passed")
    print(f"factorial: {len(fac)} prompts by cell"); summarise(fac, "cell")
    print(f"explicit: {len(exp)} prompts"); summarise(exp, "label")
    if bench:
        print(f"benchmark: {len(bench)} prompts by kind"); summarise(bench, "kind")
        print(f"  handwritten rewrites in use: {sum(r.get('author') == 'handwritten' for r in bench)}")
    write_review(fac, exp, bench)
    print("Wrote data/review_samples.md and data/handwrite_todo.md")


if __name__ == "__main__":
    main()
