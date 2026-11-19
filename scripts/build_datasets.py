"""Assemble the probe datasets offline from the generated files (no API calls).

Inputs:  data/tasks.jsonl, data/contexts.jsonl, data/bridges.jsonl, data/benchmark_items.jsonl,
         data/handwritten_deploy.txt
Outputs (label 1 = eval, 0 = deploy):
  data/factorial.jsonl     every task in all 8 label x format x register cells (core dataset)
  data/explicit.jsonl      contrastive explicit cues, same tasks
  data/benchmark.jsonl     originals, paraphrases, and topic-matched deploy rewrites (Claude + hand-written)
  data/review_samples.md   10 prompts per factorial cell + 20 explicit + 20 benchmark (CHECKPOINT 1)
  data/handwrite_todo.md   the 40 benchmark items to rewrite by hand

Checks:
  - fails if any banned eval word appears anywhere in the factorial set
  - prints a context-diversity report (distinct openings per label x register)
  - freezing: `--freeze` records a SHA-256 of every dataset in data/frozen_manifest.json. After that,
    a build that would change any dataset fails unless you pass `--change-reason "..."`, which is
    logged to data/dataset_changes.md and updates the manifest.

Usage:
  uv run python scripts/build_datasets.py                 # build (and verify against manifest if frozen)
  uv run python scripts/build_datasets.py --freeze        # build and freeze (end of CHECKPOINT 1)
  uv run python scripts/build_datasets.py --change-reason "added 12 hand-written rewrites"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eap.framings import BANNED, FORMATS, LABELS, REGISTERS, render_explicit, render_factorial
from eap.paths import DATA

DATASETS = ("factorial.jsonl", "explicit.jsonl", "benchmark.jsonl")
MANIFEST = DATA / "frozen_manifest.json"
CHANGELOG = DATA / "dataset_changes.md"


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def dumps_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


# ------------------------------------------------------------------ factorial / explicit
def build_factorial(tasks: dict[str, dict], contexts: list[dict], bridges: dict[str, dict]) -> list[dict]:
    rows = []
    for c in sorted(contexts, key=lambda c: c["task_id"]):
        t = tasks[c["task_id"]]
        for label in (1, 0):
            for reg in REGISTERS:
                key = f"{LABELS[label]}_{reg}"
                ctx = c["contexts"][key]
                for fmt in FORMATS:
                    r = render_factorial(t, c["split"], label, fmt, reg, ctx, bridges[t["id"]]["bridges"][key])
                    rows.append({"id": f"fac-{t['id']}-{r['cell']}", "task_id": t["id"], "split": c["split"],
                                 "domain": t["domain"], **r, "n_words": len(r["text"].split())})
    return rows


def build_explicit(tasks: dict[str, dict], splits: dict[str, str]) -> list[dict]:
    rows = []
    for tid in sorted(splits):
        for label in (1, 0):
            r = render_explicit(tasks[tid], splits[tid], label)
            rows.append({"id": f"exp-{tid}-{LABELS[label]}", "task_id": tid, "split": splits[tid],
                         "domain": tasks[tid]["domain"], **r, "n_words": len(r["text"].split())})
    return rows


def diversity_report(contexts: list[dict]) -> None:
    """Distinct 4-word openings per label x register. Eval and deploy should be comparable."""
    print("context diversity (distinct 4-word prefix openings / n; top repeat):")
    for lab in ("eval", "deploy"):
        for reg in REGISTERS:
            c = Counter(" ".join(x["contexts"][f"{lab}_{reg}"]["prefix"].lower().split()[:4]) for x in contexts)
            top, n = c.most_common(1)[0]
            print(f"    {lab}_{reg:7s} {len(c):4d}/{len(contexts)}   top x{n}: {top!r}")


def leak_report(fac: list[dict], contexts: list[dict]) -> None:
    """Two quick surface-leak checks to run before freezing.

    1. Length-only AUROC per format x register (word count -> label). Should be near 0.5.
    2. The context words most skewed toward one label. Some skew is the intended cue ('my', 'next');
       a word with no business being label-specific (e.g. an artefact of the generation prompt) is a leak.
    """
    import math
    import re

    from sklearn.metrics import roc_auc_score

    print("length-only AUROC (0.5 = no leak; WARN above 0.65):")
    for fmt in FORMATS:
        for reg in REGISTERS:
            rs = [r for r in fac if r["format"] == fmt and r["register"] == reg]
            auc = roc_auc_score([r["label"] for r in rs], [r["n_words"] for r in rs])
            auc = max(auc, 1 - auc)  # direction-free
            print(f"    {fmt:9s} {reg:6s} {auc:.2f}{'   <-- WARN' if auc > 0.65 else ''}")

    df = {"eval": Counter(), "deploy": Counter()}
    for c in contexts:
        for lab in df:
            for reg in REGISTERS:
                x = c["contexts"][f"{lab}_{reg}"]
                df[lab].update(set(re.findall(r"[a-z']+", f"{x['prefix']} {x['suffix']}".lower())))
    n = 2 * len(contexts)  # contexts per label
    rows = []
    for w in set(df["eval"]) | set(df["deploy"]):
        e, d = df["eval"][w], df["deploy"][w]
        if e + d >= 0.05 * n:
            rows.append((math.log((e + 1) / (d + 1)), w, e, d))
    rows.sort()
    fmt_row = lambda r: f"{r[1]} ({r[2]}/{r[3]})"
    print(f"most label-skewed context words (word (eval/deploy docs), of {n} per label):")
    print("    eval-skewed:   " + ", ".join(fmt_row(r) for r in rows[::-1][:15]))
    print("    deploy-skewed: " + ", ".join(fmt_row(r) for r in rows[:15]))


# ------------------------------------------------------------------ benchmark
def load_handwritten(path: Path) -> tuple[dict[str, str], list[str]]:
    """'[bid] text' lines -> {bid: text}; untagged lines returned separately."""
    tagged, untagged = {}, []
    if not path.exists():
        return tagged, untagged
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and "]" in line:
            bid, text = line[1 : line.index("]")], line[line.index("]") + 1 :].strip()
            tagged[bid] = text.replace("\\n", "\n")
        else:
            untagged.append(line.replace("\\n", "\n"))
    return tagged, untagged


def build_benchmark(items: list[dict]) -> list[dict]:
    """Every row carries `author` (human or claude), so analyses can compare like with like:
      - human vs human:   original (eval)   vs hand-written rewrite (deploy)
      - claude vs claude: paraphrase (eval) vs Claude rewrite (deploy)
    Claude rewrites are kept for every item even where a hand-written one exists."""
    hand, untagged = load_handwritten(DATA / "handwritten_deploy.txt")
    unknown = set(hand) - {it["bid"] for it in items}
    if unknown:
        print(f"WARNING: handwritten ids not in benchmark_items: {sorted(unknown)}")
    rows = []
    for it in items:
        base = {"task_id": it["bid"], "source": it["source"]}
        rows.append({"id": f"bench-{it['bid']}-orig", **base, "label": 1, "kind": "original",
                     "author": "human", "text": it["text"]})
        if it.get("paraphrase_text"):
            rows.append({"id": f"bench-{it['bid']}-para", **base, "label": 1, "kind": "paraphrase",
                         "author": "claude", "text": it["paraphrase_text"]})
        if it.get("rewrite"):
            rows.append({"id": f"bench-{it['bid']}-claude", **base, "label": 0, "kind": "rewrite",
                         "author": "claude", "text": it["rewrite"]})
        if it["bid"] in hand:
            rows.append({"id": f"bench-{it['bid']}-hand", **base, "label": 0, "kind": "rewrite",
                         "author": "human", "text": hand[it["bid"]]})
    for k, text in enumerate(untagged):
        rows.append({"id": f"bench-hw{k:02d}", "task_id": f"hw{k:02d}", "source": "handwritten", "label": 0,
                     "kind": "unmatched", "author": "human", "text": text})
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


# ------------------------------------------------------------------ reporting / freezing
def summarise(rows: list[dict], *keys: str) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(str(r.get(k)) for k in keys)].append(r)
    for k in sorted(groups):
        rs = groups[k]
        print(f"    {' / '.join(k):34s} n={len(rs):4d}  mean_words={sum(r['n_words'] for r in rs) / len(rs):5.1f}")


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
            tag = r.get("cell") or f"{r.get('kind', '')} ({r.get('author', '')})"
            out.append(f"\n**{r['id']}** · label {r['label']} · {tag}\n\n```\n{r['text']}\n```\n")
    (DATA / "review_samples.md").write_text("".join(out))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def check_and_write(outputs: dict[str, str], freeze: bool, reason: str | None) -> None:
    hashes = {name: sha(text) for name, text in outputs.items()}
    changed: list[str] = []
    if MANIFEST.exists() and not freeze:
        frozen = json.loads(MANIFEST.read_text())["sha256"]
        changed = sorted(n for n in hashes if frozen.get(n) != hashes[n])
        if changed and not reason:
            sys.exit(f"BUILD FAILED: datasets are frozen and these would change: {changed}.\n"
                     f"Re-run with --change-reason \"...\" to log the change, or revert the inputs.")
        if changed:
            with CHANGELOG.open("a") as f:
                f.write(f"- {dt.datetime.now().isoformat(timespec='minutes')}: changed {', '.join(changed)}. Reason: {reason}\n")
            print(f"Logged change to {CHANGELOG.name}: {changed}")
    for name, text in outputs.items():
        (DATA / name).write_text(text)
    if freeze or changed:
        MANIFEST.write_text(json.dumps({"frozen_at": dt.datetime.now().isoformat(timespec="minutes"),
                                        "sha256": hashes}, indent=2) + "\n")
        print(f"{'Froze' if freeze else 'Updated'} {MANIFEST.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="record dataset hashes (end of CHECKPOINT 1)")
    ap.add_argument("--change-reason", help="allow and log a change to frozen datasets")
    args = ap.parse_args()

    tasks = {t["id"]: t for t in read_jsonl(DATA / "tasks.jsonl")}
    contexts = read_jsonl(DATA / "contexts.jsonl")
    items = read_jsonl(DATA / "benchmark_items.jsonl")
    bridges = {b["task_id"]: b for b in read_jsonl(DATA / "bridges.jsonl")}
    no_bridge = sorted(c["task_id"] for c in contexts if c["task_id"] not in bridges)
    if no_bridge:
        print(f"WARNING: dropping {len(no_bridge)} tasks with no bridges (re-run generate_bridges.py): {no_bridge}")
        contexts = [c for c in contexts if c["task_id"] in bridges]
    if not tasks or not contexts or not items:
        sys.exit("Need data/tasks.jsonl, contexts.jsonl and benchmark_items.jsonl: run the generate_*.py scripts first.")
    splits = {c["task_id"]: c["split"] for c in contexts}

    fac = build_factorial(tasks, contexts, bridges)
    leaks = [(r["id"], m.group(0)) for r in fac if (m := BANNED.search(r["text"]))]
    if leaks:
        sys.exit(f"BUILD FAILED: banned eval words in factorial set: {leaks[:20]}")
    exp = build_explicit(tasks, splits)
    bench = build_benchmark(items)

    check_and_write({"factorial.jsonl": dumps_jsonl(fac), "explicit.jsonl": dumps_jsonl(exp),
                     "benchmark.jsonl": dumps_jsonl(bench)}, args.freeze, args.change_reason)
    write_handwrite_todo(items)
    write_review(fac, exp, bench)

    n_train = sum(s == "train" for s in splits.values())
    missing = sorted(set(tasks) - set(splits))
    print(f"tasks with contexts: {len(splits)} of {len(tasks)} (train {n_train}, test {len(splits) - n_train})"
          + (f"; MISSING contexts for {len(missing)}: re-run generate_contexts.py" if missing else ""))
    print("keyword grep passed")
    diversity_report(contexts)
    leak_report(fac, contexts)
    print(f"factorial: {len(fac)} prompts"); summarise(fac, "cell")
    print(f"explicit: {len(exp)} prompts"); summarise(exp, "label")
    print(f"benchmark: {len(bench)} prompts"); summarise(bench, "kind", "author")
    print("Wrote data/review_samples.md and data/handwrite_todo.md")


if __name__ == "__main__":
    main()
