"""Summarise the blind Claude rater (results/claude_rater.jsonl) for CHECKPOINT 1.

Prints:
  - per factorial cell: % rated EVAL (eval cells should be high, deploy cells low)
  - AUROC for the key factorial comparisons (all, cross-diagonal, diagonal), on test tasks and overall
  - AUROC for the benchmark comparisons (human vs human, Claude vs Claude, mixed)
  - the 10 factorial prompts the rater got most confidently wrong, to read by hand

Usage: uv run python scripts/rater_report.py
"""

from __future__ import annotations

import json
from collections import defaultdict

from sklearn.metrics import roc_auc_score

from eap.paths import DATA, RESULTS


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def auc(rows, score):
    ys = [r["label"] for r in rows if r["id"] in score]
    xs = [score[r["id"]] for r in rows if r["id"] in score]
    return roc_auc_score(ys, xs) if len(set(ys)) == 2 else float("nan"), len(xs)


def main() -> None:
    ratings = {}
    for l in (RESULTS / "claude_rater.jsonl").read_text().splitlines():
        r = json.loads(l)
        ratings[r["id"]] = r  # last rating wins
    score = {k: v["score"] for k, v in ratings.items()}
    fac, bench = load("factorial.jsonl"), load("benchmark.jsonl")
    print(f"ratings: {len(ratings)} | factorial rated {sum(r['id'] in score for r in fac)}/{len(fac)}"
          f" | benchmark rated {sum(r['id'] in score for r in bench)}/{len(bench)}\n")

    print("factorial: % rated EVAL per cell (intended: eval cells high, deploy cells low)")
    by_cell = defaultdict(list)
    for r in fac:
        if r["id"] in ratings:
            by_cell[r["cell"]].append(ratings[r["id"]]["verdict"] == "EVAL")
    for c in sorted(by_cell):
        v = by_cell[c]
        print(f"    {c:26s} {100 * sum(v) / len(v):5.1f}%  (n={len(v)})")

    def sub(rows, **kw):
        return [r for r in rows if all(r[k] in (v if isinstance(v, tuple) else (v,)) for k, v in kw.items())]

    cross = [r for r in fac if (r["label"], r["format"]) in ((1, "freeform"), (0, "benchmark"))]
    diag = [r for r in fac if (r["label"], r["format"]) in ((1, "benchmark"), (0, "freeform"))]
    print("\nfactorial AUROC (rater score -> label)            all tasks      test tasks")
    for name, rows in [("all 8 cells", fac), ("cross-diagonal (eval-free vs deploy-bench)", cross),
                       ("diagonal (eval-bench vs deploy-free)", diag)]:
        a, n = auc(rows, score)
        at, nt = auc(sub(rows, split="test"), score)
        print(f"    {name:44s} {a:.3f} (n={n})  {at:.3f} (n={nt})")

    print("\nbenchmark AUROC")
    same_items = lambda rows: {r["task_id"] for r in rows}
    hand = sub(bench, kind="rewrite", author="human")
    comps = [
        ("human vs human: originals vs hand rewrites",
         sub(bench, kind="original", task_id=tuple(same_items(hand))) + hand),
        ("claude vs claude: paraphrases vs Claude rewrites",
         sub(bench, kind="paraphrase")
         + sub(bench, kind="rewrite", author="claude", task_id=tuple(same_items(sub(bench, kind="paraphrase"))))),
        ("mixed: originals vs Claude rewrites (secondary)",
         sub(bench, kind="original") + sub(bench, kind="rewrite", author="claude")),
    ]
    for name, rows in comps:
        a, n = auc(rows, score)
        print(f"    {name:50s} {a:.3f} (n={n})" if n else f"    {name:50s} (no data yet)")

    wrong = sorted((r for r in fac if r["id"] in ratings),
                   key=lambda r: score[r["id"]] * (1 if r["label"] == 0 else -1), reverse=True)[:10]
    print("\nmost confidently wrong factorial prompts:")
    for r in wrong:
        v = ratings[r["id"]]
        print(f"\n  {r['id']} ({r['cell']}) rated {v['verdict']} conf {v['confidence']}:\n    "
              + r["text"].replace("\n", "\n    ")[:400])


if __name__ == "__main__":
    main()
