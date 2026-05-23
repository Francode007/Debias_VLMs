"""
Phase 0c — McNemar Paired Test (vanilla vs v4)
─────────────────────────────────────────────────────────────────────────────
The +0.11 pp delta reported in the README compared two independent binomials.
The right test on the same SB-Bench test set is McNemar's test on the
discordant pairs.

USAGE
    python -m scripts.phase0_mcnemar \\
        --vanilla_jsonl  logs/eval_vanilla.jsonl \\
        --v4_jsonl       logs/eval_v4.jsonl

OUTPUT
    Per-category contingency table  +  exact binomial p-value on b vs c,
    where
        a = both correct,  b = vanilla correct & v4 wrong,
        c = vanilla wrong & v4 correct,  d = both wrong.
"""

import argparse
import json
import re
from collections import defaultdict
from math import comb

LETTER_RE = re.compile(r"\b([ABC])\b")


def parse_letter(text: str):
    m = LETTER_RE.search(text.strip().upper())
    return m.group(1) if m else None


def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[int(r["question_id"])] = r
    return out


def is_correct(rec):
    pred = parse_letter(rec.get("text", ""))
    gold = {0: "A", 1: "B", 2: "C"}[int(rec["label"])]
    return pred == gold


def exact_binomial_two_sided(b: int, c: int) -> float:
    """Two-sided exact mid-p of McNemar (binomial(n=b+c, p=0.5))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla_jsonl", required=True)
    ap.add_argument("--v4_jsonl", required=True)
    args = ap.parse_args()

    A = load_jsonl(args.vanilla_jsonl)
    B = load_jsonl(args.v4_jsonl)
    shared = sorted(set(A) & set(B))
    print(f"Shared question ids: {len(shared)}")

    counts = defaultdict(lambda: [0, 0, 0, 0])  # a,b,c,d per category
    overall = [0, 0, 0, 0]
    for qid in shared:
        ra, rb = A[qid], B[qid]
        if ra["category"] != rb["category"] or ra["label"] != rb["label"]:
            continue
        ca, cb = is_correct(ra), is_correct(rb)
        idx = (0 if ca and cb else 1 if ca and not cb
               else 2 if cb and not ca else 3)
        counts[ra["category"]][idx] += 1
        overall[idx] += 1

    print(f"{'Category':25s} {'a':>5} {'b':>5} {'c':>5} {'d':>5}"
          f" {'acc_v':>7} {'acc_4':>7} {'p_mcnemar':>10}")
    for cat, (a, b, c, d) in sorted(counts.items()):
        n = a + b + c + d
        av = (a + b) / n if n else 0
        a4 = (a + c) / n if n else 0
        p = exact_binomial_two_sided(b, c)
        print(f"{cat:25s} {a:5d} {b:5d} {c:5d} {d:5d}"
              f" {av:7.4f} {a4:7.4f} {p:10.4f}")

    a, b, c, d = overall
    n = a + b + c + d
    print("-" * 80)
    print(f"{'OVERALL':25s} {a:5d} {b:5d} {c:5d} {d:5d}"
          f" {(a+b)/n:7.4f} {(a+c)/n:7.4f}"
          f" {exact_binomial_two_sided(b, c):10.4f}")
    print(f"\nN={n}  vanilla_correct={a+b}  v4_correct={a+c}  "
          f"diff={(a+c)-(a+b)}  discordant_pairs(b,c)=({b},{c})")


if __name__ == "__main__":
    main()
