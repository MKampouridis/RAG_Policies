#!/usr/bin/env python3
"""Compare human scores against the automatic judge's.

Every comparison in this ledger is mediated by a model scoring answers 1-5, and
nobody had ever checked those scores against a person. This answers four
questions the ledger depends on:

  1. How often do they agree exactly, and within 1?
  2. Is the judge systematically GENEROUS or HARSH? A constant offset is
     harmless for A/B work (it cancels); disagreement that varies is not.
  3. Does it rank answers the same way? A/B comparisons need ORDER, not
     absolute values - so correlation matters more than agreement.
  4. Where does it fail? Specifically, does it mistake plausible-sounding
     answers for supported ones, which is this corpus's characteristic risk.
"""
import json
import pathlib
import statistics as st
from collections import Counter


def main() -> int:
    items = json.loads(pathlib.Path("eval/judge_calibration_items.json").read_text())
    scores = {s["id"]: s for s in
              json.loads(pathlib.Path("eval/judge_calibration_scores.json").read_text())}
    pairs = []
    for it in items:
        s = scores.get(it["id"])
        if not s or s.get("human_score") is None:
            continue
        pairs.append((it, s["human_score"], it["judge_score"], (s.get("note") or "").strip()))

    n = len(pairs)
    diffs = [j - h for _, h, j, _ in pairs]          # positive = judge higher
    exact = sum(1 for d in diffs if d == 0)
    within1 = sum(1 for d in diffs if abs(d) <= 1)

    print(f"\n  n = {n} answers scored by both\n")
    print(f"  exact agreement      {exact}/{n}  ({exact/n*100:.0f}%)")
    print(f"  within 1 point       {within1}/{n}  ({within1/n*100:.0f}%)")
    print(f"  mean difference      {st.mean(diffs):+.2f}   (judge minus human)")
    print(f"  median difference    {st.median(diffs):+.1f}")
    print(f"  judge HIGHER on      {sum(1 for d in diffs if d > 0)}")
    print(f"  judge LOWER on       {sum(1 for d in diffs if d < 0)}")

    # Rank correlation - what A/B work actually relies on.
    #
    # TIE HANDLING IS NOT OPTIONAL HERE. The first version assigned ordinal
    # positions from a plain sort, so tied scores got arbitrary distinct ranks
    # decided by row order in the JSON file. On this data - which is almost all
    # ties, the human using 4 distinct levels across 30 items - that statistic
    # moved between +0.395 and +0.743 depending on row order alone, and the
    # reported +0.46 was near the bottom of its own range. Average ranks for
    # tied values is the standard correction and makes the statistic
    # well-defined. (External review, round 8.)
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    h = [p[1] for p in pairs]
    j = [p[2] for p in pairs]
    rh, rj = rank(h), rank(j)
    mh, mj = st.mean(rh), st.mean(rj)
    num = sum((a - mh) * (b - mj) for a, b in zip(rh, rj))
    den = (sum((a - mh) ** 2 for a in rh) * sum((b - mj) ** 2 for b in rj)) ** 0.5
    print(f"  rank correlation     {num/den:+.2f}   (1.0 = identical ordering)")

    print(f"\n  {'human':>6}{'judge':>7}   n")
    grid = Counter((hh, jj) for _, hh, jj, _ in pairs)
    for hh in range(1, 6):
        row = "".join(f"{grid.get((hh, jj), 0):>4}" for jj in range(1, 6))
        print(f"  {hh:>6}        {row}     <- human {hh}")
    print(f"  {'':>6}        {'   1   2   3   4   5'}")

    worst = sorted(pairs, key=lambda p: -abs(p[2] - p[1]))[:6]
    print(f"\n  biggest disagreements:")
    for it, hs, js, note in worst:
        if hs == js:
            continue
        print(f"    human {hs} vs judge {js}   {it['question'][:62]}")
        if note:
            print(f"       you: {note[:100]}")
        print(f"       judge: {(it.get('judge_just') or '')[:100]}")

    pathlib.Path("eval/judge_calibration_report.json").write_text(json.dumps({
        "n": n, "exact": exact, "within1": within1,
        "mean_diff": st.mean(diffs), "rank_corr": num / den,
        "judge_higher": sum(1 for d in diffs if d > 0),
        "judge_lower": sum(1 for d in diffs if d < 0)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
