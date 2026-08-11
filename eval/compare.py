#!/usr/bin/env python3
"""Compare two eval arms PER QUESTION instead of by their means.

WHY THIS EXISTS
Every comparison in this ledger until now was "arm A mean vs arm B mean",
judged against a single empirical noise floor (+/-0.20 on a 10-question set).
That throws away the fact that both arms answered THE SAME QUESTIONS, and it
hides the only number that usually matters: how many turns actually changed.

The adjacent-chunk decision is the worked example. Reported as "+0.05, below
the noise floor, shipped anyway". Paired, on the same stored files:

    mean paired diff  +0.050
    95% CI            [-0.250, +0.350]
    win/tie/loss      3 / 14 / 3
    turns changed     6 of 20

Same mean, completely different claim. Fourteen turns were IDENTICAL, and the
six that moved split three up, three down. "+0.05" reads like a small
improvement; "3 better, 3 worse, 14 untouched, interval straddles zero" is what
the evidence actually supports.

(The external review that prompted this tool reported +0.000 over 26 turns for
the same comparison. That does not reproduce from these files - they hold 20
scored turns - but its conclusion, that most turns were unchanged and the
effect is unresolvable, is exactly right.)

WHAT IT REPORTS
  mean / median paired difference   - the effect, with question variance removed
  95% bootstrap CI                  - resampling QUESTIONS, which is the unit
                                      of independence (a turn is not independent
                                      of the other turn of its own question)
  win / tie / loss                  - direction, per question
  turns changed                     - the honest headline for a targeted change
  hit@6 McNemar-style b/c counts    - discordant pairs only, for retrieval

A paired CI that straddles zero means "this experiment cannot resolve the
effect", NOT "the effect is zero" - a distinction the old noise-floor phrasing
blurred, and the reason "confirmed no harm" was the wrong words.

Usage:
    python eval/compare.py results_A.json results_B.json
    python eval/compare.py results_A.json results_B.json --metric hit_at_6
    python eval/compare.py A.json B.json --bootstrap 20000 --json out.json
"""

import argparse
import json
import pathlib
import random
import statistics as st

# Resamples questions, so a run is reproducible; the CI is an estimate either
# way and a wandering one invites re-rolling until it reads well.
SEED = 20260811



def load(path: pathlib.Path):
    """-> {turn_key: {"score": float|None, "hit": bool|None, "question": str}}"""
    data = json.loads(path.read_text())
    out: dict[str, dict] = {}
    rows = data if isinstance(data, list) else [data]
    for qi, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        for turn in ("primary", "follow_up"):
            sub = item.get(turn)
            if not isinstance(sub, dict):
                continue
            # The TURN's own question text is the identity. source_url is not
            # unique - three documents carry two questions each in the spanning
            # set - and keying on it silently collapsed 20 turns into 14 the
            # first time this ran. Falling back to the row index keeps arms
            # aligned when a file has no question text, but only positionally.
            question = sub.get("question") or item.get("source_url") or f"__row{qi}"
            judge = sub.get("judge") or {}
            retr = sub.get("retrieval") or {}
            score = judge.get("score")
            rank = retr.get("rank")
            out[f"{question}||{turn}"] = {
                "score": float(score) if isinstance(score, (int, float)) else None,
                # recomputed from rank, never trusted from the stored flag:
                # hit_at_6 was written uncapped until 2026-08-11 and is wrong in
                # older files (7 known false hits)
                "hit": (isinstance(rank, int) and rank <= 6) if rank is not None else None,
                "question": str(question),
            }
    return out


def bootstrap_ci(diffs_by_question: dict[str, list[float]], n: int, alpha=0.05):
    """Resample QUESTIONS with replacement, not turns. The two turns of one
    question share a retrieval and a topic, so treating them as independent
    would understate the interval."""
    keys = list(diffs_by_question)
    if not keys:
        return (float("nan"), float("nan"))
    rng = random.Random(SEED)
    means = []
    for _ in range(n):
        picked = [rng.choice(keys) for _ in keys]
        vals = [d for k in picked for d in diffs_by_question[k]]
        if vals:
            means.append(sum(vals) / len(vals))
    if not means:
        return (float("nan"), float("nan"))
    means.sort()
    lo = means[int(len(means) * alpha / 2)]
    hi = means[min(len(means) - 1, int(len(means) * (1 - alpha / 2)))]
    return (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("treatment")
    ap.add_argument("--metric", choices=("score", "hit_at_6"), default="score")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--json", help="also write the result here")
    args = ap.parse_args()

    a, b = pathlib.Path(args.baseline), pathlib.Path(args.treatment)
    for p in (a, b):
        if not p.is_file():
            print(f"missing: {p}")
            return 2
    A, B = load(a), load(b)

    field = "score" if args.metric == "score" else "hit"
    common = [k for k in A if k in B and A[k][field] is not None and B[k][field] is not None]
    only_a, only_b = len(A) - len(common), len(B) - len(common)

    print(f"\n  baseline : {a.name}   ({len(A)} turns)")
    print(f"  treatment: {b.name}   ({len(B)} turns)")
    print(f"  metric   : {args.metric}")
    print(f"  paired   : {len(common)} turns" +
          (f"   [unpaired: {only_a} baseline-only, {only_b} treatment-only]"
           if (only_a or only_b) else ""))
    if not common:
        print("\n  NOTHING TO COMPARE - the two files share no scorable turns.")
        return 1

    if args.metric == "hit_at_6":
        # discordant pairs are the whole story for a binary metric
        b_only = sum(1 for k in common if B[k]["hit"] and not A[k]["hit"])
        a_only = sum(1 for k in common if A[k]["hit"] and not B[k]["hit"])
        same = len(common) - b_only - a_only
        print(f"\n  gained by treatment : {b_only}")
        print(f"  lost by treatment   : {a_only}")
        print(f"  unchanged           : {same}")
        print(f"  net                 : {b_only - a_only:+d} turns")
        if b_only + a_only == 0:
            print("\n  IDENTICAL on every paired turn - this experiment moved nothing.")
        result = {"metric": "hit_at_6", "gained": b_only, "lost": a_only,
                  "unchanged": same, "net": b_only - a_only, "paired": len(common)}
    else:
        diffs, by_q = [], {}
        wins = ties = losses = 0
        for k in common:
            d = B[k]["score"] - A[k]["score"]
            diffs.append(d)
            by_q.setdefault(A[k]["question"], []).append(d)
            if d > 0:
                wins += 1
            elif d < 0:
                losses += 1
            else:
                ties += 1
        lo, hi = bootstrap_ci(by_q, args.bootstrap)
        changed = wins + losses
        mean = sum(diffs) / len(diffs)
        print(f"\n  mean paired diff  {mean:+.3f}")
        print(f"  median paired diff{st.median(diffs):+.3f}")
        print(f"  sd                {st.pstdev(diffs):.3f}")
        print(f"  95% bootstrap CI  [{lo:+.3f}, {hi:+.3f}]   ({args.bootstrap} resamples of "
              f"{len(by_q)} questions)")
        print(f"\n  win / tie / loss  {wins} / {ties} / {losses}")
        print(f"  TURNS CHANGED     {changed} of {len(common)}"
              f"   ({changed / len(common) * 100:.0f}%)")
        if changed == 0:
            print("\n  IDENTICAL on every paired turn.")
        elif lo <= 0 <= hi:
            print("\n  The CI straddles zero: this experiment CANNOT RESOLVE the effect.")
            print("  That is not evidence of no effect - report the turns-changed count,")
            print("  and if the mechanism targets a known subset, evaluate that subset.")
        else:
            print(f"\n  CI excludes zero: a real {'improvement' if lo > 0 else 'regression'} "
                  f"at this sample size.")
        result = {"metric": "score", "mean": mean, "median": st.median(diffs),
                  "ci95": [lo, hi], "wins": wins, "ties": ties, "losses": losses,
                  "changed": changed, "paired": len(common), "questions": len(by_q)}

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, indent=1))
        print(f"\n  wrote {args.json}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
