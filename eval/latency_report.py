#!/usr/bin/env python3
"""Latency as a USER would experience it, not as a developer's median.

WHY: the headline "9.8s median" was a median over one heavy user's traffic. A
colleague asking two questions a day is cold nearly every time and gets a
different number entirely. Reporting one median hid that (Round 11).

Reports four populations:
  session-first  - first request after >=10 min idle. What a returning user gets.
  in-session     - a request within 10 min of the previous one.
  overall        - every request.
  by stage       - where the time goes.

Usage:
    python eval/latency_report.py [data/latency.jsonl] [--since 2026-08-12]

ALWAYS scope with --since after a configuration change. The log spans every
configuration this system has had, and an unscoped median describes a machine
that never existed.
"""
import json
import pathlib
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime

IDLE_GAP = 600.0


def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--since" in sys.argv:
        i = sys.argv.index("--since")
        args = [a for a in args if a != sys.argv[i + 1]]
    path = pathlib.Path(args[0] if args else "data/latency.jsonl")
    if not path.is_file():
        print(f"missing: {path}")
        return 2
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("seconds") is not None]
    rows.sort(key=lambda r: r["ts"])

    # The log spans every configuration this system has had. Mixing them
    # produces a median of a machine that never existed - the pre-warmup,
    # pre-cache-fix era dominates by volume. --since scopes it to one config.
    since = None
    for i, a in enumerate(sys.argv):
        if a == "--since" and i + 1 < len(sys.argv):
            since = sys.argv[i + 1]
    if since:
        rows = [r for r in rows if r["ts"] >= since]
        print(f"\n  scoped to ts >= {since}")

    totals = [r for r in rows if r["stage"] == "answer_total"]
    if not totals:
        print("no answer_total rows yet")
        return 0

    first, insession = [], []
    prev = None
    for r in totals:
        t = datetime.fromisoformat(r["ts"])
        gap = (t - prev).total_seconds() if prev else 9e9
        (first if gap > IDLE_GAP else insession).append(r["seconds"])
        prev = t

    print(f"\n  requests: {len(totals)}   window: {totals[0]['ts'][:16]} -> {totals[-1]['ts'][:16]}\n")
    print(f"  {'population':<16}{'n':>5}{'p50':>8}{'p90':>8}{'max':>8}")
    for label, v in (("session-first", first), ("in-session", insession),
                     ("overall", [r['seconds'] for r in totals])):
        if v:
            print(f"  {label:<16}{len(v):>5}{st.median(v):>8.1f}{pct(v,0.9):>8.1f}{max(v):>8.1f}")

    print("\n  LEAD WITH session-first: it is what a returning user actually waits.\n")

    by = defaultdict(list)
    for r in rows:
        by[r["stage"]].append(r["seconds"])
    print(f"  {'stage':<18}{'n':>5}{'median':>9}")
    for s in sorted(by, key=lambda k: -st.median(by[k])):
        print(f"  {s:<18}{len(by[s]):>5}{st.median(by[s]):>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
