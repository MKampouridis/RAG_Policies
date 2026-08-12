#!/usr/bin/env python3
"""Derive a canary set: turns that pass COMFORTABLY today and must never break.

WHY: a regression set tells you the average moved; a canary set tells you
something that was solidly working has stopped. Those need different questions.
A turn scraping in at rank 6 will flip on noise and cry wolf; a turn that has
been at rank 1-3 flipping to a miss is a genuine alarm.

Selection: gold at rank <= 3 in the reference replay. Not hand-picked, so it
cannot be gamed, and it is regenerable after any deliberate re-baseline.

Usage:
    python eval/build_canary_set.py [replay.json] [--write]
"""
import json
import pathlib
import sys

DEFAULT = "eval/retrieval_replay_cache_off.json"
MAX_RANK = 3


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    replay = pathlib.Path(argv[0] if argv else DEFAULT)
    if not replay.is_file():
        print(f"missing: {replay}")
        return 2
    rows = json.loads(replay.read_text())
    canary = [r for r in rows if isinstance(r.get("rank"), int) and r["rank"] <= MAX_RANK]
    print(f"\n  reference : {replay.name} ({len(rows)} turns)")
    print(f"  canary    : {len(canary)} turns with gold at rank <= {MAX_RANK}")
    by_rank = {}
    for r in canary:
        by_rank[r["rank"]] = by_rank.get(r["rank"], 0) + 1
    for k in sorted(by_rank):
        print(f"     rank {k}: {by_rank[k]}")
    out = [{"results_file": r["results_file"], "source_url": r["source_url"],
            "turn": r["turn"], "query": r.get("query"), "reference_rank": r["rank"]}
           for r in canary]
    if "--write" in sys.argv:
        p = pathlib.Path("eval/canary_set.json")
        p.write_text(json.dumps(out, indent=1))
        print(f"\n  wrote {p}")
        print("  check it with: PYTHONPATH=. python eval/check_canary.py")
    else:
        print("\n  (pass --write to save)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
