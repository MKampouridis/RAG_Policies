#!/usr/bin/env python3
"""Classify the retrieval misses instead of guessing at them.

WHY
The "11% far from anything retrieved" figure (Round 8p) was produced ad hoc and
never broken down, and every mechanism tried against it has been a RANKER. That
is only the right family of fix if the gold document is actually in the
candidate pool. If it never enters the pool, no reranker can rescue it and the
whole falsified list - cross-encoders, ColBERT, LLM reranking, identity-salient
passages - was aimed at the wrong stage.

THE DECISIVE QUESTION, asked first
`FETCH_POOL_MULTIPLIER` has only ever been swept DOWNWARD (8, 4, 2). This
sweeps it UP. For each missed turn, does the gold document enter the candidate
pool at 8x, 16x, or 32x?

  ENTERS at 8x   -> already in the pool; a RANKING failure. A better ranker
                    could in principle fix it.
  ENTERS at 16/32 -> a POOL-SIZE failure. Recall, not ranking.
  NEVER enters   -> a REPRESENTATION failure. Neither dense nor BM25 puts the
                    document anywhere near this query, and no reranker at any
                    pool size can help. This is the class worth knowing about,
                    because it retroactively explains why ranking work kept
                    failing.

Reads a replay file (rank per turn), takes every turn whose gold is not in the
top 6, and re-runs retrieval at each multiplier.

Usage:
    PYTHONPATH=. python eval/far_miss_taxonomy.py [replay.json] [--limit N]
Writes eval/far_miss_taxonomy_result.json
"""

import json
import pathlib
import sys

DEFAULT_REPLAY = "eval/retrieval_replay_partnerfix.json"
MULTIPLIERS = (8, 16, 32)
OUT = pathlib.Path("eval/far_miss_taxonomy_result.json")


def main() -> int:
    argv = sys.argv[1:]
    limit = None
    if "--limit" in argv:
        i = argv.index("--limit")
        limit = int(argv[i + 1])
        del argv[i:i + 2]          # drop BOTH, or the value reads as the path
    args = [a for a in argv if not a.startswith("--")]
    replay = pathlib.Path(args[0] if args else DEFAULT_REPLAY)
    if not replay.is_file():
        print(f"missing replay file: {replay}")
        return 2

    rows = json.loads(replay.read_text())
    misses = [r for r in rows
              if not (isinstance(r.get("rank"), int) and r["rank"] <= 6)]
    if limit:
        misses = misses[:limit]
    print(f"\n  replay      : {replay.name} ({len(rows)} turns)")
    print(f"  misses      : {len(misses)}")
    print(f"  multipliers : {MULTIPLIERS}\n")

    from src import rag

    results = []
    for i, r in enumerate(misses, 1):
        gold = r.get("source_url")
        query = r.get("query") or ""
        entered_at = None
        depth = None
        for mult in MULTIPLIERS:
            # patched per iteration rather than via env, so one process can
            # sweep; retrieve() reads the module global at call time
            rag.FETCH_POOL_MULTIPLIER = mult
            try:
                res, _ = rag.retrieve(query, [])
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(misses)}] ERROR {type(exc).__name__}")
                break
            pool = rag._LAST_CANDIDATE_POOL or {}
            urls = [m.get("source_url") for m in pool.get("metadatas", [[]])[0]]
            if gold in urls:
                entered_at = mult
                depth = urls.index(gold) + 1
                break
        rag.FETCH_POOL_MULTIPLIER = MULTIPLIERS[0]

        verdict = ("RANKING" if entered_at == 8 else
                   "POOL_SIZE" if entered_at else "REPRESENTATION")
        results.append({"source_url": gold, "turn": r.get("turn"),
                        "query": query[:160], "entered_at": entered_at,
                        "pool_depth": depth, "verdict": verdict})
        print(f"  [{i}/{len(misses)}] {verdict:<15} "
              f"{'enters at ' + str(entered_at) + 'x, depth ' + str(depth) if entered_at else 'never enters'}"
              f"   {(gold or '').rsplit('/', 1)[-1][:40]}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n  ---- summary ----")
    for k in ("RANKING", "POOL_SIZE", "REPRESENTATION"):
        n = counts.get(k, 0)
        pct = (n / len(results) * 100) if results else 0
        print(f"  {k:<16}{n:>4}  ({pct:.0f}% of misses)")
    if counts.get("REPRESENTATION", 0) > counts.get("RANKING", 0):
        print("\n  Most misses never enter the pool at ANY size tested.")
        print("  Ranking work cannot address these - the falsified reranker list")
        print("  was aimed at the wrong stage.")
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\n  wrote {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
