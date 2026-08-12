#!/usr/bin/env python3
"""Retrieval baseline over the 151-question regression set. No generation.

The last end-to-end eval on record is 2026-07-24. Since then the corpus was
re-ingested, the stale ColBERT cache was removed (+5 hit@6), concise became the
default and the partner scope switch replaced the heuristic. Nothing in the
ledger describes the current system.

Generation is expensive; retrieval is free and deterministic. This scores hit@6
over the merged regression set so the retrieval half is baselined now, and the
end-to-end half can follow when someone decides what it should cost.

Usage:
    PYTHONPATH=. RAG_DETERMINISTIC=1 python eval/regression_retrieval_baseline.py
Writes eval/regression_retrieval_baseline.json
"""
import json
import pathlib
import time
from collections import Counter


def main() -> int:
    items = json.loads(pathlib.Path("eval/questions_regression.json").read_text())
    from src import rag

    rows, t0 = [], time.time()
    for i, it in enumerate(items, 1):
        q, gold = it.get("question"), it.get("source_url")
        if not (q and gold):
            continue
        res, rq = rag.retrieve(q, [])
        urls = [m.get("source_url") for m in res.get("metadatas", [[]])[0]]
        rank = urls.index(gold) + 1 if gold in urls else None
        rows.append({"question": q, "source_url": gold, "rank": rank,
                     "hit_at_6": rank is not None and rank <= 6,
                     "strata": it.get("_strata", []), "source": it.get("_source")})
        if i % 25 == 0:
            print(f"  {i}/{len(items)}  hit@6 so far "
                  f"{sum(r['hit_at_6'] for r in rows)}/{len(rows)}  "
                  f"({time.time()-t0:.0f}s)")

    hits = sum(r["hit_at_6"] for r in rows)
    print(f"\n  TOTAL hit@6: {hits}/{len(rows)} ({hits/len(rows)*100:.1f}%)\n")

    # by stratum, because an aggregate hides which question types are weak
    tags = Counter(t for r in rows for t in r["strata"] if not t.startswith("src:"))
    print(f"  {'stratum':<24}{'hit@6':>10}")
    for tag, n in tags.most_common():
        sub = [r for r in rows if tag in r["strata"]]
        h = sum(r["hit_at_6"] for r in sub)
        print(f"  {tag:<24}{h:>4}/{len(sub):<5}({h/len(sub)*100:.0f}%)")

    pathlib.Path("eval/regression_retrieval_baseline.json").write_text(
        json.dumps(rows, indent=1))
    print(f"\n  wrote eval/regression_retrieval_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
