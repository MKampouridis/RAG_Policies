#!/usr/bin/env python3
"""Re-run the canary turns and report anything that stopped working.

Not a quality metric - an alarm. Any canary turn that no longer returns its
gold document in the top 6 is a regression in something that was solidly
working, and is worth stopping for even when the aggregate looks fine.

Usage:
    PYTHONPATH=. RAG_DETERMINISTIC=1 python eval/check_canary.py
Exit code 1 if any canary broke.
"""
import json
import pathlib
import sys


def main() -> int:
    p = pathlib.Path("eval/canary_set.json")
    if not p.is_file():
        print("no canary set - run eval/build_canary_set.py --write first")
        return 2
    canary = json.loads(p.read_text())
    from src import rag

    broken = []
    for i, c in enumerate(canary, 1):
        res, _ = rag.retrieve(c["query"] or "", [])
        urls = [m.get("source_url") for m in res.get("metadatas", [[]])[0]][:6]
        if c["source_url"] not in urls:
            broken.append(c)
        if i % 20 == 0:
            print(f"  {i}/{len(canary)} checked, {len(broken)} broken")
    print(f"\n  canary turns : {len(canary)}")
    print(f"  BROKEN       : {len(broken)}")
    for c in broken:
        print(f"     was rank {c['reference_rank']}: {c['source_url'].rsplit('/', 1)[-1][:52]}")
        print(f"        {(c['query'] or '')[:88]}")
    if broken:
        print("\n  Something that was solidly working has stopped. Investigate before shipping.")
        return 1
    print("\n  all canaries still pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
