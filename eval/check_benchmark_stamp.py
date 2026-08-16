#!/usr/bin/env python3
"""Refuse to trust the eval set if the corpus has moved under it.

The eval set bundles gold documents chosen from one corpus snapshot. After a
re-ingest those documents can be superseded, and retrieval then gets scored a
MISS for returning the CURRENT edition - measured at 9 of 148 items before the
round-8 corrections.

Belongs in the post-ingest sequence beside audit_family_aliases.py,
stale_index_audit.py and colbert_index_drift.py. Exit code 1 on mismatch, so a
script can stop.
"""
import json
import pathlib
import sys

SET = pathlib.Path("eval/questions_regression.json")


def main() -> int:
    if not SET.is_file():
        print("no regression set to check")
        return 0
    rows = json.loads(SET.read_text())
    stamp = next((r for r in rows if isinstance(r, dict) and r.get("_stamp")), None)
    if not stamp:
        print(f"  {SET.name} carries NO corpus stamp - provenance unknown.")
        print("  Run eval/benchmark_provenance_audit.py, then stamp it.")
        return 1
    from src.ingest import read_corpus_version
    now = read_corpus_version()
    was = stamp.get("corpus_version")
    print(f"  eval set verified against : {str(was)[:16]}")
    print(f"  corpus is now             : {str(now)[:16]}")
    if was == now:
        print("  in sync")
        return 0
    print("\n  STALE. The corpus has changed since this eval set was verified.")
    print("  Gold documents may be superseded, which scores correct retrieval as a miss.")
    print("  Run: PYTHONPATH=. python eval/benchmark_provenance_audit.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
