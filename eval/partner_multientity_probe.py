#!/usr/bin/env python3
"""Targeted regression test for the multi-entity partner leak.

WHY A DEDICATED PROBE
`retrieval_replay` cannot see this defect: 0 of its 160 turns name >=2
departments, so it reports "no change" for something it structurally cannot
measure. Sets 5 and 6 cannot see it either - set 5's multi-entity questions
surface no partner documents, and set 6's questions NAME a partner, so the
exclusion correctly does not apply.

The defect needs a question that names >=2 departments where at least one has
NO department metadata. Those aliases fall through to a retrieval filtered on
`is_current` alone, which re-admits partner chunks that were excluded from the
main pool. Only three aliases qualify today: sociology, criminology, and
philosophical/historical/interdisciplinary studies.

Usage:
    RAG_MULTI_ENTITY_PARTNER_RECHECK=0 PYTHONPATH=. python eval/partner_multientity_probe.py
    RAG_MULTI_ENTITY_PARTNER_RECHECK=1 PYTHONPATH=. python eval/partner_multientity_probe.py

Expected: partner chunks with the flag OFF, zero with it ON.
"""
import json
import os

from src import rag
from src.entities import detect_departments
from src.rag import _is_partner_institution as is_partner

QUESTIONS = [
    "What progression rules apply to Sociology and Criminology undergraduates?",
    "Do Sociology and Psychology students have the same reassessment rules?",
    "For Criminology and CSEE students, what happens after a failed module?",
    "How do Sociology and Criminology Masters students qualify for a Merit?",
]


def main() -> int:
    flag = os.environ.get("RAG_MULTI_ENTITY_PARTNER_RECHECK", "1")
    total = 0
    print(f"\n  RAG_MULTI_ENTITY_PARTNER_RECHECK={flag}\n")
    for q in QUESTIONS:
        res, _ = rag.retrieve(q, [])
        metas = res.get("metadatas", [[]])[0]
        n = sum(1 for m in metas if is_partner(m))
        total += n
        print(f"  entities={len(detect_departments(q))}  partner_chunks={n}  {q[:56]}")
    print(f"\n  TOTAL partner chunks served: {total}")
    print("  (expected: >0 with the flag OFF, 0 with it ON)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
