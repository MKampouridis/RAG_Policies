#!/usr/bin/env python3
"""Small production-config check reporting RATES, not mean judge scores.

WHY THIS EXISTS
Every ledger baseline is measured on a LOCAL deterministic generator, while
production runs Sonnet. That is right for retrieval - it is identical in both -
but wrong for anything prompt-shaped. `USER_FACING_LANGUAGE`, the detail level,
the disclosure and the multi-entity coverage rule are all GENERATOR behaviours,
and gemma3's response to an instruction says nothing about Sonnet's. The
plumbing leak (Round: USER_FACING_LANGUAGE) is exactly this: four metrics
scored it identically before and after, because the facts were right both times.

So this runs a small set against PRODUCTION and reports binary rates:

    leaked plumbing vocabulary       yes/no
    asked the user to supply a doc   yes/no
    cited at least one source        yes/no
    abstained                        yes/no
    named every entity the question listed

Rates over n samples are far more stable than a mean of 1-5 judge scores on 20
turns, and they measure things a user would actually notice. There is no judge
here at all - every check is deterministic string work, so the only variance is
the generator's.

Usage:
    RAG_API_BASE=http://127.0.0.1:8000 python eval/cloud_smoke.py [--samples 3]
"""
import json
import os
import re
import sys

import requests

BASE = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8000")

# phrases that reveal retrieval machinery to a user who supplied nothing
PLUMBING = re.compile(
    r"\b(the (provided |supplied )?context\b|excerpts?\b|the documents provided\b"
    r"|you (have )?provided\b|paste\b|share (the|your) document)", re.I)
ASK_FOR_DOCS = re.compile(
    r"\b(please (share|provide|paste)|if you (have|can) (the|any)|could you (share|provide))", re.I)

CASES = [
    {"q": "What are the pass marks for a PGT Merit and Distinction?", "entities": []},
    {"q": "Which Masters programmes in CSEE, MSAS and Psychology are accredited?",
     "entities": ["csee", "msas", "psycholog"]},
    {"q": "In which cases is an independent chair required for PGR examinations?", "entities": []},
    {"q": "What is the University of Essex policy on time travel expenses for staff?",
     "entities": [], "expect_abstain": True},
    {"q": "How many credits are needed to pass year one?", "entities": []},
    {"q": "What support is available for students with disabilities?", "entities": []},
]
ABSTAIN = re.compile(r"\b(don't (cover|have)|do not (cover|have)|no (policy|document)|couldn't find|cannot find)\b", re.I)


def ask(q: str) -> dict:
    cid = requests.post(f"{BASE}/api/conversations", json={"title": "__smoke__"}).json()["id"]
    try:
        r = requests.post(f"{BASE}/api/conversations/{cid}/messages",
                          json={"content": q, "detail": "default"}, timeout=300).json()
        return r
    finally:
        requests.delete(f"{BASE}/api/conversations/{cid}")


def main() -> int:
    samples = 3
    if "--samples" in sys.argv:
        samples = int(sys.argv[sys.argv.index("--samples") + 1])

    counts = {"leaked": 0, "asked_for_docs": 0, "no_sources": 0,
              "missed_entity": 0, "abstain_ok": 0, "abstain_n": 0, "n": 0}
    failures = []
    for case in CASES:
        for s in range(samples):
            try:
                r = ask(case["q"])
            except Exception as exc:  # noqa: BLE001
                print(f"  request failed: {exc}")
                continue
            a = r.get("answer", "")
            counts["n"] += 1
            if PLUMBING.search(a):
                counts["leaked"] += 1
                failures.append(("leaked", case["q"][:44], PLUMBING.search(a).group(0)))
            if ASK_FOR_DOCS.search(a):
                counts["asked_for_docs"] += 1
                failures.append(("asked_for_docs", case["q"][:44], ASK_FOR_DOCS.search(a).group(0)))
            if not r.get("sources"):
                counts["no_sources"] += 1
            missing = [e for e in case["entities"] if e not in a.lower()]
            if missing:
                counts["missed_entity"] += 1
                failures.append(("missed_entity", case["q"][:44], ",".join(missing)))
            if case.get("expect_abstain"):
                counts["abstain_n"] += 1
                if ABSTAIN.search(a):
                    counts["abstain_ok"] += 1
            print(f"  [{counts['n']}] {case['q'][:46]:<46} {len(a):>5} chars")

    n = max(counts["n"], 1)
    print(f"\n  samples: {counts['n']}  ({len(CASES)} questions x {samples})\n")
    print(f"  {'check':<26}{'rate':>10}")
    print(f"  {'leaked plumbing words':<26}{counts['leaked']/n*100:>9.0f}%")
    print(f"  {'asked user for documents':<26}{counts['asked_for_docs']/n*100:>9.0f}%")
    print(f"  {'answered with no sources':<26}{counts['no_sources']/n*100:>9.0f}%")
    print(f"  {'dropped a named entity':<26}{counts['missed_entity']/n*100:>9.0f}%")
    if counts["abstain_n"]:
        print(f"  {'abstained when it should':<26}{counts['abstain_ok']/counts['abstain_n']*100:>9.0f}%")
    if failures:
        print("\n  failures:")
        for kind, q, detail in failures[:12]:
            print(f"    {kind:<16} {q}   -> {detail!r}")
    json.dump({"counts": counts, "failures": failures},
              open("eval/cloud_smoke_result.json", "w"), indent=1)
    print("\n  wrote eval/cloud_smoke_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
