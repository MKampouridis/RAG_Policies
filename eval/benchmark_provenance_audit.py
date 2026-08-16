#!/usr/bin/env python3
"""Measure how far the eval set has drifted from the corpus it was written against.

WHY
Each test item bundles three things written at the same time from one snapshot
of the documents: the question, the expected answer, and the gold `source_url`.
The corpus has since been re-ingested - 5 new documents and ~20 changed - and
nobody revisited the test set. It is a stale cache of the corpus, the same bug
class as the ColBERT embedding cache that silently cost 5 turns of hit@6.

This measures the drift BEFORE anyone spends a day on judgement calls, because
"9 of 148 gold URLs are superseded" is only one of the ways an item can rot and
the others are unmeasured.

Four checks per item, cheapest first:

  MISSING_DOC   gold `source_url` is not in the corpus at all
  NOT_CURRENT   present, but `is_current` is False - a newer edition exists, so
                retrieval returning the CURRENT document scores as a MISS
  TEXT_DRIFT    keyphrases (or salient terms from the expected answer) no longer
                appear in the gold document's indexed text
  OK            nothing detected

TEXT_DRIFT is a weaker signal than the other two: a reference can be reworded
without being wrong, and keyphrase matching is exact-substring. It is reported
as "worth a human look", never as "defective" - that distinction is the one the
round-8 review caught me collapsing.

Usage:
    PYTHONPATH=. python eval/benchmark_provenance_audit.py [questions.json]
Writes eval/benchmark_provenance_audit.json
"""

import json
import pathlib
import re
import sys
from collections import Counter


def salient(text: str, n: int = 6) -> list[str]:
    """Distinctive terms from a reference answer. Numbers carry the checkable
    content in policy text; capitalised phrases carry the named entities."""
    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", text or "")
    caps = re.findall(r"\b[A-Z][a-zA-Z]{3,}(?:\s+[A-Z][a-zA-Z]{3,})?\b", text or "")
    out: list[str] = []
    for x in nums + caps:
        if x not in out:
            out.append(x)
    return out[:n]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = pathlib.Path(args[0] if args else "eval/questions_regression.json")
    items = json.loads(path.read_text())

    from src import ingest
    coll = ingest._get_collection()

    text_by_doc: dict[str, str] = {}
    current: dict[str, bool] = {}
    offset = 0
    while True:
        got = coll.get(limit=5000, offset=offset, include=["documents", "metadatas"])
        ids = got.get("ids") or []
        if not ids:
            break
        for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
            u = (meta or {}).get("source_url")
            if not u:
                continue
            text_by_doc[u] = text_by_doc.get(u, "") + " " + (doc or "")
            if u not in current:
                current[u] = (meta or {}).get("is_current")
        offset += len(ids)
    print(f"\n  corpus: {len(text_by_doc)} documents, {offset} chunks")
    print(f"  set   : {path.name}, {len(items)} items\n")

    rows = []
    for it in items:
        q = it.get("question")
        gold = it.get("source_url")
        ref = it.get("expected_answer") or ""
        if not (q and gold):
            rows.append({"question": (q or "")[:110], "source_url": gold,
                         "verdict": "NO_GOLD", "detail": "item has no gold document"})
            continue
        if gold not in text_by_doc:
            rows.append({"question": q[:110], "source_url": gold,
                         "verdict": "MISSING_DOC", "detail": "not in the corpus"})
            continue
        if current.get(gold) is False:
            rows.append({"question": q[:110], "source_url": gold,
                         "verdict": "NOT_CURRENT",
                         "detail": "a newer edition exists; returning it scores as a miss"})
            continue
        keys = it.get("keyphrases") or salient(ref)
        low = text_by_doc[gold].lower()
        found = [k for k in keys if str(k).lower() in low]
        if keys and not found:
            rows.append({"question": q[:110], "source_url": gold, "verdict": "TEXT_DRIFT",
                         "detail": f"0 of {len(keys)} key terms found",
                         "missing": [str(k) for k in keys][:5]})
        elif keys and len(found) < len(keys):
            rows.append({"question": q[:110], "source_url": gold, "verdict": "PARTIAL_DRIFT",
                         "detail": f"{len(found)} of {len(keys)} key terms found",
                         "missing": [str(k) for k in keys if k not in found][:5]})
        else:
            rows.append({"question": q[:110], "source_url": gold, "verdict": "OK",
                         "detail": f"{len(found)} of {len(keys)} key terms found"})

    counts = Counter(r["verdict"] for r in rows)
    order = ["MISSING_DOC", "NOT_CURRENT", "TEXT_DRIFT", "PARTIAL_DRIFT", "OK", "NO_GOLD"]
    print(f"  {'verdict':<16}{'n':>5}{'share':>9}")
    for v in order:
        if counts.get(v):
            print(f"  {v:<16}{counts[v]:>5}{counts[v]/len(rows)*100:>8.0f}%")

    hard = [r for r in rows if r["verdict"] in ("MISSING_DOC", "NOT_CURRENT")]
    print(f"\n  DEFINITELY BROKEN (grades against a document that is gone or superseded): {len(hard)}")
    for r in hard:
        print(f"    {r['source_url'].rsplit('/', 1)[-1][:56]}")
        print(f"      {r['question'][:88]}")

    soft = [r for r in rows if r["verdict"] == "TEXT_DRIFT"]
    print(f"\n  WORTH A HUMAN LOOK (no key term still present in the gold document): {len(soft)}")
    for r in soft[:10]:
        print(f"    {r['source_url'].rsplit('/', 1)[-1][:50]}  missing {r.get('missing')}")

    out = pathlib.Path("eval/benchmark_provenance_audit.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n  wrote {out}")
    print("\n  NOTE: TEXT_DRIFT is a weak signal - a reference can be reworded without")
    print("  being wrong, and matching is exact-substring. It flags items for a human,")
    print("  it does not condemn them.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
