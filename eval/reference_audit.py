#!/usr/bin/env python3
"""Check whether each question's REFERENCE answer is supported by its own gold
document.

WHY: judge calibration (Round 42) found the judge scores agreement-with-
reference rather than correctness, and three of the largest human/judge gaps
were cases where the reference itself was wrong - "the reference text was wrong
but the answer from the system was right". Round 29 independently found 5 of 11
misses had gold keyphrases present in NO current document. Both point at the
references, not the judge.

For each item: take the keyphrases (or salient terms from the expected answer)
and check whether they appear in the gold document's indexed text.

  SUPPORTED   - all keyphrases found in the gold document
  PARTIAL     - some found
  UNSUPPORTED - none found; the reference cannot be checked against its own
                source, so any judge scoring against it is scoring noise

This is a corpus-truth check, not a judgement about answer quality. A reference
can be well-written and still unsupported if the document was superseded.

Usage:
    PYTHONPATH=. python eval/reference_audit.py [questions.json] [--limit N]
"""
import json
import pathlib
import re
import sys


def salient(text: str, n: int = 8) -> list[str]:
    """Distinctive terms from a reference answer: numbers and capitalised
    phrases carry the checkable content in policy text."""
    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", text or "")
    caps = re.findall(r"\b[A-Z][a-zA-Z]{3,}(?:\s+[A-Z][a-zA-Z]{3,})?\b", text or "")
    out = []
    for x in nums + caps:
        if x not in out:
            out.append(x)
    return out[:n]


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    path = pathlib.Path(argv[0] if argv else "eval/questions_regression.json")
    items = json.loads(path.read_text())
    if limit:
        items = items[:limit]

    # Read from CHROMA, the live production index. The first version of this
    # read colbert_docs.json - the ColBERT cache - which is a 20,477-chunk
    # snapshot from 2026-07-21 against Chroma's 21,709, and whose index
    # directory has since been deleted. Auditing references against a stale
    # copy of the corpus would answer the wrong question in exactly the way
    # this audit exists to detect.
    from src import ingest
    coll = ingest._get_collection()
    by_doc: dict[str, str] = {}
    offset, BATCH = 0, 5000
    while True:
        got = coll.get(limit=BATCH, offset=offset, include=["documents", "metadatas"])
        ids = got.get("ids") or []
        if not ids:
            break
        for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
            u = (meta or {}).get("source_url")
            if u:
                by_doc[u] = by_doc.get(u, "") + " " + (doc or "")
        offset += len(ids)
    print(f"  corpus: {len(by_doc)} documents, {offset} chunks (live Chroma index)")

    counts = {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0, "NO_DOC": 0}
    rows = []
    for it in items:
        gold = it.get("source_url")
        ref = it.get("expected_answer") or ""
        keys = it.get("keyphrases") or salient(ref)
        text = by_doc.get(gold)
        if not text:
            counts["NO_DOC"] += 1
            continue
        low = text.lower()
        found = [k for k in keys if str(k).lower() in low]
        verdict = ("SUPPORTED" if keys and len(found) == len(keys)
                   else "PARTIAL" if found else "UNSUPPORTED")
        counts[verdict] += 1
        rows.append({"question": it.get("question", "")[:110], "source_url": gold,
                     "verdict": verdict, "found": len(found), "checked": len(keys),
                     "missing": [k for k in keys if k not in found][:5]})

    total = sum(counts.values())
    print(f"\n  checked {total} items from {path.name}\n")
    for k in ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "NO_DOC"):
        print(f"  {k:<14}{counts[k]:>4}  ({counts[k]/max(total,1)*100:.0f}%)")

    bad = [r for r in rows if r["verdict"] == "UNSUPPORTED"]
    if bad:
        print(f"\n  UNSUPPORTED references - the reference cannot be found in its own"
              f" gold document ({len(bad)}):")
        for r in bad[:12]:
            print(f"    {r['source_url'].rsplit('/',1)[-1][:44]}")
            print(f"      q: {r['question'][:88]}")
            print(f"      looked for: {r['missing']}")
    pathlib.Path("eval/reference_audit_result.json").write_text(json.dumps(rows, indent=1))
    print(f"\n  wrote eval/reference_audit_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
