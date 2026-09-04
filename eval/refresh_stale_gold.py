#!/usr/bin/env python3
"""Repair NOT_CURRENT items in an eval set after a re-ingest supersedes gold
documents (e.g. an academic-year rollover). Companion to
benchmark_provenance_audit.py, which only MEASURES drift.

WHAT THIS DOES vs WHAT IT REFUSES TO DO
NOT_CURRENT is the one drift category this can safely auto-repair: the gold
document still exists, just under a newer edition, and document_family()
already groups yearly reissues together - the same relationship
audit_family_aliases.py relies on. So for each NOT_CURRENT item this finds the
CURRENT sibling in the same family and swaps source_url to it, but ONLY when
the item's keyphrases still match the new document's text. If they don't, the
content substantively changed between editions (e.g. a pass mark moved) and
swapping the URL while leaving a stale expected_answer would silently corrupt
the ground truth - worse than leaving it flagged. Those go to a "needs a
human" list instead, same as every PARTIAL_DRIFT and TEXT_DRIFT item: this
script never rewrites an expected_answer, because judging whether wording
drift reflects a real policy change is a domain call, not a mechanical one.

Usage:
    PYTHONPATH=. python eval/refresh_stale_gold.py [questions.json]
Writes <name>.json in place (after printing a summary) and
eval/refresh_stale_gold_report.json with the full per-item detail.
"""

import json
import pathlib
import sys

from eval.benchmark_provenance_audit import salient


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = pathlib.Path(args[0] if args else "eval/questions_regression.json")
    items = json.loads(path.read_text())

    from src import ingest
    from src.docid import document_family

    coll = ingest._get_collection()

    text_by_doc: dict[str, str] = {}
    current: dict[str, bool] = {}
    family_current: dict[str, list[str]] = {}  # family key -> current URLs
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
                if current[u]:
                    fam = document_family(u)
                    family_current.setdefault(fam, [])
                    if u not in family_current[fam]:
                        family_current[fam].append(u)
        offset += len(ids)
    print(f"\n  corpus: {len(text_by_doc)} documents, {offset} chunks\n")

    fixed, needs_human, unresolved_family = [], [], []
    for it in items:
        if not isinstance(it, dict) or "_stamp" in it:
            continue
        q, gold, ref = it.get("question"), it.get("source_url"), it.get("expected_answer") or ""
        if not (q and gold):
            continue
        if current.get(gold) is not False:
            continue  # only NOT_CURRENT is in scope here

        fam = document_family(gold)
        candidates = family_current.get(fam, [])
        if len(candidates) != 1:
            unresolved_family.append({"question": q[:110], "old_url": gold, "family": fam,
                                       "candidates": candidates})
            continue
        new_url = candidates[0]
        keys = it.get("keyphrases") or salient(ref)
        low = text_by_doc[new_url].lower()
        found = [k for k in keys if str(k).lower() in low]
        record = {"question": q[:110], "old_url": gold, "new_url": new_url,
                  "keyphrases": keys, "found": found}
        if keys and len(found) < len(keys):
            record["missing"] = [k for k in keys if k not in found]
            needs_human.append(record)
            continue
        it["source_url"] = new_url
        if it.get("source_title"):
            it["source_title"] = new_url.rsplit("/", 1)[-1]
        fixed.append(record)

    print(f"  auto-fixed (clean family match, keyphrases still present): {len(fixed)}")
    for r in fixed:
        print(f"    {r['old_url'].rsplit('/', 1)[-1]}  ->  {r['new_url'].rsplit('/', 1)[-1]}")

    print(f"\n  NEEDS A HUMAN (family match found, but content drifted - do not auto-swap): {len(needs_human)}")
    for r in needs_human:
        print(f"    {r['old_url'].rsplit('/', 1)[-1]}  ->  {r['new_url'].rsplit('/', 1)[-1]}  missing {r.get('missing')}")
        print(f"      {r['question']}")

    print(f"\n  NO CLEAN FAMILY MATCH (0 or >1 current siblings - needs a human to pick): {len(unresolved_family)}")
    for r in unresolved_family:
        print(f"    {r['old_url'].rsplit('/', 1)[-1]}  family={r['family']}  candidates={[c.rsplit('/', 1)[-1] for c in r['candidates']]}")
        print(f"      {r['question']}")

    if fixed:
        path.write_text(json.dumps(items, indent=1))
        print(f"\n  wrote {path} ({len(fixed)} source_url fields updated)")

    report_path = pathlib.Path("eval/refresh_stale_gold_report.json")
    report_path.write_text(json.dumps(
        {"fixed": fixed, "needs_human": needs_human, "unresolved_family": unresolved_family}, indent=1))
    print(f"  wrote {report_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
