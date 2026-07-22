#!/usr/bin/env python3
"""Regenerate candidate rename-split family aliases for src/docid.py's
_FAMILY_ALIASES (external code review round 3, 2026-07-22).

Essex renames a document's FILENAME FORMAT between academic years (separator
flips, added -vN versions, extra hyphens, reformatted year tokens), which
splits two editions of the same document into different document_family()
keys - so the superseded prior edition keeps is_current=True next to its
successor and pollutes the current-only retrieval pool.

This script finds candidate rename-splits by grouping kept documents on a
STRUCTURAL-TOKEN-PRESERVING normalized stem (normalizes separators and strips
only a trailing edition-year/-version suffix, so "4yr-year-1" stays distinct
from "4yr-year-2") and reporting groups that (a) span >1 family key and (b)
currently have >1 simultaneously-is_current edition - the actual pollution.
It prints a ready-to-paste Python dict. REVIEW before pasting: the audit is
conservative but the map is explicit and human-checked on purpose, since no
regex reliably tracks Essex's ad-hoc renaming.

Run after any ingest that adds/renames documents:
    python audit_family_aliases.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from src.docid import document_family, normalize_year
import reembed

MANIFEST_PATH = Path("data/manifest.json")


def tight_stem(url: str) -> str:
    fn = url.rsplit("/", 1)[-1]
    fn = re.sub(r"\.pdf$", "", fn.lower())
    fn = re.sub(r"[-_]+", "-", fn)
    fn = re.sub(r"-v\d+$", "", fn)                    # trailing version suffix
    fn = re.sub(r"-(20)?\d{2}(-\d{2,4})?$", "", fn)   # trailing edition year
    fn = re.sub(r"-(20)?\d{2}(-\d{2,4})?$", "", fn)   # again for e.g. -24-v7 leftovers
    return fn


def main() -> None:
    docs = json.loads(MANIFEST_PATH.read_text())["documents"]
    kept = [d for d in docs.values() if d.get("keep")]
    # NOTE: compute flags with the CURRENT alias map in place, so re-running
    # after a paste shows only NEW splits, not already-fixed ones.
    flags = reembed.compute_current_flags(docs)

    by_stem: dict[str, list] = defaultdict(list)
    for d in kept:
        by_stem[tight_stem(d["url"])].append(d)

    alias: dict[str, str] = {}
    groups = 0
    for members in by_stem.values():
        fams = {document_family(m["url"]) for m in members}
        cur = [m for m in members if flags.get(m["url"])]
        if len(fams) > 1 and len(cur) > 1:
            groups += 1
            newest = max(members, key=lambda m: normalize_year(m.get("academic_year")) or "")
            canon = document_family(newest["url"])
            for m in members:
                fk = document_family(m["url"])
                if fk != canon:
                    alias[fk] = canon

    print(f"# {groups} rename-split groups with >1 simultaneously-current edition")
    print(f"# {len(alias)} family-key aliases (paste into src/docid.py _FAMILY_ALIASES, review first)")
    print("_FAMILY_ALIASES = {")
    for k in sorted(alias):
        print(f"    {k!r}: {alias[k]!r},")
    print("}")


if __name__ == "__main__":
    main()
