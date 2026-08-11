#!/usr/bin/env python3
"""Assemble one stratified regression set from the six question sets.

WHY
Six sets accumulated one at a time, each aimed at a mechanism (siblings,
spanning, multi-entity, partner). Every comparison since has been run on
whichever small set was nearest to hand, at 8-23 questions against a +/-0.20
noise floor. All four external reviews said the same thing: consolidate, and
report per-question differences rather than two arm means.

This merges them, tags each question with the strata that matter here, and
drops duplicates. It does NOT invent questions - every item already existed and
was already checked. The point is one set with known composition, not more
questions.

Deduplication is by normalised question text. Where the same question appears
in two files, the FIRST occurrence wins and the source is recorded, so a
question's provenance survives the merge.

Usage:
    python eval/build_regression_set.py            # report composition only
    python eval/build_regression_set.py --write    # write eval/questions_regression.json
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# Ordered: earlier files win a duplicate. The two 40-question sets are the
# oldest and most-replayed, so they anchor the merge.
SOURCES = [
    ("main", "questions.json"),
    ("holdout", "questions_set2.json"),
    ("pgt_pgr", "questions_set3.json"),
    ("sibling", "questions_set3_sibling.json"),
    ("spanning", "questions_set4_spanning.json"),
    ("multientity", "questions_set5_multientity.json"),
    ("partner", "questions_set6_partner.json"),
    ("abstention", "questions_abstention.json"),
    ("pgr_interpretive", "questions_pgr_interpretive2.json"),
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def strata(item: dict, source: str) -> list[str]:
    """Tags used to check composition. A question can carry several."""
    tags = [f"src:{source}"]
    dt = (item.get("doc_type") or "").lower()
    if dt:
        tags.append(f"doc:{dt}")
    if item.get("expects_abstention"):
        tags.append("abstention")
    if item.get("follow_up_question"):
        tags.append("has_followup")
    if item.get("department"):
        tags.append("departmental")
    q = _norm(item.get("question"))
    if any(w in q for w in ("pgr", "doctorate", "phd", "research degree", "viva", "thesis")):
        tags.append("pgr")
    if any(w in q for w in ("pgt", "masters", "merit", "distinction")):
        tags.append("pgt")
    return tags


def main() -> int:
    merged: list[dict] = []
    seen: dict[str, str] = {}
    dupes = 0

    for source, fname in SOURCES:
        path = ROOT / fname
        if not path.is_file():
            print(f"  MISSING {fname}")
            continue
        try:
            items = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"  UNREADABLE {fname}: {exc}")
            continue
        kept = 0
        for item in items:
            if not isinstance(item, dict) or not item.get("question"):
                continue
            key = _norm(item["question"])
            if key in seen:
                dupes += 1
                continue
            seen[key] = source
            out = dict(item)
            out["_source"] = source
            out["_strata"] = strata(item, source)
            merged.append(out)
            kept += 1
        print(f"  {fname:<38} +{kept:>3} (of {len(items)})")

    print(f"\n  merged: {len(merged)} unique questions   duplicates dropped: {dupes}")

    counts: dict[str, int] = {}
    for m in merged:
        for tag in m["_strata"]:
            counts[tag] = counts.get(tag, 0) + 1
    print("\n  composition:")
    for tag in sorted(counts, key=lambda k: (-counts[k], k)):
        if not tag.startswith("src:"):
            print(f"    {tag:<22}{counts[tag]:>4}")

    thin = [t for t in ("abstention", "pgr", "pgt", "departmental") if counts.get(t, 0) < 8]
    if thin:
        print(f"\n  THIN STRATA (<8 questions): {thin}")
        print("  A comparison aimed at one of these still cannot resolve much;")
        print("  the merge fixes overall power, not per-stratum power.")

    if "--write" in sys.argv:
        out_path = ROOT / "questions_regression.json"
        out_path.write_text(json.dumps(merged, indent=1, ensure_ascii=False))
        print(f"\n  wrote {out_path} ({len(merged)} questions)")
    else:
        print("\n  (pass --write to create eval/questions_regression.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
