#!/usr/bin/env python3
"""Compare two eval result files, scoring hit@6 family-aware.

A stored results file freezes the gold URL of the day it ran. After an ingest
that document can be superseded, and a run that correctly returns the CURRENT
edition is then scored a MISS - so comparing a new run against an older
baseline shows a regression the retrieval never had. Measured on these sets:
10 of 80 questions carry a gold document that is no longer current.

Same rule retrieval_replay.py already uses: a hit counts the gold URL OR the
CURRENT edition of its document family, and only the current one, so returning
an older sibling still misses.

Usage: PYTHONPATH=. python eval/compare_family_aware.py <baseline.json> <new.json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.docid import document_family
from src.ingest import _get_collection


def corpus_state():
    fam, cur = {}, set()
    for m in _get_collection().get(include=["metadatas"])["metadatas"]:
        u = m.get("source_url", "")
        if u and m.get("is_current"):
            cur.add(u)
            fam.setdefault(document_family(u), set()).add(u)
    return fam, cur


def rescore(path, fam):
    rows = {}
    for r in json.loads(Path(path).read_text()):
        gold = r["source_url"]
        accept = {gold} | fam.get(document_family(gold), set())
        for turn in ("primary", "follow_up"):
            t = r.get(turn)
            if not t:
                continue
            urls = (t.get("retrieval") or {}).get("top_urls") or []
            hit = any(u in accept for u in urls[:6])
            rows[(r["source_title"], turn)] = {
                "hit": hit,
                "stored_hit": (t.get("retrieval") or {}).get("hit_at_6"),
                "judge": ((t.get("judge") or {}).get("score")),
            }
    return rows


def main() -> int:
    fam, cur = corpus_state()
    a, b = rescore(sys.argv[1], fam), rescore(sys.argv[2], fam)
    keys = sorted(set(a) & set(b))
    if not keys:
        print("  no overlapping turns"); return 1
    ha = sum(a[k]["hit"] for k in keys); hb = sum(b[k]["hit"] for k in keys)
    sa_raw = sum(a[k]["stored_hit"] or 0 for k in keys)
    sb_raw = sum(b[k]["stored_hit"] or 0 for k in keys)
    ja = [a[k]["judge"] for k in keys if a[k]["judge"] is not None]
    jb = [b[k]["judge"] for k in keys if b[k]["judge"] is not None]
    print(f"  turns compared: {len(keys)}")
    print(f"  hit@6 as STORED     baseline {sa_raw:3}  new {sb_raw:3}  ({sb_raw - sa_raw:+d})")
    print(f"  hit@6 FAMILY-AWARE  baseline {ha:3}  new {hb:3}  ({hb - ha:+d})")
    if ja and jb:
        print(f"  judge mean          baseline {sum(ja)/len(ja):.2f}  new {sum(jb)/len(jb):.2f}"
              f"  ({sum(jb)/len(jb) - sum(ja)/len(ja):+.2f})   [same judge only]")
    gained = [k for k in keys if not a[k]["hit"] and b[k]["hit"]]
    lost = [k for k in keys if a[k]["hit"] and not b[k]["hit"]]
    print(f"  gained {len(gained)}   lost {len(lost)}")
    for k in lost[:10]:
        print(f"    LOST  {k[0][:48]} ({k[1]})")
    for k in gained[:10]:
        print(f"    GAIN  {k[0][:48]} ({k[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
