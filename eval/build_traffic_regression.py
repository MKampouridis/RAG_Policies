#!/usr/bin/env python3
"""Build a regression set from REAL TRAFFIC, not from a curated question list.

WHY THIS REPLACES THE HAND-WRITTEN SET
The 151-question set had to be maintained: after every re-ingest its gold
documents could be superseded, which scores correct retrieval as a MISS (9 of
148 items, round 8), and right now eval/check_benchmark_stamp.py reports it
stale. That maintenance buys a proxy for what users actually ask. This set is
built from what they actually asked, and rebuilding it is one command.

WHAT THIS IS, AND EMPHATICALLY WHAT IT IS NOT

  It is a CHANGE DETECTOR. It records which documents retrieval returned for
  each real question, so a later pass can say "retrieval now returns something
  different on 14 of 117 questions - here they are".

  It is NOT a correctness oracle, and the recorded documents are NOT gold. They
  are simply what the system returned at the time. Nobody verified most of
  them. Treating them as gold would repeat the exact failure that made the old
  set stale - an unverified artifact hardening into a standard - only faster,
  because this one regenerates itself.

  So the number it reports is "turns changed", never "hit@6". A change is a
  prompt to LOOK, not a verdict. Some changes will be improvements; the tool
  cannot tell you which, and does not pretend to.

Three tiers, kept separate because they carry different evidential weight:

  anchor_positive  a turn the user thumbed UP. The closest thing here to a
                   verified expectation: a person saw the answer and said it
                   was right. Losing one of these documents is a real signal.
  anchor_neutral   an unrated turn that cited sources. Weak: the baseline is
                   "what it did", not "what it should do". Bulk coverage.
  watch_negative   a turn the user thumbed DOWN. Its documents are explicitly
                   NOT a target - they may be exactly what was wrong. Tracked
                   so a change here can be read as possible GOOD news.

Usage:
    PYTHONPATH=. python eval/build_traffic_regression.py
    PYTHONPATH=. python eval/build_traffic_regression.py --out eval/questions_traffic.json
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

OUT_DEFAULT = "eval/questions_traffic.json"
FEEDBACK = pathlib.Path("data/feedback.jsonl")


def _doc(url: str) -> str:
    return str(url).rstrip("/").rsplit("/", 1)[-1]


def _norm(q: str) -> str:
    return " ".join((q or "").lower().split())


def feedback_items() -> list:
    """Replayable items FROM THE FEEDBACK LOG ITSELF.

    Originally this read feedback only as a ratings lookup keyed on chat.db
    questions - and dropped 31 of 38 ratings, because those conversations had
    since been DELETED and the join found nothing. The rated turns are the
    highest-value entries in the whole set (a person looked at the answer and
    said whether it was right), so losing them to a join was backwards.

    A feedback row is self-sufficient: it stores the question, the
    retrieval_query and the sources at the time it was rated. It needs chat.db
    for nothing.
    """
    out = []
    for q, r in _feedback_rows().items():
        query = (r.get("retrieval_query") or "").strip()
        # Same rule as chat.db items: without a recorded rewrite, only a turn
        # whose query IS its question can be replayed honestly.
        if not query:
            query = r["question"].strip()
        if not r.get("sources"):
            continue
        out.append({
            "tier": "watch_negative" if r.get("rating") == "down" else "anchor_positive",
            "question": r["question"].strip(),
            "retrieval_query": query,
            "primary": query == r["question"].strip(),
            "baseline_docs": sorted({_doc(u) for u in r["sources"]}),
            "rating": r.get("rating"),
            "tags": r.get("tags") or [],
            "recorded_at": r.get("timestamp") or "",
            "source": "feedback",
        })
    return out


def _feedback_rows() -> dict:
    """question -> the latest rating row for it."""
    out = {}
    if not FEEDBACK.exists():
        return out
    for line in FEEDBACK.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        q = _norm(r.get("question"))
        if not q:
            continue
        # Latest rating wins: a question re-rated after a fix should count as
        # its current verdict, not its first.
        out[q] = {"rating": r.get("rating"),
                  "question": r.get("question") or "",
                  "retrieval_query": (r.get("retrieval_query") or "").strip(),
                  "sources": r.get("sources") or [],
                  "tags": r.get("tags") or [],
                  "timestamp": r.get("timestamp") or ""}
    return out


def build() -> list:
    import sqlite3
    conn = sqlite3.connect("data/chat.db")
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT id, conversation_id, role, content, meta, created_at "
        "FROM messages ORDER BY conversation_id, id")]
    conn.close()

    ratings = _feedback_rows()
    items, pending, turn_no = [], {}, {}
    for r in rows:
        cid = r["conversation_id"]
        if r["role"] == "user":
            pending[cid] = r
            continue
        if r["role"] != "assistant" or cid not in pending:
            continue
        q = pending.pop(cid)
        idx = turn_no.get(cid, 0)
        turn_no[cid] = idx + 1
        try:
            meta = json.loads(r["meta"]) if r["meta"] else {}
        except ValueError:
            meta = {}
        sources = meta.get("sources") or []
        if not sources:
            continue                      # nothing to compare against later
        usage = meta.get("usage") or {}
        fb = ratings.get(_norm(q["content"]), {})

        # The query to replay. A PRIMARY turn never runs the contextualizer, so
        # its question IS its query. A follow-up needs the recorded rewrite -
        # from telemetry, or from a feedback row if one rated that turn. Without
        # either, the turn is skipped rather than replayed with the raw question,
        # which would silently ask a DIFFERENT question ("anything on the actual
        # process?" means nothing standalone) and report the difference as a
        # retrieval change.
        primary = idx == 0
        query = (usage.get("retrieval_query") or fb.get("retrieval_query") or "").strip()
        if not query:
            if not primary:
                continue
            query = q["content"].strip()

        rating = fb.get("rating")
        tier = ("watch_negative" if rating == "down"
                else "anchor_positive" if rating == "up" else "anchor_neutral")
        items.append({
            "tier": tier,
            "question": q["content"].strip(),
            "retrieval_query": query,
            "primary": primary,
            "baseline_docs": sorted({_doc(u) for u in sources}),
            "rating": rating,
            "tags": fb.get("tags") or [],
            "recorded_at": r["created_at"],
            "source": "chat",
        })

    # Rated turns come from the feedback log directly, not via a join - see
    # feedback_items(). Conversations get deleted; the ratings outlive them.
    items.extend(feedback_items())

    # Deduplicate on the replayed query: the same question asked twice replays
    # identically, and counting it twice would weight whatever people happened
    # to repeat. A rated instance beats an unrated one for the same query.
    rank = {"anchor_positive": 0, "watch_negative": 1, "anchor_neutral": 2}
    best = {}
    for it in items:
        k = _norm(it["retrieval_query"])
        cur = best.get(k)
        # Tier decides first; recency only breaks ties WITHIN a tier, where
        # both entries came from the same source and so have comparable
        # timestamps (chat.db stores epoch floats, feedback stores ISO strings -
        # comparing across the two would raise).
        if cur is None or rank[it["tier"]] < rank[cur["tier"]] or (
                rank[it["tier"]] == rank[cur["tier"]]
                and it["source"] == cur["source"]
                and it["recorded_at"] > cur["recorded_at"]):
            best[k] = it
    return sorted(best.values(), key=lambda x: (x["tier"], x["question"].lower()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    items = build()
    tiers = {}
    for it in items:
        tiers[it["tier"]] = tiers.get(it["tier"], 0) + 1
    payload = {
        "built_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "note": ("Baselines are WHAT RETRIEVAL RETURNED, not verified gold. "
                 "Report turns changed, never hit@6."),
        "counts": tiers,
        "items": items,
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}: {len(items)} replayable question(s)")
    for t in ("anchor_positive", "anchor_neutral", "watch_negative"):
        print(f"  {t:16} {tiers.get(t, 0)}")
    prim = sum(1 for i in items if i["primary"])
    print(f"  primary {prim} · follow-up {len(items) - prim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
