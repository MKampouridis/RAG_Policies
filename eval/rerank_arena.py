#!/usr/bin/env python3
"""Reranker investigation: does the reranker earn its place, and what tuning helps?

WHY AN INSTRUMENT AND NOT JUST traffic_replay
`traffic_replay` reports TURNS CHANGED and refuses to call them better or worse -
correct for its job, useless for choosing between reranker configurations. This
adds a quality signal that is still free and still local.

ANCHOR RECALL is that signal. For every question a person thumbed UP, the
documents cited in that approved answer are treated as sufficient: a human read
the answer and said it was right, so losing those documents is a measurable
regression. Anchor recall = the share of those documents a configuration still
returns in the top-k.

WHAT IT CANNOT DO, stated plainly. It rewards REPRODUCING the validated answer,
so it can detect a regression and cannot detect an improvement over it - a
configuration that finds a better document scores worse. Treat a fall as
evidence and a rise as a prompt to look, never the reverse. There are only 14
thumbed-up questions, so this is a coarse instrument; it is reported with its
denominator every time.

THE POSITIVE CONTROL matters more than any arm. `pool100` reproduces a
configuration the ledger already falsified (J0b: 2 rescues, 5 losses, RoA hit@6
70%->62.5%). If this instrument cannot show pool100 losing ground, the
instrument is broken and no other number in the run means anything.

Usage:
    PYTHONPATH=. python eval/rerank_arena.py            # all arms
    PYTHONPATH=. python eval/rerank_arena.py norerank   # one arm
"""
import json
import os
import pathlib
import subprocess
import sys

ARMS = {
    # name          env overrides                              what it asks
    "production":  ({},                                        "shipped: colbert, pool 30"),
    "norerank":    ({"RAG_RERANK": "0"},                       "SPOILER: fusion order, no reranking"),
    "pool100":     ({"RAG_RERANK_POOL": "100"},                "POSITIVE CONTROL: known-bad (J0b)"),
    "pool50":      ({"RAG_RERANK_POOL": "50"},                 "half-way to the known-bad point"),
    "cap2":        ({"RAG_MAX_PER_DOC": "2"},                  "NEW: max 2 chunks per document"),
    "cap3":        ({"RAG_MAX_PER_DOC": "3"},                  "NEW: max 3 chunks per document"),
}
SET_PATH = "eval/questions_traffic.json"
OUT = pathlib.Path("eval/rerank_arena.json")


def run_arm(name: str) -> list:
    """Each arm runs in its OWN process: RERANK_POOL_SIZE and friends are read at
    import, so switching arms in-process would silently reuse the first one."""
    env = dict(os.environ, PYTHONPATH=".", **ARMS[name][0])
    code = r'''
import json, sys
from src.rag import retrieve
items = json.loads(open("%s").read())["items"]
out = []
for it in items:
    try:
        res, _ = retrieve(it["retrieval_query"], [])
        metas = (res.get("metadatas") or [[]])[0]
        docs, seen = [], set()
        for m in metas:
            d = str((m or {}).get("source_url") or "").rstrip("/").rsplit("/", 1)[-1]
            if d and d not in seen:
                seen.add(d); docs.append(d)
        out.append({"tier": it["tier"], "q": it["question"],
                    "anchors": it["baseline_docs"], "got": docs,
                    "n_chunks": len(metas)})
    except Exception as exc:
        out.append({"tier": it["tier"], "q": it["question"],
                    "anchors": it["baseline_docs"], "got": None,
                    "error": f"{type(exc).__name__}: {exc}"})
print("@@RESULT@@" + json.dumps(out))
''' % SET_PATH
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    marker = r.stdout.find("@@RESULT@@")
    if marker < 0:
        print(f"  {name}: FAILED\n{r.stderr[-600:]}")
        return []
    return json.loads(r.stdout[marker + 10:])


def score(rows: list) -> dict:
    up = [r for r in rows if r["tier"] == "anchor_positive" and r.get("got") is not None]
    allr = [r for r in rows if r.get("got") is not None]
    recall = []
    for r in up:
        a = set(r["anchors"])
        if a:
            recall.append(len(a & set(r["got"])) / len(a))
    distinct = [len(r["got"]) for r in allr]
    return {
        "anchor_recall": sum(recall) / len(recall) if recall else None,
        "anchor_n": len(recall),
        "full_recall_questions": sum(1 for r in recall if r == 1.0),
        "mean_distinct_docs": sum(distinct) / len(distinct) if distinct else 0,
        "questions": len(allr),
        "errors": sum(1 for r in rows if r.get("got") is None),
    }


def main() -> int:
    want = sys.argv[1:] or list(ARMS)
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in want:
        print(f"\n== {name}: {ARMS[name][1]}", flush=True)
        rows = run_arm(name)
        if rows:
            store[name] = rows
            OUT.write_text(json.dumps(store))
            s = score(rows)
            print(f"   anchor recall {s['anchor_recall']:.3f} (n={s['anchor_n']}, "
                  f"{s['full_recall_questions']} at 1.0) · "
                  f"mean distinct docs {s['mean_distinct_docs']:.2f} · "
                  f"errors {s['errors']}", flush=True)

    print("\n" + "=" * 74)
    base = store.get("production")
    print(f"{'arm':12} {'anchor recall':>14} {'vs prod':>9} {'full':>6} {'distinct':>9}")
    for name in ARMS:
        if name not in store:
            continue
        s = score(store[name])
        d = ""
        if base and name != "production":
            b = score(base)["anchor_recall"]
            d = f"{(s['anchor_recall'] - b):+.3f}"
        print(f"{name:12} {s['anchor_recall']:>14.3f} {d:>9} "
              f"{s['full_recall_questions']:>4}/{s['anchor_n']:<2} {s['mean_distinct_docs']:>9.2f}")
    print("\nRead the POSITIVE CONTROL first: if pool100 does not lose ground,")
    print("this instrument cannot detect the regression the ledger already recorded,")
    print("and nothing else in this table should be believed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
