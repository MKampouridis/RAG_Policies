#!/usr/bin/env python3
"""Fetch-pool sweep: how small can FETCH_POOL_MULTIPLIER get before recall drops?

Latency profiling (2026-08-09) found retrieval is dominated by the DENSE search -
2.73s of a 3.1s retrieve(), versus 0.16s for ColBERT reranking and 0.03s for BM25.
So the lever is how many candidates each channel fetches, not the rerank pool:
N_RESULTS(6) x FETCH_POOL_MULTIPLIER(8) = 48 per channel, ~96 after fusion, to
answer with 6.

This is STAGE 1 of two. It replays each turn's stored retrieval_query - no
generation, no judging - so it measures recall and latency only. A surviving
candidate still needs a full eval, because a smaller pool can change WHICH CHUNKS
come back even when the document is unchanged, and hit@6 compares document URLs
(the blind spot measured at 8.7 points in round 7). Screening on hit@6 alone
would repeat the mistake that cost 8.8 points: validating on a metric that cannot
see the failure mode.

Usage: PYTHONPATH=. python eval/fetch_pool_sweep.py [multipliers...]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.rag as rag

RESULTS = ["eval/results_gemma3_e2e_main.json", "eval/results_gemma3_e2e_set2.json"]
MULTIPLIERS = [int(x) for x in sys.argv[1:]] or [8, 6, 4, 3, 2]


def load_turns():
    turns = []
    for rf in RESULTS:
        p = Path(rf)
        if not p.is_file():
            continue
        for r in json.loads(p.read_text()):
            for t in ("primary", "follow_up"):
                x = r.get(t)
                if not x:
                    continue
                q = x["retrieval"].get("retrieval_query") or x.get("question")
                if q:
                    turns.append((r["source_url"], q))
    return turns


turns = load_turns()
print(f"replaying {len(turns)} stored queries per multiplier "
      f"(baseline FETCH_POOL_MULTIPLIER={rag.FETCH_POOL_MULTIPLIER})\n", flush=True)

rag.retrieve("warm up the caches", [])  # avoid charging cold start to the first config

rows = []
for mult in MULTIPLIERS:
    rag.FETCH_POOL_MULTIPLIER = mult
    hits = 0
    t0 = time.time()
    for gold, q in turns:
        res, _ = rag.retrieve(q, [])
        urls = [m.get("source_url") for m in res.get("metadatas", [[]])[0]]
        hits += gold in urls[:rag.N_RESULTS]
    elapsed = time.time() - t0
    per_query = elapsed / len(turns)
    rows.append((mult, hits, len(turns), per_query))
    print(f"  multiplier {mult:2d} (pool {rag.N_RESULTS*mult:3d}/channel): "
          f"hit@6 {hits}/{len(turns)} ({hits/len(turns)*100:5.1f}%)   "
          f"{per_query:.2f}s/query", flush=True)

base = next((r for r in rows if r[0] == 8), rows[0])
print(f"\n{'mult':>5s} {'hit@6':>8s} {'vs base':>9s} {'s/query':>9s} {'speedup':>9s}")
for mult, hits, n, per_query in rows:
    print(f"{mult:5d} {hits/n*100:7.1f}% {(hits-base[1])/n*100:+8.1f} "
          f"{per_query:8.2f}s {base[3]/per_query:8.2f}x")
print("\nSTAGE 2: any candidate that holds hit@6 still needs a full eval "
      "(./eval_session.sh) - a smaller pool can change which CHUNKS return.")
Path("eval/fetch_pool_sweep_result.json").write_text(json.dumps(
    [{"multiplier": m, "hits": h, "turns": n, "sec_per_query": s} for m, h, n, s in rows],
    indent=2))
