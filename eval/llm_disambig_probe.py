#!/usr/bin/env python3
"""Ceiling probe (#1), feasible form: can REASONING break the sibling tie that
SCORING (all the rerankers) can't? For each RANKING-failure turn (gold family in
the pool but the reranker mis-ranked it), present a capable LLM with the query
and the DISTINCT competing document identities from the pool, and ask it to pick
the single best match. If the LLM picks the gold family, reasoning solves what
scoring couldn't - motivating a reasoning-based (LLM) reranker despite its cost.
If not, the sibling problem is genuinely underspecified and no reranker helps.

This is the incisive form of the Qwen3-Reranker probe - it isolates the
disambiguation decision on exactly the turns that matter (~22), a handful of LLM
calls, instead of an un-runnable 8GB reranker over every candidate.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/llm_disambig_probe.py [judge_model]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import chat

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma3:12b"
POOLS = json.loads(Path("eval/reranker_pools.json").read_text())


def humanize(f):
    return f.replace(".pdf", "").replace("-", " ").replace("_", " ").strip()


SYS = ("You are matching a University of Essex rules-of-assessment question to the ONE document whose "
       "programme / degree-length / year / institution identity fits it best. You are given the question "
       "and a numbered list of candidate document identities. Reply with ONLY the number of the single "
       "best-matching document. If genuinely impossible to tell, reply 0.")

correct = total = abstain = 0
by_set = {}
for p in POOLS:
    gold_in_pool = p["goldfam"] in p["poolfams"]
    cur_hit = p["goldfam"] in p["cur_top6"]
    if not (gold_in_pool and not cur_hit):
        continue  # only the ranking-failure turns
    total += 1
    fams = list(dict.fromkeys(p["poolfams"]))  # distinct, order-preserved
    options = "\n".join(f"{i+1}. {humanize(f)}" for i, f in enumerate(fams))
    raw = chat(messages=[{"role": "system", "content": SYS},
                         {"role": "user", "content": f"Question: {p['query']}\n\nCandidate documents:\n{options}"}],
               model=MODEL).strip()
    digits = "".join(c for c in raw.split()[0] if c.isdigit()) if raw.split() else ""
    pick = int(digits) if digits else -1
    s = by_set.setdefault(p["set"], [0, 0])
    s[1] += 1
    if pick == 0:
        abstain += 1
    elif 1 <= pick <= len(fams) and fams[pick - 1] == p["goldfam"]:
        correct += 1
        s[0] += 1
    print(f"  [{p['set']}] {'CORRECT' if 1 <= pick <= len(fams) and fams[pick-1]==p['goldfam'] else 'wrong ' if pick>0 else 'abstain'}"
          f"  pick={pick}/{len(fams)}  {p['label'][:45]}", flush=True)

print(f"\n=== LLM disambiguation ({MODEL}) on {total} RANKING-failure turns ===")
for sn, (c, n) in by_set.items():
    print(f"  {sn}: {c}/{n} correctly disambiguated ({c/n*100:.0f}%)")
print(f"  TOTAL: {correct}/{total} correct ({correct/total*100:.0f}%), {abstain} abstained")
print(f"  => reasoning {'CAN' if correct/total > 0.4 else 'CANNOT'} reliably break the sibling tie "
      f"(scoring rerankers rescued ~2-5 of these).")
