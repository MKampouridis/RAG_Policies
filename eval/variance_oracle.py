#!/usr/bin/env python3
"""Tier-1 items 3 + 4 (round-6 / Claude): the variance oracle. The one genuinely
new, non-falsified mechanism. Class F failed as an OUTPUT format (enumeration)
because the parameters are uniform - but that uniformity is exactly the signal a
clarification trigger needs, used as INPUT to a decision rather than output.

For each RoA question, take the sibling documents that plausibly compete in its
retrieval pool, extract what EACH one says the answer is, and classify:
  - SAME  (uniform): every sibling gives the same value -> retrieving the 'wrong'
          sibling is HARMLESS (it answers correctly). A miss here is not a harm;
          the assistant should answer confidently, no ask, no hedge.
  - DIFFER (varying): siblings give different values -> the specific document
          matters; a wrong sibling gives a WRONG answer. THIS is when to
          ask/disclose.

This yields both Tier-1 deliverables:
  ITEM 4 (real-harm): among the current retrieval MISSES, how many are DIFFER
         (a genuine wrong-value harm) vs SAME (harmless)? = the true user-harm rate.
  ITEM 3 (variance-gated trigger): a rule 'ask only when the pooled siblings DIFFER'
         - measured against the current fragmentation trigger (>=6 families, ~0.45
         precision, fires on perfectly answerable questions).

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/variance_oracle.py [model]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import chat

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma3:12b"
POOLS = json.loads(Path("eval/reranker_pools.json").read_text())
MAX_FAMS = 4        # cap plausible-sibling set per question (the top competitors; 4 captures the variance)
POOL_DEPTH = 20     # only consider families appearing in the first N pool chunks (realistic competitors)
_CTX = 3500         # per-family excerpt budget - the identity-bearing text is early, keep calls fast

EXTRACT_SYS = (
    "You answer a University of Essex rules-of-assessment question using ONLY the provided document "
    "excerpts from ONE specific programme's document. Reply with the SHORT specific answer this "
    "document gives - a number, mark, threshold, credit value, or a short phrase. If this document "
    "does not contain the answer, reply EXACTLY: NOT STATED. No explanation, just the value."
)
CLASSIFY_SYS = (
    "Different University of Essex programme documents were each asked the same rules-of-assessment "
    "question. You are given their answers. Decide whether they agree on the substantive value or "
    "differ. Ignore wording/format differences (e.g. '60' vs 'a mark of 60 or more' are the SAME; "
    "'40%' vs '50%' DIFFER; a definition phrased two ways with the same meaning is SAME). "
    'Reply ONLY JSON: {"verdict": "SAME" or "DIFFER", "values": ["...distinct values..."]}'
)


def extract(query, passages_text):
    return chat(messages=[{"role": "system", "content": EXTRACT_SYS},
                          {"role": "user", "content": f"Question: {query}\n\nDocument excerpts:\n{passages_text[:_CTX]}"}],
                model=MODEL).strip()


def classify(query, answers):
    listing = "\n".join(f"- {a}" for a in answers)
    raw = chat(messages=[{"role": "system", "content": CLASSIFY_SYS},
                         {"role": "user", "content": f"Question: {query}\n\nDocuments' answers:\n{listing}"}],
               format="json", model=MODEL)
    try:
        d = json.loads(raw)
        return d.get("verdict", "?"), d.get("values", [])
    except Exception:
        return "?", []


def fams_for_turn(p):
    """Distinct competing families in the top POOL_DEPTH chunks, capped, gold included."""
    seen = []
    for f in p["poolfams"][:POOL_DEPTH]:
        if f not in seen:
            seen.append(f)
    fams = seen[:MAX_FAMS]
    if p["goldfam"] not in fams and p["goldfam"] in p["poolfams"]:
        fams = fams[:MAX_FAMS - 1] + [p["goldfam"]]
    return fams


rows = []
roa = [p for p in POOLS]  # pools are already RoA-only
for i, p in enumerate(roa, 1):
    fams = fams_for_turn(p)
    answers = {}
    for f in fams:
        text = "\n\n".join(pas for pas, pf in zip(p["passages"], p["poolfams"]) if pf == f)[:_CTX]
        if text.strip():
            answers[f] = extract(p["query"], text)
    stated = {f: a for f, a in answers.items() if "NOT STATED" not in a.upper()}
    if len(stated) >= 2:
        verdict, values = classify(p["query"], list(stated.values()))
    else:
        verdict, values = ("UNIQUE" if len(stated) == 1 else "NONE"), list(stated.values())
    is_miss = p["goldfam"] not in p["cur_top6"]
    n_fams_pool = len(set(p["poolfams"]))
    rows.append({"set": p["set"], "label": p["label"], "query": p["query"],
                 "verdict": verdict, "n_stated": len(stated), "values": values,
                 "is_miss": is_miss, "frag_trigger": n_fams_pool >= 6})
    if i % 10 == 0:
        print(f"  {i}/{len(roa)}", flush=True)

Path("eval/variance_oracle_result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

# ---- ITEM 4: real-harm on the current misses ----
misses = [r for r in rows if r["is_miss"]]
harmful = [r for r in misses if r["verdict"] == "DIFFER"]
harmless = [r for r in misses if r["verdict"] == "SAME"]
undet = [r for r in misses if r["verdict"] in ("UNIQUE", "NONE", "?")]
print(f"\n=== ITEM 4 - real-harm on current retrieval misses (n={len(misses)}) ===")
print(f"  HARMFUL (siblings DIFFER, wrong sibling => wrong value): {len(harmful)}/{len(misses)} ({len(harmful)/len(misses)*100:.0f}%)")
print(f"  HARMLESS (siblings agree, any sibling answers right):    {len(harmless)}/{len(misses)} ({len(harmless)/len(misses)*100:.0f}%)")
print(f"  UNDETERMINED (0-1 sibling stated a value):               {len(undet)}/{len(misses)}")
for r in harmful:
    print(f"    HARMFUL  {r['label']:44s} values={r['values']}")

# ---- ITEM 3: variance-gated trigger vs the current fragmentation trigger ----
def precision_recall(trigger_key):
    fired = [r for r in rows if r[trigger_key]] if trigger_key != "variance" else [r for r in rows if r["verdict"] == "DIFFER"]
    tp = sum(1 for r in fired if r["is_miss"])
    prec = tp / len(fired) if fired else 0.0
    rec = tp / len(misses) if misses else 0.0
    return len(fired), tp, prec, rec
print(f"\n=== ITEM 3 - trigger comparison (fire = 'ask the user') ===")
for key, name in [("frag_trigger", "fragmentation (>=6 families)"), ("variance", "variance-gated (siblings DIFFER)")]:
    n_fire, tp, prec, rec = precision_recall(key)
    print(f"  {name:34s} fires on {n_fire:2d} turns | precision {prec:.2f} | recall {rec:.2f}")
print(f"\n(all {len(rows)} RoA turns; misses={len(misses)}; full detail in eval/variance_oracle_result.json)")
