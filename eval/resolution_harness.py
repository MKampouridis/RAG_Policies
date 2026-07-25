#!/usr/bin/env python3
"""Tier-2 item 5 (round-6 / all reviewers converged): the resolution harness -
evaluate the D3 clarification UX with a metric that DOESN'T structurally penalise
asking. hit@6 scores a clarifying question as a miss by design; instead measure
RESOLUTION@2: when the first-turn 'guess' retrieval MISSES, if the user then
supplies the missing programme identity (what D3's clarifying question asks for),
does retrieval resolve to a hit within one more exchange?

For each RoA question:
  - GUESS baseline: retrieve(question) as-is -> hit@6? (what the system does today)
  - If it misses, simulate the clarified follow-up: the user names the programme
    (its readable document identity), so retrieve(question + identity) -> hit@6?
Resolution@2 = of the baseline MISSES, the fraction that HIT after clarification.
That is the payoff D3 buys, measured offline. The gold programme name is in the
question metadata, so the simulated user is 'cooperative' - this is the ceiling if
users answer, which is the number the ask-vs-guess product decision needs.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/resolution_harness.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.docid import document_family as fam
from src.rag import retrieve

QSETS = ["eval/questions.json", "eval/questions_set2.json"]


def hit(question, goldfam):
    res, _ = retrieve(question, [])
    return goldfam in {fam(m.get("source_url", "")) for m in res.get("metadatas", [[]])[0]}


def identity(q):
    """What a cooperative user would say to name their programme - the readable
    document title, stripped of generic 'rules of assessment' boilerplate."""
    t = q.get("source_title", "")
    for junk in ("Rules of Assessment", "rules of assessment", "Rules of assessment"):
        t = t.replace(junk, "")
    return t.strip(" -,")


rows = []
for qpath in QSETS:
    for q in json.loads(Path(qpath).read_text()):
        if q.get("doc_type") != "rules_of_assessment":
            continue
        goldfam = fam(q["source_url"])
        guess = hit(q["question"], goldfam)
        resolved = None
        if not guess:
            clarified = f"{q['question']} (for the {identity(q)} programme)"
            resolved = hit(clarified, goldfam)
        rows.append({"label": q["source_title"], "guess_hit": guess, "resolved": resolved,
                     "identity": identity(q)})
    print(f"  done {qpath}", flush=True)

Path("eval/resolution_harness_result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

n = len(rows)
guess_hits = sum(1 for r in rows if r["guess_hit"])
misses = [r for r in rows if not r["guess_hit"]]
resolved = sum(1 for r in misses if r["resolved"])
print(f"\n=== RESOLUTION HARNESS (RoA, n={n}) ===")
print(f"  GUESS baseline (today): {guess_hits}/{n} hit@6 ({guess_hits/n*100:.1f}%)")
print(f"  of the {len(misses)} misses, RESOLVED after clarification: {resolved}/{len(misses)} "
      f"({resolved/len(misses)*100:.0f}%)")
eff = guess_hits + resolved
print(f"  effective success if D3 asks-and-user-answers: {eff}/{n} ({eff/n*100:.1f}%)  "
      f"(vs {guess_hits/n*100:.1f}% guessing)")
print("\n  misses NOT resolved even with the programme named (genuine retrieval gaps):")
for r in misses:
    if not r["resolved"]:
        print(f"    {r['label']}  (named: {r['identity']})")
