"""Phase D / round 4 (Fable 5): gold-multiplicity ceiling analysis. Quantifies
how much of the strict-hit@6 residual is a TEST-SET ARTIFACT - scoring a
question against a single gold document when many current documents contain
the same answer - vs a genuine retrieval failure. Zero hand-labelling.

For each turn's keyphrases, N(q) = number of CURRENT documents whose full text
contains ALL of them (the answer is genuinely present in N documents, so any
of them is a legitimate hit). Under exchangeability, the best a single-gold
strict-hit@6 metric can score that turn is min(1, 6/N) - if N >> 6, the gold
is one of many equally-valid docs and strict hit@6 will usually mark it a
"miss" no matter how good retrieval is. The mean of min(1, 6/N) over all
scoreable turns is the achievable strict-hit@6 ceiling this corpus+metric
imposes. Reports it overall and specifically for the current misses.

Usage: PYTHONPATH=. python eval/gold_multiplicity.py [results_file]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest import _get_collection

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/results_c1_anchor_v2.json")
QUESTIONS = Path("eval/questions.json")
MANIFEST = Path("data/manifest.json")

# current documents' full texts (the retrieval-eligible pool)
coll = _get_collection()
cur_urls = {m.get("source_url", "") for m in coll.get(include=["metadatas"])["metadatas"] if m.get("is_current")}
manifest = json.loads(MANIFEST.read_text())["documents"]
texts = {}
for url in cur_urls:
    d = manifest.get(url) or {}
    p = Path(d.get("text_cache_path", ""))
    if p.exists():
        texts[url] = p.read_text(encoding="utf-8").lower()
print(f"current documents with cached text: {len(texts)}")

def N(keyphrases):
    kps = [k.lower() for k in keyphrases if k]
    if not kps:
        return None
    return sum(1 for t in texts.values() if all(k in t for k in kps))

questions = {q["source_url"]: q for q in json.loads(QUESTIONS.read_text())}
results = json.loads(RESULTS.read_text())

rows = []  # (label, doc_type, hit, N, achievable)
for r in results:
    q = questions.get(r["source_url"])
    if not q:
        continue
    for turn, kpkey in (("primary", "keyphrases"), ("follow_up", "follow_up_keyphrases")):
        n = N(q.get(kpkey) or [])
        if n is None:
            continue
        achievable = min(1.0, 6.0 / n) if n > 0 else 1.0
        rows.append((f"{r['source_title']}[{turn}]", r["doc_type"], r[turn]["retrieval"]["hit_at_6"], n, achievable))

def summ(subset, name):
    if not subset:
        return
    ach = sum(x[4] for x in subset) / len(subset)
    hit = sum(1 for x in subset if x[2]) / len(subset)
    print(f"{name:28s} n={len(subset):3d}  actual hit@6={hit*100:5.1f}%  achievable-ceiling={ach*100:5.1f}%")

print()
summ([x for x in rows if x[1] == "rules_of_assessment"], "RoA (all turns)")
summ([x for x in rows if x[1] == "policy"], "Policy (all turns)")
summ(rows, "Overall")
print()
misses = [x for x in rows if not x[2]]
print(f"--- the {len(misses)} current misses: how many current docs contain the SAME keyphrases (N) ---")
for lbl, dt, hit, n, ach in sorted(misses, key=lambda x: -x[3]):
    verdict = "AMBIGUOUS (N>6: many valid docs)" if n > 6 else ("tight (N<=6)" if n >= 1 else "N=0: keyphrases not jointly in any current doc")
    print(f"   N={n:4d}  {lbl:52s} {verdict}")
n_ambiguous = sum(1 for x in misses if x[3] > 6)
print(f"\nmisses that are gold-multiplicity artifacts (N>6): {n_ambiguous}/{len(misses)}")
