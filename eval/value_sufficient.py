#!/usr/bin/env python3
"""Tier-2 item 6 (round-6 / Claude, Gemini, DeepSeek): a VALUE-LEVEL evidence-
sufficiency metric that doesn't rubber-stamp topical vocabulary. The judge-based
check (evidence_sufficient_judge.py) over-credited siblings that merely share
words - e.g. it credited a document about *uncapped* marks for a *capped*-mark
question. The fix (Claude): extract the atomic VALUE-IN-ROLE claims from the gold
reference answer ("60 - minimum weighted average for Merit"), and ask whether a
retrieved document states THAT value in THAT role - not whether the words co-occur.
Uses the NEUTRAL phi4 judge (never a candidate/self judge).

Also addresses the one-sided-bias caveat: the old check only judged keyphrase-
INSUFFICIENT turns (can only raise the number). Here every turn is scored the same
way and a random sample of keyphrase-SUFFICIENT turns is included, so we can put a
symmetric error bar on the headline (false-credits as well as false-misses).

Validates itself on the known false-positives (glossary <- msc-ot / foundation-year)
before trusting the aggregate.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/value_sufficient.py [results_file] [judge]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import chat

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/results_c1_anchor_v2.json")
JUDGE = sys.argv[2] if len(sys.argv) > 2 else "phi4"
QUESTIONS = Path("eval/questions.json")
MANIFEST = json.loads(Path("data/manifest.json").read_text())["documents"]
_text_cache: dict[str, str] = {}


def doc_text(url):
    if url not in _text_cache:
        p = Path((MANIFEST.get(url) or {}).get("text_cache_path", ""))
        _text_cache[url] = p.read_text(encoding="utf-8") if p.is_file() else ""
    return _text_cache[url]


CLAIMS_SYS = (
    "Extract the atomic factual claims from a known-correct reference answer to a University of Essex "
    "rules-of-assessment question. Each claim is a specific VALUE paired with the ROLE it plays "
    "(a threshold, mark, credit count, condition, or definition). "
    'Reply ONLY JSON: {"claims": ["<value> - <role>", ...]} with 1-4 short claims.'
)
CHECK_SYS = (
    "You verify whether a retrieved University of Essex document SUPPLIES specific answer facts. You "
    "are given a list of value-in-role CLAIMS and one DOCUMENT. A claim is SUPPORTED only if the "
    "document states THAT value in THAT role. Topical overlap is NOT support: the same words used for "
    "a different parameter, a different programme's value, or the opposite concept (e.g. 'uncapped' "
    "for a claim about a 'capped' mark) do NOT count. "
    'Reply ONLY JSON: {"supported": <count of claims supported>, "total": <number of claims>}'
)


def extract_claims(question, gold):
    raw = chat(messages=[{"role": "system", "content": CLAIMS_SYS},
                         {"role": "user", "content": f"Question: {question}\n\nReference answer: {gold}"}],
               format="json", model=JUDGE)
    try:
        return [c for c in json.loads(raw).get("claims", []) if c][:4]
    except Exception:
        return []


def doc_supports(claims, url):
    text = doc_text(url)[:9000]
    if not text or not claims:
        return False
    listing = "\n".join(f"- {c}" for c in claims)
    raw = chat(messages=[{"role": "system", "content": CHECK_SYS},
                         {"role": "user", "content": f"CLAIMS:\n{listing}\n\nDOCUMENT:\n{text}"}],
               format="json", model=JUDGE)
    try:
        d = json.loads(raw)
        return int(d.get("supported", 0)) >= (len(claims) + 1) // 2  # majority of claims in-role
    except Exception:
        return False


def value_sufficient(top_urls, claims):
    return any(doc_supports(claims, u) for u in dict.fromkeys(top_urls))


questions = {q["source_url"]: q for q in json.loads(QUESTIONS.read_text())}
results = json.loads(RESULTS.read_text())

# --- self-validation on the known false-positives before trusting the aggregate ---
print(f"=== self-check on known rubber-stamp false-positives (judge={JUDGE}) ===", flush=True)
for r in results:
    if "glossary" not in r["source_url"]:
        continue
    for turn, kpkey in (("primary", "keyphrases"), ("follow_up", "follow_up_keyphrases")):
        q = questions[r["source_url"]]
        claims = extract_claims(r[turn]["question"], r[turn].get("expected_answer", ""))
        vs = value_sufficient(r[turn]["retrieval"]["top_urls"], claims)
        print(f"  glossary[{turn}] claims={claims} -> value_sufficient={vs} "
              f"(old judge-based credited this)", flush=True)

# --- full pass: value-level sufficient@6 across all turns ---
rows = []
for r in results:
    q = questions.get(r["source_url"])
    if not q:
        continue
    for turn, kpkey in (("primary", "keyphrases"), ("follow_up", "follow_up_keyphrases")):
        kps = q.get(kpkey) or []
        if not kps:
            continue
        claims = extract_claims(r[turn]["question"], r[turn].get("expected_answer", ""))
        vs = value_sufficient(r[turn]["retrieval"]["top_urls"], claims) if claims else None
        rows.append({"label": f"{r['source_title']}[{turn}]", "doc_type": r["doc_type"],
                     "hit": r[turn]["retrieval"]["hit_at_6"], "value_sufficient": vs})
    print(f"  scored {r['source_title']}", flush=True)

Path("eval/value_sufficient_result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def rate(sub):
    s = [x for x in sub if x["value_sufficient"] is not None]
    v = sum(1 for x in s if x["value_sufficient"])
    return f"{v}/{len(s)} ({v/len(s)*100:.1f}%)" if s else "n/a"


roa = [x for x in rows if x["doc_type"] == "rules_of_assessment"]
pol = [x for x in rows if x["doc_type"] == "policy"]
print(f"\n=== VALUE-LEVEL evidence-sufficient@6 (judge={JUDGE}) ===")
print(f"  Overall {rate(rows)} | Policy {rate(pol)} | RoA {rate(roa)}")
print(f"  (on hit turns {rate([x for x in rows if x['hit']])} | on miss turns {rate([x for x in rows if not x['hit']])})")
print("  compare: keyphrase-string 83.8%, judge-based(loose) 93.8% raw / ~91% defensible")
