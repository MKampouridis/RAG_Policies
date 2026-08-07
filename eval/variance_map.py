#!/usr/bin/env python3
"""The variance MAP (round-6 follow-up): for the common rules-of-assessment
parameters, is the value UNIFORM across programmes (so retrieving the 'wrong'
sibling is harmless and the assistant can answer confidently) or does it VARY (so
the specific document matters and a targeted disclosure is warranted)? This is the
lookup table a variance-gated disclosure would consult at query time.

Method: for each parameter, probe a fixed question consistently across a sample of
current RoA documents that mention it, extract the short value each states, then
classify the collected answers into distinct substantive values (uniform vs
varying), ignoring phrasing differences. Also flags when the variation simply
tracks UG-vs-PGT level (predictable) rather than being programme-specific.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/variance_map.py [n_docs] [model]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest import _get_collection
from src.llm import chat

N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemma3:12b"
MANIFEST = json.loads(Path("data/manifest.json").read_text())["documents"]

# (name, probe question, keyword that selects relevant documents)
PARAMS = [
    ("Merit threshold", "What is the minimum overall weighted average mark required to be awarded a Merit?", "merit"),
    ("Distinction threshold", "What is the minimum overall weighted average mark required to be awarded a Distinction?", "distinction"),
    ("Module pass mark", "What is the minimum mark required to pass an individual module?", "pass"),
    ("Condonement threshold", "What is the lowest module mark that can be condoned (compensated) rather than requiring reassessment?", "condon"),
    ("Reassessment mark cap", "When a failed module is reassessed, what is the maximum (capped) mark that can be awarded for it?", "capped"),
    ("Credits for the award", "How many credits must be passed to be awarded this degree?", "credits"),
    ("Permitted further attempts", "How many further attempts (reassessments) is a student normally permitted at a failed module?", "further"),
    ("First-class / top classification", "What overall mark is required for the highest classification (First Class or Distinction)?", "class"),
]

EXTRACT_SYS = (
    "You answer a University of Essex rules-of-assessment question using ONLY the provided document, "
    "for THIS document's programme. Reply with ONLY the short specific value (a number/mark/credit "
    "count/short phrase). If the document does not state it, reply EXACTLY: NOT STATED."
)
CLASSIFY_SYS = (
    "Several University of Essex programme documents were each asked the SAME question. Given their "
    "answers, group them into DISTINCT substantive values (ignore wording/format - '60' and 'a mark "
    "of 60 or more' are the same; '40' and '50' differ). Decide UNIFORM (one substantive value) or "
    "VARYING (two or more). If the split is clearly just undergraduate-vs-postgraduate (e.g. pass "
    "mark 40 for UG, 50 for PGT), set level_split true. "
    'Reply ONLY JSON: {"verdict":"UNIFORM"|"VARYING","values":[{"value":"..","count":N}],"level_split":bool}'
)


def doc_text(url):
    p = Path((MANIFEST.get(url) or {}).get("text_cache_path", ""))
    return p.read_text(encoding="utf-8") if p.is_file() else ""


# current RoA docs with text, deterministic order
coll = _get_collection()
urls = []
for m in coll.get(include=["metadatas"])["metadatas"]:
    if m.get("is_current") and m.get("doc_type") == "rules_of_assessment":
        u = m.get("source_url", "")
        if u and u not in urls:
            urls.append(u)
urls.sort()
texts = {u: doc_text(u) for u in urls}
texts = {u: t for u, t in texts.items() if t}
print(f"current RoA docs with text: {len(texts)} | model={MODEL}\n", flush=True)


def extract(probe, text):
    return chat(messages=[{"role": "system", "content": EXTRACT_SYS},
                          {"role": "user", "content": f"Question: {probe}\n\nDocument:\n{text[:9000]}"}],
                model=MODEL).strip()


def classify(probe, answers):
    listing = "\n".join(f"- {a}" for a in answers)
    raw = chat(messages=[{"role": "system", "content": CLASSIFY_SYS},
                         {"role": "user", "content": f"Question: {probe}\n\nAnswers:\n{listing}"}],
               format="json", model=MODEL)
    try:
        d = json.loads(raw)
        return d.get("verdict", "?"), d.get("values", []), bool(d.get("level_split", False))
    except Exception:
        return "?", [], False


rows = []
for name, probe, kw in PARAMS:
    pool = [u for u in texts if kw in texts[u].lower()][:N_DOCS]
    answers = []
    for u in pool:
        a = extract(probe, texts[u])
        if "NOT STATED" not in a.upper():
            answers.append(a)
    verdict, values, level = classify(probe, answers) if len(answers) >= 2 else ("TOO FEW", [], False)
    rows.append({"parameter": name, "probe": probe, "n_docs": len(pool), "n_stated": len(answers),
                 "verdict": verdict, "level_split": level, "values": values})
    tag = "VARYING (level-split)" if (verdict == "VARYING" and level) else verdict
    dist = ", ".join(f"{v.get('value')}×{v.get('count')}" for v in values) if values else "-"
    print(f"RESULT {name:34s} [{tag:22s}] stated {len(answers)}/{len(pool)} | {dist}", flush=True)

Path("eval/variance_map_result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

uniform = [r for r in rows if r["verdict"] == "UNIFORM"]
vary_prog = [r for r in rows if r["verdict"] == "VARYING" and not r["level_split"]]
vary_level = [r for r in rows if r["verdict"] == "VARYING" and r["level_split"]]
print(f"\n=== VARIANCE MAP SUMMARY ({len(rows)} parameters) ===")
print(f"  UNIFORM (answer confidently, no disclosure):        {len(uniform)}  {[r['parameter'] for r in uniform]}")
print(f"  VARYING by programme (targeted disclosure fires):   {len(vary_prog)}  {[r['parameter'] for r in vary_prog]}")
print(f"  VARYING but only UG-vs-PGT (disambiguated by level): {len(vary_level)}  {[r['parameter'] for r in vary_level]}")
