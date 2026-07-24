"""Class F pilot: structured parameter-extraction, "ambiguity becomes enumeration".
The proposal (round-4 reviewers): for an underspecified query like "what's the
minimum weighted average to pass with Merit?" - which names no programme and so
mis-retrieves a sibling - don't retrieve one document; instead EXTRACT the
parameter across ALL current programme documents and ENUMERATE {programme -> value}.

Bounded pre-validation before any build (project discipline: kill cheaply if the
mechanism can't work). Two questions decide it:

  (1) EXTRACTION RELIABILITY - can an LLM reliably pull a named parameter out of
      each sibling document? (facet/department extraction was already found too
      sparse in earlier rounds - if this repeats, Class F isn't buildable.)
  (2) DOES ENUMERATION ADD VALUE - only if the parameter VARIES across programmes.
      If it's uniform (every programme's Merit threshold is 60), the query was
      never really ambiguous: any retrieved sibling already answers it correctly
      (a gold-multiplicity artifact, not a real miss), so enumerating is pointless.
      If it varies AND the query names no programme, a long enumerated table is
      strictly worse UX than D3's single clarifying question.

Anchor: the "pass with Merit" misses (ma_social_work_25, masters-25, msc-physiotherapy-25)
all have gold Merit=60 / Distinction=70 - so we expect uniformity, which would
empirically demonstrate arm (2)'s "enumeration pointless" case.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/class_f_pilot.py [n_docs]
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest import _get_collection
from src.llm import chat

N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
EXTRACT_MODEL = "gemma3:12b"  # the adopted production generator - reflects what a real build would use
MANIFEST = json.loads(Path("data/manifest.json").read_text())["documents"]

SYS = (
    "You extract classification thresholds from a single University of Essex postgraduate "
    "rules-of-assessment document. Report ONLY the values this specific document states for ITS "
    "programme. Output STRICT JSON with exactly these keys and integer-or-null values:\n"
    '{"merit_min_weighted_average": <int or null>, "distinction_min_weighted_average": <int or null>}\n'
    "The value is the minimum overall weighted average mark (0-100) required for Merit / Distinction. "
    "Use null if the document does not state it. Output ONLY the JSON object, nothing else."
)


def extract(text: str) -> dict:
    raw = chat(
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": f"DOCUMENT:\n{text[:14000]}"}],
        model=EXTRACT_MODEL, format="json",
    )
    try:
        d = json.loads(raw)
        return {"merit": d.get("merit_min_weighted_average"), "distinction": d.get("distinction_min_weighted_average")}
    except Exception:
        return {"merit": None, "distinction": None, "_parse_fail": True}


# current RoA docs that mention Merit = the PGT-classification pool this parameter lives in
coll = _get_collection()
seen = {}
for m in coll.get(include=["metadatas"])["metadatas"]:
    if m.get("is_current") and m.get("doc_type") == "rules_of_assessment":
        seen.setdefault(m.get("source_url", ""), m.get("source_title", ""))

pool = []
for url, title in seen.items():
    p = Path((MANIFEST.get(url) or {}).get("text_cache_path", ""))
    if p.is_file():
        t = p.read_text(encoding="utf-8")
        if "merit" in t.lower():
            pool.append((url, title, t))

# keep the anchor docs in the sample, then fill to N_DOCS (deterministic order)
anchors = {"ma_social_work_25.pdf", "masters-25.pdf", "msc-physiotherapy-25.pdf"}
pool.sort(key=lambda x: (x[0].split("/")[-1] not in anchors, x[0]))
pool = pool[:N_DOCS]
print(f"extracting Merit/Distinction thresholds from {len(pool)} current PGT RoA docs (model={EXTRACT_MODEL})\n", flush=True)

rows = []
for i, (url, title, text) in enumerate(pool, 1):
    r = extract(text)
    rows.append((url.split("/")[-1], r.get("merit"), r.get("distinction"), r.get("_parse_fail", False)))
    if url.split("/")[-1] in anchors or i % 10 == 0:
        print(f"  [{i}/{len(pool)}] {url.split('/')[-1]:42s} merit={r.get('merit')} distinction={r.get('distinction')}", flush=True)

merit_vals = [m for _, m, _, _ in rows if isinstance(m, int)]
dist_vals = [d for _, _, d, _ in rows if isinstance(d, int)]
parse_fail = sum(1 for *_, f in rows if f)

print("\n=== (1) EXTRACTION RELIABILITY ===")
print(f"  merit extracted: {len(merit_vals)}/{len(rows)} ({len(merit_vals)/len(rows)*100:.0f}%)   "
      f"distinction: {len(dist_vals)}/{len(rows)} ({len(dist_vals)/len(rows)*100:.0f}%)   parse-fails: {parse_fail}")
print("\n=== (2) VARIANCE (does enumeration add value?) ===")
print(f"  merit value distribution:       {dict(Counter(merit_vals))}")
print(f"  distinction value distribution: {dict(Counter(dist_vals))}")

print("\n=== ANCHOR CHECK (gold: Merit=60, Distinction=70) ===")
for name, m, d, _ in rows:
    if name in anchors:
        print(f"  {name:42s} extracted merit={m} distinction={d}  "
              f"{'OK' if m == 60 else 'MISMATCH'}")

uniform = len(set(merit_vals)) <= 1
reliable = len(merit_vals) / len(rows) >= 0.7
print("\n=== VERDICT ===")
if not reliable:
    print("  EXTRACTION UNRELIABLE (<70%) -> same sparsity wall as facet extraction -> Class F not buildable.")
elif uniform:
    print("  RELIABLE but UNIFORM -> the parameter is the same across programmes, so the query was never")
    print("  truly ambiguous (any sibling answers it -> gold-multiplicity artifact). Enumeration is pointless")
    print("  here; the 'underspecified Merit misses' are not real retrieval failures.")
else:
    print("  RELIABLE and VARIES -> enumeration could carry information, BUT if the query names no programme")
    print("  a long enumerated table is worse UX than D3's single clarifying question. Weigh against D3.")
