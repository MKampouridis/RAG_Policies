"""Phase B1 (review round 3): document-level routing oracle with pre-registered
gates (Fable 5). Question: is an identity-only query precise enough that HARD
macro-routing (restrict chunk retrieval to top-K identity-routed documents)
could rescue the out-of-pool misses WITHOUT newly losing currently-hitting
turns? Two gates, both must pass to justify building it:
  RESCUE : gold in routing top-K for >= 3 of the out-of-pool miss turns
  SAFETY : ZERO currently-hitting turns have gold outside routing top-K
           (each such turn is a guaranteed new loss under hard routing)
Runs over ALL 80 logged deterministic retrieval queries from the current
production eval (c1_anchor), post-hygiene. BM25 over current-document identity
cards (programme_name + department + aliases + readable title) - the lexical
signal a router would use; deliberately generous (no is_current confound).
"""
import json, re
from pathlib import Path
from rank_bm25 import BM25Okapi
import sys
sys.path.insert(0, "/Users/mkampo/RAG_Policies")
from src.ingest import _get_collection, _load_doc_identity, _readable_title

TOKEN = re.compile(r"[a-z0-9]+")
tok = lambda t: TOKEN.findall((t or "").lower())

# current documents + identity cards
coll = _get_collection()
data = coll.get(include=["metadatas"])
cur = {}
for m in data["metadatas"]:
    if not m.get("is_current"):
        continue
    url = m.get("source_url", "")
    if url and url not in cur:
        cur[url] = m
urls = list(cur)
def card(url, m):
    idy = _load_doc_identity(url)
    parts = [_readable_title(m.get("title") or url.rsplit("/", 1)[-1]),
             idy.get("programme_name", ""), idy.get("department", "") or (m.get("department") or ""),
             " ".join(idy.get("aliases") or [])]
    return " ".join(p for p in parts if p)
cards = [tok(card(u, cur[u])) for u in urls]
bm25 = BM25Okapi(cards)
url_rank = {}  # (query, gold) -> rank

RESULTS = json.loads(Path("eval/results_c1_anchor.json").read_text())
K = 5
rows = []  # (turn_label, gold_url, currently_hit, routing_rank)
for r in RESULTS:
    for turn in ("primary", "follow_up"):
        t = r[turn]
        gold = r["source_url"]
        q = t["retrieval"]["retrieval_query"]
        scores = bm25.get_scores(tok(q))
        order = sorted(range(len(urls)), key=lambda i: scores[i], reverse=True)
        rank = next((i + 1 for i, idx in enumerate(order) if urls[idx] == gold), None)
        rows.append((f"{r['source_title']}[{turn}]", r["doc_type"], gold, t["retrieval"]["hit_at_6"], rank))

hit_rows = [x for x in rows if x[3]]
miss_rows = [x for x in rows if not x[3]]
roa_miss = [x for x in miss_rows if x[1] == "rules_of_assessment"]

print(f"total turns: {len(rows)}  currently-hit: {len(hit_rows)}  currently-miss: {len(miss_rows)} (RoA miss: {len(roa_miss)})")
print(f"\n--- SAFETY gate (currently-hit turns whose gold falls OUTSIDE routing top-{K}) ---")
unsafe = [x for x in hit_rows if x[4] is None or x[4] > K]
print(f"unsafe hit turns: {len(unsafe)} / {len(hit_rows)}  (gate PASSES only if 0)")
for lbl, dt, gold, hit, rank in sorted(unsafe, key=lambda x: (x[4] is not None, x[4] or 9999))[:20]:
    print(f"   routing_rank={rank}  {lbl}")

print(f"\n--- RESCUE gate (currently-miss turns with gold IN routing top-{K}) ---")
rescuable = [x for x in miss_rows if x[4] is not None and x[4] <= K]
print(f"rescuable miss turns: {len(rescuable)} / {len(miss_rows)}  (gate wants >= 3)")
for lbl, dt, gold, hit, rank in rescuable:
    print(f"   routing_rank={rank}  {lbl}")

print(f"\n=== VERDICT: hard macro-routing is justified only if SAFETY passes (0 unsafe) AND RESCUE >= 3 ===")
print(f"   SAFETY: {'PASS' if not unsafe else 'FAIL ('+str(len(unsafe))+' guaranteed new losses)'}")
print(f"   RESCUE: {'PASS' if len(rescuable) >= 3 else 'FAIL ('+str(len(rescuable))+' rescues)'}")
