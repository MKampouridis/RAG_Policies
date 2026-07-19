#!/usr/bin/env python3
"""Stage G: builds a separate "pseudo-query" Chroma collection so each real
chunk stays reachable through more than one representation - the single
pooled chunk embedding used today can lose fine distinctions between
near-identical RoA siblings (Luan et al. 2020). Per Tang 2021 / Lee 2025
(pseudo-query and question-generation indexing), each chunk with at least
one usable structured facet (degree_length, award_type, department) gets 1-2
deterministic, template-filled question strings embedded and indexed
*pointing back at the real chunk's own text and metadata* - so if a user's
question resembles the template, the real content still surfaces.

Deliberately NOT another LLM narrative-generation pass (that already failed,
see eval/report.md's "stage4_context_pilot") - these templates are filled
from metadata already extracted deterministically at ingest time, so this
is cheap: template strings + an embed_batch call, no chat model involved.

Queried at retrieval time by src/pseudo_query.py as an isolated fourth
channel (src/rag.py's PSEUDO_QUERY_ENABLED flag), fused via RRF alongside
dense/BM25/facet/SPLADE/ensemble - never replacing the primary collection.
"""

from src.ingest import _get_collection
from src.llm import EMBED_DOCUMENT_PREFIX, embed_batch
from src.pseudo_query import PSEUDO_COLLECTION_NAME, _get_pseudo_collection

MAX_PSEUDO_QUERIES_PER_CHUNK = 2
BATCH_SIZE = 64


def _pseudo_queries(meta: dict) -> list[str]:
    degree_length = meta.get("degree_length") or ""
    award_type = meta.get("award_type") or ""
    department = meta.get("department") or ""

    queries = []
    if degree_length and award_type:
        queries.append(f"What are the rules of assessment for a {degree_length} {award_type} programme?")
    if department:
        queries.append(f"What are the assessment rules for {department}?")
    if degree_length and department:
        queries.append(f"What are the progression rules for a {degree_length} programme in {department}?")
    if award_type and not degree_length:
        queries.append(f"What exit award rules apply for a {award_type}?")
    return queries[:MAX_PSEUDO_QUERIES_PER_CHUNK]


def run() -> None:
    source = _get_collection()
    data = source.get(include=["documents", "metadatas"])
    ids, documents, metadatas = data["ids"], data["documents"], data["metadatas"]

    pq_ids, pq_texts, pq_documents, pq_metadatas = [], [], [], []
    for chunk_id, doc, meta in zip(ids, documents, metadatas):
        for i, pq_text in enumerate(_pseudo_queries(meta)):
            pq_ids.append(f"{chunk_id}_pq{i}")
            pq_texts.append(pq_text)
            pq_documents.append(doc)
            pq_metadatas.append(meta)

    print(f"Generated {len(pq_ids)} pseudo-queries from {len(ids)} chunks", flush=True)
    if not pq_ids:
        print("Nothing to index (no chunks had a usable facet).")
        return

    dest = _get_pseudo_collection()
    for start in range(0, len(pq_ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        embeddings = embed_batch([EMBED_DOCUMENT_PREFIX + t for t in pq_texts[start:end]])
        dest.upsert(
            ids=pq_ids[start:end],
            embeddings=embeddings,
            documents=pq_documents[start:end],
            metadatas=pq_metadatas[start:end],
        )
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  [{min(end, len(pq_ids))}/{len(pq_ids)}]", flush=True)

    print(f"Done. Wrote {len(pq_ids)} entries to Chroma collection {PSEUDO_COLLECTION_NAME!r}")


if __name__ == "__main__":
    run()
