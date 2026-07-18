#!/usr/bin/env python3
"""One-time (expensive) generation of per-chunk "situating" context: an
LLM-written sentence that identifies what makes THIS document distinct from
its near-identical siblings, and what this specific chunk covers within it.

Richer, chunk-specific signal than the static chunk_header (same for every
chunk in a document) - targets the RoA sibling-confusion misses that survive
chunk headers and hybrid reranking alone (see eval/report.md's stage-4
discussion). Only scoped to documents in multi-member families (single-
document families have no sibling to confuse retrieval with, so they don't
need this and it would be wasted compute); further scoped to a specific
family allowlist for the initial pilot, since the full multi-member scope is
~14,000 chunks (~20 hours at the per-chunk rate measured in the pilot).

Cached per-document to data/chunk_context_cache/<url_hash>.json (list of
strings aligned to chunk index) so re-running only regenerates for changed
documents, same pattern as data/text_cache/ and data/manifest.json.

Usage:
    python generate_chunk_context.py                 # pilot family scope (PILOT_FAMILIES)
    python generate_chunk_context.py --all-multi      # full multi-member scope (~20h)
"""

import json
import sys
from pathlib import Path

from src.docid import document_family
from src.ingest import chunk_text, clean_text
from src.llm import chat

MANIFEST_PATH = Path("data/manifest.json")
CACHE_DIR = Path("data/chunk_context_cache")

# the families behind the 18 RoA misses in results_stage1_rerank.json - a
# bounded pilot to validate the technique before paying for the full corpus
PILOT_FAMILIES = {
    "integrated-phd-roa-model-a.pdf", "ug-principles-and-framework.pdf",
    "roa-ug-aegean-omiros-4yr-non-standard-year-1.pdf", "csee-ft-masters-accredited-variations.pdf",
    "roa-ug-glossary.pdf", "roa-ug-3yr-year-1-rules.pdf", "masters.pdf",
    "roa-ug-integrated-masters-4yr-year-1.pdf", "east.pdf",
    "roa-ug-diploma-higher-education-year-1.pdf", "ma_social_work.pdf",
}

SYSTEM_PROMPT = """Given a document's title and a short excerpt from it, write ONE concise sentence \
identifying what makes this specific document distinct from near-identical sibling documents \
(degree length, department, programme, academic year) and what topic this excerpt covers. \
Output ONLY that sentence, nothing else."""


def generate_for_chunk(title: str, chunk: str) -> str:
    return chat(messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Document: {title}\n\nExcerpt:\n{chunk}"},
    ]).strip()


def cache_path(url: str) -> Path:
    from src.ingest import url_hash
    return CACHE_DIR / f"{url_hash(url)}.json"


def run(family_filter: set[str] | None) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    kept = [d for d in manifest["documents"].values() if d.get("keep")]

    if family_filter is not None:
        scope = [d for d in kept if document_family(d["url"]) in family_filter]
    else:
        families: dict[str, list] = {}
        for d in kept:
            families.setdefault(document_family(d["url"]), []).append(d)
        scope = [d for docs in families.values() if len(docs) >= 2 for d in docs]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    for i, doc in enumerate(scope, 1):
        out_path = cache_path(doc["url"])
        text = Path(doc["text_cache_path"]).read_text(encoding="utf-8")
        chunks = chunk_text(clean_text(text))
        if not chunks:
            continue

        if out_path.exists():
            cached = json.loads(out_path.read_text())
            if len(cached) == len(chunks):
                print(f"[{i}/{len(scope)}] cached, skipping: {doc['title']}", flush=True)
                continue

        contexts = []
        for chunk in chunks:
            try:
                contexts.append(generate_for_chunk(doc["title"], chunk))
            except Exception as exc:
                contexts.append("")
                print(f"    chunk generation error: {exc}", flush=True)
        out_path.write_text(json.dumps(contexts, indent=2, ensure_ascii=False))
        total_chunks += len(chunks)
        print(f"[{i}/{len(scope)}] ({len(chunks)} chunks): {doc['title']}", flush=True)

    print(f"\nDone. {len(scope)} documents in scope, {total_chunks} chunks generated this run.")


if __name__ == "__main__":
    family_filter = None if "--all-multi" in sys.argv else PILOT_FAMILIES
    run(family_filter)
