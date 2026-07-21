#!/usr/bin/env python3
"""Re-embed all kept documents from cached text, without re-crawling or
re-classifying. Use this after changing EMBED_MODEL, or after any fix to
how chunks are embedded (chunk headers, text cleaning, prefixes) that
requires refreshing vectors already in the store.

Also computes each document's `is_current` flag (is this the most recent
academic year within its document family?) so retrieval can pre-filter the
historical archive out of the default candidate pool. The flag lives only
in chunk metadata - not in the embedded text - so future flag flips (e.g.
after Essex publishes next year's documents) only need
`recompute_current_flags()`, not a re-embed.
"""

import json
import re
from pathlib import Path

from src.docid import document_family, effective_year, normalize_year, previous_year
from src.ingest import _get_collection, bump_corpus_version, chunk_text, clean_text, upsert_document

MANIFEST_PATH = Path("data/manifest.json")

YEAR_DIR_RE = re.compile(r"/(20\d{2}-\d{2,4})/")


def compute_current_flags(documents: dict) -> dict[str, bool]:
    """URL -> is_current. A document is current when it is the most recent
    academic year within its family. URL path evidence overrides the family
    rule where present: /previous-years/ forces archived, /current/ forces
    current (Essex's UG archive reuses identical filenames across years),
    and a year-named directory (/pgt/2020-21/...) MORE THAN ONE YEAR older
    than the newest year in the corpus forces archived - this catches legacy
    families whose filename stem was later renamed (no within-family
    successor exists even though the edition is clearly superseded), while
    the one-year grace keeps departments alive during the staggered
    start-of-year rollout when their new edition hasn't been published yet.
    Per-document year comes from effective_year() (docid.py), not raw
    normalize_year() - see its docstring for the PGT "January starts"
    content/folder-year mismatch this guards against."""
    kept = [d for d in documents.values() if d.get("keep")]

    corpus_max_year = max((effective_year(d["url"], d.get("academic_year")) for d in kept), default="")
    grace_floor = previous_year(corpus_max_year)

    max_year_per_family: dict[str, str] = {}
    for doc in kept:
        family = document_family(doc["url"])
        year = effective_year(doc["url"], doc.get("academic_year"))
        if family not in max_year_per_family or year > max_year_per_family[family]:
            max_year_per_family[family] = year

    flags = {}
    for doc in kept:
        url = doc["url"]
        year = effective_year(url, doc.get("academic_year"))
        year_dir = YEAR_DIR_RE.search(url)
        if "/previous-years/" in url:
            flags[url] = False
        elif "/current/" in url:
            flags[url] = True
        elif year_dir and normalize_year(year_dir.group(1)) < grace_floor and year < grace_floor:
            flags[url] = False
        else:
            flags[url] = year == max_year_per_family[document_family(url)]
    return flags


def recompute_current_flags() -> None:
    """Update is_current and academic_year_norm in chunk metadata only - no
    re-embedding. Run after any incremental crawl (run_ingest.py does so
    automatically). Batched: one full-collection read, grouped in memory,
    one update per changed document."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    documents = manifest["documents"]
    flags = compute_current_flags(documents)
    year_norms = {
        d["url"]: effective_year(d["url"], d.get("academic_year"))
        for d in documents.values() if d.get("keep")
    }

    collection = _get_collection()
    data = collection.get(include=["metadatas"])

    by_url: dict[str, list[tuple[str, dict]]] = {}
    for id_, meta in zip(data["ids"], data["metadatas"]):
        by_url.setdefault(meta.get("source_url", ""), []).append((id_, meta))

    updated = 0
    for url, chunks in by_url.items():
        if url not in flags:
            continue
        is_current = flags[url]
        year_norm = year_norms[url]
        stale = [
            (id_, meta) for id_, meta in chunks
            if meta.get("is_current") != is_current or meta.get("academic_year_norm") != year_norm
        ]
        if not stale:
            continue
        collection.update(
            ids=[id_ for id_, _ in stale],
            metadatas=[{**meta, "is_current": is_current, "academic_year_norm": year_norm}
                       for _, meta in stale],
        )
        updated += len(stale)

    if updated:
        bump_corpus_version()
    print(f"recompute_current_flags: updated {updated} chunks")


def _load_chunk_contexts(url: str, expected_len: int) -> list[str] | None:
    """Per-chunk situating context from generate_chunk_context.py's cache, if
    present and aligned to the current chunking. Most documents have none
    (single-document families, or outside the pilot scope) - that's normal,
    not an error."""
    from src.ingest import url_hash
    path = Path("data/chunk_context_cache") / f"{url_hash(url)}.json"
    if not path.exists():
        return None
    contexts = json.loads(path.read_text())
    return contexts if len(contexts) == expected_len else None


def run() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    documents = manifest["documents"]
    kept = [d for d in documents.values() if d.get("keep")]
    flags = compute_current_flags(documents)

    for i, doc in enumerate(kept, 1):
        text_path = Path(doc["text_cache_path"])
        text = text_path.read_text(encoding="utf-8")
        metadata = {
            "title": doc["title"],
            "doc_type": doc["doc_type"],
            "department": doc.get("department"),
            "academic_year": doc.get("academic_year"),
            "is_current": flags[doc["url"]],
        }
        expected_chunks = chunk_text(clean_text(text))
        chunk_contexts = _load_chunk_contexts(doc["url"], len(expected_chunks))
        n_chunks = upsert_document(doc["url"], text, metadata, chunk_contexts=chunk_contexts)
        doc["chunk_count"] = n_chunks
        tag = " +context" if chunk_contexts else ""
        print(f"[{i}/{len(kept)}] re-embedded ({n_chunks} chunks{tag}): {doc['title']}", flush=True)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
