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

# PGRE files its progress milestones in per-year directories
# (.../pgre/milestones-2025-26/ce-phd-2025-26.pdf), so the PUBLISHER states the
# academic year in the path - a stronger signal than anything inferable from
# the filename. YEAR_DIR_RE does not match these: it requires the path segment
# to BE the year, and here the year carries a "milestones-" prefix.
#
# That mattered (2026-08-28). Ingesting the PGRE archive (578 documents, nine
# years, reached for the first time via the PGR-progress webpage) left 157
# historical documents flagged current, including 11 from 2017-18 and 33 from
# 2018-19. Cause: the family rule marks the newest edition WITHIN a family, and
# PGRE renamed its files repeatedly across those years - department codes
# changed (csee->ce, psychology->py, langling->lt, iser->rc), a "-milestones"
# token was added, "(accessible)" variants appeared - so a department that
# stopped publishing under an old filename left its last edition as that
# family's permanent maximum. 158 distinct family keys were affected.
#
# _FAMILY_ALIASES is the usual remedy, but audit_family_aliases.py proposes
# only 5 of these by design (it refuses to merge stems differing by a whole
# word, which is exactly what these renames do), and hand-writing ~158 mappings
# invites a wrong merge - `ll-phd-mono` and `ll-phd-by-papers` are genuinely
# different documents, not editions of one.
#
# So currency for these is read off the directory instead, which needs no
# filename inference at all. Same shape as the /previous-years/ and /current/
# path rules below, and the blast radius is exactly the URLs this matches:
# 578 PGRE milestone documents, no others.
PGRE_MILESTONE_DIR_RE = re.compile(r"/pgre/milestones-(20\d{2}-\d{2,4})/")


# Explicitly superseded documents (2026-08-11). Essex CONSOLIDATED the
# per-degree-length UG variations files into one file per year:
#
#   roa-ug-3yr-year-1-variations.pdf  ─┐
#   roa-ug-4yr-year-1-variations.pdf  ─┴─>  year-1-variations-ug.pdf
#
# The weekly watcher caught the old URLs going unreachable while the new ones
# appeared. Nothing else demotes them: the family rule picks the newest year
# within a family, and BOTH carry 2025-26, so it is a tie; and the filename
# schemes are too different for a _FAMILY_ALIASES entry to group them. The old
# 3yr and 4yr files were already byte-identical to each other (97,229 chars),
# so Year 1 variations existed THREE times in the index, all flagged current -
# which is why a general framework question came back with five slots of
# variations documents.
#
# Listed rather than deleted: this is reversible, and retrieval already filters
# on is_current, so demotion is sufficient. If Essex relinks them, remove the
# entry rather than re-crawling.
SUPERSEDED_URLS = {
    "https://www.essex.ac.uk/-/media/documents/directories/academic-section/rules-of-assessment/ug/current/3-year-honours-degrees/roa-ug-3yr-year-1-variations.pdf",
    "https://www.essex.ac.uk/-/media/documents/directories/academic-section/rules-of-assessment/ug/current/3-year-honours-degrees/roa-ug-3yr-year-2-variations.pdf",
    "https://www.essex.ac.uk/-/media/documents/directories/academic-section/rules-of-assessment/ug/current/3-year-honours-degrees/roa-ug-3yr-final-year-variations.pdf",
    "https://www.essex.ac.uk/-/media/documents/directories/academic-section/rules-of-assessment/ug/current/4-year-honours-degrees/roa-ug-4yr-year-1-variations.pdf",
    "https://www.essex.ac.uk/-/media/documents/directories/academic-section/rules-of-assessment/ug/current/4-year-honours-degrees/roa-ug-4yr-year-2-variations.pdf",
    "https://www.essex.ac.uk/-/media/documents/directories/academic-section/rules-of-assessment/ug/current/4-year-honours-degrees/roa-ug-4yr-year-3-variations.pdf",
    "https://www.essex.ac.uk/-/media/documents/directories/academic-section/rules-of-assessment/ug/current/4-year-honours-degrees/roa-ug-4yr-year-4-variations.pdf",
}


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
    PGRE milestone directories (/pgre/milestones-2025-26/...) are decided by
    their directory year alone - see PGRE_MILESTONE_DIR_RE for why the family
    rule cannot be trusted there.
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

    # newest PGRE milestone directory present in the corpus - the comparison
    # point for the path rule below. Scoped to PGRE's own archive rather than
    # corpus_max_year so a newer document elsewhere can never archive the whole
    # milestone set during a staggered publication window.
    pgre_max_year = max(
        (normalize_year(m.group(1))
         for d in kept
         if (m := PGRE_MILESTONE_DIR_RE.search(d["url"]))),
        default="",
    )

    flags = {}
    for doc in kept:
        url = doc["url"]
        year = effective_year(url, doc.get("academic_year"))
        year_dir = YEAR_DIR_RE.search(url)
        pgre_dir = PGRE_MILESTONE_DIR_RE.search(url)
        if url in SUPERSEDED_URLS:
            flags[url] = False
            continue
        if "/previous-years/" in url:
            flags[url] = False
        elif "/current/" in url:
            flags[url] = True
        elif pgre_dir:
            # directory year is authoritative here - see PGRE_MILESTONE_DIR_RE
            flags[url] = normalize_year(pgre_dir.group(1)) == pgre_max_year
        elif year_dir and normalize_year(year_dir.group(1)) < grace_floor and year < grace_floor:
            flags[url] = False
        else:
            flags[url] = year == max_year_per_family[document_family(url)]
    return flags


def _patch_colbert_snapshot(flags: dict[str, bool], year_norms: dict[str, str]) -> int:
    """data/colbert_docs.json is a frozen metadata snapshot taken at
    build_colbert_index.py time (external code review, 2026-07-21: found
    this snapshot still carried the pre-fix is_current for the 109 chunks
    patched by the is_current bug fix earlier this session - Chroma had the
    correct flags, the ColBERT index's own copy didn't, since this function
    only ever updated Chroma). No-ops (returns 0) if the index hasn't been
    built - nothing to patch, and this must never be the thing that requires
    building it. Metadata-only, same as the Chroma update above - the token
    embeddings themselves don't depend on currency/year."""
    path = Path("data/colbert_docs.json")
    if not path.exists():
        return 0
    snapshot = json.loads(path.read_text())
    updated = 0
    for meta in snapshot["metadatas"]:
        url = meta.get("source_url", "")
        if url not in flags:
            continue
        is_current, year_norm = flags[url], year_norms[url]
        if meta.get("is_current") != is_current or meta.get("academic_year_norm") != year_norm:
            meta["is_current"] = is_current
            meta["academic_year_norm"] = year_norm
            updated += 1
    if updated:
        path.write_text(json.dumps(snapshot, ensure_ascii=False))
    return updated


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

    colbert_updated = _patch_colbert_snapshot(flags, year_norms)
    if colbert_updated:
        print(f"recompute_current_flags: also patched {colbert_updated} chunks in data/colbert_docs.json")

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
