#!/usr/bin/env python3
"""CLI: crawl + classify + embed the configured seed URLs into the local
vector store.

Re-runnable: every page/PDF is re-fetched on each run (cheap — they're small
over HTTP), but a document whose content hash matches the last run is not
re-classified or re-embedded (the expensive steps), so refreshing the index
after Essex publishes new documents only pays for what actually changed.

Usage:
    python run_ingest.py                  # crawl the default seed URLs
    python run_ingest.py <url> [<url> ...]  # crawl the defaults plus extra seed URLs
"""

import json
import sys
from pathlib import Path

from reembed import compute_current_flags, recompute_current_flags
from src.crawler import crawl
from src.docid import document_family
from src.ingest import _get_collection, delete_document, upsert_document, url_hash
from src.relevance import classify

SEED_URLS = [
    "https://www.essex.ac.uk/governance-and-strategy/governance/policies",
    "https://www.essex.ac.uk/student/rules-of-assessment",
    "https://www.essex.ac.uk/student/rules-of-assessment/roa-pgt-dept-specific",
]

MANIFEST_PATH = Path("data/manifest.json")
TEXT_CACHE_DIR = Path("data/text_cache")


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"documents": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def _sync_family_siblings(url: str, flags: dict[str, bool]) -> None:
    """A newly-ingested current document may supersede its family siblings;
    flip any sibling whose stored chunk flag disagrees with the freshly
    computed one, so currency is correct immediately rather than only after
    the end-of-run recompute. Only runs when a new current doc arrives, so
    the per-sibling lookups are rare."""
    collection = _get_collection()
    family = document_family(url)
    for sib_url, sib_flag in flags.items():
        if sib_url == url or document_family(sib_url) != family:
            continue
        existing = collection.get(where={"source_url": sib_url}, include=["metadatas"])
        ids = existing.get("ids", [])
        if not ids or all(m.get("is_current") == sib_flag for m in existing["metadatas"]):
            continue
        collection.update(
            ids=ids,
            metadatas=[{**m, "is_current": sib_flag} for m in existing["metadatas"]],
        )


def run(seed_urls: list[str]) -> dict:
    manifest = load_manifest()
    documents = manifest["documents"]
    TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"fetched": 0, "kept": 0, "rejected": 0, "skipped_unchanged": 0, "errors": 0}

    def on_item(item):
        stats["fetched"] += 1
        print(f"[{stats['fetched']}] fetched {item.content_type}: {item.url}", flush=True)

        prior = documents.get(item.url)
        cache_path = TEXT_CACHE_DIR / f"{url_hash(item.url)}.txt"

        if prior and prior.get("content_hash") == item.content_hash:
            stats["skipped_unchanged"] += 1
            return

        cache_path.write_text(item.text, encoding="utf-8")

        if not item.text.strip():
            decision = {
                "keep": False, "doc_type": "none",
                "department": None, "academic_year": None,
                "reason": "no extractable text",
            }
        else:
            try:
                decision = classify(item.title, item.url, item.text)
            except Exception as exc:
                stats["errors"] += 1
                decision = {
                    "keep": False, "doc_type": "none",
                    "department": None, "academic_year": None,
                    "reason": f"classification error: {exc}",
                }

        entry = {
            "url": item.url,
            "title": item.title,
            "content_type": item.content_type,
            "content_hash": item.content_hash,
            "text_cache_path": str(cache_path),
            **decision,
        }
        documents[item.url] = entry

        if decision["keep"]:
            # compute the currency flag NOW, against the up-to-date in-memory
            # manifest, so chunks are never written without is_current - a
            # crawl that crashes before the end-of-run recompute must not
            # leave documents invisible to the default retrieval filter
            flags = compute_current_flags(documents)
            metadata = {
                "title": item.title,
                "doc_type": decision["doc_type"],
                "department": decision.get("department"),
                "academic_year": decision.get("academic_year"),
                "is_current": flags[item.url],
            }
            try:
                n_chunks = upsert_document(item.url, item.text, metadata)
                entry["chunk_count"] = n_chunks
                stats["kept"] += 1
                print(f"    KEEP ({decision['doc_type']}, {n_chunks} chunks): {item.title}", flush=True)
                if flags[item.url]:
                    # this doc may have just superseded family siblings -
                    # flip any sibling whose stored flag disagrees
                    _sync_family_siblings(item.url, flags)
            except Exception as exc:
                stats["errors"] += 1
                entry["embed_error"] = str(exc)
                print(f"    EMBED ERROR: {exc}", flush=True)
        else:
            if prior and prior.get("keep"):
                # content changed and no longer qualifies - drop stale chunks
                delete_document(item.url)
            stats["rejected"] += 1
            print(f"    reject ({decision.get('reason', '')})", flush=True)

        # persist after every item so an interrupted run doesn't lose progress
        save_manifest(manifest)

    crawl(seed_urls, on_item=on_item)
    save_manifest(manifest)

    # global safety net: reconcile every document's flags in one batched
    # pass (upsert-time flags above cover the common cases, but a crawl can
    # change family maxima in ways only a full recompute sees)
    recompute_current_flags()

    return stats


if __name__ == "__main__":
    extra = sys.argv[1:]
    urls = SEED_URLS + [u for u in extra if u not in SEED_URLS]
    result = run(urls)
    print("\n=== Ingestion summary ===")
    for key, value in result.items():
        print(f"{key}: {value}")
