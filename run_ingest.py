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
import os
import subprocess
import time
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
    # HTML content page, not a PDF - see _INCLUDED_HTML_URLS below for why this
    # one is exempt from the PDF-only hub-page guard.
    "https://www.essex.ac.uk/student/postgraduate-research/pgr-progress",
]

MANIFEST_PATH = Path("data/manifest.json")
TEXT_CACHE_DIR = Path("data/text_cache")

# Durable duplicate-URL exclusion (2026-07-23). Essex sometimes publishes the
# SAME document at two URLs across a filename-scheme change. These URLs are
# byte-identical duplicates of a canonical file already in the corpus and must
# not be re-indexed on a future crawl (they add zero coverage and, worse, carry
# a directory-derived academic-year label that mislabels their true cohort -
# e.g. five-year-integrated-masters-21-v7.pdf sits under /2025-26/ but its
# content is the 2021-22 cohort, identical to roa-ug-integrated-masters-5yr-
# year-5.pdf). Same spirit as the hub-page guard below, but keyed by exact URL.
_EXCLUDED_URLS = {
    "https://www.essex.ac.uk/-/media/documents/directories/academic-section/rules-of-assessment/pgt/2025-26/masters-taught-courses/five-year-integrated-masters-21-v7.pdf",
}

# Explicit exemption from the PDF-only hub-page guard below (2026-08-28). The
# guard exists because EVERY real document in this corpus used to be a PDF -
# but the PGR Progress procedure has no PDF equivalent, only this webpage, and
# its content (verified by hand: 36KB of substantive procedural text in the
# page's richtext div, not a nav/listing page) is exactly the kind of thing
# this corpus should answer questions from. Scoped to this one URL rather than
# loosening the guard generally, since the guard's purpose - keeping lexical-
# magnet nav pages like roa-pgt-previous-years out of the index - still holds
# for every other non-PDF page the crawler reaches. classify() still makes the
# keep/reject call same as any PDF; this only lets it be asked the question.
_INCLUDED_HTML_URLS = {
    "https://www.essex.ac.uk/student/postgraduate-research/pgr-progress",
}


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

        if item.url in _EXCLUDED_URLS:
            decision = {
                "keep": False, "doc_type": "none",
                "department": None, "academic_year": None,
                "reason": "excluded duplicate URL (byte-identical to a canonical file)",
            }
        elif not item.text.strip():
            decision = {
                "keep": False, "doc_type": "none",
                "department": None, "academic_year": None,
                "reason": "no extractable text",
            }
        elif not item.url.lower().endswith(".pdf") and item.url not in _INCLUDED_HTML_URLS:
            # Durable hub-page guard (external code review round 3, 2026-07-22,
            # Fable 5, verified): every real document in this corpus is a PDF;
            # the crawl's HTML pages are navigation/listing hubs (e.g.
            # /rules-of-assessment/roa-pgt-previous-years lists every historical
            # programme name). 19 of them had slipped past the LLM classifier
            # and were indexed - and because they list every programme name,
            # they're lexical magnets that surface on identity queries while
            # containing no answerable rule, actively displacing the intended
            # PDF (seen polluting the Phase 5 mt8 top-6). The crawler still
            # FOLLOWS these pages to reach the PDFs; they just aren't indexed as
            # documents themselves. _INCLUDED_HTML_URLS is the explicit,
            # by-hand exemption list for the rare non-PDF page that IS real
            # content - see its comment above.
            decision = {
                "keep": False, "doc_type": "none",
                "department": None, "academic_year": None,
                "reason": "hub/navigation page (non-PDF)",
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

        # content_changed_at (2026-08-11): the timestamp a document's CONTENT
        # last differed from what we already had. Reaching this line at all
        # means the hash differed or there was no prior entry, so this is
        # exactly the moment of change. Used to mark stored answers as
        # potentially stale - an answer given before its cited document changed
        # may be quoting a rule that has since been rewritten, which for a
        # policy tool is the failure worth surfacing.
        #
        # A FIRST ingest is not a change, so a brand-new document inherits no
        # timestamp; otherwise every answer would look stale after the first
        # crawl. Existing documents keep whatever they already had until they
        # genuinely change.
        changed_at = time.time() if prior else (prior or {}).get("content_changed_at")
        entry = {
            "url": item.url,
            "title": item.title,
            "content_type": item.content_type,
            "content_hash": item.content_hash,
            "text_cache_path": str(cache_path),
            **({"content_changed_at": changed_at} if changed_at else {}),
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


def _listeners_on(port: int) -> list[str]:
    """PIDs/commands listening on a local TCP port, via lsof. Empty list when
    nothing is listening (or lsof is unavailable - absence of evidence, so the
    guard below fails OPEN rather than blocking a legitimate ingest)."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []
    return out[1:]  # drop the header row


def _refuse_if_server_live() -> None:
    """Refuse to ingest while a server holds the same Chroma store open.

    Found the hard way (2026-08-28): a full recrawl was launched while
    production was serving on :8000. Chroma's persistent client does NOT share
    writes across processes - the crawl wrote new documents from its own
    process, and the SERVER's cached segment state went on pointing at an index
    that no longer matched the metadata on disk. Every subsequent question
    failed with `InternalError: Error executing plan: Internal error: Error
    finding id`, and it did not heal on its own: the store on disk was fine
    (verified - a fresh process read 21912 chunks and queried them happily),
    so only restarting the server cleared it. Same class as eval_session.sh's
    :8001 guard, and the same reasoning: refuse rather than silently corrupt.

    NOT COVERED, deliberately stated: this detects a LISTENING SERVER on the
    two known ports, not "any process with the store open". A second ingest,
    a reembed.py, or a python REPL holding the collection would slip past it -
    the port is a proxy for the real condition, chosen because it catches the
    failure that actually happened at the cost of one lsof call. Override with
    RAG_INGEST_ALLOW_LIVE_SERVER=1 when you know the listener does not touch
    this store.
    """
    if os.environ.get("RAG_INGEST_ALLOW_LIVE_SERVER") == "1":
        return
    ports = {int(os.environ.get("PORT", "8000")), int(os.environ.get("EVAL_PORT", "8001"))}
    busy = {p: _listeners_on(p) for p in sorted(ports)}
    busy = {p: rows for p, rows in busy.items() if rows}
    if not busy:
        return
    print("!! a server is listening while this ingest would write to the same")
    print("   Chroma store - refusing to run. Its in-memory index would go stale")
    print("   against the new writes and every answer would fail until restart.")
    for port, rows in busy.items():
        print(f"   :{port}")
        for row in rows:
            print(f"     {row}")
    plist = Path.home() / "Library/LaunchAgents/com.mkampo.ragpolicies.plist"
    if plist.exists():
        print("\n   Stop production first (KeepAlive means unload, not kill):")
        print(f"     launchctl unload {plist}")
        print("   ...then re-run this, and afterwards:")
        print(f"     launchctl load {plist}")
    print("\n   Override with RAG_INGEST_ALLOW_LIVE_SERVER=1 if the listener")
    print("   does not share this store.")
    sys.exit(1)


if __name__ == "__main__":
    _refuse_if_server_live()
    extra = sys.argv[1:]
    urls = SEED_URLS + [u for u in extra if u not in SEED_URLS]
    result = run(urls)
    print("\n=== Ingestion summary ===")
    for key, value in result.items():
        print(f"{key}: {value}")
