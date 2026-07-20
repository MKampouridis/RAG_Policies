#!/usr/bin/env python3
"""J1: one-off per-document identity extraction. For every kept document,
feed the first ~2 pages of cached text plus the filename to the local chat
model and extract a structured identity record: programme name, department,
partner institution, awards conferred, and aliases/abbreviations a user
might say. Cached permanently to data/doc_identity/<url_hash>.json - never
regenerated unless the file is deleted.

This is the per-DOCUMENT (~1,189 calls) pass that was originally mispriced
as being as expensive as the rejected per-chunk contextual-embeddings pilot
(20,498 calls) - see eval/report.md "identity-first round". The identity
facts (programme/partner/aliases) live on title pages, not in filenames,
which is why every filename-derived facet attempt hit coverage gaps.

Usage: PYTHONPATH=. python extract_doc_identity.py
Resumable: skips documents whose output file already exists.
"""

import json
import time
from pathlib import Path

from src.ingest import url_hash
from src.llm import chat

MANIFEST_PATH = Path("data/manifest.json")
OUTPUT_DIR = Path("data/doc_identity")
FIRST_PAGES_CHARS = 3000  # roughly the first two pages of a text-extracted PDF

EXTRACT_SYSTEM_PROMPT = """You extract identity metadata from the opening pages of a university \
policy or rules-of-assessment document. Respond with ONLY a JSON object with these exact keys:
{
  "programme_name": "<full programme name(s) this document governs, e.g. 'MSc Periodontology', or '' if it is a general/university-wide document>",
  "department": "<school/department/centre name, e.g. 'East 15 Acting School', or ''>",
  "partner_institution": "<partner/collaborative institution name if this is a partner-institution document, or ''>",
  "awards": ["<award types conferred or defined, e.g. 'MSc', 'PGDip', 'PGCert', 'BA', 'Integrated Masters'>"],
  "aliases": ["<other ways a user might refer to this programme/document: abbreviations, informal names, campus names, e.g. 'East15', 'perio', 'HRM'>"]
}
Extract only what the text actually states - empty string/list when absent. Keep aliases short and realistic."""


def extract_one(filename: str, text_head: str) -> dict | None:
    raw = chat(
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Filename: {filename}\n\nOpening pages:\n{text_head}"},
        ],
        format="json",
    )
    try:
        parsed = json.loads(raw)
        return {
            "programme_name": str(parsed.get("programme_name", "") or ""),
            "department": str(parsed.get("department", "") or ""),
            "partner_institution": str(parsed.get("partner_institution", "") or ""),
            "awards": [str(a) for a in parsed.get("awards", []) if a],
            "aliases": [str(a) for a in parsed.get("aliases", []) if a],
        }
    except Exception as exc:
        print(f"    parse error: {exc}", flush=True)
        return None


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())["documents"]
    kept = [(url, doc) for url, doc in manifest.items() if doc.get("keep")]
    print(f"Extracting identity for {len(kept)} kept documents ...", flush=True)

    done = skipped = failed = 0
    t0 = time.time()
    for i, (url, doc) in enumerate(kept, 1):
        out_path = OUTPUT_DIR / f"{url_hash(url)}.json"
        if out_path.exists():
            skipped += 1
            continue
        text_path = Path(doc.get("text_cache_path", ""))
        if not text_path.exists():
            print(f"[{i}/{len(kept)}] NO TEXT CACHE for {url}", flush=True)
            failed += 1
            continue
        text_head = text_path.read_text(encoding="utf-8")[:FIRST_PAGES_CHARS]
        filename = url.rsplit("/", 1)[-1]
        identity = extract_one(filename, text_head)
        if identity is None:
            failed += 1
            continue
        identity["source_url"] = url
        out_path.write_text(json.dumps(identity, indent=2, ensure_ascii=False))
        done += 1
        if done % 25 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = len(kept) - i
            print(f"[{i}/{len(kept)}] {done} extracted ({rate:.2f}/s, ~{remaining / rate / 60:.0f} min remaining)", flush=True)

    print(f"Done. extracted={done} skipped_existing={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    run()
