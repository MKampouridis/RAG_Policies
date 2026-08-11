#!/usr/bin/env python3
"""One-off: record content_changed_at for the documents that changed in the
2026-08-11 ingest.

run_ingest.py now stamps content_changed_at whenever a document's hash differs,
but that was added AFTER the ingest that refreshed ~20 documents - so the
staleness marker had no data and could never fire. The weekly watcher's report
names exactly which documents changed, and the manifest's mtime is when they
were written, so this is a record of something that genuinely happened, not a
fabricated timestamp.

Only backfills documents named in the report. Everything else is left without a
timestamp, which the staleness endpoint treats as "not known to have changed" -
the safe direction, since a false staleness marker teaches people to ignore it.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "manifest.json"
REPORT = ROOT / "data" / "new_documents_report.md"


def main() -> int:
    if not MANIFEST.is_file() or not REPORT.is_file():
        print("manifest or watch report missing")
        return 2

    report = REPORT.read_text()
    section = report.split("## Changed since last ingest")
    if len(section) < 2:
        print("no 'Changed' section in the report - nothing to backfill")
        return 0
    changed = set(re.findall(r"https://\S+", section[1].split("## ")[0]))

    # the ingest wrote the manifest, so its mtime is when those changes landed
    stamp = MANIFEST.stat().st_mtime

    data = json.loads(MANIFEST.read_text())
    docs = data.get("documents", {})
    n = 0
    for url in changed:
        rec = docs.get(url)
        if not rec or rec.get("content_changed_at"):
            continue
        # hub pages are not policy documents; a staleness marker citing one
        # would be noise
        if not rec.get("keep"):
            continue
        rec["content_changed_at"] = stamp
        n += 1

    if "--write" in sys.argv:
        MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"stamped {n} documents with content_changed_at={stamp:.0f}")
    else:
        print(f"would stamp {n} documents (pass --write to apply)")
        for url in list(changed)[:8]:
            rec = docs.get(url)
            if rec and rec.get("keep"):
                print(f"   {url.rsplit('/', 1)[-1][:66]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
