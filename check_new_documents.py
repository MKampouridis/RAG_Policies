#!/usr/bin/env python3
"""Weekly watch for new or changed Essex policy / rules-of-assessment documents.

DETECT ONLY - this never ingests anything. Automatically pulling new documents
into the corpus would silently change what every answer is based on, and
silently invalidate the eval ledger, with nobody having decided to. So this
reports and stops; `python run_ingest.py` remains a deliberate human action.

Crawls the same SEED_URLS as run_ingest.py, using the same polite crawler
(robots.txt honoured, 0.7s between requests), and compares what it finds
against data/manifest.json:

  NEW      - a document URL the manifest has never seen
  CHANGED  - a known URL whose content hash now differs (Essex republished it)
  GONE     - a known, kept document no longer reachable from the seed pages

Writes data/new_documents_report.md and prints a one-line summary. Exit code is
0 normally, so a scheduler does not treat "found something" as a failure.

Usage:
    python check_new_documents.py            # crawl and report
    python check_new_documents.py --quiet    # only write the report
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_ingest import SEED_URLS, _EXCLUDED_URLS
from src.crawler import crawl

MANIFEST_PATH = Path("data/manifest.json")
REPORT_PATH = Path("data/new_documents_report.md")
STATE_PATH = Path("data/new_documents_seen.json")


def _load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        return {}
    return json.loads(MANIFEST_PATH.read_text()).get("documents", {})


def main() -> int:
    quiet = "--quiet" in sys.argv
    known = _load_manifest()
    # URLs already reported as new, so a weekly run does not keep re-reporting
    # the same finding until someone ingests it
    seen_new = set(json.loads(STATE_PATH.read_text())) if STATE_PATH.is_file() else set()

    found: dict[str, str] = {}          # url -> content_hash

    def on_item(item):
        url = getattr(item, "url", None)
        if not url or url in _EXCLUDED_URLS:
            return
        found[url] = getattr(item, "content_hash", "") or ""

    crawl(SEED_URLS, on_item=on_item)

    new, changed = [], []
    for url, h in sorted(found.items()):
        rec = known.get(url)
        if rec is None:
            new.append(url)
        elif h and rec.get("content_hash") and h != rec.get("content_hash"):
            changed.append((url, rec.get("title") or ""))

    # only documents that were KEPT are worth flagging as gone; the crawl also
    # legitimately drops hub pages and rejected files
    gone = [u for u, r in known.items()
            if r.get("keep") and u not in found and u not in _EXCLUDED_URLS]

    unreported = [u for u in new if u not in seen_new]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Essex document watch — {ts}", "",
             f"Crawled {len(SEED_URLS)} seed pages, saw {len(found)} documents.", "",
             f"- **New:** {len(new)} ({len(unreported)} not previously reported)",
             f"- **Changed:** {len(changed)}",
             f"- **No longer reachable:** {len(gone)}", ""]
    if new:
        lines += ["## New documents", ""]
        lines += [f"- {'**(new since last check)** ' if u in unreported else ''}{u}" for u in new] + [""]
    if changed:
        lines += ["## Changed since last ingest", ""]
        lines += [f"- {t or '(untitled)'} — {u}" for u, t in changed] + [""]
    if gone:
        lines += ["## No longer reachable from the seed pages", ""]
        lines += [f"- {u}" for u in gone[:40]]
        if len(gone) > 40:
            lines += [f"- ... and {len(gone) - 40} more"]
        lines += [""]
    if new or changed:
        lines += ["---", "",
                  "To bring these into the corpus (a deliberate step — it changes what every",
                  "answer is based on, and re-baselines the eval ledger):", "",
                  "```", "python run_ingest.py", "python audit_family_aliases.py    # review new rename-split aliases",
                  "python eval/stale_index_audit.py  # confirm cleaning did not drop content", "```", ""]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    STATE_PATH.write_text(json.dumps(sorted(set(new) | seen_new), indent=1))

    if not quiet:
        print(f"[{ts}] documents seen {len(found)} | new {len(new)} "
              f"({len(unreported)} unreported) | changed {len(changed)} | gone {len(gone)}")
        print(f"report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
