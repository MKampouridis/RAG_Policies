#!/bin/zsh
# Weekly document watch, run by com.mkampo.ragpolicieswatch (Sundays 23:00).
#
# Deliberately does NOT ingest. It crawls, compares against the manifest and
# writes data/new_documents_report.md; pulling anything into the corpus stays a
# human decision, because it changes what every answer is based on and
# re-baselines the eval ledger.
#
# Safe to run while production is up: it only reads the manifest and makes
# outbound HTTP requests - no Chroma access, no model loading, so it does not
# compete with the server for RAM.
set -e
cd "$(dirname "$0")"

LOG="data/document_watch.log"
mkdir -p data
echo "=== $(date '+%F %T') starting weekly document watch ===" >> "$LOG"

if .venv/bin/python3 check_new_documents.py >> "$LOG" 2>&1; then
  SUMMARY=$(grep -m1 -E '^\- \*\*New:' data/new_documents_report.md 2>/dev/null || echo "see report")
  NEW=$(sed -n 's/^- \*\*New:\*\* \([0-9]*\).*/\1/p' data/new_documents_report.md 2>/dev/null | head -1)
  CHANGED=$(sed -n 's/^- \*\*Changed:\*\* \([0-9]*\).*/\1/p' data/new_documents_report.md 2>/dev/null | head -1)
  echo "$(date '+%F %T') done: new=${NEW:-?} changed=${CHANGED:-?}" >> "$LOG"
  # A notification is the only reason this is noticed on a Monday morning; a
  # report nobody opens is the same as no report.
  osascript -e "display notification \"New: ${NEW:-?} · Changed: ${CHANGED:-?}. See data/new_documents_report.md\" with title \"Essex policy watch\" sound name \"Glass\"" 2>/dev/null || true
else
  echo "$(date '+%F %T') FAILED - see log above" >> "$LOG"
  osascript -e 'display notification "Weekly document watch failed - see data/document_watch.log" with title "Essex policy watch"' 2>/dev/null || true
fi
