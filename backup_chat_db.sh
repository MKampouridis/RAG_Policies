#!/bin/zsh
# Nightly snapshot of data/chat.db - the conversation history.
#
# WHY: chat.db is the only state in this project that is NOT in git and NOT
# reproducible. The corpus can be re-crawled, the index re-embedded, the eval
# re-run; conversations cannot. On 2026-08-11 "Clear all history" emptied it
# and the only reason anything survived was that SQLite leaves deleted rows on
# freed pages until they are reused - luck, not a backup.
#
# Uses sqlite3 .backup rather than cp: it takes a consistent snapshot even if
# the server is mid-write, which cp does not.
set -e
cd "$(dirname "$0")"
DIR=data/backups
mkdir -p "$DIR"
STAMP=$(date '+%Y%m%d-%H%M')
sqlite3 data/chat.db ".backup '$DIR/chat-$STAMP.db'"
# keep 14 nightlies; history is small (~4MB) but not worth unbounded growth
ls -1t "$DIR"/chat-*.db 2>/dev/null | tail -n +15 | xargs -I{} rm -f {} 2>/dev/null || true

# The JSONL side-stores, which this job skipped entirely. feedback.jsonl holds
# the ratings and - more to the point - the written comments, which are the
# least reproducible data in the project: a lost answer can be regenerated from
# the corpus, a colleague's explanation of WHY an answer was wrong cannot.
# events.jsonl and alert_history.jsonl are cheaper to lose but cost nothing to
# carry. One tarball per run keeps the restore story simple: untar, and the
# files are back where they were.
SIDE=()
for f in data/feedback.jsonl data/events.jsonl data/alert_history.jsonl; do
  [ -f "$f" ] && SIDE+=("$f")
done
if [ ${#SIDE[@]} -gt 0 ]; then
  tar -czf "$DIR/side-$STAMP.tgz" "${SIDE[@]}"
  ls -1t "$DIR"/side-*.tgz 2>/dev/null | tail -n +15 | xargs -I{} rm -f {} 2>/dev/null || true
fi

echo "$(date '+%F %T') backed up -> $DIR/chat-$STAMP.db${SIDE:+ + side-$STAMP.tgz} ($(ls -1 "$DIR"/chat-*.db | wc -l | tr -d ' ') kept)"
