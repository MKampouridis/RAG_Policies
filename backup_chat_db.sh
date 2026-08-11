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
echo "$(date '+%F %T') backed up -> $DIR/chat-$STAMP.db ($(ls -1 "$DIR"/chat-*.db | wc -l | tr -d ' ') kept)"
