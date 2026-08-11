#!/usr/bin/env python3
"""Recover deleted conversations from data/chat.db's freed pages.

WHY THIS EXISTS
"Clear all history" in the preview UI deleted every conversation (2026-08-11).
The button worked; its only feedback was a small muted line on the Settings
screen, so it read as doing nothing. SQLite does not zero pages on DELETE - it
just unlinks rows from the b-tree - so the rows survive until those pages are
reused. Production was stopped immediately to prevent that.

Reads a FORENSIC COPY, never the live database, and writes a separate recovery
file. It does not touch data/chat.db, so running it cannot make things worse.

Usage:
    python recover_chat_history.py <forensic-copy.db> <output.db>
"""

import collections
import pathlib
import re
import sqlite3
import sys
import time

UUID = rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    summary TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
"""


def carve(raw: bytes):
    """Message rows store conversation_id, then role, then content adjacently,
    so the triple is still recognisable on the page after the row is unlinked."""
    messages = []
    for m in re.finditer(UUID + rb"(user|assistant)([ -~\n\t]{1,20000})", raw):
        body = m.group(2).decode("utf-8", "ignore").strip()
        if len(body) < 3:
            continue
        messages.append((m.group(0)[:36].decode(), m.group(1).decode(), body))

    titles = {}
    for m in re.finditer(UUID + rb"([ -~]{4,120})", raw):
        t = m.group(1).decode("utf-8", "ignore").strip()
        if t.startswith(("user", "assistant")):
            continue
        if 3 < len(t) < 120:
            titles.setdefault(m.group(0)[:36].decode(), t.rstrip("A").strip())
    return messages, titles


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    raw = src.read_bytes()
    messages, titles = carve(raw)
    print(f"message rows carved: {len(messages)}")

    grouped = collections.defaultdict(list)
    for cid, role, body in messages:
        # the same row can appear more than once across page copies
        if (role, body) not in grouped[cid]:
            grouped[cid].append((role, body))
    print(f"conversations with recovered messages: {len(grouped)}")

    if dst.exists():
        dst.unlink()
    db = sqlite3.connect(dst)
    db.executescript(SCHEMA)
    now = time.time()
    for i, (cid, rows) in enumerate(grouped.items()):
        title = titles.get(cid) or (rows[0][1][:60] if rows else "Recovered conversation")
        db.execute(
            "INSERT OR IGNORE INTO conversations (id,title,created_at,summary) VALUES (?,?,?,'')",
            (cid, title, now - (len(grouped) - i)),
        )
        for j, (role, body) in enumerate(rows):
            db.execute(
                "INSERT INTO messages (conversation_id,role,content,created_at) VALUES (?,?,?,?)",
                (cid, role, body, now - (len(grouped) - i) + j * 0.001),
            )
    db.commit()
    nc = db.execute("select count(*) from conversations").fetchone()[0]
    nm = db.execute("select count(*) from messages").fetchone()[0]
    print(f"\nwrote {dst}: {nc} conversations, {nm} messages")
    for (t,) in db.execute("select title from conversations limit 6"):
        print("   ", t[:66])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
