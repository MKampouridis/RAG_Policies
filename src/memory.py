"""SQLite-backed conversation memory: conversations persist across server
restarts and are resumable from either machine that points at the same
data/chat.db."""

import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = "data/chat.db"

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

# Once a conversation exceeds this many stored messages, the oldest ones are
# folded into a rolling summary so prompts don't grow unbounded on a
# long-running thread.
MAX_TURNS_BEFORE_SUMMARY = 20
TURNS_TO_KEEP_AFTER_SUMMARY = 10


_migrated = False


def _connect() -> sqlite3.Connection:
    global _migrated
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not _migrated:
        # schema creation + the summarized_through migration only need to run
        # once per process, not on every connection (every message send,
        # every history fetch) - re-running an ALTER TABLE wrapped in a
        # try/except on every call means paying real exception-handling
        # overhead on every DB access for the entire life of the process
        conn.executescript(SCHEMA)
        # migration for databases created before the summarization watermark:
        # summarized_through is the id of the last message already folded into
        # the rolling summary, so each message is summarized exactly once
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN summarized_through INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # column already exists
        _migrated = True
    return conn


def conversation_exists(conversation_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return row is not None


def create_conversation(title: str) -> str:
    conv_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
            (conv_id, title, time.time()),
        )
    return conv_id


def update_title(conversation_id: str, title: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))


def list_conversations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def add_message(conversation_id: str, role: str, content: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, time.time()),
        )


def delete_conversation(conversation_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def get_messages(conversation_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _get_summary_state(conversation_id: str) -> tuple[str, int]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT summary, summarized_through FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    if row is None:
        return "", 0
    return row["summary"] or "", row["summarized_through"] or 0


def _set_summary_state(conversation_id: str, summary: str, summarized_through: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET summary = ?, summarized_through = ? WHERE id = ?",
            (summary, summarized_through, conversation_id),
        )


def get_conversation_context(conversation_id: str) -> tuple[str, list[dict]]:
    """Returns (summary, recent_messages) for use as prompt history, prior to
    the current turn. Messages already folded into the rolling summary are
    tracked via the summarized_through watermark (a message id), so each
    message is summarized exactly once: the summarizer LLM call fires only
    when the UNSUMMARIZED tail exceeds MAX_TURNS_BEFORE_SUMMARY, roughly once
    per (MAX - KEEP) new messages - not on every turn of a long conversation."""
    summary, summarized_through = _get_summary_state(conversation_id)

    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE conversation_id = ? AND id > ? ORDER BY id ASC",
            (conversation_id, summarized_through),
        ).fetchall()
    pending = [dict(r) for r in rows]

    if len(pending) > MAX_TURNS_BEFORE_SUMMARY:
        to_fold = pending[:-TURNS_TO_KEEP_AFTER_SUMMARY]
        pending = pending[-TURNS_TO_KEEP_AFTER_SUMMARY:]
        summary = _summarize(summary, to_fold)
        _set_summary_state(conversation_id, summary, to_fold[-1]["id"])

    return summary, [
        {"role": m["role"], "content": m["content"], "created_at": m["created_at"]}
        for m in pending
    ]


def _summarize(existing_summary: str, messages: list[dict]) -> str:
    from src.llm import chat

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt = (
        "Summarize the following conversation turns into a short paragraph that "
        "preserves the topics discussed and any conclusions reached, so it can be "
        "used as context for continuing the conversation. Keep it under 200 words."
    )
    if existing_summary:
        prompt += f"\n\nExisting summary so far:\n{existing_summary}"
    prompt += f"\n\nNew turns to fold in:\n{transcript}"

    return chat(messages=[{"role": "user", "content": prompt}])
