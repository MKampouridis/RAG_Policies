"""SQLite-backed conversation memory: conversations persist across server
restarts and are resumable from either machine that points at the same
data/chat.db."""

import sqlite3
import threading
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
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets readers and the writer work at the same time. Under the default
    # rollback journal a single writer blocks every reader, so the moment two
    # people use this at once one of them gets "database is locked" - the
    # server is already multi-threaded (FastAPI runs sync endpoints in a
    # threadpool, and the streaming endpoint adds a worker thread per request),
    # so this is reachable with ONE user on two tabs. `timeout` above makes a
    # contended write wait rather than fail instantly.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
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
        # Soft delete (2026-08-11). "Clear all history" has destroyed every
        # conversation twice; the first cause was found (a button that looked
        # inert, so it was pressed repeatedly) and the SECOND IS STILL UNKNOWN.
        # Recovery both times meant carving freed SQLite pages, which only
        # worked because the pages had not been reused yet - luck, not design.
        # Rather than keep hunting an unreproduced cause, deletion no longer
        # destroys anything: it stamps deleted_at and the rows stay. Whatever
        # the cause is, it can no longer lose data.
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN deleted_at REAL DEFAULT NULL")
        except sqlite3.OperationalError:
            pass  # column already exists
        _migrated = True
    return conn


def conversation_exists(conversation_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND deleted_at IS NULL",
            (conversation_id,)
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
            "SELECT id, title, created_at FROM conversations"
            " WHERE deleted_at IS NULL ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def add_message(conversation_id: str, role: str, content: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, time.time()),
        )


def delete_conversation(conversation_id: str) -> None:
    """Soft delete: the conversation disappears from every read path but the
    rows remain. See the migration note in _connect() for why - two total
    losses, one cause still unexplained, and both recoveries depended on freed
    pages happening not to have been reused yet."""
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
            (time.time(), conversation_id),
        )


def restore_conversation(conversation_id: str) -> bool:
    """Undo a soft delete. Returns whether a deleted row was actually found."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE conversations SET deleted_at = NULL WHERE id = ? AND deleted_at IS NOT NULL",
            (conversation_id,),
        )
        return cur.rowcount > 0


def list_deleted_conversations() -> list[dict]:
    """What a soft delete is hiding - the recovery path that used to require a
    forensic page carver."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, deleted_at FROM conversations"
            " WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def purge_deleted(older_than_seconds: float = 30 * 24 * 3600) -> int:
    """Permanently remove conversations soft-deleted longer ago than the
    window. NOT called from any endpoint - deliberately a manual operation, so
    nothing automatic can ever destroy history again."""
    cutoff = time.time() - older_than_seconds
    with _connect() as conn:
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM conversations WHERE deleted_at IS NOT NULL AND deleted_at < ?",
            (cutoff,)).fetchall()]
        for cid in ids:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
    return len(ids)


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


# Per-conversation lock. get_conversation_context reads the summary watermark,
# decides whether to summarise, calls the LLM, then writes the new watermark -
# so two requests on the same conversation could both observe the same
# watermark and summarise the same messages twice. Reachable today: the server
# runs sync endpoints in a threadpool and the streaming endpoint adds a worker
# thread, so one user with two tabs is enough.
_conv_locks: dict[str, threading.Lock] = {}
_conv_locks_guard = threading.Lock()


def _conversation_lock(conversation_id: str) -> threading.Lock:
    with _conv_locks_guard:
        return _conv_locks.setdefault(conversation_id, threading.Lock())


def get_conversation_context(conversation_id: str) -> tuple[str, list[dict]]:
    """Returns (summary, recent_messages) for use as prompt history, prior to
    the current turn. Messages already folded into the rolling summary are
    tracked via the summarized_through watermark (a message id), so each
    message is summarized exactly once: the summarizer LLM call fires only
    when the UNSUMMARIZED tail exceeds MAX_TURNS_BEFORE_SUMMARY, roughly once
    per (MAX - KEEP) new messages - not on every turn of a long conversation.

    Serialised per conversation: the read-decide-summarise-write sequence below
    is not atomic, so concurrent turns on the same conversation could summarise
    the same messages twice and write conflicting watermarks."""
    with _conversation_lock(conversation_id):
        return _get_conversation_context_locked(conversation_id)


def _get_conversation_context_locked(conversation_id: str) -> tuple[str, list[dict]]:
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
