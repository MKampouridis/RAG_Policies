#!/usr/bin/env python3
"""Separate the user's real conversations from eval-generated ones.

The recovery carved 1,825 conversations out of freed pages, but most were
created by eval/run_eval.py - every eval turn opens a conversation through the
same API. Restoring all of them would bury ~115 real ones in eval noise.

Three signals, strongest first:

  KEEP  conversation_id appears in data/feedback.jsonl - the user rated it, so
        it is unambiguously theirs.
  DROP  the opening question matches a question in any eval/questions*.json -
        eval runs replay those verbatim.
  KEEP  anything else: a question nobody wrote into a question set and that the
        eval never asked is, by elimination, typed by a human.

Reports by default. Pass --write <target.db> to build a filtered database.
Never touches data/chat.db.
"""

import json
import pathlib
import re
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def eval_questions() -> set[str]:
    out = set()
    for f in sorted((ROOT / "eval").glob("questions*.json")):
        try:
            for q in json.loads(f.read_text()):
                for k in ("question", "follow_up_question"):
                    if q.get(k):
                        out.add(_norm(q[k]))
        except Exception:
            continue
    return out


def rated_conversation_ids() -> set[str]:
    out = set()
    p = ROOT / "data" / "feedback.jsonl"
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            cid = json.loads(line).get("conversation_id")
        except Exception:
            continue
        if cid:
            out.add(cid)
    return out


def main() -> int:
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not src or not src.is_file():
        print(__doc__)
        return 2
    write_to = None
    if "--write" in sys.argv:
        write_to = pathlib.Path(sys.argv[sys.argv.index("--write") + 1])

    evalq = eval_questions()
    rated = rated_conversation_ids()
    print(f"eval questions known : {len(evalq)}")
    print(f"conversations rated  : {len(rated)}\n")

    db = sqlite3.connect(src)
    rows = db.execute("select id, title from conversations").fetchall()
    keep, drop = [], []
    for cid, title in rows:
        msgs = db.execute(
            "select role, content from messages where conversation_id=? order by created_at", (cid,)
        ).fetchall()
        first_user = next((c for r, c in msgs if r == "user"), "")
        # RATED ONLY (2026-08-11). The wider heuristic - "keep anything whose
        # opening question is not in a committed question set" - kept 1801 of
        # 1825, because question-GENERATION scripts also create conversations
        # through this API and their output is in no committed set. A feedback
        # record is the only signal that is certainly a human: the user pressed
        # a thumb on it.
        if cid in rated:
            keep.append((cid, title, msgs, "rated by the user"))
        else:
            drop.append((cid, title))

    print(f"KEEP {len(keep)}   DROP {len(drop)} (eval-generated or empty)\n")
    for cid, title, msgs, why in keep[:25]:
        print(f"  [{why:18}] {len(msgs):2} msgs  {title[:58]}")
    if len(keep) > 25:
        print(f"  ... and {len(keep) - 25} more")

    if write_to:
        if write_to.exists():
            write_to.unlink()
        out = sqlite3.connect(write_to)
        out.executescript(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL,"
            " created_at REAL NOT NULL, summary TEXT DEFAULT '');"
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,"
            " created_at REAL NOT NULL);"
        )
        for i, (cid, title, msgs, _why) in enumerate(keep):
            created = db.execute(
                "select created_at from conversations where id=?", (cid,)
            ).fetchone()[0]
            out.execute(
                "INSERT INTO conversations (id,title,created_at,summary) VALUES (?,?,?,'')",
                (cid, title, created),
            )
            for j, (role, content) in enumerate(msgs):
                out.execute(
                    "INSERT INTO messages (conversation_id,role,content,created_at)"
                    " VALUES (?,?,?,?)",
                    (cid, role, content, created + j * 0.001),
                )
        out.commit()
        nc = out.execute("select count(*) from conversations").fetchone()[0]
        nm = out.execute("select count(*) from messages").fetchone()[0]
        print(f"\nwrote {write_to}: {nc} conversations, {nm} messages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
