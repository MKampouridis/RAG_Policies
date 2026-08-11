#!/usr/bin/env python3
"""List and restore soft-deleted conversations.

Replaces the forensic page carver (recover_chat_history.py), which only ever
worked because SQLite had not yet reused the freed pages - luck, not design.
Since 2026-08-11 deletion sets `deleted_at` instead of destroying rows, so
recovery is an UPDATE.

    python restore_history.py                 # list what is recoverable
    python restore_history.py <id>            # restore one
    python restore_history.py --all           # restore everything deleted
    python restore_history.py --purge 30      # PERMANENTLY drop rows deleted >30 days ago
"""
import sys
from datetime import datetime

from src import memory


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--purge":
        days = float(args[1]) if len(args) > 1 else 30.0
        n = memory.purge_deleted(older_than_seconds=days * 24 * 3600)
        print(f"permanently removed {n} conversation(s) deleted more than {days:g} days ago")
        return 0

    deleted = memory.list_deleted_conversations()
    if not deleted:
        print("nothing is soft-deleted - no conversation is recoverable this way.")
        return 0

    if not args:
        print(f"{len(deleted)} recoverable conversation(s):\n")
        for r in deleted:
            when = datetime.fromtimestamp(r["deleted_at"]).strftime("%Y-%m-%d %H:%M")
            print(f"  {r['id']}  deleted {when}  {(r['title'] or '')[:52]}")
        print("\nrestore one:  python restore_history.py <id>")
        print("restore all:  python restore_history.py --all")
        return 0

    targets = [r["id"] for r in deleted] if args[0] == "--all" else [args[0]]
    done = sum(1 for cid in targets if memory.restore_conversation(cid))
    print(f"restored {done} of {len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
