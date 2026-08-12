#!/usr/bin/env python3
"""Explain a stored answer: what produced it, and whether its sources have moved.

WHY: for a policy assistant the awkward question is not "was this right?" but
"was this right WHEN IT WAS GIVEN, and is it still right now?". An answer can be
perfectly faithful to a rule that has since been rewritten. Until 2026-08-12
neither half was answerable - nothing recorded which corpus produced an answer,
and staleness was only checkable live from the browser.

Each assistant message now stores its provenance and cited sources, so this
reads them back and compares against the current manifest.

Usage:
    PYTHONPATH=. python eval/audit_answer.py <conversation_id>
"""
import json
import pathlib
import sys
from datetime import datetime


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    from src import memory

    cid = sys.argv[1]
    if not memory.conversation_exists(cid):
        print(f"no such conversation (or it is deleted): {cid}")
        return 1

    manifest = {}
    mp = pathlib.Path("data/manifest.json")
    if mp.is_file():
        manifest = json.loads(mp.read_text()).get("documents", {})

    for m in memory.get_messages(cid):
        when = datetime.fromtimestamp(m["created_at"]).strftime("%Y-%m-%d %H:%M")
        if m["role"] != "assistant":
            print(f"\n  [{when}] USER: {m['content'][:90]}")
            continue
        meta = json.loads(m["meta"]) if m.get("meta") else None
        print(f"  [{when}] ANSWER ({len(m['content'])} chars)")
        if not meta:
            print("     no provenance recorded - predates 2026-08-12")
            continue
        p = meta.get("provenance", {})
        print(f"     generator {p.get('generator')} · corpus {str(p.get('corpus_version'))[:12]}"
              f" · build {p.get('code_revision')}")
        moved = []
        for url in meta.get("sources") or []:
            rec = manifest.get(url) or {}
            changed = rec.get("content_changed_at")
            if changed and changed > m["created_at"]:
                moved.append((url, changed))
        if moved:
            print(f"     *** {len(moved)} cited document(s) CHANGED since this answer:")
            for url, ts in moved:
                print(f"         {url.rsplit('/', 1)[-1][:56]}"
                      f"  (changed {datetime.fromtimestamp(ts):%Y-%m-%d})")
            print("     The answer may have been correct when given and wrong now.")
        else:
            print(f"     {len(meta.get('sources') or [])} cited document(s), none changed since")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
