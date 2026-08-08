#!/usr/bin/env python3
"""Stale-index audit (2026-08-08): is every current document's INDEXED text
still what the pipeline would produce from its cached source today?

Found while chasing a real user complaint. roa-ug-integrated-masters-4yr-year-1
answers nothing about failing to progress, because the words "withdraw" and
"situations" appear in NO chunk of that document - the indexed chunk reads
"...they - Where a student was absent fr..." where the source reads "...they
must withdraw from the University in any of the following situations: ...".

Root cause is NOT a chunking bug: chunk_text() is a clean sliding window, and
re-chunking the current cache reproduces the missing words. The stored index
was simply built from an older, poorer PDF extraction; the text cache was later
refreshed with a better one and never re-embedded. Diagnostic that settles it:
the index contains NO words absent from the cache (so it is not a different
document), only the reverse.

Method: for each current document, re-chunk the cached source with the real
chunk_text(), and report any word present in that fresh output but missing from
every stored chunk. Using the pipeline's own chunker matters - an earlier
ad-hoc normalisation produced a 37% false-positive-laden estimate that had to
be retracted (report.md).

Small word counts still matter: losing "withdraw" costs the operative verb of a
progression rule and makes it unretrievable, no matter how good the reranker.

Fix for anything this flags: re-embed those documents (reembed.py).

Usage: PYTHONPATH=. python eval/stale_index_audit.py
"""
import json, pathlib, re
from src.ingest import chunk_text, _get_collection
m = json.load(open('data/manifest.json'))['documents']
coll = _get_collection()
got = coll.get(include=['documents','metadatas'])
by_url, current = {}, set()
for d, mm in zip(got['documents'], got['metadatas']):
    u = mm.get('source_url')
    by_url.setdefault(u, []).append(d)
    if mm.get('is_current'):
        current.add(u)
words = lambda s: set(re.sub(r'[^a-z0-9 ]', ' ', s.lower()).split())

stale, checked, clean = [], 0, 0
for u in sorted(current):
    p = pathlib.Path((m.get(u) or {}).get('text_cache_path', ''))
    if not p.is_file() or u not in by_url:
        continue
    checked += 1
    fresh = words(' '.join(chunk_text(p.read_text(encoding='utf-8'))))
    idx = words(' '.join(by_url[u]))
    lost = fresh - idx
    if lost:
        stale.append((u.split('/')[-1], len(lost), sorted(lost)[:6]))
    else:
        clean += 1
print(f"current docs checked : {checked}")
print(f"index matches cache  : {clean}")
print(f"index MISSING content: {len(stale)}  ({len(stale)/max(checked,1)*100:.0f}%)")
print()
for f, n, ex in sorted(stale, key=lambda x: -x[1])[:10]:
    print(f"  {f[:48]:48s} {n:4d} words missing  e.g. {ex[:5]}")
