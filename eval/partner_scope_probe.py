#!/usr/bin/env python3
"""Measure the Essex/Partner scope switch on far more than set 6's 8 questions.

Set 6 is 8 hand-written questions. That is enough to show a mechanism works and
too few to trust a product decision. Two larger measurements are available
without writing any new questions:

  ESSEX SIDE (n=157): every replayed query that names no partner. These are
  ordinary Essex questions with a known gold document, so `essex_only` must not
  cost hit@6 against them. This is the regression risk.

  PARTNER SIDE (n=249): every partner document in the corpus. For each, a probe
  built from its own title asks "can retrieval find this document at all?"
  under each mode. This is a RETRIEVAL test, not a question-answering test -
  the probe is derived from the document rather than written by a person - so
  it measures reachability, which is exactly what a scope filter changes.

The partner side is deliberately generous to `essex_only` (a title probe is the
easiest possible query); if `essex_only` still cannot reach these documents,
neither can a real user.

Usage:
    PYTHONPATH=. RAG_DETERMINISTIC=1 python eval/partner_scope_probe.py [--limit N]
"""
import json
import pathlib
import re
import sys


def title_probe(url: str, rec: dict) -> str:
    t = (rec.get("title") or "").strip()
    if not t:
        t = re.sub(r"[-_]+", " ", url.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    return re.sub(r"\s+", " ", t)[:180]


def main() -> int:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    from src import rag
    from src.rag import _is_partner_institution as isp

    docs = json.loads(pathlib.Path("data/manifest.json").read_text())["documents"]
    partner = [(u, r) for u, r in docs.items() if r.get("keep") and isp({"source_url": u})]
    if limit:
        partner = partner[:limit]

    print(f"\n  PARTNER SIDE — {len(partner)} partner documents, title probes\n")
    res = {"essex_only": 0, "partner_only": 0, "default": 0}
    for i, (url, rec) in enumerate(partner, 1):
        q = title_probe(url, rec)
        for mode in res:
            pm = None if mode == "default" else mode
            r, _ = rag.retrieve(q, [], partner_mode=pm)
            urls = [m.get("source_url") for m in r.get("metadatas", [[]])[0]][:6]
            if url in urls:
                res[mode] += 1
        if i % 25 == 0:
            print(f"    {i}/{len(partner)}  essex_only={res['essex_only']} "
                  f"partner_only={res['partner_only']} default={res['default']}")

    n = len(partner)
    print(f"\n  {'mode':<16}{'found':>8}{'rate':>9}")
    for m in ("default", "essex_only", "partner_only"):
        print(f"  {m:<16}{res[m]:>8}{res[m]/n*100:>8.1f}%")
    pathlib.Path("eval/partner_scope_probe_result.json").write_text(
        json.dumps({"n": n, "found": res}, indent=1))
    print(f"\n  wrote eval/partner_scope_probe_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
