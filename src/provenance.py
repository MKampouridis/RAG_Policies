"""Provenance stamps for derived artifacts, checked at read time.

WHY THIS EXISTS
This project has now been bitten twice by the same bug, three weeks apart:

  * The ColBERT embedding cache was built 2026-07-21 against a corpus that was
    re-ingested 2026-08-11. The reranker scored chunks by superseded wording
    for three weeks, costing 5 turns of hit@6, while every measurement in the
    ledger sat on top of it (Round 27).
  * The EVAL SET bundles gold documents chosen from one corpus snapshot. After
    the same re-ingest, 9 of 148 items graded retrieval against superseded
    documents - so returning the CURRENT edition scored as a MISS (Round 54).

Both are one class: **a derived artifact whose validity depends on an input
that can change independently, with nothing recording which input it was built
from.** `lexical.py` and `doc_index.py` already rebuild when
`read_corpus_version()` moves; nothing else did.

THE RULE
Every derived artifact records the version of what it was built from, that
record is checked when the artifact is READ, and a mismatch FAILS CLOSED rather
than warning. A warning in a log nobody reads is how three weeks passed.

Counts and modification times are proxies and this project has already seen one
fail: `colbert_index_drift.py` compared chunk COUNTS, so a document edited in
place with the same number of chunks passed it - and the last re-ingest was
5 new documents and ~20 CHANGED, which is precisely what a count cannot see.
"""

import json
import pathlib
from typing import Any

STAMP_KEY = "_provenance"


class StaleArtifact(RuntimeError):
    """A derived artifact was built from a different corpus than the live one."""


def current_corpus_version() -> str | None:
    from src.ingest import read_corpus_version

    return read_corpus_version()


def stamp(payload: Any, *, built_from: str | None = None, note: str = "") -> dict:
    """Wrap `payload` with a record of the corpus it was derived from."""
    return {
        STAMP_KEY: {
            "corpus_version": built_from if built_from is not None else current_corpus_version(),
            "note": note,
        },
        "payload": payload,
    }


def read_stamped(path: pathlib.Path, *, required: bool = True) -> Any:
    """Read a stamped artifact, refusing it if the corpus has moved.

    `required=False` accepts an UNSTAMPED file - for artifacts written before
    stamping existed - but still refuses one whose stamp disagrees. Silent
    acceptance of a mismatch is the failure this module exists to prevent.
    """
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or STAMP_KEY not in data:
        if required:
            raise StaleArtifact(
                f"{path} carries no provenance stamp, so there is no way to tell "
                f"which corpus it was built from. Rebuild it, or pass required=False.")
        return data
    was = data[STAMP_KEY].get("corpus_version")
    now = current_corpus_version()
    if was != now:
        raise StaleArtifact(
            f"{path} was built from corpus {str(was)[:12]} but the corpus is now "
            f"{str(now)[:12]}. Rebuild it - using it would score against documents "
            f"that may have been superseded.")
    return data.get("payload")


def write_stamped(path: pathlib.Path, payload: Any, *, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp(payload, note=note), indent=1, ensure_ascii=False))


def describe(path: pathlib.Path) -> dict:
    """What a file claims about its provenance, without raising. For audits."""
    out = {"path": str(path), "exists": path.is_file(), "stamped": False,
           "corpus_version": None, "matches": None}
    if not out["exists"]:
        return out
    try:
        data = json.loads(path.read_text())
    except Exception:
        return out
    # TWO stamp shapes, because two kinds of file need stamping:
    #   dict wrapper  - for artifacts we own end to end (write_stamped)
    #   in-list row   - for files that MUST stay a JSON list, like the eval set,
    #                   whose consumers iterate and read .get("question")
    # A checker that knows only the first reported the eval set as unstamped
    # minutes after it was stamped, which is how one checker and two formats
    # quietly disagree.
    if isinstance(data, dict) and STAMP_KEY in data:
        out["stamped"] = True
        out["corpus_version"] = data[STAMP_KEY].get("corpus_version")
    elif isinstance(data, list):
        row = next((r for r in data
                    if isinstance(r, dict) and (r.get("_stamp") or STAMP_KEY in r)), None)
        if row:
            out["stamped"] = True
            out["corpus_version"] = row.get("corpus_version") or (
                row.get(STAMP_KEY) or {}).get("corpus_version")
    if out["stamped"]:
        out["matches"] = out["corpus_version"] == current_corpus_version()
    return out
