"""Per-stage latency instrumentation.

Split out of rag.py 2026-08-13. Pure move - no behaviour change - as the first
step of breaking up a 2,237-line module, chosen to go first BECAUSE it is the
most isolated part: it depends on nothing in the retrieval path and nothing in
the retrieval path depends on it beyond calling it.

Writes one JSON line per stage to data/latency.jsonl. Instrumentation must
never break a request, so every function here swallows its own errors.
"""

import datetime as _dt
import json
import os
import time as _perf

RAG_TIMING = os.environ.get("RAG_TIMING", "") == "1"
# Separate paths so production traffic and eval runs never mix in one file -
# they answer different questions (what users experience vs did change X help).
_TIMING_PATH = os.environ.get("RAG_TIMING_PATH", "data/latency.jsonl")


# Resolved once, not per call: the imports, the mkdir and the Path construction
# below all used to run on every stage of every request - inside the very
# window this function exists to measure.
_TIMING_READY = False


def _ensure_timing_dir() -> None:
    """Both writers must go through this.

    They did not: `_stage_note` opened the file directly while only
    `_stage_timer` did the mkdir, so on a fresh checkout a note written before
    any timer was LOST - and lost silently, because both writers swallow their
    own exceptions by design. Verified before fixing: with the directory absent,
    `_stage_note` alone wrote nothing and reported nothing.

    Instrumentation that fails invisibly is how the latency picture went wrong
    before, so the shared guard matters more than the one missing line.
    """
    global _TIMING_READY
    if not _TIMING_READY:
        os.makedirs(os.path.dirname(_TIMING_PATH) or ".", exist_ok=True)
        _TIMING_READY = True


def _write(rec: dict) -> None:
    """Single append point. Best-effort by contract - instrumentation must never
    break a request - which is exactly why the directory guard cannot live in
    only one of the two callers."""
    try:
        _ensure_timing_dir()
        with open(_TIMING_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _stage_note(kind: str, payload: dict) -> None:
    """Structured diagnostic line beside the timings. Same file, same
    best-effort contract - instrumentation must never break a request."""
    if not RAG_TIMING:
        return
    _write({"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "stage": kind, "seconds": None, "detail": payload})


def _stage_timer(stage: str, started: float) -> None:
    if not RAG_TIMING:
        return
    _write({"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(), "stage": stage,
            "seconds": round(_perf.time() - started, 3)})
