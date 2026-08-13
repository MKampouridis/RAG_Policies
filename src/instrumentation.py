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


def _stage_note(kind: str, payload: dict) -> None:
    """Structured diagnostic line beside the timings. Same file, same
    best-effort contract - instrumentation must never break a request."""
    if not RAG_TIMING:
        return
    try:
        rec = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
               "stage": kind, "seconds": None, "detail": payload}
        with open(_TIMING_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


# Resolved once, not per call: the imports, the mkdir and the Path construction
# below all used to run on every stage of every request - inside the very
# window this function exists to measure.
_TIMING_READY = False


def _stage_timer(stage: str, started: float) -> None:
    if not RAG_TIMING:
        return
    global _TIMING_READY
    try:
        rec = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(), "stage": stage,
               "seconds": round(_perf.time() - started, 3)}
        if not _TIMING_READY:
            os.makedirs(os.path.dirname(_TIMING_PATH) or ".", exist_ok=True)
            _TIMING_READY = True
        with open(_TIMING_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
