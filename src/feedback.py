"""Append-only user-feedback log (JSONL) for ratings of answers.

Early-stage by design: no DB, no schema migrations - one JSON object per line
at data/feedback.jsonl, so it's trivially greppable and feeds feedback_report.py
straight into an action plan. Each record stores enough retrieval context
(retrieval_query, sources, ranked_top_urls) to REPLAY the retrieval later and
auto-classify a thumbs-down into the failure taxonomy (retrieval miss vs
hallucination) without the user having to. Feedback is user data and lives under
the gitignored data/ dir.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_PATH = Path("data/feedback.jsonl")
_lock = threading.Lock()

# Size cap so an unattended deployment can't let the append-only log fill the
# disk (round-6 review, DeepSeek). At the threshold the current file is rotated
# to a timestamped sibling and a fresh log started; feedback_report.py already
# tolerates multiple files if you glob them, and rotations are rare in practice.
_MAX_BYTES = 50 * 1024 * 1024  # 50 MB (~100k+ records) before rotating

# Failure-tag vocabulary shown on a thumbs-down, aligned with the project's
# failure taxonomy (eval/report.md) so each tag routes to a specific lever.
TAGS = {
    "wrong_programme": "Wrong programme / document",   # retrieval miss / wrong sibling / under-specified -> retrieval, D3
    "wrong_figure": "Wrong or made-up figure",         # hallucination -> generator
    "outdated": "Out of date / wrong year",            # stale edition -> data-hygiene, _EXCLUDED_URLS
    "no_answer": "Didn't answer / too vague",          # abstention / under-specified
    "other": "Other",
}


def record_feedback(record: dict) -> None:
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:  # POST handlers can race; keep one clean line per record
        if FEEDBACK_PATH.exists() and FEEDBACK_PATH.stat().st_size >= _MAX_BYTES:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            FEEDBACK_PATH.rename(FEEDBACK_PATH.with_name(f"feedback.{stamp}.jsonl"))
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_feedback() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    out = []
    for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip a single corrupt line rather than kill the whole report
    return out
