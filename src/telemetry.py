"""Per-answer facts the pipeline already computes and used to throw away.

WHY THIS EXISTS
The feedback dashboard drew six panels from 38 self-selected ratings while the
system produced 340 answers, and nothing recorded what any of them COST, how
long they took to start, whether they cited anything, or whether the right
document was retrieved and then ignored. Every one of those was already in
memory at answer time - llm.LAST_USAGE, the sources list, ranked_top_urls, the
history length - and was discarded when the request returned.

Two stores, because they answer different questions and have different shapes:

  * message meta (chat.db)   one record per answer, joinable to the question,
                             the sources and the rating. Read by /insights.
  * data/events.jsonl        client-side signals (a copy click, a source
                             opened). Append-only, one line per event, no join
                             key beyond the message it refers to.

Latency stays in data/latency.jsonl, which already holds 2,673 stage timings -
this module does not duplicate it.

CONTRACT: like src/instrumentation.py, nothing here may break a request.
Every public function swallows its own errors and returns something usable.
"""

import datetime as _dt
import json
import os
import re

from src import llm

# Cost per million tokens, in USD. Used to turn token counts into a number the
# dashboard can add up. Deliberately a TABLE rather than a per-provider API
# call: these are published list prices that change rarely, and a wrong-by-20%
# cost line is still the difference between "no idea" and "about four pounds a
# month". Update when a provider's pricing page changes.
PRICING = {
    "openai/gpt-oss-120b": (0.15, 0.60),      # Groq, paid tier
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
# Free-tier daily token ceiling, so /health can show headroom rather than only
# reporting the outage after it happens.
FREE_TIER_DAILY_TOKENS = {"groq": 200_000}

EVENTS_PATH = os.environ.get("RAG_EVENTS_PATH", "data/events.jsonl")
# Events are a UI signal, not a metric to defend: an unknown name is dropped
# rather than stored, so a typo in a fetch() call cannot quietly become a
# category on a chart.
EVENT_KINDS = {"copy_answer", "copy_question", "open_source", "rephrase"}


def cost_usd(model: str, input_tokens, output_tokens) -> float | None:
    """None when the model is not priced, NOT zero - a missing price and a free
    call are different facts, and averaging zeros into a cost chart would
    understate the bill silently."""
    price = PRICING.get(model or "")
    if not price or input_tokens is None or output_tokens is None:
        return None
    pin, pout = price
    return round((input_tokens * pin + output_tokens * pout) / 1_000_000, 6)


# "The policies I can see don't cover X" and its older phrasing. Kept here
# rather than in the page's JavaScript because the dashboard was GUESSING this
# from the answer text and labelling it "indicative" - recording it at answer
# time, next to whether any source was cited, turns a heuristic into a fact
# the charts can rely on.
_ABSTAIN_RE = re.compile("|".join([
    r"(don'?t|do not|does not|doesn'?t) (have|contain|include|cover|specify|mention|address)",
    r"policies i can see don'?t cover",
    r"(can'?t|cannot|could not|couldn'?t|unable to) (find|locate|answer|determine)",
    r"no (information|details|document|mention) (on|about|regarding|of)",
]), re.I)


def _doc(url: str) -> str:
    return str(url).rstrip("/").rsplit("/", 1)[-1]


def answer_record(*, answer_text: str, sources: list, ranked_top_urls: list,
                  history: list, retrieval_query: str = "", question: str = "",
                  seconds: float | None = None) -> dict:
    """The per-answer telemetry block, for message meta.

    Everything here is derived from values the caller already has. It makes no
    model calls and costs nothing, which is the whole reason it can run on
    every answer rather than on a sample.
    """
    try:
        usage = dict(llm.LAST_USAGE or {})
        gen = llm.LAST_GENERATOR or {}
        model = usage.get("model") or gen.get("model") or ""
        src_docs = {_doc(u) for u in (sources or [])}
        top_docs = {_doc(u) for u in (ranked_top_urls or [])}
        rec = {
            "model": model,
            "provider": usage.get("provider") or gen.get("provider"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            # reasoning models bill invisible thinking inside output_tokens;
            # broken out where the provider reports it, because "what did I pay
            # for text the user never saw" is its own question.
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "cost_usd": cost_usd(model, usage.get("input_tokens"), usage.get("output_tokens")),
            # 'length'/'max_tokens' means the answer was CUT OFF mid-sentence.
            # Nothing noticed this before, and a truncated policy answer is
            # worse than an error because the reader cannot tell.
            "stop_reason": usage.get("stop_reason"),
            "truncated": usage.get("stop_reason") in ("length", "max_tokens"),
            "fell_back_from": gen.get("fell_back_from"),
            # The ledger splits on this constantly - the contextualizer only
            # runs on follow-ups - but production traffic was never labelled,
            # so "are follow-ups worse in real use?" was unanswerable.
            "follow_up": bool(history),
            "turn_index": len(history) // 2 if history else 0,
            "rewritten": bool(retrieval_query and question
                              and retrieval_query.strip() != question.strip()),
            "cited_any": bool(sources),
            "n_sources": len(sources or []),
            "abstained": bool(_ABSTAIN_RE.search((answer_text or "")[:300])),
            # Retrieved and then NOT used. This is the "right document, wrong
            # chunk / had the facts, didn't use them" gap that hit@6 cannot see
            # by construction, measured at 8.7 points on the main set - and it
            # is free to record here.
            "retrieved_not_cited": sorted(top_docs - src_docs),
            "answer_chars": len(answer_text or ""),
            "seconds": round(seconds, 2) if seconds is not None else None,
        }
        return {k: v for k, v in rec.items() if v is not None}
    except Exception:  # noqa: BLE001 - telemetry must never break a request
        return {}


def log_event(kind: str, detail: dict | None = None) -> bool:
    """Append one client-side signal. Returns whether it was stored.

    These are HIGH-VOLUME and LOW-PRECISION, the mirror image of the 38
    explicit ratings: a copy click means "I am using this", not "this is
    correct", and someone can copy a confidently wrong answer. Worth having
    alongside ratings, never worth merging onto the same axis as one.
    """
    if kind not in EVENT_KINDS:
        return False
    try:
        os.makedirs(os.path.dirname(EVENTS_PATH) or ".", exist_ok=True)
        with open(EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "kind": kind,
                "detail": detail or {},
            }) + "\n")
        return True
    except Exception:  # noqa: BLE001
        return False


def load_events(limit: int = 100_000) -> list[dict]:
    try:
        with open(EVENTS_PATH, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        return rows[-limit:]
    except FileNotFoundError:
        return []
    except Exception:  # noqa: BLE001
        return []
