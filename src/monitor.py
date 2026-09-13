"""Live monitoring: detectors for "something is wrong right now".

WHY, AND WHY IT IS SHAPED THIS WAY
The dashboards answer questions somebody thought to ask. Monitoring has to
work when nobody is looking - which is most of the time, and is exactly when
the free tier ran out on 2026-09-05 and every question 503'd for everyone.

Two rules govern everything here.

**Thresholds are derived from this system's own history, not guessed.** Every
number below has its derivation in a comment, computed from 329 recorded turns,
346 answers and 28 failures. A guessed threshold either never fires or fires
constantly, and the second is worse: an alert that cries wolf stops being read,
which leaves you worse off than no alert at all. `eval/monitor_backtest.py`
replays history against these rules and reports what each WOULD have fired on -
run it after changing any threshold.

**Everything is computed from data already on disk.** No model calls, so
monitoring costs nothing and can run every few minutes. The one exception is
the liveness probe, which posts a real question - and it is the check that
would have caught the 503s, because the retrieval fingerprint and the canary
both passed while no question could be answered.

Each detector returns None (quiet) or a dict describing what it saw. Nothing
here raises: a monitor that crashes is a monitor that is not monitoring.
"""

import datetime as _dt
import json
import os
import pathlib
import time

STATE_PATH = pathlib.Path(os.environ.get("RAG_MONITOR_STATE", "data/monitor_state.json"))

# ── thresholds, each with its derivation ────────────────────────────────────
# Latency: recorded p50 10.8s, p90 27.0s, p95 32.1s, p99 87.0s, max 308.9s
# (n=329). 45s sits above the historical p95, so normal slow days stay quiet
# and a genuine regression does not.
SLOW_P90_SECONDS = 45.0
# A single turn over two minutes is not a slow day, it is a stuck one. The
# recorded max was 308.9s, so this does exist and is worth naming individually.
STUCK_TURN_SECONDS = 120.0

# Failures: 28 in total across 7 days, worst day 8 - and both 8-failure days
# (2026-08-11, 2026-09-05) were real incidents. Three inside one hour is well
# clear of the ordinary one-offs while still catching both.
FAILURE_BURST = 3
FAILURE_BURST_WINDOW_H = 1

# Grounding: 14.5% of all answers cite nothing, but the DAILY rate is 0% on
# almost every day with one day at 100% - so the aggregate is one bad day, not
# a background rate. 40% over a window of at least MIN_SAMPLE answers catches
# that day and nothing else in recorded history.
UNGROUNDED_RATE = 0.40
MIN_SAMPLE = 8

# Percentile rules need a bigger sample than rate rules. Backtesting the first
# version showed `slow` firing on 21% of days - but those days had ~9 turns, and
# the p90 of 9 samples is essentially the maximum, so the rule was reporting
# "one slow answer happened" while claiming to report a trend. 20 is the point
# where the 90th percentile means something; below it the check declines to
# judge rather than guessing, which is the honest behaviour for a quiet hour.
LATENCY_MIN_SAMPLE = 20

# Free-tier headroom. Groq's daily ceiling is 200k tokens; at 80% there is
# still time to act, which is the entire point of a leading indicator. The
# 2026-09-05 outage announced itself only by failing.
FREE_TIER_WARN = 0.80

# Spend. Not an anomaly measure - a budget. At ~$0.0005/answer this needs a
# runaway loop or a fallback to a paid model to trip, both of which are exactly
# what should wake somebody.
DAILY_SPEND_USD = 2.00

# Cooldown: a condition that is still true in ten minutes is not new news.
# Re-alerting on it is the fastest way to train someone to ignore the channel.
COOLDOWN_H = 6


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _since(hours: float) -> float:
    return time.time() - hours * 3600


def _pctl(values: list, p: float):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


# ── detectors ───────────────────────────────────────────────────────────────

def check_failures(failures_by_day: list, recent_failures: int) -> dict | None:
    """Turns that produced no answer at all.

    Structurally invisible to feedback - nobody can rate a question that
    returned nothing - so this is the only place they surface.
    """
    if recent_failures >= FAILURE_BURST:
        return {"id": "failure_burst", "severity": "high",
                "title": f"{recent_failures} failed turns in the last hour",
                "detail": "Questions are being asked and getting nothing back. "
                          "Check the generator: quota, rate limit or network.",
                "value": recent_failures}
    return None


def check_ungrounded(answers: list) -> dict | None:
    """Answers citing no source at all.

    Not a quality judgement - a structural one. An answer with no source is
    ungrounded by definition, and a spike means retrieval stopped returning
    anything usable rather than that the questions got harder.
    """
    if len(answers) < MIN_SAMPLE:
        return None
    n = sum(1 for a in answers if not a.get("n_sources"))
    rate = n / len(answers)
    if rate >= UNGROUNDED_RATE:
        return {"id": "ungrounded", "severity": "high",
                "title": f"{round(rate * 100)}% of recent answers cited no source",
                "detail": f"{n} of {len(answers)} answers came back with nothing cited. "
                          "Retrieval or the index, not the generator.",
                "value": round(rate, 3)}
    return None


def check_latency(turn_seconds: list) -> dict | None:
    if len(turn_seconds) < LATENCY_MIN_SAMPLE:
        return None
    p90 = _pctl(turn_seconds, 90)
    if p90 and p90 > SLOW_P90_SECONDS:
        return {"id": "slow", "severity": "medium",
                "title": f"Recent p90 answer time is {p90:.0f}s",
                "detail": f"Baseline p90 is 27s, p95 32s (n=329). Over {len(turn_seconds)} "
                          "recent turns. Check whether Ollama or the reranker is swapping.",
                "value": round(p90, 1)}
    return None


def check_stuck(turn_seconds: list) -> dict | None:
    worst = max(turn_seconds) if turn_seconds else 0
    if worst > STUCK_TURN_SECONDS:
        return {"id": "stuck_turn", "severity": "medium",
                "title": f"A turn took {worst:.0f}s",
                "detail": "Over two minutes for one answer. Usually a provider stalling "
                          "or a retry ladder running to the end.",
                "value": round(worst, 1)}
    return None


def check_headroom(tokens_today: int, cap: int) -> dict | None:
    """Leading indicator, deliberately. The 2026-09-05 outage announced itself
    by failing; at 80% there is still time to do something about it."""
    if not cap:
        return None
    used = tokens_today / cap
    if used >= FREE_TIER_WARN:
        return {"id": "free_tier", "severity": "high" if used >= 0.95 else "medium",
                "title": f"Free tier {round(used * 100)}% used today",
                "detail": f"{tokens_today:,} of {cap:,} tokens. When this runs out every "
                          "question fails - the paid fallback is currently off.",
                "value": round(used, 3)}
    return None


def _usd(v: float) -> str:
    """Two decimals hides everything this system actually spends - a real day
    is ~$0.003, which formats as "$0.00" and reads as a bug in the alert."""
    return f"${v:.2f}" if v >= 0.01 else f"${v:.4f}"


def check_spend(spend_today: float) -> dict | None:
    if spend_today > DAILY_SPEND_USD:
        return {"id": "spend", "severity": "high",
                "title": f"Spend today is {_usd(spend_today)}",
                "detail": f"Over the {_usd(DAILY_SPEND_USD)} daily budget. At ~$0.0005 an "
                          "answer this needs a runaway loop or a paid fallback.",
                "value": round(spend_today, 4)}
    return None


def check_truncation(answers: list) -> dict | None:
    """An answer that hit max_tokens was CUT OFF mid-sentence. On a policy
    assistant that is worse than an error, because the reader cannot tell."""
    n = sum(1 for a in answers if a.get("truncated"))
    if n:
        return {"id": "truncated", "severity": "medium",
                "title": f"{n} answer(s) were cut off at max_tokens",
                "detail": "Truncated mid-sentence and indistinguishable from a complete "
                          "answer to the reader. Raise the output cap.",
                "value": n}
    return None


def check_fallback(answers: list) -> dict | None:
    n = sum(1 for a in answers if a.get("fell_back_from"))
    if n:
        return {"id": "fallback", "severity": "medium",
                "title": f"{n} answer(s) came from the paid fallback",
                "detail": "The primary generator failed and a paid model answered instead. "
                          "Expected only if the fallback was deliberately re-enabled.",
                "value": n}
    return None


def check_drift(running_rev: str | None, head_rev: str | None) -> dict | None:
    """Is the server running the code that is saved?

    A running process holds the code it loaded at startup. Edit a file
    afterwards and the server keeps serving the old version - the fix is on
    disk, visible in the editor, and not in production, with nothing anywhere
    saying so. Found on 2026-09-13: production was stamping answers with a
    revision it had never run.

    Costs nothing: one HTTP GET to /api/config on this machine (which reads a
    few environment variables and returns) and one `git rev-parse`. No model is
    involved, so this runs in the FREE hourly job and not the twice-daily probe.

    Silent when the revision is unknown or the server is unreachable - the
    first is not a fact, and the second is check_liveness's job to report.
    """
    if not running_rev or not head_rev or running_rev == "unknown":
        return None
    if running_rev == head_rev:
        return None
    return {"id": "code_drift", "severity": "medium",
            "title": f"Server is running {running_rev}, saved code is {head_rev}",
            "detail": "The running server loaded its code at startup and has not picked up "
                      "later changes. Restart it (launchctl unload/load "
                      "com.mkampo.ragpolicies) - until then, fixes on disk are not live and "
                      "every answer's recorded revision is wrong.",
            "value": f"{running_rev}->{head_rev}"}


def check_liveness(probe: dict | None) -> dict | None:
    """Can the system answer a question AT ALL.

    The check that matters most and the one every other safety net missed: a
    refactor once left a constant undefined and every answer returned 503 while
    a 161-query fingerprint and a 118-turn canary both passed, because both
    exercise retrieval and the constant is used during answer assembly.
    """
    if probe is None:
        return None
    if not probe.get("ok"):
        return {"id": "down", "severity": "critical",
                "title": "The assistant could not answer a test question",
                "detail": str(probe.get("error") or "")[:300],
                "value": 0}
    if not probe.get("sources"):
        return {"id": "no_sources", "severity": "high",
                "title": "The test question was answered with no sources",
                "detail": "The server is up and generating, but retrieval returned nothing - "
                          "the index may be missing or unreadable.",
                "value": 0}
    return None


# ── state, so a standing condition is not re-announced forever ──────────────

def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:  # noqa: BLE001 - a corrupt state file must not stop monitoring
        return {}


def save_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception:  # noqa: BLE001
        pass


def suppress(alerts: list, state: dict) -> tuple[list, dict]:
    """Drop alerts already announced inside the cooldown, and CLEAR the record
    for conditions that have gone away - so the same problem recurring next
    week alerts again, while one that never resolved does not renotify hourly.

    Returns (alerts to send, new state). Alerts are still returned in the
    report and via /api/alerts either way; suppression governs notification
    only, because a dashboard showing a standing problem is correct and a
    notification repeating it every ten minutes is noise.
    """
    now = time.time()
    fired = dict(state.get("fired") or {})
    active = {a["id"] for a in alerts}
    for key in list(fired):
        if key not in active:
            del fired[key]          # condition cleared; next occurrence is news again
    send = []
    for a in alerts:
        last = fired.get(a["id"])
        if last is None or (now - last) > COOLDOWN_H * 3600:
            send.append(a)
            fired[a["id"]] = now
    state["fired"] = fired
    state["last_run"] = _now().isoformat()
    return send, state
