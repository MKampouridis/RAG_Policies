#!/usr/bin/env python3
"""Live monitor: gather, evaluate, notify. Run from launchd every 10 minutes.

    python run_monitor.py              # full run, including the live probe
    python run_monitor.py --no-probe   # data-only, no request to the server
    python run_monitor.py --quiet      # evaluate and write, never notify

Detectors and their thresholds live in src/monitor.py, each with the number it
was derived from. This file only collects inputs and delivers output.

Writes data/alerts.json on every run - current alerts plus when it last ran -
so /health can show live status and, more importantly, can show that the
monitor itself is still running. A monitoring system that dies silently is
indistinguishable from one reporting all-clear, which is the failure mode most
worth designing against here.
"""

import argparse
import datetime as _dt
import json
import hashlib
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src import memory, monitor, telemetry  # noqa: E402

ALERTS_PATH = pathlib.Path("data/alerts.json")
REPORT_PATH = pathlib.Path("data/monitor_report.md")
BASE = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8000")
WINDOW_H = 6          # answers/turns considered "recent"
PROBE_QUESTION = "What is the pass mark for taught masters modules?"


def _access_cookie() -> dict | None:
    """The monitor must work whether or not the shell sourced the password -
    launchd does not read a profile, and a monitor that silently 401s is a
    monitor that reports all-clear forever."""
    pw = os.environ.get("RAG_ACCESS_PASSWORD", "")
    if not pw:
        env = pathlib.Path.home() / ".config" / "ragpolicies" / "env"
        if env.exists():
            for line in env.read_text().splitlines():
                if "RAG_ACCESS_PASSWORD=" in line:
                    pw = line.split("RAG_ACCESS_PASSWORD=", 1)[1].strip().strip("'\"")
    if not pw:
        return None
    return {"rag_access": hashlib.sha256(("rag-access:" + pw).encode()).hexdigest()}


def probe() -> dict:
    """Ask a real question. The only check here that costs anything, and the
    one that would have caught the 503s - both offline safety nets passed while
    no question could be answered.

    One generation per run on the free tier. At a 10-minute cadence that is
    ~144 questions a day, which WOULD exhaust the free tier on its own - so the
    launchd job runs hourly and the probe is skippable with --no-probe.
    """
    try:
        import requests
    except ImportError:
        return {"ok": False, "error": "requests not installed"}
    cookies = _access_cookie()
    try:
        cid = requests.post(f"{BASE}/api/conversations", json={"title": "__monitor__"},
                            headers={"X-User": "monitor"}, cookies=cookies, timeout=20).json()["id"]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"could not create a conversation: {exc}"}
    try:
        t0 = time.time()
        r = requests.post(f"{BASE}/api/conversations/{cid}/messages",
                          json={"content": PROBE_QUESTION},
                          headers={"X-User": "monitor"}, cookies=cookies, timeout=180)
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        d = r.json()
        return {"ok": True, "seconds": round(time.time() - t0, 1),
                "sources": len(d.get("sources") or []),
                "chars": len(d.get("answer") or "")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        # The probe must not pollute the history it is monitoring: 24 synthetic
        # conversations a day would bend every count on /insights.
        try:
            requests.delete(f"{BASE}/api/conversations/{cid}",
                            headers={"X-User": "monitor"}, cookies=cookies, timeout=15)
        except Exception:  # noqa: BLE001
            pass


def recent_turn_seconds(hours: float) -> list:
    """answer_total timings from the last `hours`, read off the timing log."""
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=hours)
    out = []
    path = os.environ.get("RAG_TIMING_PATH", "data/latency.jsonl")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if '"answer_total"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("stage") != "answer_total" or r.get("seconds") is None:
                    continue
                try:
                    ts = _dt.datetime.fromisoformat(str(r.get("ts")))
                except ValueError:
                    continue
                if ts >= cutoff:
                    out.append(r["seconds"])
    except FileNotFoundError:
        return []
    return out


def gather(with_probe: bool) -> dict:
    now = time.time()
    answers = memory.answer_telemetry(limit=5000)
    recent = [a for a in answers if (a.get("created_at") or 0) >= now - WINDOW_H * 3600]
    today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    todays = [a for a in answers
              if _dt.datetime.fromtimestamp(a.get("created_at") or 0,
                                            _dt.timezone.utc).date().isoformat() == today]
    tokens_today = sum((a.get("input_tokens") or 0) + (a.get("output_tokens") or 0)
                       for a in todays)
    spend_today = sum(a.get("cost_usd") or 0 for a in todays)

    failures = memory.failure_breakdown()
    recent_failures = memory.recent_failure_count(hours=monitor.FAILURE_BURST_WINDOW_H)

    return {
        "answers_recent": recent,
        "answers_today": todays,
        "tokens_today": tokens_today,
        "spend_today": spend_today,
        "failures": failures,
        "recent_failures": recent_failures,
        "turns_recent": recent_turn_seconds(WINDOW_H),
        "probe": probe() if with_probe else None,
    }


def evaluate(d: dict) -> list:
    cap = telemetry.FREE_TIER_DAILY_TOKENS.get("groq")
    checks = [
        monitor.check_liveness(d["probe"]),
        monitor.check_failures(d["failures"], d["recent_failures"]),
        monitor.check_ungrounded(d["answers_recent"]),
        monitor.check_headroom(d["tokens_today"], cap),
        monitor.check_spend(d["spend_today"]),
        monitor.check_truncation(d["answers_recent"]),
        monitor.check_fallback(d["answers_recent"]),
        monitor.check_latency(d["turns_recent"]),
        monitor.check_stuck(d["turns_recent"]),
    ]
    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted([c for c in checks if c], key=lambda a: rank.get(a["severity"], 9))


def notify(alerts: list) -> None:
    """The channel the weekly document watch already uses, because it is proven
    to reach this user - a new one would need its own proof."""
    if not alerts:
        return
    worst = alerts[0]
    extra = f" (+{len(alerts) - 1} more)" if len(alerts) > 1 else ""
    title = "Essex assistant: " + ("DOWN" if worst["severity"] == "critical" else "check")
    body = worst["title"].replace('"', "'") + extra
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{body}" with title "{title}" sound name "Glass"'],
            capture_output=True, timeout=15)
    except Exception:  # noqa: BLE001 - failing to notify must not fail the run
        pass


def write_outputs(alerts: list, d: dict, sent: list) -> None:
    payload = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "alerts": alerts,
        "notified": [a["id"] for a in sent],
        "window_hours": WINDOW_H,
        "context": {
            "answers_recent": len(d["answers_recent"]),
            "answers_today": len(d["answers_today"]),
            "tokens_today": d["tokens_today"],
            "spend_today": round(d["spend_today"], 4),
            "recent_failures": d["recent_failures"],
            "probe": d["probe"],
        },
    }
    try:
        ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ALERTS_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:  # noqa: BLE001
        pass
    lines = ["# Monitor report\n", f"_{payload['generated_at']}_\n"]
    if alerts:
        lines.append(f"## {len(alerts)} alert(s)\n")
        for a in alerts:
            lines.append(f"- **[{a['severity']}] {a['title']}**  \n  {a['detail']}\n")
    else:
        lines.append("## All clear\n\nNo detector fired.\n")
    c = payload["context"]
    lines.append(f"\n## Context (last {WINDOW_H}h / today)\n")
    lines.append(f"- answers in window: {c['answers_recent']} · today: {c['answers_today']}\n")
    lines.append(f"- tokens today: {c['tokens_today']:,} · spend today: ${c['spend_today']:.4f}\n")
    lines.append(f"- failed turns in the last hour: {c['recent_failures']}\n")
    if c["probe"]:
        p = c["probe"]
        lines.append(f"- live probe: {'ok' if p.get('ok') else 'FAILED'} "
                     f"({p.get('seconds', '?')}s, {p.get('sources', 0)} sources)\n")
    try:
        REPORT_PATH.write_text("".join(lines))
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-probe", action="store_true", help="skip the live question")
    ap.add_argument("--quiet", action="store_true", help="evaluate but never notify")
    args = ap.parse_args()

    d = gather(with_probe=not args.no_probe)
    alerts = evaluate(d)
    state = monitor.load_state()
    sent, state = monitor.suppress(alerts, state)
    monitor.save_state(state)
    if not args.quiet:
        notify(sent)
    write_outputs(alerts, d, sent)

    if alerts:
        for a in alerts:
            mark = "NOTIFIED" if a in sent else "suppressed"
            print(f"  [{a['severity']:8}] {a['title']}  ({mark})")
    else:
        print("  all clear")
    # Non-zero only for a system that cannot answer, so launchd's log shows the
    # difference between "monitor ran, found problems" and "monitor could not run".
    return 2 if any(a["severity"] == "critical" for a in alerts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
