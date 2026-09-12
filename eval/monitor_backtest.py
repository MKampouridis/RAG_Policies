#!/usr/bin/env python3
"""Replay history against the monitor's rules: what WOULD each have fired on?

A threshold nobody backtested is a guess. The two failure modes are opposite
and both fatal: a rule that never fires is decoration, and a rule that fires
constantly trains its reader to ignore the channel - which leaves you worse off
than no alert at all, because now you believe you are covered.

This walks the recorded history one day at a time, evaluates every detector on
what was visible at the time, and prints the days each rule fires on. Run it
after changing ANY threshold in src/monitor.py.

    python eval/monitor_backtest.py
"""
import datetime as _dt
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import memory, monitor, telemetry  # noqa: E402

def day_of(ts) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).date().isoformat()

def main() -> int:
    answers = memory.answer_telemetry(limit=100000)
    by_day = defaultdict(list)
    for a in answers:
        by_day[day_of(a.get("created_at") or 0)].append(a)

    turns = defaultdict(list)
    path = pathlib.Path("data/latency.jsonl")
    if path.exists():
        for line in path.read_text().splitlines():
            if '"answer_total"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("stage") == "answer_total" and r.get("seconds") is not None:
                turns[str(r.get("ts"))[:10]].append(r["seconds"])

    fails = defaultdict(int)
    for f in memory.failure_breakdown():
        fails[f["day"]] += f["n"]

    cap = telemetry.FREE_TIER_DAILY_TOKENS.get("groq")
    days = sorted(set(by_day) | set(turns) | set(fails))
    fired = defaultdict(list)
    print(f"Replaying {len(days)} day(s) of history\n")
    print(f"{'day':12} {'ans':>4} {'turns':>6} {'fails':>6}  alerts")
    for d in days:
        ans, tn, fl = by_day.get(d, []), turns.get(d, []), fails.get(d, 0)
        tok = sum((a.get("input_tokens") or 0) + (a.get("output_tokens") or 0) for a in ans)
        spend = sum(a.get("cost_usd") or 0 for a in ans)
        # A whole day stands in for the rolling hour: it can only OVER-report a
        # burst rule, so a rule quiet here is quiet for real.
        got = [c for c in (
            monitor.check_failures([], fl),
            monitor.check_ungrounded(ans),
            monitor.check_headroom(tok, cap),
            monitor.check_spend(spend),
            monitor.check_truncation(ans),
            monitor.check_fallback(ans),
            monitor.check_latency(tn),
            monitor.check_stuck(tn),
        ) if c]
        for c in got:
            fired[c["id"]].append(d)
        print(f"{d:12} {len(ans):>4} {len(tn):>6} {fl:>6}  "
              + (", ".join(f"{c['id']}({c['value']})" for c in got) or "-"))

    print("\nPer-rule firing rate:")
    for rid in ("down", "no_sources", "failure_burst", "ungrounded", "free_tier",
                "spend", "truncated", "fallback", "slow", "stuck_turn"):
        ds = fired.get(rid, [])
        rate = len(ds) / len(days) * 100 if days else 0
        note = ""
        if rid in ("down", "no_sources"):
            note = "  (live probe only - not replayable from history)"
        elif not ds:
            note = "  (never fired: check it CAN fire before trusting it)"
        elif rate > 30:
            note = "  (NOISY - fires on most days; raise the threshold)"
        print(f"  {rid:15} {len(ds):>3} day(s)  {rate:5.1f}%  {', '.join(ds[-4:])}{note}")
    print()
    return 0 if selftest() else 1


def selftest() -> bool:
    """Every rule must fire on an input that SHOULD trip it.

    A rule that never fired in history is either well-tuned or broken, and the
    backtest alone cannot tell those apart - "0 days, 0.0%" looks identical
    either way. This project has been bitten by exactly that: a dead-zone check
    written once passed the actual broken file it was written for. So each rule
    is handed a known failure and must report it.
    """
    cap = telemetry.FREE_TIER_DAILY_TOKENS.get("groq", 200_000)
    ungrounded = [{"n_sources": 0} for _ in range(10)]
    cases = [
        ("failure_burst", monitor.check_failures([], monitor.FAILURE_BURST)),
        ("ungrounded",    monitor.check_ungrounded(ungrounded)),
        ("free_tier",     monitor.check_headroom(int(cap * 0.9), cap)),
        ("spend",         monitor.check_spend(monitor.DAILY_SPEND_USD + 1)),
        ("truncated",     monitor.check_truncation([{"truncated": True}])),
        ("fallback",      monitor.check_fallback([{"fell_back_from": "groq"}])),
        ("slow",          monitor.check_latency([90.0] * monitor.LATENCY_MIN_SAMPLE)),
        ("stuck_turn",    monitor.check_stuck([monitor.STUCK_TURN_SECONDS + 1])),
        ("down",          monitor.check_liveness({"ok": False, "error": "boom"})),
        ("no_sources",    monitor.check_liveness({"ok": True, "sources": 0})),
    ]
    # And the mirror: healthy input must stay quiet, or a rule that always
    # fires would pass the test above while being useless.
    quiet = [
        ("failure_burst", monitor.check_failures([], 0)),
        ("ungrounded",    monitor.check_ungrounded([{"n_sources": 3}] * 10)),
        ("free_tier",     monitor.check_headroom(int(cap * 0.1), cap)),
        ("spend",         monitor.check_spend(0.01)),
        ("truncated",     monitor.check_truncation([{"truncated": False}])),
        ("fallback",      monitor.check_fallback([{}])),
        ("slow",          monitor.check_latency([5.0] * monitor.LATENCY_MIN_SAMPLE)),
        ("stuck_turn",    monitor.check_stuck([9.0])),
        ("down",          monitor.check_liveness({"ok": True, "sources": 4})),
    ]
    ok = True
    print("Self-test — each rule against a known failure, and against a healthy case:")
    for name, got in cases:
        if not got:
            print(f"  {name:15} BROKEN: did not fire on input that should trip it"); ok = False
        elif got["id"] != name:
            print(f"  {name:15} BROKEN: fired as '{got['id']}'"); ok = False
    for name, got in quiet:
        if got:
            print(f"  {name:15} BROKEN: fired on a healthy case ({got['title']})"); ok = False
    print("  all rules fire when they should and stay quiet when they should" if ok
          else "  SELF-TEST FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
