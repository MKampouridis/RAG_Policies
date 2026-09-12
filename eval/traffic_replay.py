#!/usr/bin/env python3
"""Replay the traffic regression set through retrieval. No generation, no cost.

WHAT IT ANSWERS
"I changed something - did retrieval move on questions people actually asked?"

It reports TURNS CHANGED, never hit@6, and that wording is load-bearing. The
baselines are what retrieval returned at the time, not verified gold (see
eval/build_traffic_regression.py). A change is a prompt to LOOK. Some changes
will be improvements, and this tool cannot tell you which - a report that said
"hit@6 fell 4 points" would be inventing a verdict it has no basis for.

WHY IT IS FREE AND DETERMINISTIC
Each item carries the retrieval_query that was actually used, so the
contextualizer - the only LLM left in the retrieval path, and a paid cloud call
- never runs. No generation, no judge. Minutes, no API spend, no temperature.

Usage:
    PYTHONPATH=. python eval/traffic_replay.py before
    ...make a change, restart, re-embed, whatever...
    PYTHONPATH=. python eval/traffic_replay.py after
    PYTHONPATH=. python eval/traffic_replay.py --diff before after

A single pass also compares against each item's RECORDED baseline, so one run
on its own already tells you what has drifted since the traffic was captured.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

SET_PATH = "eval/questions_traffic.json"
TOP_N = 6          # same depth the old hit@6 used, so "top documents" means the same thing


def _doc(url: str) -> str:
    return str(url).rstrip("/").rsplit("/", 1)[-1]


def out_path(label: str) -> pathlib.Path:
    return pathlib.Path(f"eval/traffic_replay_{label}.json")


def run(label: str, set_path: str) -> int:
    from src.rag import retrieve

    data = json.loads(pathlib.Path(set_path).read_text())
    items = data["items"]
    rows = []
    for i, it in enumerate(items, 1):
        try:
            res, _ = retrieve(it["retrieval_query"], [])
            metas = (res.get("metadatas") or [[]])[0]
            seen, top = set(), []
            for m in metas:
                d = _doc((m or {}).get("source_url") or "")
                if d and d not in seen:
                    seen.add(d)
                    top.append(d)
                if len(top) >= TOP_N:
                    break
        except Exception as exc:  # noqa: BLE001 - one bad item must not void the pass
            print(f"  [{i}/{len(items)}] ERROR {type(exc).__name__}: {exc}")
            top = None
        rows.append({**{k: it[k] for k in
                        ("tier", "question", "retrieval_query", "primary", "baseline_docs", "rating")},
                     "now_docs": top})
        if i % 25 == 0:
            print(f"  {i}/{len(items)}")
    out_path(label).write_text(json.dumps(rows, indent=1))
    print(f"wrote {out_path(label)}  ({len(rows)} items)")
    report(rows, f"{label} vs recorded baseline", baseline_key="baseline_docs")
    return 0


def report(rows: list, title: str, baseline_key: str = "baseline_docs",
           other: dict | None = None) -> None:
    print(f"\n=== {title} ===")
    tiers = ("anchor_positive", "anchor_neutral", "watch_negative")
    LABEL = {"anchor_positive": "thumbed UP (strongest signal)",
             "anchor_neutral": "unrated (weak baseline)",
             "watch_negative": "thumbed DOWN (change may be GOOD)"}
    changed_all = []
    for t in tiers:
        sub = [r for r in rows if r["tier"] == t and r.get("now_docs") is not None]
        if not sub:
            continue
        changed = []
        for r in sub:
            before = set(other[_key(r)]["now_docs"]) if other and _key(r) in other \
                else set(r[baseline_key])
            if other and _key(r) not in other:
                continue
            after = set(r["now_docs"])
            if before != after:
                changed.append((r, sorted(after - before), sorted(before - after)))
        changed_all.extend((t, c) for c in changed)
        pctg = len(changed) / len(sub) * 100
        print(f"  {LABEL[t]:38} {len(changed):>3} of {len(sub):<3} changed ({pctg:.0f}%)")

    if not changed_all:
        print("  nothing moved.")
        return
    print("\n  what moved (look, do not assume it is a regression):")
    for t, (r, gained, lost) in changed_all[:20]:
        flag = "!" if t == "anchor_positive" else (" " if t == "anchor_neutral" else "?")
        print(f"  {flag} [{t.split('_')[1][:3]}] {r['question'][:62]}")
        if lost:
            print(f"      - no longer: {', '.join(x[:44] for x in lost[:3])}")
        if gained:
            print(f"      + now also: {', '.join(x[:44] for x in gained[:3])}")
    if len(changed_all) > 20:
        print(f"  ... {len(changed_all) - 20} more")
    print("\n  ! = a turn a person said was RIGHT lost or gained documents - check these first."
          "\n  ? = a turn a person said was WRONG changed, which may be the fix working.")


def _key(r: dict) -> str:
    return " ".join(r["retrieval_query"].lower().split())


def diff(a: str, b: str) -> int:
    ra = json.loads(out_path(a).read_text())
    rb = json.loads(out_path(b).read_text())
    idx = {_key(r): r for r in ra if r.get("now_docs") is not None}
    rows = [r for r in rb if r.get("now_docs") is not None]
    print(f"comparing '{a}' -> '{b}' on {len(set(idx) & {_key(r) for r in rows})} shared question(s)")
    report(rows, f"{a} -> {b}", other=idx)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", help="name this pass")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    ap.add_argument("--set", default=SET_PATH)
    args = ap.parse_args()
    if args.diff:
        return diff(*args.diff)
    if not args.label:
        ap.error("give a label, or --diff A B")
    return run(args.label, args.set)


if __name__ == "__main__":
    raise SystemExit(main())
