#!/usr/bin/env python3
"""Add answer_score (completeness/helpfulness vs the gold answer, 1-5) to the
bake-off finalists - OFFLINE, reusing the answers already generated in the
bake-off (no regeneration). Groundedness (from the bake-off) measures
faithfulness; this measures whether the answer actually ANSWERS the question, so
we don't crown a model that's faithful-but-terse.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/bakeoff_answerscore.py [model ...]
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.run_eval import judge_answer
from src.llm import JUDGE_MODEL

REF = json.loads(Path("eval/results_qwen14b_full.json").read_text())
EXP = {}  # (source_title, turn) -> expected_answer
for r in REF:
    for turn in ("primary", "follow_up"):
        EXP[(r["source_title"], turn)] = r[turn]["expected_answer"]

FINALISTS = sys.argv[1:] or ["gemma3:12b", "gpt-oss:20b", "phi4", "qwen2.5:14b-instruct"]


def fname(m):
    return f"eval/bakeoff_{m.replace(':', '_').replace('/', '_')}.json"


def score_of(res):
    try:
        return int(res.get("score")) if isinstance(res, dict) else int(res)
    except Exception:
        return None


print(f"answer_score (1-5, judge={JUDGE_MODEL}) on the bake-off answers vs gold\n", flush=True)
for m in FINALISTS:
    p = Path(fname(m))
    if not p.exists():
        print(f"{m}: no bake-off file, skipped", flush=True)
        continue
    rows = json.loads(p.read_text())
    for r in rows:
        title, turn = r["label"].rsplit("[", 1)
        turn = turn.rstrip("]")
        exp = EXP.get((title, turn))
        r["ascore"] = score_of(judge_answer(r["question"], exp, r["answer"])) if exp else None
    scored = [r for r in rows if r.get("ascore") is not None]
    roa = [r for r in scored if r["doc_type"] == "rules_of_assessment"]
    pol = [r for r in scored if r["doc_type"] == "policy"]
    grounded_rate = sum(1 for r in rows if r.get("grounded")) / len(rows) * 100
    print(f"RESULT {m:24s} answer_score overall {statistics.mean(x['ascore'] for x in scored):.2f} | "
          f"RoA {statistics.mean(x['ascore'] for x in roa):.2f} | Policy {statistics.mean(x['ascore'] for x in pol):.2f}"
          f"  (grounded {grounded_rate:.0f}%)", flush=True)
    p.with_name(p.stem + "_ascore.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
