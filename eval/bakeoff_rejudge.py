#!/usr/bin/env python3
"""Cross-family re-judge: re-score the groundedness of already-generated bake-off
answers with a DIFFERENT judge model, to check the ranking isn't an artifact of
the qwen-14b judge (self-judging bias for the qwen generators). Reuses the stored
answers (no regeneration).

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/bakeoff_rejudge.py <judge_model> [gen_model ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.generator_bakeoff import GROUND_PROMPT, _clean
from src.llm import chat

JUDGE = sys.argv[1] if len(sys.argv) > 1 else "gemma3:12b"
GENS = sys.argv[2:] or ["gemma3:12b", "qwen2.5:14b-instruct"]


def judge_grounded(context, answer):
    raw = chat(messages=[{"role": "system", "content": GROUND_PROMPT},
                         {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{_clean(answer)}"}],
               format="json", model=JUDGE)
    try:
        return bool(json.loads(raw).get("grounded", True))
    except Exception:
        return None


print(f"re-judge with judge={JUDGE}\n", flush=True)
for g in GENS:
    p = Path(f"eval/bakeoff_{g.replace(':', '_').replace('/', '_')}.json")
    rows = json.loads(p.read_text())
    for r in rows:
        r["grounded_rejudge"] = judge_grounded(r["context"], r["answer"])
    scored = [r for r in rows if r["grounded_rejudge"] is not None]
    rate = lambda sub: f"{sum(1 for r in sub if r['grounded_rejudge']) / len(sub) * 100:.1f}%" if sub else "n/a"
    hit = [r for r in scored if r["hit"]]
    miss = [r for r in scored if not r["hit"]]
    roa = [r for r in scored if r["doc_type"] == "rules_of_assessment"]
    orig = sum(1 for r in rows if r.get("grounded")) / len(rows) * 100
    print(f"RESULT {g:24s} [{JUDGE}-judged] overall {rate(scored)} | hit {rate(hit)} | miss {rate(miss)} | "
          f"RoA {rate(roa)}   (qwen-judged was {orig:.0f}%)", flush=True)
