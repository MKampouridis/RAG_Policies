#!/usr/bin/env python3
"""Re-judge the synthetic miss-corpus answers with a NEUTRAL judge (phi4) to
remove the self-preference confound: the original run judged with qwen2.5:14b,
which SELF-favours the 14B's own answers and is harsh on gemma3 (the exact bias
the project documented and why the generator bake-off finalists used phi4). The
abstention numbers are regex-based (judge-independent) and unaffected; only the
GROUNDED numbers need the neutral re-judge.

Contexts aren't stored in the answer files, so they're rebuilt deterministically
from build_miss_turns() and matched by (question, sibling_url).

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/synthmiss_rejudge.py [judge_model]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import chat
from eval.synthetic_miss_corpus import GROUND_PROMPT, _clean, build_miss_turns

JUDGE = sys.argv[1] if len(sys.argv) > 1 else "phi4"
MODELS = ["gemma3:12b", "qwen2.5:14b-instruct"]

ctx_by_key = {(t["question"], t["sibling_url"]): t["context"] for t in build_miss_turns()}


def judge(context, answer):
    raw = chat(messages=[{"role": "system", "content": GROUND_PROMPT},
                         {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{_clean(answer)}"}],
               format="json", model=JUDGE)
    try:
        return bool(json.loads(raw).get("grounded", True))
    except Exception:
        return None


print(f"re-judging synthetic miss corpus with NEUTRAL judge={JUDGE}\n", flush=True)
for m in MODELS:
    rows = json.loads(Path(f"eval/synthmiss_{m.replace(':', '_').replace('/', '_')}.json").read_text())
    reg = []
    for i, r in enumerate(rows, 1):
        g = judge(ctx_by_key[(r["question"], r["sibling_url"])], r["answer"])
        reg.append(g)
        if i % 50 == 0:
            print(f"    {m} rejudge {i}/{len(rows)}", flush=True)
    scored = [g for g in reg if g is not None]
    grounded = sum(1 for g in scored if g)
    ab = sum(1 for r in rows if r["abstained"])
    halluc = sum(1 for r, g in zip(rows, reg) if g is False and not r["abstained"])
    Path(f"eval/synthmiss_{m.replace(':', '_').replace('/', '_')}_phi4.json").write_text(
        json.dumps([{**r, "grounded_phi4": g} for r, g in zip(rows, reg)], ensure_ascii=False, indent=2))
    print(f"RESULT[{JUDGE}] {m:24s} n={len(rows)}  GROUNDED(safe) {grounded}/{len(scored)} "
          f"({grounded/len(scored)*100:.1f}%)  | abstained {ab}/{len(rows)} ({ab/len(rows)*100:.1f}%)  "
          f"| HALLUCINATED {halluc}/{len(scored)} ({halluc/len(scored)*100:.1f}%)", flush=True)
