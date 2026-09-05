#!/usr/bin/env python3
"""Re-score stored bake-off answers with a DIFFERENT judge, to separate a real
quality gap from a judge artifact.

Why this exists: the 2026-09-04 free-tier bake-off rests on a 10-point
groundedness gap (gpt-oss-120b 94% vs claude-sonnet-5 84%), and this ledger
records that the judge ALONE moves this metric by ~9 points - the same order as
the finding. A second, cross-family judge on the SAME stored answers is the
cheap way to tell those apart: no regeneration, no API spend, and generation
variance is held fixed because the answers are byte-identical.

Writes <file>_<judge>.json alongside the original; never modifies it.
Saved after every judgment so a kill costs one turn, not the run.

Usage: PYTHONPATH=. python eval/bakeoff_rejudge.py <judge-model> <model-spec> [...]
  e.g. PYTHONPATH=. python eval/bakeoff_rejudge.py phi4 groq:openai/gpt-oss-120b
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.generator_bakeoff import GROUND_PROMPT, _clean, _out_path
from src.llm import chat


def judge_with(model: str, context: str, answer: str) -> bool | None:
    raw = chat(
        messages=[{"role": "system", "content": GROUND_PROMPT},
                  {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{_clean(answer)}"}],
        format="json", model=model,
    )
    try:
        return bool(json.loads(raw).get("grounded", True))
    except Exception:
        return None


def main() -> int:
    judge = sys.argv[1]
    models = sys.argv[2:]
    if not models:
        print(__doc__)
        return 2
    safe_judge = re.sub(r"[^a-zA-Z0-9_.-]", "_", judge)
    print(f"re-judging with {judge}\n", flush=True)
    for m in models:
        src = _out_path(m)
        if not src.exists():
            print(f"{m}: no bake-off file, skipped", flush=True)
            continue
        out = src.with_name(f"{src.stem}_{safe_judge}.json")
        rows = json.loads(out.read_text()) if out.exists() else json.loads(src.read_text())
        key = f"grounded_{safe_judge}"
        for r in rows:
            if key in r:
                continue
            r[key] = judge_with(judge, r["context"], r["answer"])
            out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        scored = [r for r in rows if r.get(key) is not None]
        orig = [r for r in rows if r.get("grounded") is not None]
        rate = lambda sub, k: f"{sum(1 for r in sub if r[k]) / len(sub) * 100:.1f}%" if sub else "n/a"
        roa = [r for r in scored if r["doc_type"] == "rules_of_assessment"]
        print(f"RESULT {m:26s} {judge}: overall {rate(scored, key)} | RoA {rate(roa, key)}"
              f"   (was {rate(orig, 'grounded')} under the original judge, n={len(scored)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
