#!/usr/bin/env python3
"""LLM-experiments phase, step 1: re-score existing answers with a stronger
judge model, without regenerating any answers or re-running retrieval. The
production judge (qwen2.5:7b-instruct) is the same model that generates
answers, so it may be too generous on its own mistakes - this checks how
much of the ~3.9/5 answer score gap to 5 is real vs a self-judging artifact,
before spending a full eval cycle on a generator swap.

Usage: PYTHONPATH=. python eval/rejudge.py <judge_model> [results_path]
Writes eval/results_<results_stem>_rejudged_<model>.json (same schema, judge
scores replaced) and prints an old-vs-new comparison.
"""

import json
import sys
from pathlib import Path

from eval.run_eval import judge_answer

JUDGE_MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b-instruct"
RESULTS_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("eval/results_stage_colbert.json")
OUT_PATH = Path(f"eval/results_{RESULTS_PATH.stem.replace('results_', '')}_rejudged_{JUDGE_MODEL.replace(':', '_')}.json")


def main():
    data = json.loads(RESULTS_PATH.read_text())
    old_scores, new_scores = [], []

    for i, item in enumerate(data, 1):
        for tk in ("primary", "follow_up"):
            turn = item[tk]
            old = turn["judge"]
            new = judge_answer(turn["question"], turn["expected_answer"], turn["actual_answer"], model=JUDGE_MODEL)
            turn["judge_original_7b"] = old
            turn["judge"] = new
            if old.get("score") is not None:
                old_scores.append(old["score"])
            if new.get("score") is not None:
                new_scores.append(new["score"])
        print(f"[{i}/{len(data)}] {item['source_title']}: "
              f"primary {item['primary']['judge_original_7b']['score']}->{item['primary']['judge']['score']} | "
              f"followup {item['follow_up']['judge_original_7b']['score']}->{item['follow_up']['judge']['score']}",
              flush=True)
        OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"\nOld judge (qwen2.5:7b) mean: {sum(old_scores) / len(old_scores):.3f}")
    print(f"New judge ({JUDGE_MODEL}) mean: {sum(new_scores) / len(new_scores):.3f}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
