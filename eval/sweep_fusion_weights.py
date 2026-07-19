#!/usr/bin/env python3
"""Fast retrieval-only sweep over Stage F's (DENSE_WEIGHT, BM25_WEIGHT) pairs,
used to narrow candidates before committing to the full (slow, judge-scored)
80-turn eval on the single best config. Skips the answer-generation and
judge LLM calls entirely - only retrieval is measured - so each config's
80-turn pass costs seconds per turn instead of ~150s.

The follow-up turn's history uses the *reference* expected_answer as a stand-in
for what the assistant would have said, since generating a real answer is
exactly the expensive step this script is avoiding. That's an approximation,
not production behavior - fine for *relative* ranking of weight configs, but
the eventual winner still needs a real 80-turn eval (eval/run_eval.py) before
being trusted as a production change.

Usage: python eval/sweep_fusion_weights.py [questions_path]
"""

import json
import sys
from pathlib import Path

import src.rag as rag

QUESTIONS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/questions.json")

CONFIGS = [
    ("rrf_baseline", None, None),
    ("50/50", 0.5, 0.5),
    ("60/40", 0.6, 0.4),
    ("70/30", 0.7, 0.3),
    ("40/60", 0.4, 0.6),
    ("30/70", 0.3, 0.7),
]


def hit_and_rr(metadatas: list[dict], expected_url: str) -> tuple[bool, float]:
    for i, meta in enumerate(metadatas, 1):
        if meta.get("source_url") == expected_url:
            return True, 1.0 / i
    return False, 0.0


def run_config(questions: list[dict], dense_weight, bm25_weight) -> dict:
    if dense_weight is None:
        rag.WEIGHTED_FUSION_ENABLED = False
    else:
        rag.WEIGHTED_FUSION_ENABLED = True
        rag.DENSE_WEIGHT = dense_weight
        rag.BM25_WEIGHT = bm25_weight

    hits, rrs, roa_hits, roa_rrs, roa_n = 0, [], 0, [], 0
    n = 0
    for item in questions:
        expected_url = item["source_url"]
        is_roa = item["doc_type"] == "rules_of_assessment"

        results, _ = rag.retrieve(item["question"], [])
        metas = results.get("metadatas", [[]])[0]
        hit, rr = hit_and_rr(metas, expected_url)
        hits += hit
        rrs.append(rr)
        n += 1
        if is_roa:
            roa_hits += hit
            roa_rrs.append(rr)
            roa_n += 1

        prior_history = [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["expected_answer"]},
        ]
        fu_results, _ = rag.retrieve(item["follow_up_question"], prior_history)
        fu_metas = fu_results.get("metadatas", [[]])[0]
        fu_hit, fu_rr = hit_and_rr(fu_metas, expected_url)
        hits += fu_hit
        rrs.append(fu_rr)
        n += 1
        if is_roa:
            roa_hits += fu_hit
            roa_rrs.append(fu_rr)
            roa_n += 1

    return {
        "overall_hit_at_6": hits / n,
        "overall_mrr": sum(rrs) / len(rrs),
        "roa_hit_at_6": roa_hits / roa_n if roa_n else None,
        "roa_mrr": sum(roa_rrs) / len(roa_rrs) if roa_rrs else None,
    }


def main():
    questions = json.loads(QUESTIONS_PATH.read_text())
    print(f"Sweeping {len(CONFIGS)} configs over {len(questions)} questions ({len(questions) * 2} turns each)\n")
    for name, dw, bw in CONFIGS:
        result = run_config(questions, dw, bw)
        print(f"{name:14s} overall hit@6={result['overall_hit_at_6']*100:5.1f}% mrr={result['overall_mrr']:.3f} | "
              f"RoA hit@6={result['roa_hit_at_6']*100:5.1f}% mrr={result['roa_mrr']:.3f}", flush=True)


if __name__ == "__main__":
    main()
