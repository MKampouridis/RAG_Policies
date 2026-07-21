#!/usr/bin/env python3
"""Phase 5 (external code review round 2, 2026-07-21, Fable 5): scripted
multi-turn conversation probe - the one dimension none of the project's
three existing question sets cover (all are single-topic, 2-turn primary +
one follow-up). Fable 5's point: both real bugs this project's harness ever
missed (a user manually finding the contextualizer-drift bug in a live
topic-switching conversation; Stage H's follow-up knock-on effect) live in
exactly this regime - long conversations with topic switches, "going back
to" an earlier topic, and cross-document comparisons.

Runs each scripted conversation (eval/questions_set4_multiturn.json)
sequentially against the live API within a single conversation (so memory/
history behaves exactly as it would for a real user), scoring retrieval
hit@6 per turn plus logging the contextualizer's actual retrieval_query for
every turn - especially "return"/"comparison"/"switch" turns, where a
faithful rewrite should clearly reference the expected topic and an
unfaithful one (the exact failure mode the real postfix3->postfix4 bug was)
would visibly reference the wrong one instead. Deliberately does NOT
automate a "faithfulness" classifier - logging the rewrite next to the
expected topic makes this directly reviewable rather than trusting a new,
unvalidated heuristic metric.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/run_multiturn_eval.py [output_name]
Writes eval/results_set4_multiturn_<output_name>.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_eval import create_conversation, judge_answer, keyphrase_coverage, post_message

CONVERSATIONS_PATH = Path("eval/questions_set4_multiturn.json")


def run_conversation(conv_spec: dict) -> dict:
    conv_id = create_conversation()
    turns_out = []
    for i, turn in enumerate(conv_spec["turns"], 1):
        t0 = time.time()
        api_result = post_message(conv_id, turn["question"])
        elapsed = time.time() - t0

        expected_url = turn["expected_source_url"]
        ranked_top_urls = api_result["ranked_top_urls"]
        rank = next((rk for rk, u in enumerate(ranked_top_urls, 1) if u == expected_url), None)

        turns_out.append({
            "turn_index": i,
            "turn_type": turn["turn_type"],
            "question": turn["question"],
            "expected_source_url": expected_url,
            "retrieval_query": api_result["retrieval_query"],
            "rank": rank,
            "hit_at_6": rank is not None,
            "ranked_top_urls": ranked_top_urls,
            "answer": api_result["answer"],
            "keyphrase_coverage": keyphrase_coverage(api_result["answer"], turn.get("expected_keyphrases", [])),
            "elapsed_s": round(elapsed, 1),
        })
        print(
            f"  [{i}/{len(conv_spec['turns'])}] ({elapsed:.1f}s) {turn['turn_type']:12s} "
            f"hit@6={rank is not None!s:5s} rank={rank} -- {turn['question'][:70]}",
            flush=True,
        )
        print(f"      rewrite: {api_result['retrieval_query']!r}", flush=True)

    return {"conversation_id": conv_spec["conversation_id"], "description": conv_spec["description"], "turns": turns_out}


def run(output_name: str) -> None:
    conversations = json.loads(CONVERSATIONS_PATH.read_text())
    output_path = Path(f"eval/results_set4_multiturn_{output_name}.json")

    results = []
    for i, conv_spec in enumerate(conversations, 1):
        print(f"\n=== conversation {i}/{len(conversations)}: {conv_spec['conversation_id']} ===", flush=True)
        try:
            results.append(run_conversation(conv_spec))
        except Exception as exc:
            print(f"FAILED conversation {conv_spec['conversation_id']}: {exc}", flush=True)
            raise
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    total_turns = sum(len(c["turns"]) for c in results)
    assert total_turns == sum(len(c["turns"]) for c in conversations), "turn count mismatch - a conversation was dropped"

    # summary by turn_type
    by_type: dict[str, list[bool]] = {}
    for c in results:
        for t in c["turns"]:
            by_type.setdefault(t["turn_type"], []).append(t["hit_at_6"])
    print(f"\n=== Summary ({total_turns} turns across {len(results)} conversations) ===")
    for tt, hits in sorted(by_type.items()):
        print(f"  {tt:12s} n={len(hits):3d}  hit@6={sum(hits)/len(hits)*100:5.1f}%")
    print(f"\nDone. Wrote {len(results)} conversations to {output_path}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "run"
    run(name)
