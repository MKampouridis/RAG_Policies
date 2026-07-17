#!/usr/bin/env python3
"""Run the eval question set against the live app (HTTP API, so it exercises
the real conversation/memory path) plus a direct ranked-retrieval check
(bypassing the alphabetical re-sort used for citation display), and score
both retrieval and answer quality.

Usage: python eval/run_eval.py <output_name>
Writes eval/results_<output_name>.json
"""

import json
import sys
import time
from pathlib import Path

import requests

from src.llm import chat
from src.rag import retrieve as rag_retrieve

API_BASE = "http://127.0.0.1:8000"
QUESTIONS_PATH = Path("eval/questions.json")
N_RESULTS = 6

JUDGE_SYSTEM_PROMPT = """You are grading an AI assistant's answer to a question about University of \
Essex policy/rules-of-assessment documents. You are given the question, a ground-truth reference \
answer, and the assistant's actual answer. Score the assistant's answer on a 1-5 scale:

5 = fully correct and complete, matches the reference answer's substance
4 = mostly correct, minor omission or imprecision
3 = partially correct, missing significant substance or has a minor inaccuracy
2 = largely incorrect or mostly missing the point, but not a hallucination of false facts
1 = incorrect, contradicts the reference, or hallucinates facts not in the reference

Respond with ONLY a JSON object: {"score": <int 1-5>, "justification": "<one sentence>"}
"""


def create_conversation() -> str:
    resp = requests.post(f"{API_BASE}/api/conversations", json={})
    resp.raise_for_status()
    return resp.json()["id"]


def post_message(conv_id: str, content: str) -> dict:
    resp = requests.post(f"{API_BASE}/api/conversations/{conv_id}/messages", json={"content": content})
    resp.raise_for_status()
    return resp.json()


def ranked_retrieval(question_text: str, expected_url: str, history: list[dict]) -> dict:
    """Uses the exact same retrieval path as the live app (src.rag.retrieve),
    including query contextualization and recency preference, so this metric
    reflects what production actually does - not a simplified stand-in."""
    results, retrieval_query = rag_retrieve(question_text, history)
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    rank = None
    for i, meta in enumerate(metadatas, 1):
        if meta.get("source_url") == expected_url:
            rank = i
            break
    return {
        "rank": rank,
        "hit_at_6": rank is not None,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        "top_urls": [m.get("source_url") for m in metadatas],
        "top_distances": distances,
        "retrieval_query": retrieval_query,
    }


def keyphrase_coverage(answer: str, keyphrases: list[str]) -> float:
    if not keyphrases:
        return None
    answer_lower = answer.lower()
    hits = sum(1 for kp in keyphrases if kp.lower() in answer_lower)
    return hits / len(keyphrases)


def judge_answer(question: str, expected_answer: str, actual_answer: str) -> dict:
    user_prompt = (
        f"Question: {question}\n\nReference answer: {expected_answer}\n\n"
        f"Assistant's answer: {actual_answer}"
    )
    raw = chat(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
    )
    try:
        parsed = json.loads(raw)
        return {"score": int(parsed["score"]), "justification": parsed.get("justification", "")}
    except Exception as exc:
        return {"score": None, "justification": f"judge parse error: {exc}"}


def eval_one(item: dict) -> dict:
    conv_id = create_conversation()
    result = {
        "source_url": item["source_url"],
        "source_title": item["source_title"],
        "doc_type": item["doc_type"],
    }

    # primary question - no prior history
    retrieval = ranked_retrieval(item["question"], item["source_url"], history=[])
    api_result = post_message(conv_id, item["question"])
    judge = judge_answer(item["question"], item["expected_answer"], api_result["answer"])
    result["primary"] = {
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "actual_answer": api_result["answer"],
        "api_sources": api_result["sources"],
        "retrieval": retrieval,
        "keyphrase_coverage": keyphrase_coverage(api_result["answer"], item.get("keyphrases", [])),
        "judge": judge,
    }

    # follow-up question, same conversation (tests memory-aware retrieval too) -
    # history mirrors exactly what the live app would have loaded for this turn
    prior_history = [
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": api_result["answer"]},
    ]
    fu_retrieval = ranked_retrieval(item["follow_up_question"], item["source_url"], history=prior_history)
    fu_api_result = post_message(conv_id, item["follow_up_question"])
    fu_judge = judge_answer(item["follow_up_question"], item["follow_up_expected_answer"], fu_api_result["answer"])
    result["follow_up"] = {
        "question": item["follow_up_question"],
        "expected_answer": item["follow_up_expected_answer"],
        "actual_answer": fu_api_result["answer"],
        "api_sources": fu_api_result["sources"],
        "retrieval": fu_retrieval,
        "keyphrase_coverage": keyphrase_coverage(fu_api_result["answer"], item.get("follow_up_keyphrases", [])),
        "judge": fu_judge,
    }

    return result


def run(output_name: str) -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    output_path = Path(f"eval/results_{output_name}.json")

    results = []
    for i, item in enumerate(questions, 1):
        t0 = time.time()
        try:
            r = eval_one(item)
            results.append(r)
            elapsed = time.time() - t0
            print(
                f"[{i}/{len(questions)}] ({elapsed:.1f}s) "
                f"primary hit@6={r['primary']['retrieval']['hit_at_6']} score={r['primary']['judge']['score']} | "
                f"followup hit@6={r['follow_up']['retrieval']['hit_at_6']} score={r['follow_up']['judge']['score']} "
                f"-- {item['source_title']}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{i}/{len(questions)}] FAILED for {item['source_title']}: {exc}", flush=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nDone. Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "run"
    run(name)
