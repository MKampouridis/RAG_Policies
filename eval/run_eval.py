#!/usr/bin/env python3
"""Run the eval question set against the live app (HTTP API, so it exercises
the real conversation/memory path) and score both retrieval and answer
quality.

Scores the exact retrieval that produced the answer - src.rag.answer() now
returns its own retrieval_query/ranked_top_urls (Phase 1 fix, 2026-07-21,
external code review round: this eval used to call retrieve() a second,
independently-sampled time via its own ranked_retrieval() helper, which
could diverge from what the live app actually retrieved on follow-up turns
since the query contextualizer is an LLM sample. One retrieve() call per
turn now, via the API response.

For eval runs, start both this script and the server (run_server.py) with
RAG_DETERMINISTIC=1 set (src/llm.py) - otherwise repeat runs on identical
code can still show different hit@6/answer scores from Ollama's default
sampling, not a real change.

Usage: python eval/run_eval.py <output_name>
Writes eval/results_<output_name>.json
"""

import json
import sys
import time
from pathlib import Path

import requests

from src.llm import chat

API_BASE = "http://127.0.0.1:8000"
QUESTIONS_PATH = Path("eval/questions.json")
N_RESULTS = 6
MAX_ATTEMPTS = 2

# Judge model upgraded from qwen2.5:7b-instruct (2026-07-20, eval/rejudge.py):
# re-scoring the existing baseline's answers with a stronger judge, unchanged
# otherwise, found the 7b judge - which also generates the answers - was
# specifically over-crediting RoA wrong-sibling boilerplate answers (RoA mean
# 3.80->3.48, misses-only 3.33->2.67, catching genuine factual contradictions
# the 7b judge missed) while policy scores were unaffected or slightly higher
# (3.98->4.15). All answer_score comparisons before this date used the 7b
# judge; comparing across the switch requires re-judging, not just re-running.
JUDGE_MODEL = "qwen2.5:14b-instruct"

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


def score_retrieval(expected_url: str, retrieval_query: str, ranked_top_urls: list[str]) -> dict:
    """Scores the retrieval the live app's API response already reports for
    this turn - not a second, separately-sampled retrieve() call (see module
    docstring)."""
    rank = None
    for i, url in enumerate(ranked_top_urls, 1):
        if url == expected_url:
            rank = i
            break
    return {
        "rank": rank,
        "hit_at_6": rank is not None,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        "top_urls": ranked_top_urls,
        "retrieval_query": retrieval_query,
    }


def keyphrase_coverage(answer: str, keyphrases: list[str]) -> float:
    if not keyphrases:
        return None
    answer_lower = answer.lower()
    hits = sum(1 for kp in keyphrases if kp.lower() in answer_lower)
    return hits / len(keyphrases)


def judge_answer(question: str, expected_answer: str, actual_answer: str, model: str = JUDGE_MODEL) -> dict:
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
        model=model,
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
    api_result = post_message(conv_id, item["question"])
    retrieval = score_retrieval(item["source_url"], api_result["retrieval_query"], api_result["ranked_top_urls"])
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

    # follow-up question, same conversation (tests memory-aware retrieval too)
    fu_api_result = post_message(conv_id, item["follow_up_question"])
    fu_retrieval = score_retrieval(
        item["source_url"], fu_api_result["retrieval_query"], fu_api_result["ranked_top_urls"]
    )
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


def run(output_name: str, questions_path: Path = QUESTIONS_PATH) -> None:
    """Retries a failing question once (transient Ollama/HTTP hiccups
    shouldn't silently shrink the denominator - a prior run lost 16/40
    questions to one dropped connection and nobody noticed until the summary
    line said "Wrote 24 results" instead of 40). Hard-fails after a second
    failure rather than continuing to write partial results, since a
    partial-but-silently-succeeding run is exactly the failure mode that
    caused the original bug."""
    questions = json.loads(questions_path.read_text())
    output_path = Path(f"eval/results_{output_name}.json")

    results = []
    for i, item in enumerate(questions, 1):
        t0 = time.time()
        last_exc = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                r = eval_one(item)
                elapsed = time.time() - t0
                results.append(r)
                print(
                    f"[{i}/{len(questions)}] ({elapsed:.1f}s) "
                    f"primary hit@6={r['primary']['retrieval']['hit_at_6']} score={r['primary']['judge']['score']} | "
                    f"followup hit@6={r['follow_up']['retrieval']['hit_at_6']} score={r['follow_up']['judge']['score']} "
                    f"-- {item['source_title']}",
                    flush=True,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_ATTEMPTS:
                    print(f"[{i}/{len(questions)}] attempt {attempt} FAILED for {item['source_title']}: {exc} - retrying", flush=True)
                    time.sleep(5)
        if last_exc is not None:
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            raise RuntimeError(
                f"[{i}/{len(questions)}] FAILED for {item['source_title']} after {MAX_ATTEMPTS} attempts: {last_exc}"
                f" - wrote {len(results)}/{len(questions)} results to {output_path} before stopping"
            ) from last_exc
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    assert len(results) == len(questions), (
        f"wrote {len(results)} results but expected {len(questions)} - denominator would be silently wrong"
    )
    print(f"\nDone. Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "run"
    q_path = Path(sys.argv[2]) if len(sys.argv) > 2 else QUESTIONS_PATH
    run(name, q_path)
