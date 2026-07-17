#!/usr/bin/env python3
"""Generate a grounded eval question set from the 40 selected source
documents. For each document, ask the LLM to draft one specific factual
question (answerable only from that document) plus a natural follow-up,
with ground-truth answers and key phrases for scoring. Output is reviewed
and hand-edited afterward, not used blind."""

import json
from pathlib import Path

from src.llm import chat

SELECTED_PATH = Path("eval/selected_docs.json")
OUTPUT_PATH = Path("eval/questions.json")

GEN_SYSTEM_PROMPT = """You write evaluation questions for a RAG system over University of Essex \
policy and rules-of-assessment documents. Given the title and text of ONE document, produce:

1. One specific, factual question that a student or staff member would plausibly ask, which is \
answerable ONLY from this document's content (not generic, not answerable from the title alone).
2. A concise ground-truth answer (2-4 sentences) based strictly on the given text.
3. Two or three short key phrases from the text that a correct answer must contain.
4. A natural FOLLOW-UP question a person would ask next in the same conversation (e.g. asking for \
more detail, an edge case, or a related consequence). It should still be answerable from this same \
document's content.
5. A concise ground-truth answer for the follow-up, and its key phrases.

Respond with ONLY a JSON object with keys: "question", "expected_answer", "keyphrases" (array), \
"follow_up_question", "follow_up_expected_answer", "follow_up_keyphrases" (array).
"""


def generate_for_doc(doc: dict) -> dict:
    text = Path(doc["text_cache_path"]).read_text(encoding="utf-8")[:4000]
    user_prompt = f"Title: {doc['title']}\nDocument type: {doc['doc_type']}\n\nText:\n{text}"

    raw = chat(
        messages=[
            {"role": "system", "content": GEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
    )
    parsed = json.loads(raw)
    return {
        "source_url": doc["url"],
        "source_title": doc["title"],
        "doc_type": doc["doc_type"],
        "department": doc.get("department"),
        "academic_year": doc.get("academic_year"),
        **parsed,
    }


def run() -> None:
    selected = json.loads(SELECTED_PATH.read_text())
    results = []
    for i, doc in enumerate(selected, 1):
        try:
            item = generate_for_doc(doc)
            results.append(item)
            print(f"[{i}/{len(selected)}] OK: {item['question'][:80]}", flush=True)
        except Exception as exc:
            print(f"[{i}/{len(selected)}] FAILED for {doc['title']}: {exc}", flush=True)
        OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
