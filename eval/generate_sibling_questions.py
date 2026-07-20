#!/usr/bin/env python3
"""J5(b): generate a sibling-DISCRIMINATING question set from the J1 identity
records. Unlike eval/generate_questions.py (whose questions often carry no
document-identifying detail, making many of them genuinely ambiguous across
siblings - see eval/report.md "J5"), every question here is REQUIRED to name
the programme/department/partner institution from the document's extracted
identity record, so the correct sibling is identifiable from the question
text alone. This stresses exactly the discrimination failure mode the
original set under-measures.

Selection: current RoA documents whose identity record has a non-empty
programme_name or partner_institution (the fields that distinguish true
siblings), sampled across distinct programmes.

Usage: PYTHONPATH=. python eval/generate_sibling_questions.py [n_docs]
Writes eval/questions_set3_sibling.json (same schema as questions.json,
reusable directly by eval/run_eval.py). Reviewed afterward, not used blind.
"""

import json
import sys
from pathlib import Path

from src.ingest import url_hash
from src.llm import chat

N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MANIFEST_PATH = Path("data/manifest.json")
IDENTITY_DIR = Path("data/doc_identity")
OUT_PATH = Path("eval/questions_set3_sibling.json")

GEN_SYSTEM_PROMPT = """You write evaluation questions for a RAG system over University of Essex \
rules-of-assessment documents. Given ONE document's identity (programme name, department, partner \
institution) and its text, produce:

1. One specific, factual question that EXPLICITLY NAMES the programme (and partner institution if \
one is given) so the question is unambiguous about which document it refers to, and is answerable \
from this document's content. Example style: "What is the pass mark for core modules on the MSc \
Periodontology programme?" - never a generic question like "What is the pass mark for core modules?"
2. A concise ground-truth answer (2-4 sentences) based strictly on the given text.
3. Two or three short key phrases from the text that a correct answer must contain (specific \
numbers, thresholds, or defined terms - not generic words).
4. A natural FOLLOW-UP question that also names the programme, answerable from this same document.
5. A concise ground-truth answer for the follow-up, and its key phrases.

Respond with ONLY a JSON object with keys: "question", "expected_answer", "keyphrases" (array), \
"follow_up_question", "follow_up_expected_answer", "follow_up_keyphrases" (array)."""


def pick_documents(manifest: dict) -> list[tuple[str, dict, dict]]:
    picked, seen_programmes = [], set()
    for url, doc in manifest.items():
        if not doc.get("keep") or doc.get("doc_type") != "rules_of_assessment":
            continue
        identity_path = IDENTITY_DIR / f"{url_hash(url)}.json"
        if not identity_path.exists():
            continue
        identity = json.loads(identity_path.read_text())
        key = identity.get("programme_name") or identity.get("partner_institution")
        if not key or key in seen_programmes:
            continue
        # prefer current editions - crude filename check for a recent year
        if not any(tok in url for tok in ("-25", "_25", "25-26", "/current/")):
            continue
        seen_programmes.add(key)
        picked.append((url, doc, identity))
        if len(picked) >= N_DOCS:
            break
    return picked


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())["documents"]
    picked = pick_documents(manifest)
    print(f"Generating sibling-discriminating questions for {len(picked)} documents", flush=True)

    questions = []
    for i, (url, doc, identity) in enumerate(picked, 1):
        text = Path(doc["text_cache_path"]).read_text(encoding="utf-8")[:4000]
        identity_line = " | ".join(filter(None, [
            identity.get("programme_name"),
            identity.get("department"),
            identity.get("partner_institution"),
        ]))
        raw = chat(
            messages=[
                {"role": "system", "content": GEN_SYSTEM_PROMPT},
                {"role": "user", "content": f"Identity: {identity_line}\nFilename: {url.rsplit('/', 1)[-1]}\n\nText:\n{text}"},
            ],
            format="json",
        )
        try:
            q = json.loads(raw)
        except Exception as exc:
            print(f"[{i}/{len(picked)}] parse error for {url.rsplit('/', 1)[-1]}: {exc}", flush=True)
            continue
        q.update({
            "source_url": url,
            "source_title": url.rsplit("/", 1)[-1],
            "doc_type": doc["doc_type"],
        })
        questions.append(q)
        print(f"[{i}/{len(picked)}] {q['question'][:90]}", flush=True)

    OUT_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    print(f"\nDone. Wrote {len(questions)} question pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
