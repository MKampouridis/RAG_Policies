#!/usr/bin/env python3
"""Tier-0 item 2 (round-6 / Claude): scale the generator's miss-turn faithfulness
finding from n=13 to n~200 with ZERO labelling. The bake-off's decisive claim -
gemma3 faithfully ABSTAINS on retrieval misses while the 14B GUESSES from
parametric memory - rested on 13 miss turns (gemma3 12/13 vs 14B 11/13 = one
turn). Here we synthesise miss turns in bulk: pair each RoA question with sibling
documents that DEMONSTRABLY don't contain its answer (a synthetic 'wrong sibling
retrieved'), generate, and judge faithfulness-to-context.

On a wrong context, GROUNDED = the safe behaviour (faithful abstention, or stating
only facts actually in the wrong context); NOT GROUNDED = fabricating a specific
figure the context doesn't contain (parametric-memory hallucination - the exact
failure a policy assistant must avoid). We also report an explicit ABSTENTION rate
(regex) to decompose 'grounded' into refuse-vs-context-only. This is the standing
acceptance test for any future generator swap.

Sibling selection is DETERMINISTIC and guarantees answer-absence: a sibling is
eligible only if its full text contains NONE of the question's gold keyphrases.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/synthetic_miss_corpus.py [model ...]
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest import _get_collection
from src.llm import JUDGE_MODEL, chat
from src.rag import SYSTEM_PROMPT, _format_context

# inlined from generator_bakeoff.py (avoid eval-package import fragility) - identical judge
GROUND_PROMPT = """You are auditing whether an AI assistant's answer is FAITHFUL to the retrieved \
document excerpts it was given (its "context"). Judge ONLY faithfulness-to-context, not whether the \
answer is objectively correct or whether the right document was retrieved.

An answer is GROUNDED if every specific factual claim in it (numbers, thresholds, marks, credit \
values, time limits, conditions, procedures) is directly supported by the context. It is NOT \
grounded (a hallucination) if it states a specific fact that the context does not contain or that \
the context contradicts.

Ignore: the "Sources" citation list, any hedging or "this could relate to other documents" \
disclosure, and general framing sentences. If the answer plainly says the information isn't in the \
context / it can't answer, that is GROUNDED (a faithful abstention).

Respond with ONLY a JSON object: {"grounded": true or false}"""


def _clean(ans: str) -> str:
    ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.S)
    return re.split(r"\n+Sources?:", ans, flags=re.I)[0].strip()

QSETS = ["eval/questions.json", "eval/questions_set2.json"]
MODELS = [a for a in sys.argv[1:]] or ["gemma3:12b", "qwen2.5:14b-instruct"]
K_SIBLINGS = 5  # wrong siblings per question -> 40 RoA questions x 5 = ~200 miss turns
MANIFEST = json.loads(Path("data/manifest.json").read_text())["documents"]
_ABSTAIN = re.compile(
    r"\b(not (?:provided|in|included|contain|specif|mention|state|available|found)"
    r"|does not (?:specify|mention|state|contain|include|provide)"
    r"|no (?:information|details?|mention)|cannot (?:find|answer|determine|be answered)"
    r"|isn't (?:in|provided)|is not (?:in|provided|available)|unable to)\b",
    re.I,
)


def load_doc_chunks():
    """current RoA docs -> (up-to-6 chunks as a _format_context-ready dict, full lowercased text)."""
    coll = _get_collection()
    got = coll.get(include=["documents", "metadatas"])
    by_url = {}
    for doc, meta in zip(got["documents"], got["metadatas"]):
        if not meta.get("is_current") or meta.get("doc_type") != "rules_of_assessment":
            continue
        by_url.setdefault(meta.get("source_url", ""), {"docs": [], "metas": []})
        d = by_url[meta["source_url"]]
        if len(d["docs"]) < 6:
            d["docs"].append(doc)
            d["metas"].append(meta)
    texts = {}
    for url in by_url:
        p = Path((MANIFEST.get(url) or {}).get("text_cache_path", ""))
        texts[url] = p.read_text(encoding="utf-8").lower() if p.is_file() else ""
    return by_url, texts


def build_miss_turns():
    by_url, texts = load_doc_chunks()
    all_urls = sorted(by_url)
    turns = []
    for qpath in QSETS:
        for q in json.loads(Path(qpath).read_text()):
            if q.get("doc_type") != "rules_of_assessment":
                continue
            kps = [k.lower() for k in q.get("keyphrases", []) if k]
            gold_url = q["source_url"]
            # eligible wrong siblings: a DIFFERENT current RoA doc whose text has NONE of the gold keyphrases
            eligible = [u for u in all_urls if u != gold_url and texts.get(u)
                        and not any(k in texts[u] for k in kps)]
            for u in eligible[:K_SIBLINGS]:
                res = {"documents": [by_url[u]["docs"]], "metadatas": [by_url[u]["metas"]]}
                turns.append({
                    "question": q["question"],
                    "gold_url": gold_url.split("/")[-1],
                    "sibling_url": u.split("/")[-1],
                    "context": _format_context(res),
                })
    return turns


def judge_grounded(context, answer):
    raw = chat(messages=[{"role": "system", "content": GROUND_PROMPT},
                         {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{_clean(answer)}"}],
               format="json", model=JUDGE_MODEL)
    try:
        return bool(json.loads(raw).get("grounded", True))
    except Exception:
        return None


def run_model(model, turns):
    out = Path(f"eval/synthmiss_{model.replace(':', '_').replace('/', '_')}.json")
    if out.exists():
        return json.loads(out.read_text())
    rows = []
    for i, t in enumerate(turns, 1):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{t['context']}\n\nQuestion: {t['question']}"}]
        t0 = time.time()
        ans = chat(messages=msgs, model=model)
        rows.append({**{k: t[k] for k in ("question", "gold_url", "sibling_url")},
                     "answer": ans, "abstained": bool(_ABSTAIN.search(_clean(ans))),
                     "latency": time.time() - t0})
        if i % 25 == 0:
            print(f"    {model} gen {i}/{len(turns)}", flush=True)
    for r, t in zip(rows, turns):
        r["grounded"] = judge_grounded(t["context"], r["answer"])
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def summarize(model, rows):
    scored = [r for r in rows if r["grounded"] is not None]
    g = sum(1 for r in scored if r["grounded"])
    ab = sum(1 for r in rows if r["abstained"])
    # a hallucination = answered (no abstention) AND judged not-grounded (stated a fact not in context)
    halluc = sum(1 for r in scored if not r["grounded"] and not r["abstained"])
    print(f"RESULT {model:24s} n={len(rows)}  GROUNDED(safe) {g}/{len(scored)} ({g/len(scored)*100:.1f}%)  "
          f"| abstained {ab}/{len(rows)} ({ab/len(rows)*100:.1f}%)  "
          f"| HALLUCINATED-figure {halluc}/{len(scored)} ({halluc/len(scored)*100:.1f}%)  "
          f"| mean latency {sum(r['latency'] for r in rows)/len(rows):.0f}s", flush=True)


if __name__ == "__main__":
    turns = build_miss_turns()
    print(f"synthetic miss turns: {len(turns)} (RoA questions x up to {K_SIBLINGS} answer-absent siblings); judge={JUDGE_MODEL}\n", flush=True)
    for m in MODELS:
        print(f"=== {m} ===", flush=True)
        summarize(m, run_model(m, turns))
