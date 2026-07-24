#!/usr/bin/env python3
"""Generator bake-off with CLEAN isolation: compare answer generators on the
IDENTICAL retrieved context. Full end-to-end evals let follow-up retrieval drift
per generator (different primary answer -> different contextualized query ->
different retrieval), contaminating the comparison. Here retrieval is done ONCE
(contexts reconstructed from a reference run's history, cached), so any
difference between models is purely the generator.

For each candidate: generate all 80 turns' answers from the fixed contexts
(timed), then judge groundedness (faithfulness-to-context, the metric a stronger
generator moves). Reports groundedness overall / on hit-turns / RoA, plus mean
latency - the speed/quality frontier. Cross-family judge + answer_score are a
second pass on the finalists.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/generator_bakeoff.py [model ...]
"""
import json
import re
import sys
import time
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import DETERMINISTIC, DETERMINISTIC_OPTIONS, JUDGE_MODEL, chat
from src.rag import SYSTEM_PROMPT, _format_context, retrieve


def _generate(model_spec: str, msgs: list[dict]) -> str:
    """Generate an answer. A '::nothink' / '::think' suffix on the model name
    toggles reasoning for thinking models (qwen3) - to test whether qwen3's
    strong RoA groundedness comes FROM the thinking (lost when off) or is
    inherent (kept, with a big latency win). Plain names use chat() unchanged."""
    if "::" not in model_spec:
        return chat(messages=msgs, model=model_spec)
    real, mode = model_spec.split("::", 1)
    opts = DETERMINISTIC_OPTIONS if DETERMINISTIC else {"num_ctx": 8192}
    resp = ollama.chat(model=real, messages=msgs, options=opts, think=(mode == "think"))
    return resp["message"]["content"]

REF = Path("eval/results_qwen14b_full.json")
CTX_CACHE = Path("eval/bakeoff_contexts.json")

ROSTER = [
    "llama3.2:3b", "mistral:7b", "qwen2.5:7b-instruct", "qwen3:8b", "llama3.1:8b",
    "gemma3:12b", "phi4", "qwen2.5:14b-instruct", "qwen3:14b", "gpt-oss:20b",
]

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
    ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.S)  # reasoning models (qwen3 etc.)
    return re.split(r"\n+Sources?:", ans, flags=re.I)[0].strip()


def build_contexts() -> list[dict]:
    if CTX_CACHE.exists():
        return json.loads(CTX_CACHE.read_text())
    ref = json.loads(REF.read_text())
    ctxs = []
    for r in ref:
        hist = []
        for turn in ("primary", "follow_up"):
            t = r[turn]
            res, _ = retrieve(t["question"], list(hist))
            ctxs.append({
                "label": f"{r['source_title']}[{turn}]", "doc_type": r["doc_type"],
                "question": t["question"], "context": _format_context(res),
                "hit": t["retrieval"]["hit_at_6"],
            })
            hist += [{"role": "user", "content": t["question"]},
                     {"role": "assistant", "content": t["actual_answer"]}]
        print(f"  contexts: {r['source_title']}", flush=True)
    CTX_CACHE.write_text(json.dumps(ctxs, ensure_ascii=False))
    return ctxs


def judge_grounded(context: str, answer: str) -> bool | None:
    raw = chat(
        messages=[{"role": "system", "content": GROUND_PROMPT},
                  {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{_clean(answer)}"}],
        format="json", model=JUDGE_MODEL,
    )
    try:
        return bool(json.loads(raw).get("grounded", True))
    except Exception:
        return None


def run_model(model: str, ctxs: list[dict]) -> list[dict]:
    out_path = Path(f"eval/bakeoff_{model.replace(':', '_').replace('/', '_')}.json")
    rows = []
    for i, c in enumerate(ctxs, 1):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{c['context']}\n\nQuestion: {c['question']}"}]
        t0 = time.time()
        ans = _generate(model, msgs)
        rows.append({**c, "answer": ans, "latency": time.time() - t0})
    for r in rows:  # judge after all gens so the generator model isn't swapped in/out per turn
        r["grounded"] = judge_grounded(r["context"], r["answer"])
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def summarize(model: str, rows: list[dict]) -> None:
    scored = [r for r in rows if r["grounded"] is not None]
    rate = lambda sub: f"{sum(1 for r in sub if r['grounded']) / len(sub) * 100:.1f}%" if sub else "n/a"
    hit = [r for r in scored if r["hit"]]
    roa = [r for r in scored if r["doc_type"] == "rules_of_assessment"]
    lat = sum(r["latency"] for r in rows) / len(rows)
    print(f"RESULT {model:24s} grounded: overall {rate(scored)} | hit-turns {rate(hit)} | "
          f"RoA {rate(roa)} | mean latency {lat:.0f}s/answer", flush=True)


if __name__ == "__main__":
    models = sys.argv[1:] or ROSTER
    ctxs = build_contexts()
    print(f"contexts ready: {len(ctxs)} turns; judge={JUDGE_MODEL}\n", flush=True)
    for m in models:
        done_path = Path(f"eval/bakeoff_{m.replace(':', '_').replace('/', '_')}.json")
        if done_path.exists():  # resume: skip models already completed (survives a wedge/restart)
            summarize(m, json.loads(done_path.read_text()))
            print(f"    ({m} already done - skipped)\n", flush=True)
            continue
        print(f"=== generating + judging: {m} ===", flush=True)
        t0 = time.time()
        rows = run_model(m, ctxs)
        summarize(m, rows)
        print(f"    ({m} done in {(time.time() - t0) / 60:.1f} min)\n", flush=True)
