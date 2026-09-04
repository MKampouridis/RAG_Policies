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

Cloud free-tier models (2026-09-04 investigation): pass "<provider>:<model>",
e.g. "groq:openai/gpt-oss-120b" or "gemini:gemini-2.5-flash". Requires
GROQ_API_KEY / GEMINI_API_KEY in the environment. Dispatches through
src.llm.generate() (its existing 429 retry/backoff), not a separate HTTP
path. Free-tier daily quotas are small enough that a single run can exhaust
one mid-list - run_model() saves after every turn and resumes from the last
completed one, so a quota-exhaustion crash costs one day, not all progress.
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


_CLOUD_PROVIDERS = ("groq", "gemini", "anthropic")


def _generate(model_spec: str, msgs: list[dict]) -> str:
    """Generate an answer. A '::nothink' / '::think' suffix on the model name
    toggles reasoning for thinking models (qwen3) - to test whether qwen3's
    strong RoA groundedness comes FROM the thinking (lost when off) or is
    inherent (kept, with a big latency win). "<provider>:<model>" (single
    colon, provider first) dispatches to a free-tier cloud generator. Plain
    names use chat() unchanged."""
    # Cleared per call: LAST_USAGE is a module global, so a local model (which
    # never writes it) would otherwise inherit the previous cloud call's counts
    # and silently mis-attribute them.
    import src.llm as llm
    llm.LAST_USAGE.clear()
    if ":" in model_spec and model_spec.split(":", 1)[0] in _CLOUD_PROVIDERS:
        return _cloud_generate(model_spec, msgs)
    if "::" not in model_spec:
        return chat(messages=msgs, model=model_spec)
    real, mode = model_spec.split("::", 1)
    opts = DETERMINISTIC_OPTIONS if DETERMINISTIC else {"num_ctx": 8192}
    resp = ollama.chat(model=real, messages=msgs, options=opts, think=(mode == "think"))
    return resp["message"]["content"]


def _cloud_generate(model_spec: str, msgs: list[dict]) -> str:
    """Routes through src.llm.generate() - its existing retry/backoff on 429s
    - by setting the module's GENERATOR_PROVIDER/MODEL globals rather than
    re-implementing the HTTP call here. reasoning_effort="low" for gpt-oss
    models only: unconstrained, gpt-oss-120b spent 48 of a 50-token budget on
    an invisible reasoning field before any visible answer (verified
    2026-09-04), eating free-tier quota far faster than raw TPD implies;
    "low" cut that to 17 tokens with the answer intact. Not applied to Qwen's
    thinking models - their <think> tags are visible content, not a separate
    field Groq's reasoning_effort controls, and _clean() already strips them
    before judging."""
    import src.llm as llm
    provider, model = model_spec.split(":", 1)
    llm.GENERATOR_PROVIDER = provider
    llm.GENERATOR_MODEL = model
    llm.GENERATOR_REASONING_EFFORT = "low" if "gpt-oss" in model else None
    return llm.generate(messages=msgs)


def _last_usage() -> dict:
    """Token counts for the call _generate() just made, or {} for local models
    (Ollama reports no billable usage and none is needed - local is free).
    Captured per turn because it is the ONE thing a paid comparison run cannot
    reconstruct later: the stored answer text supports re-judging any quality
    metric locally at no cost, but nothing recovers what a provider billed."""
    import src.llm as llm
    return dict(llm.LAST_USAGE)

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


def _out_path(model: str) -> Path:
    return Path(f"eval/bakeoff_{model.replace(':', '_').replace('/', '_')}.json")


def run_model(model: str, ctxs: list[dict]) -> list[dict]:
    out_path = _out_path(model)
    # Resume from a PARTIAL file, not just "fully done or not started" - a
    # free-tier daily quota can run out mid-list (this is the whole reason
    # cloud models are in scope here), so a crash must cost one day, not the
    # whole run. Saved after every generated turn, not just at the end.
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    start = len(rows)
    if start < len(ctxs):
        try:
            for c in ctxs[start:]:
                msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{c['context']}\n\nQuestion: {c['question']}"}]
                t0 = time.time()
                ans = _generate(model, msgs)
                rows.append({**c, "answer": ans, "latency": time.time() - t0,
                             "usage": _last_usage()})
                out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        except Exception as exc:
            # Judge what's already generated instead of returning nothing -
            # 69-75 of 80 turns is a large enough sample for a directional
            # read today, rather than waiting on a quota reset for the last
            # handful. Not re-raised: the caller distinguishes "complete" from
            # "partial" by comparing len(rows) to len(ctxs), not by exception.
            print(f"    stopped after {len(rows)}/{len(ctxs)} turns ({exc}); "
                  f"judging what's generated so far, re-run later to finish", flush=True)
    # judge after all gens so the generator model isn't swapped in/out per turn.
    # Saved after EVERY judgment, same reason generation is: a local 14B judge
    # over 80 turns runs longer than a harness background-task window, and
    # writing only at the end of the loop meant two consecutive kills lost all
    # judging progress and re-did it from zero (2026-09-04).
    for r in rows:
        if "grounded" not in r:
            r["grounded"] = judge_grounded(r["context"], r["answer"])
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def summarize(model: str, rows: list[dict], total: int | None = None) -> None:
    total = total if total is not None else len(rows)
    scored = [r for r in rows if r.get("grounded") is not None]
    rate = lambda sub: f"{sum(1 for r in sub if r['grounded']) / len(sub) * 100:.1f}%" if sub else "n/a"
    hit = [r for r in scored if r["hit"]]
    roa = [r for r in scored if r["doc_type"] == "rules_of_assessment"]
    lat = sum(r["latency"] for r in rows) / len(rows) if rows else 0
    partial = f" [PARTIAL {len(rows)}/{total}]" if len(rows) < total else ""
    print(f"RESULT {model:24s} grounded: overall {rate(scored)} | hit-turns {rate(hit)} | "
          f"RoA {rate(roa)} | mean latency {lat:.0f}s/answer{partial}", flush=True)


def _is_complete(rows: list[dict], total: int) -> bool:
    return len(rows) == total and all("grounded" in r for r in rows)


if __name__ == "__main__":
    models = sys.argv[1:] or ROSTER
    ctxs = build_contexts()
    print(f"contexts ready: {len(ctxs)} turns; judge={JUDGE_MODEL}\n", flush=True)
    for m in models:
        done_path = _out_path(m)
        existing = json.loads(done_path.read_text()) if done_path.exists() else []
        if _is_complete(existing, len(ctxs)):
            summarize(m, existing)
            print(f"    ({m} already done - skipped)\n", flush=True)
            continue
        print(f"=== generating + judging: {m} "
              f"({'resuming from ' + str(len(existing)) if existing else 'starting'}/{len(ctxs)}) ===",
              flush=True)
        t0 = time.time()
        # run_model() no longer raises on a quota-exhaustion stop - it judges
        # whatever was generated and returns. Completeness is read from the
        # row count, not from catching an exception, so a partial run still
        # yields a real (if partial) groundedness score today, and one
        # model's exhaustion never stops the rest of the roster from running -
        # each Groq model carries its own independent daily counter (observed
        # 2026-09-04: distinct remaining-token counts per model on one key).
        rows = run_model(m, ctxs)
        summarize(m, rows, total=len(ctxs))
        status = "done" if _is_complete(rows, len(ctxs)) else "incomplete - re-run later to finish"
        print(f"    ({m} {status}; {(time.time() - t0) / 60:.1f} min)\n", flush=True)
