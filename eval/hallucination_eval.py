"""Round 4 (Fable 5/Gemini/DeepSeek): hallucination / groundedness eval -
the measurement gap the project had no coverage for. For each answer already
generated in a results file, reconstruct the EXACT context it was generated
from (re-retrieve deterministically -> _format_context) and judge whether
every factual claim in the answer is supported by that context.

This measures FAITHFULNESS-TO-CONTEXT (intrinsic hallucination), which is
distinct from - and orthogonal to - retrieval hit@6: an answer that faithfully
reports a wrong-sibling document's rules is grounded-but-wrong (that's the
strict-vs-evidence gap, not a hallucination). An answer that states a figure
absent from its context is a hallucination even if hit@6 was a hit.

Run with RAG_DETERMINISTIC=1 for reproducibility (re-retrieval reruns the
contextualizer). Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/hallucination_eval.py [results_file]
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag import retrieve, _format_context
from src.llm import chat

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/results_c1_anchor_v2.json")
JUDGE_MODEL = "qwen2.5:14b-instruct"

JUDGE_PROMPT = """You are auditing whether an AI assistant's answer is FAITHFUL to the retrieved \
document excerpts it was given (its "context"). Judge ONLY faithfulness-to-context, not whether the \
answer is objectively correct or whether the right document was retrieved.

An answer is GROUNDED if every specific factual claim in it (numbers, thresholds, marks, credit \
values, time limits, conditions, procedures) is directly supported by the context. It is NOT \
grounded (a hallucination) if it states a specific fact that the context does not contain or that \
the context contradicts.

Ignore: the "Sources" citation list, any hedging or "this could relate to other documents" \
disclosure, and general framing sentences. If the answer plainly says the information isn't in the \
context / it can't answer, that is GROUNDED (a faithful abstention).

Respond with ONLY a JSON object: {"grounded": true or false, "unsupported": "<the single most \
clearly unsupported specific claim, or empty string if grounded>"}"""

def strip_answer(a):
    # drop the trailing Sources block for judging (it's citations, not claims)
    return re.split(r"\n+Sources?:", a, flags=re.I)[0].strip()

def judge(context, answer):
    raw = chat(messages=[
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{strip_answer(answer)}"},
    ], format="json", model=JUDGE_MODEL)
    try:
        p = json.loads(raw)
        return bool(p.get("grounded", True)), p.get("unsupported", "")
    except Exception:
        return None, "judge parse error"

results = json.loads(RESULTS.read_text())
rows = []  # (label, doc_type, hit, grounded, unsupported)
for i, r in enumerate(results, 1):
    hist = []
    for turn in ("primary", "follow_up"):
        t = r[turn]
        res, _ = retrieve(t["question"], list(hist))
        context = _format_context(res)
        g, unsup = judge(context, t["actual_answer"])
        rows.append((f"{r['source_title']}[{turn}]", r["doc_type"], t["retrieval"]["hit_at_6"], g, unsup))
        hist += [{"role": "user", "content": t["question"]}, {"role": "assistant", "content": t["actual_answer"]}]
    print(f"[{i}/{len(results)}] {r['source_title']}: "
          f"primary grounded={rows[-2][3]} | followup grounded={rows[-1][3]}", flush=True)

scored = [x for x in rows if x[3] is not None]
grounded = [x for x in scored if x[3]]
hallucinated = [x for x in scored if not x[3]]
def rate(sub): return f"{sum(1 for x in sub if x[3])/len(sub)*100:.1f}%" if sub else "n/a"
print(f"\n=== Groundedness (faithfulness-to-context) ===")
print(f"overall: {rate(scored)}  ({len(grounded)}/{len(scored)} answers grounded)")
print(f"  on hit@6 turns:  {rate([x for x in scored if x[2]])}")
print(f"  on miss turns:   {rate([x for x in scored if not x[2]])}")
print(f"  RoA:  {rate([x for x in scored if x[1]=='rules_of_assessment'])}")
print(f"  Policy: {rate([x for x in scored if x[1]=='policy'])}")
print(f"\n--- {len(hallucinated)} answers judged NOT grounded (candidate hallucinations) ---")
for lbl, dt, hit, g, unsup in hallucinated:
    print(f"   hit@6={hit}  {lbl}: {unsup}")
Path("eval/results_hallucination.json").write_text(json.dumps(
    [{"turn": x[0], "doc_type": x[1], "hit_at_6": x[2], "grounded": x[3], "unsupported": x[4]} for x in rows],
    indent=2, ensure_ascii=False))
