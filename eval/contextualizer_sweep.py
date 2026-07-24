"""Round 5 contextualizer test. The follow-up query contextualizer (rewrites
"what happens after that?" into a standalone question before retrieval) is the
one place the contextualizer model touches retrieval - so vary CONTEXTUALIZE_MODEL
and measure FOLLOW-UP retrieval only (primary turns have empty history and are
returned verbatim, so they are unaffected - asserted below as a sanity check).

Motivations: (1) does a stronger contextualizer improve follow-up retrieval? (2)
the UNIFIED-MODEL idea - if the production generator (gemma3:12b) is also a good
contextualizer, one model serves both roles and the 16GB Mac loads one big model
instead of two (halves RAM, kills the load-wedge). Baseline = current production
qwen2.5:7b-instruct.

Retrieval-only: no answer generation, no judge - cheap. family-hit@6 (gold family
in the reranked top-6) over both the tuned set and the holdout, mirroring
reranker_sweep.py, so a contextualizer that helps one set but not the other is
visible.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/contextualizer_sweep.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.rag as rag
from src.docid import document_family as fam
from src.llm import chat as _real_chat

SETS = {"main": "eval/results_qwen14b_full.json", "set2": "eval/results_set2_14b.json"}
# qwen3:8b is DISQUALIFIED as a contextualizer by latency: it ignores /no_think in
# Ollama and generates thinking tokens (~140s/turn, the bake-off bloat) - unusable
# on a follow-up's critical retrieval path regardless of hit@6. Pass models as args
# to run a subset: e.g. `... contextualizer_sweep.py gemma3:12b`
_ALL = ["qwen2.5:7b-instruct", "qwen2.5:14b-instruct", "gemma3:12b"]
MODELS = [a for a in sys.argv[1:] if not a.isdigit()] or _ALL
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def install_contextualizer(model: str):
    """Force every contextualize chat() onto `model`. qwen3 emits <think> blocks
    that would poison the rewrite (and trip _is_faithful_rewrite into discarding
    it), so disable thinking via /no_think and strip any residual think block -
    a fair capability test, not a thinking-token artifact."""
    is_qwen3 = model.startswith("qwen3")

    def _chat(messages, format=None, model=None, options=None):
        msgs = messages
        if is_qwen3:
            msgs = [dict(m) for m in messages]
            msgs[-1]["content"] += "\n/no_think"
        out = _real_chat(msgs, format=format, model=MODELS_ACTIVE[0], options=options)
        return _THINK.sub("", out).strip()

    MODELS_ACTIVE[0] = model
    rag.chat = _chat


MODELS_ACTIVE = [MODELS[0]]


def load_turns():
    turns = []  # (set, goldfam, doc_type, is_followup, question, history)
    for setname, path in SETS.items():
        for r in json.loads(Path(path).read_text()):
            goldfam = fam(r["source_url"])
            p, f = r["primary"], r["follow_up"]
            turns.append((setname, goldfam, r["doc_type"], False, p["question"], []))
            hist = [{"role": "user", "content": p["question"]},
                    {"role": "assistant", "content": p["actual_answer"]}]
            turns.append((setname, goldfam, r["doc_type"], True, f["question"], hist))
    return turns


def hit(goldfam, question, history):
    res, _ = rag.retrieve(question, list(history))
    return goldfam in {fam(m.get("source_url", "")) for m in res.get("metadatas", [[]])[0]}


turns = load_turns()
followups = [t for t in turns if t[3]]  # primary turns bypass the contextualizer (empty-history early return), so skip them
print(f"turns: {len(turns)} ({len(followups)} follow-up scored; primary turns are contextualizer-independent)\n", flush=True)

for model in MODELS:
    install_contextualizer(model)
    stats = {}  # set -> [hit, n]; split RoA vs all so the RoA-specific effect is visible
    roa = {}
    for setname, goldfam, dt, is_fu, q, hist in followups:
        h = hit(goldfam, q, hist)
        s = stats.setdefault(setname, [0, 0]); s[1] += 1; s[0] += int(h)
        if dt == "rules_of_assessment":
            r = roa.setdefault(setname, [0, 0]); r[1] += 1; r[0] += int(h)
    line = "  ".join(f"{sn} {h}/{n} ({h/n*100:.0f}%)" for sn, (h, n) in stats.items())
    tot_h, tot_n = sum(h for h, n in stats.values()), sum(n for h, n in stats.values())
    roa_h, roa_n = sum(h for h, n in roa.values()), sum(n for h, n in roa.values())
    print(f"RESULT {model:24s} follow-up hit@6:  {line}   overall {tot_h}/{tot_n} ({tot_h/tot_n*100:.1f}%)  "
          f"| RoA {roa_h}/{roa_n} ({roa_h/roa_n*100:.1f}%)", flush=True)
