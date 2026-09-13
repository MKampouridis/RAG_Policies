#!/usr/bin/env python3
"""Head-to-head on POLICY questions: is gpt-oss-120b worse than Sonnet here?

THE ONE OPEN QUESTION. The 80-turn bake-off (2026-09-04) moved production from
Sonnet to Groq's gpt-oss-120b: better groundedness (94% vs 84%), better on
rules-of-assessment (4.39 vs 4.03), ~26x cheaper, ~6x faster. Sonnet's single
remaining edge was COMPLETENESS ON POLICY-TYPE QUESTIONS (4.58 vs 4.38). That
is the only thing left unsettled, so this tests exactly that and nothing else.

DESIGN, and why each part is the way it is.

FIXED CONTEXTS. Retrieval runs ONCE per question and both models answer from
the identical context. Otherwise a difference could come from retrieval rather
than generation, and there would be no way to tell which.

REAL QUESTIONS. 40 policy-type questions taken from actual traffic (see
eval/build_traffic_regression.py), preferring ones a person rated - not a
curated list written to probe a hypothesis.

PAIRWISE, NOT SCORED AGAINST GOLD. These questions have no gold answers, and
inventing some would make the result a measurement of my invented gold. The
judge sees the context, the question and two anonymous answers, and says which
better answers it. That is blind by construction.

BOTH ORDERS. Every pair is judged twice, A/B and B/A. Position bias in LLM
judges is large and well known; running both orders MEASURES it instead of
hoping it is absent. A pair that flips when swapped is counted as a tie,
because that is what it is - the judge could not tell them apart.

NEUTRAL JUDGE. phi4, which is neither contestant. Same-family self-preference
swung a result by 24 points in this project's own history.

Usage:
  # 1. generate (cloud; production server can stay up)
  PYTHONPATH=. python eval/policy_arena.py generate
  # 2. judge (local phi4, 9.1GB - STOP the production server first)
  PYTHONPATH=. python eval/policy_arena.py judge
  # 3. report
  PYTHONPATH=. python eval/policy_arena.py report
"""
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = Path("eval/policy_arena_questions.json")
CTX_PATH = Path("eval/policy_arena_contexts.json")
GEN_PATH = Path("eval/policy_arena_answers.json")
JUDGE_PATH = Path("eval/policy_arena_judged.json")

ARMS = {"gpt-oss": "groq:openai/gpt-oss-120b",
        "sonnet": "anthropic:claude-sonnet-5"}
# Variant arms: same model, different prompt. The point of keeping Sonnet's
# answers untouched is that a prompt change applied to BOTH arms would make the
# comparison rigged - new-prompt gpt-oss against old-prompt Sonnet. So the rule
# under test is gpt-oss-ONLY, which is also how it would ship (there is
# precedent: GENERATOR_REASONING_EFFORT is already applied only to gpt-oss).
VARIANTS = {"gpt-oss-complete": ("groq:openai/gpt-oss-120b", "completeness")}
JUDGE = "phi4"
SEED = 20260912       # fixed, so the A/B assignment is reproducible


def build_contexts() -> list:
    """Retrieve once per question. Free, local, deterministic."""
    from src.rag import _format_context, retrieve
    items = json.loads(QUESTIONS.read_text())
    out = []
    for i, it in enumerate(items, 1):
        res, _ = retrieve(it["retrieval_query"], [])
        out.append({"question": it["question"],
                    "retrieval_query": it["retrieval_query"],
                    "rating": it["rating"],
                    "context": _format_context(res)})
        if i % 10 == 0:
            print(f"  context {i}/{len(items)}", flush=True)
    CTX_PATH.write_text(json.dumps(out, indent=1))
    print(f"wrote {CTX_PATH} ({len(out)} contexts)")
    return out


def generate() -> None:
    """Both arms on the identical contexts. Saves after every answer so a
    quota exhaustion or a kill costs one answer, not the whole run."""
    from eval.generator_bakeoff import _generate
    from src.prompts import SYSTEM_PROMPT
    from src import llm

    ctxs = json.loads(CTX_PATH.read_text()) if CTX_PATH.exists() else build_contexts()
    rows = json.loads(GEN_PATH.read_text()) if GEN_PATH.exists() else \
        [{"i": i, **{k: c[k] for k in ("question", "retrieval_query", "rating", "context")}}
         for i, c in enumerate(ctxs)]

    targets = dict(ARMS)
    if len(sys.argv) > 2:                      # e.g. `generate gpt-oss-complete`
        want = sys.argv[2]
        targets = {want: VARIANTS[want][0]} if want in VARIANTS else {want: ARMS[want]}
    for arm, spec in targets.items():
        suffix = ""
        if arm in VARIANTS and VARIANTS[arm][1] == "completeness":
            from src.prompts import _COMPLETENESS_RULE
            suffix = _COMPLETENESS_RULE
        todo = [r for r in rows if not r.get(arm) and not r.get(arm + "_error")]
        print(f"\n{arm} ({spec}): {len(todo)} to generate", flush=True)
        for n, r in enumerate(todo, 1):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT + suffix},
                    {"role": "user",
                     "content": f"Context:\n{r['context']}\n\nQuestion: {r['question']}"}]
            t0 = time.time()
            try:
                r[arm] = _generate(spec, msgs)
                u = dict(llm.LAST_USAGE or {})
                r[arm + "_tokens"] = [u.get("input_tokens"), u.get("output_tokens")]
                r[arm + "_seconds"] = round(time.time() - t0, 2)
            except Exception as exc:  # noqa: BLE001
                # Record and CONTINUE rather than abandoning the arm. Groq's
                # free tier refuses any single request over 8,000 tokens with a
                # 413, and 2 of these 40 policy contexts exceed that - a real
                # production limit, not a harness problem, and one oversized
                # question must not cost the other 38.
                r[arm + "_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                print(f"  [{n}/{len(todo)}] FAILED: {r[arm + '_error'][:110]}", flush=True)
                GEN_PATH.write_text(json.dumps(rows, indent=1))
                continue
            GEN_PATH.write_text(json.dumps(rows, indent=1))
            if n % 5 == 0:
                print(f"  [{n}/{len(todo)}] {r[arm + '_seconds']}s", flush=True)
    print(f"\nwrote {GEN_PATH}")
    _cost(rows)


def _cost(rows: list) -> None:
    """What this actually spent, from the recorded token counts - not the
    estimate that justified running it."""
    from src.telemetry import cost_usd
    total = 0.0
    for arm, spec in ARMS.items():
        model = spec.split(":", 1)[1]
        c = sum(cost_usd(model, *(r.get(arm + "_tokens") or [None, None])) or 0 for r in rows)
        total += c
        print(f"  {arm:8} ${c:.4f}")
    print(f"  {'total':8} ${total:.4f}  (~£{total * 0.79:.2f})")


_JUDGE_PROMPT = """You are comparing two answers to a university policy question.
Both were written from the SAME retrieved policy excerpts, shown below.

Judge ONLY on how well each answer answers the question using those excerpts:
completeness (does it cover the relevant conditions, exceptions and figures),
accuracy against the excerpts, and usefulness to a university staff member.
Do NOT reward length for its own sake. Do NOT reward confident tone.
An answer that states something the excerpts do not support is WORSE, not better.

Reply with JSON only: {{"winner": "A" | "B" | "tie", "why": "<one short sentence>"}}

EXCERPTS:
{context}

QUESTION: {question}

ANSWER A:
{a}

ANSWER B:
{b}
"""


def _pair() -> tuple:
    """Which two arms to compare. Defaults to the original head-to-head."""
    if len(sys.argv) > 3:
        return sys.argv[2], sys.argv[3]
    return "gpt-oss", "sonnet"


def _judge_path(a: str, b: str) -> Path:
    if (a, b) == ("gpt-oss", "sonnet"):
        return JUDGE_PATH            # the original run keeps its filename
    return Path(f"eval/policy_arena_judged_{a}_vs_{b}.json")


def judge() -> None:
    """Blind pairwise, both orders. Local phi4 - stop production first."""
    import ollama
    A, B = _pair()
    rows = json.loads(GEN_PATH.read_text())
    JP = _judge_path(A, B)
    done = json.loads(JP.read_text()) if JP.exists() else {}
    rnd = random.Random(SEED)
    # The A/B assignment per question is fixed by SEED so a resumed run does
    # not silently re-randomise and mix two different experiments.
    for r in rows:
        r["_first"] = "gpt-oss" if rnd.random() < 0.5 else "sonnet"   # remapped below

    todo = [r for r in rows if r.get(A) and r.get(B) and str(r["i"]) not in done]
    print(f"judging {A} vs {B}: {len(todo)} pair(s) x 2 orders with {JUDGE}", flush=True)
    for n, r in enumerate(todo, 1):
        # Same seeded A/B assignment as the original run, remapped to this pair,
        # so the two comparisons are not accidentally different experiments.
        first = A if r["_first"] == "gpt-oss" else B
        second = B if first == A else A
        verdicts = {}
        for order, (x, y) in (("fwd", (first, second)), ("rev", (second, first))):
            prompt = _JUDGE_PROMPT.format(context=r["context"][:12000],
                                          question=r["question"],
                                          a=r[x], b=r[y])
            try:
                out = ollama.chat(model=JUDGE, format="json",
                                  messages=[{"role": "user", "content": prompt}],
                                  options={"temperature": 0, "seed": 42, "num_ctx": 16384})
                v = json.loads(out["message"]["content"])
                pick = str(v.get("winner", "tie")).strip().upper()
                verdicts[order] = {"A": x, "B": y, "winner":
                                   x if pick == "A" else (y if pick == "B" else "tie"),
                                   "why": str(v.get("why", ""))[:200]}
            except Exception as exc:  # noqa: BLE001
                verdicts[order] = {"A": x, "B": y, "winner": None, "why": repr(exc)[:120]}
        done[str(r["i"])] = verdicts
        JP.write_text(json.dumps(done, indent=1))
        if n % 5 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
    print(f"wrote {JP}")


def report() -> None:
    A, B = _pair()
    rows = {r["i"]: r for r in json.loads(GEN_PATH.read_text())}
    judged = json.loads(_judge_path(A, B).read_text())
    wins = {A: 0, B: 0}
    ties = flipped = 0
    for k, v in judged.items():
        f, rv = v.get("fwd", {}), v.get("rev", {})
        if not f.get("winner") or not rv.get("winner"):
            continue
        # Agreement across the two orders is the whole point: a pair that flips
        # when swapped tells you the judge could not distinguish them, not that
        # the answer shown first was better.
        if f["winner"] == rv["winner"] and f["winner"] != "tie":
            wins[f["winner"]] += 1
        elif f["winner"] == "tie" and rv["winner"] == "tie":
            ties += 1
        else:
            flipped += 1
    n = wins[A] + wins[B] + ties + flipped
    print(f"\n=== {A} vs {B} — 40 policy questions from real traffic, judged by {JUDGE} ===")
    print(f"pairs with a verdict in both orders: {n}\n")
    print(f"  {A} wins (both orders agree) : {wins[A]}")
    print(f"  {B} wins (both orders agree) : {wins[B]}")
    print(f"  genuine ties      (tie in both)       : {ties}")
    print(f"  order-dependent   (flipped on swap)   : {flipped}  <- judge could not tell them apart")
    decisive = wins[A] + wins[B]
    if decisive:
        print(f"\n  of {decisive} decisive pairs: {A} {wins[A]/decisive*100:.0f}%"
              f" / {B} {wins[B]/decisive*100:.0f}%")
    print(f"\n  position-bias check: {flipped} of {n} pairs ({flipped/n*100:.0f}%) depended on order.")
    for arm in (A, B):
        lens = [len(rows[int(k)][arm]) for k in judged if rows.get(int(k), {}).get(arm)]
        secs = [rows[int(k)].get(arm + "_seconds") for k in judged
                if rows.get(int(k), {}).get(arm + "_seconds")]
        if lens:
            print(f"  {arm:8} median {statistics.median(lens):.0f} chars, "
                  f"{statistics.median(secs):.1f}s")
    _cost(list(rows.values()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"contexts": build_contexts, "generate": generate,
     "judge": judge, "report": report}[cmd]()
