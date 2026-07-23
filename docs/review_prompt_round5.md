# External review — Round 5

I'm building a **conversational RAG assistant over University of Essex policy and rules-of-assessment (RoA) documents**, running **entirely locally** (no paid APIs required). I've had you and other LLMs review it over several rounds; this is round 5. Please **read the code on GitHub** and give me: (a) a code review, (b) methodology critique, (c) concrete suggestions for what to try next — and be blunt about whether I've hit the ceiling.

**Repo:** https://github.com/MKampouridis/RAG_Policies

## Stack
- **Retrieval:** Chroma (dense, `nomic-embed-text`) + BM25 (`rank_bm25`) fused with **RRF**, then **ColBERT reranking** (`GTE-ModernColBERT` via pylate, MaxSim). Over-fetch pool → rerank top-6.
- **Generation:** local Ollama. Query **contextualizer** = `qwen2.5:7b-instruct`; **answer generator** = `qwen2.5:14b-instruct` (just upgraded from 7b — see below); **judge** (eval only) = `qwen2.5:14b-instruct`.
- **Corpus:** ~12.6k chunks over ~hundreds of Essex PDFs. The hard problem: many documents are **near-duplicate "siblings"** (same rules-of-assessment boilerplate, differing only by programme / degree-length / cohort-year / partner-institution). Retrieval failures are almost always "retrieved the WRONG sibling," not "off-topic."
- **Eval:** 40 questions × 2 turns (primary + follow-up) = 80 turns, deterministic (`RAG_DETERMINISTIC=1`, temp 0). Metrics: strict hit@6, **evidence-sufficient@6** (headline; any retrieved doc containing ≥half the gold keyphrases), answer_score (1–5, 14B judge), keyphrase_coverage, and a **groundedness / faithfulness-to-context** eval (14B judge checks every factual claim is supported by the retrieved context — this is the hallucination metric).

## What I did since your last review, and what I found

**1. Generator capability is the lever for hallucination — proven.** I swapped ONLY the answer generator (retrieval + judge held identical) across three tiers and measured groundedness on the RoA questions:

| RoA groundedness | 7B (old) | 14B (local, now production) | cloud gpt-oss-120B |
|---|---|---|---|
| on retrieval-HIT turns | 72.4% | 81.5% | **92.9%** |
| on retrieval-MISS turns | 54.5% | 46.2% | 41.7% |
| overall RoA | 67.5% | 70.0% | 77.5% |

Hit-turn groundedness is **monotonic in generator size**. This proves the long-standing "strict-vs-evidence gap" (a sufficient doc is retrieved but the model doesn't surface its figures) is a **generator-capability problem**, not retrieval — when the right doc is retrieved, a stronger model reports the figure faithfully instead of fabricating it. I switched production to the local 14B (+9pt, free, unlimited). I also built an env-gated cloud-generator path (Groq/Gemini, OpenAI-compatible) but Groq free tier caps at 100–200K tokens/day so it can't sustain a full eval — it only established the ceiling.

**2. But stronger models are WORSE on retrieval misses** (54.5→46.2→41.7): handed the wrong sibling, a bigger model writes a fluent, confident answer from it ("confidently wrong"), which the weaker 7B more often hedged. So miss-turn hallucination is a *separate* problem from generation.

**3. Abstention gate — FALSIFIED.** I tried to detect "retrieval probably missed → abstain/hedge." Diagnostic over 80 turns: **no retrieval signal separates hit from miss** at usable precision. Absolute ColBERT confidence: ZERO signal (misses score *marginally higher* — the wrong sibling is just as on-topic). Top1–top2 margin: zero signal. Distinct-document-families-in-top-6: the *only* informative signal but weak (≥6 families → 0.40 precision / 0.62 recall). Root cause = the core problem: **a miss retrieves a near-duplicate sibling that is high-confidence and on-topic**, so confidence can't flag it (same reason an earlier CRAG-style "does context support the answer?" gate over-fired and was rejected).

**4. Clarification UX (D3) — built, off by default.** For under-specified questions ("what's the minimum average to pass with Merit?" — which programme?), ask instead of guessing. Key finding: **listing candidate programmes is logically impossible** — on a miss the correct doc is *by definition* absent from the retrieved pool, so any offered options are all wrong (verified). So the only honest form is a **generic** "which programme did you mean?" with no options. Validated the payoff: user names the programme → the follow-up contextualizer rewrites the query → retrieval goes from total miss to a perfect hit. Trigger (fragmented pool + query names no degree/award) is only ~0.45 precision, so it ships off and must be judged on real conversations (the hit@6 eval scores a clarifying question as a "miss" by design).

**5. Data hygiene:** removed a byte-identical duplicate document that was masquerading as a current-year edition (added a durable exclusion so re-crawls don't re-add it); added degree-length digit↔word tokenizer synonyms (3yr↔three etc.) so glued filename identity tokens become lexically matchable (RoA hit@6 70→72.5%, evidence-sufficient@6 87.5→90%).

**6. Product:** added a **user feedback loop** — thumbs up/down under each answer; thumbs-down expands failure tags mapped to my failure taxonomy (wrong programme/document, wrong-or-made-up figure, outdated/wrong-year, didn't-answer). Stored as JSONL with full retrieval context so each rating can be *replayed and auto-diagnosed*; `feedback_report.py` routes tags → levers. Deployed privately over Tailscale.

## Already exhaustively tried and FALSIFIED — please don't re-suggest these
Cross-encoder rerankers, ColBERT as first-stage retriever, alternate embedding models (mxbai, bge-m3), embedding ensembles, SPLADE, weighted score fusion (vs RRF), hard + soft facet filtering (degree-length/award-type), pseudo-query expansion (HyDE-style), CRAG verification, selective multi-hop query decomposition, macro document-routing, identity-enriched reranking, header-weight boosting, wider rerank pools, inline per-claim citations (regressed groundedness — model fabricates provenance), verbatim-figure prompt rules. ~20 architecture experiments, all net-neutral or negative. The only net-positive *retrieval* gains in the whole project came from **data/lexical hygiene**, not architecture.

## What I want from you
1. **Code review** — anything wrong, fragile, or non-idiomatic; correctness bugs; things that will bite at scale.
2. **The core unsolved problem is sibling disambiguation** (retrieving the right near-duplicate among many). Given everything above is falsified, is there a *genuinely novel* angle I'm missing — e.g., structured/metadata-first retrieval, learned query→document-facet classification, a document-graph, constrained decoding, anything? Or is this fundamentally a data-modeling problem (I should represent programme/cohort as hard facets and route), not a retrieval-ranking problem?
3. **Generator next step.** The finding says capability drives grounding. For a standing-production *local* generator on a 16GB M1 Pro (14B is already tight), what's the best value — a larger quantized model, a different 14B-class model, speculative decoding, a distilled reranking-aware model? Any specific models worth testing?
4. **Is the abstention gate really dead**, or is there a signal I haven't tried (e.g., a small learned classifier over retrieval features, agreement between dense/BM25/ColBERT rankings, answer self-consistency across samples)?
5. **Methodology / eval soundness** — is the 40×2 eval representative enough? Is judging with the same 14B that now also generates a problem (self-judging bias)? Should I get an independent judge, and how without a paid API? Should I generate a different/larger question set, and how (I have a sibling-discriminating set already)?
6. **The feedback loop is now my main real-world signal.** How should I best turn accumulating thumbs-down + tags + replayed retrieval into the next experiment — any pitfalls in learning from this data?
7. **Have I hit the ceiling?** Honestly. What, if anything, is the single highest-expected-value thing left to try — and what should I stop pursuing?

Please be specific and critical. Verify claims against the actual code rather than assuming.
