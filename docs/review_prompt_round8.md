# External review — Round 8

I'm building a **conversational RAG assistant over University of Essex policy and rules-of-assessment documents**. You and other LLMs reviewed it at round 7; this covers what happened since, which is ~40 ledger rounds in four days.

Round 7's advice was acted on almost completely — **91 of 94 recommendations resolved**, each with the measurement or the reason for declining. But the most important thing that happened wasn't on your list, and it changes how much of the earlier work should be believed. Please give me (a) a code review, (b) a methodology critique, and (c) a judgement on whether the central new finding is worth publishing. **Be blunt.** Several sections below are me reporting that I was wrong.

**Repo:** https://github.com/MKampouridis/RAG_Policies — `eval/report.md` Rounds 13–51 are the full record. `docs/review_round7_actions.md` tracks all 94 items with their disposition.

---

## The headline: the LLM judge does not measure what we assumed

Round 7 reviewers said "human-calibrate the judge". I did: **30 stored answers, scored blind by a human, same rubric and same inputs the judge gets.**

| | |
|---|---|
| exact agreement | **12/30 (40%)** |
| within 1 point | 21/30 (70%) |
| mean difference | −0.57 (judge scores lower) |
| **rank correlation** | **+0.46** |

The rank correlation is the number that matters: A/B work needs the judge to *order* answers like a human, not to match absolute values. At +0.46 it barely does.

**The diagnosis is the interesting part.** Agreement is decent on bad answers and poor on good ones — of 17 answers the human called perfect, the judge scored three of them 1 or 2. The human's notes explain why:

> *"The reference text was wrong but the answer from the system was right!!"* (human 5, judge 1)
> *"Answer seems to be correct, although it doesn't match the referenced text"* (human 5, judge 2)

**The judge scores agreement-with-reference, not correctness** — which is exactly what the rubric asks for, and the wrong question whenever the reference is unreliable.

### Then a second experiment, which produced a method

If bad references drive disagreement, how common are they? Two samples:

| sample | references wrong |
|---|---|
| 9 items **selected for maximum human/judge disagreement** | **7 (78%)** |
| 15 items drawn **at random** | **0 (0%)** |

0/15 rules out a rate above ~20% (rule of three) but is consistent with 5–10%, so the claim is "not common", not "zero".

**The contrast is the finding: human/judge disagreement is a cheap detector for corrupt test items.** Nine selected items found eight defects. A keyword audit of all 151 items found two — and *zero* of the ones the human caught, because keyphrase presence cannot detect a reference that quotes its source faithfully while describing a superseded rule.

**Question 1 for you: is that publishable, and where?** My own assessment is that it's an evaluation-validity paper rather than a negative-results paper, and that it needs multiple annotators and a public dataset to be credible. Tell me if I'm overrating it.

## What this costs the earlier work

Every judge-scored comparison in this ledger is noisier than recorded — and randomly noisier, which does *not* cancel between two arms the way a constant bias would. Eight defective items sit in the main 40-question set, the most-replayed set in the project.

Unaffected: hit@6, span/department/keyphrase coverage, and every deterministic string check. Those never involve the judge.

## A three-week-old defect nobody could see

Investigating a *latency* flag turned into a correctness fix. The ColBERT reranker's embedding cache was built 2026-07-21; the corpus was re-ingested 2026-08-11. The index held 20,477 chunks against Chroma's 21,709, so **the reranker was scoring some chunks by superseded wording**.

| | cache ON | cache OFF |
|---|---|---|
| hit@6 | 121/160 | **126/160** |
| warm retrieval | ~5.0s | **1.45s** |
| server RSS | ~5.0GB | **2.12GB** |

Faster, lighter *and* more accurate. `run_ingest.py` updates Chroma and leaves the ColBERT index alone, and nothing detected the drift. It also explained an anomaly I had recorded and misdiagnosed a day earlier.

**Question 2: what else in a RAG stack goes stale silently like this?** I now have a drift check for this one instance. I'd like the general class.

## Retrieval is done, and I can now show it

| | |
|---|---|
| measured hit@6 (main set) | **86.2%** |
| achievable ceiling for this corpus+metric | **84.3%** |

Retrieval scores *above* the exchangeability ceiling. Of 11 residual misses: 5 have gold keyphrases present in no current document, 2 are gold-multiplicity artifacts, **only 4 are reachable targets**.

Separately, classifying all 39 document-level misses by sweeping the fetch pool *upward* (it had only ever been swept down): **59% ranking, 10% pool-size, 31% never enter the pool at any size tested**. That third could never have been fixed by any reranker — which retroactively explains why a decade of reranker experiments here all failed.

On that basis I **closed five retrieval proposals from round 7 without trying them** (RM3, document-level rerank, family retriever, parent-child chunking, cross-reference extraction) — marked "not falsified, unjustified". **Question 3: is closing on a ceiling argument legitimate, or am I using it to avoid work?**

## Things I got wrong, in public

- **Claimed the reranker was demoting correct documents** on 5 shallow-pool cases. Inspected them: 1 was genuine, 1 was a title-page artifact where the reranker was *right*, 1 was partner exclusion working as designed. **Retracted.**
- **Claimed bad references are rare**, citing an audit I had *already documented as unable to detect that failure*. Tested it: caught 0 of 7. **Retracted.**
- **Said the batched-rerank optimisation would be a large win** on the assumption transformers batch well. Built it: **11%**, because ColBERT's cost is per-passage compute. Not adopted.
- **Shipped a blank page.** A `const` used 760 lines before its declaration threw at top level and aborted the whole script. Every check I'd run — CSS balanced, element present in HTML, correct documents on 217 cases — was true and none tested whether the page *executes*.

## The refactor, and what its safety nets missed

`rag.py` 2,237 → **1,270 lines** across six single-purpose modules, plus `preview.html` 1,667 → 156 (CSS/JS extracted), research routes separated from the product app, `/classic` removed. Pure moves, verified against a byte-exact fingerprint of 161 queries.

**It broke twice.** The second is instructive: a constant was swept into the wrong module by a slice that spanned it, and **every answer failed with 503**. The 161-query fingerprint passed. The 118-turn canary passed. Both exercise *retrieval*; the constant is used during *answer assembly*.

**Two independent safety nets, both green, on a system that could not answer a single question.** Only an end-to-end request caught it — the cheapest check available.

**Question 4: how would you have designed the verification differently?**

## Also shipped, each measured

Streaming (dead time 14.9s → 7.2s, then a steady-rate renderer because output arrives in 154-char blocks 614ms apart) · soft delete making the twice-unexplained data loss non-destructive · per-answer provenance (corpus version, code revision, models) · a strict Essex/Partner scope switch replacing a heuristic, measured on 217 cases where set 6's 8 questions had hidden that the control was *contradicting its own label* · name-based history separation · human error messages replacing raw API JSON · a production smoke set reporting **rates** not judge means, which immediately caught an intermittent (~1 in 13) plumbing leak.

## Deployment, measured

4 concurrent users 1.1×, 8 users 1.8× (the contended resource is reranking, not generation — generation is network wait). **1.32GB under 8-way load.** **No GPU needed** — reranking 30 passages is 0.27s on an Apple GPU and 0.29s on CPU. **$0.0197/question**, so ~$43/month for 10 users at 10 questions/day.

## What I want from you

1. **Is the judge/reference finding publishable, and where?** Be realistic about venue and about what evidence level would survive review.
2. **What else goes stale silently** in a RAG stack, as the ColBERT cache did?
3. **Is closing retrieval work on a ceiling argument legitimate?**
4. **How should the refactor verification have been designed** so that two green safety nets couldn't coexist with a totally broken system?
5. **Code review** of the six new modules — verify against the actual code; several of my confident claims this round were wrong and cheap to check.
6. **What is the single highest-value thing left?** The honest candidates are: fix 8 defective test items, run the 302-turn end-to-end baseline that has never been run, get 5–10 real users, or write the paper. If the answer is "stop building and go get users", say so plainly.

Please be specific and critical.
