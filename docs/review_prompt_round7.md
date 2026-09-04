# External review — Round 7

I'm building a **conversational RAG assistant over University of Essex policy and rules-of-assessment (RoA) documents**. You and other LLMs have reviewed it over six rounds. This covers everything since round 6 — roughly 110 commits, and the single densest stretch of the project.

Please give me: (a) a code review, (b) a methodology critique, (c) concrete next steps. **Be blunt.** Several sections below are me reporting that my own hypotheses were falsified; I'd rather that continue than be flattered.

**Repo:** https://github.com/MKampouridis/RAG_Policies — `eval/report.md` is the ledger and holds the full record, including retractions. `CLAUDE.md` states the working discipline.

> **Numbering note:** the ledger's internal "Rounds" and these review rounds are different counters. This review round 7 covers **ledger Rounds 7–10**.

---

## What changed most since round 6: production is no longer fully local

Round 6 told you this ran entirely locally on a 16GB M1 Pro. **That is no longer true of production**, and it is the biggest single change:

- **Answer generator:** `claude-sonnet-5` (was `gemma3:12b`)
- **Query contextualizer:** `claude-haiku-4-5` (was `qwen2.5:7b-instruct`; Sonnet was tried and was not better — a null)
- **Retrieval:** unchanged and fully local — Chroma (`nomic-embed-text`) + BM25 fused with RRF, then ColBERT reranking (`GTE-ModernColBERT`, MaxSim), over-fetch pool → top-6
- **Falls back to local automatically** when no API key is present

**The eval server stays local and deterministic** (`RAG_DETERMINISTIC=1`). This split is deliberate: cloud generation cannot be temperature-pinned, and every ledger baseline was measured locally. `eval_session.sh` now *enforces* the topology — it stops production, starts a local server on :8001, and refuses to run if anything already holds that port. That refusal exists because a leftover cloud-configured server once answered the health check and silently served a "local" run.

**Corpus:** 1,172 kept documents / 246 current after archive-filtering, 21,709 chunks in the production collection.

**Eval sets:** the original 40×2 main set and 40-question holdout, plus **four new sets built this round** — set 3 (23 PGT/PGR incl. abstention), set 4 (10 corpus-spanning), set 5 (10 multi-entity), set 6 (8 partner-institution). New metrics: **span coverage@6**, **department coverage**, **evidence-sufficient@6**.

---

## 1. A real-user feedback loop now drives the work (34 ratings)

The largest change in *how* I choose what to build. Real thumbs-down ratings in `data/feedback.jsonl`, replayed against live retrieval, generated most of this round's shipped work. Four mechanisms below came from actual complaints, not from my speculation. **I think this is the single best methodological change of the round** and I'd like it critiqued — particularly the risk of overfitting to one user (n=1; it's my own use plus a small number of colleagues).

## 2. A guard I shipped cost 8.8 points, and how it was caught

`_has_extraneous_family` was enabled on the strength of **two hand-checked cases**. A full eval later showed **−8.8 points of follow-up hit@6**. Now disabled, and it became the project's governing rule: new retrieval/prompt mechanisms default to `False` and ship only after an eval justifies it. Falsified mechanisms stay in the code, off, with the falsification recorded.

Related, found later: the same guard's *faithfulness* check discards good rewrites of elliptical follow-ups. Production rate turned out to be ~2%, so it was left alone — a ledger problem, not a production one.

## 3. Mechanisms shipped, each with the measurement

| mechanism | evidence | status |
|---|---|---|
| **Stale-edition alias fix** (3 rename-split families) | stale editions were live in retrieval; verified neutral on hit@6 | **on** |
| **Multi-entity retrieval** (per-entity budget) | real failure: department coverage 1/6 → 5/6; set 5: **60.4% → 100%**; 160-turn control: no collateral damage | **on** |
| **Faculty → department expansion** | faculty questions now match explicitly-named-department questions | **on** |
| **Partner exclusion when unnamed** | **4/17 real complaints → 0/17**, hit@6 +1, 0 lost | **on** (see §5) |
| **Adjacent-chunk expansion** (rank-1 only, max +2) | the real case 0/7 → 5/7 and a false denial removed; A/B +0.05 (below noise) | **on, narrow** |
| **`USER_FACING_LANGUAGE`** | plumbing-leaking answers 4/4 → 2/4 | **on, partial** |
| **Detail level (concise/detailed)** | concise loses 0 entities, 0 list items, −27% length | **on**, default unchanged |

**Adjacent-chunk expansion is the one I most want challenged.** I measured *before* building: of 160 turns, 79% already had the answer chunk, **5% had it immediately adjacent**, 11% were "far", 4% had none. So the ceiling was ~8 turns in 160. The A/B delta (+0.05) is **below my own noise floor** and I shipped anyway, on the argument that a concentrated benefit can't show up in a 20-question mean and that no harm was detectable. I narrowed the blast radius (top-3 expansion touched 97% of turns; rank-1 touches 81%) precisely because that 97%-touch-for-5%-benefit ratio is what `_has_extraneous_family` had. **Is that reasoning sound, or am I rationalising an unmeasurable change?**

## 4. Falsified this round — please don't re-suggest these

- **Duplicate-chunk crowding** as a failure cause — the control *inverted* it.
- **Chunk order in the prompt** — my own hypothesis, falsified **twice**, on both context shapes. I ran the spoiler arm first (`reversed`, worst chunk first); it killed the hypothesis outright. A `rank` vs `grouped` comparison would have produced a small difference and I'd have read it as confirmation.
- **Structure-aware chunking** — v1 made list-splitting *worse* (30% → 50%); v2 (extend-forward) reached 24% with no retrieval gain. Not adopted.
- **Bigger chunks** — measured against a parallel index: list-splitting 30% → 8% with **zero retrieval gain**. Not adopted.
- **`FETCH_POOL_MULTIPLIER=4`** — costs quality, saves ~1s of 42s.
- **Haiku 4.5 as generator** — null; kept Sonnet 5.
- **Cloud contextualizer (Sonnet)** — null; the query rewriter was not the bottleneck. Haiku adopted for cost, not quality.
- **D3 clarify-on-underspecified** — **declined**, not deferred. Round 6 called it the last lever; I now think the offline metric structurally can't evaluate it and the live signal isn't there.

Still falsified from earlier rounds (full list in round 6): cross-encoder rerankers, ColBERT first-stage, alternate embedders/ensembles, SPLADE, weighted fusion, facet filtering, doc routing, HyDE, CRAG, multi-hop decomposition, inline citations, structured parameter enumeration, retrieval-confidence abstention gates.

## 5. A trade-off I accepted with the cost on the record

Set 6 was built specifically to test partner-institution questions in the direction that had never been measured — **where a partner document IS the gold answer**. It found that `PARTNER_EXCLUDE_WHEN_UNNAMED` **over-corrects**:

| group | exclusion OFF | exclusion ON |
|---|---|---|
| partner NAMED | 4/4 | 4/4 |
| **partner UNNAMED** (names a partner *programme*) | 2/2 | **0/2** |
| home control | 2/2 | 2/2 |

This was invisible before: 9 incidental partner-gold questions in older sets gave +1 with 0 losses, which *read* as safe but was simply unmeasured in this direction. **Decision: leave as is.** The intended audience is Essex staff asking about Essex programmes, and their complaint is fixed. A partner-college administrator would hit the inverse case immediately. Both softening options add permanent maintenance. **Tell me if you think this call is wrong.**

## 6. Methodology: the noise floor, and a retraction

**The noise floor.** Two runs of the **same configuration** on a 10-question set scored **4.05 and 3.85**. So on a 10-question / 20-turn set with a cloud generator, **any delta below ~0.20 is uninterpretable**. Three findings recorded that same day sat at or below it — one reported as an effect, one whose direction *reversed* on retest. The rule now: before believing a small cloud delta, **run one arm twice**. The repeat costs exactly what the comparison cost.

**A retraction.** I judged Sonnet-generated answers with `claude-sonnet-5`. Same-family self-preference inflated the result by **+0.42**; a separate case swung **24 points**. Re-judged with `phi4` (neutral, cross-family) and retracted the headline 4.62 → 4.20. A second comparison had **mixed judges across arms** — re-judged both sides, and the true gap was **+0.22, not +0.52**. The judge alone moves the threshold metric by ~9 points.

**Other corrections I logged rather than buried:** a contiguity-grouping justification whose causal claim was falsified; a Round 8f claim that the old answer *did* name the departments I'd said it missed; a "useful-answer rate is the wrong headline" correction.

**A metric honesty note:** hit@6 compares document URLs, so it cannot see "right document, wrong chunk" or "had the facts, didn't use them" — measured at **8.7 points** on the main set. Two mechanisms this round (adjacent chunks, detail level) are **structurally invisible** to it.

## 7. Corpus re-ingest, a weekly watcher, and a deliberate re-baseline

I built `check_new_documents.py` — a **detect-only** weekly crawler (Mondays 11:00 via launchd, moved from Sunday 23:00 after the overnight notification was found to be silently suppressed). It never ingests: auto-ingesting would silently change what every answer is based on and silently invalidate the ledger, with nobody deciding to.

Its first run found **5 new and ~20 changed documents** — the system had been answering from superseded text, including final-year UG rules. Ingested deliberately, checks in order: 10 indexed, 0 errors, 0 new rename-split aliases, 0 missing content.

**Retrieval replay: 123/160 → 122/160 (net −1).** The lost turn is a *general* framework question now outranked by new department-specific variations files. Accepted — serving superseded rules is a failure at the tool's actual purpose — but recorded as a **watch item**, because those files are large and generically worded and may be broadly "attractive" to UG queries.

**The ledger is now re-baselined.** Rounds 1–8 numbers remain valid as a record and comparable to each other, but are **no longer comparable to anything measured from here on**.

## 8. The lesson evals could not have caught

Production answers said *"the context you've provided across both turns"* and *"if you have excerpts, please share them"*. The user supplied nothing — the retriever did. So this reads as either a mistake or a request they cannot act on.

**Every metric in the ledger scored these answers identically before and after the fix, because the facts were right both times.** No amount of measurement would have surfaced it; only reading the output as a user would. The fix (`USER_FACING_LANGUAGE`) cut plumbing-leaking answers from 4/4 to **2/4, not zero** — "the excerpts I can see" survives an explicit instruction not to say it. I report the residual rather than the headline.

I also rebuilt the UI substantially (Essex branding, mobile drawer, source modal, staleness marker that flags stored answers whose cited policy has since changed, generated conversation titles).

## 9. Latency — measured, largely unaddressed

Responses still feel slow, so I instrumented per-stage timing (`RAG_TIMING=1` → `data/latency.jsonl`, 473 records over 118 real requests). **The headline is that there are two very different systems depending on whether the server is warm:**

| | n | median total | retrieve | generate | contextualize |
|---|---|---|---|---|---|
| **warm** (request within 10 min of the last) | 101 | **9.8s** | 0.5–3.4s | 6.8s | ~0.5s |
| **cold** (>10 min idle) | 17 | **28.2s** | **22.6s** | 6.8s | ~0.5s |

p90 total is 28.2s; worst observed 84s.

**When warm, the cloud generator is the cost** (6.8s of 9.8s) and retrieval is nearly free. **When cold, retrieval costs ~21 extra seconds.** I reproduced this outside the server — 20.87s first call vs 0.50s third call in a fresh process — and then isolated it by pre-warming each loader in turn:

| component | cold cost |
|---|---|
| torch / ColBERT **first-encode warmup** | **~8.0s** |
| ColBERT model load (`pylate`) | ~4.1s |
| BM25 index build (`rank_bm25`, 21.7k chunks) | ~2.5s |
| module import | ~2.5s |
| Chroma first dense query | ~0.3s |
| *(≈17s of the ~21s; remainder is run-to-run variance)* | |

**Why this matters more than the raw numbers suggest:** everything here is lazily initialised on first use, and the server is under launchd `KeepAlive`, so every restart resets it. Only 14% of my *logged* requests were cold — but I am a heavy user. **An alpha tester asking two questions a day is cold essentially every time**, so the experience I've measured as "9.8s median" would be ~28s for them.

**Since writing this I built and measured the fix** — one throwaway retrieval in a daemon thread at startup: **first query 16.97s → 3.40s** (6 fresh processes, arms alternated, spread 0.4s), server still serving in 0.4ms while it warms, results identical. So the cold case is largely solved and **the warm 9.8s — of which 6.8s is one cloud generation call — is now the whole problem.**

**There is a companion prompt dedicated to latency** (`docs/review_prompt_round7_latency.md`) with the full breakdown, what's already been ruled out, and specific questions. If latency is your area, use that one.

## 10. Ops failures I'd like judged

- **Two data-loss incidents.** "Clear all history" deleted every conversation; its only feedback was a muted line, so it read as doing nothing and was pressed repeatedly. Recovered by carving freed SQLite pages (DELETE unlinks rows without zeroing them). Now: OK/Cancel confirm, progress, nightly `sqlite3 .backup` keeping 14. **The second incident's cause is still unknown, and I state it as unknown.**
- My first recovery filter kept **1801 of 1825** conversations — too permissive, because question-*generation* scripts also create conversations through the same API. Only "the user rated it" was a reliable human signal.
- Mistakes that cost real time: `git ls-files` exits 0 on no match (fooled me **twice**); `setsid` doesn't exist on macOS (I claimed a crawler had relaunched when it had died instantly); a wrong reject-log filename led me to conclude a log was empty.

---

## What I want from you

1. **Code review.** Correctness, fragility, anything that bites at scale. **Verify against the actual code** — several of my own confident claims this round were wrong and cheap to check.

2. **Is shipping below the noise floor defensible?** §3's adjacent-chunk expansion and several small decisions rest on samples that cannot resolve their own effect size. I argue "measured the ceiling first, narrowed the blast radius, confirmed no harm." **Is that a principled rule or a rationalisation?** If it's the latter, what's the alternative that doesn't mean never shipping a concentrated fix?

3. **My eval sets are 8–10 questions and my noise floor is ±0.20.** That's a bad ratio and I know it. Given a fixed compute budget, what's the highest-value fix — bigger sets, more repeats per arm, paired/bootstrap statistics, or moving evaluation back to a deterministic local generator and accepting it doesn't match production?

4. **The 11% "far" chunk misses.** Measured, unaddressed, deliberately deferred because everything I know how to try is on the falsified list. Is there a genuinely novel angle, given that list? Specifics only.

5. **Production is cloud, evals are local.** Every ledger baseline is measured on a configuration production doesn't run. I think this is the least-bad option (cloud can't be temperature-pinned). **Is it? What am I failing to detect because of it?**

6. **Is the feedback loop (n≈1 user, 34 ratings) a sound way to choose work, or am I overfitting to myself?** Four shipped mechanisms came from it.

7. **The partner trade-off in §5** — right call or wrong?

8. **Latency (§9).** The cold case is now fixed; the warm 9.8s is not, and 6.8s of it is one cloud generation call. Full treatment and specific questions are in the companion prompt `docs/review_prompt_round7_latency.md` — answer there if this is your area.

9. **Single highest-expected-value thing left, and what to STOP.** Round 6 asked this and the answer was "clarification UX"; I've since declined it. If the honest answer is "this is done, stop optimising and get it in front of users," say so — the real blockers now are hosting and per-user separation, not retrieval.

Please be specific and critical.
