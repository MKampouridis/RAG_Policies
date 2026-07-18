# Retrieval & Answer-Quality Evaluation

**Date:** 2026-07-16 (updated same day with the retrieval improvement round — see
"Retrieval improvement round" below for the current production configuration)
**Corpus:** 1,188 kept documents (88 policy, ~1,100 rules-of-assessment), 12,624 chunks
**Question set:** 40 questions (20 policy, 20 rules-of-assessment) + 40 follow-ups = 80 scored turns per pass

## Summary

The system was evaluated three times against the same 40-question set: once as originally
shipped, once after fixing two retrieval bugs found during evaluation, and once more after
swapping the embedding model to test whether a stronger model would help further.

**Bottom line:** two logic fixes (query contextualization for follow-ups, and preferring the
most recent academic year among near-duplicate documents) delivered a real, broad improvement.
Testing a stronger embedding model (`mxbai-embed-large`) was **not** a net win — it improved
policy retrieval but regressed rules-of-assessment retrieval below baseline, and RoA is most of
the corpus. **Recommendation followed: keep `nomic-embed-text`, keep both logic fixes.** This is
the configuration now running in production.

## Methodology

**Question generation.** 40 source documents were hand-selected from the ingestion manifest for
topical diversity (20 policy: whistleblowing, academic offences, freedom of speech, fitness to
practise, student engagement, etc.; 20 rules-of-assessment: UG framework/glossary, 3-year/4-year
honours rules, foundation year, integrated masters, PGT department-specific rules for CSEE, East
15, Social Work, Physiotherapy, Periodontology, HRM, MBA, credit framework, integrated PhD,
appeals procedure, and one partner-institution document). For each, the chat model (given the
document's cached text) drafted one specific factual question plus a natural in-conversation
follow-up, a ground-truth answer, and key phrases a correct answer should contain. A sample was
manually spot-checked for accuracy; one class of question was flagged as unreliable (see
Limitations).

**Scoring, per turn:**
- **Retrieval hit@6 / rank / MRR** — was the question's source document among the top-6 retrieved
  chunks, and at what rank (reciprocal rank = 1/rank, 0 if absent). Retrieval was measured via the
  *exact* code path the live app uses (`src/rag.py:retrieve()`), not a simplified stand-in — this
  matters because query contextualization and recency filtering change what actually gets
  retrieved for follow-ups, and an earlier version of this harness silently missed that (see
  Process notes).
- **Answer score (1-5)** — LLM-judged (using the same `qwen2.5:7b-instruct` model) against the
  ground-truth answer, on correctness/completeness.
- **Keyphrase coverage** — objective, non-LLM: fraction of the ground-truth key phrases that
  appear verbatim in the generated answer.

Each question was run as a fresh conversation via the live HTTP API (not a unit-test shortcut),
so results reflect real end-to-end behavior including conversation memory.

## Results

All figures are hit@6 / MRR / mean answer score (1-5) over 80 scored turns (40 questions × 2).

| | Baseline | + Logic fixes | + mxbai-embed-large |
|---|---|---|---|
| **Overall** | 50.0% / 0.35 / 3.64 | 56.3% / 0.43 / 3.66 | 51.3% / 0.44 / 3.76 |
| **Policy** | 77.5% / 0.63 / 3.78 | 77.5% / 0.71 / 3.78 | **82.5% / 0.74 / 3.95** |
| **Rules of assessment** | 22.5% / 0.06 / 3.50 | **35.0% / 0.15 / 3.55** | 20.0% / 0.13 / 3.58 |
| Primary questions only | 55.0% / 0.39 / 3.83 | 65.0% / 0.49 / 3.78 | 55.0% / 0.47 / 3.88 |
| Follow-up questions only | 45.0% / 0.30 / 3.45 | 47.5% / 0.37 / 3.55 | 47.5% / 0.40 / 3.65 |

## What was found and fixed

### Bug 1 (found before formal eval, fixed in baseline): missing embedding task prefixes

`nomic-embed-text` requires Nomic's `search_document:` / `search_query:` prefixes to produce
useful similarity scores. Without them, a direct query for "What does the Whistleblowing Policy
cover?" didn't retrieve the whistleblowing document even in the top 20 results out of 12,624
chunks. This was caught and fixed during initial smoke-testing, before the formal eval began, so
all three passes above already include this fix — it's not reflected in the baseline-vs-fixed
delta.

### Bug 2 (found during eval): follow-up questions retrieved with no conversation context

The original `rag.answer()` embedded only the current turn's raw text for retrieval. A follow-up
like *"What happens after a concern is disclosed under this policy?"* carries no signal about what
"this policy" refers to once stripped of conversation context, so it matched on generic vocabulary
against unrelated documents instead.

**Example (item 1, whistleblowing follow-up):**

| | Baseline | Fixed |
|---|---|---|
| Retrieved (top 2) | `tavistock-pg-dip-19.pdf`, `tavistock-pg-diploma-20.pdf` (unrelated PGT diplomas) | `policy-whistleblowing.pdf` (both) |
| Answer | "The provided context does not specify..." | Correct, detailed procedure |
| Judge score | 2 | 4 |

**Fix:** `src/rag.py:_contextualize_query()` rewrites the follow-up into a standalone question
using recent conversation history before embedding it (e.g. → *"What happens after a concern is
disclosed under **the University of Essex whistleblowing policy**?"*). The answering model still
receives the original question plus full history, since it can already resolve the reference
itself — only the retrieval step needed the rewrite.

**Evidence this was the right diagnosis, not a guess:** in the baseline data, follow-up turns
where retrieval happened to hit scored 4.0/5 on average — identical to primary questions. Turns
where it missed scored 2.9/5. The generation model was never the problem; it just weren't being
given the right context to work with.

### Bug 3 (found during eval): year-duplicate documents crowd out the current year

Rules-of-assessment documents are reissued almost unchanged every year across dozens of
departments (the crawl found copies going back to 2010-11). A fixed top-6 retrieval window
routinely filled up with several near-identical old-year chunks of the *same* document, pushing
the current year's version out entirely — this alone explains most of the 22.5% baseline hit
rate on RoA questions.

**Example (`procedures-fitness-to-practise.pdf`):**

| | Baseline top 3 | Fixed top 3 |
|---|---|---|
| 1 | `...-2019-20.pdf` | `...fitness-to-practise.pdf` (current) |
| 2 | `...-2020-21.pdf` | `professional-clinical-appearance-code-of-practice.pdf` |
| 3 | `...-2022-23.pdf` | `fitness-to-study.pdf` |

**Fix:** `src/rag.py:_prefer_most_recent_year()` over-fetches a larger candidate pool (24 instead
of 6), groups candidates by document family (filename with the year suffix stripped — chosen over
path-based grouping because Essex's "current" and "previous-years" archives use different folder
structures for the same document), and keeps only the most recent academic year within each
family before truncating to the top 6. This is skipped when the query itself mentions a specific
year (so "what were the rules in 2019-20" still works).

**Result:** RoA hit@6 rose from 22.5% → 35% (MRR 0.06 → 0.15, more than doubled) with this fix
alone.

## The embedding-model experiment

Given the residual gap, `mxbai-embed-large` (334M params, 1024-dim, MTEB-competitive) was pulled
and the full corpus re-embedded from cached text (~20 min, no re-crawl needed). The full
comparison (see Results table) shows a genuine trade-off, not a clean win:

- **Policy:** best of all three configurations on every metric (82.5% hit@6, 0.74 MRR, 3.95 score).
- **Rules of assessment:** *worse* than the fixed-nomic pass on hit@6 (20% vs 35%) and even
  slightly worse than the unfixed baseline (20% vs 22.5%).

**Working hypothesis:** `mxbai-embed-large` has a 512-token context window, versus
`nomic-embed-text`'s 2048+. Our chunks are ~300 words (~400-500 tokens) of dense, jargon-heavy RoA
text — course codes, mark thresholds, department names — that plausibly gets truncated, losing the
specific details that discriminate one RoA document from a near-identical one. Policy prose is
more front-loaded and narrative, so it's less exposed to this. This wasn't independently confirmed
(would require inspecting truncation directly) but is consistent with the direction and size of
the effect, and with `mxbai`'s embedding wins concentrating on questions with short, high-signal
answers.

**Decision: kept `nomic-embed-text`.** Rules-of-assessment documents are ~93% of the corpus
(1,100 of 1,188 kept documents) and equally core to the assistant's purpose, so a model that wins
on the smaller content type while losing on the larger one is not a net improvement. The
`mxbai-embed-large` collection (`policies_mxbai-embed-large`) was left in the Chroma store rather
than deleted, in case future work (e.g. a hybrid or per-doc-type embedding strategy) wants to
revisit it.

## Process notes worth recording

- **The eval harness itself had a bug that would have produced a misleading comparison.** The
  first "fixed" pass measured retrieval by calling the vector store directly with raw follow-up
  text, bypassing the very contextualization fix it was supposed to measure — so it would have
  shown no follow-up improvement despite the live app genuinely producing one. Caught by manually
  diff-ing `api_sources` (from the real endpoint) against the harness's own retrieval check before
  trusting the numbers. Fixed by refactoring `src/rag.py` to expose a single `retrieve()` function
  that both the app and the eval harness call, so the two can never drift apart again.
- **The `_prefer_most_recent_year` filter had a sentinel bug on first implementation**: documents
  with no `academic_year` (e.g. evergreen policies) computed an empty-string "best year," and
  `"" > ""` being false meant the family was never registered, causing a `KeyError` crash the first
  time it ran against live traffic. Fixed with an explicit membership check. Worth knowing if this
  pattern (using a dict `.get(key, default)` comparison to conditionally populate that same dict)
  shows up elsewhere.
- **A long-running ingestion job was killed 4 times** by what turned out to be a roughly 30-minute
  cap on this harness's own background-task tracking — not disk space or system sleep (both
  independently ruled out). Long jobs going forward should be launched fully detached
  (`nohup ... & disown`, reparented to `launchd`) rather than relying on the harness's
  `run_in_background` tracking.

## Limitations of this evaluation

- **LLM-as-judge uses the same model family that generates the answers** (`qwen2.5:7b-instruct`).
  This is a standard but real limitation — self-grading bias can inflate scores for the model's own
  stylistic tendencies. The objective keyphrase-coverage metric is included specifically to give a
  non-LLM cross-check; it moves in the same direction as the judge scores throughout, which is
  reassuring but not proof of an unbiased judge.
- **Some RoA questions are grounded in near-generic content.** A few partner-institution documents
  (flagged during question review, e.g. the Aegean Omiros reassessment-principles question) share
  boilerplate language across many documents. For these, strict single-URL retrieval scoring may
  be harsher than fair, since a different-but-equally-boilerplate document could serve the user
  just as well.
- **A handful of documents never enter the candidate pool at all**, regardless of the fixes tested
  — e.g. `roa-ug-3yr-year-1-rules.pdf` (confirmed via direct inspection to have correct
  `academic_year` metadata, so this isn't the duplication bug; it's a genuine embedding-similarity
  miss). These persisted across all three passes including the alternative embedding model,
  suggesting a genuine content-matching gap rather than a pipeline bug — worth a closer look if
  RoA retrieval quality needs to improve further.
- **Sample size.** 40 questions / 80 turns is enough to see clear, consistent directional effects
  (especially the RoA collapse and recovery), but individual document-level results have real
  variance from LLM sampling in both the contextualization rewrite and the judge — several
  individual items flipped hit/miss between passes in ways that didn't hold up as a pattern. The
  aggregate numbers, not any single question, should drive conclusions.

---

# Retrieval improvement round (same day, follow-up work)

A dedicated retrieval-optimization round followed the evaluation above, targeting hit@6 and
MRR with emphasis on RoA. It began with a failure analysis of `results_fixed.json` that
established three facts:

1. **Recall problem, not ranking**: of 26 RoA misses, only 7 had the expected document
   anywhere in the raw top-24 pool (8 in top-50).
2. **Anonymous chunks**: document identity (degree type, department, year) exists only on the
   PDF title page; all later chunks are boilerplate that is near-identical across sibling
   documents — 24/26 RoA misses retrieved a sibling (wrong degree type, wrong department, or
   wrong year).
3. **Archive pollution**: ~70-80% of chunks are historical reissues competing with current
   documents in every query.

## Changes shipped (in production now)

- **Stage 0 — better instrument**: `score_summary.py` now reports hit@k for k=1..6 plus a
  family-lenient scoring view (same document family + same academic year counts as a hit).
- **Stage 1 — contextual chunk headers + text cleaning**: every chunk is embedded with a
  one-line identity header (`Document: <readable title> | <doc type> | department | academic
  year`); extraction noise (TOC dot-leaders, repeated page headers/footers) is stripped before
  chunking. The stored chunk text is unchanged — only the embedded text gets the header, which
  is also kept in `chunk_header` metadata.
- **Stage 2 — is_current archive pre-filtering**: each document gets an `is_current` metadata
  flag (most recent academic year within its filename family; URL evidence like
  `/previous-years/`, `/current/`, and year-named directories overrides; academic-year strings
  normalized before comparison because LLM-extracted years are messy). Default retrieval
  filters to `is_current=True` (251 current docs vs 937 archived); queries mentioning a
  specific year search the full archive unfiltered. Flags are metadata-only, so a future year's
  publication only needs `recompute_current_flags()` (run automatically at the end of
  `run_ingest.py`), not a re-embed.
- **Stage 3 — hybrid retrieval**: in-memory BM25 index (`src/lexical.py`, built lazily at
  startup over header+chunk text, ~12k chunks) fused with dense results via reciprocal-rank
  fusion (k=60), then the family-recency safety net, then top-6.

## Tested and rejected

- **LLM listwise reranking (stage 4)**: fused top-24 reranked to top-6 by `qwen2.5:7b`. The
  gate analysis showed 6 of 16 remaining RoA misses sat at pool ranks 7-24 (rescuable), but in
  the full eval the reranker *hurt everything*: policy hit@6 97.5%→87.5%, RoA 60%→47.5%, answer
  score 3.88→3.62, plus ~15-40s extra latency per query. A 7B model reordering 24 near-identical
  snippets breaks more correct rankings than it fixes. Reverted; results kept in
  `results_stage4.json`. A dedicated cross-encoder reranker might do better — untested, and not
  worth it while the current numbers hold.

## Results progression (strict scoring, 80 turns per pass)

| Pass | Policy hit@6 / MRR | RoA hit@6 / MRR | Overall hit@6 / MRR | Answer |
|---|---|---|---|---|
| Pre-round ("fixed") | 77.5% / 0.71 | 35.0% / 0.15 | 56.3% / 0.43 | 3.66 |
| Stage 1+2 (headers+cleaning+filter) | 87.5% / 0.75 | 47.5% / 0.37 | 68.8% / 0.56 | 3.74 |
| **Stage 3 (+ hybrid BM25) — production** | **97.5% / 0.85** | **60.0% / 0.42** | **78.8% / 0.64** | **3.88** |
| Stage 4 (+ LLM rerank) — rejected | 87.5% / 0.73 | 47.5% / 0.36 | 67.5% / 0.54 | 3.62 |

Versus the original baseline at the very start of the day (50% overall hit@6, RoA at 22.5%),
production retrieval is now +29pp overall and RoA hit@1 went from 2.5% to 35%.

## Remaining known misses (16 RoA turns at stage 3)

Ten of sixteen are absent from every retrieval pool: the UG glossary (definition list diluted
across many terms; "capped mark" unigrams saturate the corpus so BM25 can't isolate it either),
CSEE variations (persistent across every configuration tested today), and the templated
partner-institution documents (Aegean Omiros) whose question text is genuinely generic. These
are content-matching limits, not pipeline bugs. If they matter in practice, the most promising
untested options are a proper cross-encoder reranker and per-document summary embeddings
(embed an LLM-written description of each document alongside its chunks).

---

# Code-review fix round (2026-07-17)

A high-effort code review of the whole codebase surfaced 10 verified correctness findings
(9 CONFIRMED, 1 PLAUSIBLE). All 10 were fixed, unit-verified, and confirmed live:

1. **Stale-chunk cleanup dead branch** (`prior.get("kept")` vs stored `"keep"`) — rejected
   documents' chunks are now actually deleted on re-crawl.
2. **Staggered-rollout archiving** — the year-directory rule now has a one-year grace window,
   so a department whose new edition isn't published yet stays retrievable each September.
3. **Mid-crawl invisibility** — `is_current` is computed and written at upsert time (with
   family-sibling sync), so a crashed crawl can no longer leave documents invisible to the
   default retrieval filter.
4. **Raw year-string comparisons** — one canonical `normalize_year` + `document_family` now
   live in `src/docid.py` and are used by ingestion, retrieval, flag computation, and eval
   scoring alike; unknown-year documents are no longer dropped by the recency filter.
5. **Year-mention gate** — anchored regex (money/course codes no longer trip it), and a
   mentioned year is now a *soft preference* (year-labeled pool fused with the current pool)
   rather than a filter-disable or a hard filter. The hard-filter variant was tried first and
   measurably regressed questions where a year appears incidentally (a quoted statistic range,
   a cohort start year) — the soft fusion recovers those while keeping genuine edition requests
   ("appeals rules in 2021-22") working, which the old unfiltered path handled badly.
6. **Per-turn re-summarization** — a `summarized_through` watermark means each message is folded
   into the rolling summary exactly once (~once per 11 new messages, not every turn).
7. **Stale BM25 snapshot** — a corpus-version marker (bumped on every upsert/delete/flag change,
   cross-process) triggers index rebuild, so the lexical side can't serve deleted chunks or
   stale currency flags after a crawl; index construction is also now lock-guarded.
8. **Empty-collection crash** — BM25 init guarded; a fresh or model-switched setup no longer
   500s on first query.
9. **Lenient eval scoring** — year-format variants of the same year now match; unknown-year
   pairs are only credited in evergreen (never-dated) families.
10. **Orphan conversations** — unknown conversation IDs now 404; SQLite foreign keys enabled.

Bonus from fix 3's batching: the post-crawl flag recompute dropped from ~1,200 sequential
Chroma queries (minutes) to one batched pass (~13s for 12k chunks).

## Post-fix eval (same 80-turn set)

| Pass | Policy hit@6 / MRR | RoA hit@6 / MRR | Overall hit@6 / MRR | Answer |
|---|---|---|---|---|
| Stage 3 (pre-fix production) | 97.5% / 0.85 | 60.0% / 0.42 | 78.8% / 0.64 | 3.88 |
| Post-fix, hard year filter (superseded) | 90.0% / 0.81 | 57.5% / 0.43 | 73.8% / 0.62 | 3.75 |
| **Post-fix, soft year preference — production** | 95.0% / 0.81 | 55.0% / 0.41 | 75.0% / 0.61 | 3.88 |

The final configuration measures 1-4pp below stage 3 on strict retrieval. Flip analysis
attributes this to one debatable year-path turn (a follow-up about a 2023-24 cohort now
retrieves the 2023-24-labeled payment policy instead of the expected 2025-26 edition — arguably
a legitimate source) plus two items on untouched code paths that have flipped between runs all
day (LLM sampling variance in the contextualize/judge loop; observed run-to-run variance on
identical configs is ±2-4pp). Answer quality is identical (3.88). In exchange, the system no
longer has the silent failure modes the eval can't see: mid-crawl corruption windows, the
September rollout cliff, unbounded long-conversation summarization cost, stale-index ghost
citations, and a first-query crash on fresh setups.

---

# Generalization check: independent holdout question set (2026-07-17)

Every eval pass to this point used the same 40 documents/questions the pipeline was iteratively
tuned against all day — a real risk of overfitting to that specific set rather than improving
retrieval generally. To check, a second, disjoint set of 40 documents (20 policy, 20
rules-of-assessment, zero URL overlap with the original set — `selected_docs_set2.json`) was
selected, a fresh 40 questions + follow-ups generated the same way (`questions_set2.json`), and
run against the live production server with **no code changes**.

## Result: retrieval quality generalizes

| | Production set (tuned-against) | Holdout set2 (raw) | Holdout set2 (corrected*) |
|---|---|---|---|
| Policy hit@6 / MRR | 95.0% / 0.81 | 70.0% / 0.60 | **93.3% / 0.80** |
| RoA hit@6 / MRR | 55.0% / 0.41 | 52.5% / 0.38 | 52.5% / 0.38 |
| Answer score | 3.88 | 3.81 | 3.79 |

RoA — the domain every fix today specifically targeted — needed no correction at all: 52.5% vs
55.0% on entirely different departments, degree types, and document families is well within the
noise band already established between same-config reruns elsewhere in this document. The gains
from `is_current` filtering, canonical year comparison, and hybrid BM25 are not artifacts of the
specific 20 RoA documents they were tuned against.

**Policy's raw 70% initially looked like a real regression; it wasn't.** Diagnosis: 5 of the 6
policy misses shared `is_current: False` and their questions never mention a year. All 5 are
superseded-year editions (`academic-offences-procedure-2024-25.pdf`,
`procedures-fitness-to-practise-2024-25.pdf`, etc.) that I selected as ground truth when picking
new documents to avoid overlapping the original set's topics — but the corpus's richest policy
topics were already used there, so the remaining unused pool skewed toward older editions of
those same families. `is_current` correctly prefers the current edition over the one I picked,
which the strict metric then scores as a miss even though the live answer is arguably *more*
correct than the test expected. \* Excluding those 5 confounded documents (35 of 40 remain):
policy hit@6 is 93.3%, matching production's 95.0% within noise.

One genuine, uncorrelated miss remains (`compensation-and-refund-policy.pdf`, `is_current: True`,
no year confound) — a real single-item gap, not a pattern.

**Conclusion: today's retrieval work generalizes.** The lesson for future holdout sets: exclude
`is_current: False` documents at selection time unless the question explicitly targets a past
year, or the confound above will recur by construction.

---

# User-reported bug: contextualizer topic drift in long conversations (2026-07-17)

Manually testing the live app (not via the eval harness), the user asked two questions in a
single conversation that had already drifted across several unrelated topics (exam-chair
requirements → CSEE programme listings → back to Professional Doctorate governance). Both got a
"the provided context does not include..." non-answer, despite both being directly answerable
from ingested documents.

## Root cause

`_contextualize_query()` rewrites every follow-up question into a standalone form before
retrieval, using the last 6 messages of conversation history. With a long, topic-switching
history, the small local contextualizer model sometimes **echoed a different, earlier question
from the transcript instead of rewriting the actual new one**. Reproduced exactly against the
real stored conversation: asked about "Professional Doctorate Director responsibilities," the
rewrite came back as "Are any of the CSEE programs listed earlier running in 2025-26?" — a
question from six turns earlier. Retrieval dutifully fetched CSEE documents; the real answer was
never in context. The next question suffered the same failure in reverse. Neither of this
project's eval sets exercises multi-topic conversations (every eval conversation is one topic,
two turns), so this failure mode was invisible to all retrieval testing done today.

## Fix and a lesson about isolating changes

`_is_faithful_rewrite()` in `src/rag.py`: a deterministic content-word-overlap check comparing
the rewrite against the original question. If the rewrite shares too few significant words with
the original, it's discarded in favor of the raw question — a topic-hijacked rewrite and the
original share essentially zero content words, so this catches the failure directly regardless
of why the small model drifted.

The first attempt at this fix (bundled a reworded contextualize prompt — "if already
self-contained, output unchanged" — together with the new guard) regressed RoA hit@6 by 7.5pp in
the standard eval. Diffing retrieved documents between passes showed why: the reworded prompt
made the model skip adding disambiguating programme/document names to follow-ups that read as
grammatically self-contained but still needed that detail to distinguish among near-identical RoA
sibling documents (e.g. "what happens if a student fails a core module?" needs "...in the MSc HRM
programme" injected to retrieve the right one out of dozens of structurally identical
department-specific rules documents). Reverting to the original prompt wording and keeping only
the guard (current production) restored RoA to its prior level while still catching the original
bug, re-verified against the exact real conversation that triggered it.

| Pass | Policy hit@6 / MRR | RoA hit@6 / MRR | Overall hit@6 / MRR | Answer |
|---|---|---|---|---|
| Before this fix | 95.0% / 0.81 | 55.0% / 0.41 | 75.0% / 0.61 | 3.88 |
| Guard + reworded prompt (rejected) | 95.0% / 0.81 | 47.5% / 0.35 | 71.3% / 0.58 | 3.77 |
| **Guard only, original prompt — production** | 95.0% / 0.84 | 55.0% / 0.43 | 75.0% / 0.64 | 3.73 |

**Lesson applied going forward:** don't bundle a prompt wording change with a structural/code
fix in the same eval pass — attributing a regression to the right one of two simultaneous changes
after the fact takes a full extra eval cycle that isolating them up front would have avoided.

Separately, this incident is also the reason the ~450 conversations the day's automated eval runs
had accumulated in `data/chat.db` were cleared out (identified precisely by matching each
conversation's message count and first-message text against the known eval question sets, so the
user's own real conversations — including the one that surfaced this bug — were left untouched).

---

# Second RoA improvement round (2026-07-18)

RoA retrieval held at 55% hit@6 / 0.43 MRR even after the fixes above — much better than the
original 22.5%, but still weak in absolute terms, and confirmed on the independent holdout set
too, so a real ceiling rather than a measurement artifact. A fresh failure analysis of the 18
current RoA misses found the same pattern almost everywhere: retrieval lands in the exact right
topical neighborhood but picks the wrong sibling document (wrong year, wrong degree length, wrong
department, wrong award type) - the underlying boilerplate-similarity problem the chunk headers
only partially fixed. A larger-pool check found 13 of 18 misses (72%) were present somewhere in a
top-50 union, just poorly ranked - a genuine reranking opportunity - while 5 of 18 (28%) were
absent even from a top-50 pool, which no reranker can fix.

## What was tested

| Change | Result | Kept? |
|---|---|---|
| **Smaller chunks** (300→175 words, overlap 50→30) | RoA hit@6 55%→62%, MRR 0.43→0.45, no policy regression | **Yes** |
| **Cross-encoder reranker** (`BAAI/bge-reranker-base`, rescoring top-30 fused candidates; `sentence-transformers`/`torch` were already installed, no new dependency) | Policy hit@6 reached 100%; RoA hit@3/MRR improved, hit@6 flat (62%→60%, within noise) | **Yes** |
| BM25 header-boost (repeat `chunk_header` 5x so identity terms outweigh boilerplate body text) | Regressed RoA hit@6 60%→53% - amplifies the header's generic shared words right along with the genuinely distinguishing ones | No, reverted |
| Embedding model retest: **`bge-m3`** (8192-token context, no truncation risk) | Wash-to-slight-regression on RoA (hit@6 60%→57%, hit@3 57%→50%); answer quality improved slightly | No, reverted |

**Cross-encoder reranker note:** the first implementation scored the raw stored chunk text,
which doesn't include `chunk_header` (only prepended at embedding time, never stored) - so the
reranker was working with less identity signal than the embedder had, and failed a manual
exemplar test as a result. Fixed by scoring `chunk_header + chunk_text` together, which is what
actually delivered the policy/RoA gains above.

**On "would testing different models help" specifically:** two embedding models were tested
today (`mxbai-embed-large` in the first round, `bge-m3` in this one) and neither beat
`nomic-embed-text` - `mxbai` regressed RoA outright (its 512-token window likely truncates dense
technical chunks), and `bge-m3` (chosen specifically to rule out that truncation risk) still came
out a wash-to-slight-regression. This is reasonably strong evidence that the embedding model
isn't the lever that unlocks further RoA gains on this corpus - the near-duplicate boilerplate
structure of the documents themselves is the dominant constraint, which is why chunk size and
reranking (which change how the *existing* signal is used) worked while model swaps didn't. The
generation/chat model was not retested, since every pass all day shows answer quality staying
strong (3.5-4.1/5) whenever retrieval succeeds - no evidence points at generation as a bottleneck.

## Where this leaves things

| | Start of this round (`postfix4`) | Current production (`stage1_rerank`) |
|---|---|---|
| RoA hit@1 / hit@3 / hit@6 / MRR | 38% / 50% / 55% / 0.43 | 38% / 57% / 60% / 0.45 |
| Policy hit@6 / MRR | 95% / 0.84 | 100% / 0.86 |
| Overall hit@6 / MRR | 75% / 0.64 | 80% / 0.66 |

A real, if modest, RoA improvement (+5pp hit@6, +7pp hit@3) and a clean policy improvement, with
two honestly-reported negative results (header-boost, bge-m3) along the way. RoA hit@6 remains
below the plan's original "pursue contextual per-chunk embeddings" threshold of ~70%, so that
option (a one-time LLM-generated situating description per chunk, substantial engineering and
compute cost) remains on the table but undecided - it's a bigger investment than anything tried
today, and the two most recent experiments' mixed results are a reason for a deliberate go/no-go
decision rather than an automatic next step.

## Files in this folder

- `selected_docs.json`, `questions.json` — the original 40-document/question set (tuned-against)
- `selected_docs_set2.json`, `questions_set2.json` — the independent holdout set
- `results_baseline.json`, `results_fixed.json`, `results_mxbai.json` — raw per-question
  results for the original evaluation round
- `results_stage2.json`, `results_stage3.json`, `results_stage4.json` — raw results for the
  first retrieval improvement round
- `results_postfix.json`, `results_postfix2.json` — raw results for the code-review fix round
- `results_holdout_set2.json` — raw results for the generalization check
- `results_postfix3.json` (rejected), `results_postfix4.json` — raw results for the
  contextualizer-drift fix
- `results_stage0_chunks.json`, `results_stage1_rerank.json` (current production),
  `results_stage2_header_boost.json` (rejected), `results_stage3_bgem3.json` (rejected) — raw
  results for the second RoA improvement round described above
- `EXPERIMENTS.md` — exact parameters and headline metrics for every pass, for fast comparison
  and reverting via git if a future change regresses
- `run_eval.py`, `score_summary.py`, `generate_questions.py` — the eval harness itself, reusable
  for future re-evaluation after any further changes (both now accept a question-set path as a
  CLI argument, so a third set doesn't require duplicating either script)
