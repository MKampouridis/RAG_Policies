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

## Stage 4: contextual per-chunk embeddings (piloted, rejected)

RoA hit@6 (60%) sat below the plan's ~70% threshold for considering this option, so with the
user's go-ahead it was attempted - but the full scope (843 documents, 14,006 chunks, one LLM call
per chunk to generate a situating sentence) turned out to cost an estimated **~20 hours**, far
more than anticipated. Before committing to that, a **bounded pilot** was run: just the 34
documents/580 chunks behind the current 18 known misses (~48 minutes of generation), re-embedded,
and measured.

An initial single-call-per-document design (send the whole document once, ask for all its
chunks' situating context back in one structured response) failed outright - the local 7B model
couldn't reliably track alignment between many chunks and produced malformed output. Switched to
one call per chunk (simpler task, ~5-7s each, reliable output) - which is what set the ~20-hour
full-scope estimate.

**Pilot result: negative.** The 80-turn aggregate looks close to neutral, but that's misleading -
only 34 of 843 in-scope documents were touched, so 758 documents' worth of untouched questions
dominate the aggregate. Isolating just the 22 turns that actually target pilot-scope documents:
**zero turns improved, one regressed** (a glossary follow-up that previously ranked the correct
document at rank 4 now doesn't retrieve it at all). Manually re-tested the specific hard exemplar
this was meant to fix (4-year vs 5-year integrated masters confusion) - still wrong, identically
to every other configuration tried today, suggesting this is a case where the two documents'
content is genuinely too similar for any of today's techniques to disambiguate reliably, not a
gap contextual embeddings happened to fill.

Reverted (`rm -rf data/chunk_context_cache`, re-embedded from cache) rather than accept a known
regression on the exact cases it was meant to fix, or spend ~20 more hours chasing a technique
whose validation pilot came back negative. This is the third consecutive rejected experiment
after header-boost and bge-m3 - reasonably strong evidence that the remaining RoA gap (documents
whose content is nearly indistinguishable from a sibling's) needs either a fundamentally
different approach (e.g. a bigger/differently-trained reranker, or restructuring the corpus
itself) or may simply be close to this system's practical ceiling without much larger
engineering investment than justified by the size of the remaining gap.

## Where this leaves things

| | Start of the day | After first RoA round (`postfix4`) | Current production (`stage1_rerank`) |
|---|---|---|---|
| RoA hit@1 / hit@3 / hit@6 / MRR | 2.5% / — / 22.5% / 0.06 | 38% / 50% / 55% / 0.43 | 38% / 57% / 60% / 0.45 |
| Policy hit@6 / MRR | 77.5% / 0.63 | 95% / 0.84 | 100% / 0.86 |
| Overall hit@6 / MRR | 50% / 0.35 | 75% / 0.64 | 80% / 0.66 |

RoA hit@6 went from 22.5% to 60% over the full day (both improvement rounds combined) - a real,
substantial gain, even though the final three experiments (header-boost, bge-m3, contextual
embeddings) all came back negative. Two embedding models were tested and both underperformed
`nomic-embed-text`; a purpose-built reranker and smaller chunks both helped; a richer per-chunk
LLM-generated context signal, at real compute cost, didn't. The practical takeaway for future
work on this corpus: further gains are more likely to come from changes to how retrieval *uses*
existing signal (chunking, ranking) than from bigger models or richer per-chunk metadata.

---

# Literature-grounded improvement round (2026-07-18, later)

With RoA hit@6 stuck at 60% after three consecutive negative experiments, the user asked for
deep research into the academic and practitioner literature on this exact failure class before
proposing more changes, plus a direct answer on whether different models would help.

## What "the RoA problem" actually is

Academic-year duplication was a large piece of the *original* problem, and it's already solved
(the `is_current` filter + canonical year normalization earlier today). What remains is the same
underlying phenomenon - near-identical boilerplate regulatory text - recurring along dimensions
that don't have year's clean, regex-matchable structure: wrong degree length (4yr vs 5yr
Integrated Masters), wrong department/programme (CSEE vs Social Work), wrong award type (Grad
Cert vs Grad Dip), and primary questions that are genuinely underspecified (no identifying detail
at all, ambiguous across dozens of documents).

## Research findings

Eight web searches plus three full-paper fetches (arXiv + practitioner sources):

- **This is a named, well-documented problem class.** Legal-tech practitioners describe the
  identical failure in NDA retrieval - documents "structurally almost identical...differing only
  in critical variables like party names or dates" confusing vector similarity - solved there via
  checksum-guarded citations and metadata-encoded disambiguation rules.
- **["When More Documents Hurt RAG"](https://arxiv.org/pdf/2606.11350)** names our exact symptom
  ("vector search dilution") and proposes **domain-scoped retrieval**: hard-partition the corpus
  by metadata *before* ranking, rather than soft-preferring it after. A materially different
  mechanism than the soft RRF-fusion already used for academic year.
- **["Metadata, Structure, or Strategy?"](https://arxiv.org/pdf/2606.29645)** explains *why* two
  of today's earlier experiments (BM25 header-boost, contextual embeddings) backfired: retrieval
  *strategy* (which documents get selected/ranked) dominates outcomes, while chunk-level metadata
  /context enrichment has diminishing returns that go **negative** past a threshold - independent
  literature confirmation of an independently-discovered result.
- **["Retrieval Improvements Do Not Guarantee Better Answers"](https://arxiv.org/html/2603.24580v1)**
  (a 947-document AI-policy corpus study) found retrieval gains don't help when the query is
  genuinely ambiguous - the generator produces a fluent, confidently wrong answer instead of
  surfacing uncertainty. Matches several of our remaining misses precisely (primary questions
  with zero identifying information).
- **ColBERT-style late interaction** (via [RAGatouille](https://github.com/AnswerDotAI/RAGatouille)
  / [PyLate](https://github.com/lightonai/pylate)) and **SPLADE learned sparse retrieval** were
  identified as established, locally-deployable alternative mechanisms not yet tried (token-level
  MaxSim matching and learned sparse term expansion, respectively - both fundamentally different
  from the single-vector dense embeddings and raw-frequency BM25 used all day).

## What was tried: two ideas killed by pre-validation, one real win

**Facet-based hard filtering (degree-length/award-type) - killed before writing any code.**
Rigorously checked whether the distinguishing fact for each of the 16 current misses appears
*anywhere in the question text* (not just in the document): **13 of 16 (81%) don't mention it at
all.** No filter, however well-designed, can act on a fact the question never states - the same
coverage problem that killed the original department-field attempt, caught this time before any
implementation cost.

**Ambiguity detection + clarifying question - killed before writing any code.** Tested the two
candidate signals available in the pipeline (family diversity in the top-6; query content-word
count as a proxy for "genericness") against the full eval set. Family diversity correlates in
aggregate (hits average 3.5 same-family repeats in top-6; misses average 1.6) but the
distributions overlap too much for a safe threshold - even the strictest cutoff caught only 56%
of misses while wrongly flagging 14% of *currently-correct* answers. Content-word count didn't
separate hits from misses at all: "minimum weighted average...to pass a Master's degree with
Merit" (a hit) and "...for a student to pass with Merit" (a miss) are nearly identical phrasings
with opposite outcomes - whether a query hits or misses depends on retrieval internals invisible
from the query's surface form. Building a clarification trigger on either signal would have meant
either missing most real ambiguity or interrupting currently-working answers for uncertain gain.

**ColBERT reranker (PyLate, `lightonai/GTE-ModernColBERT-v1`) - the day's best single result.**
RAGatouille (the more famous wrapper) turned out to be broken against the currently-installed
langchain version (an unrelated transitive-dependency incompatibility); PyLate, a more actively
maintained library the RAGatouille project itself is migrating to, worked cleanly instead - both
installed via pip, no new infrastructure. Swapped in as a direct replacement for the existing
`BAAI/bge-reranker-base` cross-encoder over the same fused candidate pool (`src/rerank.py`,
`BACKEND = "colbert"`).

| Pass | Policy hit@6 / MRR | RoA hit@6 / MRR | Overall hit@6 / MRR | Answer |
|---|---|---|---|---|
| Cross-encoder (prior production) | 100% / 0.86 | 60% / 0.45 | 80% / 0.66 | 3.81 |
| **ColBERT late interaction — production** | **100% / 0.91** | **70% / 0.45** | **85% / 0.68** | 3.89 |

RoA hit@6 jumped 10 percentage points - the single largest gain since the original hybrid
dense+BM25 retrieval fix. Flip analysis: 5 turns gained, 1 lost, spread across 5 different
document families - a genuine, well-distributed improvement, not a fluke concentrated in one
area. Worth noting for process: manual spot-checks on the hardest known exemplar (4-year vs
5-year Integrated Masters) looked unconvincing beforehand (near-identical MaxSim scores, wrong
document still ranking first) - the full 80-turn aggregate told a materially better story. Same
lesson as the header-boost experiment, in the opposite direction: trust the full eval over
hand-picked spot-checks, whether they look promising or not.

**Not yet tried at the time** (remaining items from the research): SPLADE as a third RRF-fused
retrieval channel, and a cheap embedding-model ensemble (`nomic-embed-text` + the already-embedded
`bge-m3` collection, fused via the existing RRF mechanism) - both attempted in the follow-up
round below.

# Try-everything round: eight more experiments, all reverted (2026-07-19)

The user explicitly asked to try every remaining idea from the research plan (`tender-strolling-
storm.md`), including the two killed by pre-validation above, plus a fresh literature review via
Consensus focused specifically on retrieval over corpora with **overlapping, non-mutually-
exclusive facets** (this corpus's actual structure: a masters RoA document can legitimately hold
the correct answer to a diploma-exit-award question). Every one of the eight experiments below
was measured against the same 80-turn eval and the `stage_colbert` production baseline (RoA
hit@6 70%, overall 85%, policy 100%); every one regressed or washed. All are implemented behind
off-by-default flags in `src/rag.py` rather than deleted, in case a future refinement makes one
of them viable.

| Pass | Policy hit@6 / MRR | RoA hit@6 / MRR | Overall hit@6 / MRR | Answer | Verdict |
|---|---|---|---|---|---|
| `stage_colbert` (baseline) | 100% / 0.91 | 70% / 0.45 | 85% / 0.68 | 3.89 | — |
| **Stage A**: hard facet filter | 100% / 0.89 | 47.5% / 0.33 | 73.8% / 0.61 | 3.73 | rejected |
| **Stage A v2**: hard filter, `3yr`/`4yr` regex bug fixed | 100% / 0.92 | 57.5% / 0.41 | 78.8% / 0.66 | 3.74 | rejected |
| **Stage A2**: soft RRF-fuse instead of hard filter | 100% / 0.89 | 60.0% / 0.40 | 80.0% / 0.65 | 3.86 | rejected |
| **Stage F**: weighted fusion vs RRF (fast sweep, not a full eval) | — | best tie 67.5% | best tie 83.8% | — | rejected (no config beat RRF) |
| **Stage G**: deterministic pseudo-query index | 100% / 0.90 | 70.0% / 0.45 | 85.0% / 0.68 | 3.81 | rejected (net-zero wash) |
| **Stage H**: CRAG-style verification/gating | 100% / 0.86 | 65.0% / 0.42 | 82.5% / 0.64 | 2.88 | rejected |
| **Stage D**: SPLADE third retrieval channel | 100% / 0.91 | 65.0% / 0.45 | 82.5% / 0.68 | 3.90 | rejected |
| **Stage E**: embedding ensemble (nomic + bge-m3) | 100% / 0.89 | 57.5% / 0.40 | 78.8% / 0.65 | 3.80 | rejected |

**Stage A / A2 (facet filtering, hard then soft).** The user asked for this despite the earlier
pre-validation finding (13/16 misses don't mention a facet in the question text at all) - correctly
predicted low value, but the actual result was worse than "low value": a **regression**. First
attempt had a real bug (degree-length regex required spelled-out "year" but filenames abbreviate
`3yr`/`4yr`/`5yr`, so many documents' own facet extraction came back empty even when the query's
did match) - fixing it recovered some ground (RoA 47.5%→57.5%) but the regression persisted.
Root cause, confirmed by manual inspection: this corpus's facets are **not mutually-exclusive
partitions** - a masters-labeled document can legitimately hold the correct diploma-exit-award
answer, so excluding non-matching documents throws away real answers (exactly what "When More
Documents Hurt RAG"'s domain-scoped-filtering fix assumes ISN'T true, but is here). Converting to
a soft RRF-fuse (same mechanism already proven for year-mentions) per a follow-up Consensus
literature review recovered further ground (RoA 57.5%→60%) but still net-regressed, because
extraction-side gaps mean many correct documents (filenames like `east15-25.pdf`,
`mscperiodontology_25.pdf`) never get tagged with any facet at all, so they get no boost from the
soft preference while occasional false-positive matches on unrelated documents do - net negative
even without ever hard-excluding anyone. Conclusion: this corpus's facet signal is too sparse and
non-exclusive for either mechanism; not worth pursuing further without a fundamentally different
(e.g. graph-based, non-exclusive) representation - see Consensus's rank-6 suggestion, held as a
conditional future option.

**Stage F (weighted score fusion vs RRF, Bruch et al. 2022).** Built a fast retrieval-only sweep
(`eval/sweep_fusion_weights.py`, skips answer-generation/judge calls entirely) across five
dense/BM25 weight splits plus the RRF baseline. No weighted config decisively beat RRF on RoA
hit@6 - best case was an exact tie (50/50 and 40/60 both matched RRF's 67.5%); pushing weight
toward either extreme actively hurt (70/30→62.5%, 30/70→65%). Concluded RRF is already
competitive and didn't spend a full 80-turn production eval validating a config that only ties at
best - consistent with Bruch et al.'s own caveat that weighted fusion needs real in-domain tuning
signal to beat RRF, which this sweep didn't surface. `_weighted_dense_bm25()` and
`WEIGHTED_FUSION_ENABLED`/`DENSE_WEIGHT`/`BM25_WEIGHT` remain in `src/rag.py`, off by default.

**Stage G (deterministic pseudo-query index, Tang 2021 / Lee 2025).** Built a second Chroma
collection (`build_pseudo_query_index.py`) indexing each chunk with 1-2 metadata-filled question
templates (e.g. "What are the rules of assessment for a {degree_length} {award_type}
programme?"), deliberately *not* another LLM narrative-generation pass like the rejected
stage4_context_pilot - templates are filled from already-extracted metadata, no chat calls.
11,202 pseudo-queries generated from 20,498 chunks. Queried as a fourth RRF channel
(`src/pseudo_query.py`), with a new post-fusion dedup step (`_dedup_by_chunk`) added to collapse
the same real chunk surfacing under two different ids (its own + a pseudo-query's). Result: an
exact hit@6 tie with baseline at every level (100%/70%/85%) - flip analysis showed this wasn't a
true no-op but a genuine 1-for-1 cancel-out (gained one follow-up turn, lost a different one).
Not harmful, but not worth the added complexity (a second collection to keep in sync, one more
embedding call per query) for zero net benefit. Off by default.

**Stage H (CRAG-style retrieval verification, Yan et al. 2024).** Added a lightweight LLM check
(`_context_supports_answer()`) before generation: ask the same local chat model whether the
retrieved excerpts actually support answering, and return an explicit uncertainty message instead
of a confident guess when they don't - intended as a better-founded alternative to the
family-fragmentation heuristic considered (and rejected) for Stage B. Regressed clearly, for two
distinct reasons, not just the anticipated "judge doesn't reward abstention" scoring caveat: (1)
the verifier massively **over-triggered** - 66 of 80 turns (82.5%), including turns where
retrieval had actually succeeded, meaning the local model's calibration on "confidently and
specifically" is far too conservative as currently prompted; (2) a genuine, previously
unanticipated **architectural side effect**: gating the primary turn's answer with a generic
uncertainty message means the follow-up turn's query-contextualizer sees that generic message in
history instead of a real answer, which measurably regressed follow-up-turn retrieval itself
(34/40→32/40) even though primary-turn retrieval was completely unaffected (34/40→34/40, since
`retrieve()` itself doesn't read this flag) - a single-turn QA benchmark (CRAG's original
evaluation context) wouldn't surface this multi-turn knock-on cost. Off by default.

**Stage D (SPLADE third retrieval channel).** Built a `naver/splade-cocondenser-ensembledistil`
sparse index over all 20,498 chunks (`build_splade_index.py`, ~105 minutes on this hardware,
cached to `data/splade_matrix.npz`), queried via `src/splade.py` as a third RRF channel alongside
dense+BM25. Regressed: RoA hit@6 70%→65%, overall 85%→82.5% (net -2 turns: +3/-5), almost
entirely on follow-up retrieval - the extra channel appears to add noise to the 3-way RRF fusion
that disproportionately affects follow-up queries, similar to Stage G's dilution risk but without
Stage G's offsetting gains. Combined with the real build cost, not worth keeping. Off by default.

**Stage E (embedding-model ensemble, nomic + bge-m3).** Queried the already-populated
`policies_bge-m3` collection (left over from the earlier `stage3_bgem3` experiment) as a second
dense channel, RRF-fused alongside the primary `nomic-embed-text` channel. The worst regression
of the eight: RoA hit@6 70%→57.5%, overall 85%→78.8%. Consistent with `stage3_bgem3`'s original
finding that bge-m3 alone was a wash/slight regression on RoA specifically - fusing its weaker RoA
rankings in via RRF introduces enough noise to displace `nomic-embed-text`'s correct results from
the top ranks rather than complementing them. Off by default.

## Where this leaves things (2026-07-19)

Production configuration is unchanged from `stage_colbert`: hybrid dense+BM25, `is_current`
pre-filtering, family-based recency dedupe, ColBERT late-interaction reranking. RoA hit@6 remains
at 70% (up from 22.5% at the start of the original round). Eight further ideas - two from the
original research plan (SPLADE, embedding ensemble) and five suggested by a second, more targeted
literature review (soft facet fusion, weighted fusion, pseudo-query indexing, CRAG verification,
plus the facet-filtering retry) - were each implemented, measured in isolation, and reverted. The
recurring theme across all eight: **this corpus's remaining misses don't respond to more
retrieval-side machinery** - every new signal either has too many extraction/coverage gaps to be
trustworthy (facets, pseudo-queries), or adds noise that a 2-source RRF fusion doesn't have
(SPLADE, bge-m3 ensemble), or fails in a way specific to this being a *conversational* system
rather than a single-shot QA benchmark (CRAG's follow-up knock-on effect). The genuinely
unexplored options left from the original research are the higher-effort, more architecturally
different ones: a small deterministic facet-*overlap* graph (Consensus rank 6, modeling
cross-references explicitly rather than assuming exclusivity) and selective multi-hop
decomposition (Consensus rank 4, triggered only on detected cross-document ambiguity) - both
still on the table as conditional next steps, not attempted this round given the consistent
negative signal from cheaper mechanisms tried first.

## Pre-validation: facet-overlap graph killed before writing any code (2026-07-19, later)

Before building the facet-overlap graph, categorized all 12 current RoA misses (`stage_colbert`
baseline, across both primary and follow-up turns) by actual failure mode:

- **7/12 turns: genuinely underspecified query, no identifying facet mentioned at all** (Capped
  Mark glossary term, "types of classifications", CSEE variations x2, Diploma in HE x2, MA Social
  Work "minimum weighted average to pass with Merit"). No department, degree length, award type,
  or programme name appears in the question - there's nothing for any facet mechanism, graph or
  otherwise, to route on.
- **5/12 turns: same-facet-family sibling confusion, wrong granularity** (Integrated Masters x2,
  MSc Periodontology, Aegean-Omiros partner programme x2). The wrongly-retrieved document shares
  the *same* degree_length/award_type as the correct one - the miss is between siblings that only
  differ by department, specific programme, or partner institution, none of which are extracted
  reliably (department-field coverage was already checked earlier in the corpus and found too
  sparse to use - see the second RoA improvement round).
- **0/12: genuine cross-reference/overlap** (a document tagged with one facet legitimately
  holding the answer to a different facet's question) - the motivating case for the graph. The
  one real example we found (`masters-25.pdf` correctly holding a Postgraduate Diploma
  exit-award answer) is a **hit** in the `stage_colbert` baseline; it only became a miss when
  Stage A2's own soft facet-preference mechanism wrongly deprioritized it. That failure was
  self-inflicted by our attempted fix, not a naturally occurring gap in the unmodified pipeline.

**Conclusion: not building the facet-overlap graph.** It targets a failure mode that doesn't
appear in the current miss set, and even if built, it would only re-enable a facet-preference
mechanism already shown net-negative for unrelated reasons (sparse extraction, insufficient
granularity - see Stage A/A2 above). The dominant real failure modes (underspecified queries;
same-family sibling confusion needing finer-grained identifiers than we can reliably extract) are
not addressed by modeling facet overlap.

## Stage I: selective multi-hop query decomposition (2026-07-19, later still)

Tried anyway, at the user's request, despite the pre-validation finding above predicting low
value (neither dominant failure mode obviously calls for decomposition). Triggered only when the
initial reranked top-6 is fragmented across many document families (reusing Stage B's validated
`_top_family_count` signal - a false-positive trigger here only costs extra retrieval/rerank
compute, not a wrong response type, so the same imprecise signal is more defensible here than it
was for Stage B's clarifying-question behavior). On trigger, asks the local chat model to rewrite
the ambiguous question into up to 3 concrete, document-specific hypotheses (one per plausible
candidate family found), retrieves for each, and RRF-fuses the union with the original pool
before a second rerank pass.

| Pass | Policy hit@6 / MRR | RoA hit@6 / MRR | Overall hit@6 / MRR | Answer |
|---|---|---|---|---|
| `stage_colbert` (baseline) | 100% / 0.91 | 70.0% / 0.45 | 85.0% / 0.68 | 3.89 |
| `stageI_multihop_decomposition` (rejected) | 100% / 0.89 | 62.5% / 0.40 | 81.2% / 0.65 | 3.88 |

Regressed: RoA hit@6 70%→62.5% (net -3 turns: +1/-4). Manual spot-checks on the two dominant
failure-mode exemplars (Capped Mark glossary term; MSc Periodontology home-vs-partner-institution
confusion) both still missed after decomposition, exactly as predicted - neither underspecified
queries (nothing to hypothesize a distinguishing fact from) nor same-family sibling confusion
(the generated hypotheses don't reliably surface facts like "home institution, not a partner
variant" that aren't implied by the question or the candidate titles) benefit from this
mechanism. The one flip analysis surprise: it *did* recover one genuine former miss
(`roa-ug-aegean-omiros-4yr-non-standard-year-1.pdf` follow-up) - decomposition can occasionally
help - but the 4 turns lost elsewhere show the added candidate pool more often dilutes the rerank
step with a wrong hypothesis's results, displacing documents the original single-shot retrieval
had already found correctly. Off by default (`MULTIHOP_DECOMPOSITION_ENABLED` in `src/rag.py`);
kept for reference, not a dead end worth deleting.

This brings the total to nine tried-and-reverted ideas from the literature-grounded round, with a
consistent verdict: this corpus's remaining RoA misses are dominated by (a) genuinely
underspecified questions with no exploitable signal, and (b) same-family sibling confusion
needing a finer-grained document identifier than anything reliably extractable so far - neither
of which has responded to any retrieval-side mechanism tried (facet filtering/preference,
weighted fusion, pseudo-query indexing, verification/gating, SPLADE, embedding ensemble, or
query decomposition).

# Identity-first round ("J round", 2026-07-19/20)

Plan synthesized from four external LLM responses (Fablo/ChatGPT/Gemini/DeepSeek via Grok) to a
detailed problem prompt. Their convergent diagnosis: the remaining misses are entity-
identification failures, the per-document extraction cost had been mispriced ~19x, and the eval
itself was becoming a bottleneck. Ten stages run; one kept (J6), the rest were diagnostics
(several highly valuable) or reverted experiments.

| Stage | What | Verdict |
|---|---|---|
| J0 | Diagnostics: pool-recall split + judge scores on the 12 misses | 4 misses out-of-pool, 4 in-pool-beyond-rerank-window, 4 seen-and-misranked; all 12 misses still judged 3-4 |
| J0b | Widen `RERANK_POOL_SIZE` 30→100 | **Reverted**: rescued 2 deep-pool targets but lost 5 (RoA 70→62.5%) - more candidates = more indistinguishable boilerplate |
| J1 | Per-document identity extraction (1,188 docs, first ~2 pages + filename → programme/dept/partner/awards/aliases JSON) | **Kept as data asset** (`data/doc_identity/`): 0 failures, all 5 sibling-confusion miss docs got real identity |
| J2 | Identity-enriched chunk headers (re-embed) | **Reverted**: RoA 70→60% despite +1 target rescue and improved MRR - documents with EMPTY identity records still flipped hit→miss because ~450 other docs' embeddings moved (corpus-wide displacement) |
| J3 | Document-level identity index + soft routing prior (1,188 identity cards, chunk embeddings untouched) | **Reverted**: 0 rescues / 3 losses, all four metrics down - true siblings' identity cards are themselves near-identical (home vs partner MSc Periodontology) |
| J4 | User-turns-only follow-up contextualizer | **Reverted**: small net regression incl. the follow-up-only split it targeted - assistant answers DO carry referents follow-ups point at; reconsider only if answer-gating returns |
| J5a | Evidence-sufficiency metric (`eval/score_evidence_sufficiency.py`) | **Key finding**: 7 of 12 strict misses retrieved a sibling containing ≥half the key facts - effective RoA evidence retrieval is **87.5%**, true deficit is 5 turns |
| J5b | Sibling-discriminating question set (`eval/questions_set3_sibling.json`, 20 programme-named pairs from identity records) | **Key finding**: when the question names the programme, production scores **90% hit@6 / 95% primary** (2 of 4 misses are test artifacts - superseded-edition gold docs). Sibling discrimination is already strong when identity is in the query |
| J6 | Disclose-don't-gate: append a source-naming disclosure when the top-6 is family-fragmented | **KEPT (production)**: retrieval untouched, answer score ~flat (3.89→3.86, within noise), fired on 9/12 actual misses (converting silent wrong-sibling answers into correctable ones) at a truthful-caveat cost on 26% of hits. Also incidentally measured the eval's noise floor (~1-2 turns of hit@6 swing between runs with zero retrieval changes) |
| J7 | "Quote figures verbatim" rule in SYSTEM_PROMPT | **Reverted**: overall keyphrase +1.7pp but RoA keyphrase -1.4pp, answer -0.06 - the 7B generator doesn't reliably follow the instruction; retry with a stronger generator (deferred LLM phase) |
| J8 | Nameable-identity clarifying question: ask only when the candidate pool's J1 identity records contain >=2 distinct nameable labels; else fall through to J6 | **Killed by manual pre-validation, no full eval run** - see below |

**J8 in detail.** Prompted by the user asking whether the system could proactively identify
sibling documents and ask the user to confirm which one they meant. Manual simulation first
tested the core hypothesis directly: for the 7 "underspecified" misses, reformulating the query
with the correct missing fact (from the target document's own J1 identity record) and re-running
retrieval. Result: all 3 documents that actually have a real identity to name (CSEE, MA Social
Work) recovered cleanly (rank 1-2); the other 4 turns (2 genuinely university-wide documents -
the glossary and the generic Diploma of Higher Education - with no programme to name at all) stayed
broken even with a realistic "no, it's not programme-specific" answer, since that carries almost
no distinguishing content. This confirmed clarification only helps when a real identity exists to
solicit - a materially sharper diagnosis than Stage B's original blanket signal.

But implementing the auto-detection step (which candidates to name) surfaced a harder problem:
scanning identity labels among documents retrieval ALREADY GOT WRONG has no way to surface the
CORRECT option. Tested on the MA Social Work miss: the wrongly-retrieved candidates' identities
were MSc AI, East 15, Sport/Rehab Science, and CSEE - four confident, plausible-sounding, entirely
wrong choices, none of them Social Work. Re-sourcing candidates from the J3 document-identity
index queried against the raw question text (rather than the retrieved chunk pool) failed
identically, for the identical root reason: a genuinely underspecified query carries no signal for
any index - chunk-level or document-level - to match "Social Work" against. Conclusion: you
cannot auto-detect good clarification options for exactly the queries that need them most; the
missing fact is only recoverable by asking a fully generic question with no named guesses (which
risks nothing since it commits to no guess) - and that's what J6's disclosure already does,
without reintroducing the gating cost Stage H demonstrated. Killed before any full eval was run;
`NAMEABLE_CLARIFICATION_ENABLED` in `src/rag.py` stays `False`, code kept only as documented
reference.

**Where this leaves the system (2026-07-20).** Production = `stage_colbert` retrieval + J6
disclosure. Strict RoA hit@6 remains 70%, but the round's diagnostics reframed what that number
means: evidence-sufficient retrieval is 87.5% (only 5 turns fail to bring the key facts into
context), sibling discrimination is ~95% when the query names its programme, and the answers on
strict misses still score 3-4 because sibling boilerplate carries most of the substance. The
dominant remaining costs are (a) underspecified queries - now mitigated by J6's disclosure
inviting correction - and (b) generation-side imprecision (68% keyphrase coverage even on hits),
which is the deferred LLM-experiments phase's target. The J1 identity records and the set3
question set remain as durable assets for future retrieval exploration, which the user intends
to continue.

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
- `results_stage0_chunks.json`, `results_stage1_rerank.json` (superseded),
  `results_stage2_header_boost.json` (rejected), `results_stage3_bgem3.json` (rejected),
  `results_stage4_context_pilot.json` (rejected, reverted) — raw results for the second RoA
  improvement round
- `generate_chunk_context.py` — the stage-4 contextual-embedding pilot script (kept for
  reference/reuse; not part of the active pipeline since the pilot was rejected)
- `results_stage_colbert.json` (current production) — raw results for the literature-grounded
  round's ColBERT reranker swap
- `results_stageA_facets.json`, `results_stageA_facets_v2.json`, `results_stageA2_soft_facets.json`
  (all rejected) — raw results for the facet-filtering retry (hard, hard+regex-fix, soft)
- `eval/sweep_fusion_weights.py`, `eval/sweep_fusion.log` — the fast retrieval-only weighted-fusion
  sweep (Stage F); no full 80-turn results file since no config beat RRF enough to warrant one
- `results_stageG_pseudo_query.json` (rejected — net-zero wash) — raw results for the
  deterministic pseudo-query index; `build_pseudo_query_index.py`/`src/pseudo_query.py` build and
  query it, gated by `PSEUDO_QUERY_ENABLED` in `src/rag.py`
- `results_stageH_crag_verification.json` (rejected) — raw results for CRAG-style retrieval
  verification, gated by `CRAG_VERIFICATION_ENABLED`
- `results_stageD_splade.json` (rejected) — raw results for the SPLADE third retrieval channel;
  `build_splade_index.py`/`src/splade.py` build and query it, gated by `SPLADE_ENABLED`
- `results_stageE_embedding_ensemble.json` (rejected) — raw results for the nomic+bge-m3
  embedding ensemble; `src/ensemble.py` queries the existing `policies_bge-m3` collection, gated
  by `EMBEDDING_ENSEMBLE_ENABLED`
- `EXPERIMENTS.md` — exact parameters and headline metrics for every pass, for fast comparison
  and reverting via git if a future change regresses
- `run_eval.py`, `score_summary.py`, `generate_questions.py` — the eval harness itself, reusable
  for future re-evaluation after any further changes (both now accept a question-set path as a
  CLI argument, so a third set doesn't require duplicating either script)

# LLM-experiments phase: judge upgrade + generator bake-off (2026-07-20)

## Judge upgrade

`qwen2.5:7b-instruct` both generated and judged answers in every eval to date - a self-judging
risk. Re-scored the existing `stage_colbert` and `j6_disclose_ambiguity` results with
`qwen2.5:14b-instruct` as an independent judge, regenerating nothing (`eval/rejudge.py`).

| Group | 7B judge | 14B judge | Delta |
|---|---|---|---|
| Policy | 3.98 | 4.15 | +0.18 |
| **RoA** | **3.80** | **3.48** | **-0.32** |
| RoA misses only | 3.33 | 2.67 | **-0.66** |
| RoA hits only | 4.00 | 3.82 | -0.18 |

Not a uniform rescale: the 7B judge specifically over-credited RoA wrong-sibling boilerplate
answers (justifications show it catching genuine factual contradictions the 7B judge missed,
e.g. "contradicts the reference by incorrectly defining a capped mark" on an answer 7B scored
3/5). The true policy-vs-RoA answer-quality gap is ~4x wider than previously reported (0.18 vs
0.67). `JUDGE_MODEL = "qwen2.5:14b-instruct"` is now the standard judge for all future evals
(`eval/run_eval.py`); comparing scores across the switch requires re-judging, not just re-running.

## Generator bake-off

Tested `qwen2.5:14b` and `llama3.1:8b` as CHAT_MODEL replacements, each judged independently by
`qwen2.5:14b` where possible. First pass conflated the generator swap with an unintended
contextualizer swap (`CHAT_MODEL` was used for both roles) - `CONTEXTUALIZE_MODEL` was split out
as its own constant in `src/llm.py`, pinned to the validated `qwen2.5:7b-instruct`, so later
passes test generation in isolation.

| Metric | Production (7B gen) | llama3.1:8b (indep. judged) | qwen2.5:14b (self-judged, caveat) |
|---|---|---|---|
| Overall hit@6 | 83.8% | 82.5% | 82.5% |
| Overall answer | 3.98 | 3.84 | 4.00 |
| RoA answer | 3.70 | 3.58 | 3.67 |
| Follow-up hit@6 | 82.5% | 80.0% | 80.0% |

**Both rejected.** `llama3.1:8b` is cleanly, independently judged worse across the board -
higher keyphrase coverage (+2.9pp) but lower holistic answer quality, suggesting it states more
raw figures without getting them more consistently right. `qwen2.5:14b` looks best but is
self-judged (it was also `JUDGE_MODEL` for that pass) - given the judge-upgrade finding above
already proved self-judging bias as large as +0.3 on RoA specifically, this number isn't
trustworthy without an independent judge stronger than 14B, which isn't practical on this
hardware. `qwen2.5:7b-instruct` remains `CHAT_MODEL`; `qwen2.5:14b-instruct` stays installed only
as `JUDGE_MODEL` (not used in the live app - only invoked by `eval/run_eval.py`).

Deferred next step, not yet attempted: retrying the "quote figures verbatim" prompt rule (J7,
rejected under the 7B generator) under a stronger generator, since the mechanism needing a more
capable model to follow the instruction was J7's own stated hypothesis.

# Code review round (2026-07-20)

Full read-through of `src/` end to end (`rag.py`, `ingest.py`, `lexical.py`, `rerank.py`,
`docid.py`, `doc_index.py`, `memory.py`, `app.py`, `llm.py`, `splade.py`, `ensemble.py`,
`pseudo_query.py`, `reembed.py`, `run_ingest.py`).

**Fixed:**
- **Critical**: `CHAT_MODEL` was still `qwen2.5:14b-instruct` in production - the bake-off's
  revert-to-7B decision was stated but never applied to code. Live traffic had been running on
  the unproven 14B generator since the bake-off concluded. Reverted and restarted.
- `src/memory.py`: schema creation + the `summarized_through` ALTER TABLE migration (wrapped in a
  try/except) ran on every single DB connection - every message send, every history fetch - for
  the life of the process. Now runs once per process.
- `src/ingest.py`: `upsert_document`/`delete_document` fetched full documents+metadatas via
  `collection.get()` just to read `ids` for deletion. Added `include=[]`.
- `src/rag.py`: `degree_length`/`award_type` were extracted from every query regardless of
  whether `FACET_PREFERENCE_ENABLED`/`SPLADE_ENABLED` (both off) would ever consume them. Now
  skipped when neither is on.
- `src/doc_index.py`: its BM25 cache had no staleness check against the corpus version marker
  (unlike `src/lexical.py`'s equivalent) - would have silently served a stale identity index
  forever if `DOC_ROUTING_ENABLED` were ever reactivated after a re-embed. Fixed to match
  `lexical.py`'s pattern.
- `src/splade.py`: documented (not auto-fixed - a ~105-minute offline rebuild shouldn't happen
  silently) that its index has no staleness check; re-run `build_splade_index.py` by hand after
  any re-embed if `SPLADE_ENABLED` is ever reactivated.
- `src/ensemble.py`: documented that its `bge-m3` Ollama model dependency was removed during a
  disk cleanup this session - reactivating `EMBEDDING_ENSEMBLE_ENABLED` now needs `ollama pull
  bge-m3` first, or `query()` fails immediately.

All fixes verified behavior-preserving: full module import check, live `retrieve()`/`answer()`
smoke test, and a 10-question regression check comparing fresh retrieval output against the
stored `stage_colbert` baseline - 0 mismatches.

**Not changed**: the ~11 experimental flags accumulated in `retrieve()`/`answer()` (one per
reverted stage) add real reading complexity to the hot path, but restructuring them into a
cleaner extension-point pattern is a larger, riskier change than this pass's fixes - proposed to
the user as a separate decision rather than done unprompted.

**New ideas surfaced for RoA retrieval, not yet attempted:**
1. **Precompute and cache ColBERT document-side (token) embeddings.** `src/rerank.py`'s
   `_rerank_colbert()` currently re-encodes all ~30 pool documents from scratch on *every single
   query* - the same chunk gets re-embedded by the BERT model again and again across different
   queries, pure waste (production ColBERT/PLAID systems precompute document embeddings once,
   only encoding the query at search time). Fixing this is also the natural stepping stone to:
2. **ColBERT as a genuine first-stage retrieval channel, not just a reranker.** J0's diagnostic
   found 4 of 12 misses were never even in the dense+BM25 candidate pool to begin with - no
   reranker can rescue a document reranking never sees. ColBERT's token-level MaxSim was the
   single biggest win this project found (RoA 60%->70%) precisely because it can discriminate on
   the identity terms in `chunk_header` that dense pooling washes out; extending it to score
   against the full corpus (or a much wider candidate set) rather than only reranking whatever
   dense+BM25 already surfaced could directly address the out-of-pool miss class. Requires #1's
   caching to be computationally practical.
3. **Show J1 identity data in the LLM's answer context, not in embeddings or retrieval.**
   J2 (embedding-time) and J3 (retrieval-time) both failed by touching retrieval; neither touched
   the one place identity data can't perturb retrieval at all - the context block
   (`_format_context()`) shown to the answering model *after* retrieval is already done. Adding
   the target document's own `programme_name`/`department`/`partner_institution`/`aliases` there
   is zero-retrieval-risk and could sharpen both the J6 disclosure's specificity (name the actual
   differentiator, e.g. "3yr vs 4yr", not a generic "tell me the programme") and general answer
   precision (the deferred LLM-phase's keyphrase-coverage goal).
4. **Targeted rerank-pool widening.** J0b widened `RERANK_POOL_SIZE` globally (30->100) and lost
   more than it gained (noise on the ~90% of queries that didn't need the extra depth). A
   version gated on `_top_family_count` already looking fragmented at 30 - i.e. only pay the
   depth cost on the specific queries where the right document plausibly isn't in the shallow
   pool - could isolate the 2-rescue benefit without the 5-turn cost.

## Following up on the code review's 4 ideas (2026-07-20/21)

| Idea | Result |
|---|---|
| 4: targeted rerank-pool widening | **Rejected** - worse than J0b's naive global widening (0 rescues / 4 losses vs 2/5) |
| 3: identity data in answer context | **Rejected** - net negative on RoA specifically |
| 1: cache ColBERT doc embeddings | In progress - see below |
| 2: ColBERT as first-stage retrieval | In progress - see below |

**Idea 4 (targeted rerank-pool widening) - rejected.** Gated `RERANK_POOL_SIZE`'s widening
(30->100) on `_top_family_count` already looking fragmented at the shallow depth, hoping to
isolate J0b's 2-rescue benefit without its 5-turn cost. Result was worse than J0b's blunt global
version: **0 rescues, 4 losses** (RoA hit@6 70%->60%, MRR 0.450->0.388), all losses on follow-up
turns. The pre-rerank family-fragmentation signal doesn't correlate with "the right document is
deeper in the pool" - it fired on queries where extra depth only added noise, and never once on
an out-of-pool case. `TARGETED_WIDENING_ENABLED` reverted to `False` in `src/rerank.py`.

**Idea 3 (J1 identity data in answer context) - rejected.** Added the retrieved document's
`programme_name`/`partner_institution`/`aliases` to `_format_context()`'s per-chunk header and
sharpened the J6 disclosure to name the actual differentiator when available - all strictly
post-retrieval, so (confirmed) retrieval itself barely moved (1 turn lost, within the established
noise band). Mixed answer-quality result that looked positive in aggregate but wasn't where it
mattered: overall/policy answer score rose (3.89->3.95, 3.98->4.25), but policy documents rarely
have identity data populated, so that's likely noise from a feature that barely engages there.
**RoA - where it actually fires - moved the wrong way on both quality metrics together**: answer
score 3.80->3.65, keyphrase coverage 55.2%->53.4%. Extra identity fields in the context block
appear to add clutter the 7B generator doesn't parse more precisely, rather than sharpening it.
`IDENTITY_CONTEXT_ENABLED` reverted to `False` in `src/rag.py`; worth retrying if the deferred
stronger-generator phase changes the outcome.

**Ideas 1+2 (cache ColBERT embeddings + first-stage retrieval) - implementation.** PyLate ships
a full multi-vector retrieval stack (`indexes.Voyager`, an HNSW-based persistent index - already
installed, no new dependency; `retrieve.ColBERT`, ANN token search + exact MaxSim rerank over the
retrieved candidates) rather than needing this built from scratch. One offline-built index now
serves both ideas:
- `build_colbert_index.py` encodes every chunk (header+text, matching `src/rerank.py`'s existing
  `_passages()` convention) once and persists to `data/colbert_index/`.
- `src/colbert_index.py`'s `query()` does first-stage ANN retrieval over the FULL corpus (Idea 2,
  gated by `COLBERT_FIRST_STAGE_ENABLED` in `src/rag.py`), targeting the out-of-pool miss class
  J0 found that no reranker can rescue.
- The same module's `get_cached_embeddings_by_meta()` looks up a candidate's precomputed
  embedding by `(source_url, chunk_index)` - a key that survives `src/rag.py`'s whole fusion/
  dedup pipeline unchanged, unlike a raw Chroma id, which doesn't. `src/rerank.py`'s
  `_rerank_colbert()` now uses this instead of unconditionally re-encoding every candidate's text
  from scratch on every query (`USE_CACHED_COLBERT_EMBEDDINGS`, on by default) - falls back to
  fresh encoding per-candidate when the index isn't built yet or a candidate isn't in it, verified
  by inspection to be logically identical to the old unconditional encode call in that case (all
  entries fall back, in original order).

**Idea 1 (cache ColBERT doc embeddings) - kept.** Verified correct by inspection (fallback path
for uncached candidates is logically identical to the old unconditional-encode call) and by a
live 10-question regression check against the stored `stage_colbert` baseline - 0 mismatches.
Measurably beneficial: ~47% reduction in average retrieval latency (1.69s vs 3.21s average).
Pure efficiency win, no retrieval-quality tradeoff, so kept on unconditionally
(`USE_CACHED_COLBERT_EMBEDDINGS = True` in `src/rerank.py`) independent of Idea 2's outcome.
Note this saving is scoped to the reranking step alone - it doesn't move the needle on total
per-question eval wall-clock time, which is dominated by the contextualize/generate/judge LLM
round trips (each 80-turn eval question now runs 150-350s end to end; the ColBERT
retrieval/rerank step inside that is single-digit seconds even before caching).

## Idea 2 (ColBERT first-stage retrieval) - implemented, evaluated, rejected (2026-07-21)

Machine transfer (M1 -> M1 Pro) picked this up mid-flight: `COLBERT_FIRST_STAGE_ENABLED = True`
was already set but the eval hadn't been re-run since a bugfix. First eval attempt after the
transfer hit `queryEf must be equal to or greater than the requested number of neighbors` on
40/40 turns - `query()`'s over-fetch (`n_results * 6`, up to `n_results=48` in production ->
k=288) exceeds Voyager's constructor default `ef_search=200`, and Voyager's underlying HNSW
search requires `ef_search >= k`. `ef_search` is a per-instance query-time knob
(`pylate/indexes/voyager.py`: stored as `self.ef_search`, only read in `__call__`'s
`query_ef=self.ef_search`), not baked into the persisted graph, so safe to raise without
rebuilding the index. Fixed with `EF_SEARCH = 400` in `src/colbert_index.py` (comfortable
headroom over the 288 max, not tied exactly to it so a future pool_size bump doesn't silently
reopen this).

Verified the fix with two 10-question retrieval-only regression checks (no live server, direct
`src.rag.retrieve()` calls) before committing to a full run: 10 policy questions (0/10 hit@6
changes vs `stage_colbert`) and, since the first check's sample happened to be entirely policy
documents (already 100% hit@6, so it couldn't show a RoA gain either way), a second check
targeting the first 10 RoA questions specifically (also 0/10 hit@6 changes, no crashes). Both
clean - confirmed the `ef_search` fix works and introduces no regressions - but neither sample
showed the target out-of-pool misses being rescued, so inconclusive on whether Idea 2 actually
helps.

**Full 80-turn eval (`idea2_colbert_firststage`) - net regression, rejected:**

| | Policy hit@6/MRR | RoA hit@6/MRR | Overall hit@6/MRR | Answer score |
|---|---|---|---|---|
| `stage_colbert` (baseline) | 100.0% / 0.91 | 70.0% / 0.45 | 85.0% / 0.68 | 3.89 |
| `idea2_colbert_firststage` (rejected) | 100.0% / 0.90 | **65.0% / 0.43** | 82.5% / 0.66 | 3.84 |

RoA - exactly where Idea 2 was meant to help - regressed on both hit@6 (-5pp) and answer score
(3.80->3.55, a real quality drop, not just a retrieval-metric wobble). Policy stayed flat as
expected (Idea 2 only adds a competing channel, doesn't touch the already-saturated policy pool).
Flip analysis: 2 gained / 4 lost (net -2 turns) - gained
`roa-ug-integrated-masters-4yr-year-1.pdf` [follow-up] and
`roa-ug-aegean-omiros-4yr-non-standard-year-1.pdf` [follow-up]; lost `roa-ug-glossary.pdf`
[follow-up], `roa-ug-3yr-year-1-rules.pdf` [follow-up], `pgt-credit-framework-25.pdf` [primary],
`integrated-phd-roa-model-a-25.pdf` [follow-up].

**Root-cause investigation of the 4 losses** (compared exact top-6 URLs and retrieval queries,
baseline vs new, for each): in all 4 cases the correct document was already only marginally in
the pool at baseline (rank 4-6, right at the edge of top-6) - Idea 2 doesn't cause wildly wrong
retrievals, it adds 1-2 more RRF channels that dilute any document only weakly supported by a
couple of existing channels.
- `roa-ug-glossary.pdf` - confounded, not really an Idea 2 effect. The query contextualizer
  produced a materially different rewrite between runs ("Can the Capped Mark exceed 40..." vs
  "Can the capped mark be exceeded by...", dropping the "40" and the exact glossary term) -
  known Ollama non-determinism noise (no `temperature`/`seed` set), not something Idea 2 caused.
- `roa-ug-3yr-year-1-rules.pdf` - genuine sibling over-recall: the new ColBERT channel pulled in
  several "variations" sibling documents (same family, different content) that outcompeted the
  already-marginal (rank 6) correct "rules" document in the fusion.
- `pgt-credit-framework-25.pdf` - clean case, identical retrieval query both runs. The new
  channel surfaced unrelated partner-institution documents (`roa-ug-aegean-omiros-*`, even
  duplicated at ranks 3+4) sharing generic assessment-framework boilerplate language, displacing
  the marginal (rank 6) correct document.
- `integrated-phd-roa-model-a-25.pdf` - clean case, identical query. Traced to a **pre-existing
  `is_current` metadata bug**, unrelated to Idea 2's design: the `pgt-model-1-january-starts-
  rules-of-assessment` document family has multiple editions simultaneously tagged
  `is_current: True` (at least the current `jan-26` edition, correctly, and the superseded
  `jan-25` edition, incorrectly - the latter also mistagged `academic_year_norm: "2025-26"`
  despite living in the `2024-25` URL path). Idea 2's new channel was simply sensitive enough to
  newly surface this already-mistagged sibling, displacing the marginal (rank 4) correct
  document. Filed separately below - fixing it doesn't change Idea 2's verdict (3 of 4 losses are
  unrelated to it).

Same lesson as J0b/Idea 4 (targeted rerank-pool widening, also rejected): adding retrieval depth
or channels rescues out-of-pool misses rarely and dilutes already-fragile marginal hits often -
the dilution cost outweighs the rescue benefit on this corpus. `COLBERT_FIRST_STAGE_ENABLED`
reverted to `False` in `src/rag.py`. Idea 1 (embedding caching) is unaffected and stays kept -
it's a pure latency win independent of whether the first-stage channel is enabled.

### `is_current` metadata bug fix (unrelated to Idea 2, fixed as a follow-up)

Investigated the `integrated-phd-roa-model-a-25.pdf` loss's root cause further since it traced to
a data bug rather than Idea 2's mechanism. Confirmed via direct inspection of
`data/manifest.json` that the `pgt-model-1-january-starts-rules-of-assessment` family had **3** of
its 6 editions simultaneously tagged `is_current: True` (should only ever be 1: the newest).
Root cause: `reembed.py`'s `compute_current_flags()` picks the max-year member per family using
each document's *content-extracted* `academic_year` field - but PGT "January starts" documents
describe the academic year the cohort **finishes in**, not the document's own edition/publish
year, so a superseded `jan-25` edition (filed under `.../pgt/2024-25/...`) had its
content-extracted `academic_year` misread as `"2025-26"` - tying it with the true current `jan-26`
edition and marking both `is_current: True`.

Corpus-wide scan (all 1,188 kept documents, `compute_current_flags()` re-run standalone) found 4
families total with this "≥2 simultaneous `is_current: True`" symptom. Investigated all 4 before
fixing anything, since a broad fix risked being worse than the narrow bug:
- **2 confirmed as the same PGT January-starts content/folder-year mismatch**
  (`pgt-model-1-january-starts-rules-of-assessment`, `pgt-model-2-january-starts-rules-of-assessment`)
  - fixed, see below.
- **`student-engagement-policy.pdf`** - a one-off: the `-2024-25.pdf`-suffixed edition's own
  content-extracted `academic_year` reads `"2025-26"`, contradicting its own filename. No
  generalizable pattern found (isolated extraction anomaly on this one document) - left as-is,
  flagged for manual follow-up if it surfaces in a future eval miss.
- **`roa-ug-northwest.pdf`** - NOT clearly a bug: both editions (`-2022.pdf` covering 2022-23/
  2023-24, `-2017-2021.pdf` covering 2017-2021) live under Essex's literal `/current/` UG-archive
  folder, which unconditionally forces `is_current: True` by design (`"Essex's UG archive reuses
  identical filenames across years"` per `compute_current_flags`'s own docstring). Plausibly
  intentional - different partner-institution cohorts may legitimately follow different-vintage
  rules simultaneously. Left as-is; fixing would risk hiding a legitimately-current document for
  some cohort.

**Considered and rejected a broad fix first**: tried unconditionally preferring the URL's
year-folder over content-extracted `academic_year` wherever they disagree. A corpus-wide scan
found 61 such mismatches - but 52 of them were UG `/previous-years/` archive documents where the
folder-year is consistently *one year ahead* of the content-year by an apparently intentional,
different convention (already harmless, since `/previous-years/` unconditionally forces
`is_current: False` regardless of any year computation) - broadly "fixing" this would have
silently changed a correct, unrelated convention. Narrowed to the 9 mismatches that actually
participate in live `is_current` computation (not already covered by an override); a first
attempt at unconditional preference fixed the 2 target families but **introduced a new bug**:
`part-time-taught-masters-24.pdf` (content year understating its own folder year) tied with the
true current `part-time-taught-masters-25.pdf`, creating a 5th "≥2 True" family that didn't exist
before.

**Shipped fix**: `effective_year()` in `src/docid.py` - `normalize_year()`, capped (never raised)
at the URL's year-folder, scoped to `/rules-of-assessment/pgt/` paths only. One-directional by
design (only ever lowers a document's effective year, never raises it) - matches
`compute_current_flags`'s existing convention that all its path-based overrides only ever force
`False`, never `True`. Verified via a full corpus diff (all 1,188 kept documents, old flags vs
new): exactly 3 flags changed, all `True -> False`, all on the 2 target families, zero collateral
changes elsewhere. Wired into all 3 places that independently computed this value before
(`reembed.py`'s `compute_current_flags()` and `recompute_current_flags()`, `src/ingest.py`'s
`upsert_document()`) so they can't drift apart again, matching `docid.py`'s existing "single
shared definition" charter. Applied live via `reembed.recompute_current_flags()` (metadata-only,
no re-embed needed) - 109 chunks updated across the 3 affected documents. Verified: direct Chroma
query confirms the `jan-25` edition now reads `is_current: False`, `academic_year_norm: "2024-25"`;
a live retrieval check for the original `integrated-phd-roa-model-a-25.pdf` follow-up question now
ranks the target document 3rd (was 4th in the `stage_colbert` baseline, and dropped out of top-6
entirely under Idea 2) with the stale `jan-25` sibling no longer in the pool at all. A 10-question
policy+RoA regression check post-fix showed 0 hit@6 changes vs baseline elsewhere.

### `current_prod_verify` - full-eval validation of today's net changes (= new current production)

Today's session nets out to: Idea 1 (kept) + the `is_current` fix (kept) on top of
`j6_disclose_ambiguity`, with Idea 2 tried and reverted (net zero vs off). Neither Idea 1 nor the
metadata fix had been through a full 80-turn eval on their own - both were only spot-checked
(10-question regression checks, plus the one specific document the metadata fix targeted). Ran a
full eval to get a clean, complete number for what's actually live now, comparing against
`j6_disclose_ambiguity` (the correct prior-production baseline - the earlier Idea 2 comparison in
this document used the older `stage_colbert` checkpoint instead, since that's what the prior
session's Idea 1-4 work was itself measured against; this run reconciles back to the actual
current-production lineage).

| | Policy hit@6/MRR | RoA hit@6/MRR | Overall hit@6/MRR | Answer score |
|---|---|---|---|---|
| `j6_disclose_ambiguity` (prior production) | 100.0% / 0.87 | 67.5% / 0.42 | 83.8% / 0.65 | 3.86 |
| `current_prod_verify` (new) | 100.0% / 0.92 | 62.5% / 0.40 | 81.2% / 0.66 | 3.99 |

Mixed result: hit@6 dipped slightly (RoA -5pp, overall -2.6pp; flip analysis: 2 losses / 0 gains,
both follow-up turns - `roa-ug-3yr-year-1-rules.pdf` and `east15-25.pdf`), but answer quality rose
across the board (3.86->3.99 overall, up in both policy and RoA). Both losses fall within this
project's own previously-documented noise floor (~1-2 turns from Ollama's unset
temperature/seed in the contextualizer, per the `j6_disclose_ambiguity` J-round note above) -
and neither Idea 1 (independently verified retrieval-neutral) nor the `is_current` fix
(independently verified to touch only 3 unrelated documents, neither of which appears in the loss
list) offers a mechanism that would explain a genuine regression here. Read as noise rather than a
real problem, but flagged plainly rather than asserted with more confidence than the data
supports - if a future eval pass shows the same 2 documents losing again, that would upgrade this
from "probably noise" to "worth investigating."

**Operational note from this run**: the server process died mid-run (turns 24-25). Initially
attributed this to an OOM-driven kill under the session's memory pressure, based on circumstantial
evidence (no traceback, no crash report, macOS's unified log produced no definitive signature
either, coinciding with a ~12GB drop in swap usage) - flagged as "consistent with", never
confirmed. **Correction (same day): this was the user manually killing the Ollama server, not a
spontaneous crash.** The RAM/swap pressure observed throughout this session (three concurrent
Ollama models - `qwen2.5:14b-instruct` judge + `qwen2.5:7b-instruct` chat + `nomic-embed-text` - on
a 16GB machine, repeatedly measured near-full via `vm.swapusage`) is still real and independently
measured, just not the cause of *this specific* incident - don't conflate a genuinely tight memory
budget with an unconfirmed causal story for one event. Resumed from the point of failure (a small
ad hoc script re-ran just the 16 failed questions and merged into the existing 24-entry results
file) rather than restarting the full run.

`current_prod_verify` is now the reference "current production" row in `EXPERIMENTS.md`,
superseding `j6_disclose_ambiguity`.

## External code review round 2 (2026-07-21): four independent LLM reviews, verified and acted on

Sent the project (public GitHub repo, full history in this file) to four LLM tools for a second
review round, asking for code review, code/methodology improvement suggestions, a ceiling
assessment, and an opinion on whether a new eval question set was worth building. **One of the
four (DeepSeek) fabricated its entire review** - every file it claimed to read (`src/retrieval/`,
`src/metadata_manager.py`, `eval/harness.py`, `eval/judge.py`, `src/config.yaml`, etc.) does not
exist in this repo; it never actually cloned or read the code and invented a plausible-looking
fictional structure instead. Discarded entirely - none of its file/line-level claims are usable,
though a couple of its generic points happened to overlap with the other three reviews' verified
findings. Grok's and Fable 5's reviews referenced real files and were spot-checked against the
actual code before acting on anything; all checked claims were accurate. Gemini gave strategic/
methodology feedback without file-specific claims, so there was nothing to hallucination-check
there.

### Phase 1: fix eval determinism and the double-retrieval-invocation bug

Grok, Fable 5, and (genuinely, despite the fabricated file citations) DeepSeek all independently
flagged the same root issue: no Ollama call site anywhere set `temperature`/`seed`, so the
project's own documented "~1-2 turn noise floor" wasn't inherent - it was optional. Fable 5 also
found something none of the others caught: `eval/run_eval.py`'s `ranked_retrieval()` called
`retrieve()` a second, independently-sampled time to score retrieval quality, separate from the
`retrieve()` call inside the live app's `answer()` that actually produced the answer - on
follow-up turns (where the contextualizer's rewrite is a real LLM sample) these two calls could
diverge, meaning the eval could score a retrieval that wasn't the one the answer was actually
generated from.

Both fixed together: `src/llm.py`'s `chat()` gained an `options` parameter; `RAG_DETERMINISTIC=1`
pins `temperature=0/seed=42/num_ctx=8192` by default for any call that doesn't pass explicit
options, covering the contextualizer, generator, and judge from one change. `src/rag.py`'s
`answer()` now returns its own `retrieval_query`/`ranked_top_urls` instead of discarding them, the
API surfaces them, and `run_eval.py` scores those directly - one `retrieve()` call per turn, not
two. Also fixed a bug hit firsthand this session: a mid-run server crash had silently dropped
16/40 questions from `current_prod_verify`'s results with no error, just a smaller "Wrote N
results" count nobody would notice without checking. `run_eval.py` now retries once, hard-fails
loudly on a second failure, and asserts the final count matches `len(questions)`.

**Verification, in order of rigor:**
1. Two-question spot check (identical `retrieval_query`, `ranked_top_urls`, and answer text across
   two independent calls to the same question under `RAG_DETERMINISTIC=1`) - passed.
2. 3-question smoke test through the real `eval_one()` - passed.
3. Full 80-turn run (`current_prod_deterministic_run1`).
4. A second full 80-turn run on identical code (`current_prod_deterministic_run2`), diffed
   programmatically against run1 across all 80 turns' answer text, retrieval query, retrieved
   URLs, and judge score: **0 differences.** Full determinism confirmed at scale, not just on a
   handful of spot-checked questions.

**Headline numbers (now the confirmed, noise-free reference)**:

| | Policy hit@6/MRR | RoA hit@6/MRR | Overall hit@6/MRR | Answer score |
|---|---|---|---|---|
| `current_prod_deterministic` (run1 = run2, exact) | 100.0% / 0.87 | 62.5% / 0.40 | 81.2% / 0.63 | 3.84 |

RoA hit@6 (62.5%) exactly matches `current_prod_verify`'s earlier non-deterministic number, not
`j6_disclose_ambiguity`'s 67.5% - initial evidence that 62.5% was already the true, stable value
and the "regression" investigated earlier in this document was substantially a measurement
artifact, not a real code regression.

**But the full story needed one more step.** Diffing the deterministic run directly against
`j6_disclose_ambiguity` (not `current_prod_verify`) found a *different* net -2 (0 gained, 2 lost:
`roa-ug-4yr-year-1-rules.pdf` and `ug-grad-cert-year-1.pdf`, both follow-up turns) than the
non-deterministic comparison had shown. Investigated both properly rather than writing this off as
noise, since "noise" was the exact assumption this work was supposed to stop taking on faith:

- `ug-grad-cert-year-1.pdf`: the deterministic contextualizer rewrote the follow-up as "...can
  still be awarded **the certificate**..."; the old `j6_disclose_ambiguity` run's rewrite (a
  different, non-deterministic sample) said "...awarded **the Graduate Certificate**...". Dropping
  "Graduate" cost exactly the identity-bearing word the corpus needs to disambiguate this document
  from its many siblings (`ug_grad-dip-year-2.pdf`, `ug-grad-dip-year-1.pdf`, etc.) - the target
  fell from rank 3 to out-of-pool entirely.
- `roa-ug-4yr-year-1-rules.pdf`: similarly, the deterministic rewrite's phrasing shifted enough
  that the candidate pool changed completely - `j6_disclose_ambiguity`'s pool was entirely 4-year
  honours-degree siblings (target at rank 3); the deterministic run's pool was unrelated
  integrated-masters/nursing documents, none from the right family at all.

**This refines what "noise floor" actually meant.** It wasn't that hit@6 randomly wobbles - under
a fixed seed it's now proven to be exactly 0. What's real is *seed sensitivity*: a fixed seed
value is still an arbitrary choice, and different fixed seeds can produce different (but each
internally reproducible) contextualizer rewrites for the same follow-up question - some
better-specified, some worse. `seed=42` happened to land on a slightly weaker rewrite than
whatever non-deterministic sample `j6_disclose_ambiguity` happened to draw for these 2 specific
questions. This is a genuinely different, more precise finding than any of the four external
reviews anticipated - they expected fixing determinism to cleanly settle "is the regression real,"
and it did, but it also surfaced that determinism trades one kind of uncertainty (run-to-run
variance) for another (seed-choice sensitivity) rather than eliminating uncertainty altogether.
Not a regression to chase further - the underlying cause (contextualizer rewrite quality on
follow-ups) is a known, already-tracked class of variance, and now it's at least fully traceable
turn-by-turn instead of hand-waved as noise.

`current_prod_deterministic` (run1) is now the reference "current production" row in
`EXPERIMENTS.md`, superseding `current_prod_verify`. `RAG_DETERMINISTIC=1` should be set for any
future eval run intended for headline-number comparison; leave it unset for normal day-to-day use
of the live app (production traffic keeps natural sampling variation - determinism was never about
changing what users experience, only about making evals trustworthy).

### Phase 2 & 3: safe fixes and eval-harness upgrades from the same review round

Verified-and-shipped, all confirmed against real code or corpus-wide data before landing (same
"verify before shipping" discipline as the `is_current` fix):
- `src/colbert_index.py`: `ef_search` now clamps dynamically (`max(current, k)`) per query
  instead of relying on a fixed constant's headroom, closing the risk Fable 5 flagged - a future
  `pool_size` increase could otherwise silently reopen the exact crash `EF_SEARCH=400` was raised
  to fix.
- `reembed.py`: `recompute_current_flags()` now also patches `data/colbert_docs.json` (the
  ColBERT index's frozen metadata snapshot), not just live Chroma - Fable 5 caught that this had
  already drifted stale after the `is_current` fix earlier the same day (109 chunks), since
  nothing previously kept the two in sync. Patched live.
- `eval/score_summary.py` / `eval/score_evidence_sufficiency.py`: fixed to use `effective_year()`
  instead of stale `normalize_year()`, and to key questions by a per-URL queue instead of a flat
  `{source_url: question}` dict that would silently collide if a future set ever has 2+ questions
  on one document.
- `eval/score_summary.py`: evidence-sufficient@6 (J5a) promoted from a one-off diagnostic script
  to a standard column reported alongside strict/lenient hit@6 and answer-score mean/stdev for
  every group - Fable 5's specific point that it's the number tracking what a user actually
  experiences. Verified the integration reproduces the exact previously-documented J5a numbers
  (RoA strict hit@6=70%, evidence_sufficient@6=87.5%) before trusting it.

**Investigated and rejected**: Fable 5's proposed `document_family()` fix (make the year-suffix
separator mandatory, to prevent a hypothetical cross-family merge like "east15"/"east16" siblings
colliding). Corpus-wide audit (same methodology as the `is_current` fix) found this would be a net
regression - Essex's dominant real filename convention is a *bare* 2-digit year suffix with no
separator (`ug-dip-he22.pdf`, `variations22.pdf`, `mlang20.pdf`, confirmed via manifest inspection
to be genuine same-document yearly reissues, not distinct documents) - making the separator
mandatory broke 45 documents' correct family grouping to guard against a case that doesn't
currently exist in this corpus (`east15-25.pdf`/`east15-23.pdf`, the concrete example raised,
already group correctly under the existing regex). Left as-is, documented inline in `docid.py` for
future reference.

### Phase 4, experiment 1: identity-enriched rerank passages - evaluated, rejected

Fable 5's proposed answer to "have we hit a ceiling": J2 (eval/report.md, "Identity-first round")
enriched `chunk_header` with the J1 per-document identity record (programme name, department,
partner institution, awards, aliases) and re-embedded - regressed RoA 70%->60% despite the
identity data itself being locally effective (MRR rose), because re-embedding moved ~450 *other*
documents' embeddings corpus-wide, a side effect of changing indexed text unrelated to whether the
identity data helps. Fable 5's version enriches only the reranker's passage text for the
already-small candidate pool - no embedding in the vector store is touched, so corpus-wide
displacement is structurally impossible, not just "wasn't observed this time." Implemented in
`src/rerank.py` (`_identity_suffix()`, reusing the existing `_load_doc_identity()` helper),
`IDENTITY_ENRICHED_RERANK_ENABLED` flag. Also fixed a stale-cache bug this change would otherwise
introduce (the ColBERT embedding cache is keyed by chunk identity, not content - enriching a
candidate's passage without invalidating its cache entry would score it against the wrong,
pre-enrichment embedding; `_rerank_colbert()` now forces a cache miss only for candidates that
actually got a non-empty suffix).

**Full 80-turn eval - net regression, rejected:**

| | Policy hit@6/MRR | RoA hit@6/MRR | Overall hit@6/MRR | Answer score |
|---|---|---|---|---|
| `current_prod_deterministic` (baseline) | 100.0% / 0.87 | 62.5% / 0.40 | 81.2% / 0.63 | 3.84 |
| `identity_rerank_only` (rejected) | 100.0% / 0.83 | 57.5% / 0.43 | 78.8% / 0.63 | 3.56 |

Flip analysis: 4 gained / 6 lost (net -2). Gained `roa-ug-4yr-year-1-rules.pdf` [follow-up],
`roa-ug-integrated-masters-4yr-year-1.pdf` [primary], `ug-grad-cert-year-1.pdf` [follow-up],
`mscperiodontology_25.pdf` [follow-up] - notably, 3 of these 4 gains are exactly the turns Phase
1's determinism work had identified as real (non-noise) losses vs `j6_disclose_ambiguity`,
suggesting the identity signal genuinely helps some sibling-confusion cases. But lost
`east15-25.pdf` [follow-up], `msc-physiotherapy-25.pdf` [primary], `mba-25.pdf` [primary],
`pgt-credit-framework-25.pdf` [primary], `masters-25.pdf` [primary], and
`integrated-phd-roa-model-a-25.pdf` [primary] (the exact document the `is_current` fix targeted,
hitting cleanly in every run since) - 5 of these 6 losses on primary turns.

**Root-cause investigation** (checked the actual J1 identity records for the lost documents, not
just the retrieval output): enrichment isn't neutral across candidates. `masters-25.pdf` and
`pgt-credit-framework-25.pdf` are generic PGT "framework" documents that don't belong to any one
specific programme - their identity records are thin-to-empty (`masters-25.pdf`'s is just
`{"awards": ["MSc"]}`, everything else blank). Their competitors in the pool
(`mres-gov-25.pdf`: full `programme_name`, `department`, `aliases` including "government MRes")
have rich records. On a generically-worded query ("minimum overall weighted average... to pass a
Master's degree"), the rich sibling's newly-added text picks up semantic proximity to the query
that its old passage didn't have, raising its rerank score - while the correct-but-generic
document gets essentially no boost, since it has almost nothing to enrich with. The correct
document doesn't get pushed down because it's wrong; it gets diluted because it's generic in a
corpus where "generic" and "programme-specific" documents now compete on structurally uneven
footing once enrichment is added.

This is a different failure mechanism than every previous "adding depth/channels dilutes marginal
hits" rejection (Idea 2, Idea 4, J0b) - those diluted via RRF fusion math (more candidate lists,
same document, lower fused rank); this dilutes via asymmetric content enrichment (same candidate
list, but some candidates get more new signal than others, unrelated to relevance). Reverted
(`IDENTITY_ENRICHED_RERANK_ENABLED = False`), mechanism documented inline in `src/rerank.py`. A
real fix would need to gate enrichment on relative fairness within the pool (e.g. only enrich when
multiple pool candidates have identity records, so a lone generic document isn't structurally
disadvantaged) rather than enriching unconditionally whenever data happens to exist - not
attempted this round; the two remaining Phase 4/5 items (home-institution tie-break, multi-turn
conversation probe) were prioritized instead.

### Phase 4, experiment 2: home-institution tie-break - null result, rejected

Fable 5's second proposal: post-rerank, when the final top-k contains both a partner-institution
edition and a home edition of what looks like the same programme, and the home edition currently
ranks worse, promote it - the same species of deterministic, high-precision post-rerank rule as
`_prefer_most_recent_year`, targeting the Periodontology-class sibling-confusion misses J3's
post-mortem specifically named (home vs partner-institution MSc Periodontology identity cards are
"themselves near-identical").

**Verified the proposed detection signal before implementing it**: Fable 5's design used the J1
identity record's `partner_institution` field to detect partner editions, but a corpus-wide check
found only ~63% coverage - the Alexandria periodontology programme's own identity record has this
field blank despite genuinely being a partner edition. Combined it with the URL path
(`/partner-institutions/`, a structural signal Essex's site consistently uses, same category as
the `/previous-years/`/`/current/` overrides `compute_current_flags` already trusts) for full
coverage, and used J1 alias overlap (both the home and Alexandria periodontology documents list
"perio") to detect "same programme" rather than exact `programme_name` matching, since those
strings differ even for genuine home/partner pairs ("MSc Periodontology (36 months...)" vs
"MSc Periodontology Science and Practice"). Implemented in `src/rag.py`
(`_prefer_home_institution()`, `HOME_INSTITUTION_TIEBREAK_ENABLED`), unit-tested against the real
periodontology pair before running the eval - confirmed the home candidate gets correctly promoted
above the partner one when both are present.

**Full 80-turn eval - null result**: 0 gained / 0 lost vs `current_prod_deterministic`, headline
numbers identical to two decimal places on every metric (Policy 100.0%/0.87, RoA 62.5%/0.40,
overall 81.2%/0.63, answer score 3.84). Verified this wasn't a silently-broken no-op by diffing the
*entire* retrieved `top_urls` list (not just hit@6 status) across all 80 turns against the
baseline - 0 differences anywhere, confirming the mechanism never actually fired, not that it
fired and happened to net out neutral. Its precondition (a partner candidate and an alias-sharing
home candidate both present in the *same final top-6*, with home ranking worse) never arose for
any of these 40 questions - even the periodontology test question itself never surfaced the
Alexandria variant into its own final pool, so there was nothing to break a tie on.

Following this project's own precedent (Stage G's "net-zero wash" was reverted despite causing no
measured harm, since added complexity without demonstrated benefit isn't worth keeping): reverted
(`HOME_INSTITUTION_TIEBREAK_ENABLED = False`). Correctly implemented and unit-verified against a
real confusable pair - simply unproven on this specific 40-question eval set, not disproven as an
idea. Would need either a broader/differently-constructed question set that actually surfaces
partner-institution ambiguity in its top-6 candidates, or a corpus-wide audit of how often
partner/home pairs co-occur in real retrieval pools at all, to get a real read on this mechanism's
value - neither attempted this round.

## Phase 5: scripted multi-turn conversation probe

Fable 5's fourth question-set proposal (this project's other 3 sets are all single-topic, 2-turn
primary+follow-up): 8 scripted conversations (`eval/questions_set4_multiturn.json`, 30 turns
total), grounded in 8 real documents (3 policy, 5 RoA) with ground truth reused from
`eval/questions.json`, covering clean topic switches, switch-then-explicit-return,
cross-document comparison, 3-topic return-to-first, return-to-middle-of-three (the specific
ambiguity class the real live contextualizer-drift bug hit), same-vocabulary cross-family
switching, rapid single-turn switching, and deep-then-distant-return. `eval/run_multiturn_eval.py`
runs each conversation sequentially against the live API (real conversation memory/history, not a
simplified stand-in), scoring hit@6 per turn and logging the contextualizer's actual
`retrieval_query` for every turn rather than automating an unvalidated "faithfulness" classifier.

**Result by turn type** (30 turns, `RAG_DETERMINISTIC=1`):

| Turn type | n | hit@6 |
|---|---|---|
| new_topic | 8 | 100.0% |
| switch | 11 | 100.0% |
| comparison | 1 | 100.0% |
| continuation | 5 | 80.0% |
| return | 5 | 80.0% |

**Topic switching itself is flawless** (19/19 clean, including a long-distance return past 2
intervening switches and a return specifically to the *middle* of three prior topics - the exact
ambiguity class the historical live bug hit). The 2 misses are both in `mt8_deep_then_distant_return`
and are two genuinely different failure mechanisms, not one:

- **Turn 3** ("What happens if a student exceeds that trailing credit limit?"): the contextualizer's
  rewrite is fully correct - `"What happens if a student exceeds the 30-credit trailing limit for
  the MSc Periodontology or Postgraduate Diploma in Periodontology programmes?"` - but retrieval
  still misses. A genuine retrieval-pipeline gap (this specific procedural "what if exceeded"
  scenario isn't well-matched against the corpus text), not a contextualizer problem.
- **Turn 5** ("Back to the very first thing I asked about the credit limit - which department
  administers that programme?"): a real, precisely-diagnosed bug, confirmed by reproducing the
  exact contextualizer call standalone. The LLM's raw rewrite attempt was actually **correct**:
  `"Which department administers the MSc and Postgraduate Diploma in Periodontology programme?"`
  - but `src/rag.py`'s `_is_faithful_rewrite()` guard rejected it (27% content-word overlap with
  the original, just under its 30% threshold), falling back to the raw unresolved question. That
  raw text - dominated by pure referential scaffolding ("back", "very", "first", "thing", "asked",
  "credit", "limit" - 7 of 11 content words, none of them the actual topic) - then retrieved and
  generated an answer about a completely unrelated document ("MRes Government programme's rules...
  administered by the Government department"), reproducing the same class of wrong-document
  hallucination as the original live-reported bug (`eval/report.md`, "postfix3 -> postfix4"), just
  via a different mechanism (guard-rejects-a-good-rewrite, not contextualizer-echoes-a-different-
  question).

**Root cause of the guard's gap**: `_is_faithful_rewrite()`'s content-word-overlap heuristic
assumes "a faithful rewrite keeps most of the original's content words" - true for the failure
mode it was built to catch (hijacking to an unrelated question), but structurally false for a
distant "going back to the very first/distant thing" reference, where a *correct* rewrite must
drop nearly all of the referential scaffolding and substitute in the real topic name, guaranteeing
low literal overlap with the original by construction. The guard can't currently distinguish
"rewrite dropped the original's words because it hijacked to something unrelated" from "rewrite
dropped the original's words because it correctly resolved a heavy reference" - both look
identical under a pure overlap-ratio metric.

**Not fixed this round** - a real fix (e.g. weighting the overlap check by whether the *dropped*
words were referential/scaffolding vs topical, or checking whether the rewrite's added content
appears verbatim earlier in the conversation transcript rather than being novel) needs the same
corpus-wide-safe verification discipline as every other change this session, and this was found
via the newly-built probe rather than something with an existing broader validation harness to
lean on. Flagged as a known, precisely-diagnosed gap for a future session rather than patched
reactively off one example.

This is exactly the class of finding Fable 5 predicted this probe would surface that the existing
2-turn question sets structurally cannot - not a retrieval-signal-engineering problem at all, but
a measurement gap (single-follow-up sets never construct a reference distant enough to hit this
guard's blind spot).

## Follow-up items 1-4 (2026-07-21, continuation)

Four open items after Phases 1-5: (1) fix the `_is_faithful_rewrite()` distant-reference gap;
(2) investigate the genuine retrieval-pipeline miss; (3) the deferred architectural direction
(hierarchical retrieval / boilerplate dedup); (4) a third reviewer round.

### Item 1: faithful-rewrite fix - KEPT (pure improvement, zero regression)

Excluded conversation-reference scaffolding (`_REFERENTIAL_WORDS`: "back", "earlier", "first",
"asked"...) from the original's word set in the overlap check. Verified the exact Phase 5 failing
case now passes at 60% topical overlap (was 27%), a simulated hijack still rejects at 0% (the fix
doesn't weaken the guard's actual purpose), and a normal follow-up still passes. Re-running the
multi-turn probe showed the fix delivers a real answer-quality win the strict metric hides: the
guard now accepts the correct rewrite instead of falling back to the raw question, turning a
wrong-document hallucination ("MRes Government programme... Government department") into the
factually correct answer ("Postgraduate Diploma Periodontology... Health and Social Care
department"). Strict hit@6 on that turn is still a miss, but only because of the separate
sibling-confusion problem in items 2/3, not the contextualizer.

**Full 80-turn regression eval (`faithfulfix_regression`, `RAG_DETERMINISTIC=1`)**: 0 gained / 0
lost vs `current_prod_deterministic`, headline numbers byte-identical (Policy 100.0%/0.87, RoA
62.5%/0.40, overall 81.2%/0.63, answer score 3.84). Critically, **0/40 follow-up retrieval queries
differ** from baseline - on the standard 40-question set the guard never once behaved differently,
because none of its follow-ups are distant-reference questions that hit the blind spot. So the fix
is a pure improvement: measurably helps the distant-reference case (a real hallucination-to-correct
-answer flip on the multi-turn probe), provably zero effect on everything else. This is itself the
point - the bug lives in a conversational regime the single-follow-up sets structurally never
construct, which is exactly why building the multi-turn probe (Phase 5) was necessary to find it.
Kept in production - the first substantive retrieval-path change this session to survive its eval
(every Phase 4 experiment was reverted), and it survived precisely because it's a targeted
correctness fix for a real bug, not a speculative retrieval-signal tweak.

### Item 2: the genuine retrieval-pipeline miss - it's the same sibling confusion

The Phase 5 turn-3 miss ("What happens if a student exceeds that trailing credit limit?") had a
fully-correct contextualizer rewrite yet still missed. Diagnosed it directly: the correct query
retrieves periodontology documents, but the WRONG ones - the Alexandria partner-institution MSc
variants and the PG Dip crowd the home `mscperiodontology_25.pdf` out of the top-6 entirely. It's
the same home-vs-partner sibling confusion the Phase 4 tie-break targeted, except here the home
document falls out of the candidate pool completely (not merely ranks below the partner), which is
precisely why that tie-break couldn't have helped it - there was no home candidate in the pool to
promote. The retrieval failure and the Phase 5 turn-5 residual miss are the same underlying
problem: when a query names a programme generically ("Periodontology"), the near-identical
sibling/partner editions are collectively closer than any single one, and the intended home
edition doesn't reliably survive into the pool.

### Item 3: architectural direction - quantified, and dedup is the WRONG lever

Multiple reviewers proposed boilerplate deduplication (collapse identical chunks to one vector with
multiple source URLs). Measured whether it applies here:
- **31.7% of all chunks (6,508 of 20,498) are exact duplicates** (whitespace/case-normalized) of
  another chunk's BODY text, spanning 2-12 source documents each - the boilerplate problem is real
  and large.
- **But only 0.1% (12 chunks) are duplicates of the actually-EMBEDDED text** (header+body). The
  per-document `chunk_header` - prepended at embedding time - makes each vector distinct even when
  the body is identical boilerplate.

So the vectors are already ~not duplicated; the "duplication" lives only in the stored body, and
the system deliberately keeps siblings distinct via headers. **Naive dedup would be actively
harmful** - collapsing 6,508 chunks to shared vectors would strip exactly the header signal that
is the only thing letting retrieval tell siblings apart at all. The problem this corpus has isn't
redundant vectors to remove; it's that the header identity signal is too WEAK to reliably rank the
right sibling on top - a different problem with a different (non-dedup) set of levers.

Mapped where the current RoA misses actually live (15 miss turns; primary-turn pool checks are
exact, follow-up pool checks approximate since they omit conversation history):
- **~5 out-of-pool**: gold document absent from even a 48-candidate fused pool. Only
  document-level routing (surface the right document BEFORE chunk retrieval) could rescue these.
- **~9 in-pool but ranked deep** (ranks 8-76): present but not pulled into the top-6. A stronger
  reranker/discriminator could rescue these - but J0b/Idea 4 already showed global rerank-pool
  widening hurts more than it helps.
- **~1 lost in reranking**: in the top-6 of raw fusion, dropped by the reranker.

**Honest conclusion**: the remaining headroom is NOT dedup, and NOT another RRF channel (four
rejected this session, ~20 before). It splits between (a) genuinely underspecified questions with
no exploitable signal (unfixable at retrieval), and (b) a hard sibling/partner-discrimination
problem where the home edition either falls out of the pool or ranks below near-identical variants.
The one architectural lever not yet tried in a HARD form is document-level macro-routing (restrict
chunk retrieval to the top-K identity-routed documents) - but J3's SOFT version (routing prior as
an extra RRF list) already failed (0 rescues / 3 losses), and a hard version has an obvious
unrecoverable failure mode (wrong routing = guaranteed miss with no fallback). The smallest
de-risking experiment before any build: measure document-level routing precision in isolation - for
the ~5 out-of-pool misses, does an identity-only query over the document index (src/doc_index.py)
even rank the correct document top-K? If it can't, hard routing can't help and the honest answer is
this corpus is at its retrieval ceiling; the remaining gains are generation-side or UX
(clarification on underspecified queries), not retrieval.

### Item 4: third reviewer round

Prepared a follow-up prompt (results of acting on round 2 + the two open failure modes + the
revisited ceiling question + the specific dedup tradeoff) to send back to the genuine reviewers -
see the session for the text.

### Phase A re-baseline: the first NET-POSITIVE retrieval result since the original hybrid/ColBERT gains

Data hygiene (A1 rename-split + A2 hub removal + A3a alpha<->digit token split), full 80-turn
deterministic eval vs `current_prod_deterministic`:

| | Policy hit@6/MRR | RoA hit@6/MRR | RoA evid@6 | Overall hit@6 | Answer score |
|---|---|---|---|---|---|
| `current_prod_deterministic` | 100.0% / 0.87 | 62.5% / 0.40 | 82.5% | 81.2% | 3.84 |
| `hygiene_A1A2A3a` (new prod) | 100.0% / 0.87 | **65.0% / 0.44** | **85.0%** | **82.5%** | **3.90** |

Flip: 2 gained / 1 lost, **net +1 turn**. This matters out of proportion to its size: every retrieval
experiment across three sessions AFTER the original hybrid-retrieval + ColBERT-reranker wins (Idea 2,
identity enrichment, home-institution tie-break, SPLADE, ensembles, facets, routing, decomposition,
pool-widening - ~20 in total) was a regression, wash, or null. The data-hygiene fixes are the first
thing since to move RoA hit@6 UP - direct confirmation of round-3's central thesis (the remaining gap
sat in the DATA, below the retrieval architecture, not in the retrieval signal). Gains: the two hard
sibling-confusion follow-ups `roa-ug-integrated-masters-4yr-year-1` and `ug-grad-cert-year-1` -
exactly the class stale-edition pollution was hurting.

**The 1 loss is diagnosed and is NOT a hygiene defect** - it's a C1 test case. `east15-25` [follow-up]
regressed rank 2 -> out-of-pool, but the retrieval query itself is the cause: the baseline follow-up
rewrite kept the identity anchor ("...at East 15 Acting School's Masters degree programs?"), the new
one DROPPED it ("...non-core taught modules?"). Under determinism this can only happen if the
contextualizer's INPUT changed - and it did: A3a's token-split slightly reordered the primary turn's
pool, shifting the primary ANSWER, which changed the follow-up's conversation history, which produced
a rewrite that lost "East 15". This is the identity-token-loss / accretion failure mode - exactly what
Fable 5's proposed alias-anchor guard (C1, next) targets: re-append the active J1-alias when a rewrite
drops it. So the east15 follow-up becomes a concrete C1 acceptance test alongside the seed-sensitivity
turns.

**Confirmed A3b is needed** (deferred glued-title fix): periodontology's follow-up still misses
identically to baseline (`mscperiodontology_25` primary-hit / follow-up-miss unchanged) - A1 removing
the stale Alexandria-24 edition was not enough; the home doc is still lexically invisible on
"Periodontology" (the glued-alpha token case A3a can't split). A3b (audited title-repair + targeted
re-embed of the ~3 genuinely-glued stems: mscperiodontology, mscinursing, pgcertpwp) is the next
retrieval step. csee's 2025 doc also still misses despite A1 demoting its 2024 sibling - same lexical
class.

### Phase A complete (A3b glued-title + Alexandria paren-split): RoA 62.5% -> 67.5%

Full 80-turn deterministic eval, cumulative vs the pre-hygiene baseline:

| | pre-hygiene `current_prod_deterministic` | full Phase A `hygiene_A3b` |
|---|---|---|
| RoA hit@6 / MRR | 62.5% / 0.40 | **67.5% / 0.45** |
| RoA evidence@6 | 82.5% | **87.5%** |
| Overall hit@6 / evid@6 | 81.2% / 91.2% | 83.8% / 93.8% |
| Answer score | 3.84 | 3.90 |

**Net +2 turns (3 gained / 1 lost).** A3b + the Alexandria paren-split addendum added the third gain
(mscperiodontology follow-up) on top of A1+A2+A3a's two, with zero new losses. The three gains are
precisely the hard sibling-confusion follow-ups that stale-edition pollution and lexical-
invisibility were hurting: roa-ug-integrated-masters-4yr-year-1, ug-grad-cert-year-1,
mscperiodontology_25. Evidence-sufficient@6 87.5% is back to the original J5a level, now under the
honest deterministic + single-invocation eval regime.

Perspective: across three sessions, ~20 retrieval-SIGNAL experiments after the original hybrid+
ColBERT wins produced zero net RoA improvement (all regression/wash/null). The Phase A data-
HYGIENE programme - stale-edition family-split correction, hub-page removal, and lexical-visibility
repair, none of which touches the retrieval model or adds a channel - delivered +5pp RoA hit@6.
This is the round-3 thesis vindicated end to end: the remaining gap lived in the DATA, below the
architecture. `hygiene_A3b` is the new production baseline.

The one outstanding loss (east15 follow-up, rank 2 -> out-of-pool) is unchanged and remains a C1
acceptance test: A3a's reordering shifted the primary answer, cascading into a follow-up rewrite
that dropped the "East 15" anchor - the identity-token-loss mode the alias-anchor guard (C1, next)
is designed to catch.

### Phase C1 (alias-anchor guard): RoA 67.5% -> 70.0%, and the round-3 programme lands at net +3 / 0 losses

Full 80-turn deterministic eval. C1 recovers the one outstanding Phase-A loss (east15 follow-up,
out-of-pool -> both turns hit, answer score 5) with ZERO new losses or regressions - the guard fired
only where intended.

Cumulative arc of the entire round-3 data-hygiene + guard programme, vs the pre-hygiene
`current_prod_deterministic` baseline:

| | pre-hygiene | A1+A2+A3 (hygiene_A3b) | +C1 (c1_anchor) |
|---|---|---|---|
| RoA hit@6 | 62.5% | 67.5% | **70.0%** |
| RoA MRR | 0.40 | 0.45 | 0.44 |
| RoA evidence@6 | 82.5% | 87.5% | **87.5%** |
| Overall hit@6 | 81.2% | 83.8% | **85.0%** |
| Overall evidence@6 | 91.2% | 93.8% | **93.8%** |
| Answer score | 3.84 | 3.90 | **3.92** |

**Net +3 turns, 0 losses.** The three gains - roa-ug-integrated-masters-4yr-year-1, ug-grad-cert-year-1,
mscperiodontology follow-ups from hygiene; plus east15 follow-up from C1 (the hygiene loss, now
recovered) - are all the hard sibling-confusion / identity-anchor turns the round-3 defects were
hurting. RoA hit@6 70.0% matches the historical stage_colbert peak, but that peak was measured under
the OLD non-deterministic + double-retrieval-bug eval; this 70.0% is under the honest deterministic,
single-invocation regime, so it's a genuine advance, not a re-tie.

The headline of the whole multi-session effort: ~20 retrieval-SIGNAL experiments after the original
hybrid+ColBERT wins produced zero net RoA gain (all regression/wash/null). The round-3 DATA-layer
programme - stale-edition family-split correction (A1), hub-page removal (A2), lexical-visibility
repair (A3a/A3b), and a deterministic identity-anchor guard (C1), none of which adds a retrieval
channel or touches the embedding model - delivered +7.5pp RoA hit@6 with zero losses. The gap was in
the data and in conversational identity-tracking, below the retrieval architecture, exactly as the
round-3 reviews (Fable 5's log-level analysis especially) argued. `c1_anchor` is the new production
baseline.

### C1 switch-safety validation (multi-turn probe)

Re-ran the 30-turn multi-turn probe with C1 on. **Switch-safety confirmed: 11/11 switches hit@6
(100%), plus 8/8 new_topic and 1/1 comparison** - C1 never once appended a stale anchor to a query
that named its own topic, exactly the design guarantee. Return 4/5 and continuation 4/5 (same
totals as before C1).

Investigated the two return-category turns that changed vs the pre-C1 (post-faithfulfix) probe,
and NEITHER is attributable to C1:
- conv8 turn5 (distant return) rescued (miss -> rank 5): identical rewrite both runs; the rescue is
  A3b making mscperiodontology lexically retrievable, not C1 (the rewrite already named
  periodontology, so C1 didn't fire).
- conv6 turn3 (ambiguous "does a capped mark apply to that Year One reassessment") regressed
  (rank 4 -> miss): both rewrites already name "Four-Year Honours Degree", so C1 never fired here
  either. The contextualizer itself produced a subtly different rewrite ("what does Capped Mark
  MEAN..." leaning to the glossary definition vs "does a capped mark APPLY to reassessment..."
  leaning to the 4yr rules) because upstream answer changes shifted its history - the same
  accretion/seed-sensitivity drift documented earlier, orthogonal to C1.

Net: C1 is switch-safe and causes no harm on the multi-turn set; combined with the 80-turn result
(east15 recovered, 0 losses), it's validated and kept.

## Phase B: offline backtests to decide (before building) whether any architectural lever remains

### B1 routing oracle - hard macro-routing REJECTED at pre-validation (the cheap kill)

Fable 5's pre-registered two-gate oracle (`eval/b1_routing_oracle.py`), over all 80 logged
deterministic retrieval queries from current production, BM25 over current-document identity cards:

| Gate | Result |
|---|---|
| SAFETY (0 currently-hit turns may have gold outside routing top-5) | **FAIL - 28 of 68** hit turns have gold outside top-5 (ranks up to 60-65) |
| RESCUE (gold in routing top-5 for >= 3 of the 12 misses) | **FAIL - only 1 of 12** |

Hard macro-routing (restrict chunk retrieval to the top-K identity-routed documents) would
GUARANTEE ~28 new losses to rescue at most 1 - a catastrophic trade, and no larger K fixes it
(the unsafe golds sit at ranks 6-65). Mechanism: most eval questions ask about CONTENT ("what
penalties apply?", "what happens if a student fails?"), not IDENTITY ("MSc Periodontology"), so
identity-only routing discards exactly the content signal that chunk-level dense+BM25 provides.
This is the cheap kill Fable 5 predicted and answers review-round-3 Q4 / Gemini's macro-routing
test decisively: do NOT build macro-routing. J3's earlier soft-routing failure (0 rescues / 3
losses) was the same signal at lower stakes.

### B2 attribution + B3 diversity-cap ceilings - both ZERO; retrieval ceiling confirmed

`eval/b2b3_attribution_diversity.py`, over the 12 current misses using their logged queries:

- **B2 (citation-attribution tie-break)**: 0/12. No current miss has a reranked-top-6 chunk whose
  normalised body is byte-identical to a gold-document chunk - so re-attributing shared boilerplate
  to the gold doc has zero rescue ceiling. (After Phase A the remaining misses aren't "retrieved a
  byte-identical sibling chunk"; they're genuinely-different content or out-of-pool.)
- **B3 (diversity cap, max 2 chunks/doc)**: 0/12. Verified the mechanism directly: for misses WITH
  duplicate-filled slots (e.g. glossary follow-up: top-6 held sres-modular x4, only 3 distinct
  docs) the gold document is OUT-OF-POOL entirely (absent from reranked-30), so freeing slots can't
  surface it; for misses WITHOUT duplicates (e.g. roa-ug-4yr follow-up: 6 distinct docs, gold at
  rank 10) there is nothing to cap. Neither shape helps.

**Phase B verdict**: all three remaining architectural/mechanism levers - hard macro-routing (B1),
citation attribution (B2), diversity cap (B3) - are rejected offline before any build. Combined
with ~20 rejected retrieval-signal experiments and the Phase A data-hygiene gains now banked, this
confirms the retrieval ceiling for this corpus is genuinely reached at RoA hit@6 70% strict /
87.5% evidence-sufficient. The residual 12 misses are underspecified questions (no retrieval fix
exists) or out-of-pool cases whose gold-document content simply does not match the query text - both
generation-side / UX territory, not retrieval. This is the reviewers' convergent conclusion
(ChatGPT/Gemini/Grok/Fable 5) now confirmed by direct measurement rather than asserted.

## Phase D1: experiment taxonomy and falsification ledger (what to STOP pursuing)

ChatGPT's round-3 framing: the highest-value output now is not another experiment but a taxonomy
that says which whole CLASSES of technique this corpus has already falsified. Grouping every
experiment across all sessions:

**Class A - Representation** (change what/how documents are embedded): better embed model (mxbai,
bge-m3), contextual per-chunk embeddings, identity-enriched headers (J2), chunk size. Verdict:
**exhausted**. Chunk-size + the original nomic choice were early wins; everything since regressed or
washed. Embedding-model swaps and header enrichment perturb the whole corpus's neighbourhood and
lose more than they gain.

**Class B - Retrieval fusion** (add/combine retrieval channels): SPLADE, embedding ensemble,
weighted score fusion, pseudo-query index, ColBERT first-stage (Idea 2), global + targeted rerank-
pool widening (J0b/Idea 4). Verdict: **exhausted and falsified**. Every added channel or widening
diluted already-marginal correct docs via RRF math; net negative or wash without exception. The
one Class-B-adjacent win, ColBERT as a RERANKER (not a channel), is kept.

**Class C - Metadata guidance** (route/prefer by extracted facets): year handling, degree/award
facet preference (Stage A/A2), document-identity soft routing (J3), hard macro-routing (B1). Verdict:
**falsified for RETRIEVAL-TIME preference**. Soft routing (J3) lost 0/3; hard routing (B1) fails
pre-validation 28-losses/1-rescue. Facet metadata is too sparse and sibling identity cards too
near-identical to route on. BUT the same metadata used for DATA HYGIENE (not retrieval preference)
was the round-3 win - see Class E.

**Class D - Post-retrieval reasoning**: ColBERT MaxSim reranking (KEPT - a real win), LLM listwise
rerank (worse), CRAG verification gate (Stage H, worse), multi-hop decomposition (Stage I, worse).
Verdict: **one durable win (ColBERT rerank), the rest falsified**. Generative/agentic post-retrieval
steps consistently underperform a purpose-built cross-encoder/late-interaction scorer here.

**Class E - Data layer** (round 3; NEW class, not in ChatGPT's original A-D): stale-edition family-
split correction (A1), hub-page removal (A2), lexical-visibility repair (A3), and a deterministic
identity-anchor guard on the contextualizer (C1). Verdict: **the only unexhausted class - +7.5pp
RoA hit@6, net +3 turns, 0 losses**. This is the round-3 discovery: after Classes A-D were mined
out, the remaining gap was not in the retrieval MODEL but in the DATA feeding it (mis-flagged
editions, magnet hub pages, lexically-invisible identity tokens) and in conversational identity
TRACKING (anchor loss across turns) - both below the retrieval architecture.

### Falsified - stop pursuing (evidence is now strong enough):
- Adding any retrieval channel or widening the pool (Class B, ~8 experiments, unanimous regression).
- Retrieval-time metadata/facet/identity routing, soft or hard (Class C, J3 + B1 + Stage A/A2).
- Generative post-retrieval gating/decomposition (Class D minus the reranker; Stage H/I).
- Embedding-model swaps and corpus-wide header enrichment (Class A; mxbai/bge-m3/J2).
- Boilerplate chunk deduplication (measured wrong-lever: 0.1% embedded-dup; would delete the only
  sibling discriminator).
- Citation-attribution tie-break and diversity cap (B2/B3, 0/12 ceiling each).

### Still open (generation-side / UX, NOT retrieval):
- The strict-vs-evidence gap (70% vs 87.5%): the system often RETRIEVES a sufficient document but
  the 7B generator doesn't always use it. The deferred J7 keyphrase-prompt retry is now a FAIR test
  (num_ctx pinned removes the truncation confound that may have muddied the original).
- Underspecified questions (~half the residual misses, e.g. bare "what is a capped mark?"): no
  retrieval fix exists; the honest lever is proactive clarification / disclosure (J6 already does a
  soft version). ChatGPT's "measure the Bayes error" point applies - for these, 70% strict may be
  at or near the human ceiling given query text alone.

This ledger is the round-3 deliverable ChatGPT asked for: the project has moved from exploring the
RAG design space to having MAPPED its boundary for this corpus. Future effort should assume Classes
A-D are closed and spend only on Class E (data quality, as new documents arrive) and generation/UX.

## Phase D2: J7 verbatim-figures retry - REJECTED (confirmed under fair conditions)

Retried J7's "quote specific figures verbatim" rule under the now-fair regime (deterministic +
num_ctx=8192 pinned, removing the truncation confound Fable 5 flagged as a possible explanation for
the original null). Result almost exactly replicates the original J7:

| | keyphrase coverage | answer score | hit@6 |
|---|---|---|---|
| Policy | 60.3% -> 58.2% (-2.1) | 4.17 -> 4.30 (+0.12) | 100% -> 100% |
| RoA | 56.7% -> 58.2% (+1.5) | 3.67 -> 3.60 (-0.07) | 70% -> 70% |
| Overall | 58.5% -> 58.2% (-0.3) | 3.92 -> 3.95 (+0.03) | 85% -> 85%, 0 flips |

A wash: RoA keyphrase nudges up but RoA answer score nudges down, policy keyphrase drops. The
num_ctx confound was NOT the explanation - the fair retest confirms the 7B generator genuinely
doesn't benefit from the instruction. This definitively resolves the long-deferred J7 question:
the strict-vs-evidence gap (70% vs 87.5%) is not closable by prompt instruction on this model.
Consistent with the generator bake-off (the 14B only looked better under self-judging), closing
that gap needs a genuinely stronger, independently-validated generator, not prompt engineering -
i.e. generation-side is also near its ceiling with the local 7B. Reverted
(QUOTE_FIGURES_VERBATIM=False), following the "don't keep washes" precedent (Stage G).

## Round 4: reviewers converge on "ship it + rework the metric"; one real C1 bug found and fixed

Sent round 3's outcome (RoA 62.5%->70.0%) to the reviewers. Unanimous: retrieval ceiling reached,
stop retrieval-signal work, promote evidence-sufficient@6 to the headline metric, and quantify how
much of the residual is genuine query ambiguity (a test-set artifact of scoring ambiguous queries
against one gold doc). The genuinely-new suggestions are all generation/UX: context ASSEMBLY (full
top-1-2 docs vs 6 chunks - untouched by D2's gating/prompting falsification, and the honest next
lever since J7 proved a model can't quote figures its window never contained), a structured-
parameter-extraction "Class F" (ambiguity becomes enumeration), clarification UX (which IS
measurable via scripted 2-turn resolution conversations - correcting the earlier "can't measure"
framing), and a hallucination eval set (a real gap - none exists).

### C1 false-anchor bug (Fable 5) - verified, fixed

Fable 5 found C1 could append a nonsensical programme anchor to a generic question. Verified: the
glossary and DipHE follow-ups were getting a "musculoskeletal/public-health" anchor off a LONE
distinctive token - "term" (from "what does the term..."), "conditions", "principles" - common
English words that register as distinctive because docfreq is computed over identity records only,
not query/corpus frequency. (Fable 5's specific culprit "assessment" was wrong - not distinctive -
but the mechanism and fix are exact.) Fixed by requiring >=2 overlapping distinctive tokens before
firing: the 6 legitimate multi-token anchors all keep firing (east15 {east,acting}, physiotherapy
{credit,physiotherapy}, whistleblowing, etc.), the 6 spurious single-token ones stop.

`c1_anchor_v2` full eval: hit@6 unchanged (0 flips vs c1_anchor), RoA MRR 0.44->0.46 (spurious
anchors were mildly degrading ranking on some hits), all spurious anchors eliminated (0 turns still
carrying the musculoskeletal/public-health text). All three former single-token HITS
(professional-doctorates, foundation-year, pgt-credit) held after their append was removed. Clean
bug fix, kept - production stays at RoA 70% strict / 87.5% evidence-sufficient, now without the
false-anchor mode.

## Round 4, item 2: gold-multiplicity ceiling - strict hit@6 is AT its achievable limit

`eval/gold_multiplicity.py` (Fable 5's method, zero hand-labelling): for each turn, N(q) = number
of CURRENT documents whose full text contains ALL its keyphrases; under exchangeability the best a
single-gold strict-hit@6 can score is min(1, 6/N). Result:

| | actual strict hit@6 | achievable ceiling (single-gold metric) |
|---|---|---|
| RoA | 70.0% | **68.6%** |
| Overall | 85.0% | **84.3%** |

**Actual is AT (slightly above) the achievable ceiling** - the system already beats random
tie-breaking among equally-valid documents. Strict hit@6 has essentially zero headroom left; it is
now measuring the metric's single-gold artifact, not retrieval quality. This is the quantitative
confirmation (not assertion) the reviewers asked for that retrieval is done.

The 12 misses decompose (verified by inspecting the gold documents):
- **2 pure gold-multiplicity artifacts** (N>6): `ma_social_work[primary]` N=**78** (its keyphrases
  "overall weighted average"/"60 or more"/"Pass with Merit" are in the gold AND 77 other current
  documents - generic Merit boilerplate), `diploma[follow_up]` N=22. Scoring these against one gold
  is meaningless.
- **6 keyphrase-proxy failures** (N=0): the keyphrases aren't jointly present in ANY current
  document, gold included - e.g. `roa-ug-4yr-year-1[follow_up]` gold has "failed" + "resit
  assessment period" but phrases the third keyphrase ("subsequent year") differently. The answer IS
  in the gold; the literal keyphrase string is too brittle. This is a keyphrase-METRIC limitation
  (and it caps evidence-sufficient@6 too, since that also uses keyphrase presence - Fable 5's
  suggested reference-answer-containment variant would be more robust).
- **4 tight cases** (N=1-6): csee (N=6 x2), diploma[primary] (N=4), aegean[follow_up] (N=1) - the
  closest thing to genuine residual retrieval limits, and even these are borderline.

So the residual is dominated by measurement artifacts (gold multiplicity + keyphrase brittleness),
not retrieval failure. Only ~4 of 12 misses are even arguably "real", and the achievable-ceiling
math says the system is at the limit of what single-gold strict hit@6 can reward.

## Round 4, item 3: metric rework - evidence-sufficient@6 is now the headline

Given the above, strict hit@6 is demoted from headline to attribution diagnostic, and
**evidence-sufficient@6 (RoA 87.5% / overall 93.8%) is the primary retrieval metric** - it credits
retrieving ANY document that contains the answer, which is what a user experiences and what the
gold-multiplicity analysis shows is the honest target. `eval/score_summary.py` now leads each
group's output with `evidence_sufficient_at_6`; strict/lenient hit@6 remain as diagnostics.
Caveat carried forward: evidence-sufficient@6 shares the keyphrase-proxy brittleness (the 6 N=0
turns), so a reference-answer-containment or judge-based sufficiency variant is the natural refinement
if this metric is ever used for a fine-grained verdict.

## Round 4, item 6: hallucination / groundedness eval (the measurement gap) - baseline established

`eval/hallucination_eval.py`: for each answer already generated at current production, reconstruct
the exact context it was generated from (deterministic re-retrieve -> _format_context) and have the
14B judge check whether every specific factual claim is supported by that context. Measures
FAITHFULNESS-TO-CONTEXT (intrinsic hallucination), orthogonal to hit@6.

**Baseline groundedness: 78.8% (63/80 answers)**:

| split | grounded |
|---|---|
| on hit@6 turns | 83.8% |
| on miss turns | **50.0%** |
| RoA | **65.0%** |
| Policy | 92.5% |

Clear pattern: hallucination concentrates where retrieval failed (miss turns 50%) and where answers
hinge on specific figures (RoA 65% vs policy 92.5%). Even ~16% of hit turns hallucinate. The
unsupported claims are concrete fabricated figures - "the capped mark is 50", "40 credits at Level 6
and the remaining at Level 5", "150 credits at Masters Level... pass all except SE760", "70 or more
across 135 PG Diploma credits" - and one sibling cross-contamination (the roa-ug-3yr[primary] HIT
answer imported the FOUR-year degree's "120 credits at Level 4" rule into a THREE-year question).

This is the measurable form of the RoA answer-quality gap: the 7B fabricates a plausible figure
rather than abstaining when the exact one isn't in its 175-word retrieved chunk. It directly
motivates item 5 (context assembly): if the real figure is in the full document but not the
retrieved chunk, fuller context would let the model quote it instead of inventing it. Caveats: the
14B judge runs somewhat strict, and a few "not grounded" verdicts are faithful abstentions
over-flagged; but even discounting that, a real ~21% hallucination rate concentrated on RoA specifics
and misses is established as the baseline to improve against. Results per-turn in
`eval/results_hallucination.json`.

## Round 4, items 2 + 4a: inline citations REJECTED (regressed groundedness), degree-length tokenizer KEPT

Two round-4 experiments run back-to-back (deterministic, full 80-turn A/B vs `c1_anchor_v2`).

**Item 2 - per-claim inline citations (REJECTED).** Added a system-prompt rule asking the 7B to
attribute every specific factual claim to its exact `source_url` inline (`INLINE_CITATIONS` flag).
Hypothesis: forcing per-claim provenance would reduce fabrication. Result: answer_score a wash
(3.91->3.90, as D2's verbatim-figures retry predicted), retrieval unchanged - but **groundedness
REGRESSED 78.8% -> 67.5% (-11.3pts)**, every split worse (RoA 65->55, Policy 92.5->80, miss-turns
50->30.8). The mechanism is visible in the judge's flagged claims: the citation itself becomes a new
hallucination surface - the 7B confidently attributes a real figure to the WRONG filename (e.g.
"pass mark is 50 [five-year-integrated-masters-21-v7.pdf]" when that document says 40). Asking a
small model to cite provenance per claim makes it fabricate provenance on top of the facts.
Citations don't self-verify. Reverted to end-of-answer Sources only. This is a concrete data point
for the "stronger generator" future item: provenance discipline is exactly the kind of thing a
larger model does reliably and a 7B does not.

**Item 4a - degree-length yr<->word tokenizer (KEPT, new production).** Essex RoA filenames encode
degree length as the glued BM25 token `3yr`/`4yr`/`5yr` - the ONLY token distinguishing a
three-year from a four-year from a five-year programme, since the three siblings otherwise share
generic `year`/`rules`/`masters` boilerplate. Queries say "Four-Year"/"three year", so the glued
`4yr` matched neither `four` nor `year` and the home document was lexically invisible on its own
degree length. Fix (`_DEGREE_SYNONYMS` in `src/lexical.py`): emit the spelled number (+`year`) for
an `Nyr` token, gated with the existing alpha/digit split; offline-verified that `3yr`->`three` and
`4yr`->`four` do not cross-match. Full 80-turn deterministic A/B: clean **+1 / -0** per-turn - the
single flipped turn is `roa-ug-4yr-year-1-rules` follow-up (miss->hit), exactly the targeted
sibling. **RoA hit@6 70->72.5%, evidence-sufficient@6 87.5->90.0%, overall hit@6 85->86.3%, zero
regressions.** Answer_score dipped within the noise floor (RoA 3.65->3.55). This is the 2nd
net-positive RETRIEVAL change since the round-3 data-hygiene work - and like those, it's a
DATA/lexical fix below the architecture, not signal-engineering (which stayed falsified across ~20
attempts). Item 4b (five-year rename demotion) left for user judgment: two parallel current
lineages exist (`roa-ug-integrated-masters-5yr-year-N.pdf` vs `five-year-integrated-masters-21-v7
.pdf`) and it's ambiguous whether the latter is a stale duplicate or a legitimately distinct
document - needs a human to decide before demoting either.

## Round 4, item 3: stronger generator (cloud gpt-oss-120b) — the retrieved-but-not-surfaced gap IS a generator problem

The strict-vs-evidence gap (a sufficient document is retrieved but the local 7B doesn't surface its key figures) and the item-6 hallucination finding (8/17 hallucinations were figures that WERE in the retrieved chunks, fabricated anyway) both pointed at generator capability, but D2 only proved a *prompt rule* can't close it on the 7B. This item tests the actual lever: swap the GENERATOR for a genuinely stronger model, holding retrieval and the (local 14B) judge identical.

**Setup.** Added a cloud generator path (`generate()` in src/llm.py, `GENERATOR_PROVIDER`/`GENERATOR_MODEL` env, OpenAI-compatible; only the answer-generation call moves to cloud - contextualizer, judge, summarizer, relevance stay local). Provider Groq free tier. First choice llama-3.3-70b-versatile was abandoned: its free daily cap is 100K tokens (TPD) - a full 80-turn eval needs ~450K, so it can't finish. gpt-oss-120b has a 200K TPD, which comfortably covers the 40 RoA turns (~112K) - so the test was run on the RoA subset (`eval/questions_roa_only.json`), which is exactly where hallucination concentrates (RoA groundedness 65% vs Policy 92.5%; Policy has almost nothing to gain). Deterministic (temp 0). 7B baseline is the same 20 RoA questions from `results_item4a.json`, re-judged for groundedness on identical retrieval (`results_item4a_roa.json`).

**Result (groundedness, faithfulness-to-context; both arms identical retrieval + judge):**

| RoA groundedness | 7B (baseline) | gpt-oss-120b | Δ |
|---|---|---|---|
| on hit@6 turns | 72.4% | **92.9%** | **+20.5** |
| on miss turns | 54.5% | 41.7% | **−12.8** |
| overall RoA | 67.5% | 77.5% | **+10.0** |
| answer_score (14B judge) | 3.55 | 3.98 | +0.43 |

**Two findings, split cleanly by cause:**

1. **When retrieval succeeds, a stronger generator dramatically reduces hallucination: +20.5pts (72.4%->92.9%).** This is the first hard confirmation that the retrieved-but-not-surfaced gap is a GENERATOR-CAPABILITY problem, not retrieval and not promptable on the 7B. Given the same context the 7B fabricated from, gpt-oss-120b reports the figure faithfully ~93% of the time. Answer_score corroborates (+0.43).

2. **When retrieval FAILS, the stronger generator is WORSE: -12.8pts (54.5%->41.7%).** The "more confidently wrong" effect: handed the wrong document, the 120B writes a fluent, confident answer the judge flags as ungrounded, whereas the weaker 7B more often hedged/waffled (which reads as faithful abstention). On miss turns, fluency hurts groundedness.

**Strategic consequence — the residual hallucination splits by cause, and so does the fix:**
- retrieved-but-not-surfaced -> generator problem -> SOLVED by a stronger generator (+20.5).
- miss-turn hallucination -> NOT a generator problem -> needs retrieval improvement OR an ABSTENTION GATE (don't confidently answer from weak context). A stronger generator without an abstention gate trades hit-turn gains for miss-turn losses; the net (+10 overall) is positive only because hit turns outnumber miss turns.

**Caveats.** RoA-only (20 questions/40 turns), not the full 80 - but RoA is where the gap lives. gpt-oss-120b runs on Groq free tier (200K TPD), so it establishes the *ceiling a stronger generator reaches*, not a standing production config (reverted to local 7B after the run; the cloud path is env-gated and off by default). The 14B judge is identical across both arms, so the comparison is fair even if the judge runs somewhat strict in absolute terms. Per-turn files: `results_gptoss120b_roa.json`, `results_hallucination_gptoss_roa.json`, `results_hallucination_item4a_roa_7b.json`.

### Item 3 follow-up: local 14B generator (full 40-question run) - the monotonic ladder

Repeated item 3 with the LOCAL qwen2.5:14b-instruct as generator (no cloud limits, so the full 40
questions - 20 RoA + 20 Policy - were run, unlike the cloud subset). Same retrieval, same local 14B
judge. Caveat: this arm is SELF-JUDGED (generator == judge model), which the bake-off showed inflates
answer_score; groundedness is more objective but may still be slightly optimistic - yet the 14B still
lands below the independently-judged 120B, so the ordering is safe.

**Three-way RoA groundedness (identical retrieval + judge), the headline of item 3:**

| RoA groundedness | 7B (baseline) | 14B (local) | gpt-oss-120B (cloud) |
|---|---|---|---|
| on hit@6 turns | 72.4% | 81.5% | 92.9% |
| on miss turns | 54.5% | 46.2% | 41.7% |
| overall RoA | 67.5% | 70.0% | 77.5% |
| answer_score | 3.55 | 3.70 | 3.98 |

**Groundedness on hit turns is MONOTONIC in generator size: 72.4 -> 81.5 -> 92.9.** When retrieval
succeeds, faithful reporting of the figure is purely a generator-capability function. The local 14B
captures ~half the cloud 120B's gain (+9pt vs +20.5pt) while remaining local/free/unlimited.
Miss-turn groundedness is monotonic the OTHER way (54.5 -> 46.2 -> 41.7): bigger models are more
confidently wrong from the wrong document - so an abstention gate is needed regardless of generator.

14B full-run splits: overall 81.2% (65/80), hit-turn 88.1%, miss-turn 46.2%, RoA 70.0%, Policy 92.5%
(Policy unchanged across all three generators - no headroom). Answer_score full: 14B RoA 3.70 vs 7B
3.55 (+0.15, self-judged so an upper bound); Policy 4.15 vs 4.17 (flat). Files:
`results_qwen14b_full.json`, `results_hallucination_qwen14b_full.json`.

**Deployment implication.** The 14B is the realistic standing-production upgrade path: a genuine
+9pt hit-turn groundedness gain, LOCAL (no cloud daily caps), at the cost of ~2x generation latency
and 16GB-RAM tightness. The cloud 120B shows the ceiling (+20.5) but isn't a standing-prod config on
the free tier. Whether the 14B's grounding gain is worth the latency/RAM cost is a user deployment
decision (not yet made; production remains 7B).

## Round 4, item 3 follow-up: abstention gate for miss-turn hallucination - FALSIFIED (diagnostic only)

Item 3 showed a stronger generator worsens groundedness on retrieval MISSES ("confidently wrong"
from the wrong doc). Natural fix: an abstention gate that hedges/abstains when retrieval probably
missed. Before building (the earlier CRAG hard-gate was rejected for over-firing, answer_score
3.90->2.88), diagnosed whether ANY retrieval signal separates HIT from MISS turns
(scratchpad/abstain_diag*.py, 80 turns of results_qwen14b_full.json):

| signal | HIT | MISS | best gate |
|---|---|---|---|
| top ColBERT score (abs confidence) | mean 24.1 | mean 24.9 | precision 0.16 = base rate (ZERO signal) |
| margin (top1 - top2) | median 0.05 | median 0.01 | precision <=0.21 (false-gates ~all hits) |
| distinct families in top-6 | median 2 | median 6 | precision 0.40 / recall 0.62 @ >=6 (real but weak) |

**Falsified.** Absolute confidence and margin carry NO signal because a miss retrieves the wrong
near-duplicate SIBLING, which is just as on-topic and high-scoring as the right document - the
reranker is confidently wrong, mirroring the generator. Only fragmentation (how scattered the top-6
is across document families) separates them, and even at its best (all 6 slots different families)
it's precision 0.40: a hard gate would prevent ~8 hallucinations while killing ~12 good answers (a
losing trade, and a repeat of CRAG). A soft hedge at that precision is just the shipped J6
disclosure and doesn't stop the wrong figure being stated.

**Constructive output:** the fragmentation signal (>=6 distinct families in top-6, ~62% of true
misses) is a concrete, data-backed TRIGGER for the D3 clarification UX - the previously-missing
"when to ask" condition (pool maximally scattered => query too under-specified => ask which
programme/document instead of guessing). Miss-turn hallucination thus folds into (a)
sibling-disambiguating retrieval (hard, mostly exhausted) and (b) D3 clarification UX; there is no
standalone confidence-based abstention gate at acceptable precision.

## Round 4, item 4b: byte-identical duplicate removed from corpus

`five-year-integrated-masters-21-v7.pdf` (PGT directory) was proven byte-identical (20596 chars,
2021-22 cohort) to the canonical `roa-ug-integrated-masters-5yr-year-5.pdf` (UG /current/ per-year
set) - a leftover from Essex's pre-2025 filename scheme, and mislabeled academic_year 2025-26 from
its directory path though its content is the 2021 cohort (it was the wrong-attribution target in the
item-2 hallucination test). Removed outright (adds zero coverage; the whole 2021 cohort is already
served by year-5). Because COLBERT_FIRST_STAGE_ENABLED is off (the ColBERT index is only a rerank
embedding cache, not a retrieval channel), removal needed no ColBERT rebuild: deleted its 21 chunks
from Chroma (delete_document, which bumps the corpus version so BM25 rebuilds), tidied the inert
ColBERT snapshot + manifest, and added the URL to a durable `_EXCLUDED_URLS` guard in run_ingest.py
(same spirit as the hub-page guard) so a future crawl won't re-add it. Regression: a five-year
integrated-masters query now returns only the canonical per-year files (year-1..5), the removed
duplicate cannot surface, canonical year-5 intact (21 chunks). Data changes are local (data/ is
gitignored, rebuilt from run_ingest.py); the exclusion in run_ingest.py is what makes it durable.

## Round 4, D3: clarify-on-underspecified gate (generic hard-ask) - BUILT, off by default

The abstention-gate diagnostic reframed miss-turn hallucination as partly an UNDER-SPECIFICATION
problem: ~half the fragmented-pool misses are queries about programme-specific rules that name no
programme ("minimum average to pass with Merit?" - which programme?). D3 asks instead of guessing.

**Trigger (measured on results_qwen14b_full.json, 80 turns):** fragmented pool (>= 6 distinct
document families in top-6) AND under-specified query (extract_degree_length + extract_award_type
both empty). Precision: fragmentation-only 0.40 -> +under-spec filter 0.45 (fires 11/80, catches 5
of the miss turns; the under-spec filter correctly drops named-programme hits like "Four-Year
Honours"/"Foundation Year" that fragment but are answerable). Confirmed the gate fires on
under-specified misses and NOT on 3yr/4yr/masters-named queries.

**Design forced to GENERIC ask (no listed options).** The intuitive version - list the candidate
programmes and let the user pick - is provably broken and was already killed (J8/NAMEABLE_
CLARIFICATION): on a retrieval MISS (hit@6=False) the correct document is BY DEFINITION absent from
the pool, so options sourced from the pool are all wrong (verified: all 5 caught misses have the
correct doc absent from the offered families; J8 empirically offered 4 confidently-wrong names).
The only honest form is a generic "which programme did you mean?" with no guesses, letting the USER
supply the missing fact.

**Payoff validated (2-turn simulation).** Turn 1 "pass with Merit?" fires the clarification; turn 2
user replies "MA Social Work"; the contextualizer rewrites to "...pass with Merit in the MA Social
Work programme" and retrieval goes from a total MISS to all 6 top-6 slots = ma_social_work. The
clarification converts an unanswerable query into a perfect hit - it is not a dead-end.

**Shipping status: OFF by default** (`CLARIFY_UNDERSPECIFIED_ENABLED=False`). A clarifying question
is scored as a MISS by the hit@6 eval by design, so this can only be judged on real conversations,
and at 0.45 precision it interrupts some answerable general questions (framework/procedure queries
that name no programme). Flip the flag to evaluate the ask-vs-guess tradeoff live. Production
unaffected until then.

## Round 5: generator bake-off (10 local models) + the miss-turn faithfulness finding

Round-5 reviews converged on "the generator is the lever." Ran a rigorous bake-off: 10 local models
generate answers on IDENTICAL fixed contexts (eval/generator_bakeoff.py - contexts reconstructed once
from the 14B reference run's history, so retrieval is held constant and only the generator varies;
cleaner than end-to-end evals where follow-up retrieval drifts per model). Judged groundedness
(faithfulness-to-context) with qwen2.5:14b; then answer_score (completeness vs gold) offline on the
same answers (eval/bakeoff_answerscore.py). NOTE: qwen-generator rows are self-judged (inflated);
gemma3/gpt-oss/phi4/llama/mistral rows are cross-family judged (clean).

**Groundedness + latency frontier (sorted by RoA):**

| model | hit-turn | RoA grounded | latency | RAM | judge |
|---|---|---|---|---|---|
| gpt-oss:20b | 95.5% | 95.0% | 29s | 13GB | clean |
| gemma3:12b | 97.0% | 92.5% | 31s | 8.1GB | clean |
| qwen3:14b | 95.5% | 90.0% | 141s(!) | 9GB | clean |
| phi4 | 97.0% | 87.5% | 35s | 9GB | clean |
| qwen2.5:14b (PROD) | 94.0% | 85.0%* | 26s | 9GB | self* |
| qwen3:8b | 88.1% | 77.5% | 42s | 5GB | clean |
| llama3.1:8b | 88.1% | 72.5% | 16s | 4.9GB | clean |
| qwen2.5:7b | 94.0% | 70.0%* | 15s | 4.7GB | self* |
| mistral:7b | 85.1% | 67.5% | 19s | 4.4GB | clean |
| llama3.2:3b | 74.6% | 65.0% | 29s | 2GB | clean |

Every ~12-20B model beats the current production qwen2.5:14b on RoA groundedness - and the 14B's 85%
is self-judged, so it's genuinely near the BOTTOM of the big-model tier. Clear ~12B capability
threshold (8B plateau ~70-77% RoA; 12-14B jump to 87-95%). qwen3:14b at 141s/answer is thinking-token
bloat (think-off variant tested separately).

**The key finding - split answer_score by hit vs miss turn:**

| model | ans HIT | ans MISS | grounded HIT | grounded MISS |
|---|---|---|---|---|
| gpt-oss:20b | 4.24 | 3.31 | 96% | 92% |
| qwen2.5:14b (PROD) | 4.12 | 3.38 | 94% | **69%** |
| phi4 | 3.96 | 2.69 | 97% | 77% |
| gemma3:12b | 3.94 | 2.69 | 97% | **92%** |

gemma3's lower headline answer_score is NOT terseness - on HIT turns it matches everyone (3.94 vs
14B's 4.12). The gap is entirely MISS turns, where gemma3 FAITHFULLY ABSTAINS (92% grounded, low
answer_score) instead of guessing, while the current 14B GUESSES FROM PARAMETRIC MEMORY (69% grounded
= 31% hallucinating a figure not in the rules; lucky guesses inflate its answer_score). For a
policy/rules assistant a wrong pass-mark is worse than "not found", so answer_score was rewarding the
14B's hallucination and penalizing gemma3's honesty.

**SECONDARY FINDING - overturns item 3.** Item 3 concluded "stronger models are MORE confidently wrong
on misses" (54.5->46.2->41.7). FALSE in general - it was specific to the qwen line. gemma3 and
gpt-oss:20b stay faithful on misses (92% grounded), breaking the trend. So choosing the right
generator IS the abstention solution the round-4 gate diagnostic couldn't build - a model that
naturally says "not found" on weak context, no gate required.

**RECOMMENDATION: switch production qwen2.5:14b -> gemma3:12b.** Comparable completeness on hits,
dramatically more faithful on misses (92% vs 69% grounded -> hallucination on failed retrieval drops
~31%->~8%), RAM-safe (8.1GB vs 9GB), and faster. gpt-oss:20b is the ceiling (best on everything) but
13GB is impractical for production on the 16GB Mac (must coexist with the 7B contextualizer +
retrieval stack). Pending: cross-family re-judge to strip the self-judged rows; a real production
switch is a one-line LOCAL_GENERATOR_MODEL change in src/llm.py once confirmed.

### Round 5 generator - CORRECTION after cross-family + neutral re-judging (2026-07-24)

The single-judge (qwen2.5:14b) numbers above OVERSTATED gemma3's advantage. Re-judged the finalists
with a lenient same-family judge (gemma3) and a NEUTRAL cross-family judge (phi4 - cross-family to all
candidates, not itself a candidate). Neutral phi4-judged groundedness:

| model | overall | RoA | hit | miss | latency | RAM |
|---|---|---|---|---|---|---|
| gemma3:12b | 97.5% | 95.0% | 98.5% | 92.3% | 31s | 8.1GB |
| qwen2.5:14b (old prod) | 92.5% | 92.5% | 94.0% | 84.6% | 26s | 9GB |
| gpt-oss:20b | 91.2% | 90.0% | 94.0% | 76.9% | 29s | 13GB |
| qwen3:8b::nothink | 88.8% | 80.0% | 95.5% | 53.8% | 17s | 5GB |

Corrections to the headline: (1) gemma3 IS the robustly best model (top under BOTH neutral judges) and
the switch is validated - but the margin over the old 14B is MODERATE (+2.5 RoA / +7.7 miss under
phi4), NOT the "hallucinates 4x less" claimed. Neutral-judged 14B miss-hallucination is ~15% (not
31%); gemma3 ~8%. (2) gpt-oss:20b's apparent lead was a qwen-JUDGE ARTIFACT - it drops to bottom under
phi4 (judge-volatile) + is 13GB, so NOT the ceiling; skip it. (3) gemma3 is slightly SLOWER than the
14B (31 vs 26s), not faster (earlier error). (4) Small-model check: qwen3:8b::nothink matches on HIT
turns but COLLAPSES on misses (53.8% grounded) - disqualifying for a policy assistant despite 17s/5GB.
Methodology lesson: never trust a single judge for close calls; gemma3-as-judge is too lenient (100%
self), qwen is harsh, phi4 is the usable neutral one. Production stays gemma3:12b (validated).

---

# Round 5: retrieval bake-off — the RoA sibling frontier is an underspecification problem, not a retrieval one (2026-07-24)

The generator was bake-off'd rigorously (10 models); retrieval never had the same treatment (only
3 embedders + 2 rerankers across the whole project). This round gave it that treatment — decomposed
into a staged pipeline so recall failures and ranking failures separate cleanly, rather than a blind
embedder×reranker grid. **Conclusion up front: the residual RoA misses are not fixable by any
reranker or embedder or even LLM reasoning — the queries are genuinely underspecified, which
scientifically vindicates the D3 clarification UX as the correct (and only) remaining lever.**

## Stage (a): pool-recall diagnostic — the misses are mostly RANKING, not recall

`eval/retrieval_recall_diag.py` splits every current RoA miss by whether the gold document is present
in the candidate pool *before* reranking (a ranking failure the reranker could rescue) or absent from
it entirely (a recall failure only a better embedder could fix). Run on BOTH the tuned set and the
independent holdout:

| set | ranking failures (gold in pool) | recall failures (gold absent) |
|---|---|---|
| main | 9/13 (69%) | 4/13 (31%) |
| set2 (holdout) | 13/14 (93%) | 1/14 (7%) |
| **combined** | **22/27 (81%)** | **5/27 (19%)** |

81% of misses have the gold document sitting in the pool, mis-ranked below a wrong sibling. That
points the high-leverage lever at the **reranker**, not the embedder — and bounds the embedder's
maximum possible contribution at the 19% recall tail.

## Stage (b): reranker sweep on fixed pools — every cross-encoder plateaus at ~+3

`eval/reranker_sweep.py` captures the candidate pools once (`eval/reranker_pools.json`, both sets) and
re-ranks them with different models — no re-embedding, so this is cheap and isolates the ranking
decision. Baseline is the current production ColBERT (`GTE-ModernColBERT`): main 27/40, set2 26/40
family-hit@6. Tested cross-encoders (`bge-reranker-v2-m3`, `bge-reranker-base`, `mxbai-rerank-base`,
a second ColBERT) plus the advanced late-interaction / LLM-logit backends
(`eval/reranker_sweep_advanced.py`):

- Every cross-encoder caps at roughly **+3 family-hit** over baseline and is **holdout-unstable** (a
  gain on one set washes or reverses on the other). `bge-reranker-base` was the only clean net-positive
  (main +3 / set2 +1); the rest broke even or regressed on set2.
- The current ColBERT is already at/near the top of the pack — no swap decisively beats it.
- `jina-reranker-v2` was dropped (transformers API incompatibility); `mxbai-rerank-large` OOM'd on the
  16GB MPS GPU (used `mxbai-base` instead).

The mechanism is the recurring project theme: near-identical sibling documents read *the same* to any
scorer, so a better scorer can't separate them.

## Stage (b'): identity-salience formatting — marginal

`eval/reranker_salient.py` tests a data-side idea instead of a model swap: prepend a humanized
programme/degree/year identity line to each passage (repeated for emphasis) so the one discriminating
signal is prominent, then re-rank. Result: marginal (+2 family-hit at best, ±0 on the other set).
Formatting the identity more loudly doesn't help when the *query* doesn't name the identity to match
it against.

## Stage (c): the capstone — can REASONING break the tie that SCORING can't? No.

The incisive form of the "would a bigger/reasoning reranker help" question: on exactly the 22
ranking-failure turns (gold in pool, mis-ranked), give a capable LLM the query plus the distinct
competing document identities from the pool and ask it to pick the single best match
(`eval/llm_disambig_probe.py`). If reasoning picks the gold family, an LLM reranker would be worth its
cost; if not, the tie is genuinely unbreakable from the query alone.

**Result: 3/22 correct (14%), 0 abstentions.** Verified genuine (not a parse bug): the picks are
varied and it got a few right by real identity-matching (e.g. matched a "three-year" follow-up to the
`3yr` family). Nuance worth recording: the option sets are large — **17 to 64 distinct competing
families per turn** — so 14% is actually *above* random (~2.5% for ~40 options); reasoning has *some*
identity signal but nowhere near enough to disambiguate reliably. The sheer crowding of the sibling
space is itself part of the wall.

## What this closes

Putting the three stages together:

- 81% of RoA misses are ranking failures — the right document **is** retrieved into the pool.
- **Scoring** (5 rerankers, cross-encoder + late-interaction): modest (~+3 max), holdout-unstable.
- **Identity-salient formatting**: marginal (+2 at best).
- **Reasoning** (LLM disambiguation): weak signal, 14% — above random but unreliable.

No lever available to a retriever — better scoring, louder formatting, or explicit reasoning —
reliably tells the siblings apart, because the distinguishing fact is **not present in the query**.
This is the rigorously-earned confirmation of what earlier rounds kept implying: **the RoA sibling
frontier is an underspecification problem, not a retrieval problem.** The only thing that resolves it
is obtaining the missing identity from the user → the **D3 clarification UX** (built, off by default)
is the correct and only remaining lever. The embedder bake-off (phase 2) was therefore **not run**:
it targets only the 19% recall tail, and even a perfect embedder cannot answer a question that never
names which programme it is about.

**Production retrieval is unchanged and considered closed:** hybrid dense (`nomic-embed-text`) + BM25
with RRF fusion, `is_current` pre-filtering, family-recency dedupe, ColBERT late-interaction
reranking. RoA hit@6 stays 70% strict / 87.5% evidence-sufficient. The retrieval investigation has
reached its natural end; further RoA gains require the product-side D3 decision, not another
retrieval experiment.

---

# Round 5 wrap-up item: routing pre-validation — resolved by existing evidence, no new build (2026-07-24)

Round-5 review Q2 asked whether the corpus should "represent programme/cohort as hard facets and
route" (a learned query→document-facet classifier before retrieval); the reviewers split (DeepSeek
leaned toward building it, Fable 5 against). No new experiment is warranted — this exact class of
mechanism has **three independent falsifications** on this corpus, and the retrieval-bake-off
capstone above is the decisive nail:

- **B1 hard macro-routing oracle** (Phase B): restricting chunk retrieval to a document router's
  top-5 would GUARANTEE ~28 new losses (currently-hit turns whose gold sits at router-rank 6–65) to
  rescue at most 1 of 12 misses. Mechanism: most questions ask about CONTENT ("what penalties
  apply?"), not IDENTITY ("MSc Periodontology"), so identity routing discards the content signal
  that chunk-level dense+BM25 relies on.
- **Stage A / A2 facet filtering** (hard then soft RRF-fused): both net-regressed RoA, because this
  corpus's facets are **not mutually-exclusive partitions** (a masters-labelled document legitimately
  holds the correct diploma-exit answer) and extraction coverage is too sparse to tag many correct
  documents at all.
- **J3 soft routing prior**: 0 rescues / 3 losses — the same signal at lower stakes.

The DeepSeek variant (a *learned* classifier rather than B1's BM25 router) does not change the
verdict, because the blocker is not router accuracy — it is that **81% of the misses don't name the
facet in the query at all** (capstone above; earlier pre-validation found 13/16). A more accurate
router still has nothing to route on when the query contains no routable signal, and still pays B1's
~28-loss safety cost on the content-question majority. **Verdict: do not build facet routing.** The
disagreement resolves in Fable 5's favour on existing measured evidence. This is the same conclusion
as the capstone from a second angle: the residual is underspecification, addressable only by asking
the user (D3), not by any pre-retrieval routing layer.

---

# Round 5 wrap-up item: keyphrase-metric fix — judge-based evidence sufficiency (2026-07-24)

Round-4 item 2 flagged that evidence-sufficient@6 inherits the keyphrase proxy's brittleness: it
credits a retrieved document only when ≥ half the turn's keyphrases appear as EXACT case-insensitive
substrings, so a document that genuinely contains the answer but phrases a keyphrase differently
("subsequent year" vs "the following year of study") is under-credited. Fable 5's recommended
refinement is a reference-answer-containment / judge-based sufficiency check, built here as
`eval/evidence_sufficient_judge.py`: for every turn scored *insufficient* by the keyphrase rule, ask
the 14B judge whether ANY of the turn's top-6 retrieved documents actually contains the information
in the gold reference answer (early-exit on first yes). Only keyphrase-insufficient turns are judged,
so the number can only rise, and every rescue is printed for inspection.

**Result (on `results_c1_anchor_v2.json`, 80 turns):**

| | keyphrase-string evid-suff@6 | judge-refined | rescued |
|---|---|---|---|
| RoA | 82.5% | 90.0% | +3 |
| Policy | 85.0% | 97.5% | +5 |
| **Overall** | **83.8%** | **93.8%** | **+8** |

**But the rescues were spot-checked, and they are not all equal — the headline 93.8% is slightly
optimistic:**

- **5 policy rescues: rescuing document == the gold document.** Unambiguous — the gold was retrieved
  and genuinely contains the answer; only the literal keyphrase string missed it. These are exactly
  the brittleness the fix targets. Solid.
- **1 RoA rescue (`mscperiodontology[follow_up]` ← the Alexandria partner sibling):** cross-document
  but a near-identical same-family sibling that genuinely carries the same answer — a fair
  evidence-sufficiency credit (the user gets the correct answer).
- **2 RoA rescues are the judge over-crediting vocabulary overlap** — the topical-overlap trap the
  prompt explicitly warned against: `roa-ug-glossary[primary]` ← `foundation-year-rules` (which
  *uses* "capped at 40" but never *defines* the glossary term), and `roa-ug-glossary[follow_up]` ←
  `msc-ot_25` (which is about *un*capped module marks — the opposite concept). These should not
  count.

So the **defensible correction is +6 clean rescues → ~91.3% overall** (Policy 97.5%, RoA ~85%); the
two glossary rescues inflate the raw judge number. Two takeaways, both reinforcing rather than
softening the capstone:

1. The keyphrase-string metric genuinely was ~7–8pp pessimistic (the +6 clean rescues), overwhelmingly
   on the **policy** side — a real measurement fix. **evidence-sufficient@6 should be read as ~91% overall
   / ~97.5% policy / ~85% RoA once string brittleness is removed.**
2. **5 turns remain genuine still-misses even under a deliberately lenient judge** — including
   `roa-ug-4yr-year-1[follow_up]`, the exact "subsequent year" case round-4 item 2 called a keyphrase
   artifact. The judge confirms the answer is NOT in any retrieved document there: it's a *real*
   retrieval miss, not brittleness. The judge-based check thus does what the string metric couldn't —
   separate true metric artifacts (rescued) from genuine retrieval failures (still-miss). The genuine
   residual is the RoA sibling/underspecification wall from the capstone, not a scoring illusion.

Process note: the RoA glossary false-positives are themselves a small piece of capstone evidence —
even an LLM judge, given a glossary question and a sibling document, rubber-stamps shared vocabulary,
the same failure mode as the 14% disambiguation-probe ceiling. The script stays in `eval/` as the
honest sufficiency instrument; `score_summary.py`'s fast keyphrase-string view is kept as the cheap
default with this ~7–8pp pessimism bias now quantified.

---

# Round 5: contextualizer test + unified-model idea — production choice validated, unification rejected (2026-07-24)

The follow-up query contextualizer (rewrites "what happens after that?" into a standalone question
before retrieval) is the one place the contextualizer model touches retrieval, so it was bake-off'd
the same way as the generator: fix everything else, vary `CONTEXTUALIZE_MODEL`, measure FOLLOW-UP
retrieval only (primary turns have empty history and are returned verbatim — contextualizer-independent).
`eval/contextualizer_sweep.py`, retrieval-only (no generation/judge), family-hit@6 over both the tuned
set and the holdout, mirroring the reranker sweep.

| Contextualizer | follow-up hit@6 (main / set2 / overall) | RoA |
|---|---|---|
| **qwen2.5:7b-instruct (production)** | 82% / 78% / **80.0%** | **62.5%** |
| qwen2.5:14b-instruct | 82% / 82% / 82.5% (+2.5) | 65.0% (+2.5) |
| qwen3:8b | — DISQUALIFIED — | — |
| gemma3:12b (unified-model candidate) | 75% / 72% / 73.8% (**−6.2**) | 47.5% (**−15**) |

Two conclusions:

1. **Production qwen2.5:7b-instruct is the right contextualizer.** The 14B buys only +2.5pp
   (2 turns) — inside the run-to-run noise band established all project — and costs 2× RAM and
   latency on a call that sits on every follow-up's critical path. Not worth switching.

2. **Unified-model idea (one model for contextualize + generate) — REJECTED.** Using the production
   generator gemma3:12b *also* as the contextualizer regresses follow-up retrieval sharply (−6.2pp
   overall, **−15pp RoA**). Faithful query-rewriting is a narrow instruction-following task where the
   small, instruct-tuned qwen2.5:7b genuinely beats the larger, more generative gemma3:12b — the same
   direction as the earlier finding (report: "contextualizer topic drift") that reasonable-looking
   contextualizer changes regress RoA by changing how disambiguating programme names get injected into
   the rewrite. The roles cannot be collapsed; production keeps three models — gemma3:12b (generate) +
   qwen2.5:7b (contextualize) + nomic-embed (embed). Since generate and contextualize are now different
   models, production must run `OLLAMA_MAX_LOADED_MODELS ≥ 2` (or unset) so gemma3 (8GB) and nomic
   (0.4GB) coexist in the 16GB budget without the load-thrash observed during this sweep.

**qwen3:8b disqualified — latency, not accuracy.** qwen3:8b ignores Ollama's `/no_think` soft switch
and emits reasoning tokens (~140s/turn, the same bloat seen in the generator bake-off). A
contextualizer runs synchronously before every follow-up retrieval, so ~140s of think latency is
disqualifying regardless of hit@6 — the run was killed rather than completed. (This also settles the
end-of-experiments model cleanup: the contextualizer test does NOT adopt qwen3:8b, so it can be
removed.)

---

# Round 5: Class F structured-parameter extraction pilot — falsified, branch dropped (2026-07-24)

The round-4 reviewers proposed "Class F": for an underspecified query that names no programme
("what's the minimum weighted average to pass with Merit?") and so mis-retrieves a sibling, don't
retrieve one document — EXTRACT the parameter across ALL current programme documents and ENUMERATE
`{programme → value}`. "Ambiguity becomes enumeration." Piloted on branch `feature/class-f-extraction`
(`eval/class_f_pilot.py`), a bounded pre-validation before any build.

Two questions decide it, both measured by extracting Merit/Distinction classification thresholds
across 40 current PGT RoA documents with gemma3:12b:

| | result |
|---|---|
| (1) Extraction reliability | Merit 34/40 (85%), Distinction 33/40 (82%), 0 parse-fails — **reliable** |
| (2) Variance across programmes | Merit = **{60: 34}**, Distinction = **{70: 33}** — **perfectly uniform** |
| Anchor check (gold 60/70) | ma_social_work, msc-physiotherapy, masters all extracted 60/70 ✓ |

**Verdict: reliable but uniform → enumeration is pointless here, and Class F is falsified.** Because
every programme's Merit threshold is the same value (60), the "underspecified Merit misses" were never
real retrieval failures: any retrieved sibling already answers correctly — this is precisely the
`ma_social_work[primary]` N=78 gold-multiplicity artifact round-4 item 2 identified, now confirmed
from the document side. Enumerating would emit "60, 60, 60, … 60" — zero information.

The general argument (not just this parameter) closes the idea:
- **Uniform parameters** (Merit=60, Distinction=70, most classification boundaries in this corpus):
  no ambiguity exists — any sibling answers — so enumeration adds nothing.
- **Varying parameters** (programme-specific rules): either the query NAMES the programme (then it's
  not underspecified and normal retrieval works), or it doesn't — in which case a long enumerated
  table where the user must still recognise their own programme is strictly worse UX than **D3's
  single clarifying question**. And the per-programme facet extraction such enumeration would need was
  already found too sparse to trust (Stage A facet filtering, department-field coverage).

So Class F cannot beat the already-built D3 clarification UX on the underspecified misses, and is
redundant with gold-multiplicity for the uniform ones. Falsified, in the same clean-kill discipline as
the contextual-embeddings and facet-overlap-graph pilots. The pilot script is kept on main as part of
the falsification ledger; the `feature/class-f-extraction` branch is dropped (it held only the pilot,
now preserved here — nothing to merge).

This closes the round-5 experiment queue. Every generation/UX lever the reviewers proposed for the
residual misses is now resolved: stronger generator (adopted gemma3:12b), abstention (solved by the
generator choice, not a gate), Class F enumeration (falsified), and clarification (D3, built and
vindicated by the retrieval capstone). The remaining open item is the product-side D3 live-judgment
call, not another experiment.

---

# Round 6 (external council) Tier 0: validating the decisions already made (2026-07-24)

Five external reviews (DeepSeek, ChatGPT, Grok, Gemini, Claude Opus 5) converged: retrieval is
closed, gemma3:12b is the right generator, D3 is the remaining lever, ship it. Claude (the only
code-verified reviewer, weighted heaviest) caught two things the others missed, both of which Tier 0
was built to check: (1) the gemma3 switch was never validated *end-to-end* — the generator bake-off
held retrieval constant by construction, so gemma3's abstention-on-miss (its selling point) was never
tested for its known cost of poisoning follow-up contextualization; (2) the decisive miss-turn
faithfulness finding rested on n=13.

## Item 1 — end-to-end gemma3 validation: no follow-up-retrieval regression (VALIDATED)

Ran the full deterministic eval with gemma3 in production position (generator's own answers feed the
follow-up contextualizer, so any abstention-poisoning would show), on both sets, and compared
FOLLOW-UP hit@6 to the 14B end-to-end baselines. Also re-ran the 30-turn multi-topic probe.

| Follow-up hit@6 | 14B baseline | gemma3 E2E | Δ |
|---|---|---|---|
| Main (n=40) | 33/40 (82.5%) | 35/40 (87.5%) | +2 |
| set2 holdout (n=40) | 26/40 (65.0%) | 25/40 (62.5%) | −1 |
| **Pooled (n=80)** | 59/80 (73.8%) | **60/80 (75.0%)** | **+1 (net-flat)** |

Pooled, gemma3 is net-flat on follow-up retrieval (+1 turn, inside the ±2–4pp noise band) — **no
regression.** Two validity checks confirm a clean A/B: primary hit@6 is identical on both sets
(34/34, 29/29), exactly as required since primary retrieval is generator-independent. The 30-turn
multi-topic probe is **byte-identical to the c1/phase5 baselines** (28/30; switch turns 11/11) — the
topic-switch regime where abstention-poisoning would surface shows zero damage. The mechanism Claude
predicted holds: `_anchor_from_history` reads user turns only, so programme identity survives an
abstention. **The gemma3 production switch is validated end-to-end.**

## Item 2 — synthetic miss corpus (n=200): gemma3 does NOT hallucinate more (the n=13 scare retired)

`eval/synthetic_miss_corpus.py` scales the miss-turn faithfulness test from n=13 to n=200 with zero
labelling: pair each RoA question with 5 sibling documents that contain NONE of its gold keyphrases
(a synthetic "wrong sibling retrieved"), generate, and judge faithfulness-to-context. On a wrong
context, GROUNDED = safe (faithful abstention or context-only facts); NOT GROUNDED = fabricating a
figure the context lacks.

**First pass used qwen2.5:14b as judge and produced an alarming flip** (gemma3 76% grounded / 4.5%
hallucinated vs 14B 96% / 0.5%) — but that was **entirely judge self-preference bias**: the 14B judge
favours the 14B's own answers and is harsh on gemma3 (the documented confound, and why the bake-off
finalists used neutral phi4). Re-judged with neutral **phi4** (`eval/synthmiss_rejudge.py`):

| Synthetic misses (n=200) | 14B-judged (confounded) | **phi4-judged (neutral)** |
|---|---|---|
| gemma3 grounded / hallucinated | 76.0% / 4.5% (9) | **100.0% / 0.0% (0)** |
| qwen2.5:14b grounded / hallucinated | 96.0% / 0.5% | **99.0% / 0.5% (1)** |

Under a neutral judge, **both models are near-perfectly faithful on off-topic miss contexts — a wash**
(gemma3 0 fabrications, 14B 1). gemma3 does NOT hallucinate more than the 14B; the scare was 100% a
judge artifact. They differ only in style (14B abstains 98%, gemma3 79% but stays grounded). The
abstention rate is regex-based / judge-independent and unaffected by the re-judge. **Methodology
lesson, re-confirmed: never judge close calls with a candidate model — the self-preference swing here
was 24 percentage points.**

**Honest limitation:** this corpus tests the EASY failure mode (off-topic documents, which both models
trivially abstain on), not the production-relevant HARD one — a near-duplicate sibling that discusses
the topic with a *different* value. So the bake-off's slight gemma3 edge on 13 *real* near-duplicate
misses is neither confirmed nor refuted at scale. That hard case is deferred to Tier 1's real-harm
measurement (it only exists when a parameter varies across siblings — for uniform ones like Merit=60
any sibling answers correctly, so a "miss" is not a harm), which is more production-relevant than a
synthetic v2 corpus.

## Tier 0 verdict

The gemma3:12b production decision is **confirmed**: validated end-to-end (no follow-up/multi-turn
regression) and, under a neutral judge, at-worst-a-wash and fabrication-free on misses. The two things
Claude flagged as unverified are now verified and both landed favourably. Retire from the writeup: the
"hallucination drops ~31%→~8%" phrasing (the underlying n=13 subgroup gap is one turn) — replaced by
the n=200 neutral-judge finding that gemma3 is fabrication-free on off-topic misses and net-flat on
conversational retrieval.

---

# Round 6 Tier 1: the variance oracle - real-harm quantified + a stakes classifier (2026-07-25)

Claude's one genuinely-new, non-falsified mechanism: reuse the Class F extraction as an ORACLE (input
to a decision) rather than an output format. `eval/variance_oracle.py`: for each RoA question, take the
sibling documents competing in its retrieval pool, extract what EACH one says the answer is, and
classify whether they AGREE (uniform - any sibling answers correctly, a miss is harmless) or DIFFER
(varying - the specific document matters, a wrong sibling gives a wrong answer). Run over all 80 RoA
turns (both sets), gemma3 extraction, gemma3 SAME/DIFFER classification.

## Item 4 - real-harm on the 27 current misses: "a miss is not a harm", quantified

| of 27 retrieval misses | count | meaning |
|---|---|---|
| HARMFUL (siblings DIFFER) | **9 (33%)** | wrong sibling => genuinely wrong value (60 vs 30, 40 vs 50, 30 vs 15) |
| HARMLESS (siblings agree) | **13 (48%)** | uniform answer (e.g. Merit=60 everywhere) - any sibling answers right |
| UNDETERMINED (0-1 sibling stated a value) | 5 (19%) | too few sibling values to judge |

**The real user-harm rate is ~11% (9/80), not the 34% (27/80) the raw strict-miss rate implies.**
Nearly half the misses are harmless because the answer doesn't vary across the siblings - the strict
metric was penalising retrieval for picking a different-but-equally-correct document. This is the
data-backed form of the point the Class F pilot first hinted at (Merit=60 uniform), now measured
across the whole miss set: the sibling problem is roughly a third as harmful as the miss count says.

## Item 3 - the variance signal is a STAKES classifier, not a miss predictor

Measured naively as a miss predictor, the variance-gated trigger does NOT beat the old fragmentation
trigger (precision 0.31 vs 0.34 ~ base rate) - expected, because answers vary on retrieval HITS too,
so "siblings differ" doesn't predict a miss. Its real, deployable value is different: it partitions
questions by stakes.

| stakes (all 80 RoA turns) | count | |
|---|---|---|
| HIGH - answer varies across siblings (DIFFER) | 29 (36%) | getting the right document matters |
| LOW - uniform answer (SAME) | 37 (46%) | any sibling answers; a miss is harmless |
| undetermined | 14 (18%) | |

This enables **variance-gated disclosure**: surface "based on <document X>; these rules differ by
programme, so confirm this is yours" ONLY on the 36% high-stakes turns, versus today's always-on J6
disclosure (100%). It fires exactly where a wrong sibling would cause harm, and stays quiet on the
~46% where it wouldn't - a principled, targeted improvement over blanket disclosure, and it covers
both the 9 harmful misses and the 20 high-stakes HITS (where retrieval succeeded but the user still
benefits from knowing which document was used). Full detail: `eval/variance_oracle_result.json`.

**Takeaway:** the novel mechanism works, but as a targeted-disclosure gate, not an ask-trigger. The
headline is Item 4's ~11% real-harm rate - the honest, much lower measure of how often the sibling
problem actually hurts a user.

---

# Round 6 Tier 2: value-level sufficiency metric + resolution harness (2026-07-25)

## Item 6 - value-level evidence sufficiency (replaces vocabulary rubber-stamping)

`eval/value_sufficient.py` (Claude/Gemini/DeepSeek converged): extract the atomic VALUE-IN-ROLE claims
from each gold answer ("60 - minimum weighted average for Merit") and ask, with the NEUTRAL phi4 judge,
whether a retrieved document states THAT value in THAT role - not whether the words co-occur.

| evidence-sufficient@6 | Overall | Policy | RoA |
|---|---|---|---|
| keyphrase-string (old headline) | 83.8% | 85.0% | 82.5% |
| judge-based (loose) | 93.8% raw | 97.5% | 90.0% |
| **value-level (phi4, value-in-role)** | **96.2%** | **97.5%** | **95.0%** |

The value-level number came out HIGHER than the loose judge, not lower - because it correctly credits
HARMLESS misses (a sibling supplies the right value even when the gold document wasn't retrieved), the
same phenomenon as Tier 1's real-harm finding. Spot-check of the 10 credited misses:
`ma_social_work[primary]` (Merit=60) is a legitimate credit (every PGT-masters sibling states 60);
but `glossary[follow_up]` is residual LENIENCY - the retrieved doc only mentions "*un*capped" marks,
which phi4 over-credited for a "capped mark cannot be exceeded" claim. Conclusion: value-level is a
genuine improvement on NUMERIC/threshold claims (the ones that matter) but retains some leniency on
DEFINITIONAL claims where "value in role" is inherently fuzzy. The honest right-value-available rate
is ~90-93% once the definitional over-credits are discounted - which CONVERGES with the two other
Tier-1/2 measures.

**Convergence (the real result):** three independent analyses now agree the effective retrieval-failure
rate is ~10%, not the 34% strict single-gold miss rate - variance real-harm ~11%, value-availability
~90-96%, evidence-sufficient ~91%. Strict hit@6 was measuring a single-gold test-set artifact, not
user-facing failure.

## Item 5 - resolution harness: D3's ceiling is 100%

`eval/resolution_harness.py`: hit@6 penalises asking, so measure RESOLUTION@2 instead - when the
first-turn guess MISSES, if the user supplies the missing programme identity (what D3's clarifying
question asks for), does retrieval resolve within one more exchange?

| RoA (n=40) | result |
|---|---|
| GUESS baseline (today) | 28/40 hit@6 (70.0%) |
| of the 12 misses, RESOLVED after clarification | **12/12 (100%)** |
| effective success if D3 asks and the user answers | **40/40 (100.0%)** vs 70.0% guessing |

Every single missed question resolves once the programme is named - zero genuine retrieval gaps remain.
This is the cleanest possible proof that the residual is UNDERSPECIFICATION, not a retrieval deficiency:
the missing information is exactly the programme identity, and supplying it fixes 100% of misses.
Caveat: this is the COOPERATIVE-user ceiling (the simulated user gives the exact identity from
metadata), so live resolution will be lower - but a 100% ceiling is a strong green light for D3.

## Tier 3 - abstention snapshot test: PASS

`eval/test_abstention_snapshot.py` (regression canary for gemma3's faithful abstention): 5/5
answer-absent contexts correctly abstained ("cannot be answered from the provided context"). The
property the generator was chosen for holds; this test guards against a future model/quantisation swap
silently breaking it.

## Round 6 overall

Retrieval is confirmed closed and the residual reframed honestly: the effective harm rate is ~10-11%
(not 34%), all of it underspecification, and 100% of it resolvable by asking the user for the programme.
The one new mechanism (variance oracle) is best deployed as a targeted-disclosure gate (fire on the
~36% high-stakes turns). The generator decision (gemma3:12b) is validated end-to-end and fabrication-
free under a neutral judge. Everything points to the same next step: D3 clarification/disclosure is the
lever, and it is a product judgment on live traffic, not another experiment.

---

# Round 6 Tier 1 follow-up: the variance MAP - a parameter-level lookup table (2026-08-07)

The variance oracle (Tier 1, above) answers "for THIS query's retrieval pool, do the sibling documents
agree?" - a per-turn, per-pool judgment. `eval/variance_map.py` builds the complementary STATIC table:
for 8 common rules-of-assessment parameters, independent of any specific query, is the value UNIFORM
across current RoA documents (so any sibling answers correctly - no disclosure needed) or does it VARY
(so the specific document matters)? Method: probe each parameter's fixed question against every current
RoA document that mentions it (gemma3:12b extraction, up to 30 docs/parameter), then classify the
collected values as UNIFORM/VARYING (gemma3:12b), flagging splits that just track UG-vs-PGT level.

| parameter | verdict | values (count) |
|---|---|---|
| Merit threshold | **UNIFORM** | 60 (21) |
| Distinction threshold | **UNIFORM** | 70 (14) |
| First-class / top classification | **UNIFORM** | 70 (3) |
| Module pass mark | VARYING | 50 (22), 40 (1), Pass/Fail (1) |
| Condonement threshold | VARYING | 40 (23), 50 (1) |
| Reassessment mark cap | VARYING | 50 (17), 40 (3) |
| Credits for the award | VARYING | 140 (5), 260 (4), 160 (4), 60 (2), 90 (1), 180 (1) |
| Permitted further attempts | VARYING | 2 (6), 1 (4), 60 (2) |

Full detail: `eval/variance_map_result.json`.

**3 of 8 parameters are UNIFORM** (Merit=60, Distinction=70, First-class=70) - confirms, at the
parameter level, Tier 1's finding that a wrong sibling is often harmless: for these three, ANY RoA
document answers correctly regardless of which programme's sibling retrieval surfaces. These are safe
to answer confidently with no disclosure.

**5 are VARYING, but not uniformly interesting.** Credits-for-the-award and permitted-further-attempts
vary *by design* (credits scale with programme length/type; resit attempts are a genuine policy
difference) - these are exactly where the sibling problem bites and targeted disclosure earns its
keep. Module pass mark, condonement threshold, and reassessment cap each show a dominant value with a
small number of outliers (1-3 documents) - worth a manual spot-check to confirm those aren't stale
editions or genuine programme exceptions before trusting the dominant value as a fallback.

**Data-quality flag, not a real finding:** "Permitted further attempts" includes a value of "60" (2
documents) - implausible as an attempt count and almost certainly a misextraction (the probe likely
pulled a credit or page figure instead). Doesn't change the VARYING verdict (1 vs 2 already differ) but
should not be trusted as a literal value without a manual check of the source documents.

**Use:** this table is a fast, static reference for scoping the variance-gated disclosure (Tier 1, item
3) - parameters here can be hardcoded as UNIFORM/VARYING rather than re-derived per query, cutting the
oracle's per-turn extraction cost for these 8 common cases.

---

# Real-user feedback fix: contextualizer topic-switch injection (2026-08-07)

First look at `data/feedback.jsonl` (32 real thumbs up/down ratings collected since the UI shipped)
turned up a live bug distinct from anything the offline eval harness probes: full analysis in
`eval/FEEDBACK_FINDINGS.md`. The largest cluster (~6 of 17 thumbs-down) was the contextualizer treating
a genuine topic SWITCH as a follow-up needing prior-turn context, injecting unrelated programme names
into the rewritten retrieval query. Two cases reproduced directly against live code:

- "what is the minimum and maximum duration for a phd?" (after an MSc Artificial Intelligence question)
  came back rewritten with five unrelated MSc programme names bolted on from the prior turn.
- "duration of phd" (after a Professional Doctorate question) got anchored back to Professional
  Doctorate - a different award entirely.

**Root cause:** two gaps in the existing rewrite guards (`_is_faithful_rewrite`,
`ALIAS_ANCHOR_GUARD_ENABLED`). (1) The faithfulness check only verified the rewrite RETAINED the
original's words - a rewrite that keeps every original word but ADDS unrelated content passes
unchecked. (2) "phd" is deliberately excluded from the distinctive-identity-token vocabulary (its
docfreq across department records is too high to anchor ON), but that same genericness meant a
question naming it wasn't recognized as already self-sufficient, so the anchor guard wrongly
re-attached the stale prior-topic anchor.

**Fix (`src/rag.py`):** `_has_extraneous_family` rejects a rewrite that introduces identity tokens for
a programme family beyond the single legitimate history anchor (same one-anchor model
`ALIAS_ANCHOR_GUARD` already uses) - the multi-programme-injection case. `_names_award_type` broadens
"this question already names its own topic" to a small award-type vocabulary (phd, mphil, professional
doctorate, doctor of philosophy) so these no longer trigger a wrongful anchor re-append even though
they're too generic to be identity anchors themselves.

**Validation:** both reproduced bug cases now come back clean (verified directly, and case 2 stable
across 4 repeated runs despite LLM sampling). Three legitimate-follow-up shapes checked for regression
- pronoun resolution, explicit topic switch, elliptical continuation - all still behave correctly. Full
30-turn multi-turn regression (`eval/run_multiturn_eval.py`, `results_set4_multiturn_contextfix.json`)
vs the established gemma3 baseline: **zero hit@6 regressions, one turn improved (28/30 -> 29/30)** - all
rewrite-text diffs are harmless LLM paraphrase noise, not topic drift.

---

# The hit@6 blind spot: what the metric cannot see (2026-08-07)

Chasing three thumbs-down on "when is an independent chair required" surfaced a structural gap in
the harness rather than a bug in the system. `score_retrieval()` in `eval/run_eval.py` compares
**document URLs**, so hit@6 answers "was the right DOCUMENT retrieved" - not "were the
answer-bearing FACTS put in front of the generator". A turn can score a clean hit while the user
gets a wrong or empty answer, and that entire failure class is invisible to every number in this
report.

## Sizing it (no new runs - computed from committed results)

| turns | main set (n=80) | set 2 (n=80) |
|---|---|---|
| hit@6 TRUE + useful answer (judge >=3) | 62 (77.5%) | 51 (63.7%) |
| **hit@6 TRUE + failed answer (judge <=2)** | **7 (8.8%)** | **3 (3.8%)** |
| hit@6 FALSE (real retrieval miss) | 11 (13.8%) | 26 (32.5%) |
| **headline hit@6 vs actually-useful** | **86.2% vs 77.5%** | **67.5% vs 63.7%** |

**hit@6 overstates end-to-end usefulness by 8.7 points on the main set**, 3.8 on set 2 - about 6% of
all 160 turns are scored as successes the user would call failures.

## Splitting the blind spot by cause (`eval/chunk_blindspot.py`)

For each hit-but-failed turn, re-run retrieval with that turn's own stored `retrieval_query` and ask
where the gold keyphrases are - present in the retrieved chunks, present in the gold document but
not retrieved, or absent from the gold document entirely.

| verdict | n | meaning |
|---|---|---|
| **GENERATOR** | 6 (60%) | keyphrases WERE in the retrieved context; the answer failed anyway - synthesis, not retrieval |
| **CHUNK_MISS** | 2 (20%) | keyphrases are in the gold document but in a chunk that was never retrieved - right document, wrong chunk. No generator can fix this |
| **WEAK_TEST** | 2 (20%) | keyphrases appear nowhere in the gold document, so the item cannot be satisfied - a defect in the question set, not the system |

**The largest slice is generation, not retrieval.** That matches the live evidence: given the chunk
containing all six independent-chair criteria, gemma3:12b reported two while claude-sonnet-5 reported
all six (2026-08-07 provider test). So ~60% of this blind spot is addressable by generator choice
alone, and Round 5's "retrieval frontier is closed" conclusion survives - the residual here was never
mostly retrieval.

**Two eval-data defects found in passing.** `independent-chairs-policy.pdf` [primary] scores
`keyphrase_in_gold_doc = 0.0` while `keyphrase_in_retrieved = 0.25` - its expected keyphrases do not
appear in its own gold document but do appear in others, so the item can never be satisfied and has
been silently depressing scores. `roa-ug-integrated-masters-4yr-year-1.pdf` [follow_up] is the same
shape. Both need rewriting before they are counted again.

## What this changes

- **Report both numbers.** hit@6 alone overstates; pair it with the useful-answer rate (hit AND
  judged >=3), which is the honest end-to-end figure.
- The remaining retrieval work is **chunk-level**, not document-level - and it is small (2 turns
  across 160), consistent with the closed retrieval frontier.
- It plausibly contributes to the gap between ~90% eval scores and 47% real-user satisfaction,
  though the feedback failures (multi-entity, topic-switch) are a separate and larger cause.

## Correction and follow-up: what the WEAK_TEST turns actually were (2026-08-08)

Investigating the two turns `chunk_blindspot.py` labelled WEAK_TEST showed the label conflated two
different things, and one of them is a real defect in the index rather than in the question set.

**1. `independent-chairs-policy.pdf` [primary] - the GOLD ANSWER IS WRONG.** The question asks how
many staff a department with *more than 149* PGR students must nominate. The policy's table reads
`1-99 -> 2`, `100-149 -> 3`, `150+ -> 4`, so the answer is **4**. The expected answer says **3**.
The keyphrase `"3 members of staff"` was never going to be found because the document does not say
it. This item has been penalising correct behaviour and should be corrected to 4 before it is
counted again.

**2. `roa-ug-integrated-masters-4yr-year-1.pdf` [follow_up] - CONTENT MISSING FROM THE INDEX.** Not a
test defect at all. The source text says *"...they must withdraw from the University in any of the
following situations: - Where the Year Mark is below 20; ..."*, and the indexed chunk reads
*"...they - Where a student was absent fr..."* - **text dropped mid-sentence inside the chunk body**,
not at a boundary. The words `withdraw` and `maximum period` appear nowhere in any of the document's
18 chunks despite being in the cached source. The progression/withdrawal rule is therefore
unretrievable, and no generator or reranker can recover it.

This also means `gold_document_text()` in `chunk_blindspot.py` reads the INDEX, so its WEAK_TEST
verdict means "keyphrase absent from the indexed document" - which covers both a bad question and a
document whose content never made it in. Those need separating in any future run.

### How widespread is the content loss? Not established - and a first estimate was wrong

A sentence-level scan over 40 sampled current documents suggested 14/38 (37%) had source text missing
from the index. **That figure is not trustworthy and is retracted.** Verifying six of the flagged
documents with bare distinctive keywords (immune to normalisation artefacts) showed **five were false
positives**: the scan had split the cached text into "sentences" that interleaved navigation and page
furniture with prose, producing strings that legitimately never appear in the cleaned chunks.

Only `roa-ug-integrated-masters-4yr-year-1.pdf` survives verification - **1 confirmed case out of 38
sampled documents**. The true rate is somewhere at or above that and is not yet known.

**Recommended next step (not attempted):** a proper audit comparing cached source against indexed
chunks with the *same* cleaning the ingest applies, rather than ad-hoc normalisation. Until that runs,
treat this as one verified defect and an open question, not a corpus-wide claim. If it does turn out
to be widespread it would be a more likely explanation for residual retrieval misses than anything in
the Round 5 retrieval bake-off, since no amount of reranking can retrieve text that was never indexed.

---

# The clean_text content-deletion bug, and what fixing it did (2026-08-08)

## Root cause: not a stale index, a live bug

Chasing the missing withdrawal rule, I first concluded the index was stale and re-embedded. **That
changed nothing** - because `reembed.py` runs the same `clean_text()`. The defect is live in the
ingest path.

`clean_text()` removed any line shorter than 120 chars repeating >= 5 times, on the assumption that
repetition means a running header/footer. **Rules-of-assessment documents are built from parallel
per-year sections, so genuine policy clauses recur verbatim** - and the rule deleted them from EVERY
occurrence. For `roa-ug-integrated-masters-4yr-year-1` all six of its matches were content and not
one was a header:

| times | line removed |
|---|---|
| x10 | `or` |
| x9 | `and` |
| x6 | `must withdraw from the University in any of the following situations:` |
| x6 | `- Where the Year Mark is below 20;` |
| x5 | `- Where a student cannot complete their studies within the maximum` |
| x5 | `period.` |

Losing the operative verb made "what happens if I fail to progress" unanswerable **from the entire
corpus** - no retriever or generator could recover it. Corpus-wide the rule stripped **2351 distinct
line-types, of which only ~10 matched any page-furniture pattern.**

**Fix:** a repeated line is now removed only if it also *looks* like furniture (page marker,
contents/nav line, bare number), with a prose veto for anything ending in sentence punctuation or
opening with a bullet/enumerator. Biased toward keeping text - a surviving running title costs slight
chunk dilution, a deleted clause is unanswerable and invisible.

**Effect on the corpus:** 20,436 -> **21,283 chunks (+847)**, and the stale-index audit goes from
72/245 documents missing content to **0/245**.

## Effect on retrieval: +1 net, and that number is misleading

Measured with `eval/retrieval_replay.py` - retrieval only, replaying each turn's stored
`retrieval_query`, so the index is the only variable and there is no generation or judging noise.
The before pass reproduced the committed baseline exactly (86.2% / 67.5%), confirming the harness is
faithful.

| | before | after |
|---|---|---|
| main set | 69/80 (86.2%) | 69/80 (86.2%) |
| set 2 | 54/80 (67.5%) | 55/80 (68.8%) |
| **total** | **123/160 (76.9%)** | **124/160 (77.5%)** |

Net **+1** (3 gained, 2 lost; rank changed on 5 turns held by both, 3 improved). On its face: a wash.

**But the headline turn makes the case for a better metric better than any argument could.** The turn
that hit@6 records as LOST is `roa-ug-integrated-masters-4yr-year-1 [follow_up]` - the very document
the fix repaired. Its gold document now falls outside the top 6, so hit@6 calls it a regression. Yet
checking the retrieved context directly:

- `withdraw` - **present**
- `year mark is below 20` - **present**
- `maximum period` - **present**

all supplied by sibling documents (year-2, year-3, the 22-v4 edition). **The user now gets a correct
answer to a question that, before the fix, was unanswerable from any document in the corpus** - and
hit@6 scores that change as negative.

## Conclusion

Two things stand. First, the bug was real, corpus-wide, and is fixed: content that existed in no
indexed document is now retrievable. Second, **hit@6 could not see any of it** - it recorded the
repair as a small net wash and the flagship case as a loss, because it measures which document was
retrieved rather than whether the answer was available. That is the same blind spot measured earlier
today (8.7 points on the main set), now demonstrated in the opposite direction: not only can a scored
hit be a user-facing failure, a scored miss can be a user-facing success.

Adopting the useful-answer rate, or a claim-level metric, is no longer a refinement - without it,
work like this fix is invisible or actively mis-signposted.

---

# Round 7: real-user feedback, a self-inflicted regression, and the cloud-component question (2026-08-08/09)

Distinct in kind from rounds 3-6, which were external-review cycles. This one started from 32 real
thumbs-up/down ratings in `data/feedback.jsonl` and followed the data. Three of its six findings are
negative, and one is a correction to a recommendation made earlier in the same round.

## 1. A guard I added cost 8.8 points, and how it was found

Fixing a real user complaint (a topic-switched question answered from the previous topic), I added
`_has_extraneous_family` to reject contextualizer rewrites that introduce unrelated programme names.
It shipped **enabled** on the strength of two hand-checked cases — against this project's own
convention that new mechanisms default off.

The full eval caught it, but only when split by turn type:

| | baseline | with guard | after disabling |
|---|---|---|---|
| primary hit@6 (no contextualizer) | 78.8% | 80.0% | 80.0% |
| **follow-up hit@6** | **75.0%** | **66.2%** | **75.0%** |

`data/contextualizer_rejects.jsonl` confirmed the mechanism: **33 rejections in a day against 1 in
all prior history**, each falling back to the raw context-free question. The rejected rewrites were
*correct* — naming the programme carried in from history is the contextualizer's whole job.

The design cannot be repaired by tuning: `_family_labels_named` counts document FAMILIES and one
programme spans many, so naming one programme is indistinguishable from naming several. Raising the
threshold to >=2 recovered only 3 of 33. Disabled.

**Two lessons.** Validate on the metric the change can actually move — the 30-turn multi-turn probe
used to justify it is topic-switch-heavy and structurally blind to follow-up regressions. And
primary turns are a free control for anything touching query rewriting.

## 2. Cloud components: one win, two nulls

All measured on the same 80 questions, same local judge, one variable at a time.

**Contextualizer (claude-sonnet-5 vs local qwen2.5:7b) — NULL.** Follow-up hit@6 75.0% -> 76.2%
(one turn), useful answer +0.0. Primary turns identical to two decimals, confirming isolation. Query
rewriting is not capability-bound here; Round 5's local sweep had already found the same flatness.

**Generator (claude-sonnet-5 vs local gemma3:12b) — REAL, and concentrated where it was not expected.**

| mean judge score | gemma3 | sonnet-5 |
|---|---|---|
| primary, hit turns | 4.19 | 4.31 |
| follow-up, hit turns | 3.77 | 3.92 |
| **primary, MISS turns** | **3.50** | **4.00** |
| **follow-up, MISS turns** | **2.90** | **3.62** |

The gain is almost entirely on turns where retrieval *failed* — extracting a usable answer from
sibling documents. Inspection of high-scoring miss turns shows hedged extraction, not fabrication
("the excerpts describe the concept of capping but do not include an explicit definition"), so
Round 4's "stronger models are more confidently wrong on misses" does not reproduce here — as
Round 5 already found for gemma3 and gpt-oss.

**Generator (claude-haiku-4-5) — NULL, and below local.** Primary mean 3.94 vs gemma3's 4.05,
follow-up mean identical at 3.55. It recovers about half the follow-up miss-turn gain (3.30 vs
sonnet 3.62) but is worse than local on primary miss turns. Extraction-under-retrieval-failure looks
like a Sonnet-tier capability; halving cost is not available.

## 3. Correction: the useful-answer rate is the wrong headline

Earlier in this round the useful-answer rate (hit@6 AND judge >= 3) was recommended as the headline
metric, on the strength of the 8.7-point blind-spot finding. **The generator experiment falsifies
that recommendation.** The metric *requires* a hit, so it is structurally blind to miss turns — which
is exactly where the generator gain lives. Sonnet's +0.50/+0.72 on miss turns registers as +0.0 on
the useful-answer rate.

**Mean judge score is the better headline**: it counts partial credit on miss turns, which is where
real user value sits given ~half of misses are harmless (Round 6 variance oracle). Report hit@6 for
retrieval, mean score for end-to-end quality, and treat any hit-gated metric as a retrieval measure
wearing a quality costume.

## 4. The judge flatters follow-ups

Re-judging the same 160 answers with claude-sonnet-5 (~$0.60): mean scores near-identical (3.80 vs
3.78) but the useful-answer rate drops 71.2% -> 62.3%. Per-turn the judges agree on 70 turns, with
Sonnet stricter on 41 and *more lenient* on 48 — it redistributes rather than tightening, and the
redistribution straddles the >= 3 threshold.

Split by turn type it is not symmetric: Sonnet rates primary answers **higher** (4.05 -> 4.24) and
follow-ups **lower** (3.55 -> 3.32; useful 66.2% -> 51.9%). The local judge is most generous exactly
where the system is weakest and the judgement is hardest. Real follow-up performance is worse than
the ledger records.

**Never mix judges within one comparison** — the judge alone moves the threshold metric by ~9 points,
dwarfing most effects being measured.

## Round 7 verdict

Production runs the cloud generator (measured) and a Haiku contextualizer (latency-measured, quality
inferred from the Sonnet null); embedding and reranking stay local as no API alternative exists.
Retrieval is now ~40% of a ~9-12s turn and is the remaining latency lever. The follow-up gap is
**neither** retrieval nor query rewriting — both were falsified this round — and is only partly the
generator; on hit turns a ~0.4 mean gap survives every swap tried.

---

# Round 8 — PGT/PGR question set 3 (2026-08-09)

**Not comparable to the ledger above.** Set 3 is a new 23-question file scoped to PGT/PGR, run on
production's configuration (cloud Sonnet generator, Haiku contextualizer) with a frontier judge.
Different questions, different generator, different judge — no number here should be diffed against
a Round 1-7 number. It exists to characterise *this* question type, not to move the ledger.

Composition: 5 factual, 15 interpretive, 3 abstention. The interpretive questions test rule
COMBINATION and authority boundaries, not lookup, because that is what the feedback log shows
failing. Abstention items are held out of all retrieval metrics (no gold document).

## Results

**RETRACTED AND RE-MEASURED (same day).** The first version of this section
reported mean judge 4.62 for the cloud run. That run was generated by
claude-sonnet-5 and judged by claude-sonnet-5 - the same-family self-judging
this ledger already forbids ("never judge close calls with a candidate model").
Re-judged with phi4 (cross-family to BOTH generators) the same answers score
**4.20**. The self-preference was worth **+0.42** on the headline. The rule was
in CLAUDE.md, written from a prior 24-point incident, and was still walked into
- when the judge and the generator are configured two steps apart it does not
announce itself. Every number below is phi4-judged.

| bucket | n turns | ev-suff@6 | strict hit@6 | mean judge (phi4) |
|---|---|---|---|---|
| overall (excl. abstention) | 40 | 100.0% | 97.5% | 4.20 |
| primary | 20 | 100.0% | 100.0% | 4.45 |
| follow-up | 20 | 100.0% | 95.0% | 3.95 |
| abstention | 6 | n/a | n/a | 5.00 |

**Interpretive questions are not harder than factual ones for this system** (4.60 vs 4.70, n=10 vs
30 — the gap is well inside noise at this sample size). The prior assumption that rule-combination
questions would expose a weakness the factual set hides is **not supported** on set 3. Note the
confound: these were authored against documents, so every one is answerable from a single document.
A genuinely corpus-spanning interpretive set is untested and is the honest version of this probe.

**Abstention behaviour is correct: 6/6 turns, mean 5.00 under the neutral judge.** The system declined to answer questions
with no corpus basis (fee amounts, named Independent Chairs, full-time/part-time PhD switching)
rather than confabulating from adjacent provisions. This was the failure the user reported from real
use, and it does not reproduce under the cloud generator. The lexical abstention marker fired on only
67% of those turns, which is why the marker is a tripwire and the judge is the metric.

## The remaining failure mode is chunk-level, not document-level

Only 3 of 40 turns scored <= 3, and **two share one root cause: the fact was in the retrieved
document but not in the retrieved chunk.**

- *"What is expected of a student entering a completion period?"* — score 1, **hit@6 = True**. The
  sentence is in the retrieved document verbatim ("Students permitted to enter a completion period
  ... do so on the understanding that their full thesis will be ready for submission by the end of
  that completion period"). The system said the context did not address it. It abstained rather than
  confabulating, which is the right behaviour on a chunk miss, but the user still gets nothing.
- *"Can supervision meetings be held over Zoom [for an immigration-sponsored student]?"* — score 3,
  hit@6 = True. Both required phrases are in the document. The answer surfaced the general
  virtual-meetings clause and the immigration caveat, but missed "the majority of contact should be
  face-to-face" — so it led with "Yes, but" where the document leads with a restriction.

With document-level retrieval at 97.5% on this set, **document-level metrics have no headroom left
to measure and chunk selection is where the remaining loss is.** This is the blind spot documented
in Round 4 (8.7 points on the main set), now the dominant term rather than a secondary one.

The third failure is different and partly a question defect: *"How many credits can be condoned in
total?"* (hit@6 = False, score 3) is underspecified without the primary turn's programme context,
and the system answered across programmes — but it also conflated "credits that may be failed and
still pass" (90 of 120) with condonement (20 credits, non-core, mark >= 40). That conflation is
real and worth a targeted probe; the question wording is not clean enough to call it alone.

## Method note

`expects_abstention` items record rank/hit as `None`, not `False` — `False` reads as a retrieval
failure in any aggregate that sums it, `None` forces exclusion. Building them surfaced a resume bug:
`run_eval` keyed resume state on `source_url`, which is null for every abstention item, so all three
collapsed to one key and two would be silently dropped on any resumed run. Now keyed on
`(source_url, question)`.


## Local vs cloud generator on set 3 (both phi4-judged, like-for-like)

| | cloud (sonnet) | local (gemma3) | delta |
|---|---|---|---|
| overall mean | 4.20 | 3.98 | +0.22 |
| primary | 4.45 | 4.10 | +0.35 |
| follow-up | 3.95 | 3.85 | +0.10 |
| **abstention** | **5.00** | **3.33** | **+1.67** |
| strict hit@6 | 97.5% | 95.0% | +2.5pts |
| latency / question | ~20s | ~105s | 5.3x |

**The general quality gap is small: +0.22 overall.** The mixed-judge comparison
first computed (sonnet-judged cloud vs qwen-judged local) showed +0.52 - more
than half of the apparent generator advantage was judge artifact. Anyone
choosing a generator on that number would have been buying ~0.2 points of real
quality at ~$4.75/month while believing they were buying 0.5.

**The abstention gap is NOT small: +1.67, and it reproduces the user's reported
bug.** Asked what paperwork is involved in switching between full-time and
part-time PhD registration - not covered anywhere in the corpus - the local
generator answered from Supervisory Panel and RSPB confirmation-of-PhD-status
provisions (judged 1). The cloud generator said the context does not cover it
(judged 5). This is the failure that motivated the abstention items, and it is
**generator-specific, not retrieval-specific**: both configurations retrieved
the same chunks, since retrieval is local in both.

Practical consequence: the cheap-generator option is cheap on ordinary questions
and expensive exactly where a policy assistant is most dangerous - confidently
answering a governance question the policies do not address.

**hit@6 95.0% vs 97.5% is a contextualizer difference, not a retrieval one.**
Embeddings, lexical and ColBERT are identical across both runs; only the
follow-up query rewriter changed (local qwen2.5:7b vs claude-haiku-4-5), and
the 2.5-point gap sits entirely in follow-up turns.
## Round 8b — stale-edition alias fix (2026-08-09)

Traced from the set-3 completion-period failure (judged 1 with hit@6=True):
slots 2 and 3 of the top 6 were the **2020-21 archived edition** of the PGR
Code of Practice, crowding the 2025-26 edition into one slot holding the wrong
chunk. Retrieval filters `is_current=True`, so archived editions should not have
been reachable at all.

Cause: rename-splits `document_family()` cannot see. The PGR Code was renamed
twice over its life (`code-practice-X` -> `code-of-practice-X` ->
`code-practice-X-degrees`), producing THREE family keys for one lineage, each
electing its own newest member as current. Same defect on Academic Appeals
(the "and" in `undergraduate-and-postgraduate-taught` dropped for 2022-24 and
re-added). `audit_family_aliases.py` cannot propose these: it merges only stems
differing by non-structural tokens, which is the conservatism that stops it
merging `4yr-year-1` with `4yr-year-2`. Whole-word renames need the hand-audited
map.

Verified as single lineages by YEAR-CONTIGUITY, not name similarity: the PGR
Code runs 2015-16..2025-26 unbroken across its three keys. Two parallel
documents cannot produce that.

**Effect: 3 documents demoted, 0 spurious promotions (245 -> 242 current).**

| | pre-alias | post-alias |
|---|---|---|
| hit@6, 160 turns (main + set 2) | 124/160 (77.5%) | 124/160 (77.5%) |
| per-turn | — | gained 0, lost 0, 3 ranks improved |

Neutral on hit@6 and non-regressive, which is the expected shape: the demoted
editions are SIBLINGS of the gold documents, so crowding them out frees chunk
slots without changing which document is judged correct. The completion-period
query now retrieves its gold chunk at **rank 1** (was absent from the top 6).
The set-3 Zoom/immigration failure is **unchanged** - that one is genuine
chunk ranking, which this fix was never going to touch.

### Method error worth recording

The first verification compared a post-alias replay against `hit@6` stored in
the older `results_gemma3_e2e_*` files, and reported "3 gained, 2 lost". That
comparison is invalid - the stored baseline was measured on a different index
state after several intervening changes, so nothing in it is attributable to
the alias change. Re-run properly (aliases disabled, flags recomputed, same
replay, then restored) the true answer is **0 gained, 0 lost**.

The two apparent "losses" were wrong-year-of-study siblings (`mlang-year-3`
holding four of six slots on one query) - a real and separate defect, but not
caused by this change. Reported as caused by it, the obvious response would
have been to revert or complicate a fix that is in fact clean. `retrieval_replay`
is cheap precisely so it can be run on BOTH sides; one side is not a baseline.

## Round 8c — FALSIFIED: duplicate-chunk crowding is not a failure cause

Hypothesis (from one vivid case: `roa-ug-mlang-year-3.pdf` holding four of six
slots while the year-1 gold was missed): near-duplicate siblings and repeated
chunks of one document waste top-6 slots and cause misses. Proposed fix was a
per-document slot cap. **Both halves are falsified.** 160 turns, retrieval-only.

| | hits (n=124) | misses (n=36) |
|---|---|---|
| turns with duplicate-document slots | **86.3%** | 52.8% |
| mean DISTINCT documents in top 6 | **3.10 / 6** | **4.64 / 6** |
| duplicated document is the gold | 92 of 107 | 2 of 19 |

**Concentration predicts success; diversity predicts failure - the opposite of
the hypothesis.** The system succeeds by locking onto one document and returning
several of its chunks (gold repeated on 92 of 107 duplicating hits). Misses are
*more* diverse, averaging 4.64 distinct documents: on a miss the pool is spread
across more documents, not fewer. So the gold is not being crowded out of a
slot it would otherwise have won - it simply never ranks.

A per-document slot cap would have attacked the mechanism that hits depend on.
Shipped on the strength of the mlang case alone - which is exactly how
`_has_extraneous_family` was shipped, on two hand-checked cases, for -8.8 points
- it would plausibly have cost real accuracy.

The year-of-study sibling variant is also small: only **4 of 36 misses** (11.1%)
have a year sibling of the gold anywhere in the top 6.

**Consequence for where effort goes.** Chunk-level loss on set 3 is real (Round
8, 2 of 3 sub-4 turns were CHUNK_MISS), but it is not slot contention and no
slot-level reranking rule addresses it. The remaining misses need the gold to
*rank higher*, which is embedding/reranking work, not diversity work. Note also
that hits average only 3.10 distinct documents in six slots: for questions whose
answer genuinely spans documents, effective breadth is about half of N_RESULTS.
That is a separate and still-untested concern, not a defect.

## Round 8d — Set 4: corpus-spanning questions (2026-08-09)

Built to remove set 3's stated confound (every question answerable from ONE
document) and to test the breadth concern raised by 8c (hits average ~3
distinct documents in six slots). 10 PGT/PGR questions, each requiring facts
from 2 documents, verified mechanically: for every question NO single required
document contains all its keyphrases. Cloud generator, phi4 judge.

### hit@6 reports perfection on a set that is 15% short of its evidence

| metric | value |
|---|---|
| strict hit@6 (single gold doc), primary | **10/10** |
| mean span coverage@6 (all required docs) | **85.0%** |
| questions with ALL required docs in top 6 | **7/10** |
| mean distinct documents in top 6 | **2.80 / 6** |

Three questions are missing a required document outright, and the document-level
metric scores every one of them as a clean hit. This is the blind spot the
metric was built to expose, now demonstrated rather than argued: **hit@6 cannot
see a partially-evidenced answer.** Any future multi-document work must report
span coverage, or it will be measuring a number that is already saturated.

The 2.80 distinct documents figure also confirms 8c from the other direction:
the system concentrates on one document. That is *why* it does well on
single-document questions and it is the same behaviour that caps this set.

### But spanning questions are only slightly harder

| | set 3 (single-doc) | set 4 (spanning) |
|---|---|---|
| primary | 4.45 | 4.40 |
| follow-up | 3.95 | 3.70 |
| overall | 4.20 | 4.05 |

**Set 3's finding largely survives the confound.** Removing single-document
answerability moved the headline by 0.15 and primary turns by 0.05. The
prediction that spanning questions would expose a large hidden weakness is NOT
supported. As with set 3 the loss concentrates in follow-ups (3.70), which is
where every measurement in this project has put it.

Coverage does track quality in the expected direction - full-coverage questions
mean 4.57 vs 4.00 for partial - but **n=3 partial, which is far too small to
call an effect.** Recorded as directional only.

### The most informative failure is generation, not retrieval

Q7 follow-up ("is the Tavistock route different from an Essex-based
professional doctorate?") scored 1 with hit@6=True and FULL span coverage. Both
codes were retrieved and the system answered "No - it's the same route/process",
when the Dean and the reporting line differ (Dean of Partnerships / Portfolio
Manager vs Faculty Dean Postgraduate / Head of Department). Having both
near-identical documents in context did not help it distinguish them; arguably
it caused the error. That is a distinct failure mode from the crowding
hypothesis falsified in 8c, and it is not addressable by retrieval work.

Cost: ~$0.15. The retrieval half (span coverage) was free and generator-
independent, and was run FIRST so the generator run could be skipped if breadth
had turned out to be broken.

## Round 8e — Real feedback replay + near-duplicate probe (2026-08-10)

### Feedback replay: the "outdated sources" complaint is fixed

All 17 thumbs-down questions from `data/feedback.jsonl` replayed through
current retrieval (free, retrieval-only).

| complaint class | then | now |
|---|---|---|
| archived editions among sources | reported by user (item 16) | **0/17 questions** |
| partner documents among sources | reported by user (items 1, 15) | **4/17 questions** |

Item 16 ("Sources were older versions of the right document and also
Tavistock") is **half fixed by the Round 8b alias work**: no archived edition
appears for any of the 17 questions, but the Tavistock code still does. The
partner half is unresolved - `PARTNER_INSTITUTION_DEMOTE_ENABLED` demotes but
does not exclude, and 4 of 17 real complaints still show a partner document
(kol-pg-masters-roa-25, sak-principles, tavistock x2).

The remaining down-votes are dominated by ENUMERATION failures (items 5-10:
"I gave you 6 schools, you answered about one"), which are generation-side and
untouched by any retrieval work this round.

### Near-duplicate confusion does NOT reproduce - Q7 was not systematic

Set 4's Q7 (scored 1: "No - it's the same route/process" when Tavistock and
Essex professional doctorate routes differ) motivated a discrimination probe:
6 standalone questions naming ONE institution where the answer differs between
the near-identical codes.

**6/6 correct.** Two initially scored CONFLATED, and both were my scoring
error, not the system's: the Essex code genuinely defines "Dean" as "the
Executive Dean (or their deputy) or the Dean of Partnerships", so quoting that
is accurate, and the second answer's Tavistock mention was a correctly-labelled
contrast. The check penalised a term APPEARING rather than being misattributed.

Re-testing the actual Q7 shape (contrast asked as a conversational follow-up):
- Essex -> Tavistock contrast: **correct** ("Not quite - the terminology and
  reporting line differ"), naming both distinct roles.
- Tavistock -> Essex contrast: **incomplete but honest** - "I can only confirm
  details for the Tavistock route" rather than Q7's confident "it's the same".

**No mechanism should be built for this.** One judged-1 answer did not
reproduce across 8 further attempts. Note the methodological limit: production
uses the cloud generator, which cannot be temperature-pinned, so a single bad
answer is not evidence of a stable defect and a non-reproduction is not proof
of absence. Both directions of that claim need n>1 here.

## Round 8f — Enumeration failures are retrieval; the leak is prose (2026-08-10)

### The enumeration complaints are a retrieval ceiling, not a prompt failure

Feedback items 5-10 ("I gave you 6 schools, you answered about one"). Measured
rather than assumed:

| | |
|---|---|
| slots returned for the 6-department question | 6 |
| DISTINCT documents among them | 3 |
| departments represented | **1 of 6** (CSEE only) |

The old answer's claim - "the provided context ONLY contains information
regarding the CSEE department" - was **true**. `MULTI_ENTITY_COVERAGE` is a
prompt rule and cannot conjure documents; it was already on and could not help.

**The documents do exist.** A first pass matched department tokens against
FILENAMES and found 0 documents for MSAS, HSC and Life Sciences, which would
have meant nothing to fix. Searching document TEXT instead: MSAS 4, HSC 25,
Life Sciences 11, SRES 13, Psychology 24, CSEE 21. The filename count was the
artifact - the same class of error as the "37% missing content" retraction.
So retrieval is genuinely leaving ~98 candidate documents on the floor and
returning 3 CSEE files. Per-entity fan-out is justified; note the adjacent
`MULTIHOP_DECOMPOSITION_ENABLED` was rejected (-7.5pts RoA) but on a DIFFERENT
trigger - it inferred candidates from fragmentation, whereas enumeration
questions name their entities explicitly and require no hypothesising.

### Sonnet does not fix retrieval - it changes the consequence

Re-running the complaint under the current cloud generator: sources are still
all-CSEE, 1/6 departments.

**CORRECTION (same day).** This section first claimed the old answer left the
user to notice the missing schools themselves, and that the gain was honesty
about coverage. That is **wrong** - the original 2026-07-27 answer already
ended "The documents do not list accredited programs for MSAS, Psychology, HSC,
SRES, or Life Sciences", naming all five. The local generator was already doing
that. The claim was written without re-reading the logged answer it described.

The differences that survive checking:
- **A factual fix.** The old answer listed "MSc Computer Systems Engineering
  (GH64)" as a 9th ACCREDITED programme. The source files it under
  "Non-accredited Masters Award Title", alongside MSc Computing (G456), as the
  exit award for students who miss accreditation requirements. The new answer
  gets this right. The user's own comment was "CSEE's answer was good" - the
  error was in the part nobody complained about.
- **Ordering and next step.** The coverage gap now leads instead of trailing,
  and the answer names where the missing information would live.
- **Follow-up behaviour.** The old follow-up replied "You are absolutely correct
  to call me out on that. My apologies" and repeated CSEE. That capitulation is
  gone; the current answer restates the limit.

So fan-out is NOT deprioritised on the grounds first written here. The user
still cannot get the answer they asked for, and the gap was always stated -
what improved is accuracy within the one department retrieval could reach.

### USER_FACING_LANGUAGE - enabled after measurement

Production answers described their own plumbing: "the context you've provided
across both turns", "if you have excerpts, please share them". The user
supplied nothing - the retriever did.

Like-for-like A/B, same 4 turns, same detector:

| | OFF | ON |
|---|---|---|
| answers leaking plumbing | 4/4 | **2/4** |
| still declining correctly | 4/4 | **4/4** |

Enabled. Abstention is unchanged, which was the risk - a rule about how to say
"I don't have this" sits directly on top of measured abstention behaviour.

Two honest limits. **The residual is real**: "the excerpts I can see" survives
in secondary sentences despite an explicit instruction not to say it; only the
opening framing is reliably fixed. And **the original defect was not in the
A/B at all** - "please share them" appeared 0/4 in BOTH arms because it
occurred on a follow-up and the probe used primary turns. Retested directly on
the follow-up with the rule on: no request to supply documents, no "context
you've provided", no "excerpt", and it now points the user to where the
information would live.

**No metric in this ledger would have caught this defect.** hit@6, span
coverage, judge score and keyphrase coverage all score these answers
identically before and after - the facts were right both times. It cost trust,
not accuracy.

## Round 8g — Multi-entity retrieval: coverage 1/6 -> 5/6 (2026-08-10)

Closes the COVERAGE half of the six-school failure that the
`MULTI_ENTITY_COVERAGE` note called "not fixable by prompting". `N_RESULTS`
caps CHUNKS, not documents: the real question returned 6 chunks resolving to 3
documents, all CSEE.

`src/entities.py` maps department aliases (CSEE, MSAS, HSC, SRES...) to the
corpus's own `department` metadata values, which are inconsistent enough
(case variants, singular/plural, parenthesised abbreviations) that the mapping
is curated and auditable rather than inferred. When >= 2 departments are named,
each gets a reserved slot budget filled by a retrieval FILTERED to its metadata
values; the rest of the widened set (cap 14) comes from the ordinary ranking.

| | OFF | ON |
|---|---|---|
| chunks / distinct documents | 6 / 3 | 14 / 8 |
| departments covered | **1 of 6** | **5 of 6** |
| CSEE accredited programmes named | 8/8 | 8/8 |
| latency on a triggering question | 26.1s | 37.8s |
| hit@6, 160 existing turns | 124/160 | **124/160** (0 gained, 0 lost) |

### Two implementation defects found by measuring, not by review

**1. The fan-out bypassed the recency pass.** Per-entity results surfaced
`csee_ft_masters_accredited_variations_24` (2024-25) above the 2025-26 edition.
Not an is_current bug: both are flagged current because the older one lives
under `/ug/current/`, a path rule that deliberately overrides the family-max
rule. `is_current` is a coarse pre-filter, not a within-family ordering - the
unfiltered path applies `_prefer_most_recent_year` on top of it and this path
was not. Fixed.

**2. Interleaving cost accuracy on the one department that already worked.**
First working version reported only 5 of CSEE's 8 programmes as accredited,
demoting four genuinely accredited ones into the non-accredited exit-award
group - a REGRESSION on the part the unfiltered path got right. CSEE received
6 chunks either way, so this was not a budget effect: the reserved chunks sat
at the top and the fill chunks for the SAME document landed at the bottom,
separated by four other departments' material, and the generator read one
strong source as two weak ones. Grouping each document's chunks contiguously
restored 8/8 with the exit-award mapping correct.

Worth generalising: **the same chunks in a different order produced a
materially wrong answer.** Nothing in this ledger measures chunk ORDER - hit@6,
span coverage and evidence-sufficiency are all set-membership tests.

### Limits, recorded as limits

- **Life Sciences is still missed (5/6, not 6/6).** It appears in 11 current
  documents' TEXT but never as a `department` metadata value, so there is
  nothing to filter on and the query-side fallback does not surface it.
- **Evidence is thin: one target question and one control.** The safety
  argument for enabling is blast radius, not weight of evidence - the trigger
  fires on 0/160 existing eval turns (verified empirically by replay, not just
  by the detector), so no committed number can move and single-entity
  questions take the unchanged path. A multi-entity question set is the
  outstanding follow-up and would be the first set in this project able to
  measure the mechanism properly.
- Costs ~11s on triggering questions only.

## Round 8h — Set 5 validates multi-entity retrieval (2026-08-10)

Set 5 exists because Round 8g enabled `MULTI_ENTITY_RETRIEVAL` on one user
question plus one control, which was thin. 10 questions naming 2-3 departments,
including 2 deliberate TRAPS that name departments but whose answers live in
general documents - there to catch the fan-out starving the ordinary ranking.

### Retrieval (free, generator-independent)

| | OFF | ON |
|---|---|---|
| mean department coverage | 60.4% | **100.0%** |
| full coverage | 3/8 | **8/8** |
| traps: gold document retrieved | 1/2 | 1/2 (unchanged) |
| mean distinct documents | 4.3 | 6.1 |

The traps are unchanged, which is the result that matters most here: the budget
spent on departments does not come out of the general ranking. The remaining
1/2 is a pre-existing miss present in BOTH arms.

**Correcting the impression left by Round 8g.** That round reported 1/6 -> 5/6
from the CSEE case and it read as rescuing a system that was failing outright.
Across 8 questions the OFF baseline is **60.4%, not near-zero** - several
two-department questions already retrieved both. The mechanism closes a
40-point gap and makes coverage reliable; it does not rescue a broken system.

### End-to-end (cloud generator, LOCAL judge both arms)

| | OFF (qwen) | ON (qwen) | OFF (phi4) | ON (phi4) |
|---|---|---|---|---|
| primary | 3.70 | 3.80 | 3.20 | 3.60 |
| follow-up | 3.40 | 4.30 | 2.90 | 3.50 |
| overall | 3.55 | **4.05** | 3.05 | **3.55** |
| hit@6 primary / follow-up | 5/10 / 5/10 | **7/10 / 8/10** | (same retrieval) | (same retrieval) |

**CORRECTION (2026-08-10, same day).** This first read "the gain concentrates
in FOLLOW-UPS (+0.90 vs +0.10 on primary), the opposite of this project's usual
pattern". Re-judged with phi4 (neutral, free) that is mostly a JUDGE ARTIFACT:

| | primary | follow-up | overall |
|---|---|---|---|
| qwen2.5:14b delta | +0.10 | +0.90 | +0.50 |
| phi4 delta | +0.40 | +0.60 | **+0.50** |

What survives is the part that matters: **overall +0.50 replicates exactly
across two judges**, so the mechanism's benefit is real and judge-independent.
What does not survive is the interesting story - under phi4 the split is
+0.40/+0.60, a mild asymmetry, not an inversion of the project's follow-up
pattern. phi4 is also stricter in absolute terms (3.05 -> 3.55, vs qwen's
3.55 -> 4.05).

Recorded because the follow-up claim was the most eye-catching thing in this
round and it was written from a single local judge, hours after this same
ledger retracted a headline for exactly that reason.

### Three limits

- **Both judges now reported** (qwen2.5:14b and phi4). The +0.50 overall gain
  is identical under both; only the primary/follow-up split moves.
- **This set was built for this mechanism.** It validates that the mechanism
  does what it is designed to do. It is not evidence of general benefit, and
  the 0/160 trigger rate means it cannot be.
- **ON-arm latency is void.** Production was restarted mid-run (my error),
  putting a second ~5GB server up and driving free memory to 0.2GB; ON turns
  read 116s/98s against OFF's ~40s. That measures swap contention, not the
  mechanism. The clean figure from Round 8g (~11s added) stands; nothing in
  this run's timings does. Scores are unaffected - retrieval is deterministic
  and generation is cloud-side.

## Round 8i — Partner exclusion: 4/17 real complaints -> 0/17 (2026-08-10)

`PARTNER_INSTITUTION_DEMOTE_ENABLED` re-sorts WITHIN the already-chosen top 6,
so it never frees a slot. That is why partner documents still reached the user
on 4 of 17 real thumbs-down questions despite the demotion being on - the
complaint raised twice in the user's own words ("Answers need to focus on Essex
programmes, not partners").

`PARTNER_EXCLUDE_WHEN_UNNAMED` drops partner chunks from the CANDIDATE POOL
before rerank, so the freed slots are filled by Essex documents. Gated on the
conversation not naming a partner.

| | before | after |
|---|---|---|
| real complaints with a partner doc in top 6 | **4/17** | **0/17** |
| partner-held slots on those questions | 5.5% | 0.0% |
| hit@6, 160 turns (history-aware replay) | 123 | **124** (0 lost) |

The three complaint documents removed are exactly the ones the user named:
`kol-pg-masters-roa` on the MSc AI exit-awards question, `sak-principles` on
the longest-UG-degree question, and the Tavistock code on both independent-chair
questions.

### Why the gate reads the CONVERSATION, not the question

First version read only the current query and scored **net -1 (gained 1, lost
2)**. Both losses were follow-ups, and they were different failures:

- **Tavistock**: a real bug. The primary named the partner; the contextualizer's
  follow-up rewrite dropped it ("what actions should new Professional Doctorate
  students take after receiving..."), so the gate excluded partner documents on
  a turn whose own conversation was explicitly about a partner. Fixed by reading
  prior user messages.
- **SKKU**: not a bug. Neither turn names a partner - the question is "maximum
  non-core credits condoned across a three-year degree", entirely generic, with
  an SKKU gold by test-set assignment. Excluding partners there is arguably
  CORRECT and was being scored as a loss by an arbitrary label.

With the conversation-aware gate: **net +1, 0 lost**.

### A harness limitation worth recording

`eval/retrieval_replay.py` replays each stored query in ISOLATION, so history is
always empty - it structurally cannot test a conversation-aware mechanism, and
its first verdict on this change (-1) was an artifact of that. Any future gate
that reads conversation state needs a replay that reconstructs the conversation
from the primary turn, as this round did. The isolated replay remains correct
for anything that depends only on the query.

Note also that **hit@6 understates this change by construction**: it asks
whether the gold was retrieved, not whether a wrong-institution document took a
slot. The 4/17 -> 0/17 figure is the one that tracks the complaint.

## Round 8j — Chunk order: the general claim is FALSIFIED (2026-08-10)

Round 8g observed that the same chunks in a different order produced a
materially wrong answer, and concluded "nothing in this ledger measures chunk
ORDER". `RAG_CHUNK_ORDER` re-orders the final context without changing its
membership, so an A/B isolates ordering alone. `reversed` (worst-ranked first)
was run FIRST, deliberately, as a spoiler: if quality survives that, the
pipeline is order-insensitive and the Round 8g reading was wrong.

**Set 4, ordinary 6-chunk contexts:**

| | primary | follow-up | overall | hit@6 | keyphrase |
|---|---|---|---|---|---|
| rank | 4.60 | 4.30 | **4.45** | 10/10 | 0.55 |
| reversed | 4.50 | 4.10 | **4.30** | 10/10 | **0.58** |

Reversing the entire context - the most aggressive perturbation available -
costs 0.15 on a 5-point scale over 20 turns, and keyphrase coverage moves the
OTHER way. **The general claim that chunk order matters materially is not
supported.** On ordinary contexts this pipeline is largely order-insensitive.

Running the spoiler first is what produced this. A `rank` vs `grouped`
comparison would have shown some small difference and been read as
confirmation of the hypothesis that motivated it.

### A scope flaw in the test itself

Set 4 does not trigger multi-entity retrieval, so these were 6-chunk contexts.
The Round 8g observation was on a **14-chunk** context in which one document's
chunks were split top-and-bottom by four other departments' material - a far
more severe fragmentation than reversing six chunks. **Set 4 therefore tested a
weaker version of the claim than the one that was made**, and should have been
recognised as doing so before it was run, not after.

The direct test (set 5, 14-chunk multi-entity contexts, where `reversed` breaks
the document-contiguity fix added in 8g) follows. Three outcomes were named in
advance: a real effect narrows the claim to large fragmented contexts; no
effect means the CSEE accreditation error had another cause and the contiguity
fix was justified by coincidence; a noise-sized effect means dropping the
ordering claim entirely.

### Set 5, 14-chunk multi-entity contexts - falsified again, and harder

| | primary | follow-up | overall |
|---|---|---|---|
| rank | 3.70 | 4.00 | **3.85** |
| reversed | 4.20 | 3.90 | **4.05** |

Reversing a 14-chunk context - which shreds the document-contiguity the 8g fix
was built to preserve - scored **higher**, not lower. Direction opposite to the
hypothesis, magnitude inside noise (below).

**The Round 8g explanation was wrong.** The CSEE accreditation error was
attributed to chunk interleaving, code was changed on that basis, and the
effect does not reproduce when tested directly on the very context shape where
it was observed. The contiguity grouping in `_multi_entity_results` is still
defensible code - grouping a document's chunks is reasonable on its own terms -
but the reason recorded for it is not supported, and the real cause of that one
wrong answer is unexplained. With cloud generation unpinned, one wrong answer
may simply have been variance.

## Round 8k — The noise floor: +/-0.20 on a 10-question set (2026-08-10)

Measured accidentally and worth more than the experiment that produced it. Two
runs of the SAME configuration on set 5 (multi-entity ON, cloud generator,
local judge) scored **4.05 and 3.85**. Nothing differed but the run.

Cloud generation cannot be temperature-pinned (Sonnet rejects the parameter),
so this is irreducible for any cloud-generator comparison. **On a 10-question /
20-turn set, a delta below ~0.20 is uninterpretable.**

Consequences for numbers already in this ledger:

| finding | delta | verdict |
|---|---|---|
| Round 8h multi-entity gain | +0.50 (both judges) | clears the floor |
| Round 8j set-4 chunk order | 0.15 | **below the floor - not an effect** |
| Round 8j set-5 chunk order | 0.20, wrong direction | **at the floor - not an effect** |
| Round 8d spanning vs set 3 | 0.15 | **below the floor - reported as "slightly harder", should be "indistinguishable"** |

The Round 8d wording ("only slightly harder") should be read as "no measurable
difference". The conclusion it supported - that removing the single-document
confound did not expose a hidden weakness - is unchanged and arguably
strengthened.

**Rule: before believing a small cloud-generator delta, run one arm twice.**
The repeat costs the same as the comparison and is the only thing that
distinguishes a real effect from a reroll. No number below ~0.2 on a set this
size should be reported as an effect without it.

## Round 8l — Faculty -> department expansion (2026-08-10)

Complaint #5 ("accredited programmes offered by Schools/Departments in the
Faculty of Science and Health" answered only about HSC) could not be fixed by
multi-entity retrieval, because that triggers on named DEPARTMENTS and the
question names none - it names a FACULTY. `FACULTY_DEPARTMENTS` in
`src/entities.py` expands a named faculty to its member departments, taken
verbatim from the University's own faculty pages (user-supplied, 2026-08-10).

| question | OFF | ON |
|---|---|---|
| "...Faculty of Science and Health?" | 1/6 departments (4 docs) | **5/6 (11 docs)** |
| "...CSEE, MSAS, Psychology, HSC, SRES, Life Sciences?" | 1/6 (3 docs) | 5/6 (8 docs) |

The faculty form now behaves identically to the explicit six-department form,
which is the point: the user asked both and got the same bad answer for
different reasons.

Trigger rate on 226 existing eval turns: **0**. Unchanged from before the
expansion, so no committed number can move.

### A bug the expansion exposed in the mechanism's own budget

Arts, Humanities and Social Sciences has 11 member departments. At the fixed 2
slots each that is 22 reserved against a cap of 14 - the final slice would have
silently dropped the last four departments, which is precisely the failure this
mechanism exists to prevent, reintroduced by its own budget. The per-entity
allocation now scales: `max(1, min(2, cap // n))`, so 11 departments get 1 slot
each and 6 still get 2. Found by testing the second faculty, not the first.

### Rosters recorded even where the corpus is empty

Sociology, Criminology, ISER and UK Data Archive appear on the faculty page but
carry no `department` metadata (and in two cases no documents at all). They are
recorded in the roster anyway, with the existing empty-alias convention marking
them as detectable-but-not-filterable. A roster that silently omitted them
would be wrong as a statement of fact about the University, and the resulting
behaviour - "I have nothing for Sociology" - is the correct answer rather than
a gap to hide.

## Round 8m — The faithfulness guard discards good rewrites of elliptical follow-ups (2026-08-10)

`data/contextualizer_rejects.jsonl` holds 43 discarded rewrites. (An earlier
note in this session claimed the log was empty - that was a wrong FILENAME
check, `rewrite_rejects.jsonl`, and the conclusion drawn from it was void.)

Split by date and by which guard fired:

| period | rejections | cause |
|---|---|---|
| <= 2026-08-08 | 33 | `_has_extraneous_family`, since DISABLED (-8.8pts) |
| >= 2026-08-09 | **7** | `_is_faithful_rewrite`, live |

**All 7 live rejections are GOOD rewrites.** Examples:

| original | rewritten | overlap |
|---|---|---|
| "They still owe some fees - does that matter?" | "Do students need to have paid all their registration fees and debts before they can submit their thesis?" | 20% |
| "you only answered about CSEE, but I gave you 5 other schools, too!" | "What are the accredited programmes offered by MSAS, Psychology, HSC, SRES, and Life Sciences?" | 0% |

The guard requires >= 30% content-word overlap with the ORIGINAL. An elliptical
follow-up has almost no content words, so **the better the rewrite, the lower
the overlap** - the check is structurally biased against precisely the turns
the contextualizer exists to serve. Its own docstring exempts questions with
fewer than 3 topical words; these have 5-7, so they are judged and fail.

### Consequence: a discarded rewrite means retrieval uses the RAW question

Feedback complaint #7 is explained end to end. The rewrite naming all five
missing schools was generated correctly, discarded by the guard, and retrieval
then searched the corpus for the literal complaint text "you only answered
about CSEE, but I gave you 5 other schools, too!". The answer's insistence that
it had only CSEE material was therefore accurate about what it was given, and
caused by the guard rather than by retrieval or the generator.

This is a candidate explanation for the standing follow-up gap (3.70-3.95 vs
4.40-4.45 primary in every round). Retrieval and query rewriting were both
previously recorded as falsified causes - but those tests measured the
contextualizer's OUTPUT, not whether that output was subsequently thrown away.

**Not fixed in this round, deliberately.** The obvious repair - judge the
rewrite against the original PLUS history - does not work: the failure mode
this guard exists for (echoing a different earlier question) also overlaps
history, so both the good and bad cases look alike under that test. The
previous attempt to harden this area cost 8.8 points. Anything here needs a
measured A/B on follow-up turns specifically, with the reject log as the
instrument.

### Also: topic-switch probe

4 deliberate switches after an established topic. 3 handled correctly,
including both real user complaints (CSEE -> PhD duration, professional
doctorate -> PhD). One leaked: "what is the pass mark for an undergraduate
module?" was rewritten to "... (Professional Doctorates)" and retrieved PG
documents for a UG question. `_is_faithful_rewrite` cannot catch that shape -
appending a parenthetical preserves every original word, so overlap is 100%.

### Step 0 — base rate, and a correction to this round's framing

Free proxy: `_contextualize_query` returns the ORIGINAL question when it
discards a rewrite, so on a follow-up turn `retrieval_query == question` marks
a turn that retrieved on the raw, unresolved question.

| contextualizer | follow-ups retrieving on the raw question |
|---|---|
| local qwen2.5:7b (what every ledger baseline uses) | **27%** (28/103) |
| cloud claude-haiku-4-5 (what PRODUCTION uses) | **2%** (3/123) |

Cost on the affected turns, both main sets:

| | hit@6 | judge |
|---|---|---|
| raw question used (n=22) | 80% / 50% | 3.00 / 3.50 |
| rewrite used (n=58) | 90% / 68% | 3.70 / 3.82 |

**This corrects the framing earlier in Round 8m.** That entry offered the guard
as a candidate explanation for the standing follow-up gap. It may well explain
part of the gap *in the ledger*, whose baselines all run the local
contextualizer at 27%. It cannot explain much in PRODUCTION, which runs Haiku
at 2%. The 7 real reject-log entries are real, and rare.

The consequence for effort: **fixing this guard would improve the ledger's
numbers far more than the live system.** That is close to the definition of
work not worth doing, and it would have been easy to spend a full A/B cycle
discovering it afterwards rather than in a free base-rate check.

Two further cautions on the cost table above:
- **It is correlational.** Turns whose rewrite is discarded are plausibly the
  more elliptical and harder ones to begin with; the comparison does not
  isolate the guard.
- `retrieval_query == question` is an UPPER BOUND on guard rejections - it also
  counts turns where the contextualizer legitimately returned the question
  unchanged because it was already standalone.

**Recommendation: do not proceed to Steps 1-4 as a priority.** The instrument
exists (22 affected turns in the 160-turn baseline) if it is ever wanted, but
the live impact does not justify the risk of touching a guard whose predecessor
cost 8.8 points.

## Round 8n — Life Sciences, and the #11 defect diagnosed (2026-08-10)

### #11 is a real, stable defect - not variance

"In which cases is an independent chair required for the examination of PGR
degrees?" scored 1 of 6 circumstances on **three consecutive runs**. Its own
follow-up in the same conversation gets 6. Tested standalone to separate
phrasing from conversation context:

| query | circumstances named |
|---|---|
| #11 phrasing | **1/6** |
| #12 phrasing ("what are the circumstances that require...") | **5/6** |
| explicit ("what circumstances require the Faculty Dean to appoint...") | 4/6 |

Root cause is chunk-level and precise: the list lives in **exactly one chunk**,
`independent-chairs-policy.pdf` index 2. #11 retrieves chunks **1 and 6** of
that document and not 2; #12 gets chunk 2 at rank 2. `hit@6` is True in both
cases, so every document-level metric scores them identically.

The answer-bearing chunk is the immediate neighbour of a chunk already at rank
1. That makes adjacent-chunk expansion the obvious candidate fix, and it is
NOT attempted here - it would need its own flag and measurement, and this
project's history with unmeasured retrieval mechanisms is the reason.

### Life Sciences: partially closed, and honestly partial

The gap was real (0 Life Sciences chunks in the six-department fan-out), not a
metric artifact - checked before acting. Two causes, two fixes:

1. **The school does have reachable metadata.** "Biomedical Science" is a
   programme area of the School of Life Sciences and IS carried as a department
   value. `life sciences -> ["Biomedical Science"]` reaches part of the school
   rather than nothing.
2. **The metadata-less fallback was diluted.** It prepended the alias to the
   WHOLE multi-entity question, so five other departments' terms dominated the
   embedding. Stripping the other named entities focuses it. This now applies
   to Sociology, Criminology and the Philosophical/Historical school rather
   than to Life Sciences.

| | before | after |
|---|---|---|
| explicit six-department question | 5/6 | **6/6** |
| faculty form | 5/6 | **6/6** |
| set 5 department coverage (regression) | 100% | 100% (8/8 unchanged) |

**What "6/6" does and does not mean.** The Life Sciences slot is filled by
`biomedical-science-year-one/two.pdf` - genuinely part of that school, but a
programme area, not the school's full rules. The school's broader UG rules live
as a "BS - Life Sciences" SECTION inside multi-department variations documents
that carry no department at all, and those are still not reached on a
multi-entity question. A user asking specifically about Life Sciences does get
them (tested: 6/6 chunks), because a focused single-entity query is not
diluted. So the coverage metric now reads 6/6 while the underlying situation is
"partially covered" - recorded so the number is not mistaken for completeness.

## Round 8p — Adjacent-chunk expansion (2026-08-10)

Traced from a real user question. "In which cases is an independent chair
required for the examination of PGR degrees?" answered *"the materials
available don't specify the exact triggering circumstances"* - **false**. The
policy lists seven in section 3.1. That list lives in ONE chunk
(`independent-chairs-policy.pdf` index 2) and the query retrieved chunks 1 and
6 of the same document. `hit@6` was True, so nothing in this ledger could see
it. A confident "not specified" about a clearly specified policy is the worst
failure mode available to this tool.

### Measured before building

Across 160 turns, classifying where the answer-bearing chunks sit:

| | turns | |
|---|---|---|
| answer chunk already retrieved | 127 | 79% |
| **adjacent to a retrieved chunk** | **8** | **5%** |
| far from anything retrieved | 18 | 11% |
| no identifiable answer chunk | 7 | 4% |

**Of the 26 turns missing the answer chunk, 31% have one immediately beside a
retrieved chunk.** Ceiling on benefit is therefore ~8 turns in 160; the larger
11% "far" slice needs ranking work this cannot touch.

### Result

| | |
|---|---|
| the real case | 0/7 circumstances + false denial -> **5/7, no denial** |
| A/B, 20 questions (sets 4+5) | 4.17 -> 4.22, **delta +0.05** |
| noise floor | +/-0.20 |

**No detectable harm, no detectable broad benefit** - the expected shape when a
benefit concentrates in 5% of turns and the sample is 20 questions.

### The blast radius mattered more than the delta

Expanding around the top 3 hits touched **97% of turns**, adding 2.62 chunks
each: a system-wide context change for a benefit concentrated in 5%. That ratio
is what `_has_extraneous_family` had when it cost 8.8 points. Narrowed to the
rank-1 chunk only: still fixes the real case, touches 81%, adds 1.26 chunks -
same fix, less than half the context growth. Shipped narrow.

**Two honest limits.** The A/B measured the WIDER setting; the shipped one is a
strict subset, so no-harm carries over a fortiori but is an inference rather
than a measurement of the exact configuration. And `hit@6` is structurally
blind here - neighbours come from documents already retrieved, so the document
set was unchanged on all 160 turns and the free replay could only ever report
"no change". End-to-end was the only instrument that could detect dilution.

## Round 8q — Structure-aware chunking: prototyped, NOT adopted (2026-08-10)

Motivated by the independent-chairs failure: the seven qualifying circumstances
are split across chunks, so no chunk holds the whole list. `chunk_text()` is a
blind sliding word window (175 words, 30-word overlap) with no awareness of
sentences, clauses or lists.

### Step 0 - and a correction to its own first measurement

A first pass reported **97% of chunk boundaries fall mid-sentence**. That is
true and nearly meaningless: the 30-word overlap means the next chunk begins 30
words BEFORE the previous ended, so a cut sentence reappears intact. Reporting
97% would have justified a large risky project on a self-healing symptom.

The measurement that matters is whether a LIST survives intact somewhere:

| | |
|---|---|
| lists of 3+ items in current documents | 360 |
| split so that NO chunk holds the whole list | **144 (40%)** |
| of those, small enough to fit in 175 words | **91 (63%)** |
| median split-list length | 129 words |

So most split lists are not too big - they are **cut in the wrong place**.

### Step 1 - the obvious prototype made it WORSE

v1 slid the cut BACKWARDS to land before a new structural unit. Result:
**30% -> 50% of fittable lists split**. Shortening every chunk to improve its
boundary traded away the capacity the lists needed - a straight own-goal, and
exactly what a free offline prototype is for.

v2 keeps the full window and EXTENDS forward past the cut when a list is in
progress, up to 60% over length:

| | fittable lists split |
|---|---|
| current sliding window | 91/307 (30%) |
| prototype v2 | **75/307 (24%)** |

### NOT adopted - the gain does not justify the cost

v2 recovers **16 lists across the whole corpus**. Against that:

- **It invalidates the ledger.** Every baseline in Rounds 1-8 was measured on
  the current chunking. Re-chunking makes them incomparable - and that record
  is the most valuable thing this project owns.
- **Blast radius is 100% of turns**, for a benefit measured in a sub-metric
  affecting a minority of questions. Adjacent-chunk expansion was narrowed
  earlier this session for a far better ratio than this.
- **It requires re-embedding ~21,000 chunks**, through the same `clean_text`
  that once silently deleted policy clauses corpus-wide.
- **Adjacent-chunk expansion already recovers part of this** at zero re-ingest
  cost, including the case that started it.

Recorded rather than pursued. If chunking is ever revisited - most likely
alongside a deliberate re-baselining - the v2 extend-forward approach is the
one that worked, and the backwards-cut approach is the one to avoid.

### Step 2 - bigger chunks: measured, and NOT adopted

Built a parallel index at 350 words (`data/chroma_350`, ~10k chunks vs 21k)
and replayed the same 160 stored queries. Free - local embedding, retrieval
only, no API.

| | hit@6 | evidence-sufficient@6 |
|---|---|---|
| 175-word (production) | **78.1%** | 92.5% |
| 350-word | 76.2% (-1.9) | 93.8% (+1.3) |

**Fixing list-splitting from 30% to 8% produced no retrieval gain.** Bigger
chunks dilute the embedding (document precision -3 turns) while carrying more
content (evidence coverage +2 turns). It nets to nothing, and both moves are
within a handful of turns out of 160.

So the whole chunking thread ends without a change:

| approach | lists split | retrieval effect |
|---|---|---|
| current sliding window | 30% | baseline |
| structure-aware v1 (cut back) | 50% - WORSE | not measured |
| structure-aware v2 (extend forward) | 24% | not measured |
| section-based (keep<=350) | 20% | not measured |
| uniform 350 | **8%** | **-1.9 hit@6, +1.3 evidence** |

Against a null retrieval result, re-chunking would still cost: every Round 1-8
baseline invalidated, ~2x context per turn (~1,985 words vs ~1,027), and a
re-embed through the same `clean_text` that once deleted policy clauses
corpus-wide. Adjacent-chunk expansion already recovers the motivating case at
zero re-ingest cost.

### Two validity traps found while setting this up

Both would have produced clean-looking numbers measuring nothing:

1. **`reembed.py` rewrites `data/manifest.json`**, so the experimental run
   overwrote production's `chunk_count` values. Backed up beforehand and
   restored. (`chunk_count` turns out to be cosmetic - no retrieval or scoring
   code reads it - but the write is silent.)
2. **The ColBERT embedding cache is keyed by `(source_url, chunk_index)` with
   NO text validation.** Against a differently-chunked index, chunk 2 of a
   document would have been reranked using the embedding of the OLD chunk 2 -
   entirely different text. The comparison would have looked fine and measured
   nothing. `RAG_COLBERT_CACHE=0` now disables it; it was set on BOTH arms so
   neither borrowed the other's vectors.

`RAG_CHROMA_DIR`, `RAG_CHUNK_WORDS` and `RAG_COLBERT_CACHE` are now
env-overridable (defaults unchanged) so a parallel index can be built and
compared without touching production.

## D3 (clarify-on-underspecified) — DECLINED, 2026-08-10

Stays OFF by the user's decision, after the trade was set out: ~0.45 trigger
precision means roughly one interrupted answerable question for every two
correct stops, and no metric in this ledger can adjudicate it - a clarifying
question is scored as a MISS by construction, because hit@6 asks whether the
right document was retrieved and asking a question retrieves nothing.

Recorded as a closed decision rather than an open item, so it is not
re-proposed. The mechanism and its rationale remain in `src/rag.py` should the
product judgement ever change.

## Round 8r — Set 6: the partner test set finds an over-correction (2026-08-10)

Closes to-do item 6, open since 2026-08-08: **33% of current documents are
partner editions yet none was the gold answer in any eval question**, so the
partner mechanisms could never be validated in the direction that matters -
questions where a partner document IS the right answer.

8 questions in three deliberately different groups, 34 keyphrases verified
against source text, 0 failing:

| group | exclusion OFF | exclusion ON |
|---|---|---|
| partner NAMED ("at Colchester Institute...") | 4/4 | **4/4** |
| partner UNNAMED ("BA (Hons) 3D Design & Craft...") | 2/2 | **0/2** |
| home control (Essex questions) | 2/2 | **2/2** |

**`PARTNER_EXCLUDE_WHEN_UNNAMED` over-corrects on questions that name a partner
PROGRAMME but not the institution.** "Which rules of assessment apply to the BA
(Hons) 3D Design & Craft course?" and "under the BSc Economics link agreement,
how many credits are needed to pass Year One?" both lose their partner gold.

This was invisible before: the 9 incidental partner-gold questions in the older
sets gave net +1 with 0 losses, which read as safe. It was not safe - it was
unmeasured in this direction, exactly as item 6 predicted.

**Not changed, because it is a product call rather than a defect.** The gate
does what it was asked to do: the user's complaint, raised twice, was Essex
questions returning partner documents ("Answers need to focus on Essex
programmes, not partners"), and that is fixed - 4/17 real complaints to 0/17,
with the home controls here unaffected. The cost is the inverse case, which
that user may never hit but a partner-college administrator would hit
immediately.

**DECIDED 2026-08-11: leave as is** (user's call, trade stated in full). The
direction that matters to the intended audience - Essex staff asking about
Essex programmes - is fixed, and the two softening options both add ongoing
maintenance: a curated list of partner PROGRAMME names would need extending
with every new partnership agreement, and "demote unless no Essex alternative
covers the topic" requires knowing what a document covers before retrieving it.

Recorded as a closed decision so it is not re-proposed. The cost is real and
stays on the record: a question naming a partner course but not its institution
loses the document that answers it.

Recorded with the measurement so the choice is made on evidence rather than
rediscovered later.

## Round 9 — Corpus re-ingest, and a deliberate ledger re-baseline (2026-08-10)

The weekly watcher (`check_new_documents.py`, first run) found 5 new documents
and ~20 changed since the last ingest - meaning the system was answering from
superseded text on documents including `roa-ug-3yr-final-year-rules.pdf`.
Ingested deliberately, with the checks run in order.

| step | result |
|---|---|
| `retrieval_replay` before | 123/160 (76.9%) |
| `run_ingest.py` | 10 indexed, 1151 unchanged, 23 rejected, **0 errors** |
| `audit_family_aliases.py` | **0** new rename-split groups |
| `eval/stale_index_audit.py` | 246 docs, **0** missing content |
| `retrieval_replay` after | 122/160 (76.2%) — **net -1** (0 gained, 1 lost) |

### The one regression, and why it was accepted

Lost: "What does the University of Essex Rules of Assessment cover regarding
the final award?" - a GENERAL framework question whose gold is
`ug-principles-and-framework.pdf`. The new `/ug/current/variations/` files
(year-1..4, ~107 chunks each) now occupy 5 of the 8 slots and push it out. So
a general question is now answered from department-specific variations.

Accepted, because the alternative is worse: serving superseded rules from ~20
documents is a failure at the tool's actual purpose, while this is one turn out
of 160 on a metric already known to over- and under-state in different
directions. Recorded rather than buried, because if more general-framework
questions degrade later, this is the cause.

**Watch item:** the new variations files are large and generically worded, so
they are broadly "attractive" to UG queries. Only 1 of 160 turns moved, so the
effect is not widespread today - but a future set with more general UG
questions should be checked against them specifically.

### THE LEDGER IS NOW RE-BASELINED

Every number in Rounds 1-8 was measured on the pre-ingest corpus. They remain
valid as a record of what was true then, and remain comparable to each other,
but **they are no longer comparable to anything measured from here on**. The
new baseline is `eval/retrieval_replay_post_ingest.json` at 122/160 (76.2%).

This was a deliberate decision taken with the cost stated in advance, not a
side effect discovered afterwards - which is the whole reason the ingest was
sequenced this way rather than just run.

## Round 10 — Answer detail level (concise / detailed) (2026-08-11)

The redesign's Settings screen had a Concise/Detailed control that saved to
localStorage and did nothing - the backend accepted only `content`. Built for
real, per-request rather than as a global flag: the preference travels with the
message, and DEFAULT is unchanged behaviour, so a user who never touches the
control gets exactly what production gives today.

### The risk, and why the rule is worded the way it is

This project's two prior base-prompt rules went in opposite directions -
`INLINE_CITATIONS` cost 11 points of groundedness, `MULTI_ENTITY_COVERAGE`
helped. A naive "be brief" instruction plausibly damages the exact failure the
user complained about twice: dropping named entities and list items. So the
concise rule protects those explicitly:

> "Do NOT drop any of the following to save space: an entity the question
> named, an item of a list the document gives, or the source citation -
> brevity must never turn a complete answer into a partial one."

### Measured on the two questions most at risk

| question | default | concise |
|---|---|---|
| six departments named | 6/6 entities, 4641 chars | **6/6 entities**, 3393 chars |
| seven-item circumstances list | 6/7 items, 783 chars | **6/7 items**, 797 chars |
| source cited | yes | yes |

**Concise loses no entities and no list items** - it removes 27% of the length
on the enumeration question while keeping every department, and is length-
neutral on the short list question, which is the correct behaviour: there was
nothing padding it to remove.

Not a general quality A/B - two questions chosen because they are where a
brevity instruction would do damage if it were going to. A broader regression
is not possible here anyway: default is byte-identical to current production,
so nobody who leaves the setting alone is affected.

## Round 11 — Latency: warm 9.8s, cold 28.2s (2026-08-11)

Prompted by "it still takes considerable time to respond". `RAG_TIMING=1` has
been writing `data/latency.jsonl` since 2026-08-09; 473 records over 118 real
requests.

**The system has two very different behaviours depending on warmth.**

| | n | median total | retrieve | generate | contextualize |
|---|---|---|---|---|---|
| warm (<10 min since last request) | 101 | **9.8s** | 0.5-3.4s | 6.8s | ~0.5s |
| cold (>10 min idle) | 17 | **28.2s** | **22.6s** | 6.8s | ~0.5s |

p90 total 28.2s; worst observed 84.3s.

Warm, the cloud generator IS the latency (6.8s of 9.8s) and retrieval is nearly
free. Cold, retrieval costs ~21 extra seconds.

### Reproduced outside the server, then decomposed

Fresh process: `retrieve()` #1 20.87s, #2 3.19s, #3 0.50s - the 20.4s gap
matches the log-derived ~21s independently. Pre-warming each loader in turn:

| component | cold cost |
|---|---|
| torch / ColBERT first-encode warmup | **~8.0s** |
| ColBERT model load (pylate) | ~4.1s |
| BM25 index build (rank_bm25, 21.7k chunks) | ~2.5s |
| module import | ~2.5s |
| Chroma first dense query | ~0.3s |

~17s of the ~21s; the remainder is run-to-run variance. Everything here is
lazily initialised on first use, and launchd `KeepAlive` restarts reset all of
it.

### Why the median understates the problem

Only 14% of LOGGED requests were cold - but the log is dominated by my own
heavy use. **A tester asking two questions a day is cold essentially every
time**, so the experience measured here as "9.8s median" is ~28s for them. Any
future latency claim should state which population it describes.

**Not fixed.** The obvious move - one dummy retrieval in a background thread at
startup - is unbuilt and unmeasured, and is recorded here as a hypothesis, not
a result. Note it only moves the cold case; warm latency is the generator.

**Probe artifact, for the record:** an early probe called Chroma with
`query_texts` instead of `query_embeddings`, which triggered Chroma's DEFAULT
onnx embedder (all-MiniLM-L6-v2, 384-dim vs the collection's 768) and
downloaded 79MB into `~/.cache/chroma`. Not a production path - production
embeds via `llm.embed` and passes `query_embeddings` - but it is why that cache
directory exists.

## Round 12 — Startup warmup: first query 17.0s -> 3.4s (2026-08-11)

Round 11 measured a ~21s cold penalty and recorded pre-warming as a hypothesis.
Built and measured.

`src/app.py` runs one throwaway `retrieve()` in a daemon thread on startup
(`RAG_STARTUP_WARMUP`, default on). It pays the lazy initialisation before a
user asks.

### A/B: 6 fresh processes, arms alternated

One process per trial - the effect IS process-level lazy init, so reusing a
process would measure nothing. Arms alternated (A,B,A,B,A,B) so the OS page
cache, warmed by trial 1 regardless of arm, cannot systematically favour
whichever ran first.

| arm | first query | all trials |
|---|---|---|
| no warmup | **16.97s** | 16.88 / 16.97 / 17.28 |
| warmup | **3.40s** | 3.38 / 3.40 / 3.41 |

**-13.6s on the first query**, spread 0.4s across trials - far tighter than
this project's usual noise, because no generator is involved.

The warmup itself costs ~17s, absorbed in the background. The first real query
still costs 3.4s rather than the ~1.5s of a fully warm one: the warmup query is
not the user's, so per-query embedding and ColBERT encoding of new candidates
remain.

### Verified on the real server

Restarted under launchd: `Application startup complete` and pages served in
**0.4ms while the warmup was still running** - it does not block boot or the
health check. `[warmup] retrieval stack ready in 17.7s` then appears; RSS goes
400MB -> 4.87GB, confirming the model is resident rather than the log line
merely being printed.

### It cannot change answers - checked, not assumed

Same `retrieve()` every request uses, keeping nothing. Compared top-6 document
sets across 4 queries with warmup ON vs OFF: **identical**.

### A concurrency bug this introduced, and the fix

`colbert_index.get_model()` and `_load()` were check-then-set on module globals.
That was safe when only request threads reached them; the warmup thread now
runs *concurrently with real requests*, so two threads could pass `is None`
together and each load a full ColBERT model - GBs of duplicate allocation on a
16GB machine. Added `_load_lock` with double-checked locking. Tested with 4
racing threads: no deadlock, no duplicate load, one model instance.

### Limits

Only the COLD case moves. Warm latency is unchanged and is dominated by the
cloud generator (6.8s of 9.8s) - pre-warming cannot touch it. A user who asks
within ~18s of a restart still races the warmup rather than benefiting from it.

## Round 13 — External review: what was verified, and one finding of my own (2026-08-11)

Four models reviewed rounds 7-12. Every checkable claim was verified against
code and data rather than accepted.

### CONFIRMED - real defects

**1. `hit@6` did not cap at 6.** `run_eval.py` and `run_multiturn_eval.py`
scored `hit_at_6: rank is not None`, scanning the WHOLE `ranked_top_urls` list.
Since 2026-08-10 that list is no longer 6 long (adjacent expansion appends up to
2, multi-entity up to 14). `retrieval_replay.py:106` had it right (`rank <= 6`)
- the harnesses disagreed with each other.

Actual damage, re-derived from stored `rank` fields without re-running:
**7 false hits, all in `results_o5_reversed_set5_multientity.json`** (ranks 7-10
scored True); that file's hit@6 goes **15 -> 8 of 20**. No other result file is
affected. Round 8j/8o's chunk-order conclusion rests on judge means, not hit@6,
so it stands.

Also confirmed: the adjacent-chunk A/B was **unfair by construction** - the OFF
arm's lists are exactly 6 URLs, the ON arm's are 7-9, so the ON arm had extra
slots to hit in. No hit landed beyond rank 6, so the reported delta is
unchanged, but the comparison was not like-for-like and I could not have seen it.

**2. `_names_partner_institution` matched unbounded substrings.** `"eput"` is
inside *deputy* and *reputation*; `"sak"` inside *for the sake of*. Verified
live: "Who is the Deputy Dean?", "the deputy chair", "for the sake of clarity"
and "University reputation policy" ALL triggered partner mode, each silently
switching `PARTNER_EXCLUDE_WHEN_UNNAMED` off - and since the gate reads the
whole conversation, one "deputy" disabled exclusion for the rest of it. Fixed
with a word-boundary regex compiled at import; 4/4 false positives gone, 8/8
true positives kept.

### NOT CONFIRMED

`FETCH_POOL_MULTIPLIER=4` "costs quality" was challenged as unsupported by
`fetch_pool_sweep_result.json` (124/160 at both 8 and 4). Correct about that
file - but the claim came from **stage 2**, an 80-question end-to-end run
(`results_pool4.json`, commit fc1bf45): primary hit@6 80.0% -> 77.5%, mean
4.05 -> 3.91. The two-stage protocol exists precisely because stage 1 was blind
to it. Caveat now recorded: those deltas are small, and the hit@6 difference is
one turn.

### MY OWN ERROR, which shaped three of the four reviews

I reported "0.76s to ColBERT-encode one short query" in the latency prompt.
**Wrong: warm steady-state is 19ms.** The 0.76s was the FIRST encode - the
torch/MPS warmup already counted in the cold budget. I double-counted it, and
three reviewers built device/dtype/`pylate-rs`/quantisation recommendations on
a number that does not exist in the warm path.

### THE ACTUAL WARM BOTTLENECK - not what anyone (including me) said

Attributing a warm `retrieve()` by wrapping the hot functions:

| | share of warm retrieve |
|---|---|
| `vector_query` (embed + Chroma) | **72-84%** |
| `_adjacent_chunks` | 5-12% |
| `rerank` (ColBERT encode + cache + MaxSim) | **6-13%** |
| BM25 | 2-3% |

Reranking - which every reviewer targeted - is 0.20-0.25s total. Inside
`vector_query`, embedding is ~34ms, so the cost is the Chroma call.

**And it is memory-pressure-dependent.** Chroma's filtered query
(`where={"is_current": True}`, n=48), same process, same queries:

| | |
|---|---|
| before `colbert_index._load()` | **81ms** |
| after `_load()` | **2543ms** |

`_load()` takes RSS 1.39GB -> 5.11GB (the Voyager index, not the model -
`get_model()` alone changes nothing). **A 31x slowdown of dense retrieval,
caused by the residency of an index whose only job is a rerank cache that saves
0.2s.** On 16GB shared with Ollama this is compressed-memory thrashing.

Fully hot, the cache is still net-positive (1.09s vs 1.30s with
`RAG_COLBERT_CACHE=0`) but costs 2.6GB (4.97GB vs 2.36GB peak RSS). So warm
latency is **bimodal in memory pressure**, which explains the 0.5-3.4s spread
in `latency.jsonl` that I had recorded as ordinary variance.

**Not yet decided.** The trade is 0.2s of rerank speed against 2.6GB of RSS
that can cost 2.5s per query when pressure is high. Needs measuring under
realistic pressure (browser + Ollama + server) before changing a default.

## Round 14 — Streaming, and why the ColBERT cache was NOT turned off (2026-08-11)

Built streaming so the answer appears as it is written instead of after a
blocking wait. Threaded an `on_token` callback through the EXISTING
`rag.answer()` rather than writing a second streaming implementation - there is
one answer path and nothing to drift. The ambiguity disclosure is appended
after generation, so it streams as a final chunk and the stored text still
equals what the user saw.

`POST /api/conversations/{id}/messages/stream` (SSE) sits beside the blocking
endpoint, which stays: the eval harness uses it, and the UI falls back to it if
SSE fails, so a transport problem cannot cost a user their answer. Storage is
written from `answer()`'s RETURN value, never from concatenated tokens - a
provider that ignores `on_token` still returns complete text.

### Measured on the live server

| | dead time before text appears | total |
|---|---|---|
| blocking | 14.86s | 14.86s |
| streaming | **7.17s** | 14.86s |

Decomposed: 5.34s retrieval + ~1.8s until the generator's first token.
**Streaming cannot cover retrieval** - there is nothing to show until the
context exists - so retrieval is now the whole of the remaining wait.

### The cache experiment: a large win that changes answers, so NOT shipped

Round 13 found Chroma's filtered query degrades 81ms -> 2543ms once
`colbert_index._load()` puts the Voyager index resident. Tested
`RAG_COLBERT_CACHE=0` on a live server:

| | retrieve | dead time | server RSS | warmup |
|---|---|---|---|---|
| cache ON (production) | 5.34s | 7.17s | ~5.0GB | 16.8s |
| cache OFF | **1.40s** | **3.3s** | **2.11GB** | 11.4s |

Retrieval 3.8x faster and 2.9GB lighter, because the cache saves ~0.2s of
reranking and costs ~3.9s of memory pressure.

**Not adopted.** Retrieval results are NOT identical: on 5 probe queries, 2
returned different document sets. The cached embeddings were computed offline
and are not equivalent to freshly-encoded ones, so this is a retrieval-changing
mechanism, not a config tweak. It goes through an eval like anything else.
Recorded as the most promising latency lever measured so far, pending that.

The near-miss is worth stating plainly: the latency case was strong enough that
shipping it on those numbers alone would have been easy, and it would have
silently changed what documents answers are based on.

## Round 15 — Paired comparison tooling, and what it says about Round 8p (2026-08-11)

All four external reviews put this first: stop comparing two arm means when
both arms answered the same questions. `eval/compare.py` reports paired
per-question differences, a bootstrap CI over QUESTIONS (not turns - the two
turns of one question share a retrieval), win/tie/loss, and the turns-changed
count. Zero compute; it reads stored result files.

### Re-analysis of the adjacent-chunk decision (Round 8p)

| | |
|---|---|
| originally reported | +0.05, "below the noise floor, no harm detected" |
| paired mean | **+0.050** |
| 95% bootstrap CI | **[-0.250, +0.350]** |
| win / tie / loss | **3 / 14 / 3** |
| turns changed | **6 of 20** |
| hit@6 (paired) | gained 0, lost 0, **identical on all 18** |

Same mean, different claim. **Fourteen turns were identical and the six that
moved split three up, three down.** "+0.05" reads as a small improvement;
"3 better, 3 worse, 14 untouched, interval straddles zero" is what the data
supports. The decision to ship stands (it was justified on a named failure
case, not on this mean), but the ledger's language was stronger than its
evidence.

**"Confirmed no harm" was the wrong words** and is retracted. A 20-turn A/B can
only fail to DETECT harm above roughly its resolution. The tool now prints that
distinction rather than leaving it to the writer.

### Two bugs found by validating the tool rather than trusting it

1. First run paired only 14 of 20 turns. The key was `source_url`, which is
   **not unique** - three documents carry two questions each in the spanning
   set - so turns collided and overwrote each other. Now keyed on the turn's
   own question text. This is the exact failure the tool exists to prevent, and
   it was in the tool.
2. `hit` is recomputed from `rank <= 6`, never read from the stored
   `hit_at_6` flag, because that flag was written uncapped until 2026-08-11 and
   is wrong in older files.

### On the external figure

The review that prompted this reported +0.000 over 26 turns for this
comparison. That does not reproduce - the files hold 20 scored turns - but its
conclusion (most turns unchanged, effect unresolvable) is exactly right, and
was right for reasons the mean alone could not show.

## Round 16 — Multi-entity partner leak: real, latent, and hard to exhibit (2026-08-11)

External review found that `_multi_entity_results` issues FRESH retrievals after
`_exclude_partner_institutions` has run, so partner chunks re-enter through the
per-entity slots. Confirmed by reading the call order (rag.py ~1550-1565):
exclusion applies to `candidates`, multi-entity runs after the rerank, and
`_demote_partner_institutions` only reorders what is already there.

### Three instruments could not see it

| instrument | why it is blind |
|---|---|
| `retrieval_replay` (160 turns) | **0 turns name >=2 departments** |
| set 5 (multi-entity, 10 q) | those questions surface no partner documents at all |
| set 6 (partner, 8 q) | they NAME a partner, so the gate correctly disables exclusion |
| 5 constructed adversarial questions | 0 partner chunks in either arm |

Running any of them would have reported "no change" and been read as "the fix
does nothing" - for a defect that is genuinely there.

### Exhibiting it required finding the exposed path

Per-entity retrieval filters on `department` metadata. Partner documents do not
carry those values, so that path CANNOT return them. The exposure is the ELSE
branch: aliases with no department metadata fall back to `where={"is_current":
True}` alone. Exactly **three** aliases qualify - sociology, criminology,
philosophical/historical/interdisciplinary studies.

| | partner chunks served |
|---|---|
| fix OFF | **3** (on 3 of 4 questions) |
| fix ON | **0** |

Collateral: **0 of 23** other questions changed (sets 5 and 6, plus 5
constructed multi-department questions).

Shipped as `MULTI_ENTITY_PARTNER_RECHECK`, default on. `eval/partner_multientity_probe.py`
is the permanent regression test, because nothing else in the eval suite can
detect a recurrence.

### The generalisable point

"Ran the eval, no change" is not evidence a fix is unnecessary when the eval
contains no case the fix applies to. The blindness has to be checked
explicitly - here by counting how many turns even trigger the mechanism (0 of
160) BEFORE running anything.

### An unattributed -1 on the replay, recorded not buried

Current code scores **121/160**, twice, byte-identical across runs (so the
replay is deterministic and this is not noise). The Round 9 post-ingest
baseline was 122/160. The lost turns are two "Essex Abroad / Term Abroad"
questions whose gold is `roa-ug-3yr-year-1-variations.pdf`.

**Not caused by the partner-gate fix**: neither query matches any partner token
under the old substring rule OR the new word-boundary rule, so the gate returns
False in both versions. The cause is somewhere in the ~6 commits between the
Round 9 baseline and now. Not bisected - one turn in 160, and the reviewers'
unanimous advice was to stop spending on exactly this. 121/160 is the reference
from here.
