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
