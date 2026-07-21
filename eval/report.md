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
