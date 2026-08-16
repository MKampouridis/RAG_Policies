# Round 8 — what to do next

From three external reviews (Claude, ChatGPT, Gemini), filtered to what is worth
doing. Items all three converged on are marked **[3/3]**. Things I've decided
against are at the bottom with reasons, so they aren't re-proposed.

**The shape of the advice:** all three said stop optimising retrieval, fix the
benchmark's provenance rather than 8 items, add authentication, get real users,
and treat the judge/reference finding as an evaluation-validity paper needing a
second annotator. The ordering below follows that.

---

## P0 — Before anyone else uses it

| # | Item | Why | Est |
|---|---|---|---|
| 1 | ~~**Authentication**~~ **DONE** [3/3] | `_owner()` trusts a header the user types. Anyone can read anyone's history by sending a different name. For "what happens if I fail my resit", that is not a v2 item. Shared password + the existing name separation is enough for a trial | 1–2h |
| 2 | ~~**Research routes truncate data with no auth**~~ **DONE** — both remaining `write_text` calls are inside `_save_versioned` (timestamped copy + pointer), verified | `research_routes.py` does `write_text()` — full replace, no auth, no backup. A POST of `[]` erases the 30 blind human judgements the paper depends on. Write timestamped + symlink latest, or copy-before-write | 30m |
| 3 | ~~**Streaming has no completion check**~~ **DONE** — `saw_stop` in `llm.py` | `llm.py` returns `"".join(parts)` when the stream ends, so a dropped connection stores a **silently truncated answer**, indistinguishable from a complete one. Observe `message_stop`; flag or raise if absent. On a policy tool, an answer ending mid-sentence at "students may appeal if" is worse than an error | 45m |
| 4 | ~~**Push `owner` into the SQL**~~ **DONE** | `delete_conversation`, `restore_conversation` and `list_deleted_conversations` have no owner in their WHERE clauses — the invariant lives in the caller. `list_deleted_conversations` has no owner parameter at all, so a recovery UI would show everyone's deleted conversations | 30m |

## P1 — Benchmark provenance (do as one job, not 8 edits)

| # | Item | Why | Est |
|---|---|---|---|
| 5 | **Re-derive the eval set** — **PARTIALLY DONE, needs you** [3/3] | The test set is a **stale cache of the corpus** — same bug class as the ColBERT index. Verified: **9 of 148 gold URLs point at NON-current documents**, so retrieval is scored a miss for returning the current edition. Patching 8 known-bad items leaves 143 with unverified provenance | 1–2d |
| 6 | ~~**Stamp `corpus_version` into `questions.json`**~~ **DONE** — verified stamped and matching the live corpus | And check it in the post-ingest sequence. Converts "someone remembers to look" into "fails closed" | 1h |
| 7 | ~~**Record correction provenance**~~ **DONE** — `eval/provenance_corrections.json`, 10 corrections with old/new URL, reason, who, and keyphrases verified | For each fixed reference: old text, new text, source URL, passage, why it was wrong, who, when. This becomes part of the benchmark's published provenance | with #5 |
| 8 | ~~**Fix the 8 defective references**~~ **subsumed into #5** | Now a subset of #5 rather than its own task. Page already built at `/reference-fix` | with #5 |

## P2 — Stale-artifact hygiene (the general class)

| # | Item | Why | Est |
|---|---|---|---|
| 9 | ~~**Provenance stamp on every derived artifact**~~ **DONE** [3/3] — `src/provenance.py` | The rule the ColBERT incident implies. Counts and mtimes are proxies; hashes are the thing. `lexical.py` and `doc_index.py` already invalidate against `read_corpus_version()` — ColBERT never did | 2–3h |
| 10 | ~~**The drift check compares counts only**~~ **DONE** — `colbert_index_drift.py` now checks the provenance stamp and fails closed on mismatch; the count is kept only to localise. **Caveat: it currently reaches nothing** — the ColBERT index is not built (cache defaults OFF), so the check exits 0 without testing anything. It will bite on the next rebuild | A document edited in place with the same chunk count passes it. The last re-ingest was *5 new and ~20 changed* — the changed ones are exactly what a count check cannot see | 30m |
| 11 | ~~Audit the other unstamped artifacts~~ **AUDITED, not all fixed** — `eval/artifact_provenance_audit.py` found `data/colbert_index/`, `data/splade_matrix.npz` and `data/doc_identity/` (1,188 files) unprotected. All three are OFF-by-default paths, so they are recorded rather than stamped | SPLADE, pseudo-query and ensemble indexes (off but live), `data/doc_identity/*.json` (never invalidated when its document changes), the process-lifetime caches | 1h |

## P3 — Verification that covers what it claims to

| # | Item | Why | Est |
|---|---|---|---|
| 12 | ~~**Verification ladder, cheapest first**~~ [3/3] **DONE** | `verify.py`: pyflakes → import → JS parses → no top-level dead zone → live POST. `--static` skips the server. Each step tested against the bug it claims to catch | done |
| 13 | ~~**A golden-request test**~~ **DONE** | Step 5 of `verify.py`: creates a conversation, POSTs a real question, asserts 200 + non-empty answer + non-empty sources, deletes it in a `finally`. Live: 1,174 chars, 4 sources |  done |
| 14 | **A browser smoke test** — **BLOCKED, partially covered** | No JS runtime available (node/deno/bun/esbuild all absent) and **Chrome headless times out in both `--headless=new` and `--headless=old`**, so the real thing cannot be built here. *Partially* covered instead by `verify.py` step 4, a static top-level dead-zone check — **verified to catch the actual blank-page bug** by running it against the pre-fix file, and verified not to fire on the working one. Uncovered: runtime errors that are not top-level dead zones | blocked |
| 15 | ~~Write the coverage rule into `CLAUDE.md`~~ **DONE** | Own section, with both failures as evidence, the `verify.py` command, and an explicit statement of what it does **not** cover | done |

## P4 — The paper (start the slow part now)

| # | Item | Why | Est |
|---|---|---|---|
| 16 | **Recruit 2 more annotators** [3/3] | The blocker, and a calendar problem — start today. Single-rater ground truth with no reliability coefficient is a desk reject for an evaluation-validity paper. An hour of two colleagues' time | start now |
| 17 | **Blind and interleave the samples** | I knew which items were the disagreement set. Interleave the random and contested items and shuffle | 1h |
| 18 | **Report Krippendorff's α** | With 3 raters on the adjudicated subset | 1h |
| 19 | **The proper two-stage study** | Stratify by disagreement level (0, 1, ≥2), adjudicate a sample of each, then report precision/recall/enrichment. Turns the observation into a method | 1–2d |
| 20 | **Lead with the counterfactual** | The keyword audit found 2 defects and **0 of the 7** the human caught. A baseline with ~0% recall on the target defect class is the strongest thing in the result — stronger than the correlation | framing |
| 21 | **Release the dataset** | 151 items, judge and human scores, adjudications. The corpus is public university policy so it can be released, and the dataset is a contribution independent of the finding | 1d |
| 22 | Frame as item *validity*, not bad references | Two defect types found: bad reference (human 5, judge 1) and bad question (human 1, judge 4). Same phenomenon — the item is not a valid measurement instrument | framing |
| 23 | Compute a proper CI, and the binary analysis | n=30 gives roughly [+0.26, +0.80] — spans weak to strong. Also report AUC for the high/low discrimination, which is where the judge is actually usable | 1h |

**Venue:** an evaluation/resources venue, not negative results. ChatGPT named JUDGe 2026 and EvalEval as strong fits and cited a **29 August 2026** deadline — *unverified, check the actual CFP before planning around it.*

## Reopened by round 56

| # | Item | Why |
|---|---|---|
| 33 | **Parent-child chunking — REOPENED, unmeasured** | Closed in round 7 on the ceiling argument, which round 56 falsified. Unlike the other four it changes what is **indexed**, so the 31%-never-enter-the-pool finding does not cover it — those items could move. It was the weakest of the five closures and it is now back on the list with no measurement either way |
| 34 | **The keyphrase sets are unreliable** | Round 56: the gold document fails its own keyphrase conjunction in **44% of turns** (35/80), mean 50% present. Every keyphrase-derived number inherits this — N, the ceiling, keyphrase coverage, and the TEXT_DRIFT/PARTIAL_DRIFT verdicts in the benchmark audit. Bigger than #5's remaining 6 items and it should probably absorb them |

## P5 — Then, and only then

| # | Item | Why |
|---|---|---|
| 24 | **5–10 real users** [3/3] | After #1–4. All three reviewers rate this highest-value once auth is done. Deployment is already measured as adequate: 1.8× at 8 users, 1.32GB, ~$43/month |
| 25 | **Run the 302-turn baseline once** | As a release-candidate snapshot, after #5 — before the corrections it would precisely measure a corrupt instrument. Free locally in two passes |
| 26 | ~~Partition the ceiling analysis~~ **DONE — and it falsified the ceiling** | `eval/ceiling_partition.py`. Partitioning moves it 84.3% → 90.1% on **5 of 80 turns**. But the run exposed that **the gold document fails its own keyphrase conjunction in 44% of turns** (35/80, mean 50% of keyphrases present). N cannot count "documents holding the answer" when ground truth is uncounted 44% of the time, so **neither ceiling is a bound** and the apparent +5.1 headroom is as untrustworthy as the -0.7 it replaced. Fixing N = fixing keyphrases = items 5–8, not retrieval |
| 27 | ~~Re-close the retrieval proposals on the right argument~~ **DONE** | RM3, document-level rerank, family retriever, cross-reference extraction: closed on the **59/10/31 decomposition**, measured directly and independent of N. **Parent-child chunking is NOT retired** — it changes what is *indexed* and could move items out of that 31%; it returns to the open list, unmeasured |

## Smaller code items

| # | Item |
|---|---|
| 28 | ~~`_stage_note` writes before the directory exists~~ **DONE** — reproduced first (a note written before any timer was lost silently, both swallow exceptions by design), then routed both writers through one `_write` with a shared directory guard |
| 29 | ~~`_top_family_count` defined identically in two live modules~~ **DONE** — moved to `docid.py` beside `document_family`; it cannot live in `rag.py` because `rag` imports `rerank`. Both now resolve to the same function object (asserted) |
| 30 | ~~`_rrf_fuse` drops `ids` while `_dense_as_hits` reads them~~ **DONE** — feeding a fused dict back in hit a `.get("ids", [[]])` default and returned an **empty list rather than raising**. No live path does that today, so this closes a trap rather than fixing an observed defect. `_dedup_by_chunk` carries ids through too |
| 31 | ~~Clear the 29 pyflakes messages~~ **DONE — 31 → 0** | Most were deliberate re-exports external callers depend on; deleting them would have broken `eval/`. They already had `# noqa: F401`, which **pyflakes does not honour** (that is flake8) — which is exactly why they survived. Declared via `__all__`, which pyflakes does honour. Four dead imports and one no-op `global` removed |
| 32 | **DECLINED, with reasons** — move falsified-mechanism rationale out and drop dead paths | There are **21 flag-gated mechanisms defaulting to False**, each carrying the falsification that put it there. `CLAUDE.md` names this as the project's convention ("New retrieval/prompt mechanisms default to `False`... each with its falsification recorded"), and the project's stated value is that *the record of what didn't work is worth more than the code*. Separating a mechanism from the reason it failed makes the next person likelier to rebuild it. The reviewer's underlying concern was module bloat, and that was already addressed by the refactor (`rag.py` 2,237 → 1,270). Deleting live-but-off code paths would also risk behaviour change for zero measured benefit. **Done instead:** asserted all 21 flags still evaluate to `False` at runtime — a flag silently flipping ON is the failure actually worth guarding against, and nothing was checking it |

---

## Decided against, with reasons

| Item | Why not |
|---|---|
| Cross-model judge validation | Real point — the finding may be "Qwen is a bad judge for this task". But it multiplies the annotation burden, and #16 must come first. Revisit once 3 raters exist |
| Replicate on a public benchmark (HotpotQA etc.) | Needed for a strong submission, not for deciding whether there's a paper. After #19 |
| Pydantic `BaseSettings` + typed dataclasses between modules | Good architecture, no current defect motivating it, and a large diff across every module days after a refactor that broke twice |
| Content-hashed IDs across stores | The specific failure it prevents (vector/lexical ID misalignment) has never occurred here — both stores key on `source_url` + `chunk_index`. #9 covers the real gap |
| Tokenizer/vocabulary drift | Speculative. No instance observed. #9 would catch it if it appeared |
| More retrieval mechanisms | 31% of misses never enter the pool; all three reviewers said stop |
