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
| 1 | **Authentication** [3/3] | `_owner()` trusts a header the user types. Anyone can read anyone's history by sending a different name. For "what happens if I fail my resit", that is not a v2 item. Shared password + the existing name separation is enough for a trial | 1–2h |
| 2 | **Research routes truncate data with no auth** | `research_routes.py` does `write_text()` — full replace, no auth, no backup. A POST of `[]` erases the 30 blind human judgements the paper depends on. Write timestamped + symlink latest, or copy-before-write | 30m |
| 3 | **Streaming has no completion check** | `llm.py` returns `"".join(parts)` when the stream ends, so a dropped connection stores a **silently truncated answer**, indistinguishable from a complete one. Observe `message_stop`; flag or raise if absent. On a policy tool, an answer ending mid-sentence at "students may appeal if" is worse than an error | 45m |
| 4 | **Push `owner` into the SQL** | `delete_conversation`, `restore_conversation` and `list_deleted_conversations` have no owner in their WHERE clauses — the invariant lives in the caller. `list_deleted_conversations` has no owner parameter at all, so a recovery UI would show everyone's deleted conversations | 30m |

## P1 — Benchmark provenance (do as one job, not 8 edits)

| # | Item | Why | Est |
|---|---|---|---|
| 5 | **Re-derive the eval set against the current corpus** [3/3] | The test set is a **stale cache of the corpus** — same bug class as the ColBERT index. Verified: **9 of 148 gold URLs point at NON-current documents**, so retrieval is scored a miss for returning the current edition. Patching 8 known-bad items leaves 143 with unverified provenance | 1–2d |
| 6 | **Stamp `corpus_version` into `questions.json`** | And check it in the post-ingest sequence. Converts "someone remembers to look" into "fails closed" | 1h |
| 7 | **Record correction provenance** | For each fixed reference: old text, new text, source URL, passage, why it was wrong, who, when. This becomes part of the benchmark's published provenance | with #5 |
| 8 | **Fix the 8 defective references** | Now a subset of #5 rather than its own task. Page already built at `/reference-fix` | with #5 |

## P2 — Stale-artifact hygiene (the general class)

| # | Item | Why | Est |
|---|---|---|---|
| 9 | **Provenance stamp on every derived artifact, checked at read, failing closed** [3/3] | The rule the ColBERT incident implies. Counts and mtimes are proxies; hashes are the thing. `lexical.py` and `doc_index.py` already invalidate against `read_corpus_version()` — ColBERT never did | 2–3h |
| 10 | **The drift check compares counts only** | A document edited in place with the same chunk count passes it. The last re-ingest was *5 new and ~20 changed* — the changed ones are exactly what a count check cannot see | 30m |
| 11 | Audit the other unstamped artifacts | SPLADE, pseudo-query and ensemble indexes (off but live), `data/doc_identity/*.json` (never invalidated when its document changes), the process-lifetime caches | 1h |

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

## P5 — Then, and only then

| # | Item | Why |
|---|---|---|
| 24 | **5–10 real users** [3/3] | After #1–4. All three reviewers rate this highest-value once auth is done. Deployment is already measured as adequate: 1.8× at 8 users, 1.32GB, ~$43/month |
| 25 | **Run the 302-turn baseline once** | As a release-candidate snapshot, after #5 — before the corrections it would precisely measure a corrupt instrument. Free locally in two passes |
| 26 | Partition the ceiling analysis | Apply the exchangeability bound only to turns with **no** disambiguating entity; elsewhere the bound is 1.0. Then re-state the residual honestly |
| 27 | Re-close the retrieval proposals on the right argument | The 31%-never-enter-pool finding, not the ceiling. Note this does **not** retire parent-child chunking, which changes what is indexed |

## Smaller code items

| # | Item |
|---|---|
| 28 | `_stage_note` writes before the directory exists; only `_stage_timer` does the mkdir. Instrumentation that fails invisibly is how the latency picture went wrong before |
| 29 | `_top_family_count` defined identically in `rag.py` and `rerank.py`, both live — the next edit to one is a silent divergence |
| 30 | `_rrf_fuse` drops `ids` while `_dense_as_hits` produces them — two functions in one small module disagreeing about shape |
| 31 | Clear the 29 pyflakes messages (unused imports, bogus `global` declarations). They are noise in the one tool that would have saved the 503 |
| 32 | Move the falsified-mechanism rationale to `eval/report.md` / decision records and drop dead paths from production modules |

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
