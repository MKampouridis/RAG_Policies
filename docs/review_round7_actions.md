# Round 7 external review — full action backlog

Every distinct recommendation from the four reviews (Opus 5 ×2, Gemini, ChatGPT,
DeepSeek), 2026-08-11. Nothing curated out — an earlier summary showed ~15 of
these and that was misleading.

**Status key:** `DONE` · `CONFIRMED` (verified real, not fixed) · `OPEN`
(plausible, unverified) · `REJECTED` (checked, not a defect) · `MOOT`
(superseded by a measurement)

Attribution: **O**=Opus 5, **G**=Gemini, **C**=ChatGPT, **D**=DeepSeek.

**Progress:** 68 of 94 resolved. Phases 1-6 complete, Phase 7 substantially done; measurements in `eval/report.md` Rounds 13-29.

---

## 1. Correctness defects

| # | Item | Who | Status |
|---|---|---|---|
| 1 | `hit@6` scored `rank is not None` — no cap; 7 false hits | O | **DONE** |
| 2 | `_names_partner_institution` unbounded substrings ("eput" in *deputy*) | O | **DONE** |
| 3 | Multi-entity re-admits partner docs AFTER exclusion; demotion only reorders | O | **DONE** — `MULTI_ENTITY_PARTNER_RECHECK`. Exhibited on the 3 aliases lacking department metadata: **3 partner chunks → 0**, 0 of 23 other questions changed. `eval/partner_multientity_probe.py` is the regression test |
| 4 | Multi-entity per-entity retrievals are dense-only — no BM25, no RRF | O | **MEASURED NULL, LEFT OFF** — `MULTI_ENTITY_LEXICAL`. Set 5 coverage 22/23 both arms, 0 of 10 changed. Verified the path fires first (3 BM25 calls, 144 hits). Real defect, no consequence |
| 5 | `_adjacent_chunks` drops `distances` | O | **DONE** — preserved; neighbours carry `None` rather than a fabricated distance |
| 6 | `_adjacent_chunks` does 2 Chroma round-trips | O | **DONE** — single `$or` query |
| 7 | `_adjacent_chunks` uses a metadata scan | O | **DONE via #6** — one scan instead of two; rows verified rather than trusted by position |
| 8 | No assertion that an adjacent chunk belongs to the same document | D | **DONE** — each row checked against the requested (url, index) set |
| 9 | Entity budget leaves only 3 of 14 slots for base ranking | O | **INSTRUMENTED, not changed** — #10 now logs fill per entity; changing the budget moves retrieval and needs the 151-question set first |
| 10 | Log per-entity candidate counts | D | **DONE** — candidates/took/starved under `RAG_TIMING`. First run: MSAS 3 candidates vs CSEE 48 |
| 11 | Conversation summarisation race | C | **DONE** — per-conversation lock; 4 concurrent calls summarise **once** |
| 12 | Mid-stream failure leaves the conversation inconsistent | D | **DONE** — and it found a bug in MY streaming fallback: the question could be stored twice. Retry is now allowed only before the first SSE byte |
| 13 | `FETCH_POOL_MULTIPLIER=4` "costs quality" unsupported | O | **REJECTED** — came from stage-2 e2e run (fc1bf45), not the sweep file |

## 2. Security, data safety, multi-user

| # | Item | Who | Status |
|---|---|---|---|
| 14 | **No auth/ownership: `/api/conversations` returns everyone's** | O,C,G | **DEFERRED BY DECISION (2026-08-11)** — single-user local machine; auth is a precondition for SHARING, not for today. Revisit before any colleague uses it |
| 15 | Add `user_id` to the schema NOW | C | **DEFERRED BY DECISION** — follows #14. Noted that doing it later costs a migration on a DB with a data-loss history |
| 16 | SQLite has no WAL mode → "database is locked" with 2 users | O,G,C | **DONE** — `journal_mode=WAL` verified active |
| 17 | `busy_timeout` unset | G,C | **DONE** — 5000ms + `timeout=5.0` on connect |
| 18 | FK declared without `ON DELETE CASCADE`; deletion is 2 manual statements | C | **REJECTED** — `with _connect()` already wraps both DELETEs in one transaction, so deletion is atomic. CASCADE needs a table rebuild on a DB with a data-loss history: cosmetic gain, real risk |
| 19 | Soft deletes instead of destructive DELETE | G | **DONE** — `deleted_at`; no endpoint removes rows. `restore_history.py` replaces the forensic carver |
| 20 | Mass delete should be an atomic background job, not client-driven | G | **PARTLY MOOT** — clear-all is now a sequence of reversible soft deletes, so partial failure is recoverable rather than destructive. Still client-driven |
| 21 | **Second data-loss cause still unexplained** | O | **MITIGATED, NOT EXPLAINED** — cause still unknown and stays open; #19 makes it non-destructive, so it can now only hide conversations |
| 22 | Append-only writes / backup before every destructive op | O | **DONE via #19** — deletion is now non-destructive by construction, which is stronger than backing up before it. Nightly `.backup` still runs |
| 23 | Feedback payload is client-asserted | C | **DONE** — question/answer now read from the server's stored conversation; a client mismatch is flagged (`client_server_mismatch`) rather than silently overwritten, and provenance is attached |
| 24 | Per-user database isolation | G | **DEFERRED BY DECISION** — follows #14 |
| 24b | Server binds `0.0.0.0`, reachable at 192.168.86.103:8000 by anyone on the LAN | *(found while checking #14)* | **DECIDED 2026-08-11: leave as is** — deliberate, it is what makes the mobile UI reachable from a phone. Trusted home network. Cost stated: the exposure is real on campus/public wifi |
| 25 | Feedback JSONL has no rotation or size cap | D | **REJECTED** — rotation already exists (`_MAX_BYTES` = 50MB, `src/feedback.py:24`) |

## 3. Provenance & auditability

| # | Item | Who | Status |
|---|---|---|---|
| 26 | Record provenance per answer | C | **DONE** — corpus version, code revision, generator, contextualizer on every answer and `/api/config` |
| 27 | Promote staleness from UI feature to first-class answer property | C | OPEN |
| 28 | Store staleness in DB metadata at query time for audit compliance | G | OPEN |

## 4. Evaluation methodology — the strongest consensus

| # | Item | Who | Status |
|---|---|---|---|
| 29 | **Paired diff + bootstrap CI + win/tie/loss** instead of comparing two means | O,G,C,D | **DONE** — `eval/compare.py`. Re-analysed Round 8p: mean +0.050, CI [-0.250,+0.350], **3 win / 14 tie / 3 loss** |
| 30 | Report "turns changed" beside every mean | O | **DONE** — printed by default; 6 of 20 for Round 8p |
| 31 | Build `eval/compare.py` | O | **DONE** — both `score` and `hit_at_6` modes |
| 32 | Stop saying "confirmed no harm" | O | **DONE** — retracted in Round 15; the tool now prints the distinction so it is not left to prose |
| 33 | Evaluate the ~8 turns a mechanism CAN change, binary, not a 20-turn mean | O | OPEN |
| 34 | Core regression set 100–150, stratified | C,G,D | **DONE** — `eval/questions_regression.json`, **151 questions / 302 turns**, built by `build_regression_set.py`; thin strata reported on every run |
| 35 | Targeted failure set (10–30) per mechanism | C | **STARTED** — `eval/partner_multientity_probe.py` is the first; built because 3 existing instruments were blind to the defect |
| 36 | Canary set (~50) that must never regress | C | OPEN |
| 37 | Decouple retrieval eval from generation eval | G,D | **ALREADY TRUE** — `retrieval_replay.py` scores hit@6 with no generation or judging; used throughout Rounds 13-21 |
| 38 | Cloud smoke set ~12 turns × n=3, report **rates** not mean judge scores | O | OPEN |
| 39 | Periodic shadow eval (20 q) on production config | D | OPEN |
| 40 | Human-calibrate the judge on ~30 questions | C | OPEN |
| 41 | Repeats strategically (dev 1× → promising 2× → headline both arms) | C | OPEN |
| 42 | Three shipping classes: A broad/statistical, B targeted, C UX-by-judgement | C | OPEN |
| 43 | Tag each mechanism "wrong for anyone" vs "right for me" | O | OPEN |
| 44 | Use feedback only to author test cases, never tune against the log | G | OPEN |
| 45 | Eval harness should refuse to run without the deterministic env var | D | **DONE** — `run_eval.py` refuses; explicit override for cloud arms |
| 46 | Config schemas (`.env.eval` / `.env.prod`) rather than a port-collision guard | G | OPEN |
| 47 | Adjacent-chunk: shipped config ≠ tested config — "accepted low-risk", not validated | C | **DONE** — Round 15 restates it as a targeted decision on a named failure case, not a measured improvement |

## 5. Latency

| # | Item | Who | Status |
|---|---|---|---|
| 48 | Stream the answer | O,G,C,D | **DONE** — dead time 14.9s → 7.2s |
| 49 | Pooled `requests.Session` (fresh TLS per call) | O | **DONE** |
| 50 | ColBERT query encode is 0.76s → check device/dtype/`torch.compile` | O,G,C,D | **MOOT** — actually **19ms** warm; my 0.76s was the first-encode warmup, double-counted |
| 51 | `pylate-rs` / WARP / PLAID / ONNX-INT8 backends | G,C | **MOOT** — rerank is only 0.20–0.25s total |
| 52 | pylate `get_documents_embeddings` does a full `pickle.load` per call | O | **DONE** — 127.5ms → **3.0ms**; embeddings verified identical across 30 docs |
| 53 | **ColBERT index residency degrades Chroma 81ms → 2543ms** | *(mine)* | **SHIPPED after an eval (Round 27)** — cache was also 3 weeks STALE. hit@6 **121 → 126**, retrieve ~5.0s → **1.45s**, RSS 5.0 → **2.12GB**. Drift detector added |
| 54 | Progress indicator during retrieval | O,G,C,D | **DONE** — SSE `stage` events drive "Searching…" → "Writing…" |
| 55 | Warmup only warms the PRIMARY path; follow-ups still cold | O | **DONE** — `_identity_anchor_index()` (1,188 files, **161ms**) now built at startup |
| 56 | Timers inside `retrieve()` | O,C | **DONE** — warm: rerank **1.15s of 1.42s (82%)**, dense 0.18s. Target has moved to reranking |
| 57 | `_stage_timer` does mkdir+open+append+close per stage per request, imports in function body | O | **DONE** — imports + mkdir hoisted; verified still writing |
| 58 | Log `usage.input_tokens`/`output_tokens` beside generate seconds | O,C | **DONE** — `data/usage.jsonl`, both streaming and blocking paths |
| 59 | `max_tokens=2048` is not a latency lever | O,C,D | **CLOSED — confirmed a non-lever.** A real answer used 996 of 2048 and stopped on `end_turn`; 0 of 2 turns hit the cap |
| 60 | Prompt caching | O,G,C | **CLOSED — no qualifying prefix.** System prompt ~385 tokens (<1024 minimum) and retrieved context changes every turn |
| 61 | **`concise` as default** | O | **DONE (user's instruction)** — 21–55% shorter on 3 enumeration/list cases, all keyphrases kept, identical content coverage |
| 62 | Batch/parallelise multi-entity queries (`asyncio.gather`) | G,C | OPEN |
| 63 | Do entity queries need ColBERT each, or one final rerank? | C | OPEN |
| 64 | Sweep `FETCH_POOL_MULTIPLIER` **upward** as a diagnostic | O | **DONE** — `eval/far_miss_taxonomy.py`. 59% ranking / 10% pool-size / **31% never enter the pool at all** |
| 65 | Partition `latency.jsonl` at the adjacent-expansion commit — free check | O | **DONE, NULL** — retrieve median 6.54s before → 3.03s after. No step-up. (n=12 before, so weak, and confounded by other changes) |
| 66 | py-spy once, after stage instrumentation | C | **NOT NEEDED** — the in-stage timers (#56) attribute retrieval to 0.02s granularity; a profiler would add nothing |
| 67 | Report 4 distributions incl. session-start latency; targets TTFT<4s, p50<8s, p90<12s | C | OPEN |
| 68 | BM25 sort / `_prefer_most_recent_year` are NOT levers (5ms, sub-ms) | O | Noted |

## 6. Operational risk

| # | Item | Who | Status |
|---|---|---|---|
| 69 | Warmup + `KeepAlive` = possible OOM boot loop; check `ThrottleInterval` | O,G,D | **CHECKED, BOUNDED** — `ThrottleInterval` already 30s; 0 crashes/tracebacks in the log. A loop would be ~50% duty cycle, not tight. No change made |
| 70 | Warmup failure only prints to a log nobody reads | O | **DONE** — `WARMUP_STATE` exposed at `/api/config` |
| 71 | Explicit readiness state STARTING→WARMING→READY | G,C | **DONE** — starting/warming/ready/failed at `/api/config` |
| 72 | A/B can't detect paging after hours idle — consider a 5-min keep-alive tickle | O | OPEN |
| 73 | Concurrency: p90 is a single-user p90 | O,C | **FALSIFIED** — 4 concurrent requests, slowest 12.4s vs 12.2s serial: **1.0x**. Most of a turn is cloud network wait |
| 74 | Test 4 simultaneous requests | C,O | **DONE** — see #73 |
| 75 | Monitor swap/compressed memory on 16GB unified | C,G | OPEN |
| 76 | **Silent cloud→local fallback should fail loud** | G | **DONE** — daemon sets `RAG_DEGRADED`; UI shows a red banner. Deliberate local mode is not flagged |

## 7. Code structure

| # | Item | Who | Status |
|---|---|---|---|
| 77 | `rag.py` is 1,688 lines — split into retrieval/generation/conversation modules | C | OPEN |
| 78 | Move falsified mechanisms out of production code; git history is the reference | C | OPEN |
| 79 | Deterministic output scrubber for plumbing language | G | **DONE** — post-generation substitution, narrow by design; testing caught a sentence-capitalisation bug and a mid-stream flicker, both recorded |
| 80 | Title generation is another non-determinism source | D | OPEN |

## 8. Retrieval research — none of these are on the falsified list

| # | Item | Who | Status |
|---|---|---|---|
| 81 | **Classify the misses before any new mechanism** | C | **DONE** — 59% ranking / 10% pool-size / **31% never enter the pool**. The follow-up "reranker demotes depth-1 docs" reading was **RETRACTED** (Round 22): 4 of 5 were a title-page artefact, partner exclusion, or a better document served |
| 82 | Run `gold_multiplicity.py` on the misses | O | **DONE** — actual 86.2% vs achievable ceiling 84.3%. **7 of 11 misses are metric/test artifacts**; only 4 are reachable |
| 83 | RM3 pseudo-relevance feedback on the BM25 channel | O | **CLOSED ON EVIDENCE (Round 29)** — at most 4 of 80 turns on the main set are reachable retrieval targets, and measured hit@6 (86.2%) already exceeds the achievable ceiling (84.3%). Not falsified — unjustified. Reopen only if a specific failing case survives the Round 29 classification |
| 84 | Document-level rerank by aggregated chunk scores | D | **CLOSED ON EVIDENCE (Round 29)** — at most 4 of 80 turns on the main set are reachable retrieval targets, and measured hit@6 (86.2%) already exceeds the achievable ceiling (84.3%). Not falsified — unjustified. Reopen only if a specific failing case survives the Round 29 classification |
| 85 | Query→document-family retriever as a candidate gate | C | **CLOSED ON EVIDENCE (Round 29)** — at most 4 of 80 turns on the main set are reachable retrieval targets, and measured hit@6 (86.2%) already exceeds the achievable ceiling (84.3%). Not falsified — unjustified. Reopen only if a specific failing case survives the Round 29 classification |
| 86 | Parent-child indexing | G | **CLOSED ON EVIDENCE (Round 29)** — at most 4 of 80 turns on the main set are reachable retrieval targets, and measured hit@6 (86.2%) already exceeds the achievable ceiling (84.3%). Not falsified — unjustified. Reopen only if a specific failing case survives the Round 29 classification. Also note Round 8q measured structure-aware chunking as *worse* |
| 87 | Extract cross-references at ingest into metadata | G | **CLOSED ON EVIDENCE (Round 29)** — at most 4 of 80 turns on the main set are reachable retrieval targets, and measured hit@6 (86.2%) already exceeds the achievable ceiling (84.3%). Not falsified — unjustified. Reopen only if a specific failing case survives the Round 29 classification |

## 9. Product decisions

| # | Item | Who | Status |
|---|---|---|---|
| 88 | Partner: demote + **cap at one slot** instead of excluding (~5 lines, keeps rank-6 reachable) | O | OPEN |
| 89 | Partner: soft boost (+15%) to Essex rather than a hard gate | G | OPEN |
| 90 | Partner: expose `include_partner_docs` as a toggle | D | OPEN |
| 91 | Partner: re-measure set 6 now the gate bug is fixed — some of 2/2→0/2 may have been gate error | O | OPEN |
| 92 | Make "audience = Essex staff, default institution = Essex" an explicit documented product policy | C | OPEN |
| 93 | Get 5–10 real users, prioritising diversity of question style over volume | O,G,C,D | OPEN |
| 94 | Write the paper — the falsification record is itself the contribution | D | OPEN |

## 10. What all four said to STOP

- Stop tuning retrieval hyper-parameters (rerankers, fusion weights, top-k, chunk size)
- Stop building new eval sets — six exist, each told you less than the last
- Stop shipping mechanisms that can't clear the noise floor
- Stop running end-to-end LLM evals for retrieval-only changes
- Stop trying to prove wording changes statistically

---

## Recommended order

1. **#3, #4** — confirmed defects affecting production now
2. **#29/#31** — paired comparison tooling; ~1h, no compute, improves every later decision
3. **#14–#22** — auth, WAL, and the unexplained data loss: the actual gate to other people using this
4. **#54, #61, #53** — progress indicator, concise-by-default, then the cache experiment behind an eval
5. **#81** — far-miss taxonomy before any new retrieval mechanism
