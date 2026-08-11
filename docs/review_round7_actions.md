# Round 7 external review — full action backlog

Every distinct recommendation from the four reviews (Opus 5 ×2, Gemini, ChatGPT,
DeepSeek), 2026-08-11. Nothing curated out — an earlier summary showed ~15 of
these and that was misleading.

**Status key:** `DONE` · `CONFIRMED` (verified real, not fixed) · `OPEN`
(plausible, unverified) · `REJECTED` (checked, not a defect) · `MOOT`
(superseded by a measurement)

Attribution: **O**=Opus 5, **G**=Gemini, **C**=ChatGPT, **D**=DeepSeek.

---

## 1. Correctness defects

| # | Item | Who | Status |
|---|---|---|---|
| 1 | `hit@6` scored `rank is not None` — no cap; 7 false hits | O | **DONE** |
| 2 | `_names_partner_institution` unbounded substrings ("eput" in *deputy*) | O | **DONE** |
| 3 | Multi-entity re-admits partner docs AFTER exclusion; demotion only reorders | O | **CONFIRMED** |
| 4 | Multi-entity per-entity retrievals are dense-only — no BM25, no RRF | O | **CONFIRMED** |
| 5 | `_adjacent_chunks` drops `distances` that `_exclude_partner_institutions` keeps | O | OPEN |
| 6 | `_adjacent_chunks` does 2 Chroma round-trips where one `$or` would do | O | OPEN |
| 7 | `_adjacent_chunks` uses a metadata scan; chunk ids are deterministic → use `ids=[...]` | O | OPEN |
| 8 | No assertion that an adjacent chunk belongs to the same document | D | OPEN |
| 9 | With 11 aliases the entity budget leaves only 3 of 14 slots for base ranking | O | OPEN |
| 10 | Multi-entity budget can starve genuine docs — log per-entity candidate counts | D | OPEN |
| 11 | Conversation summarisation race: two requests can both summarise | C | OPEN |
| 12 | Mid-stream cloud failure may leave the conversation inconsistent | D | OPEN |
| 13 | `FETCH_POOL_MULTIPLIER=4` "costs quality" unsupported | O | **REJECTED** — came from stage-2 e2e run (fc1bf45), not the sweep file |

## 2. Security, data safety, multi-user

| # | Item | Who | Status |
|---|---|---|---|
| 14 | **No auth/ownership: `/api/conversations` returns everyone's** | O,C,G | OPEN — **precondition to sharing** |
| 15 | Add `user_id` to the schema NOW; `WHERE id=? AND user_id=?` | C | OPEN |
| 16 | SQLite has no WAL mode → "database is locked" with 2 users | O,G,C | OPEN |
| 17 | `busy_timeout` unset | G,C | OPEN |
| 18 | FK declared without `ON DELETE CASCADE`; deletion is 2 manual statements | C | OPEN |
| 19 | Soft deletes (`is_archived`) instead of destructive DELETE | G | OPEN |
| 20 | Mass delete should be an atomic background job, not client-driven | G | OPEN |
| 21 | **Second data-loss cause still unexplained** — stop-ship until found | O | OPEN |
| 22 | Append-only writes / backup before every destructive op | O | OPEN |
| 23 | Feedback payload is client-asserted; derive question/answer/sources server-side from `conversation_id`+`message_id` | C | OPEN |
| 24 | Per-user database isolation | G | OPEN |
| 25 | Feedback JSONL has no rotation or size cap | D | OPEN |

## 3. Provenance & auditability

| # | Item | Who | Status |
|---|---|---|---|
| 26 | Record `corpus_version`, `retriever_version`, generator, contextualizer per answer | C | OPEN |
| 27 | Promote staleness from UI feature to first-class answer property | C | OPEN |
| 28 | Store staleness in DB metadata at query time for audit compliance | G | OPEN |

## 4. Evaluation methodology — the strongest consensus

| # | Item | Who | Status |
|---|---|---|---|
| 29 | **Paired diff + bootstrap CI + win/tie/loss** instead of comparing two means. Opus ran it on my adj0/adj1: mean **+0.000**, **20 of 26 turns unchanged** | O,G,C,D | OPEN — **highest value, ~1h, zero compute** |
| 30 | Report "turns changed" beside every mean | O | OPEN |
| 31 | Build `eval/compare.py` | O | OPEN |
| 32 | Stop saying "confirmed no harm" — say "failed to detect harm >0.2" | O | OPEN (wording, real) |
| 33 | Evaluate the ~8 turns a mechanism CAN change, binary, not a 20-turn mean | O | OPEN |
| 34 | Core regression set 100–150, stratified by question type | C,G,D | OPEN |
| 35 | Targeted failure set (10–30) per mechanism | C | OPEN |
| 36 | Canary set (~50) that must never regress | C | OPEN |
| 37 | Decouple retrieval eval (deterministic, £0) from generation eval | G,D | OPEN |
| 38 | Cloud smoke set ~12 turns × n=3, report **rates** not mean judge scores | O | OPEN |
| 39 | Periodic shadow eval (20 q) on production config | D | OPEN |
| 40 | Human-calibrate the judge on ~30 questions | C | OPEN |
| 41 | Repeats strategically (dev 1× → promising 2× → headline both arms) | C | OPEN |
| 42 | Three shipping classes: A broad/statistical, B targeted, C UX-by-judgement | C | OPEN |
| 43 | Tag each mechanism "wrong for anyone" vs "right for me" | O | OPEN |
| 44 | Use feedback only to author test cases, never tune against the log | G | OPEN |
| 45 | Eval harness should refuse to run without the deterministic env var | D | OPEN |
| 46 | Config schemas (`.env.eval` / `.env.prod`) rather than a port-collision guard | G | OPEN |
| 47 | Adjacent-chunk: shipped config ≠ tested config — call it "accepted low-risk", not validated | C | OPEN (wording, correct) |

## 5. Latency

| # | Item | Who | Status |
|---|---|---|---|
| 48 | Stream the answer | O,G,C,D | **DONE** — dead time 14.9s → 7.2s |
| 49 | Pooled `requests.Session` (fresh TLS per call) | O | **DONE** |
| 50 | ColBERT query encode is 0.76s → check device/dtype/`torch.compile` | O,G,C,D | **MOOT** — actually **19ms** warm; my 0.76s was the first-encode warmup, double-counted |
| 51 | `pylate-rs` / WARP / PLAID / ONNX-INT8 backends | G,C | **MOOT** — rerank is only 0.20–0.25s total |
| 52 | pylate `get_documents_embeddings` does a full `pickle.load` per call — memoise | O | CONFIRMED (23MB, 136ms/query) |
| 53 | **ColBERT index residency degrades Chroma 81ms → 2543ms** — cache off: retrieve 5.34→1.40s, RSS 5.0→2.1GB | *(mine)* | **CONFIRMED, NOT SHIPPED** — changes retrieval on 2/5 probes; needs an eval |
| 54 | Progress indicator during retrieval | O,G,C,D | OPEN — streaming can't cover the ~5s search |
| 55 | Warmup only warms the PRIMARY path; follow-ups still cold (`_identity_anchor_index` globs `data/doc_identity/`) | O | OPEN |
| 56 | 8–13 timers inside `retrieve()` | O,C | OPEN |
| 57 | `_stage_timer` does mkdir+open+append+close per stage per request, imports in function body | O | OPEN |
| 58 | Log `usage.input_tokens`/`output_tokens` beside generate seconds | O,C | OPEN |
| 59 | `max_tokens=2048` is not a latency lever — check `stop_reason` first | O,C,D | OPEN (probably no-op) |
| 60 | Prompt caching: system prompt is ~385 tokens, below the 1024 minimum — cost lever, not latency | O,G,C | OPEN |
| 61 | **`concise` = −27% output, already measured quality-neutral ≈ −1.8s, still not default** | O | OPEN |
| 62 | Batch/parallelise multi-entity queries (`asyncio.gather`) | G,C | OPEN |
| 63 | Do entity queries need ColBERT each, or one final rerank? | C | OPEN |
| 64 | Sweep `FETCH_POOL_MULTIPLIER` **upward** (16, 32) on the FAR subset as a diagnostic | O | OPEN |
| 65 | Partition `latency.jsonl` at the adjacent-expansion commit — free check | O | OPEN |
| 66 | py-spy once, after stage instrumentation | C | OPEN |
| 67 | Report 4 distributions incl. session-start latency; targets TTFT<4s, p50<8s, p90<12s | C | OPEN |
| 68 | BM25 sort / `_prefer_most_recent_year` are NOT levers (5ms, sub-ms) | O | Noted |

## 6. Operational risk

| # | Item | Who | Status |
|---|---|---|---|
| 69 | Warmup + `KeepAlive` = possible OOM boot loop (18s CPU + 4.5GB per cycle); check `ThrottleInterval` | O,G,D | OPEN |
| 70 | Warmup failure only prints to a log nobody reads | O | OPEN |
| 71 | Explicit readiness state STARTING→WARMING→READY | G,C | OPEN |
| 72 | A/B can't detect paging after hours idle — consider a 5-min keep-alive tickle | O | OPEN |
| 73 | Concurrency: FastAPI threadpool, one ColBERT, GIL — **p90 is a single-user p90** | O,C | OPEN |
| 74 | Test 4 simultaneous requests | C,O | OPEN |
| 75 | Monitor swap/compressed memory on 16GB unified | C,G | OPEN |
| 76 | **Silent cloud→local fallback should fail loud** — an expired key silently degrades quality | G | OPEN |

## 7. Code structure

| # | Item | Who | Status |
|---|---|---|---|
| 77 | `rag.py` is 1,688 lines — split into retrieval/generation/conversation modules | C | OPEN |
| 78 | Move falsified mechanisms out of production code; git history is the reference | C | OPEN |
| 79 | Replace prompt-based plumbing suppression with a deterministic output scrubber (`USER_FACING_LANGUAGE` only got 4/4→2/4) | G | OPEN |
| 80 | Title generation is another non-determinism source | D | OPEN |

## 8. Retrieval research — none of these are on the falsified list

| # | Item | Who | Status |
|---|---|---|---|
| 81 | **Classify the 11% far-misses into a taxonomy (A–I) and count** before any new mechanism | C | OPEN — cheapest, most informative |
| 82 | Run `gold_multiplicity.py` on the FAR subset — some isn't fixable, remove from denominator | O | OPEN |
| 83 | RM3 pseudo-relevance feedback on the BM25 channel only | O | OPEN |
| 84 | Document-level rerank: group pool by document, score by aggregated top-3 chunk MaxSim | D | OPEN |
| 85 | Train a query→document-FAMILY retriever from known failures, as a candidate gate (not another RRF channel) | C | OPEN |
| 86 | Parent-child indexing: small child chunks for matching, inject parent section | G | OPEN |
| 87 | Extract explicit cross-references at ingest ("see Regulation 9.1") into metadata | G | OPEN |

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
