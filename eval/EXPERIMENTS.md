# Experiment Log

Parameter and result tracking for every eval pass run against this system, so a
future regression can be traced to an exact configuration and, if needed,
reverted via git (`git log --oneline`, `git checkout <hash> -- src/ reembed.py
run_ingest.py`). Prose narrative for these same experiments is in `report.md`;
this file is the fast-scan parameter reference.

All passes: `qwen2.5:7b-instruct` chat model, same 40-question original set
unless noted. `CHUNK_WORDS=300`/`CHUNK_OVERLAP_WORDS=50` through `postfix4`;
changed to **175/30** at `stage0_chunks` (a later improvement round) and
unchanged since.

| Experiment | Embed model | Chunk headers | Text cleaning | `is_current` filter | Recency dedupe | Hybrid BM25 | Reranker | Year-mention handling | Results file |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | nomic-embed-text | no | no | no | no | no | no | n/a | `results_baseline.json` |
| `fixed` | nomic-embed-text | no | no | no | raw-string family max | no | no | n/a | `results_fixed.json` |
| `mxbai` | mxbai-embed-large | no | no | no | raw-string family max | no | no | n/a | `results_mxbai.json` |
| `stage2` | nomic-embed-text | yes | yes | v1 (no grace period, raw years) | raw-string family max | no | no | unanchored regex → fully unfiltered | `results_stage2.json` |
| `stage3` **(pre-fix production)** | nomic-embed-text | yes | yes | v1 | raw-string family max | yes (RRF k=60) | no | unanchored regex → fully unfiltered | `results_stage3.json` |
| `stage4` (rejected) | nomic-embed-text | yes | yes | v1 | raw-string family max | yes | yes (qwen listwise, top-24→6) | unanchored regex → fully unfiltered | `results_stage4.json` |
| `postfix` (superseded) | nomic-embed-text | yes | yes | v2 (1yr grace, normalized years, sibling-sync) | canonical-year family max, unknown-year always kept | yes | no | anchored regex → **hard filter** to that year | `results_postfix.json` |
| `postfix2` **(= "Final", current production)** | nomic-embed-text | yes | yes | v2 | canonical-year family max, unknown-year always kept | yes | no | anchored regex → **soft preference** (RRF-fuse year pool + current pool) | `results_postfix2.json` |
| `holdout_set2` | nomic-embed-text | yes | yes | v2 | canonical-year family max | yes | no | soft preference | `results_holdout_set2.json`, question set: `questions_set2.json` |
| `postfix3` (superseded) | nomic-embed-text | yes | yes | v2 | canonical-year family max | yes | no | soft preference + contextualize faithfulness guard + reworded contextualize prompt | `results_postfix3.json` |
| `postfix4` (superseded) | nomic-embed-text | yes | yes | v2 | canonical-year family max | yes | no | soft preference + contextualize faithfulness guard, **original contextualize prompt wording** | `results_postfix4.json` |
| `stage0_chunks` (superseded) | nomic-embed-text | yes | yes | v2 | canonical-year family max | yes | no | same as postfix4 | `results_stage0_chunks.json` — **CHUNK_WORDS=175, CHUNK_OVERLAP_WORDS=30** (was 300/50) |
| `stage1_rerank` (superseded) | nomic-embed-text | yes | yes | v2 | canonical-year family max | yes | **yes, `BAAI/bge-reranker-base` over top-30 fused candidates** | same as stage0_chunks | `results_stage1_rerank.json` — pool size widened `FETCH_POOL_MULTIPLIER` 4→8 (24→48 candidates) to give the reranker real depth to work with |
| `stage_colbert` **(= current production)** | nomic-embed-text | yes | yes | v2 | canonical-year family max | yes | **yes, ColBERT late-interaction (`lightonai/GTE-ModernColBERT-v1` via PyLate) replacing the cross-encoder** | same as stage1 | `results_stage_colbert.json` — `src/rerank.py` `BACKEND = "colbert"`; new deps `pylate`, `ragatouille` (the latter installed but unused after a langchain-version incompatibility, kept installed as a no-op) |
| `stage2_header_boost` (rejected — regressed RoA) | nomic-embed-text | yes | yes | v2 | canonical-year family max | yes | yes | same as stage1 | `results_stage2_header_boost.json` — BM25 `chunk_header` repeated 5x in indexed text; regressed RoA hit@6 60%→53%, reverted (`HEADER_WEIGHT=1` in `src/lexical.py`) |
| `stage3_bgem3` (rejected — wash/regression) | **bge-m3** (8192 ctx, no prefix) | yes | yes | v2 | canonical-year family max | yes | yes | same as stage1 | `results_stage3_bgem3.json` — apples-to-apples embedding swap on top of stage1's full pipeline; RoA hit@6 60%→57%, hit@3 57%→50%, reverted (`EMBED_MODEL` back to `nomic-embed-text` in `src/llm.py`); `policies_bge-m3` collection left in Chroma, non-destructive |
| `stage4_context_pilot` (rejected — no improvement, one regression) | nomic-embed-text | yes | yes | v2 | canonical-year family max | yes | yes | same as stage1 | `results_stage4_context_pilot.json` — per-chunk LLM-generated situating context (`generate_chunk_context.py`), piloted on the 34 documents/580 chunks behind current misses only (full 843-doc/14,006-chunk scope estimated at ~20h, not run). Isolated to just the 22 pilot-scope turns: 0 new hits, 1 regression (glossary follow-up, rank 4→absent). Reverted (`rm -rf data/chunk_context_cache`, re-embedded) |

## Headline metrics (strict, 80 turns unless noted)

| Experiment | Policy hit@6 / MRR | RoA hit@6 / MRR | Overall hit@6 / MRR | Answer score |
|---|---|---|---|---|
| `baseline` | 77.5% / 0.63 | 22.5% / 0.06 | 50.0% / 0.35 | 3.64 |
| `fixed` | 77.5% / 0.71 | 35.0% / 0.15 | 56.3% / 0.43 | 3.66 |
| `mxbai` | 82.5% / 0.74 | 20.0% / 0.13 | 51.3% / 0.44 | 3.76 |
| `stage2` | 87.5% / 0.75 | 47.5% / 0.37 | 68.8% / 0.56 | 3.74 |
| `stage3` | 97.5% / 0.85 | 60.0% / 0.42 | 78.8% / 0.64 | 3.88 |
| `stage4` (rejected) | 87.5% / 0.73 | 47.5% / 0.36 | 67.5% / 0.54 | 3.62 |
| `postfix` (superseded) | 90.0% / 0.81 | 57.5% / 0.43 | 73.8% / 0.62 | 3.75 |
| `postfix2` (superseded — pre user-reported-bug fix) | 95.0% / 0.81 | 55.0% / 0.41 | 75.0% / 0.61 | 3.88 |
| `holdout_set2` (raw) | 70.0% / 0.60 | 52.5% / 0.38 | 61.3% / 0.49 | 3.81 |
| `holdout_set2` (confound-corrected*) | 93.3% / 0.80 | 52.5% / 0.38 | 70.0% / 0.56 | 3.79 |
| `postfix3` (rejected — prompt-wording confound) | 95.0% / 0.81 | 47.5% / 0.35 | 71.3% / 0.58 | 3.77 |
| `postfix4` (superseded) | 95.0% / 0.84 | 55.0% / 0.43 | 75.0% / 0.64 | 3.73 |
| `stage0_chunks` (superseded) | 95.0% / 0.83 | 62.0% / 0.45 | 79.0% / 0.64 | 3.81 |
| `stage1_rerank` (superseded) | 100.0% / 0.86 | 60.0% / 0.45 | 80.0% / 0.66 | 3.81 |
| `stage2_header_boost` (rejected) | 97.0% / 0.88 | 53.0% / 0.43 | 75.0% / 0.65 | 3.90 |
| `stage3_bgem3` (rejected) | 97.0% / 0.86 | 57.0% / 0.44 | 78.0% / 0.65 | 3.90 |
| `stage4_context_pilot` (rejected, then reverted) | 97.0% / 0.88 | 57.0% / 0.45 | 78.0% / 0.66 | 3.95 |
| `stage1_rerank` config, restored (superseded) | 100.0% / 0.86 | 60.0% / 0.45 | 80.0% / 0.66 | 3.81 |
| **`stage_colbert` (= current production)** | **100.0% / 0.91** | **70.0% / 0.45** | **85.0% / 0.68** | 3.89 |
| `stageA_facets` (rejected — hard filter, regex bug) | 100.0% / 0.89 | 47.5% / 0.33 | 73.8% / 0.61 | 3.73 |
| `stageA_facets_v2` (rejected — hard filter, regex fixed) | 100.0% / 0.92 | 57.5% / 0.41 | 78.8% / 0.66 | 3.74 |
| `stageA2_soft_facets` (rejected — soft RRF-fuse) | 100.0% / 0.89 | 60.0% / 0.40 | 80.0% / 0.65 | 3.86 |
| `stageG_pseudo_query` (rejected — net-zero wash) | 100.0% / 0.90 | 70.0% / 0.45 | 85.0% / 0.68 | 3.81 |
| `stageH_crag_verification` (rejected) | 100.0% / 0.86 | 65.0% / 0.42 | 82.5% / 0.64 | 2.88 |
| `stageD_splade` (rejected) | 100.0% / 0.91 | 65.0% / 0.45 | 82.5% / 0.68 | 3.90 |
| `stageE_embedding_ensemble` (rejected) | 100.0% / 0.89 | 57.5% / 0.40 | 78.8% / 0.65 | 3.80 |
| `stageI_multihop_decomposition` (rejected) | 100.0% / 0.89 | 62.5% / 0.40 | 81.2% / 0.65 | 3.88 |
| `j0b_wide_rerank_pool` (rejected) | 100.0% / 0.88 | 62.5% / 0.42 | 81.2% / 0.65 | 3.92 |
| `j2_identity_headers` (rejected) | 100.0% / 0.91 | 60.0% / 0.50 | 80.0% / 0.70 | 3.84 |
| `j3_doc_routing` (rejected) | 100.0% / 0.90 | 62.5% / 0.42 | 81.2% / 0.66 | 3.80 |
| `j4_user_turns_contextualizer` (rejected) | 100.0% / 0.93 | 67.5% / 0.41 | 83.8% / 0.67 | 3.88 |
| `set3_sibling_baseline` (measurement, 40 turns, programme-named questions) | — | 90.0% / 0.72 | 90.0% / 0.72 | 4.12 |
| **`j6_disclose_ambiguity` (= current production)** | **100.0% / 0.87** | **67.5% / 0.43** | **83.8% / 0.65** | **3.86** |
| `j7_keyphrase_prompt` (rejected) | 100.0% / 0.91 | 62.5% / 0.40 | 81.2% / 0.66 | 3.80 |

J-round note: `j6_disclose_ambiguity` changes NO retrieval code (it appends a source-naming
disclosure to answers when the top-6 is family-fragmented), so its hit@6 deltas vs
`stage_colbert` are a direct measurement of the eval's run-to-run noise floor (~1-2 turns,
Ollama nondeterminism via the contextualizer). Judge that band into every ±1-2-turn verdict in
this table. Full J-round narrative incl. the J1 identity-extraction asset
(`data/doc_identity/`), the evidence-sufficiency metric (RoA 87.5% vs strict 70%), and the
sibling-discriminating question set (`questions_set3_sibling.json`, 90% hit@6) is in report.md's
"Identity-first round" section.

(Stage F, weighted score fusion vs RRF, isn't in this table - it was decided from a fast
retrieval-only sweep, not a full 80-turn/judge-scored pass, since no weight config beat RRF
enough to warrant one; see `eval/sweep_fusion_weights.py` / `eval/sweep_fusion.log`.)

All eight rows above are detailed in report.md's "Try-everything round" section and the later
"Stage I: selective multi-hop query decomposition" section - each implemented behind an
off-by-default flag in `src/rag.py` (`FACET_PREFERENCE_ENABLED`, `WEIGHTED_FUSION_ENABLED`,
`PSEUDO_QUERY_ENABLED`, `CRAG_VERIFICATION_ENABLED`, `SPLADE_ENABLED`,
`EMBEDDING_ENSEMBLE_ENABLED`, `MULTIHOP_DECOMPOSITION_ENABLED`), all confirmed back
to `False` in the current production checkout.

Note on `stage4_context_pilot`'s topline row: only 34 of 843 in-scope documents were actually
touched, so the 80-turn aggregate (which looks almost flat/slightly positive on answer score) is
mostly noise from the 758 untouched documents. The real signal is in the isolated pilot-scope
comparison below (22 turns actually affected) - 0 improvements, 1 regression - which is what
drove the revert, not the aggregate row.

**`stage_colbert`: the day's best single result** (literature-motivated: see the "second literature
research round" section of report.md). Swapped the cross-encoder reranker for ColBERT-style late
interaction (`lightonai/GTE-ModernColBERT-v1` via the `pylate` library) over the same fused
candidate pool. RoA hit@6 60%->70%, the single largest jump since the original hybrid-retrieval
fix. Flip analysis: 5 turns gained, 1 lost, all in RoA, spread across 5 different document
families (not concentrated in one) - a genuine, well-distributed improvement, not a fluke.
Notable process point: manual spot-checks on the hardest known exemplar (4yr vs 5yr integrated
masters) looked unconvincing before running the eval (near-identical scores, wrong doc still on
top) - the full 80-turn aggregate told a different, better story. Same lesson as
`stage2_header_boost` in the opposite direction: trust the full eval over spot-checks, both when
spot-checks look promising and when they don't.

**Stage 1 (cross-encoder reranker):** first implementation scored the raw stored chunk text,
which does NOT include `chunk_header` (that's only prepended at embedding time, never stored) -
so the reranker was working with strictly less identity signal than the embedder had, and a
manual test (4yr vs 5yr integrated masters sibling confusion) confirmed it failed to rescue a
target document even though it was in its scoring pool at rank 16. Fixed by scoring
`chunk_header + chunk_text` pairs instead. Result: policy hit@6 now 100%, RoA hit@3/MRR improved,
RoA hit@6 roughly flat (62%->60%, within observed noise). Net positive, kept in production.
Required widening `FETCH_POOL_MULTIPLIER` 4->8 so there's real depth (48 candidates) for the
reranker to search, per the failure analysis finding relevant documents as deep as rank 60.

**Stage 2 (BM25 header boost) — rejected, see table above.** Originally scoped as literal
department-metadata-field matching (per the approved plan), but a fresh diagnostic found only
1 of 8 documents behind current misses has a department value that would plausibly appear in
natural query text (the rest are either untagged or tagged with an administrative unit name,
e.g. "Health and Social Care" instead of the actual programme name "Social Work") - low expected
coverage. Pivoted (user-approved) to boosting the BM25 weight of `chunk_header` instead, a more
general mechanism reusing existing infrastructure. Manual exemplar tests looked strong (CSEE and
"MA Social Work" queries both correctly surfaced their target document top-of-list), but the full
80-turn eval showed a net regression - boosting amplifies the header's shared generic words
("masters", "rules", "year") right along with the genuinely distinguishing ones, and apparently
the corpus has proportionally more of the former. Reverted (`HEADER_WEIGHT=1` in
`src/lexical.py`, no re-embed needed - BM25 indexes lazily from stored Chroma data at server
start, so this revert took effect on restart alone).

\* 5 of 30 holdout-set2 policy documents were selected at question-construction
time as superseded-year editions (2024-25/2023-24) whose questions don't
mention a year; `is_current` correctly excludes them from default retrieval
in favor of the current edition, which the strict metric scores as a miss.
This is a test-set construction flaw (fixed by excluding those 5 documents),
not a retrieval regression — see report.md's generalization section.

**postfix3 → postfix4**: a user manually testing the live app (not via the eval
harness) found a real bug — in a long, topic-switching conversation, the
follow-up query contextualizer sometimes echoed a completely unrelated
question from earlier in the transcript instead of rewriting the actual new
one, causing wrong-document retrieval. Fixed with `_is_faithful_rewrite()` in
`src/rag.py`: a deterministic content-word-overlap check that falls back to
the original question when the rewrite shares too little content with it.
The first fix attempt (`postfix3`) bundled this guard with a reworded
contextualize prompt ("if already self-contained, output unchanged"), which
regressed RoA hit@6 by 7.5pp — the reworded prompt made the model skip adding
disambiguating document/programme names to follow-ups that read as
grammatically self-contained but still needed that detail to distinguish
among near-identical RoA siblings. Isolated by diffing retrieved documents
between `postfix2`/`postfix3`/`postfix4`: reverting to the original prompt
wording while keeping only the guard (`postfix4`) restored RoA to `postfix2`
levels and still catches the original reported bug (re-verified against the
exact real conversation history that triggered it). Lesson: don't bundle a
prompt change with a structural fix in the same eval pass — they need
separate before/after measurements to attribute an effect correctly.

## Non-retrieval parameters also in play

- `N_RESULTS=6`, `FETCH_POOL_MULTIPLIER=4` (pool size 24) — unchanged across every pass.
- `RRF_K=60` — since `stage3`.
- Ollama `chat()` calls use no explicit `temperature`/`seed` (Ollama defaults) in
  every pass to date — a known source of run-to-run noise on top of whatever a
  code change contributes; not yet controlled for (see report.md's discussion
  with the user on separating signal from noise).
- `MAX_TURNS_BEFORE_SUMMARY=20`, `TURNS_TO_KEEP_AFTER_SUMMARY=10` in
  `src/memory.py` — unchanged; only the *mechanism* changed at `postfix`/`postfix2`
  (summarized_through watermark so each message is folded once instead of on
  every turn past the threshold).

## Reverting to a prior configuration

Every commit corresponds to a stable checkpoint (see `git log`). To fall back:
`git log --oneline` to find the commit, then `git diff <hash> HEAD -- src/
reembed.py run_ingest.py` to see exactly what changed before deciding whether
to revert individual files or the whole checkpoint. Re-run
`python reembed.py` afterward if the reverted code changes what gets embedded
or how `is_current`/`academic_year_norm` are computed — those are metadata-only
and safe to recompute without a full re-crawl.
