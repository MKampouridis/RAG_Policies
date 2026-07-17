# Experiment Log

Parameter and result tracking for every eval pass run against this system, so a
future regression can be traced to an exact configuration and, if needed,
reverted via git (`git log --oneline`, `git checkout <hash> -- src/ reembed.py
run_ingest.py`). Prose narrative for these same experiments is in `report.md`;
this file is the fast-scan parameter reference.

All passes: `qwen2.5:7b-instruct` chat model, `CHUNK_WORDS=300`,
`CHUNK_OVERLAP_WORDS=50` (never varied — see report.md's rejected-proposals
table for why), same 40-question original set unless noted.

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
| `holdout_set2` (in progress) | nomic-embed-text | yes | yes | v2 | canonical-year family max | yes | no | soft preference | `results_holdout_set2.json`, question set: `questions_set2.json` |

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
| `postfix2` (production) | 95.0% / 0.81 | 55.0% / 0.41 | 75.0% / 0.61 | 3.88 |
| `holdout_set2` | _pending_ | _pending_ | _pending_ | _pending_ |

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
