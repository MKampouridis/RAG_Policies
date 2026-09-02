# Working notes for Claude on RAG_Policies

A retrieval-augmented assistant over University of Essex policy and rules-of-assessment documents.
The value of this project is its measurement discipline — most ideas tried here have been *falsified*,
and the record of what didn't work is worth more than the code. Everything below exists because
ignoring it has already cost something.

## Evidence discipline

**State causal explanations as hypotheses until measured.** If a check is cheap, run it before
asserting. Distinguish "the data shows" from "this is likely" — and say which one you mean.

This is the rule most often broken here. Recent examples, all wrong and all cheap to check:
"37% of documents have missing content" (a scan artifact — 5 of 6 spot-checks were false positives);
"set 2 is RoA-heavy" (both sets are exactly 20 policy + 20 RoA); "the slowdown is from paging" (it
was document type — RoA questions take ~45% longer in *both* sets).

A retraction rate that tracks how often the user pushes back means the *unchallenged* claims are
also unverified. Verify first instead.

## Don't ship unmeasured mechanisms ON

New retrieval/prompt mechanisms default to `False` and are enabled only after an eval justifies it.
See `INLINE_CITATIONS`, `HOME_INSTITUTION_TIEBREAK_ENABLED`, `MULTIHOP_DECOMPOSITION_ENABLED`,
`CLARIFY_UNDERSPECIFIED_ENABLED` in `src/rag.py` — all off, each with its falsification recorded.

`_has_extraneous_family` was shipped enabled on the strength of two hand-checked cases and cost
**-8.8 points of follow-up hit@6** before a full eval caught it. Small, flag-gated, measured.

## The answer is a user-facing surface, not a debug log

Answers must read as if written by someone who knows the policies — never as a program
describing its own retrieval. Production answers said *"the context you've provided across both
turns"* and offered *"if you have excerpts, please share them"*: the user supplied nothing, the
retriever did, so this reads as either a mistake or a request they cannot act on. `USER_FACING_LANGUAGE`
in `src/rag.py` now forbids that vocabulary — say "the policies I can see don't cover X", never
"the context does not contain X", and never ask the user to paste or share documents.

**The general point is about what evals cannot see.** This defect cost trust, not accuracy. Every
metric in the ledger — hit@6, span coverage, judge score, keyphrase coverage — would score these
answers identically before and after, because the facts were right both times. No amount of
measurement would have surfaced it; only reading the output as a user would. When changing
anything that shapes prose, read whole answers rather than only the scores.

Expect partial compliance and measure it like anything else: the rule cut plumbing-leaking answers
from 4/4 to 2/4, not to zero — "the excerpts I can see" survives an explicit instruction not to say
it. Report the residual rather than the headline.

## Three kinds of change, three standards of proof

Applying one bar to everything is why `+0.05, below the noise floor` got shipped
with the words "confirmed no harm". External review (2026-08-11) proposed this split
and it matches what the ledger already does implicitly:

- **Broad** (embedder, reranker, fusion, chunking, pool size) — must show a
  detectable aggregate improvement. `eval/compare.py` for paired diffs and a
  bootstrap CI; a CI that straddles zero means *unresolved*, not *no effect*.
- **Targeted** (partner exclusion, multi-entity, adjacent chunks) — needs a named
  failure case, a mechanism that addresses it, a targeted probe, no regression on a
  broad control, and a **quantified blast radius**. `_has_extraneous_family` had no
  denominator and cost -8.8 points; adjacent-chunk expansion measured its ceiling
  first (5% of turns) and narrowed from 97% to 81% touch. Report *turns changed*,
  never a diluted mean.
- **Wording/UX** (`USER_FACING_LANGUAGE`, detail level, source presentation) — judged
  by reading whole answers. Trying to prove these statistically is how the plumbing
  leak survived four metrics unchanged.

Two corollaries. **Feedback authors test cases; it never tunes the pipeline directly** —
a thumbs-down is a hypothesis, and four of this round's mechanisms came from replaying
one, but each was then measured independently. And **tag each mechanism "wrong for
anyone" vs "right for me"**: a false denial about a specified policy is wrong for
everyone; preferring Essex documents over partner ones is a product choice for one
audience, and set 6 measured exactly what that choice costs.

## A safety net's coverage is what it EXECUTES, not how many cases it runs

Before trusting a check, state which code path it does **not** reach.

A refactor left a constant undefined and **every answer returned 503**. The 161-query
retrieval fingerprint passed. The 118-turn canary passed. Both exercise *retrieval*; the constant
is used during *answer assembly* — so two green safety nets, costing minutes, coexisted with a
system that could not answer a single question. `pyflakes` finds it in one second. Separately, a
`const` used 760 lines before its declaration aborted the whole client script and blanked the page;
every check run that day — CSS balanced, element present in the HTML, correct documents on 217
cases — was true, and none tested whether the page *executes*.

161 queries and 118 turns *sound* thorough. Both exercised one path. Breadth of cases is not
breadth of coverage.

**Run `python verify.py`** (add `--static` for no server) before believing a change is safe:
pyflakes → `import src.app` → JS parses → no top-level use-before-declaration → one live POST.
Cheapest and broadest first; the fingerprint is expensive and narrow, so it runs separately.

Each step was tested against the bug it claims to catch — the dead-zone check was written twice,
because the first version passed the actual broken file (it skipped function bodies, and the bug
reached `settings` through a top-level *call*). **A check you have not run against a known failure
is a check with unknown coverage.** Not covered: runtime errors other than top-level dead zones —
a browser smoke test would be needed, and Chrome headless does not run in this environment.

## Measuring changes

- **Validate on the metric the change can actually move.** The contextualizer only runs on
  follow-ups, so a topic-switch-heavy probe cannot detect a follow-up regression. Split by
  `primary` vs `follow_up` — primary is a free control for anything touching query rewriting.
- **Report hit@6 AND the useful-answer rate** (hit AND judge >= 3). hit@6 compares document URLs, so
  it cannot see "right document, wrong chunk" or "had the facts, didn't use them" — measured at
  8.7 points on the main set.
- **Change one thing at a time.** Five bundled changes made attribution impossible and took a
  turn-type split plus a rejection log to untangle.
- **`eval/retrieval_replay.py`** scores hit@6 by replaying stored queries — no generation, no
  judging, minutes not hours. Use it for anything retrieval-only.

## Spend repeats where they change the decision

Not every arm needs running twice. Escalate:

- **Exploring** — one run per arm. Enough to discard the obviously-dead.
- **Looks promising** — repeat the same arm on the same questions. This is where
  "run one arm twice" belongs; two identical configs scored 4.05 and 3.85 (Round 8k).
- **Survives that** — run it on the 151-question regression set (`eval/questions_regression.json`),
  not the 10-question set it was developed against.
- **Headline claim** — repeat BOTH arms, and report paired diffs with
  `eval/compare.py`, never two means.

Before any of it, ask **how many turns the mechanism can even touch**. The
multi-entity partner leak was invisible to three instruments because 0 of 160 replay
turns name two departments (Round 16); `_adjacent_chunks` had a measured ceiling of
~8 turns in 160. A run on a set containing no applicable case reports "no change" for
a defect that is really there.

And check **how often it fires in real traffic** before optimising it: batched
reranking was correct, identical in output, and abandoned because it bought 11% of a
stage on ~1 in 10 questions (Round 31).

## Know the noise floor before believing a delta

Two runs of the **same** configuration on a 10-question set scored 4.05 and 3.85 (Round 8k).
Cloud generation cannot be temperature-pinned, so **on a 10-question / 20-turn set any delta
below ~0.20 is uninterpretable**. Three findings recorded the same day sat at or below it,
including one reported as an effect and one whose direction reversed on retest.

Before believing a small cloud-generator delta, **run one arm twice**. The repeat costs exactly
what the comparison cost and is the only thing separating a real effect from a reroll. Larger
sets lower the floor; local+deterministic runs remove it, which is why the ledger's baselines
are local.

Related: when testing whether something matters, **run the spoiler first** — the arm that should
break it if the mechanism is real. Chunk order was investigated by running `reversed` (worst
chunk first) before `grouped`, and it falsified the hypothesis outright; a `rank` vs `grouped`
comparison would have produced a small difference and been read as confirmation.

## Eval hygiene

- **Use `./eval_session.sh <name> [questions.json ...]`** — it stops production, starts a local
  deterministic server on :8001, runs the sets, and restores production via an EXIT trap even on
  failure. It refuses to run if anything already holds :8001, because a stale server answers the
  health check and silently serves the eval with *its* configuration (caught exactly that in
  testing: a leftover cloud-generator server served a "local" run).
- The eval drives a **server over HTTP**, so one server = one configuration. Production may use
  cloud models; **the eval server must stay local and deterministic** (`RAG_DETERMINISTIC=1` on
  *both* server and script) — cloud calls cannot be temperature-pinned, and every ledger number was
  measured locally. Point the eval at its own instance with `RAG_API_BASE`.
- **Never judge close calls with a candidate model.** Same-family self-preference swung one result
  by 24 points. `phi4` is the neutral cross-family judge; `JUDGE_PROVIDER=anthropic` gives a
  frontier judge. Never mix judges within one comparison — the judge alone moves the threshold
  metric by ~9 points.
- **RAM, as of 2026-09-02 — measure `phys_footprint`, NOT `rss`.** The figures in this
  table were wrong by ~40% for a year because they were resident-size readings, and on
  macOS RSS does not mean what it looks like: the memory compressor pages out an idle
  process, so the SAME server measured minutes apart read 22MB, 198MB, 801MB and 271MB.
  A 2-day-idle process read 11MB. None of those is the memory it needs. `footprint -p
  <pid>` reports the physical footprint, which is the number to provision against —
  and a Linux host has no such compressor, so RSS there behaves differently again.

  | | phys_footprint |
  |---|---|
  | server, warm, under use | **~3.9GB** (peak ~4.2GB) |
  | Ollama, embed model loaded | ~70MB charged, ~325MB resident |
  | **production total** | **~4GB** |

  Measured twice on separate days (3948MB/4219MB peak, then 3863MB/4177MB) so the
  figure is corroborated, not a single reading.

  Two things the old table hid. **~2.0GB of that footprint is in the "graphics"
  category**: nothing in `src/rerank.py` or `src/colbert_index.py` ever sets a device,
  pylate auto-selects, and MPS is available — so the reranker runs on the Apple GPU
  against unified memory. On a CPU-only host that work moves to the CPU; what it costs
  there is UNMEASURED. And **Ollama is cheaper than stated** because llama.cpp mmaps
  the model: `llama-server` shows 303MB resident but only 42MB charged, the other
  278MB being clean file-backed pages the kernel can reclaim.

  The conclusion the old figure supported still holds on the corrected one:
  retrieval-only eval work does not need production stopped (~4GB + an eval process on
  a 16GB machine), and the machine is no longer the constraint it was — the older notes
  said ~5GB per instance and ~13GB for three local models, both predating the cloud
  move. But it holds with less headroom than 1.5-2.6GB implied.

  **Local EVAL is a different profile.** `gemma3:12b` (8.1GB) and a judge (`phi4` 9.1GB
  or `qwen2.5:14b` 9.0GB) will not both fit in 16GB — run generation and judging as
  SEPARATE passes (`eval/rejudge.py` re-scores a stored results file), or the models
  evict each other every turn.
- **Stop the production server during LOCAL-GENERATION evals**
  (`launchctl unload ~/Library/LaunchAgents/com.mkampo.ragpolicies.plist` — a plain
  `kill` respawns it), since those load an 8GB+ model.
- Runs over ~45 minutes should be launched **detached** (`nohup`), since harness-tracked background
  tasks get killed around 60–80 minutes. Results are written incrementally and `run_eval.py`
  resumes; `RAG_EVAL_NO_RESUME=1` forces a clean run after any system change.

## Who this is for (a product decision, not a retrieval one)

**Audience: University of Essex staff asking about Essex programmes.** That single
line resolves arguments the retrieval metrics cannot. `PARTNER_EXCLUDE_WHEN_UNNAMED`
drops partner-edition documents when the query names no partner — measured cost, twice:
a question naming a partner PROGRAMME but not its institution loses the document that
answers it (set 6: NAMED 4/4, UNNAMED 0/2, HOME 2/2, unchanged after the gate fix).

That is the right trade *for this audience* and the wrong one for a partner-college
administrator. Three softening options exist (demote-and-cap, soft boost, a toggle) and
all are recorded as unmotivated while the audience holds. **If the audience widens,
revisit that decision first** — it is a product assumption wearing retrieval clothing,
and it will not announce itself.

## Data

`clean_text` in `src/ingest.py` once deleted repeated *policy clauses* as "page furniture", making
some rules unanswerable from the whole corpus. `eval/stale_index_audit.py` detects recurrence — run
it after any ingest or cleaning change. Re-embedding does **not** fix cleaning bugs, because
`reembed.py` runs the same `clean_text`.

After `run_ingest.py`: re-run `python audit_family_aliases.py` and review new rename-split aliases
in `src/docid.py`, run `python eval/colbert_index_drift.py`, and run
`python eval/check_benchmark_stamp.py` — the EVAL SET is also a derived artifact of the
corpus, and after a re-ingest its gold documents can be superseded, which scores correct
retrieval as a miss (9 of 148 items, round 8). Ingest updates Chroma and leaves
the ColBERT index alone, so the reranker's embedding cache silently describes the OLD text - it was
three weeks stale before anyone noticed, cost 5 turns of hit@6, and was found only because a
latency experiment happened to A/B the flag (Round 27). The cache now defaults OFF; the drift
check exists so re-enabling it is a decision made on evidence.

## Conventions

- Commit directly to `main`; no PR workflow.
- `eval/report.md` is the ledger — record falsifications as carefully as successes, including
  retractions.
- Don't delete files without asking.
