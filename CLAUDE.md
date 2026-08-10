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
- **Stop the production server during evals** (`launchctl unload ~/Library/LaunchAgents/com.mkampo.ragpolicies.plist`
  — a plain `kill` respawns it). Each server instance costs ~5GB on a 16GB machine.
- Runs over ~45 minutes should be launched **detached** (`nohup`), since harness-tracked background
  tasks get killed around 60–80 minutes. Results are written incrementally and `run_eval.py`
  resumes; `RAG_EVAL_NO_RESUME=1` forces a clean run after any system change.

## Data

`clean_text` in `src/ingest.py` once deleted repeated *policy clauses* as "page furniture", making
some rules unanswerable from the whole corpus. `eval/stale_index_audit.py` detects recurrence — run
it after any ingest or cleaning change. Re-embedding does **not** fix cleaning bugs, because
`reembed.py` runs the same `clean_text`.

After `run_ingest.py`: re-run `python audit_family_aliases.py` and review new rename-split aliases
in `src/docid.py`.

## Conventions

- Commit directly to `main`; no PR workflow.
- `eval/report.md` is the ledger — record falsifications as carefully as successes, including
  retractions.
- Don't delete files without asking.
