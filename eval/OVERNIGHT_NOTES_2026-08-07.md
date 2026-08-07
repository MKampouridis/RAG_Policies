# Overnight notes — 7 August 2026

*Plain English, for a morning read. Technical detail is in `eval/report.md` and
`eval/FEEDBACK_FINDINGS.md`. Everything described as "committed" is pushed to origin/main.*

---

## The one-liner

Your real user feedback turned out to be far more useful than another eval round: it surfaced
**four concrete bugs the offline test suite cannot see**, one of which I fixed and verified, and one
of which is a **blind spot in the hit@6 metric itself** — the system can score a perfect "hit" while
giving the user a half-answer.

---

## What's done and pushed (3 commits)

### 1. Variance map (`f4f6e3e`)
The static lookup table for the variance-gated disclosure idea. For 8 common rules-of-assessment
parameters, is the value the same across all programmes or does it genuinely vary?

- **Uniform (3):** Merit threshold (60), Distinction threshold (70), First-class (70). Retrieving the
  "wrong" programme's document still gives the right answer — no disclosure needed.
- **Varying (5):** pass mark, condonement, reassessment cap, credits for the award, further attempts.
  These are where getting the right document actually matters.
- One data-quality flag: "further attempts" shows a value of "60", which is almost certainly a
  misextraction (you can't have 60 resit attempts). Doesn't change the verdict, but don't quote it.

### 2. Feedback analysis (`9f6133c`) — the valuable one
First proper look at `data/feedback.jsonl`: **32 ratings, 15 up / 17 down = 47% satisfaction.**
That's much lower than the eval suite's ~90%, and the gap is the point — real use is harder than a
fixed question set. Full write-up in `eval/FEEDBACK_FINDINGS.md`.

**Not a single complaint was about a made-up number.** Every one was about *scope* — wrong topic
carried over, missing entities, missing list items, wrong institution. The groundedness work you did
in rounds 4–6 has held; the next frontier is somewhere else entirely.

### 3. Contextualizer topic-switch fix (`56649bd`) — a real bug, fixed and verified
The biggest cluster of thumbs-down (~6 of 17). When you changed topic, the system assumed you were
still on the old one and bolted the old topic onto your new question. Two of your own examples:

- You asked about MSc Artificial Intelligence, then asked about **PhD duration**. The system rewrote
  your question with **five unrelated MSc programme names** attached.
- You asked about Professional Doctorates, then asked about **PhD** duration — it answered about
  Professional Doctorates again.

**Cause:** two gaps. The existing safety check only verified the rewrite *kept* your words — it never
checked whether it *added* unrelated ones. And "phd" is deliberately excluded from the list of
distinctive programme names (it's too common across departments), which meant a question saying "phd"
wasn't recognised as already naming its own topic, so the old topic got re-attached.

**Fixed and verified:** both cases now come back clean (case 2 stable across 4 repeat runs). Three
legitimate follow-up patterns re-tested for regressions — all fine. Full 30-turn multi-turn probe:
**zero regressions, one turn improved (28/30 → 29/30).**

---

## New finding: hit@6 has a blind spot (no fix yet — needs your call)

Chasing the "independent chair" complaints (three thumbs-down in one conversation) turned up
something structural.

The policy document lists **six** circumstances requiring an independent chair. Every answer surfaced
only one or two. I checked whether the document was being chunked badly — **it isn't**, one chunk
cleanly contains all six. So I replayed all three questions to see what actually got retrieved:

| Question | Right document? | Right *chunk*? | So what went wrong |
|---|---|---|---|
| "In which cases is an independent chair required…" | Yes | **No** (got chunks 1 and 6, not 2) | Retrieval failure at chunk level |
| "But what are the circumstances…" | Yes | **Yes** | Generator had all six, reported two |
| "Is an independent chair required for a reexamination…" | Yes | **Yes** | Same — had the list, under-reported |

Two different causes wearing the same costume: one genuine retrieval miss, two generator
under-enumeration failures.

**The important part:** `hit@6` is scored by comparing **document URLs** (`score_retrieval` in
`eval/run_eval.py`). All three of these score as **HITS** — the right document was retrieved every
time. But the user got an incomplete answer in all three cases. *The metric cannot see this failure
class at all.* Every "right document, wrong chunk" or "had the facts, didn't list them" failure is
invisible to your entire eval suite.

This may partly explain the gap between 90%+ eval scores and 47% real-user satisfaction.

**Not fixed** — it's a measurement question (do you want a chunk-level or claim-completeness metric?)
plus possibly a generator prompt change ("enumerate every item in the list you find"). Your call.

---

## Running while you slept

A full 80-turn eval is regression-testing the **partner-institution fix** (see below). It was ~14/40
at the time of writing. Results and a verdict will be at the top of my next message.

**The partner-institution fix itself** (not yet committed, pending that eval): three thumbs-down were
partner documents (Kaplan, Tavistock) outranking Essex's own for questions that never mentioned a
partner. Your "exit awards for MSc AI" question sourced its answer from a **Kaplan** document while
four CSEE documents sat right there in the same results. There was an old mechanism for this
(`HOME_INSTITUTION_TIEBREAK`, built in Phase 4) but it can't fire here — it needs the two documents to
share an alias, and the Kaplan document's identity record is completely empty. So I added a simpler
rule: partner documents sink below Essex ones whenever both appear. Verified on your exact case —
Kaplan went from rank 1–2 to rank 5–6, all four CSEE documents now rank above it.

---

## Still open (not started)

- **Multi-entity questions** (~5 of 17 thumbs-down). Ask about six schools, get one. "I've asked
  information on 6 schools, which I've explicitly listed, but I'm still only getting info on one."

  I dug into this at the code level (read-only — didn't want to run anything heavy while the eval
  had the RAM). Two things worth knowing before you pick an approach:

  **(a) There's a structural ceiling: `N_RESULTS = 6`.** A question naming six schools can retrieve
  at most six chunks total — roughly one per school if perfectly distributed, and in practice they
  all come from whichever school ranks best. Even flawless ranking cannot answer that question under
  the current budget. Any fix has to widen k for this question shape.

  **(b) There is already a decomposition mechanism, and it was falsified — but for a different
  case.** `MULTIHOP_DECOMPOSITION_ENABLED` (Stage I) exists and is off: it regressed RoA hit@6
  70%→62.5%. **I don't think that result transfers here**, and the distinction matters. Stage I
  decomposed on a *guess* — the trigger was "the pool looks fragmented, so maybe the user meant one
  of these documents" — and it lost because a wrong hypothesis diluted the pool. A multi-entity
  question needs no guessing: the user *literally listed* the six schools. Decomposing on what was
  explicitly written is a different mechanism from decomposing on a hypothesis about what was meant.

  So the options, cheapest first: (1) have the generator explicitly state which of the N requested
  items it found nothing for — turns a silent wrong answer into an honest partial one, no retrieval
  change; (2) detect explicitly-enumerated entities and retrieve per-entity with a widened k. I'd
  do (1) first regardless — it's cheap and it fixes the *trust* problem even if coverage stays hard.
- The hit@6 blind spot above.

---

## On the $20 of Claude credits

Short version: **use it as a neutral judge, not as a generator.** Your own report documents a
**24-percentage-point swing** caused purely by which model graded the results — that's the weakest
link in the whole measurement chain, and it's currently a small local model (phi4) making subtle
groundedness calls.

Rough costs for one full 80-turn judging run: **Haiku 4.5 ~$0.40, Sonnet 5 ~$0.80, Opus 5 ~$2.00**
(Sonnet 5 is on introductory pricing until 31 August, so it's ~33% cheaper this month). The Batch API
halves all of those, and it's a perfect fit — offline eval has no latency requirement.

**Recommendation: Sonnet 5 via the Batch API, ~$0.40 per full re-judge → roughly 50 complete eval
runs for $20.** Enough that you don't need to ration it. Best first spends: (1) re-judge the close
calls phi4 decided, especially the value-level sufficiency run where leniency was flagged; (2) judge
the D3 ask-vs-guess tradeoff on real conversations — the open item that can't be validated by string
matching; (3) size the feedback failure modes properly instead of by my manual read of 17 comments.

Not worth it: a cloud generator for production. Your report already settled that — daily caps make it
unfit, and gemma3 is validated.
