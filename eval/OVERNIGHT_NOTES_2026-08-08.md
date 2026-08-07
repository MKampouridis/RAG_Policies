# Overnight notes — 8 August 2026 (list run-through)

*Continues `OVERNIGHT_NOTES_2026-08-07.md`. Everything below is committed and pushed.*

---

## Headline

Worked through the open list. The most valuable result is **#4**: the eval's headline metric
overstates end-to-end usefulness by up to **8.7 points**, and the gap is now measured and decomposed.
The second is that the **multi-entity fix works on the real failing question** — pending a collateral
check that was still running when I wrote this.

---

## #4 — hit@6 blind spot: MEASURED and DECOMPOSED ✅

hit@6 compares document URLs, so it asks "was the right DOCUMENT retrieved", not "were the
answer-bearing FACTS available". A turn can score a clean hit while the user gets a wrong answer.

**Size** (computed from already-committed results — no new runs, no cost):

| | main set | set 2 |
|---|---|---|
| hit@6 TRUE but answer judged ≤2 | **8.8%** | **3.8%** |
| headline hit@6 vs actually-useful | **86.2% → 77.5%** | **67.5% → 63.7%** |

**Cause split** (`eval/chunk_blindspot.py`, new): of the 10 hit-but-failed turns —

- **60% GENERATOR** — the facts *were* in the retrieved context; synthesis failed anyway.
- **20% CHUNK_MISS** — facts are in the gold document but in a chunk never retrieved. Right
  document, wrong chunk. No generator can fix these.
- **20% WEAK_TEST** — keyphrases appear nowhere in the gold document, so the item is unsatisfiable.

**Two things follow.** The biggest slice is *generation*, which matches the live evidence (gemma3
gave 2 of 6 independent-chair criteria where Sonnet gave all 6), so Round 5's "retrieval frontier is
closed" survives intact. And **two eval items are defective** — their keyphrases aren't in their own
gold documents, so they've been silently depressing scores. Worth rewriting before they're counted
again: `independent-chairs-policy.pdf` [primary] and
`roa-ug-integrated-masters-4yr-year-1.pdf` [follow_up].

**Recommendation:** report hit@6 *and* the useful-answer rate (hit AND judged ≥3). The second is the
honest end-to-end number.

## #7 — Variance-gated disclosure: BUILT and ON ✅

The J6 caveat ("rules often differ by programme — tell me which you mean") fired on every
fragmented-pool turn, including questions where the answer is identical corpus-wide. Merit is 60
everywhere, so on that question the caveat is pure noise that trains people to ignore it.

Now gated on the measured variance map: suppressed only when the question names a parameter measured
UNIFORM (Merit, Distinction, first-class) and names no varying one. Conservative by construction — an
unrecognised question keeps the caveat, a mixed question ("Merit *and* the pass mark?") keeps it, and
a missing variance map disables the gate rather than failing open. Unit-tested on 7 cases, all pass.

Note the eval **cannot** validate this either way: the disclosure is appended text no metric scores.
It's a UX judgement resting on measured data, not an eval result. `VARIANCE_GATED_DISCLOSURE = False`
reverts it.

## #2 — Multi-entity: BUILT, default OFF pending the collateral check ⏳

On your actual failing question ("accredited programmes offered by CSEE, MSAS, Psychology, HSC,
SRES, and Life Sciences"):

- **OFF:** *"I am sorry, but the provided context does not list the accredited programs…"* — and
  gives nothing, withholding even the CSEE data it had.
- **ON:** names the five it lacks explicitly, **and** lists the actual CSEE programmes.

Strictly better on the real case. **Left OFF anyway**, because this project has exactly one
precedent for base-prompt rules — `INLINE_CITATIONS` — and it regressed groundedness by 11 points.
A collateral check on ordinary single-entity questions was running when I wrote this; result at the
top of my next message. Flip `MULTI_ENTITY_COVERAGE = True` if it comes back clean.

**The other half of this failure is not prompt-fixable:** `N_RESULTS = 6` is a hard ceiling, so a
six-school question cannot retrieve enough chunks to answer six schools. That needs per-entity
retrieval or a widened k for this question shape — not attempted.

## #5 — Partner-institution test coverage: SCOPED ✅

Tonight's void eval happened partly because the question set contains **zero** partner-institution
ambiguity. Quantified now: **77 of 244 current documents (32%) are partner editions**, yet none ever
appears in an eval top-6 — the question set materially under-represents a third of the corpus.

A targeted set is feasible. Three programmes have genuine home/partner competition (shared J1 alias):

| programme | partner edition | home edition |
|---|---|---|
| Periodontology | `msc-periodontology-science-(alexandria)` | `mscperiodontology_25.pdf` |
| Sports Therapy | `portobello-variations-year-2/3` | `pt_msc_sports_therapy_24.pdf` |
| UG rules of assessment | `roa-ug-colchester-institute-year-…` | `roa-ug-4yr-year-3-rules.pdf` |

That's enough to test the partner demotion properly — and the same gap Phase 4 left open.

## #9 — Data hygiene: CLEAN ✅

20,436 chunks / 1,169 documents indexed against 1,189 in the manifest. The 20-document difference is
**entirely the hub/index pages** (`policies`, `rules-of-assessment`, `roa-masters`…) that Round 3
deliberately excluded — not drift. Zero orphans in either direction. No `run_ingest.py` has run, so
the alias map needs no re-audit.

---

## Not done, and why

- **#6 D3 clarification live-judgment** — genuinely needs you. It's a product call about how often
  the assistant should ask versus guess, judged on real conversations; I can't substitute for that.
- **#8 value-level metric as headline** — the cloud judge is now wired, so re-judging the
  definitional-claim turns phi4 was lenient on is ready to run. Held back because a judge swap moves
  the scale itself, so it needs your call on how to present numbers that aren't comparable to the
  existing ledger.
- **#10 credits** — infrastructure done and used tonight (the collateral check is cloud-judged).
  Spend so far is a few cents.
- **Full 80-turn A/B of the multi-entity rule** — ~90 minutes of contended RAM; the 6-question
  collateral check is the honest interim.

---

# Final results (written ~01:30)

## #2 multi-entity — now ENABLED ✅

The collateral check finished clean: **6 ordinary single-entity questions, mean judge score 5.00 with
the rule and 5.00 without.** Combined with the clear win on your real failing question, I enabled
`MULTI_ENTITY_COVERAGE`. The server has been restarted so it is live.

Honest limit: those 6 questions all scored 5/5, so the sample is saturated at the judge's ceiling — it
shows *no obvious harm*, not proven safety. The 80-turn A/B is still the proper validation.

## A bigger finding, and a correction I have to make

Chasing the two "WEAK_TEST" turns from #4 turned up something more serious than a bad test item.

**`independent-chairs-policy.pdf` [primary] has a factually wrong gold answer.** It asks how many
staff a department with *more than 149* PGR students must nominate. The policy table says
`1-99 → 2`, `100-149 → 3`, `150+ → 4`, so the answer is **4**; the expected answer says **3**. That
item has been penalising correct behaviour.

**`roa-ug-integrated-masters-4yr-year-1.pdf` — text is missing from the index.** The source says
*"…they must withdraw from the University in any of the following situations: — Where the Year Mark
is below 20…"*. The indexed chunk reads *"…they — Where a student was absent fr…"*. Text is dropped
**mid-sentence inside the chunk body**, not at a boundary. The words `withdraw` and `maximum period`
appear in **none** of that document's 18 chunks. That progression rule is unretrievable — no
reranker or generator can recover text that was never indexed.

**The correction:** I ran a sentence-level scan suggesting **37% of sampled documents** had missing
text. **That number is wrong and I've retracted it in `report.md`.** Verifying six flagged documents
with bare keywords showed **five were false positives** — my scan had spliced page furniture into
"sentences" that legitimately never appear in the cleaned chunks. Only one case survives
verification: **1 confirmed out of 38 sampled.**

So the honest position: **one verified content-loss defect, and an open question about how common it
is.** I did not establish a rate. Please don't quote 37% anywhere.

**Why it matters enough to chase:** if it turns out to be common, it is a better explanation for
residual retrieval misses than anything in the Round 5 retrieval bake-off — reranking cannot retrieve
what was never indexed. The proper next step is an audit comparing cached source to indexed chunks
using the *ingest's own* cleaning rather than ad-hoc normalisation, which is what tripped me up.

## Where things stand

Live: local gemma3, launchd-managed with auto-restart, both tonight's retrieval fixes plus the two
new prompt/disclosure rules active. Eval runs local and deterministic; the cloud judge is one env var
away. Clean tree, everything pushed.

**Top of the list now:** the content-loss audit. It is cheap to run, and it either clears the corpus
or reframes a chunk of the retrieval work.
