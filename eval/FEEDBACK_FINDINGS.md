# Real-user feedback findings (2026-08-07)

*Plain English. Source: `data/feedback.jsonl` (32 ratings, 2026-07-23 to 2026-08-07, 27 conversations /
146 messages in `data/chat.db`) and `python feedback_report.py`. Everything below is read from real
usage on localhost, not the offline eval harness — it surfaces failure modes the eval questions don't
probe (multi-turn topic switches, multi-entity questions, partner-institution filtering).*

## The headline number

**15 up / 17 down = 47% satisfaction** on rated turns. Lower than the eval-suite hit rates (~90%+),
which makes sense: real usage is adversarial in ways a fixed question set isn't — people ask follow-ups
that change topic, ask about several things in one question, and probe partner-institution edge cases.

## Four real failure patterns, ranked by frequency

### 1. Contextualizer wrongly treats a topic-*switch* as a follow-up (~6 of 17 downs — the biggest cluster)

The clearest, most fixable bug in the log. Two smoking-gun examples where `retrieval_query` (what the
contextualizer rewrote the question into) is visible in the log:

- User asked about **MSc Artificial Intelligence** exit awards, then asked *"what is the minimum and
  maximum duration for a phd?"* — an unrelated new topic. The contextualizer rewrote it to: *"What is
  the minimum and maximum duration for a PhD? (MSc Applications of Artificial Intelligence, MSc
  Financial Technology (Computer Science), MSc Computing (non-admitting exit route), ...)"* — it bolted
  five programme names from the previous turn onto a question that never asked about them, sending
  retrieval down the wrong path.
- User asked about **Professional Doctorate** duration, then asked *"duration of phd"* — the system
  answered about Professional Doctorates again. Comment: *"I asked about phd, and it incorrectly
  assumed I was still talking about prof doctorate."* PhD and Professional Doctorate are different
  awards; the contextualizer conflated them.
- Same shape twice more: switching F/T↔P/T PhD question got answered with irrelevant supervisory-panel
  detail carried from context; "duration of phd" after professional doctorate.

**Diagnosis:** the contextualizer is too eager to fold prior-turn entities into a new query, with no
topic-boundary check. This is a known risk area (memory already flags Stage H follow-up regression from
the gemma3 switch) but this is a *different* failure — not "abstention breaks follow-up," but "the
rewrite injects the wrong entity when the topic actually changed."

**Lever:** tighten the contextualizer prompt/logic to detect topic discontinuity (e.g. a new query
naming a different award type, or sharing little lexical overlap with the prior turn) and rewrite it
standalone instead of appending prior context.

### 2. Multi-entity questions get answered for only one entity (~5 of 17 downs, overlaps with #1)

When a question names several things at once — "accredited programmes offered by CSEE, MSAS,
Psychology, HSC, SRES, and Life Sciences," "PGT programmes above 180 credits," "names and departments of
those programmes" — the answer consistently covers only one entity (usually CSEE, the first-embedded
match) and drops the rest, even on direct follow-up ("Follow up sucks... I'm still only getting info on
one school").

**Diagnosis:** top-6 retrieval can't cover six different schools' documents in one pass, and there's no
query-decomposition step for enumerable/multi-entity questions.

**Lever:** detect multi-entity questions (comma-separated lists, "all schools/programmes", "which
programmes...") and either retrieve per-entity (loop + merge) or explicitly widen k for this question
shape. Cheaper first step: have the generator explicitly say which of the N requested entities it does
NOT have information on, rather than silently answering one and ignoring the rest.

### 3. Partner-institution documents surface for Essex-only questions (3 of 17 downs)

- "Exit awards for MSc AI" → sourced from a **Kaplan** (partner) document instead of Essex's own CSEE
  programme docs. Comment: *"you should have based your answer on CSEE's programme."*
- "Longest-duration UG degree" → partner doc (`sak-principles...`) in the mix. *"Answers need to focus
  on Essex programmes, not partners."*
- Independent-chair re-examination question → sources included a **Tavistock** partner document and an
  older edition. Tagged `outdated`.

**Diagnosis:** nothing in retrieval currently down-ranks or excludes partner-institution documents by
default; a generic query with no partner named can still surface them ahead of the direct Essex
document.

**Lever:** default-exclude (or heavily down-rank) partner-institution-tagged documents unless the query
names a partner institution or an explicitly partner-taught programme. Cheapest version: a metadata
filter, same shape as the existing `is_current` filter.

### 4. Retrieved the right document but didn't use all of it (independent-chair cluster, same session, 3 turns)

One user asked three phrasings of "when is an independent chair required" in one conversation; all three
got thumbs-down for incompleteness. I checked the source: `independent-chairs-policy.pdf` §3.1 lists
**six** explicit circumstances (no prior examining experience, referral/resubmission viva, appeal viva,
candidate circumstances, staff candidate, ...). The document **was** in the retrieved top-6 for turn 2,
but every answer surfaced only 1–2 of the six criteria, leaning on a vaguer companion document
(`code-practice-vivas.pdf`) instead of enumerating the specific policy's list.

**Diagnosis:** this isn't a retrieval miss (the right doc was there) — it's a generation/synthesis gap,
possibly compounded by the bullet list being split across chunk boundaries so no single chunk contains
the full enumeration.

**Lever:** check whether §3.1's six bullets survive as one chunk (`data/text_cache/ed39f081c70e889d.txt`,
manifest doc `independent-chairs-policy.pdf`); if split, this is a chunking fix. Separately, nudge the
generator to enumerate all list items found in context rather than paraphrasing narrowly.

## What this is NOT flagging

No thumbs-down in this log mentions a hallucinated *figure* (the thing Round 6 obsessed over) — every
complaint is about **completeness/scope** (wrong topic carried over, missing entities, missing list
items, wrong institution), not fabricated numbers. That's a useful signal: the generator-faithfulness
work paid off, and the next-highest-value lever is now retrieval/contextualizer *scope*, not groundedness.

## Suggested priority

1. **Contextualizer topic-boundary fix** (#1) — highest frequency, clearest evidence, already has exact
   reproduction cases in the log.
2. **Partner-institution filter** (#3) — cheap (metadata filter, same pattern as `is_current`), clear win.
3. **Multi-entity query handling** (#2) — real but more involved (retrieval-loop or prompt change).
4. **Independent-chair chunking check** (#4) — single-document investigation, low effort to verify.
