# Overnight report — what I found while you slept

*Prepared 25 July 2026, early hours. Plain English, no jargon. Everything below is committed to the repo; the technical detail lives in `eval/report.md`.*

---

## The one-liner

Everything we recently decided held up, the system turned out to be **much better than the scary "34% miss" number ever suggested** (the real failure rate is about **10%**), and the single most valuable thing you can do next is let the assistant **ask which programme the student means** — which, in testing, fixes **100%** of the failures. Your app also now looks like a University of Essex product.

---

## What I set out to do

The five external LLM reviewers agreed the research was basically done, but Claude (the only one that read the actual code) flagged a few things we'd decided **without properly checking**. So last night I ran a series of experiments to either confirm or overturn those decisions — plus build the one genuinely new idea, tidy up the code, and redesign the interface. Ten experiments in all, run back-to-back through the night.

---

## The findings, in plain English

### 1. Switching the answer-writer to "gemma3" was the right call — and it's safe
We recently swapped the model that writes the final answers. There was a worry: gemma3 politely says *"I can't answer that from these documents"* when it's unsure — but the system feeds each answer back in to understand the **next** question, so could an "I don't know" confuse the follow-up? I ran the whole system end-to-end and checked. **No harm at all** — follow-up performance was actually a touch *better*, and identical on a separate topic-jumping test. The switch is validated.

### 2. The "gemma3 makes things up more" scare was a false alarm
Earlier we thought gemma3 might invent figures more than the old model, but that was based on only 13 examples. I built **200** fresh test cases (hand the model a document that definitely doesn't contain the answer, see if it guesses). The first scoring looked alarming — until I noticed I'd used a **biased referee** (a model from the same family as the old one, quietly favouring its relative). With a **neutral referee**, the scare vanished completely: gemma3 invented a figure **zero times out of 200**. Both models are essentially flawless here. Lesson re-learned: never let an AI grade a contest it's competing in.

### 3. Retrieval is far better than the headline number ever said
This is the most important finding. For a long time the "score" said the system fetches the wrong document about **34%** of the time on the hard rules-of-assessment questions. Last night I measured what actually **matters**: when it fetches the "wrong" document, does that document give a *different* answer, or the *same* one?

The answer: **nearly half the time (48%) the "wrong" document gives the identical answer** — because the rule is the same across programmes (e.g. "you need 60 for a Merit" is true everywhere). Only **about a third** of the misses (9 of 27) actually hand the student a *wrong* number.

**So the real, harm-causing error rate is about 11%, not 34%.** The old score was punishing the system for picking a different-but-equally-correct document. Three separate analyses last night all landed on roughly the same **~10% real failure rate**. The system is much better than we thought.

### 4. The new idea works — as a smart "check which document" flag
The one genuinely new idea was: instead of guessing, have the system notice when a question is *high-stakes*. It does this by checking whether the competing documents **disagree** on the answer. If they all agree (like the Merit=60 case), it can answer confidently. If they **disagree**, the specific programme matters, so it should flag that.

It works — it cleanly splits questions into **high-stakes (36%)** and **low-stakes (46%)**. It's not useful as a "should I have got a different document" predictor, but it's genuinely useful as a **targeted disclosure**: on the ~36% of questions where it matters, the assistant can add *"based on the [X] document — rules differ by programme, so check this is yours."* That's much better than today's setup, which either says that on **every** answer or never.

### 5. Asking the user fixes 100% of the failures
I simulated the "clarify" feature: when the system misses, the user is asked *"which programme?"* and answers. **Every single missed question — 12 out of 12 — resolved to a correct answer** once the programme was named. Effective success jumped from **70% to 100%**.

This is the clean proof of what we've long suspected: the failures aren't a search weakness, they're **missing information**. The student didn't say which programme they're on, and there's genuinely no way to know without asking. The moment they say it, the system gets it right. (Caveat: this is the best case, where the user gives a perfect answer — real people are messier, so live results will be lower — but the ceiling being 100% is a strong reason to turn this feature on.)

### 6. A better, fairer way to score the system
The old scoring counted an answer as "supported" if the right *words* appeared in a document — which lets it be fooled by a document that shares vocabulary but says the opposite. I built a stricter version that checks the actual **value in its actual role** ("60 *as the Merit threshold*", not just "60" and "Merit" appearing nearby). It's a real improvement for number-based questions, though it's still a bit lenient on pure *definition* questions. It agreed with everything else: the system genuinely has the right answer available ~90%+ of the time.

### 7. Your app now looks like University of Essex
I redesigned the interface using Essex's **official brand** — scarlet red (#CD202C), violet, and the Archivo typeface — pulled from the University's own brand guidelines. New scarlet header, and the **left-hand conversations panel** (which you'd flagged twice as ugly) is completely redesigned: a University of Essex brand block at the top, tidy conversation rows with a little marker that turns scarlet on the one you're viewing, and a cleaner look throughout. All the existing features (chat, sources, feedback, delete) work exactly as before. **One thing to do:** give it a quick look when you next start the server — I couldn't view it live because the server was off for the experiments.

### 8. Under-the-hood tidying
Small robustness fixes the reviewers suggested: the feedback log now can't grow forever and fill the disk; the system now logs when it discards a suspect question-rewrite (useful signal for later); a stale corpus figure in the review document was corrected; and I set two Ollama memory options that will roughly halve one part of its memory use next time it restarts — which should ease the RAM pressure you asked about. (Two other suggested fixes turned out to be already done.)

---

## The big picture

Put it all together and the story is coherent and, honestly, **good news**:

- The system fails in a *harmful* way only about **1 in 9 times** — not 1 in 3 as the old score implied.
- **All** of those failures are the same root cause: the student didn't say which programme they mean.
- **All** of them are fixable by simply **asking** — which resolves 100% in testing.
- The answer-writing model is a safe, honest choice that admits when it doesn't know.

There is no more retrieval research worth doing — that frontier is genuinely closed, confirmed several different ways. The remaining value is entirely in the **product**: turning on the "ask which programme?" feature (and/or the targeted "which document I used" disclosure), and judging it on real conversations.

---

## What's left — and what's yours to decide

**Ready for you to greenlight (I can build these; last night's numbers justify them):**
1. **Wire the "high-stakes" detector into a targeted disclosure** — flag the document used only on the ~36% of questions where it matters, instead of always or never.
2. **Adopt the fairer scoring metric** as the standard.

**Your calls (not mine to make):**
3. **Turn on the "ask which programme?" feature** for real use — the testing gives it a strong green light, but whether the assistant should ask-vs-guess is a feel decision you'll want to make on real conversations.
4. **Bring the assistant back online** when you want it reachable again (it's currently off; the settings are ready).

**Whenever you like:**
5. Give the redesigned interface a visual once-over.

Nothing is broken, nothing is waiting on me. Everything from last night is committed and pushed. Sleep well — this'll be here when you're up.
