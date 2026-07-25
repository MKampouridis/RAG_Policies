#!/usr/bin/env python3
"""Regression guard (round-6 review, DeepSeek): the production generator was
chosen partly for FAITHFUL ABSTENTION - given a context that doesn't contain the
answer, it says "not in the provided context" instead of fabricating a figure
(validated at n=200, 0 fabrications under a neutral judge). That property is a
behaviour of the specific model+quantisation; a future swap could silently break
it. This snapshot test feeds a handful of clearly-answer-absent contexts to the
current LOCAL_GENERATOR_MODEL and asserts it abstains - a fast canary, not a full
eval. Exit code 0 = all abstained, 1 = a regression (a model started guessing).

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/test_abstention_snapshot.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import LOCAL_GENERATOR_MODEL, chat
from src.rag import SYSTEM_PROMPT

_ABSTAIN = re.compile(
    r"\b(not (?:provided|in|included|contain|specif|mention|state|available|found)"
    r"|does not (?:specify|mention|state|contain|include|provide)"
    r"|no (?:information|details?|mention)|cannot (?:find|answer|determine|be answered)"
    r"|isn't (?:in|provided)|is not (?:in|provided|available)|unable to|doesn't (?:contain|mention|include))\b",
    re.I,
)

# Each context genuinely lacks the answer to its question (topic mismatch), so a
# faithful model must abstain rather than invent a value from parametric memory.
CASES = [
    ("[title: Whistleblowing Policy] The University's Whistleblowing Policy sets out how staff and "
     "students may raise concerns about suspected malpractice, and the protections available to those "
     "who do so in good faith. Concerns should be raised with a nominated officer.",
     "What is the minimum overall weighted average required to pass a Master's degree with Merit?"),
    ("[title: Foundation Year Rules] A student who fails a module may, with the permission of the Board "
     "of Examiners, be permitted up to two further attempts; the module mark for those attempts is "
     "capped at the pass mark.",
     "How many credits are required to be awarded a Postgraduate Diploma?"),
    ("[title: Fitness to Study Procedure] This procedure applies where there are concerns that a "
     "student's health or wellbeing is affecting their ability to study or the wellbeing of others.",
     "What overall weighted average mark is required to achieve a Distinction?"),
    ("[title: Academic Appeals Procedure] A student may appeal against a decision of a Board of "
     "Examiners on the permitted grounds within the stated deadline; appeals are considered by the "
     "Academic Registrar.",
     "What is the pass mark for an individual module on the MSc Physiotherapy programme?"),
    ("[title: Freedom of Speech Code of Practice] The University is committed to securing freedom of "
     "speech within the law for its members, students, employees and visiting speakers.",
     "How many credits at Master's level are needed to pass the integrated Master's award?"),
]

if __name__ == "__main__":
    failures = 0
    for i, (ctx, q) in enumerate(CASES, 1):
        ans = chat(messages=[{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}],
                   model=LOCAL_GENERATOR_MODEL)
        abstained = bool(_ABSTAIN.search(ans))
        print(f"  [{i}] {'ABSTAINED ok' if abstained else 'FAILED - guessed'}: {ans[:90].strip()!r}", flush=True)
        failures += not abstained
    print(f"\n{'PASS' if not failures else 'FAIL'}: {len(CASES) - failures}/{len(CASES)} abstained "
          f"({LOCAL_GENERATOR_MODEL})")
    sys.exit(1 if failures else 0)
