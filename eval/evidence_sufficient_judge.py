"""Keyphrase-metric fix (round 5, Fable 5's refinement). The keyphrase-based
evidence-sufficient@6 in score_summary.py credits a retrieved document only when
>= half the turn's keyphrases appear as EXACT case-insensitive substrings. Round-4
item 2 found 6 of the 12 current misses are N=0 keyphrase-proxy artifacts: the gold
answer IS in a retrieved document, but a keyphrase is phrased differently there
("subsequent year" vs "the following year of study"), so the literal-substring
test under-credits it. That brittleness also caps the headline evidence-sufficient@6.

This is the recommended refinement: a JUDGE-BASED (reference-answer-containment)
sufficiency check. For each turn scored INSUFFICIENT by the keyphrase rule, ask the
judge whether ANY of the turn's top-6 retrieved documents actually contains the
information in the gold reference answer. Early-exit on the first "yes". Only the
keyphrase-insufficient turns are judged (targeted + cheap - the concern is false
MISSES from string brittleness, not false positives), so this can only raise the
number, never lower it, and every rescue is printed for inspection.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/evidence_sufficient_judge.py [results_file] [questions_file]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import JUDGE_MODEL, chat

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/results_c1_anchor_v2.json")
QUESTIONS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("eval/questions.json")
MANIFEST = Path("data/manifest.json")

manifest = json.loads(MANIFEST.read_text())["documents"]
_text_cache: dict[str, str] = {}


def doc_text(url: str) -> str:
    if url not in _text_cache:
        p = Path((manifest.get(url) or {}).get("text_cache_path", ""))
        # is_file(), not exists(): an empty text_cache_path yields Path(".") which
        # exists() reports True (the cwd) and then read_text() raises IsADirectoryError.
        _text_cache[url] = p.read_text(encoding="utf-8") if p.is_file() else ""
    return _text_cache[url]


def keyphrase_sufficient(top_urls, keyphrases) -> bool | None:
    """The existing string-substring rule from score_summary.py, replicated so the
    two views are computed on identical inputs."""
    if not keyphrases:
        return None
    for url in dict.fromkeys(top_urls):
        text = doc_text(url).lower()
        if text and sum(1 for kp in keyphrases if kp.lower() in text) >= (len(keyphrases) + 1) // 2:
            return True
    return False


JUDGE_SYS = (
    "You are a strict retrieval-sufficiency checker for a university rules/policy assistant. "
    "You are given a REFERENCE ANSWER (known-correct) and one retrieved DOCUMENT. Decide only "
    "whether the document CONTAINS the specific factual information asserted in the reference "
    "answer - the actual figures, rules, and conditions - regardless of wording. Do not reward "
    "mere topical overlap or generic boilerplate that happens to share vocabulary. "
    'Reply with ONLY "yes" or "no".'
)


def judge_contains(url: str, expected_answer: str) -> bool:
    text = doc_text(url)
    if not text:
        return False
    # num_ctx=8192 (~6k tokens); keep the document within budget after the system
    # prompt + reference answer. Docs longer than this are truncated - noted as a
    # limitation, acceptable because this only re-checks a handful of flagged turns.
    text = text[:22000]
    raw = chat(
        messages=[
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": f"REFERENCE ANSWER:\n{expected_answer}\n\nRETRIEVED DOCUMENT:\n{text}"},
        ],
        model=JUDGE_MODEL,
    ).strip().lower()
    return raw.startswith("y")


def judge_sufficient(top_urls, expected_answer) -> tuple[bool, str]:
    for url in dict.fromkeys(top_urls):
        if judge_contains(url, expected_answer):
            return True, url
    return False, ""


questions = {q["source_url"]: q for q in json.loads(QUESTIONS.read_text())}
results = json.loads(RESULTS.read_text())

# rows: (label, doc_type, kp_sufficient, judge_rescued, rescuing_url)
rows = []
for r in results:
    q = questions.get(r["source_url"])
    if not q:
        continue
    for turn, kpkey in (("primary", "keyphrases"), ("follow_up", "follow_up_keyphrases")):
        kps = q.get(kpkey) or []
        kp_suf = keyphrase_sufficient(r[turn]["retrieval"]["top_urls"], kps)
        if kp_suf is None:
            continue
        rescued, url = (False, "")
        if not kp_suf:
            exp = r[turn].get("expected_answer") or q.get(kpkey.replace("keyphrases", "expected_answer")) or ""
            if exp:
                rescued, url = judge_sufficient(r[turn]["retrieval"]["top_urls"], exp)
                print(f"  {'RESCUED' if rescued else 'still-miss'}  {r['source_title']}[{turn}]"
                      f"{'  <- ' + url.split('/')[-1] if rescued else ''}", flush=True)
        rows.append((f"{r['source_title']}[{turn}]", r["doc_type"], bool(kp_suf), rescued, url))


def summ(subset, name):
    if not subset:
        return
    kp = sum(1 for x in subset if x[2]) / len(subset)
    jd = sum(1 for x in subset if x[2] or x[3]) / len(subset)
    print(f"{name:22s} n={len(subset):3d}  keyphrase evid-suff@6={kp*100:5.1f}%  "
          f"judge-refined={jd*100:5.1f}%  (+{sum(1 for x in subset if x[3])} rescued)")


print(f"\n=== evidence-sufficient@6: keyphrase-string vs judge-refined ({JUDGE_MODEL}) ===")
summ([x for x in rows if x[1] == "rules_of_assessment"], "RoA")
summ([x for x in rows if x[1] == "policy"], "Policy")
summ(rows, "Overall")
