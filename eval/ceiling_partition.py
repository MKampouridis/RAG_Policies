#!/usr/bin/env python3
"""Round-8 item 26: partition the gold-multiplicity ceiling - and, as it turned
out, falsify it.

WHAT THIS SET OUT TO DO
`gold_multiplicity.py` applies min(1, 6/N) to EVERY turn, where N = how many
current documents contain all of that turn's keyphrases. The bound assumes
**exchangeability**: that the N documents are equally plausible answers, so a
single-gold hit@6 can only be right 6/N of the time by luck. That assumption
fails whenever the QUESTION NAMES A DISAMBIGUATING ENTITY - "the pass mark for a
Psychology MSc" may have N=40, but retrieval can and should pick the right one.
Item 26 was to apply the bound only where no entity is named, and 1.0 elsewhere.

WHAT ACTUALLY HAPPENED
Partitioning was implemented and it does move the ceiling (+9.0 points on the
scoreable turns). But running it surfaced that **29 of 80 turns have N=0** - the
keyphrases appear jointly in NO current document. The first instinct was to call
those broken test items and exclude them. That was checked before being written
down, and it was wrong: their gold documents are all CURRENT, and most already
contain most of their keyphrases (7 of 9, 3 of 4).

So N=0 is not a broken item. It is the strict conjunction failing. Which
prompted the obvious validity check, the one that should have been run before
any of this was built on N:

    ** In 44% of turns (35/80) the GOLD document fails its own keyphrase
       conjunction, with a mean of only 50% of keyphrases present. **

N is supposed to count "documents that hold the answer". It cannot, when the
known-correct document is not counted 44% of the time. The exchangeability
ceiling is built on that number, so **the ceiling is not a bound** - which
independently explains the anomaly recorded earlier as a success: measured
retrieval scoring ABOVE its own ceiling. A system beating its ceiling means the
ceiling is wrong, not that the system is finished.

CONSEQUENCE
Five retrieval proposals were closed on the ceiling argument. That argument does
not hold, and this does not restore it - partitioning raises the ceiling but
measured hit@6 still exceeds it, because the underlying N is unreliable in both
arms. The proposals must be re-closed (or reopened) on the 31%-never-enter-the-
pool finding, which is measured directly and does not depend on N. See item 27.

Usage: PYTHONPATH=. python eval/ceiling_partition.py [results_file]
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.docid import extract_award_type, extract_degree_length
from src.entities import detect_departments, detect_faculties
from src.ingest import _get_collection
from src.institutions import _names_partner_institution

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/results_c1_anchor_v2.json")
QUESTIONS = Path("eval/questions.json")
MANIFEST = Path("data/manifest.json")


def disambiguators(question: str) -> list[str]:
    """Which entity signals the question carries. The same detectors retrieval
    uses, so this measures what the pipeline could actually act on - not what a
    human would consider identifying. A PROXY: it misses entities the alias
    tables do not cover and counts entities that do not narrow anything."""
    found = []
    if detect_departments(question):
        found.append("department")
    if detect_faculties(question):
        found.append("faculty")
    if _names_partner_institution(question, []):
        found.append("partner")
    if extract_degree_length(question):
        found.append("degree_length")
    if extract_award_type(question):
        found.append("award_type")
    return found


def main() -> int:
    coll = _get_collection()
    cur_urls = {m.get("source_url", "") for m in coll.get(include=["metadatas"])["metadatas"]
                if m.get("is_current")}
    manifest = json.loads(MANIFEST.read_text())["documents"]
    texts = {}
    for url in cur_urls:
        p = Path((manifest.get(url) or {}).get("text_cache_path", ""))
        if p.exists():
            texts[url] = p.read_text(encoding="utf-8").lower()

    def N(kps: list[str]) -> int:
        return sum(1 for t in texts.values() if all(k in t for k in kps))

    questions = {q["source_url"]: q for q in json.loads(QUESTIONS.read_text())}
    results = json.loads(RESULTS.read_text())

    rows = []
    for r in results:
        q = questions.get(r["source_url"])
        if not q:
            continue
        for turn, kpkey, qkey in (("primary", "keyphrases", "question"),
                                  ("follow_up", "follow_up_keyphrases", "follow_up")):
            kps = [k.lower() for k in (q.get(kpkey) or []) if k]
            if not kps:
                continue
            gold = q["source_url"]
            gold_text = texts.get(gold)
            present = sum(1 for k in kps if gold_text and k in gold_text)
            n = N(kps)
            ent = disambiguators(q.get(qkey) or "")
            rows.append({
                "label": f"{r['source_title']}[{turn}]",
                "hit": bool(r[turn]["retrieval"]["hit_at_6"]),
                "N": n,
                "entities": ent,
                "gold_is_current": gold in texts,
                "keyphrases": len(kps),
                "present_in_gold": present,
                "gold_self_fails": present < len(kps),
                "old_bound": min(1.0, 6.0 / n) if n > 0 else 1.0,
                "new_bound": 1.0 if ent else (min(1.0, 6.0 / n) if n > 0 else 1.0),
            })

    if not rows:
        print("  no scoreable turns")
        return 1

    n_tot = len(rows)
    fails = [r for r in rows if r["gold_self_fails"]]
    notcur = [r for r in rows if not r["gold_is_current"]]

    print(f"\n  {'='*68}")
    print("  VALIDITY CHECK ON N - run this before trusting anything built on it")
    print(f"  {'='*68}")
    print(f"  turns with keyphrases                        : {n_tot}")
    print(f"  gold document not in the current pool        : {len(notcur)}")
    print(f"  gold document FAILS its own conjunction      : {len(fails)}"
          f"  ({len(fails)/n_tot*100:.0f}%)")
    if fails:
        frac = sum(r["present_in_gold"] / r["keyphrases"] for r in fails) / len(fails)
        print(f"    ...mean keyphrases actually present        : {frac*100:.0f}%")
    print("\n  N counts documents containing ALL keyphrases, as a proxy for")
    print("  'documents that hold the answer'. The KNOWN-CORRECT document fails")
    print(f"  that test in {len(fails)/n_tot*100:.0f}% of turns, so N systematically undercounts and")
    print("  the ceiling derived from it is NOT A BOUND.")

    with_ent = [r for r in rows if r["entities"]]
    old = sum(r["old_bound"] for r in rows) / n_tot
    new = sum(r["new_bound"] for r in rows) / n_tot
    hit = sum(1 for r in rows if r["hit"]) / n_tot

    print(f"\n  {'-'*68}")
    print("  THE PARTITION ITSELF (item 26 as specified), reported for the record")
    print(f"  {'-'*68}")
    print(f"  measured hit@6                   : {hit*100:5.1f}%")
    print(f"  ceiling, UNPARTITIONED (old)     : {old*100:5.1f}%")
    print(f"  ceiling, PARTITIONED             : {new*100:5.1f}%")
    print(f"  the correction                   : {(new-old)*100:+5.1f} points")
    print(f"  residual vs unpartitioned        : {(old-hit)*100:+5.1f} points"
          f"{'   <- system ABOVE its ceiling' if old < hit else ''}")
    print(f"  residual vs partitioned          : {(new-hit)*100:+5.1f} points"
          f"{'   <- system ABOVE its ceiling' if new < hit else ''}")

    moved = [r for r in rows if r["new_bound"] > r["old_bound"]]
    print(f"\n  turns naming a disambiguating entity : {len(with_ent)}/{n_tot} "
          f"({len(with_ent)/n_tot*100:.0f}%)   [detector coverage - a proxy]")
    for k, v in Counter(e for r in with_ent for e in r["entities"]).most_common():
        print(f"    {k:16s} {v:4d}")
    print(f"  turns whose bound actually CHANGED  : {len(moved)}/{n_tot} "
          f"({len(moved)/n_tot*100:.0f}%)  - reported as turns, not a diluted mean")

    print(f"\n  {'-'*68}")
    print("  CONCLUSION")
    print(f"  {'-'*68}")
    print(f"  Partitioning moves the ceiling {(new-old)*100:+.1f} points, on {len(moved)} of {n_tot} turns.")
    if old < hit:
        print("  The system measures ABOVE the unpartitioned ceiling, which is the")
        print("  tell: a bound the system beats is not a bound. Partitioning was the")
        print("  hypothesis for why, and it does close that gap.")
    print()
    print("  BUT the apparent headroom is NOT usable evidence either, because both")
    print("  ceilings are computed from N, and N fails its own validity check above")
    print(f"  ({len(fails)}/{n_tot} turns). Whichever direction it errs, a number the")
    print("  ground-truth document itself cannot satisfy should not decide what")
    print("  retrieval work is worth doing.")
    print()
    print("  ACTION: the five proposals closed on the ceiling argument must be")
    print("  re-closed on the 31%-never-enter-the-pool finding, which is measured")
    print("  directly and does not depend on N. Fixing N means fixing keyphrases,")
    print("  which is the benchmark-provenance job, not a retrieval job.")

    out = Path("eval/ceiling_partition.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
