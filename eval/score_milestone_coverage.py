#!/usr/bin/env python3
"""Score whether an answer covers every milestone a document defines.

Two ways an answer legitimately presents a milestone, and a scorer that sees
only one of them reports nonsense:

  LABELLED  "**M2.7:** Demonstrate effective project management ..."
  PROSE     "* Demonstrate effective project management ..."

A label-only count scored five complete prose answers 0/N (Round 34i). A
description-only count under-counted a labelled answer that cited all 19 codes
at 6/19, because the word-overlap proxy is loose (Round 34j). Neither signal is
sufficient alone, so a milestone counts as covered when EITHER its code appears
or its description does - and the two are reported separately so a
disagreement between them is visible rather than averaged away.

Usage:
    PYTHONPATH=. python eval/score_milestone_coverage.py            # all
    PYTHONPATH=. python eval/score_milestone_coverage.py <substring> # subset
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CODE = re.compile(r"\b([MC]\d+\.\d+)")
STOP = set("the a an of and to for in with on its their this that be is are as at by or "
           "will must should which where when what who whom".split())
# term/credit columns sit between the code and its description in these
# table-derived documents ("M1.1 1 1 Assess training needs ..."), so leading
# digits are skipped rather than treated as the description's first words
LEADING_NUMS = re.compile(r"^[\s\d/]+")


def milestones(text: str) -> dict:
    """code -> description, taken from the document itself."""
    out = {}
    for m in re.finditer(r"\b([MC]\d+\.\d+)\s+(.{0,110})", text):
        out.setdefault(m.group(1), LEADING_NUMS.sub("", m.group(2)))
    return out


def covered(code: str, desc: str, answer_low: str) -> tuple[bool, bool]:
    """(by_label, by_description) for one milestone."""
    by_label = code.lower() in answer_low
    words = [w for w in re.findall(r"[a-z]{4,}", desc.lower()) if w not in STOP][:5]
    by_desc = bool(words) and sum(1 for w in words if w in answer_low) >= max(2, len(words) - 2)
    return by_label, by_desc


def score(answer: str, doc_text: str) -> dict:
    # the fallback note names the codes it could NOT cover; counting them would
    # credit the note for the very milestones it is reporting as missing
    body = answer.split("_This list may be incomplete")[0].lower()
    defined = milestones(doc_text)
    by_label = by_desc = both = 0
    missing = []
    for code, desc in defined.items():
        lab, dsc = covered(code, desc, body)
        by_label += lab
        by_desc += dsc
        if lab or dsc:
            both += 1
        else:
            missing.append(code)
    return {"total": len(defined), "covered": both, "by_label": by_label,
            "by_description": by_desc, "missing": missing}


def main() -> int:
    from src.ingest import _get_collection
    import src.rag as R

    want = sys.argv[1] if len(sys.argv) > 1 else ""
    body = collections.defaultdict(dict)
    for m, doc in zip(*[_get_collection().get(include=["metadatas", "documents"])[k]
                        for k in ("metadatas", "documents")]):
        u = m.get("source_url", "")
        if u and m.get("is_current"):
            body[u][m.get("chunk_index") or 0] = doc

    items = json.load(open("eval/questions_milestones.json"))["items"]
    items = [i for i in items if want in i["source_title"]]
    rows, full = [], 0
    for i, item in enumerate(items, 1):
        text = " ".join(body[item["source_url"]][k] for k in sorted(body.get(item["source_url"], {})))
        ans, *_ = R.answer(item["question"], [])
        s = score(ans, text)
        s.update({"programme": item["programme"], "department": item["department"],
                  "source_title": item["source_title"]})
        rows.append(s)
        full += s["covered"] == s["total"]
        print(f"[{i}/{len(items)}] {item['source_title'][:44]:44} "
              f"{s['covered']}/{s['total']}  (label {s['by_label']}, desc {s['by_description']})",
              flush=True)
    out = Path("eval/milestone_coverage_scores.json")
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    tc = sum(r["covered"] for r in rows); tt = sum(r["total"] for r in rows)
    print(f"\n  {full}/{len(rows)} programmes complete   {tc}/{tt} milestones "
          f"({100 * tc / max(tt, 1):.1f}%)   -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
