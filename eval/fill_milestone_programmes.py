#!/usr/bin/env python3
"""Fill empty programme_name on PGRE milestone identity records.

extract_doc_identity.py asks a local model for the programme name and left it
EMPTY on 59 of the 80 current milestone documents - the prompt tells it to
return "" for a general/university-wide document, and these read as generic.
But these documents state their programme deterministically in the first line:

    2025-2026 Entry  <department>  Faculty of <x>  <PROGRAMME>  Students
    Postgraduate Research milestones are used to ...

Parsing that is exact where the model guessed, and it is the same parser
eval/build_milestone_questions.py already uses to generate 78 of 80 questions.
Only fills BLANKS - a name the model did extract is never overwritten.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest import _get_collection, url_hash

PAT = re.compile(r"Entry\s+(.*?)\s+Students\s+Postgraduate Research milestones", re.I)
FACULTY = re.compile(r"\s*Faculty of (?:Social Sciences|Science and Health|Arts and Humanities)\s*")


def main() -> int:
    d = _get_collection().get(include=["metadatas", "documents"])
    head = {}
    for m, doc in zip(d["metadatas"], d["documents"]):
        u = m.get("source_url", "")
        if "/pgre/milestones-" in u and (m.get("chunk_index") or 0) == 0:
            head[u] = doc[:260].replace("\n", " ")

    filled = skipped = already = 0
    for url, text in head.items():
        path = Path("data/doc_identity") / f"{url_hash(url)}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if rec.get("programme_name"):
            already += 1
            continue
        m = PAT.search(text)
        prog = FACULTY.split(m.group(1).strip())[-1].strip() if m else ""
        if not prog or len(prog) > 80:
            skipped += 1
            continue
        rec["programme_name"] = prog
        rec["programme_name_source"] = "parsed from document header"
        path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        filled += 1
    print(f"  filled {filled}   already had one {already}   unparseable {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
