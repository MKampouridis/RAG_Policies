#!/usr/bin/env python3
"""Byte-exact snapshot of what retrieval returns, for verifying a refactor.

A module split is a PURE MOVE: no logic changes, so the output must be
IDENTICAL - not "as good as", not "within the noise floor". That is a far
stricter test than this project usually gets, and it is only available because
nothing is meant to change.

Records, per replayed query, the ordered (source_url, chunk_index) of every
returned chunk plus the retrieval query. Any difference at all is a bug
introduced by the move.

    PYTHONPATH=. python eval/refactor_fingerprint.py before
    ...refactor...
    PYTHONPATH=. python eval/refactor_fingerprint.py after
    PYTHONPATH=. python eval/refactor_fingerprint.py --compare before after
"""
import hashlib
import json
import pathlib
import sys


def path_for(label): return pathlib.Path(f"eval/fingerprint_{label}.json")


def capture(label: str) -> int:
    from src import rag
    rows = json.loads(pathlib.Path("eval/retrieval_replay_cache_off.json").read_text())
    out = []
    for i, r in enumerate(rows, 1):
        q = r.get("query") or ""
        res, rq = rag.retrieve(q, [])
        metas = res.get("metadatas", [[]])[0]
        out.append({"query": q, "retrieval_query": rq,
                    "chunks": [[m.get("source_url"), m.get("chunk_index")] for m in metas]})
        if i % 40 == 0:
            print(f"  {i}/{len(rows)}")
    p = path_for(label)
    p.write_text(json.dumps(out, indent=1))
    digest = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16]
    print(f"\n  captured {len(out)} queries -> {p}")
    print(f"  fingerprint: {digest}")
    return 0


def compare(a: str, b: str) -> int:
    A = json.loads(path_for(a).read_text())
    B = json.loads(path_for(b).read_text())
    if len(A) != len(B):
        print(f"  DIFFERENT LENGTHS: {len(A)} vs {len(B)}")
        return 1
    diffs = [i for i, (x, y) in enumerate(zip(A, B)) if x != y]
    print(f"\n  queries compared : {len(A)}")
    print(f"  IDENTICAL        : {len(A) - len(diffs)}")
    print(f"  DIFFERENT        : {len(diffs)}")
    for i in diffs[:8]:
        print(f"\n    q: {A[i]['query'][:80]}")
        print(f"      before: {[c[0].rsplit('/',1)[-1][:26] for c in A[i]['chunks'][:4]]}")
        print(f"      after : {[c[0].rsplit('/',1)[-1][:26] for c in B[i]['chunks'][:4]]}")
    if not diffs:
        print("\n  The refactor changed nothing. That is the only acceptable result.")
    return 1 if diffs else 0


if __name__ == "__main__":
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        raise SystemExit(compare(sys.argv[i + 1], sys.argv[i + 2]))
    raise SystemExit(capture(sys.argv[1] if len(sys.argv) > 1 else "before"))
