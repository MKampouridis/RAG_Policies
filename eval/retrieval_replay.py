#!/usr/bin/env python3
"""Retrieval-only replay: score hit@6 without generating or judging anything.

Built to measure one thing cleanly - what re-embedding the 71 stale-index
documents (eval/stale_index_audit.py) does to retrieval - and deliberately NOT
by re-running the 80-turn eval, which is the wrong instrument here:

  * ATTRIBUTION. The 80-turn eval compares against a baseline measured before
    several other changes (contextualizer topic-switch fix, partner demotion,
    multi-entity rule, variance-gated disclosure). A move in that number cannot
    be attributed to the index. Here the index is the only thing that differs
    between two passes.
  * DETERMINISM. hit@6 is a pure retrieval metric, so generation and judging -
    where nearly all run-to-run variance lives - are pure overhead AND noise.
    Replaying each turn's STORED retrieval_query also removes the contextualizer,
    the only LLM left in the retrieval path. (An eval run against a server that
    was not started with RAG_DETERMINISTIC=1 is how a run got voided on
    2026-08-07; this design cannot repeat that.)
  * COST. Minutes rather than ~90 of contended RAM, so it is cheap to run on
    both sides of the re-embed.

Method: read a committed results file, take each turn's recorded
retrieval_query verbatim, push it through retrieve(), and score whether the
gold source_url appears in the top 6 - the same rule as run_eval's
score_retrieval, so numbers are directly comparable to that file's hit@6.

Usage:
  PYTHONPATH=. python eval/retrieval_replay.py <label> [results.json ...]
Writes eval/retrieval_replay_<label>.json

Compare two passes:
  PYTHONPATH=. python eval/retrieval_replay.py before
  ... re-embed ...
  PYTHONPATH=. python eval/retrieval_replay.py after
  PYTHONPATH=. python eval/retrieval_replay.py --diff before after
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_RESULTS = ["eval/results_gemma3_e2e_main.json", "eval/results_gemma3_e2e_set2.json"]


def load(label):
    return json.loads(Path(f"eval/retrieval_replay_{label}.json").read_text())


def diff(a_label, b_label):
    a, b = {}, {}
    for row in load(a_label):
        a[(row["results_file"], row["source_url"], row["turn"])] = row
    for row in load(b_label):
        b[(row["results_file"], row["source_url"], row["turn"])] = row
    keys = sorted(set(a) & set(b))
    gained = [k for k in keys if not a[k]["hit_at_6"] and b[k]["hit_at_6"]]
    lost = [k for k in keys if a[k]["hit_at_6"] and not b[k]["hit_at_6"]]
    ha = sum(1 for k in keys if a[k]["hit_at_6"])
    hb = sum(1 for k in keys if b[k]["hit_at_6"])
    print(f"turns compared: {len(keys)}")
    print(f"  {a_label:>10s} hit@6: {ha}/{len(keys)} ({ha/len(keys)*100:.1f}%)")
    print(f"  {b_label:>10s} hit@6: {hb}/{len(keys)} ({hb/len(keys)*100:.1f}%)")
    print(f"  net: {hb - ha:+d}   gained {len(gained)}, lost {len(lost)}")
    for tag, ks in (("GAINED", gained), ("LOST", lost)):
        for k in ks:
            print(f"    {tag:6s} {k[1].split('/')[-1][:52]:52s} [{k[2]}]")
    # rank movement on turns that hit in both passes
    moved = [(k, a[k]["rank"], b[k]["rank"]) for k in keys
             if a[k]["hit_at_6"] and b[k]["hit_at_6"] and a[k]["rank"] != b[k]["rank"]]
    if moved:
        better = sum(1 for _, ra, rb in moved if rb < ra)
        print(f"  rank changed on {len(moved)} turns held by both ({better} improved)")


def main():
    if sys.argv[1:2] == ["--diff"]:
        diff(sys.argv[2], sys.argv[3])
        return

    from src.rag import retrieve

    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    results = sys.argv[2:] or DEFAULT_RESULTS

    rows = []
    for rf in results:
        p = Path(rf)
        if not p.is_file():
            print(f"skip (missing): {rf}", flush=True)
            continue
        data = json.loads(p.read_text())
        hits = total = 0
        for r in data:
            gold = r["source_url"]
            for turn in ("primary", "follow_up"):
                t = r.get(turn)
                if not t:
                    continue
                query = t["retrieval"].get("retrieval_query") or t.get("question")
                if not query:
                    continue
                res, _ = retrieve(query, [])
                urls = [m.get("source_url") for m in res.get("metadatas", [[]])[0]]
                rank = urls.index(gold) + 1 if gold in urls else None
                hit = rank is not None and rank <= 6
                hits += hit
                total += 1
                rows.append({"results_file": p.name, "source_url": gold, "turn": turn,
                             "query": query, "rank": rank, "hit_at_6": hit})
        print(f"  {p.name}: hit@6 {hits}/{total} ({hits/max(total,1)*100:.1f}%)", flush=True)

    out = Path(f"eval/retrieval_replay_{label}.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    h = sum(1 for r in rows if r["hit_at_6"])
    print(f"\nTOTAL hit@6: {h}/{len(rows)} ({h/max(len(rows),1)*100:.1f}%)  -> {out}")


if __name__ == "__main__":
    main()
