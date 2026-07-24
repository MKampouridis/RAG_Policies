#!/usr/bin/env python3
"""Advanced reranker backends for the sweep - the genuinely-DIFFERENT mechanisms
(the cross-encoders all plateaued ~+3 because they read near-identical siblings
the same way). Reuses the cached pools from reranker_sweep.py.

  colbert:<hf_model>  - late-interaction (token-level; theoretically best fit for
                        the token-distinguished sibling problem) via pylate
  qwen:<hf_model>     - LLM-based reranker: an LLM *reasons* about which sibling
                        matches the query, scored by yes/no-token probability

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/reranker_sweep_advanced.py <spec> [<spec> ...]
  e.g. colbert:answerdotai/answerai-colbert-small-v1  qwen:Qwen/Qwen3-Reranker-4B
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
POOLS = Path("eval/reranker_pools.json")


def report(model_name, per_pool_order):
    """per_pool_order: list of (pool, ordered_top_indices). Reports per-set."""
    pools = json.loads(POOLS.read_text())
    stats = {}
    for p, order in zip(pools, per_pool_order):
        s = stats.setdefault(p["set"], [0, 0, 0, 0])
        s[1] += 1
        new_hit = p["goldfam"] in {p["poolfams"][i] for i in order}
        gold_in_pool = p["goldfam"] in p["poolfams"]
        cur_hit = p["goldfam"] in p["cur_top6"]
        if new_hit:
            s[0] += 1
        if gold_in_pool and not cur_hit:
            s[3] += 1
            if new_hit:
                s[2] += 1
    for setname, (hit, n, rescued, ipm) in stats.items():
        print(f"RESULT {model_name:46s} [{setname}] family-hit@6 {hit}/{n} = {hit / n * 100:.1f}% "
              f"| ranking-failures rescued {rescued}/{ipm}", flush=True)


def colbert_orders(model_name, pools, top_n=6):
    from pylate import models, rank
    m = models.ColBERT(model_name_or_path=model_name)
    orders = []
    for p in pools:
        q = m.encode([p["query"]], is_query=True)
        d = m.encode(p["passages"], is_query=False)
        res = rank.rerank(documents_ids=[list(range(len(p["passages"])))],
                          queries_embeddings=q, documents_embeddings=[d])
        orders.append([r["id"] for r in res[0][:top_n]])
    return orders


def qwen_orders(model_name, pools, top_n=6):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mdl = mdl.to(dev)
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    task = "Given a user question about University of Essex rules of assessment, retrieve the single document whose programme/degree/year identity matches the question."
    pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
           'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def score(query, doc):
        text = f"{pre}<Instruct>: {task}\n<Query>: {query}\n<Document>: {doc[:1200]}{suf}"
        inp = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        with torch.no_grad():
            logits = mdl(**inp).logits[0, -1]
        y, n = logits[yes_id].float(), logits[no_id].float()
        return torch.softmax(torch.stack([n, y]), 0)[1].item()

    orders = []
    for i, p in enumerate(pools, 1):
        sc = [score(p["query"], d) for d in p["passages"]]
        orders.append(sorted(range(len(sc)), key=lambda j: sc[j], reverse=True)[:top_n])
        if i % 10 == 0:
            print(f"    qwen {i}/{len(pools)}", flush=True)
    return orders


if __name__ == "__main__":
    pools = json.loads(POOLS.read_text())
    print(f"pools: {len(pools)} (baseline ColBERT main 27/40, set2 26/40)\n", flush=True)
    for spec in sys.argv[1:]:
        backend, model = spec.split(":", 1)
        print(f"=== {spec} ===", flush=True)
        orders = colbert_orders(model, pools) if backend == "colbert" else qwen_orders(model, pools)
        report(spec, orders)
