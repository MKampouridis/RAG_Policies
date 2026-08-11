# External review — Round 7, companion prompt: **latency**

A focused companion to my main round-7 review prompt. **This one is only about speed.** The system answers correctly often enough that response time is now the thing users actually complain about, and I'd like specific, critical advice on attacking it.

**Repo:** https://github.com/MKampouridis/RAG_Policies — `eval/report.md` Rounds 11 and 12 are the full latency record.

Everything below is **measured**, not estimated. Where something is a hypothesis I say so.

---

## The system

A conversational RAG assistant over ~1,170 University of Essex policy and rules-of-assessment PDFs (21,709 chunks).

**Retrieval — entirely local, on the machine below:**
1. Dense: Chroma, `nomic-embed-text` embeddings served by Ollama
2. Lexical: BM25 (`rank_bm25`), in-process
3. Fusion: RRF over the two lists
4. Rerank: **ColBERT** (`lightonai/GTE-ModernColBERT-v1` via `pylate`, MaxSim) over a **top-30 candidate pool** → returns top-6
   - Document embeddings for the rerank pool are **precomputed offline and cached** (verified: full cache hit on a real 48-candidate pool). Only the *query* is encoded at request time.
5. A few extra dense queries when a question names multiple entities (departments/faculties)

**Generation — cloud:** `claude-sonnet-5`, `max_tokens=2048`, **not streamed** (single blocking call, whole answer returned at once).
**Query contextualizer — cloud:** `claude-haiku-4-5`, runs **only on follow-up turns**.

**Hardware:** MacBook Pro, **Apple M1 Pro, 16GB**, macOS. Served by uvicorn under launchd (`KeepAlive`). Ollama holds the embedding model in a separate process.

---

## What I measured

Per-stage timing (`RAG_TIMING=1` → `data/latency.jsonl`): **473 records over 118 real requests**, 2026-08-09 → 11.

### There are two systems, depending on warmth

| | n | median total | retrieve | generate | contextualize |
|---|---|---|---|---|---|
| **warm** (request within 10 min of the last) | 101 | **9.8s** | 0.5–3.4s | 6.8s | ~0.5s |
| **cold** (>10 min idle) | 17 | **28.2s** | **22.6s** | 6.8s | ~0.5s |

p90 total **28.2s**; worst observed **84.3s**.

**Warm, the cloud generator is essentially the entire cost.** Cold, retrieval cost ~21 extra seconds.

### The cold penalty was lazy initialisation — and I've now fixed it

Reproduced outside the server (20.87s first call vs 0.50s third, fresh process), then decomposed by pre-warming each loader in turn:

| component | cold cost |
|---|---|
| torch / ColBERT **first-encode warmup** | ~8.0s |
| ColBERT model load (`pylate`) | ~4.1s |
| BM25 index build (`rank_bm25`, 21.7k chunks) | ~2.5s |
| module import | ~2.5s |
| Chroma first dense query | ~0.3s |

Everything is lazily initialised on first use, and `KeepAlive` restarts reset all of it. **Only 14% of my logged requests were cold — but that log is my own heavy use. A tester asking two questions a day was cold nearly every time.**

**Fix built and measured:** one throwaway `retrieve()` in a daemon thread at startup.

| arm | first query | all trials |
|---|---|---|
| no warmup | **16.97s** | 16.88 / 16.97 / 17.28 |
| warmup | **3.40s** | 3.38 / 3.40 / 3.41 |

6 fresh processes (one per trial — the effect *is* process-level lazy init), arms alternated A,B,A,B,A,B so the OS page cache couldn't systematically favour whichever ran first. **−13.6s on the first query**, spread 0.4s.

Verified on the real server: pages serve in **0.4ms while the warmup is still running**, `[warmup] retrieval stack ready in 17.7s`, RSS 400MB → 4.87GB. Top-6 document sets **identical** with warmup on vs off across 4 queries. It also introduced a race (two threads could each load a full ColBERT model) which I fixed with double-checked locking, tested with 4 racing threads.

### So this is where it now stands

| | before | after |
|---|---|---|
| first query after restart/idle | ~17s (retrieval alone) | **3.4s** |
| warm end-to-end | 9.8s | 9.8s (**unchanged**) |

**Warm latency is untouched, and that's now the whole problem.** Breakdown of a warm 3.45s retrieve:

| | |
|---|---|
| Ollama embed (nomic) | 0.10s |
| BM25 query | 0.12s |
| **ColBERT query encode** | **0.76s** |
| remainder (MaxSim rank, multi-entity extra queries, fusion) | ~2.5s |

And the warm total of 9.8s is **~6.8s generation + ~3s retrieval**.

---

## Already tried, ruled out, or known — please don't re-suggest

- **Smaller fetch pool** (`FETCH_POOL_MULTIPLIER` 8→4): **costs quality, saved ~1s of 42s.** Rejected and recorded.
- **Haiku 4.5 as the generator instead of Sonnet 5**: not a speed shortcut — it was **worse on quality**, not equal (primary mean 3.94 vs Sonnet 4.25; *below the local model* on several metrics). So "just use a smaller model" is already falsified here.
- **Cloud contextualizer choice**: Sonnet→Haiku was adopted for cost; the contextualizer is ~0.5s and only runs on follow-ups, so it isn't the lever.
- **Local generation** (`gemma3:12b`) is still supported and is what all eval baselines use — but it's *slower* than the cloud call on this hardware, not faster, and scores below Sonnet.
- **Document embeddings for reranking are already precomputed and cached.** Verified full cache hit; only the query is encoded live.

## Constraints that make this harder than a generic "make it fast"

1. **Quality cannot regress.** This project's discipline is that mechanisms ship only after an eval justifies them, and a lot of retrieval quality was hard-won. A speedup that costs accuracy is not a speedup.
2. **The noise floor is ±0.20 on a 10-question set** with a cloud generator, so small quality regressions from a latency change are genuinely hard for me to detect. Latency itself I can measure tightly (spread of 0.4s above).
3. **16GB, shared.** Ollama's embedding model, the ColBERT model (~4.9GB RSS once warm), and Chroma all coexist. Anything that adds a resident model has to fit.
4. Single machine, no GPU beyond the M1 Pro's.

---

## What I want from you

1. **The warm case is now 6.8s generation + ~3s retrieval. Where would you attack first, and why?**

2. **Generation (6.8s of 9.8s).** Is **streaming** the honest answer here — i.e. accept that total time is unchanged and just make time-to-first-token small — or is there a *real* saving available that doesn't cost answer quality? Prompt-caching the system prompt and retrieved context? Shrinking `max_tokens=2048`? Cutting the number of chunks in the prompt? I'd like to know which of these actually move wall-clock for a single-user request versus which only help throughput.

3. **Retrieval (~3s warm).** 0.76s of it is encoding **one short query** with ColBERT on an M1 Pro, and ~2.5s is MaxSim + fusion + extra entity queries. Is 0.76s reasonable for a single query encode on this hardware, or does that indicate something misconfigured (device placement, float32 vs float16, `torch.compile`, MPS vs CPU)? What's the realistic floor?

4. **Is my warm/cold framing right, or is it hiding something?** I chose a 10-minute idle threshold to split the log. Should I instead be reporting the distribution a *new user* experiences? Is a median over one heavy user's traffic a misleading way to describe latency at all?

5. **The warmup fix — any way it bites me later?** It adds ~18s of background work and ~4.5GB RSS at every restart, on a 16GB machine under `KeepAlive`. I've verified it doesn't block boot and doesn't change results. What would you worry about that I haven't checked?

6. **What would you measure next that I haven't?** I have per-stage timing but no flame graph and no breakdown *inside* the ~2.5s retrieval remainder. If you think I'm optimising blind, say what instrumentation to add first.

7. **Is ~10s actually a problem?** Users compare this to ChatGPT, which streams immediately. If the honest answer is "your latency is fine, you just have no streaming and no progress indication, fix the *perception*," say that plainly — I'd rather hear it than build the wrong thing.

Please be specific and critical, and prefer concrete numbers or code paths over general advice.
