# Essex Policies & Rules of Assessment Assistant

A conversational RAG assistant over University of Essex policy documents and
rules-of-assessment documents. Runs entirely locally (open-source models via
Ollama), keeps conversation history so you can resume a topic later, and
never stores the source PDFs — only extracted text, embeddings, and a
crawl/relevance manifest.

**Current status: local-only.** It's not yet reachable from work — see
"Hosting" below for what that will take.

## How it works

1. `run_ingest.py` crawls a small set of seed pages on `essex.ac.uk`,
   follows in-content links (policies and rules-of-assessment pages/PDFs
   only), asks the local LLM to judge whether each document is actually a
   policy or rules-of-assessment document, and — for the ones it keeps —
   chunks, embeds, and stores them in a local Chroma vector store.
2. `run_server.py` starts a small web app (chat UI + API) that answers
   questions by retrieving relevant chunks and generating a cited answer,
   while keeping a per-conversation history in SQLite so you can pick up
   a thread later.

## One-time setup

```
python3 -m pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

(Ollama must be running — the Ollama.app menu-bar app or `ollama serve`.)

## Day-to-day use

Start the server:

```
python run_server.py
```

Then open http://localhost:8000. Conversations persist in `data/chat.db`;
reopening the app later will show past conversations in the sidebar.

## Refreshing the index

When Essex publishes new documents (e.g. a new academic year's rules of
assessment), re-run:

```
python run_ingest.py
```

This re-crawls the seed pages (cheap — pages/PDFs are small over HTTP) but
only re-runs the expensive steps (LLM relevance classification + embedding)
for documents whose content actually changed since the last run. Progress
and a kept/rejected summary print as it runs; the full audit trail (what was
kept, what was rejected, and why) is in `data/manifest.json`.

To crawl additional seed pages beyond the three built-in ones, pass extra
URLs:

```
python run_ingest.py https://www.essex.ac.uk/some/other/policy-hub-page
```

## Project layout

```
requirements.txt
data/                # manifest.json, text_cache/, chroma/, chat.db (gitignored)
src/
  crawler.py          # BFS crawl + fetch + text extraction
  relevance.py         # LLM keep/reject classification
  ingest.py             # chunk + embed + upsert into Chroma
  llm.py                 # Ollama chat + embeddings wrappers (model names live here)
  rag.py                  # retrieval + prompt assembly + generation
  memory.py                # SQLite conversation storage/summarization
  app.py                    # FastAPI app: chat + conversation endpoints, serves UI
static/index.html      # single-page chat UI (no build step)
run_ingest.py         # CLI: crawl+classify+embed (re-runnable, incremental)
run_server.py        # CLI: start the web app
```

## Swapping models

Both models are named in one place: `src/llm.py` (`CHAT_MODEL`,
`EMBED_MODEL`). Pull the new model with `ollama pull <name>`, update the
constant, and re-run `run_ingest.py` if you change `EMBED_MODEL` (embeddings
from different models aren't compatible with each other).

## Hosting (not done yet)

This build runs on `127.0.0.1` only. To reach it from work as well as home,
the app is already written so it can bind to `0.0.0.0` — the remaining work
is picking a way to expose it (e.g. a small VPS, Tailscale, or a reverse
proxy) and deciding on access control now that it's not just `localhost`.
That's a deliberate follow-up, not a rewrite.
