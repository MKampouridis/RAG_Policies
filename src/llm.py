"""Thin wrappers around the local Ollama models. Swap models by changing the
constants below — nothing else in the codebase needs to change."""

import ollama

CHAT_MODEL = "qwen2.5:7b-instruct"

# Embedding model + its required task prefixes (asymmetric embedding models
# need different prefix text for indexed documents vs search queries, and get
# it wrong silently - always set all three together when swapping EMBED_MODEL).
# nomic-embed-text was evaluated against mxbai-embed-large (eval/report.md) -
# mxbai won on policy content but regressed on rules-of-assessment content
# (likely its 512-token context window truncating dense technical chunks),
# and RoA is most of the corpus, so nomic-embed-text stays the default.
EMBED_MODEL = "nomic-embed-text"
EMBED_DOCUMENT_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "
# mxbai-embed-large alternative (stronger on policy, weaker on RoA - see eval/report.md):
# EMBED_MODEL = "mxbai-embed-large"
# EMBED_DOCUMENT_PREFIX = ""
# EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def chat(messages: list[dict], format: str | None = None, model: str = CHAT_MODEL) -> str:
    response = ollama.chat(model=model, messages=messages, format=format)
    return response["message"]["content"]


def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]
