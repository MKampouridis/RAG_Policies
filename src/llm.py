"""Thin wrappers around the local Ollama models. Swap models by changing the
constants below — nothing else in the codebase needs to change."""

import ollama

CHAT_MODEL = "qwen2.5:7b-instruct"

# Embedding model + its required task prefixes (asymmetric embedding models
# need different prefix text for indexed documents vs search queries, and get
# it wrong silently - always set all three together when swapping EMBED_MODEL).
# Two alternatives tested and rejected (see eval/EXPERIMENTS.md): mxbai-embed-large
# won on policy but regressed RoA (likely its 512-token window truncating dense
# chunks); bge-m3 (8192-token context, so no truncation risk) was a wash-to-
# slight-regression on RoA anyway - the corpus's near-duplicate-boilerplate
# structure appears to be the dominant constraint, not embedding model choice.
EMBED_MODEL = "nomic-embed-text"
EMBED_DOCUMENT_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "
# bge-m3 alternative (wash/slight regression on RoA, no prefix needed):
# EMBED_MODEL = "bge-m3"
# EMBED_DOCUMENT_PREFIX = ""
# EMBED_QUERY_PREFIX = ""
# mxbai-embed-large alternative (stronger on policy, weaker on RoA):
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
