"""FastAPI app: conversation + chat endpoints, serves the single-page UI."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import feedback as feedback_store
from src import memory
from src.rag import answer as rag_answer

app = FastAPI(title="Essex Policies & Rules of Assessment Assistant")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class NewConversation(BaseModel):
    title: str | None = None


class NewMessage(BaseModel):
    content: str


class Feedback(BaseModel):
    rating: str  # "up" | "down"
    question: str
    answer: str
    conversation_id: str | None = None
    retrieval_query: str | None = None
    sources: list[str] = []
    ranked_top_urls: list[str] = []
    tags: list[str] = []
    comment: str | None = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/conversations")
def api_list_conversations():
    return memory.list_conversations()


@app.post("/api/conversations")
def api_create_conversation(payload: NewConversation):
    title = payload.title or "New conversation"
    conv_id = memory.create_conversation(title)
    return {"id": conv_id, "title": title}


@app.get("/api/conversations/{conversation_id}/messages")
def api_get_messages(conversation_id: str):
    if not memory.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return memory.get_messages(conversation_id)


@app.post("/api/conversations/{conversation_id}/messages")
def api_post_message(conversation_id: str, payload: NewMessage):
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")
    if not memory.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")

    summary, history = memory.get_conversation_context(conversation_id)
    is_first_message = not summary and not history

    memory.add_message(conversation_id, "user", payload.content)
    if is_first_message:
        memory.update_title(conversation_id, payload.content[:60])

    history_for_prompt = [{"role": m["role"], "content": m["content"]} for m in history]
    answer_text, sources, retrieval_query, ranked_top_urls = rag_answer(payload.content, history_for_prompt, summary)

    memory.add_message(conversation_id, "assistant", answer_text)

    return {
        "answer": answer_text,
        "sources": sources,
        # exposed so callers (the eval harness) can score the exact retrieval
        # this answer was generated from, instead of re-deriving it via a
        # second, independently-sampled retrieve() call - see rag.answer()'s
        # docstring. The UI also echoes these back with any feedback, so a
        # rating carries the retrieval context needed to auto-diagnose it.
        "retrieval_query": retrieval_query,
        "ranked_top_urls": ranked_top_urls,
    }


@app.post("/api/feedback")
def api_feedback(fb: Feedback):
    if fb.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    feedback_store.record_feedback(fb.model_dump())
    return {"ok": True}
