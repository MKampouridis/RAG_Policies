"""FastAPI app: conversation + chat endpoints, serves the single-page UI."""

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import feedback as feedback_store
from src import memory
from src.rag import answer as rag_answer
from src.rag import GENERATED_TITLES, generate_title

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


@app.delete("/api/conversations/{conversation_id}")
def api_delete_conversation(conversation_id: str):
    if not memory.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    memory.delete_conversation(conversation_id)
    return {"ok": True}


def _retitle(conversation_id: str, question: str) -> None:
    """Best-effort replacement of the truncated fallback title. Swallows every
    error - a background task that raises would log noise for a cosmetic
    feature, and the conversation already has a usable title either way."""
    try:
        title = generate_title(question)
        if title:
            memory.update_title(conversation_id, title)
    except Exception:
        pass


@app.post("/api/conversations/{conversation_id}/messages")
def api_post_message(conversation_id: str, payload: NewMessage, background: BackgroundTasks):
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

    if is_first_message:
        # after the response, never before: a title is cosmetic and must not
        # add latency to the turn the user is waiting on. _retitle keeps the
        # truncated fallback already stored if generation fails.
        background.add_task(_retitle, conversation_id, payload.content)

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
        # The generated title lands ~5s after this response (background task),
        # so the client must be told to re-read the conversation list; without
        # this it renders the truncated fallback once and never looks again.
        "title_pending": is_first_message and GENERATED_TITLES,
    }


@app.post("/api/feedback")
def api_feedback(fb: Feedback):
    if fb.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    feedback_store.record_feedback(fb.model_dump())
    return {"ok": True}
