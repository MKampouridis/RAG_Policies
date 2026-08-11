"""FastAPI app: conversation + chat endpoints, serves the single-page UI."""

import json
import os
import queue
import threading
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import feedback as feedback_store
from src import ingest
from src import memory
from src.rag import answer as rag_answer
from src.rag import GENERATED_TITLES, generate_title

app = FastAPI(title="Essex Policies & Rules of Assessment Assistant")

# Every retrieval dependency is lazily initialised on first use, so the FIRST
# request after a restart pays ~21s that later requests do not (Round 11:
# torch/ColBERT first-encode ~8s, ColBERT model load ~4s, BM25 index build
# ~2.5s, module import ~2.5s). launchd KeepAlive restarts reset all of it, and
# an occasional user is cold nearly every time - so that first-request cost is
# the experience most testers actually get.
#
# This runs one throwaway retrieval in a background thread at startup to pay it
# before anyone asks. It CANNOT change any answer: it calls the same retrieve()
# every request calls, keeps nothing, and only populates process-level caches
# that would otherwise be filled by the first real query.
STARTUP_WARMUP = os.environ.get("RAG_STARTUP_WARMUP", "1") == "1"
WARMUP_QUERY = "What are the rules of assessment?"


# Readiness is observable rather than inferred from "HTTP responds": the server
# answers immediately while the retrieval stack is still loading, so uptime
# alone says nothing about whether the next question will be fast. A FAILED
# warmup previously only printed to a log nobody reads.
WARMUP_STATE = {"status": "starting", "seconds": None, "error": None}


def _warmup() -> None:
    """Pre-load the retrieval stack. Never raises into the server: a failed
    warmup must degrade to today's lazy behaviour, not stop the app booting."""
    from src import rag

    started = time.time()
    WARMUP_STATE["status"] = "warming"
    try:
        rag.retrieve(WARMUP_QUERY, [])
    except Exception as exc:  # noqa: BLE001 - warmup is best-effort by design
        WARMUP_STATE.update(status="failed", seconds=round(time.time() - started, 1),
                            error=repr(exc)[:300])
        print(f"[warmup] failed after {time.time() - started:.1f}s: {exc!r}", flush=True)
        return
    WARMUP_STATE.update(status="ready", seconds=round(time.time() - started, 1))
    print(f"[warmup] retrieval stack ready in {time.time() - started:.1f}s", flush=True)


@app.on_event("startup")
def _start_warmup() -> None:
    if not STARTUP_WARMUP:
        return
    # daemon: never hold up shutdown, and never block the health check - the
    # server must answer immediately even while the warmup is still running.
    threading.Thread(target=_warmup, name="retrieval-warmup", daemon=True).start()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class NewConversation(BaseModel):
    title: str | None = None


class NewMessage(BaseModel):
    content: str
    # Answer detail level, chosen in Settings and sent with the message rather
    # than stored server-side - there is no user account to store it against.
    # Unknown values fall back to default in rag.system_prompt_for().
    detail: str = "default"


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


class SourceLookup(BaseModel):
    urls: list[str]
    question: str | None = None


@app.get("/")
def index():
    """The redesigned UI is now the default (2026-08-11). The previous one is
    still served at /classic rather than deleted: it is the interface that has
    actually been used daily, and a one-word URL change is a faster way back
    than a git checkout if something here turns out to be wrong."""
    return FileResponse(STATIC_DIR / "preview.html")


@app.get("/classic")
def classic():
    """The previous UI, kept as a short-term fallback while the new page is
    unproven in daily use.

    NOT a peer: the detail-level control, the staleness marker and the source
    modal exist only on the page at /. Falling back here means losing features,
    not switching to an equivalent - which is why this is temporary. Two
    front-ends on one backend is a tax that compounds, and this one has been
    diverging since 2026-08-11. Delete it once a normal working week has passed
    without anyone reaching for it.

    The landing-page drafts that lived at /drafts were removed the same day:
    the design review ended when draft 6 was built. They remain in git history.
    """
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/sources")
def api_sources(payload: SourceLookup):
    """Document metadata + a matching passage for the cited URLs, for the source
    modal. Read-only; see ingest.passages_for_documents for what the passage is
    and, importantly, what it is not (it is not the generator's exact context)."""
    urls = [u for u in payload.urls if u][:8]  # bounded: one embed + one query per url
    if not urls:
        return []
    return ingest.passages_for_documents(urls, payload.question or "")


@app.get("/api/config")
def api_config():
    """What the server is actually running. `degraded` means it wanted the
    cloud generator and could not reach it, so answers come from the weaker
    local model - the UI surfaces that rather than leaving it in a log."""
    return {
        "degraded": os.environ.get("RAG_DEGRADED") == "1",
        "generator": os.environ.get("GENERATOR_PROVIDER") or "local",
        "warmup": WARMUP_STATE,
    }


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
    answer_text, sources, retrieval_query, ranked_top_urls = rag_answer(
        payload.content, history_for_prompt, summary, detail=payload.detail
    )

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


@app.post("/api/conversations/{conversation_id}/messages/stream")
def api_post_message_stream(conversation_id: str, payload: NewMessage):
    """Server-sent-events variant of the endpoint above. Same work, same
    storage, same response fields - the only difference is that answer text
    reaches the client as it is generated rather than after it is complete.

    The non-streaming endpoint stays: the eval harness uses it, and a client
    that cannot do SSE still works. Both call the same rag.answer().

    NOTE what streaming can and cannot do here. It removes the ~7s the user
    spent watching nothing while the generator wrote; it CANNOT remove the
    retrieval that precedes the first token, because there is nothing to show
    until the context exists. The `stage` events carry that phase instead.
    """
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

    q: "queue.Queue[tuple[str, object]]" = queue.Queue()

    def work() -> None:
        try:
            q.put(("stage", "retrieving"))
            result = rag_answer(
                payload.content, history_for_prompt, summary, detail=payload.detail,
                on_token=lambda t: q.put(("token", t)),
            )
            answer_text, sources, retrieval_query, ranked_top_urls = result
            # Store from the RETURNED text, never from the concatenated tokens:
            # a provider that ignores on_token still returns a complete answer,
            # and what is stored must equal what answer() produced either way.
            memory.add_message(conversation_id, "assistant", answer_text)
            if is_first_message:
                _retitle(conversation_id, payload.content)
            q.put(("done", {
                "answer": answer_text,
                "sources": sources,
                "retrieval_query": retrieval_query,
                "ranked_top_urls": ranked_top_urls,
                "title_pending": is_first_message and GENERATED_TITLES,
            }))
        except Exception as exc:  # noqa: BLE001 - must reach the client as an event
            q.put(("error", str(exc)))
        finally:
            q.put(("__end__", None))

    threading.Thread(target=work, name="answer-stream", daemon=True).start()

    def events():
        while True:
            kind, data = q.get()
            if kind == "__end__":
                return
            yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # nginx and friends buffer SSE by default, which would defeat the point
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class StalenessQuery(BaseModel):
    urls: list[str] = []
    since: float = 0.0


@app.post("/api/staleness")
def api_staleness(payload: StalenessQuery):
    """Which of these documents changed AFTER the given timestamp.

    Lets a stored answer be marked when the policy it cited has since been
    rewritten - the case that actually matters for a policy tool, because the
    answer can be perfectly faithful to a rule that no longer applies. Needs no
    per-user state: the conversation already records what was cited and when.

    Read-only, and silent about documents it does not know: an unknown URL is
    reported as not-stale rather than as changed, because claiming a false
    change would train people to ignore the marker.
    """
    import json as _json
    from pathlib import Path as _Path

    manifest_path = _Path("data/manifest.json")
    if not manifest_path.is_file() or not payload.urls:
        return {"stale": [], "checked": 0}
    docs = _json.loads(manifest_path.read_text()).get("documents", {})
    stale = []
    for u in payload.urls:
        rec = docs.get(u)
        if not rec:
            continue
        changed = rec.get("content_changed_at")
        if changed and payload.since and changed > payload.since:
            stale.append({"url": u, "title": rec.get("title") or u.rsplit("/", 1)[-1],
                          "changed_at": changed})
    return {"stale": stale, "checked": len(payload.urls)}


@app.post("/api/feedback")
def api_feedback(fb: Feedback):
    if fb.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    feedback_store.record_feedback(fb.model_dump())
    return {"ok": True}
