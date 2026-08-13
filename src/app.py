"""FastAPI app: conversation + chat endpoints, serves the single-page UI."""

import json
import os
import queue
import threading
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
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
_GIT_REV = None


def _warmup() -> None:
    """Pre-load the retrieval stack. Never raises into the server: a failed
    warmup must degrade to today's lazy behaviour, not stop the app booting."""
    from src import rag

    started = time.time()
    WARMUP_STATE["status"] = "warming"
    try:
        rag.retrieve(WARMUP_QUERY, [])
        # retrieve() with EMPTY history never touches the follow-up path, so
        # the first follow-up after a restart still paid for building the
        # identity anchor index - a glob and json.loads over ~1,188 files.
        # Built directly rather than by warming with a fake history, because
        # that would run the contextualizer, which is a paid cloud call.
        rag._identity_anchor_index()
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
    # Partner-institution handling, chosen in Settings. "exclude" (default)
    # keeps Essex answers free of partner documents - measured at 0 of 157
    # ordinary queries serving one. "boost" recovers questions that name a
    # partner COURSE without its college, at the cost of partner documents
    # appearing on 29% of ordinary questions (Round 35).
    partner_mode: str = "exclude"


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


# Failures users can act on. The raw exception is an API error object -
# "anthropic generator HTTP 404: {"type":"error"...}" was going straight into
# the chat, which is the debug-log-as-user-surface failure CLAUDE.md names,
# surviving in the one path nobody read. The detail is still logged; only the
# user-facing sentence changes.
def _friendly_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "429" in text or "rate" in low and "limit" in low:
        return ("The assistant is being rate-limited right now. Wait a few seconds "
                "and ask again - your question was not lost.")
    if "529" in text or "overloaded" in low:
        return ("The assistant's model is overloaded at the moment. This usually "
                "clears within a minute; please try again.")
    if "timeout" in low or "timed out" in low:
        return ("That took too long and was stopped. Long or very broad questions "
                "are the usual cause - try asking something more specific.")
    if "connection" in low or "network" in low or "dns" in low or "resolve" in low:
        return ("I could not reach the language model - this machine may have lost "
                "its network connection. The policy documents are local, so this is "
                "not a problem with the documents themselves.")
    if "401" in text or "403" in text or "api_key" in low or "authentication" in low:
        return ("The assistant is not configured with a valid API key, so it cannot "
                "write an answer. This needs an administrator, not a retry.")
    return ("Something went wrong while writing the answer. It has been logged. "
            "Please try again, and mention this if it keeps happening.")


def _owner(x_user: str | None) -> str:
    """Who is asking. A name the browser sends, not a credential - see the
    OWNER note in src/memory.py. Blank falls back to the legacy owner so a
    client that never set one still sees a coherent history rather than an
    empty app."""
    name = (x_user or "").strip()[:60]
    return name or memory.OWNER_LEGACY


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


@app.get("/reference-fix")
def reference_fix_page():
    """Correct the reference answers judged wrong. These sit in the main
    40-question set, so every judge-scored comparison on it has included them."""
    page = STATIC_DIR / "reference_fix.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="page not built")
    return FileResponse(page)


@app.post("/api/reference-fix")
def api_reference_fix(fixes: list[dict]):
    """Saves the corrections. Deliberately does NOT write them into the question
    files - applying edits to eval data is a separate, reviewable step, and an
    endpoint that rewrites the test sets from a browser is how test data gets
    quietly changed."""
    out = Path("eval/reference_fixes.json")
    out.write_text(json.dumps(fixes, indent=1))
    acted = sum(1 for f in fixes if f.get("action") in ("rewrite", "drop"))
    return {"ok": True, "done": acted, "total": len(fixes), "path": str(out)}


@app.get("/reference-random")
def reference_random_page():
    """RANDOM sample of references, to estimate how common bad ones are. The
    /reference-review set was chosen for maximum human/judge disagreement, so
    its 78%-wrong rate says nothing about the corpus. This one can."""
    page = STATIC_DIR / "reference_random.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="page not built")
    return FileResponse(page)


@app.post("/api/reference-random")
def api_reference_random(verdicts: list[dict]):
    out = Path("eval/reference_random_verdicts.json")
    out.write_text(json.dumps(verdicts, indent=1))
    done = sum(1 for v in verdicts if v.get("verdict"))
    return {"ok": True, "done": done, "total": len(verdicts), "path": str(out)}


@app.get("/reference-review")
def reference_review_page():
    """Second-stage review: for the answers where the human and the judge
    disagreed most, is the REFERENCE right? Round 42 found the judge scores
    agreement-with-reference, so a suspect reference is the likeliest
    explanation for a large gap."""
    page = STATIC_DIR / "reference_review.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="reference review page not built")
    return FileResponse(page)


@app.post("/api/reference-review")
def api_reference_review(verdicts: list[dict]):
    out = Path("eval/reference_review_verdicts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdicts, indent=1))
    done = sum(1 for v in verdicts if v.get("verdict"))
    return {"ok": True, "done": done, "total": len(verdicts), "path": str(out)}


@app.get("/guide")
def guide_page():
    """What testers should know before they start: what it is, what it is not,
    the Essex/Partner switch, and what makes feedback useful. Most unusable
    trial feedback comes from people not knowing what they were testing."""
    page = STATIC_DIR / "guide.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="guide not built")
    return FileResponse(page)


@app.get("/feedback")
def feedback_page():
    """Read the ratings. Feedback has been write-only: 38 records in a JSONL
    file readable by running a script in a terminal. Once real users are on
    this, their feedback IS the evaluation - and a signal nobody looks at is
    not a signal."""
    page = STATIC_DIR / "feedback.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="page not built")
    return FileResponse(page)


@app.get("/api/feedback")
def api_feedback_list(limit: int = 200):
    """Newest first. Read-only; returns what was recorded, including the
    server-derived question/answer and the provenance of the answer rated."""
    rows = feedback_store.load_feedback()
    rows = sorted(rows, key=lambda r: r.get("timestamp") or "", reverse=True)
    return rows[:limit]


@app.get("/calibration")
def calibration_page():
    """Judge-calibration scoring page (eval tool, not part of the product).

    Served over HTTP rather than opened as a file because `localStorage` throws
    on file:// in Safari, which silently killed the page's whole script. Over
    http:// it works, so progress survives a reload.
    """
    page = STATIC_DIR / "judge_calibration.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="calibration page not built")
    return FileResponse(page)


@app.post("/api/calibration")
def api_calibration(scores: list[dict]):
    """Save human scores straight to disk, so there is no download or
    copy-paste step to fail. Overwrites: the page always posts the full set,
    including unscored items as null, so a partial pass is still a complete
    record of what was decided so far."""
    out = Path("eval/judge_calibration_scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=1))
    done = sum(1 for s in scores if s.get("human_score") is not None)
    return {"ok": True, "scored": done, "total": len(scores), "path": str(out)}


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
        "provenance": provenance(),
    }


def provenance() -> dict:
    """What produced an answer: corpus version, code revision, and the models.
    For a POLICY tool this matters as much as the citation - six months on,
    "why did this say 40?" is unanswerable without knowing which corpus and
    which generator produced it. Cheap to record, impossible to reconstruct
    later."""
    global _GIT_REV
    if _GIT_REV is None:
        try:
            import subprocess
            _GIT_REV = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                text=True, timeout=5, cwd=Path(__file__).resolve().parent.parent,
            ).stdout.strip() or "unknown"
        except Exception:
            _GIT_REV = "unknown"
    try:
        corpus = ingest.read_corpus_version()
    except Exception:
        corpus = None
    return {
        "corpus_version": corpus,
        "code_revision": _GIT_REV,
        "generator": os.environ.get("GENERATOR_MODEL")
                     or ("claude-sonnet-5" if os.environ.get("GENERATOR_PROVIDER") == "anthropic"
                         else "gemma3:12b"),
        "contextualizer": os.environ.get("ANTHROPIC_CONTEXTUALIZE_MODEL")
                          if os.environ.get("CONTEXTUALIZE_PROVIDER") == "anthropic"
                          else "qwen2.5:7b-instruct",
    }


@app.get("/api/conversations")
def api_list_conversations(x_user: str | None = Header(default=None)):
    return memory.list_conversations(owner=_owner(x_user))


@app.post("/api/conversations")
def api_create_conversation(payload: NewConversation, x_user: str | None = Header(default=None)):
    title = payload.title or "New conversation"
    conv_id = memory.create_conversation(title, owner=_owner(x_user))
    return {"id": conv_id, "title": title}


@app.get("/api/conversations/{conversation_id}/messages")
def api_get_messages(conversation_id: str, x_user: str | None = Header(default=None)):
    if not memory.conversation_exists(conversation_id, owner=_owner(x_user)):
        raise HTTPException(status_code=404, detail="conversation not found")
    return memory.get_messages(conversation_id)


@app.delete("/api/conversations/{conversation_id}")
def api_delete_conversation(conversation_id: str, x_user: str | None = Header(default=None)):
    if not memory.conversation_exists(conversation_id, owner=_owner(x_user)):
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
def api_post_message(conversation_id: str, payload: NewMessage, background: BackgroundTasks,
                     x_user: str | None = Header(default=None)):
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")
    if not memory.conversation_exists(conversation_id, owner=_owner(x_user)):
        raise HTTPException(status_code=404, detail="conversation not found")

    summary, history = memory.get_conversation_context(conversation_id)
    is_first_message = not summary and not history

    memory.add_message(conversation_id, "user", payload.content)
    if is_first_message:
        memory.update_title(conversation_id, payload.content[:60])

    history_for_prompt = [{"role": m["role"], "content": m["content"]} for m in history]
    try:
        answer_text, sources, retrieval_query, ranked_top_urls = rag_answer(
            payload.content, history_for_prompt, summary, detail=payload.detail,
            partner_mode=payload.partner_mode,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[answer] {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(status_code=503, detail=_friendly_error(exc))

    memory.add_message(conversation_id, "assistant", answer_text,
                       meta={"provenance": provenance(), "sources": sources})

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
        "provenance": provenance(),
    }


@app.post("/api/conversations/{conversation_id}/messages/stream")
def api_post_message_stream(conversation_id: str, payload: NewMessage,
                            x_user: str | None = Header(default=None)):
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
    if not memory.conversation_exists(conversation_id, owner=_owner(x_user)):
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
                partner_mode=payload.partner_mode,
                on_token=lambda t: q.put(("token", t)),
            )
            answer_text, sources, retrieval_query, ranked_top_urls = result
            # Store from the RETURNED text, never from the concatenated tokens:
            # a provider that ignores on_token still returns a complete answer,
            # and what is stored must equal what answer() produced either way.
            memory.add_message(conversation_id, "assistant", answer_text,
                               meta={"provenance": provenance(), "sources": sources})
            if is_first_message:
                _retitle(conversation_id, payload.content)
            q.put(("done", {
                "answer": answer_text,
                "sources": sources,
                "retrieval_query": retrieval_query,
                "ranked_top_urls": ranked_top_urls,
                "title_pending": is_first_message and GENERATED_TITLES,
                "provenance": provenance(),
            }))
        except Exception as exc:  # noqa: BLE001 - must reach the client as an event
            print(f"[answer-stream] {type(exc).__name__}: {exc}", flush=True)
            q.put(("error", _friendly_error(exc)))
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
def api_feedback(fb: Feedback, x_user: str | None = Header(default=None)):
    """Records a rating. The QUESTION and ANSWER are taken from the server's own
    stored conversation, not from the client's copy.

    This whole project's method is feedback replay: a thumbs-down is re-run
    against live retrieval to diagnose it. That is only sound if the recorded
    question and answer are what the system actually produced. A client-supplied
    copy can drift - a stale tab, an edited field, a retry - and the resulting
    diagnosis would be of something that never happened.

    The client's values are kept as `client_*` rather than discarded, so a
    mismatch is visible instead of silently overwritten.
    """
    if fb.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    record = fb.model_dump()
    record["provenance"] = provenance()
    record["owner"] = _owner(x_user)

    if fb.conversation_id and memory.conversation_exists(fb.conversation_id):
        msgs = memory.get_messages(fb.conversation_id)
        last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), None)
        last_assistant = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), None)
        if last_user is not None and last_assistant is not None:
            if (last_user, last_assistant) != (fb.question, fb.answer):
                record["client_question"] = fb.question
                record["client_answer"] = fb.answer
                record["client_server_mismatch"] = True
            record["question"] = last_user
            record["answer"] = last_assistant
            record["source"] = "server"
    record.setdefault("source", "client")   # no conversation_id: nothing to verify against
    feedback_store.record_feedback(record)
    return {"ok": True}
