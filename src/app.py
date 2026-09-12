"""FastAPI app: conversation + chat endpoints, serves the single-page UI."""

import hashlib
import hmac
import json
import os
import queue
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import feedback as feedback_store
from src import ingest
from src import llm
from src import memory
from src import telemetry
from src.rag import answer as rag_answer
from src.rag import GENERATED_TITLES, generate_title

app = FastAPI(title="Essex Policies & Rules of Assessment Assistant")

# Research/eval pages live in their own router so a tester-facing deployment
# can leave them unmounted. Mounted here today: pure move, no behaviour change.
from src.research_routes import router as _research_router  # noqa: E402
app.include_router(_research_router)

# ── access control ───────────────────────────────────────────────────────────
# One shared password for the whole site, checked on every request. NOT user
# accounts: the per-person separation is still the name in X-User, which is a
# label rather than a credential. This closes the different hole - that anyone
# who can reach the server reads everything - which matters the moment this is
# reachable by more than one person.
#
# Set RAG_ACCESS_PASSWORD to enable. Unset (the default) leaves the server open,
# which is correct for a single-user machine and is why this is opt-in rather
# than a hardcoded secret nobody can change.
ACCESS_PASSWORD = os.environ.get("RAG_ACCESS_PASSWORD", "")
# Only the login page and the favicon. /static/ was exempt here and should not
# have been: the review and dashboard pages are served from it, so /feedback
# would have been protected while /static/feedback.html still returned 200 (the
# shell only - its data comes from /api/feedback, which 401s - but the same held
# for guide.html, provenance_review.html and the rest). Nothing needs the
# exemption: the login page is entirely inline HTML/CSS and pulls no assets, and
# the app's CSS/JS load after sign-in, when the browser sends the cookie with
# subresource requests anyway. Verified both directions after narrowing.
_OPEN_PATHS = ("/login", "/favicon")


@app.middleware("http")
async def require_password(request, call_next):
    if not ACCESS_PASSWORD:
        return await call_next(request)
    path = request.url.path
    if any(path.startswith(p) for p in _OPEN_PATHS):
        return await call_next(request)
    # constant-time compare so the cookie cannot be guessed a character at a time
    cookie = request.cookies.get("rag_access", "")
    if hmac.compare_digest(cookie, _access_token()):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"detail": "not authorised"}, status_code=401)
    return RedirectResponse("/login", status_code=302)


def _access_token() -> str:
    """Derived from the password, so the cookie never contains it."""
    return hashlib.sha256(("rag-access:" + ACCESS_PASSWORD).encode()).hexdigest()


@app.get("/login")
def login_page():
    return HTMLResponse(
        '<!doctype html><meta charset="utf-8"><title>Sign in</title>'
        '<style>body{font:16px/1.5 -apple-system,sans-serif;background:#faf9fb;'
        'display:flex;align-items:center;justify-content:center;height:100vh;margin:0}'
        'form{background:#fff;border:1px solid #e5e1e8;border-radius:10px;padding:26px 28px;'
        'max-width:340px}h1{margin:0 0 6px;font-size:18px;color:#622567}'
        'p{margin:0 0 16px;font-size:13.5px;color:#6b6472}'
        'input{width:100%;padding:9px 11px;border:1px solid #e5e1e8;border-radius:7px;'
        'font:inherit;margin-bottom:11px}button{width:100%;background:#622567;color:#fff;'
        'border:0;border-radius:7px;padding:10px;font:inherit;font-weight:600;cursor:pointer}'
        '</style><form method="post" action="/login">'
        '<h1>Essex Policy Assistant</h1>'
        '<p>Enter the shared password you were given.</p>'
        '<input type="password" name="password" autofocus placeholder="Password">'
        '<button type="submit">Continue</button></form>')


# Failed sign-ins per client IP. The password is a short human-memorable phrase
# shared with colleagues, and without this an attacker who can reach the server
# gets unlimited guesses at it - which is the whole ballgame for a shared
# secret. Tailscale-only reachability keeps the risk low today; this is what
# stops that being the ONLY thing standing in the way.
#
# In-memory on purpose: the window is minutes, the server restarts rarely, and
# persisting it would hand an attacker a way to lock the real user out by
# filling a file. A restart clearing the counters is an acceptable loss.
_LOGIN_FAILURES: dict = {}
LOGIN_MAX_FAILURES = 8        # generous: a human mistyping a 2-word phrase
LOGIN_WINDOW_S = 900          # 15 minutes


def _login_blocked(ip: str) -> int:
    """Seconds to wait, or 0 to allow. Prunes as it reads, so the dict cannot
    grow without bound from scanning traffic."""
    now = time.time()
    tries = [t for t in _LOGIN_FAILURES.get(ip, []) if now - t < LOGIN_WINDOW_S]
    if tries:
        _LOGIN_FAILURES[ip] = tries
    else:
        _LOGIN_FAILURES.pop(ip, None)
    if len(tries) >= LOGIN_MAX_FAILURES:
        return int(LOGIN_WINDOW_S - (now - tries[0])) + 1
    return 0


def _login_failed(ip: str) -> None:
    _LOGIN_FAILURES.setdefault(ip, []).append(time.time())


@app.post("/login")
async def login_submit(request: Request):
    # Parsed by hand rather than with request.form(), which needs
    # python-multipart - a dependency this project does not have and does not
    # need for one field. A urlencoded body is two lines to parse.
    ip = (request.client.host if request.client else "?") or "?"
    wait = _login_blocked(ip)
    if wait:
        # 429 BEFORE comparing, so a blocked client learns nothing about the
        # password from timing or response shape.
        return HTMLResponse(
            '<!doctype html><meta charset="utf-8"><title>Too many attempts</title>'
            '<body style="font:16px/1.5 -apple-system,sans-serif;background:#faf9fb;'
            'display:flex;align-items:center;justify-content:center;height:100vh;margin:0">'
            f'<p style="max-width:340px">Too many failed sign-ins. Try again in '
            f'{max(1, wait // 60)} minute(s).</p>', status_code=429)

    raw = (await request.body()).decode("utf-8", "replace")
    submitted = parse_qs(raw).get("password", [""])[0]
    if hmac.compare_digest(submitted, ACCESS_PASSWORD):
        _LOGIN_FAILURES.pop(ip, None)   # a success clears the client's record
        r = RedirectResponse("/", status_code=302)
        # session cookie: no expiry, so it dies with the browser
        r.set_cookie("rag_access", _access_token(), httponly=True, samesite="lax")
        return r
    _login_failed(ip)
    return RedirectResponse("/login", status_code=302)



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


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    # StaticFiles sets no Cache-Control at all, so browsers fall back to
    # heuristic caching and can keep serving a pre-edit app.js/app.css for a
    # long time with no re-check - happened 2026-09-04, one tab showed stale
    # JS, another stale CSS, from the same deploy. no-cache forces a
    # conditional GET (cheap - 304 off the existing ETag) before ever reusing
    # a cached copy, so an edit is visible on the next load, not the next
    # hard refresh.
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

# Locally-ingested documents (ingest_local.py) are keyed by the URL they are
# served from, so the source modal's "open" link resolves to the real file the
# answer was drawn from. Crawled documents keep their essex.ac.uk URLs and
# never reach this mount. Created on demand: a corpus with no local documents
# has no such directory, and mounting a missing one raises at import time.
LOCAL_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "local_documents"
LOCAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/documents", StaticFiles(directory=LOCAL_DOCS_DIR), name="documents")


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
def _failure_reason(exc: Exception) -> str:
    """A machine-readable failure category for the `status` column.

    Every failure was stored as the single string "generation_failed" - 28 rows
    of it - so a quota exhaustion, a timeout, a network drop and a retrieval
    error were indistinguishable in the data. They need different fixes, and
    /health cannot chart what it cannot tell apart.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if "daily quota" in text:
        return "quota_exhausted"
    if "429" in text or ("rate" in text and "limit" in text):
        return "rate_limited"
    if "529" in text or "overloaded" in text:
        return "overloaded"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "network" in text or "dns" in text or "resolve" in text:
        return "network"
    return "generation_failed"


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
    # Out of credits is NOT a retry-able fault and NOT a bad key, but it
    # previously fell through to the generic "something went wrong ... please
    # try again" - the same words the user sees for a transient blip, so an
    # account that simply needs topping up looked like an intermittent bug and
    # was retried indefinitely (2026-08-29, seen in production). The API says
    # "credit balance is too low"; say that, and say who can fix it.
    if "credit balance" in low or "billing" in low or "quota" in low:
        return ("The assistant's API account has run out of credit, so it cannot "
                "write answers until that is topped up. Retrying will not help - "
                "this needs whoever manages the account.")
    if "401" in text or "403" in text or "api_key" in low or "authentication" in low:
        return ("The assistant is not configured with a valid API key, so it cannot "
                "write an answer. This needs an administrator, not a retry.")
    return ("Something went wrong while writing the answer. It has been logged. "
            "Please try again, and mention this if it keeps happening.")


def _owner(x_user: str | None) -> str:
    """Who is asking. A name the browser sends, not a credential - see the
    OWNER note in src/memory.py.

    Blank lands in OWNER_UNNAMED, NOT the legacy owner. It used to be the
    latter, which put anyone who signed in and skipped the name prompt straight
    into the first user's conversation history with full read and delete
    access - invisible while the server was single-user, and the default path
    the moment a shared password was handed out.
    """
    name = (x_user or "").strip()[:60]
    return name or memory.OWNER_UNNAMED


@app.get("/")
def index():
    """The single UI. The previous interface lived at /classic until
    2026-08-13, when it was removed - two front-ends on one backend is a tax
    that compounds, and it had been diverging since the day it was demoted. It
    remains in git history if anyone ever wants it back."""
    return FileResponse(STATIC_DIR / "preview.html")


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


@app.get("/insights")
def insights_page():
    """What the system is doing - every answer, not the rated 11%."""
    page = STATIC_DIR / "insights.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="page not built")
    return FileResponse(page)


@app.get("/health")
def health_page():
    """Speed, spend and failures. Separate from /feedback deliberately: latency
    percentiles and thumbs-down tags do not belong in one visual conversation,
    and they are drawn from datasets two orders of magnitude apart in size."""
    page = STATIC_DIR / "health.html"
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


@app.get("/api/feedback/context")
def api_feedback_context():
    """Counts the feedback log cannot see, so the dashboard can say what share
    of answers was rated (ratings are self-selected) and surface the failures
    nobody could rate - see memory.usage_stats()."""
    return memory.usage_stats()


class ClientEvent(BaseModel):
    kind: str
    detail: dict = {}


@app.post("/api/event")
def api_event(ev: ClientEvent):
    """One client-side signal: a copy click, a source opened.

    Explicit ratings cover 11% of answers and come from whoever remembered to
    click a thumb. These cover every answer and need nobody to remember - but
    they are WEAKER evidence, not stronger: a copy means "I am using this", not
    "this is correct", and someone can copy a confidently wrong answer. Worth
    having alongside ratings, never worth plotting on the same axis as one.

    Unknown kinds are dropped rather than stored, so a typo in a fetch() call
    cannot quietly become a category on a chart.
    """
    stored = telemetry.log_event(ev.kind, ev.detail)
    return {"stored": stored}


@app.get("/api/insights")
def api_insights(limit: int = 5000):
    """Per-answer telemetry for /insights - every answer, not the rated 11%."""
    return {"answers": memory.answer_telemetry(limit=limit),
            "events": telemetry.load_events()}


@app.get("/api/alerts")
def api_alerts():
    """Current monitor state, written by run_monitor.py.

    Includes when the monitor last RAN, not only what it found. A monitoring
    system that has quietly died is indistinguishable from one reporting
    all-clear, and that is the failure mode most worth designing against - so
    the page can say "the monitor has not checked in for N hours" rather than
    showing a reassuring green panel produced by nobody.
    """
    path = Path("data/alerts.json")
    # History is read alongside current state: a banner is transient and easily
    # missed, so the page must be able to answer "did anything fire overnight?"
    # and not only "is anything wrong this minute".
    history = []
    hist_path = Path("data/alert_history.jsonl")
    if hist_path.is_file():
        try:
            lines = hist_path.read_text().splitlines()[-200:]
            history = [json.loads(l) for l in lines if l.strip()]
        except Exception:  # noqa: BLE001
            history = []
    if not path.is_file():
        return {"alerts": [], "generated_at": None, "never_run": True, "history": history}
    try:
        data = json.loads(path.read_text())
        data["history"] = history
        return data
    except Exception:  # noqa: BLE001
        return {"alerts": [], "generated_at": None, "unreadable": True, "history": history}


@app.get("/api/health/stats")
def api_health_stats():
    """Latency, spend and failures for /health.

    Stage timings come from data/latency.jsonl (2,673 rows and counting);
    tokens and cost come from message meta. Kept as two sources rather than
    merged, because they are written by different code paths at different
    times and pretending otherwise would invent joins that do not exist.
    """
    return {
        "stages": _latency_rows(),
        "answers": memory.answer_telemetry(limit=5000),
        "failures": memory.failure_breakdown(),
        "free_tier_daily_tokens": telemetry.FREE_TIER_DAILY_TOKENS,
        "pricing": telemetry.PRICING,
    }


def _latency_rows(limit: int = 20000) -> list[dict]:
    """Stage timings, newest-last. Best-effort: an unreadable telemetry file
    must degrade the chart, never the endpoint."""
    path = os.environ.get("RAG_TIMING_PATH", "data/latency.jsonl")
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("seconds") is not None:
                    rows.append({"ts": r.get("ts"), "stage": r.get("stage"),
                                 "seconds": r.get("seconds")})
    except FileNotFoundError:
        return []
    except Exception:  # noqa: BLE001
        return rows[-limit:]
    return rows[-limit:]


class ReplayRequest(BaseModel):
    question: str
    # The stored answer and sources, so the server can diff rather than making
    # the client send back what it thinks they were.
    previous_answer: str = ""
    previous_sources: list[str] = []


@app.post("/api/feedback/replay")
def api_feedback_replay(payload: ReplayRequest):
    """Re-ask a rated question against TODAY's index and diff the result.

    Half the thumbs-down in the log predate the plagiarism retrieval fix, the
    partner-exclusion work and two re-ingests, so some are already fixed and the
    log cannot show which - it records what the system said in July, forever.
    This is the project's own method (a thumbs-down is a hypothesis, replayed
    against live retrieval) made clickable.

    Deliberately standalone: history=[] and no summary, so this replays the
    QUESTION, not the conversation. For an original follow-up turn that makes
    the replay a different question from the one that was rated - the client
    warns about this and offers the stored retrieval_query, which is the
    contextualizer's already-resolved standalone form, instead.

    This SPENDS a generation per click (production generator), which is why it
    is one explicit button per row rather than a batch 'replay everything'.
    """
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")
    t0 = time.time()
    # NO fallback for replays. The Sonnet fallback exists so a person waiting on
    # a real question gets an answer instead of a 503, and ~26x the cost is
    # plainly worth that. A replay is not that: it is a diagnostic clicked row
    # by row, so an exhausted Groq quota should stop it rather than run up
    # credit one click at a time. Failing is also the more USEFUL outcome here -
    # a silent Sonnet answer would attribute the replay's verdict to a model
    # that is not the one serving production.
    try:
        with llm.no_fallback():
            answer_text, sources, retrieval_query, ranked_top_urls = rag_answer(q, history=[])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(f"Replay did not run: the {llm.GENERATOR_PROVIDER or 'local'} generator "
                    f"failed and replays deliberately do not fall back to a paid model. "
                    f"({type(exc).__name__}: {str(exc)[:200]})"),
        )
    before = {_doc_name(u) for u in (payload.previous_sources or [])}
    after = {_doc_name(u) for u in sources}
    return {
        "question": q,
        "answer": answer_text,
        "sources": sources,
        "retrieval_query": retrieval_query,
        "ranked_top_urls": ranked_top_urls,
        "seconds": round(time.time() - t0, 1),
        "provenance": provenance(),
        # Set arithmetic on filenames, not URLs: the same document at a new URL
        # after a re-ingest is the same document to a reader asking "did the
        # right thing come back this time".
        "sources_gained": sorted(after - before),
        "sources_lost": sorted(before - after),
        "sources_same": before == after and bool(before),
        "answer_identical": answer_text.strip() == (payload.previous_answer or "").strip(),
    }


def _doc_name(url: str) -> str:
    return str(url).rstrip("/").rsplit("/", 1)[-1]


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
        # What ACTUALLY produced the answer, not what was configured to. A
        # fallback hop (see llm.GENERATOR_FALLBACK) would otherwise be recorded
        # under the primary's name, making the stored provenance and the
        # feedback dashboard's per-generator panel quietly wrong - which defeats
        # the one question provenance exists to answer.
        "generator": (llm.LAST_GENERATOR.get("model")
                      or os.environ.get("GENERATOR_MODEL")
                      or ("claude-sonnet-5" if os.environ.get("GENERATOR_PROVIDER") == "anthropic"
                          else "gemma3:12b")),
        **({"fell_back_from": llm.LAST_GENERATOR["fell_back_from"]}
           if llm.LAST_GENERATOR.get("fell_back_from") else {}),
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
    memory.delete_conversation(conversation_id, owner=_owner(x_user))
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
    _t_turn = time.time()
    try:
        answer_text, sources, retrieval_query, ranked_top_urls = rag_answer(
            payload.content, history_for_prompt, summary, detail=payload.detail,
            partner_mode=payload.partner_mode,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[answer] {type(exc).__name__}: {exc}", flush=True)
        # The question was committed before generation was attempted, so a
        # failure here would otherwise leave it in history with no answer and
        # nothing to explain why (26 such orphans accumulated before this).
        memory.mark_last_message_failed(conversation_id, _failure_reason(exc))
        raise HTTPException(status_code=503, detail=_friendly_error(exc))

    memory.add_message(conversation_id, "assistant", answer_text,
                       meta={"provenance": provenance(), "sources": sources,
                             "usage": telemetry.answer_record(
                                 answer_text=answer_text, sources=sources,
                                 ranked_top_urls=ranked_top_urls,
                                 history=history_for_prompt,
                                 retrieval_query=retrieval_query,
                                 question=payload.content,
                                 seconds=time.time() - _t_turn)})

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

    # Time to FIRST token, which is the latency a reader of a streamed answer
    # actually experiences. answer_total (~9s) measures something else: the
    # whole turn including the generation the user watches arrive. Neither
    # substitutes for the other and only one of them was recorded.
    ttft: dict = {}

    def work() -> None:
        _t_turn = time.time()

        def _on_token(t):
            if not ttft:
                ttft["seconds"] = round(time.time() - _t_turn, 3)
            q.put(("token", t))

        try:
            q.put(("stage", "retrieving"))
            result = rag_answer(
                payload.content, history_for_prompt, summary, detail=payload.detail,
                partner_mode=payload.partner_mode,
                on_token=_on_token,
            )
            answer_text, sources, retrieval_query, ranked_top_urls = result
            _usage = telemetry.answer_record(
                answer_text=answer_text, sources=sources,
                ranked_top_urls=ranked_top_urls, history=history_for_prompt,
                retrieval_query=retrieval_query, question=payload.content,
                seconds=time.time() - _t_turn)
            # Absent when the provider ignored on_token (only the Anthropic path
            # streams), so it is omitted rather than recorded as 0 - a
            # non-streaming turn has no time-to-first-token, and a zero would
            # drag every percentile down while looking like a real measurement.
            if ttft:
                _usage["ttft_seconds"] = ttft["seconds"]
            # Store from the RETURNED text, never from the concatenated tokens:
            # a provider that ignores on_token still returns a complete answer,
            # and what is stored must equal what answer() produced either way.
            memory.add_message(conversation_id, "assistant", answer_text,
                               meta={"provenance": provenance(), "sources": sources,
                                     "usage": _usage})
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
            memory.mark_last_message_failed(conversation_id, _failure_reason(exc))  # see the non-streaming path
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
