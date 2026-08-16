"""Research and evaluation pages, kept OUT of the product app.

Split from src/app.py 2026-08-13. Pure move - the routes, their paths and their
behaviour are unchanged.

These pages edit or inspect EVALUATION data: the judge-calibration scoring page
and the three reference-review pages. They were reachable from the product's
Settings menu, which meant a trial user clicking around could open "Fix 8
defective items" and start editing the test set. Separating them means the
router can simply not be mounted on a deployment that has testers on it.

Mounted by src/app.py today, so nothing changes until someone chooses not to.
"""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

router = APIRouter()

def _save_versioned(path: Path, payload) -> dict:
    """Write research data WITHOUT destroying what is already there.

    These files hold human annotation - 30 blind judgements that took real
    effort and are the ground truth a paper would rest on. Every one of these
    endpoints used to do a bare `write_text`: full truncate-and-replace, no
    auth, no backup. A POST of `[]` erased them, and this project has already
    lost its conversation history twice.

    Each save goes to a timestamped file and `path` is repointed at it, so
    every prior version survives and the newest is always where readers expect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    versioned = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
    versioned.write_text(json.dumps(payload, indent=1))
    path.write_text(json.dumps(payload, indent=1))
    kept = sorted(path.parent.glob(f"{path.stem}.*{path.suffix}"))
    return {"path": str(path), "version": str(versioned), "versions_kept": len(kept)}



@router.get("/reference-fix")
def reference_fix_page():
    """Correct the reference answers judged wrong. These sit in the main
    40-question set, so every judge-scored comparison on it has included them."""
    page = STATIC_DIR / "reference_fix.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="page not built")
    return FileResponse(page)


@router.post("/api/reference-fix")
def api_reference_fix(fixes: list[dict]):
    """Saves the corrections. Deliberately does NOT write them into the question
    files - applying edits to eval data is a separate, reviewable step, and an
    endpoint that rewrites the test sets from a browser is how test data gets
    quietly changed."""
    saved = _save_versioned(Path("eval/reference_fixes.json"), fixes)
    acted = sum(1 for f in fixes if f.get("action") in ("rewrite", "drop"))
    return {"ok": True, "done": acted, "total": len(fixes), **saved}


@router.get("/provenance-review")
def provenance_review_page():
    """Repoint test items whose gold document Essex has superseded. Those items
    score correct retrieval as a miss, so they are worse than useless."""
    page = STATIC_DIR / "provenance_review.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="page not built")
    return FileResponse(page)


@router.post("/api/provenance-review")
def api_provenance_review(choices: list[dict]):
    saved = _save_versioned(Path("eval/provenance_review_choices.json"), choices)
    done = sum(1 for c in choices if c.get("choice"))
    return {"ok": True, "done": done, "total": len(choices), **saved}


@router.get("/reference-random")
def reference_random_page():
    """RANDOM sample of references, to estimate how common bad ones are. The
    /reference-review set was chosen for maximum human/judge disagreement, so
    its 78%-wrong rate says nothing about the corpus. This one can."""
    page = STATIC_DIR / "reference_random.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="page not built")
    return FileResponse(page)


@router.post("/api/reference-random")
def api_reference_random(verdicts: list[dict]):
    saved = _save_versioned(Path("eval/reference_random_verdicts.json"), verdicts)
    done = sum(1 for v in verdicts if v.get("verdict"))
    return {"ok": True, "done": done, "total": len(verdicts), **saved}


@router.get("/reference-review")
def reference_review_page():
    """Second-stage review: for the answers where the human and the judge
    disagreed most, is the REFERENCE right? Round 42 found the judge scores
    agreement-with-reference, so a suspect reference is the likeliest
    explanation for a large gap."""
    page = STATIC_DIR / "reference_review.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="reference review page not built")
    return FileResponse(page)


@router.post("/api/reference-review")
def api_reference_review(verdicts: list[dict]):
    saved = _save_versioned(Path("eval/reference_review_verdicts.json"), verdicts)
    done = sum(1 for v in verdicts if v.get("verdict"))
    return {"ok": True, "done": done, "total": len(verdicts), **saved}


@router.get("/calibration")
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


@router.post("/api/calibration")
def api_calibration(scores: list[dict]):
    """Save human scores straight to disk, so there is no download or
    copy-paste step to fail. Overwrites: the page always posts the full set,
    including unscored items as null, so a partial pass is still a complete
    record of what was decided so far."""
    saved = _save_versioned(Path("eval/judge_calibration_scores.json"), scores)
    done = sum(1 for s in scores if s.get("human_score") is not None)
    return {"ok": True, "scored": done, "total": len(scores), **saved}
