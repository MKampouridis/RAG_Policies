#!/usr/bin/env python3
"""Verification ladder: cheapest checks first, broadest coverage first.

WHY THIS SHAPE
Two failures this month were caught by nothing, or by the wrong thing:

  * A refactor left a constant undefined. EVERY answer returned 503. The
    161-query retrieval fingerprint passed. The 118-turn canary passed. Both
    exercise retrieval; the constant is used during answer assembly. Two green
    safety nets on a system that could not answer a question. `pyflakes` finds
    it in ONE SECOND.
  * A `const` used 760 lines before its declaration threw at the top level and
    blanked the page. Every check run at the time - CSS balanced, element
    present in the HTML, correct documents on 217 cases - was true, and none
    tested whether the page EXECUTES.

The lesson, worth stating plainly: **a safety net's coverage is what it
executes, not how many cases it runs.** 161 queries and 118 turns sound
thorough; both exercised one code path.

So: cheap and broad first, expensive and narrow last.

    1  pyflakes            whole Python codebase, ~1s
    2  import the app      catches import cycles and module-scope errors
    3  JS syntax           whole client, ~1s
    4  JS top-level TDZ    the specific blank-page bug
    5  live request        the ONLY check that caught the 503
    6  fingerprint         expensive, narrow - run separately

Usage:
    python verify.py            # steps 1-5 (step 5 needs a running server)
    python verify.py --static   # steps 1-4 only, no server needed
"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
FAIL = []


def step(n: int, name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {n}. {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(f"{n}. {name}: {detail}")


def s1_pyflakes() -> None:
    files = sorted(str(p) for p in (ROOT / "src").glob("*.py"))
    r = subprocess.run([sys.executable, "-m", "pyflakes", *files],
                       capture_output=True, text=True)
    undefined = [l for l in r.stdout.splitlines() if "undefined name" in l]
    other = [l for l in r.stdout.splitlines() if "undefined name" not in l]
    step(1, "pyflakes: no undefined names", not undefined,
         undefined[0] if undefined else f"{len(other)} other messages")


def s2_import() -> None:
    r = subprocess.run([sys.executable, "-c", "import src.app"],
                       capture_output=True, text=True, cwd=ROOT)
    step(2, "app imports cleanly", r.returncode == 0,
         (r.stderr.strip().splitlines() or [""])[-1][:100])


def _js_sources() -> list[pathlib.Path]:
    return sorted((ROOT / "static").glob("*.js"))


def s3_js_syntax() -> None:
    try:
        import esprima
    except ImportError:
        step(3, "JS syntax", False, "esprima not installed (pip install esprima)")
        return
    bad = []
    for p in _js_sources():
        try:
            esprima.parseScript(p.read_text())
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{p.name}: {exc}")
    step(3, "JS parses", not bad, bad[0] if bad else f"{len(_js_sources())} file(s)")


def s4_js_tdz() -> None:
    """Top-level use of a const/let BEFORE its declaration.

    This is the blank-page bug: `let settings` was declared 760 lines below a
    top-level call that read it, which throws ReferenceError and aborts the
    whole script. Unlike `function`, const/let do not hoist. A syntax check
    cannot see this; it is legal JavaScript that always fails.
    """
    try:
        import esprima
    except ImportError:
        step(4, "JS top-level dead zone", False, "esprima not installed")
        return
    problems = []
    for p in _js_sources():
        src = p.read_text()
        try:
            tree = esprima.parseScript(src, {"range": True})
        except Exception:
            continue  # step 3 reports it
        decl_at = {}
        for node in tree.body:
            if getattr(node, "type", "") == "VariableDeclaration" and node.kind in ("const", "let"):
                for d in node.declarations:
                    if getattr(d.id, "name", None):
                        decl_at[d.id.name] = node.range[0]
        # Function bodies, so a top-level CALL can be followed into them. The
        # real blank-page bug was exactly this shape:
        #     function buildComposer() { ... settings ... }   // body reads it
        #     const c = buildComposer();                      // called at top level
        #     let settings = loadSettings();                  // declared 760 lines later
        # A check that skips function bodies misses it - verified: the first
        # version of this check passed the broken file.
        bodies = {}
        for node in tree.body:
            if getattr(node, "type", "") == "FunctionDeclaration" and getattr(node.id, "name", None):
                bodies[node.id.name] = node

        def called_names(node, out):
            """Calls that run SYNCHRONOUSLY at module scope.

            It must not descend into function bodies. A first version did, and
            flagged `menuBtn` on a working page: the name is read inside an
            onclick handler defined above its declaration but only INVOKED on a
            click, long after the whole script has run. Deferred calls are not
            dead-zone reads.
            """
            if isinstance(node, list):
                for x in node:
                    called_names(x, out)
                return
            if not getattr(node, "type", None):
                return
            if node.type in ("FunctionDeclaration", "FunctionExpression",
                             "ArrowFunctionExpression"):
                return
            if node.type == "CallExpression" and getattr(node.callee, "name", None):
                out.add(node.callee.name)
            for key in dir(node):
                if key.startswith("_") or key in ("type", "range"):
                    continue
                called_names(getattr(node, key), out)

        # walk top-level statements, flagging reads of a name declared later
        def walk(node, limit, seen):
            if isinstance(node, list):
                for x in node:
                    walk(x, limit, seen)
                return
            t = getattr(node, "type", None)
            if not t:
                return
            if t == "Identifier":
                n = getattr(node, "name", None)
                if n in decl_at and decl_at[n] > limit:
                    seen.add(n)
                return
            if t in ("FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression"):
                return  # bodies run later, not at module scope
            for key in dir(node):
                if key.startswith("_") or key in ("type", "range"):
                    continue
                walk(getattr(node, key), limit, seen)

        def walk_body(node, limit, seen):
            """Like walk, but DOES descend into function bodies - used when a
            function is invoked at module scope, so its body runs then."""
            if isinstance(node, list):
                for x in node:
                    walk_body(x, limit, seen)
                return
            t_ = getattr(node, "type", None)
            if not t_:
                return
            if t_ == "Identifier":
                n = getattr(node, "name", None)
                if n in decl_at and decl_at[n] > limit:
                    seen.add(n)
                return
            for key in dir(node):
                if key.startswith("_") or key in ("type", "range"):
                    continue
                walk_body(getattr(node, key), limit, seen)

        for node in tree.body:
            if getattr(node, "type", "") in ("FunctionDeclaration",):
                continue
            seen = set()
            walk(node, node.range[0], seen)
            calls = set()
            called_names(node, calls)
            for fn in calls:
                if fn in bodies:
                    walk_body(bodies[fn].body, node.range[0], seen)
            for name in sorted(seen):
                problems.append(
                    f"{p.name}: '{name}' is read at module scope before its declaration")
    step(4, "no top-level use before declaration", not problems,
         problems[0] if problems else "checked const/let at module scope")


def s5_live_request(base: str = "http://127.0.0.1:8000", attempts: int = 2) -> None:
    """The golden request. The ONLY check that caught the 503, and the cheapest
    end-to-end signal available: does the product actually answer?

    ONE retry, and it REPORTS having needed it. Generation is a network call to a
    third party, so a transient upstream blip fails this step - observed once,
    passing on the immediate retry with the same code. A check that cries wolf
    gets ignored, which would cost more than it saves; a check that retries
    SILENTLY hides a real outage as "flaky". So: retry, and say so.
    """
    try:
        import requests
    except ImportError:
        step(5, "live request", False, "requests not available")
        return
    h = {"X-User": "verify"}
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(f"{base}/api/config", headers=h, timeout=10)
            if r.status_code != 200:
                step(5, "live request", False, f"/api/config returned {r.status_code}")
                return
            cid = requests.post(f"{base}/api/conversations", headers=h,
                                json={"title": "__verify__"}, timeout=15).json()["id"]
            try:
                a = requests.post(f"{base}/api/conversations/{cid}/messages", headers=h,
                                  json={"content": "What are the pass marks for a PGT Merit?",
                                        "detail": "default", "partner_mode": "essex_only"},
                                  timeout=300)
                if a.status_code == 200:
                    body = a.json()
                    ok = bool(body.get("answer", "").strip()) and bool(body.get("sources"))
                    note = f"{len(body.get('answer',''))} chars, {len(body.get('sources',[]))} sources"
                    if ok and attempt > 1:
                        note += f"  (needed {attempt - 1} retry — upstream was flaky)"
                    step(5, "live request answers with sources", ok, note)
                    return
                last = f"HTTP {a.status_code}: {a.text[:70]}"
            finally:
                requests.delete(f"{base}/api/conversations/{cid}", headers=h, timeout=15)
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"[:80]
    step(5, "live request", False, f"{last}  (failed {attempts} attempts)")


def main() -> int:
    static_only = "--static" in sys.argv
    print("\n  verification ladder — cheapest and broadest first\n")
    s1_pyflakes()
    s2_import()
    s3_js_syntax()
    s4_js_tdz()
    if not static_only:
        s5_live_request()
    else:
        print("  [skip] 5. live request (--static)")
    print()
    if FAIL:
        print(f"  {len(FAIL)} CHECK(S) FAILED:")
        for f in FAIL:
            print(f"    {f}")
        print()
        return 1
    print("  all checks passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
