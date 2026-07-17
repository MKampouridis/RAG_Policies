#!/usr/bin/env python3
"""CLI: start the web app.

Run `python run_server.py` and open http://localhost:8000. Binds to
127.0.0.1 (local-only) for now; switching to 0.0.0.0 + a reverse proxy is a
one-line change here when it's time to make this reachable from elsewhere."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
