#!/usr/bin/env python3
"""CLI: start the web app.

Run `python run_server.py` and open http://localhost:8000.

Binds to 127.0.0.1 (local-only) by DEFAULT. To reach it from your other
devices over a private Tailscale network, bind to all interfaces so the
tailnet can connect:

    HOST=0.0.0.0 python run_server.py

then open http://<this-mac's-tailscale-name-or-100.x.y.z>:8000 from a device
logged into the same tailnet. Keep the Mac awake while serving (it is the
compute): `caffeinate -s` in another terminal, or disable sleep in Settings.
HOST/PORT are env-overridable; the default stays localhost so nothing is
exposed unless you opt in.
"""

import os

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.app:app", host=host, port=port, reload=False)
