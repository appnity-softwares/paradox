
"""FastAPI application composition root.

All endpoints live in ``dashboard/routers/*``; engines, agents and request
models live in ``dashboard/state.py``. This module only wires them together.
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

PROTOTYPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROTOTYPE_ROOT)

from dashboard import state  # noqa: F401,E402  (engines/agents initialise here)
from dashboard.state import DATA_MANAGER, ORCHESTRATOR  # noqa: F401,E402  (public re-exports)
from dashboard.routers import agents as agents_router  # noqa: E402
from dashboard.routers import backtest as backtest_router  # noqa: E402
from dashboard.routers import data as data_router  # noqa: E402
from dashboard.routers import execution as execution_router  # noqa: E402
from dashboard.routers import mt5 as mt5_router  # noqa: E402
from dashboard.routers import research as research_router  # noqa: E402
from dashboard.routers import platform as platform_router  # noqa: E402
from dashboard.routers import charts as charts_router  # noqa: E402

app = FastAPI(title="AI Quant Terminal", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8123", "http://localhost:8123"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router.router)
app.include_router(mt5_router.router)
app.include_router(backtest_router.router)
app.include_router(agents_router.router)
app.include_router(execution_router.router)
app.include_router(research_router.router)
app.include_router(platform_router.router)
app.include_router(charts_router.router)


PAGE_FILES = {"terminal": "terminal.html", "client": "client.html",
              "legacy": "index.html"}


@app.get("/")
def index():
    """Default landing: whichever page the admin panel selects."""
    from dashboard.admin_config import load_config
    page = load_config()["default_page"]
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     PAGE_FILES[page]))


@app.get("/vendor/lightweight-charts.js")
def vendor_lwcharts():
    """Vendored TradingView Lightweight Charts (Apache-2.0)."""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "vendor", "lightweight-charts.js"),
                        media_type="application/javascript")


@app.get("/client")
def client_portal():
    """Simple client-facing portal (also selectable as the landing page)."""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "client.html"))


@app.get("/admin")
def admin():
    """Admin panel: UI routing, AI provider, platform controls."""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "admin.html"))


@app.get("/console")
def console():
    """Part 14.1 console: honest labels, kill switch always visible,
    live controls locked until sign-off."""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "console.html"))


@app.get("/legacy")
def legacy():
    """Pre-platform prototype UI (mock data, MT5 bridge demo).
    Kept for reference only — not the product."""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "index.html"))
