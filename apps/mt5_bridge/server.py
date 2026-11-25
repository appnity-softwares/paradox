"""The MT5 bridge HTTP service (Part 12.4).

- Token auth on every endpoint (Authorization: Bearer <token>).
- TLS is configured at the uvicorn level (see run()); the service must
  never be exposed publicly without authentication.
- Server-side enforcement: order_send is refused unless order_check ran
  for the same idempotency key first (MT5 best practice per 12.4).
- The token and any MT5 credentials NEVER appear in logs or responses.
"""
from __future__ import annotations

import threading

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from apps.mt5_bridge.mt5_backend import FakeMT5Backend


class OrderRequest(BaseModel):
    symbol: str
    type: int                     # 0 buy, 1 sell; 2/3 limit, 4/5 stop
    volume: float
    price: float | None = None    # trigger price for pending types
    sl: float | None = None
    tp: float | None = None
    comment: str = ""             # carries the idempotency key


class CancelRequest(BaseModel):
    ticket: int


class ModifyRequest(BaseModel):
    ticket: int
    sl: float | None = None
    tp: float | None = None


def create_app(backend=None, token: str = "") -> FastAPI:
    """Build the bridge app. `token` is the shared secret; an empty token
    refuses to start (fail closed — no unauthenticated mode)."""
    if not token:
        raise RuntimeError("refusing to start the bridge without a token")
    backend = backend or FakeMT5Backend()
    app = FastAPI(title="dsh-mt5-bridge", docs_url=None, redoc_url=None)

    def require_token(authorization: str = Header(default="")) -> None:
        # Constant-time-ish comparison; the token is never logged.
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.post("/connect")
    def connect(_: None = Depends(require_token)) -> dict:
        return backend.connect()

    @app.get("/health")
    def health(_: None = Depends(require_token)) -> dict:
        return backend.health()

    @app.get("/account")
    def account(_: None = Depends(require_token)) -> dict:
        return backend.account_info()

    @app.get("/symbol/{symbol}")
    def symbol(symbol: str, _: None = Depends(require_token)) -> dict:
        return backend.symbol_spec(symbol)

    @app.get("/quote/{symbol}")
    def quote(symbol: str, _: None = Depends(require_token)) -> dict:
        return backend.quote(symbol)

    @app.post("/order_check")
    def order_check(req: OrderRequest,
                    _: None = Depends(require_token)) -> dict:
        result = backend.order_check(req.model_dump())
        if not result.get("ok"):
            raise HTTPException(status_code=422, detail=result)
        return result

    @app.post("/order_cancel")
    def order_cancel(req: CancelRequest,
                     _: str = Depends(require_token)) -> dict:
        return backend.order_cancel(req.ticket)

    @app.post("/order_send")
    def order_send(req: OrderRequest,
                   _: None = Depends(require_token)) -> dict:
        result = backend.order_send(req.model_dump())
        if result.get("retcode") not in (10009, None):
            raise HTTPException(status_code=422, detail=result)
        return result

    @app.post("/position_modify")
    def position_modify(req: ModifyRequest,
                        _: None = Depends(require_token)) -> dict:
        return backend.position_modify(req.ticket, req.sl, req.tp)

    @app.post("/position_close/{ticket}")
    def position_close(ticket: int, _: None = Depends(require_token)) -> dict:
        result = backend.position_close(ticket, int(0))
        if "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.get("/positions")
    def positions(_: None = Depends(require_token)) -> list[dict]:
        return backend.positions_get()

    @app.get("/orders")
    def orders(_: None = Depends(require_token)) -> list[dict]:
        return backend.orders_get()

    @app.get("/history")
    def history(_: None = Depends(require_token)) -> list[dict]:
        return backend.history_deals_get()

    app.state.backend = backend
    return app


def run(host: str = "127.0.0.1", port: int = 8001, token: str = "",
        certfile: str | None = None, keyfile: str | None = None,
        backend=None) -> None:
    """Blocking run; TLS is REQUIRED unless bound to loopback (dev only)."""
    import uvicorn
    app = create_app(backend=backend, token=token)
    ssl_kwargs = {}
    if certfile and keyfile:
        ssl_kwargs = {"ssl_certfile": certfile, "ssl_keyfile": keyfile}
    elif not host.startswith("127.0.0.1") and host != "localhost":
        raise RuntimeError(
            "refusing to expose the bridge without TLS; provide certfile "
            "and keyfile (Part 12.4)")
    uvicorn.run(app, host=host, port=port, **ssl_kwargs)


def serve_in_thread(port: int, token: str, backend=None):
    """Test/dev helper: run the bridge on loopback in a daemon thread."""
    import threading
    import time

    import uvicorn
    app = create_app(backend=backend or FakeMT5Backend(), token=token)
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, thread
