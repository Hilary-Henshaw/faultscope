"""FastAPI middleware for the FaultScope inference service.

Provides three middleware components:

``ApiKeyMiddleware``
    Validates the ``X-API-Key`` header on every request except
    ``/health``, ``/ready``, ``/docs``, ``/redoc``, and ``/openapi.json``.
    Returns HTTP 401 on missing or wrong key, 403 for explicitly forbidden
    paths.

``RequestIdMiddleware``
    Injects a UUID v4 ``X-Request-ID`` into each request (generating one
    if the caller did not provide it) and echoes it in the response.

``configure_rate_limiting``
    Attaches ``slowapi`` with the specified per-minute limit keyed on the
    client IP address.

Usage (in ``create_app``)::

    app.add_middleware(ApiKeyMiddleware, api_key=cfg.api_key)
    app.add_middleware(RequestIdMiddleware)
    configure_rate_limiting(app, limit_per_minute=100)
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from faultscope.common.logging import get_logger

_log = get_logger(__name__)

# Paths that do not require authentication.
_AUTH_EXEMPT_PREFIXES: frozenset[str] = frozenset(
    {"/health", "/ready", "/docs", "/redoc", "/openapi.json", "/metrics"}
)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Validates ``X-API-Key`` header on every non-exempt request.

    Exempt paths: ``/health``, ``/ready``, ``/docs``, ``/redoc``,
    ``/openapi.json``, ``/metrics``.

    Parameters
    ----------
    app:
        The ASGI application to wrap.
    api_key:
        Expected API key value (plain string, not hashed).
    """

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Validate the API key or pass through exempt paths."""

        path = request.url.path

        # Allow health/docs/metrics without authentication.
        if any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES):
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key", "")
        if not provided_key:
            _log.warning(
                "api_key_missing",
                path=path,
                remote=request.client.host if request.client else "?",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "X-API-Key header is required.",
                    "error": "unauthorized",
                },
            )

        if provided_key != self._api_key:
            _log.warning(
                "api_key_invalid",
                path=path,
                remote=request.client.host if request.client else "?",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid API key.",
                    "error": "unauthorized",
                },
            )

        return await call_next(request)


def configure_rate_limiting(
    app: FastAPI,
    limit_per_minute: int,
) -> None:
    """Attach slowapi rate limiter to the FastAPI application.

    Rate-limits all endpoints by client IP address.  The limit is
    expressed as ``"{limit_per_minute}/minute"``.  When a client
    exceeds the limit, a ``429 Too Many Requests`` response is
    returned automatically.

    Parameters
    ----------
    app:
        The FastAPI application instance.
    limit_per_minute:
        Number of requests allowed per minute per IP address.
    """
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{limit_per_minute}/minute"],
    )
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,  # type: ignore[arg-type]
        _rate_limit_exceeded_handler,  # type: ignore[arg-type]
    )
    _log.info(
        "rate_limiting_configured",
        limit_per_minute=limit_per_minute,
    )


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Injects a ``X-Request-ID`` header into each request and response.

    If the incoming request already carries an ``X-Request-ID`` header
    its value is reused (tracing propagation).  Otherwise a new UUID v4
    is generated.

    The request ID is also bound to the structlog context so that all
    log events emitted during the request automatically include it.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Generate or propagate the request ID."""
        import structlog

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        structlog.contextvars.unbind_contextvars("request_id")
        return response
