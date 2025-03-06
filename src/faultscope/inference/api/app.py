"""FastAPI application factory for the FaultScope inference service.

``create_app`` builds and fully configures the FastAPI instance.
It is the single entry point used by uvicorn and by the test suite.

Lifecycle
---------
On startup:
    1. Configure structlog.
    2. Optionally enable OpenTelemetry.
    3. Initialise the SQLAlchemy async engine.
    4. Start ``ModelVersionStore`` (initial model load + polling).
    5. Build ``PredictionEngine``.
    6. Start the Kafka ``EventPublisher``.
    7. Bind ``app.state`` with all shared objects.

On shutdown:
    - Stop ``ModelVersionStore`` polling.
    - Stop the Kafka publisher.
    - Dispose the DB engine.

Registered routes
-----------------
    GET  /health
    GET  /ready
    GET  /api/v1/models
    POST /api/v1/models/refresh
    POST /api/v1/predict/remaining-life
    POST /api/v1/predict/health-status
    POST /api/v1/predict/batch
    GET  /metrics   (Prometheus)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from pydantic import ValidationError as PydanticValidationError

from faultscope.common.db.engine import initialize_engine
from faultscope.common.exceptions import (
    ModelLoadError,
)
from faultscope.common.exceptions import (
    ValidationError as FaultScopeValidationError,
)
from faultscope.common.kafka.producer import EventPublisher
from faultscope.common.logging import configure_logging, get_logger
from faultscope.common.telemetry import setup_telemetry
from faultscope.inference.api.middleware import (
    ApiKeyMiddleware,
    RequestIdMiddleware,
    configure_rate_limiting,
)
from faultscope.inference.api.routes import catalog, health, predictions
from faultscope.inference.config import InferenceConfig
from faultscope.inference.engine.predictor import PredictionEngine
from faultscope.inference.engine.version_store import ModelVersionStore

_log = get_logger(__name__)


def create_app(config: InferenceConfig) -> FastAPI:
    """Build and configure the FastAPI application.

    Parameters
    ----------
    config:
        Resolved ``InferenceConfig`` instance.

    Returns
    -------
    FastAPI
        Fully configured application ready to be served by uvicorn.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage startup and shutdown of all shared resources."""
        configure_logging(level="INFO", fmt="json")

        setup_telemetry(
            service_name="faultscope-inference",
            enabled=config.otel_enabled,
            endpoint=None,
        )

        from faultscope.common.config import DatabaseSettings

        db_settings = DatabaseSettings(
            host=config.db_host,
            port=config.db_port,
            name=config.db_name,
            user=config.db_user,
            password=config.db_password,
        )
        initialize_engine(db_settings)
        _log.info("inference_db_engine_initialised")

        version_store = ModelVersionStore(
            mlflow_tracking_uri=config.mlflow_tracking_uri,
            rul_model_name=config.mlflow_rul_model_name,
            health_model_name=config.mlflow_health_model_name,
            reload_interval_s=config.model_reload_interval_s,
        )

        prediction_engine = PredictionEngine(version_store=version_store)

        publisher = EventPublisher(
            bootstrap_servers=config.kafka_bootstrap_servers
        )

        await version_store.start()
        await publisher.start()

        # Bind shared objects to app.state for route handlers.
        app.state.version_store = version_store
        app.state.prediction_engine = prediction_engine
        app.state.publisher = publisher
        app.state.startup_time = time.monotonic()
        app.state.config = config

        _log.info("inference_service_started")
        try:
            yield
        finally:
            _log.info("inference_service_shutting_down")
            await version_store.stop()
            await publisher.stop()
            _log.info("inference_service_stopped")

    app = FastAPI(
        title="FaultScope Inference Service",
        description=(
            "Real-time predictive maintenance inference API. "
            "Provides RUL regression and health classification "
            "endpoints backed by MLflow-managed models."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware (applied in reverse registration order) ────────────
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        ApiKeyMiddleware,
        api_key=config.api_key.get_secret_value(),
    )
    configure_rate_limiting(app, limit_per_minute=config.rate_limit_per_minute)

    # ── Exception handlers ────────────────────────────────────────────
    @app.exception_handler(FaultScopeValidationError)
    async def faultscope_validation_handler(
        request: Request,
        exc: FaultScopeValidationError,
    ) -> JSONResponse:
        _log.warning(
            "api_validation_error",
            path=request.url.path,
            error=str(exc),
            context=exc.context,
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.message,
                "context": exc.context,
                "error": "validation_error",
            },
        )

    @app.exception_handler(PydanticValidationError)
    async def pydantic_validation_handler(
        request: Request,
        exc: PydanticValidationError,
    ) -> JSONResponse:
        _log.warning(
            "api_pydantic_validation_error",
            path=request.url.path,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed.",
                "errors": exc.errors(),
                "error": "validation_error",
            },
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_error_handler(
        request: Request,
        exc: ModelLoadError,
    ) -> JSONResponse:
        _log.error(
            "api_model_load_error",
            path=request.url.path,
            error=str(exc),
            context=exc.context,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": exc.message,
                "context": exc.context,
                "error": "model_unavailable",
            },
        )

    # ── Routers ───────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(catalog.router)
    app.include_router(predictions.router)

    # ── Prometheus metrics ────────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    _log.info(
        "inference_app_created",
        host=config.host,
        port=config.port,
        workers=config.workers,
    )

    return app
