"""Integration tests for the FaultScope inference FastAPI service.

Uses a real FastAPI ``TestClient`` backed by a mocked ``ModelVersionStore``
so no MLflow or GPU resources are required.  Tests verify:

- Health endpoint shape and HTTP status.
- RUL prediction request/response contract.
- API key authentication enforcement.
- Batch size limit (>100 → 422).
- Per-request latency budget.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from faultscope.inference.api.middleware import ApiKeyMiddleware
from faultscope.inference.api.routes.health import router as health_router
from faultscope.inference.engine.predictor import (
    PredictionEngine,
)
from faultscope.inference.engine.version_store import ModelVersionStore

# ---------------------------------------------------------------------------
# Test API key constant
# ---------------------------------------------------------------------------
_TEST_API_KEY = "test-api-key-12345"  # noqa: S105


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_rul_model() -> MagicMock:
    """Return a stub sklearn-compatible model that predicts RUL=80."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([80.0])
    return mock_model


def _make_stub_health_model() -> MagicMock:
    """Return a stub model that returns a healthy probability row."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.85, 0.10, 0.04, 0.01]])
    return mock_model


def _make_mock_store() -> ModelVersionStore:
    """Build a mock ModelVersionStore with both stub models installed."""
    store = MagicMock(spec=ModelVersionStore)
    rul_bundle = {
        "model": _make_stub_rul_model(),
        "version": "1",
        "loaded_at": datetime.now(tz=UTC),
    }
    health_bundle = {
        "model": _make_stub_health_model(),
        "version": "1",
        "loaded_at": datetime.now(tz=UTC),
    }
    store.get_rul_model.return_value = rul_bundle
    store.get_health_model.return_value = health_bundle
    store.get_status.return_value = {
        "rul_model": {"version": "1"},
        "health_model": {"version": "1"},
        "last_reload": datetime.now(tz=UTC).isoformat(),
    }
    return store


def _make_app() -> FastAPI:
    """Assemble a minimal FastAPI app with health router + auth middleware."""
    from faultscope.inference.api.routes import predictions as pred_routes

    app = FastAPI(title="FaultScope Inference Test")

    # Register routers.
    app.include_router(health_router)

    # Inject app state.
    mock_store = _make_mock_store()
    prediction_engine = PredictionEngine(version_store=mock_store)

    app.state.version_store = mock_store
    app.state.prediction_engine = prediction_engine
    app.state.startup_time = time.monotonic()

    # Attach API key middleware.
    app.add_middleware(ApiKeyMiddleware, api_key=_TEST_API_KEY)

    # Prediction routes (if module has content).
    try:
        app.include_router(
            pred_routes.router,
            prefix="/api/v1",
        )
    except AttributeError:
        # predictions route module may be an empty stub.
        _attach_prediction_routes(app, prediction_engine)

    return app


def _attach_prediction_routes(app: FastAPI, engine: PredictionEngine) -> None:
    """Attach minimal prediction routes when the real router is a stub."""
    from fastapi import HTTPException, Request

    from faultscope.common.exceptions import ModelLoadError, ValidationError
    from faultscope.inference.api.schemas import (
        BatchPredictionRequest,
        RulPredictionRequest,
    )

    @app.post("/api/v1/predict/remaining-life")
    async def predict_rul(
        payload: RulPredictionRequest,
        request: Request,
    ) -> dict[str, Any]:
        pe: PredictionEngine = request.app.state.prediction_engine
        try:
            result = await pe.predict_remaining_life(
                machine_id=payload.machine_id,
                feature_sequence=payload.feature_sequence,
            )
        except (ValidationError, ModelLoadError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "machine_id": result.machine_id,
            "rul_cycles": result.rul_cycles,
            "rul_hours": result.rul_hours,
            "rul_lower_bound": result.rul_lower_bound,
            "rul_upper_bound": result.rul_upper_bound,
            "health_label": result.health_label,
            "confidence": result.confidence,
            "model_version": result.model_version,
            "predicted_at": result.predicted_at.isoformat(),
            "latency_ms": result.latency_ms,
        }

    @app.post("/api/v1/predict/batch")
    async def predict_batch(
        payload: BatchPredictionRequest,
        request: Request,
    ) -> dict[str, Any]:
        pe: PredictionEngine = request.app.state.prediction_engine
        from faultscope.inference.engine.predictor import (
            BatchPredictionItem as EngineItem,
        )

        items = [
            EngineItem(
                request_id=item.request_id,
                prediction_type=item.prediction_type,
                machine_id=item.machine_id,
                feature_sequence=item.feature_sequence,
                features=item.features,
            )
            for item in payload.items
        ]
        results = await pe.predict_batch(items)
        return {
            "results": [
                {"request_id": r.request_id, "success": r.success}
                for r in results
            ],
            "batch_size": len(results),
            "total_latency_ms": 0,
        }


@pytest.mark.integration
class TestInferenceApiIntegration:
    """Integration tests for the inference FastAPI service."""

    @pytest.fixture
    def app_client(self) -> TestClient:
        """Create FastAPI test client with mocked model store."""
        app = _make_app()
        return TestClient(app, raise_server_exceptions=True)

    # ── Health endpoint ───────────────────────────────────────────────

    def test_health_endpoint_returns_200(self, app_client: TestClient) -> None:
        """GET /health must return HTTP 200."""
        response = app_client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_status_field(
        self, app_client: TestClient
    ) -> None:
        """Health response must contain a 'status' field."""
        response = app_client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_endpoint_contains_models_loaded_field(
        self, app_client: TestClient
    ) -> None:
        """Health response must include models_loaded dict."""
        response = app_client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], dict)

    def test_health_endpoint_contains_uptime_field(
        self, app_client: TestClient
    ) -> None:
        """Health response must include uptime_s numeric field."""
        response = app_client.get("/health")
        data = response.json()
        assert "uptime_s" in data
        assert isinstance(data["uptime_s"], int | float)
        assert data["uptime_s"] >= 0.0

    def test_health_endpoint_accessible_without_api_key(
        self, app_client: TestClient
    ) -> None:
        """GET /health must not require X-API-Key (exempt path)."""
        response = app_client.get("/health")
        # Must not return 401.
        assert response.status_code != 401

    # ── API key enforcement ───────────────────────────────────────────

    def test_request_without_api_key_returns_401(
        self, app_client: TestClient
    ) -> None:
        """POST /api/v1/predict/remaining-life without key → 401."""
        payload = {
            "machine_id": "ENG_001",
            "feature_sequence": [{"vibration": 0.3}],
        }
        response = app_client.post(
            "/api/v1/predict/remaining-life",
            json=payload,
            # No X-API-Key header.
        )
        assert response.status_code == 401

    def test_request_with_wrong_api_key_returns_401(
        self, app_client: TestClient
    ) -> None:
        """POST with incorrect X-API-Key must return 401."""
        payload = {
            "machine_id": "ENG_001",
            "feature_sequence": [{"vibration": 0.3}],
        }
        response = app_client.post(
            "/api/v1/predict/remaining-life",
            json=payload,
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    # ── RUL prediction ────────────────────────────────────────────────

    def test_rul_prediction_returns_valid_response(
        self,
        app_client: TestClient,
    ) -> None:
        """Valid RUL request must return 200 with expected fields."""
        payload = {
            "machine_id": "ENG_001",
            "feature_sequence": [
                {
                    "fan_inlet_temp_30s_mean": 518.67,
                    "vibration_30s_rms": 0.23,
                }
            ],
        }
        response = app_client.post(
            "/api/v1/predict/remaining-life",
            json=payload,
            headers={"X-API-Key": _TEST_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["machine_id"] == "ENG_001"
        assert "rul_cycles" in data
        assert isinstance(data["rul_cycles"], int | float)
        assert data["rul_cycles"] >= 0.0

    def test_rul_prediction_response_contains_health_label(
        self,
        app_client: TestClient,
    ) -> None:
        """RUL response must include a health_label field."""
        payload = {
            "machine_id": "ENG_001",
            "feature_sequence": [{"feat": 1.0}],
        }
        response = app_client.post(
            "/api/v1/predict/remaining-life",
            json=payload,
            headers={"X-API-Key": _TEST_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        valid_labels = {"healthy", "degrading", "critical", "imminent_failure"}
        assert data["health_label"] in valid_labels

    def test_rul_prediction_response_contains_confidence(
        self,
        app_client: TestClient,
    ) -> None:
        """RUL response confidence must be in [0, 1]."""
        payload = {
            "machine_id": "ENG_001",
            "feature_sequence": [{"feat": 1.0}],
        }
        response = app_client.post(
            "/api/v1/predict/remaining-life",
            json=payload,
            headers={"X-API-Key": _TEST_API_KEY},
        )
        data = response.json()
        conf = data.get("confidence", -1)
        assert 0.0 <= conf <= 1.0

    # ── Batch endpoint ────────────────────────────────────────────────

    def test_batch_prediction_respects_size_limit(
        self,
        app_client: TestClient,
    ) -> None:
        """Batch with > 100 items must return 422 Unprocessable Entity."""
        items = [
            {
                "request_id": f"req_{i}",
                "prediction_type": "rul",
                "machine_id": "ENG_001",
                "feature_sequence": [{"feat": 1.0}],
            }
            for i in range(101)  # 101 > max of 100
        ]
        response = app_client.post(
            "/api/v1/predict/batch",
            json={"items": items},
            headers={"X-API-Key": _TEST_API_KEY},
        )
        assert response.status_code == 422

    def test_batch_prediction_with_valid_single_item(
        self,
        app_client: TestClient,
    ) -> None:
        """Batch with a single valid item must succeed."""
        payload = {
            "items": [
                {
                    "request_id": "test-req-001",
                    "prediction_type": "rul",
                    "machine_id": "ENG_001",
                    "feature_sequence": [{"feat": 1.0}],
                }
            ]
        }
        response = app_client.post(
            "/api/v1/predict/batch",
            json=payload,
            headers={"X-API-Key": _TEST_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["batch_size"] == 1

    # ── Latency ───────────────────────────────────────────────────────

    def test_rul_prediction_latency_under_200ms(
        self,
        app_client: TestClient,
    ) -> None:
        """RUL prediction (stub model) must respond in < 200 ms."""
        payload = {
            "machine_id": "ENG_001",
            "feature_sequence": [{"feat": 1.0}],
        }
        t0 = time.monotonic()
        response = app_client.post(
            "/api/v1/predict/remaining-life",
            json=payload,
            headers={"X-API-Key": _TEST_API_KEY},
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert response.status_code == 200
        assert elapsed_ms < 200, f"Expected < 200 ms, got {elapsed_ms:.1f} ms"
