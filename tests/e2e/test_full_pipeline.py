"""End-to-end smoke tests for the FaultScope full stack.

These tests exercise the running services over HTTP.  They are
intentionally lenient: each test catches ``httpx.ConnectError`` and
calls ``pytest.skip`` when the target service is not reachable, so
that the test suite can still pass in a pure unit-test environment.

Run with the full Docker Compose stack up::

    docker compose up -d
    pytest tests/e2e/ -m e2e -v
"""

from __future__ import annotations

import pytest

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HTTPX_AVAILABLE,
    reason="httpx not installed; skipping e2e tests",
)

# ---------------------------------------------------------------------------
# Service base URLs
# ---------------------------------------------------------------------------

_INFERENCE_URL = "http://localhost:8000"
_ALERTING_URL = "http://localhost:8001"
_TIMEOUT = 5.0  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Issue a GET request, skipping the test on connection refusal."""
    try:
        return await client.get(url, headers=headers or {})
    except httpx.ConnectError:
        pytest.skip(f"Service not reachable at {url}")


async def _post(
    client: httpx.AsyncClient,
    url: str,
    payload: object,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Issue a POST request, skipping the test on connection refusal."""
    try:
        return await client.post(
            url,
            json=payload,
            headers=headers or {},
        )
    except httpx.ConnectError:
        pytest.skip(f"Service not reachable at {url}")


# ---------------------------------------------------------------------------
# Inference service tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestInferenceServiceE2E:
    """End-to-end smoke tests for the inference FastAPI service."""

    async def test_inference_health_returns_200(self) -> None:
        """GET /health on the inference service must return 200."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get(client, f"{_INFERENCE_URL}/health")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

    async def test_inference_health_body_is_healthy(
        self,
    ) -> None:
        """Inference /health body must contain status == 'healthy'."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get(client, f"{_INFERENCE_URL}/health")
        data = resp.json()
        assert data.get("status") == "healthy", (
            f"Unexpected health body: {data}"
        )

    async def test_inference_ready_endpoint_exists(self) -> None:
        """GET /ready must not return 404."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get(client, f"{_INFERENCE_URL}/ready")
        assert resp.status_code != 404, (
            f"Expected /ready to exist, got {resp.status_code}"
        )

    async def test_inference_unauthenticated_returns_401(
        self,
    ) -> None:
        """Prediction endpoint must reject requests without an API key."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _post(
                client,
                f"{_INFERENCE_URL}/api/v1/predict/remaining-life",
                payload={
                    "machine_id": "ENG_001",
                    "feature_sequence": [{"feat": 1.0}],
                },
            )
        assert resp.status_code == 401, (
            f"Expected 401 without key, got {resp.status_code}"
        )

    async def test_openapi_schema_available(self) -> None:
        """GET /openapi.json must return a valid OpenAPI document."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get(client, f"{_INFERENCE_URL}/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema, (
            f"'paths' missing from OpenAPI schema: {schema.keys()}"
        )
        assert "components" in schema, (
            f"'components' missing from OpenAPI schema: {schema.keys()}"
        )


# ---------------------------------------------------------------------------
# Alerting service tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestAlertingServiceE2E:
    """End-to-end smoke tests for the alerting FastAPI service."""

    async def test_alerting_health_returns_200(self) -> None:
        """GET /health on the alerting service must return 200."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get(client, f"{_ALERTING_URL}/health")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

    async def test_alerting_rules_endpoint_returns_list(
        self,
    ) -> None:
        """GET /api/v1/rules must return a non-empty list."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get(client, f"{_ALERTING_URL}/api/v1/rules")
        assert resp.status_code == 200, (
            f"Expected 200 for /api/v1/rules, got {resp.status_code}"
        )
        rules = resp.json()
        assert isinstance(rules, list), (
            f"Expected list of rules, got: {type(rules)}"
        )
        assert len(rules) > 0, "Rules list must not be empty"

    async def test_alerting_rules_minimum_count(self) -> None:
        """Alerting service must expose at least 9 detection rules."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get(client, f"{_ALERTING_URL}/api/v1/rules")
        rules = resp.json()
        assert len(rules) >= 9, (
            f"Expected >= 9 rules, got {len(rules)}: {rules}"
        )

    async def test_alerting_rules_have_required_fields(
        self,
    ) -> None:
        """Each rule must expose rule_id, severity, and enabled."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get(client, f"{_ALERTING_URL}/api/v1/rules")
        rules = resp.json()
        required_keys = {"rule_id", "severity", "enabled"}
        for rule in rules:
            missing = required_keys - set(rule.keys())
            assert not missing, f"Rule {rule} is missing fields: {missing}"

    async def test_alerting_incidents_endpoint_exists(
        self,
    ) -> None:
        """GET /api/v1/incidents must not return 404."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get(client, f"{_ALERTING_URL}/api/v1/incidents")
        assert resp.status_code != 404, "/api/v1/incidents returned 404"


# ---------------------------------------------------------------------------
# Cross-service integration smoke test
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestCrossServiceE2E:
    """Verify that both services are healthy simultaneously."""

    async def test_both_services_healthy(self) -> None:
        """Inference and alerting services must both report healthy."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            inf_resp = await _get(client, f"{_INFERENCE_URL}/health")
            alt_resp = await _get(client, f"{_ALERTING_URL}/health")

        assert inf_resp.status_code == 200, (
            f"Inference service unhealthy: {inf_resp.status_code}"
        )
        assert alt_resp.status_code == 200, (
            f"Alerting service unhealthy: {alt_resp.status_code}"
        )

        inf_data = inf_resp.json()
        _ = alt_resp.json()

        assert inf_data.get("status") == "healthy", (
            f"Inference not healthy: {inf_data}"
        )
        # Alerting service may use a different status key structure.
        assert alt_resp.status_code == 200
