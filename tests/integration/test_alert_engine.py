"""Integration tests for the IncidentCoordinator alerting service.

Tests use the session-scoped asyncpg pool from conftest.py and
a fully wired IncidentCoordinator instance.  They verify that:

- Critical predictions create persisted incidents.
- Incidents can be acknowledged.
- Cooldown prevents duplicates within the cooldown window.
- Healthy predictions produce no incidents.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import asyncpg
import pytest
import pytest_asyncio

from faultscope.alerting.config import AlertingConfig
from faultscope.alerting.coordinator import IncidentCoordinator
from faultscope.common.kafka.schemas import RulPrediction


def _critical_prediction(machine_id: str = "ENG_INT_001") -> RulPrediction:
    """Return a RulPrediction that will fire the rul_critical rule."""
    return RulPrediction(
        machine_id=machine_id,
        predicted_at=datetime.now(tz=UTC),
        rul_cycles=3.0,  # < 10 → rul_critical fires
        rul_hours=6.0,
        rul_lower_bound=1.0,
        rul_upper_bound=5.0,
        health_label="imminent_failure",
        health_probabilities={
            "healthy": 0.0,
            "degrading": 0.0,
            "critical": 0.1,
            "imminent_failure": 0.9,
        },
        anomaly_score=0.92,
        confidence=0.95,
        rul_model_version="v1.0.0",
        health_model_version="v1.0.0",
    )


def _healthy_prediction(machine_id: str = "ENG_INT_002") -> RulPrediction:
    """Return a RulPrediction that should fire no rules."""
    return RulPrediction(
        machine_id=machine_id,
        predicted_at=datetime.now(tz=UTC),
        rul_cycles=200.0,
        rul_hours=400.0,
        rul_lower_bound=190.0,
        rul_upper_bound=210.0,
        health_label="healthy",
        health_probabilities={
            "healthy": 0.92,
            "degrading": 0.05,
            "critical": 0.02,
            "imminent_failure": 0.01,
        },
        anomaly_score=0.02,
        confidence=0.97,
        rul_model_version="v1.0.0",
        health_model_version="v1.0.0",
    )


def _make_alerting_config() -> AlertingConfig:
    """Build an AlertingConfig that skips env file loading."""
    return AlertingConfig.model_construct(
        host="0.0.0.0",  # noqa: S104
        port=8001,
        kafka_bootstrap_servers="localhost:9092",
        kafka_consumer_group="test-group",
        topic_rul_predictions="test.predictions",
        topic_incidents="test.incidents",
        db_host="localhost",
        db_port=5432,
        db_name="faultscope",
        db_user="faultscope",
        db_password=MagicMock(get_secret_value=lambda: "testpassword"),
        aggregation_window_s=300,
        email_smtp_host="",
        email_recipients=[],
        slack_webhook_url=MagicMock(get_secret_value=lambda: ""),
        webhook_url="",
        log_level="ERROR",
        log_format="console",
        otel_enabled=False,
    )


@pytest.mark.integration
@pytest.mark.asyncio
class TestAlertEngineIntegration:
    """Integration tests for the alerting IncidentCoordinator."""

    @pytest_asyncio.fixture
    async def coordinator(
        self,
        db_pool: asyncpg.Pool,  # type: ignore[type-arg]
    ) -> IncidentCoordinator:
        """Create a coordinator wired to the real test database.

        Ensures the alerting.incidents table exists using the public
        schema (TimescaleDB not required for integration tests).
        """
        # Ensure the alerting schema + incidents table exist.
        async with db_pool.acquire() as conn:
            await conn.execute("CREATE SCHEMA IF NOT EXISTS alerting;")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerting.incidents (
                    incident_id      TEXT        PRIMARY KEY,
                    rule_id          TEXT        NOT NULL,
                    machine_id       TEXT        NOT NULL,
                    severity         TEXT        NOT NULL,
                    title            TEXT        NOT NULL,
                    status           TEXT        NOT NULL DEFAULT 'open',
                    details          JSONB       NOT NULL DEFAULT '{}',
                    triggered_at     TIMESTAMPTZ NOT NULL,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    acknowledged_at  TIMESTAMPTZ,
                    acknowledged_by  TEXT,
                    closed_at        TIMESTAMPTZ,
                    resolution_note  TEXT
                );
                """
            )

        config = _make_alerting_config()
        return IncidentCoordinator(
            config=config,
            db_pool=db_pool,
            notifiers=[],  # no external notifications in tests
        )

    async def test_critical_prediction_creates_incident(
        self,
        coordinator: IncidentCoordinator,
        db_pool: asyncpg.Pool,  # type: ignore[type-arg]
    ) -> None:
        """A critical RUL prediction must create at least one incident row."""
        machine_id = f"ENG_CRIT_{uuid.uuid4().hex[:6]}"
        prediction = _critical_prediction(machine_id=machine_id)

        incident_ids = await coordinator.process_prediction(prediction)

        assert len(incident_ids) >= 1, (
            "Expected at least one incident for critical prediction"
        )

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT incident_id, rule_id, status "
                "FROM alerting.incidents "
                "WHERE machine_id = $1",
                machine_id,
            )

        assert len(rows) >= 1
        rule_ids = {row["rule_id"] for row in rows}
        assert "rul_critical" in rule_ids

    async def test_incident_can_be_acknowledged(
        self,
        coordinator: IncidentCoordinator,
        db_pool: asyncpg.Pool,  # type: ignore[type-arg]
    ) -> None:
        """Row status must be 'acknowledged' after acknowledge_incident()."""
        machine_id = f"ENG_ACK_{uuid.uuid4().hex[:6]}"
        prediction = _critical_prediction(machine_id=machine_id)

        incident_ids = await coordinator.process_prediction(prediction)
        assert len(incident_ids) >= 1

        first_id = incident_ids[0]
        await coordinator.acknowledge_incident(
            first_id, acknowledged_by="test-operator"
        )

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT status, acknowledged_by "
                "FROM alerting.incidents "
                "WHERE incident_id = $1",
                first_id,
            )

        assert row is not None
        assert row["status"] == "acknowledged"
        assert row["acknowledged_by"] == "test-operator"

    async def test_cooldown_prevents_duplicate_incidents(
        self,
        coordinator: IncidentCoordinator,
        db_pool: asyncpg.Pool,  # type: ignore[type-arg]
    ) -> None:
        """Two immediate predictions for the same machine should produce
        fewer incidents on the second call due to rule cooldown."""
        machine_id = f"ENG_COOL_{uuid.uuid4().hex[:6]}"
        prediction = _critical_prediction(machine_id=machine_id)

        ids_first = await coordinator.process_prediction(prediction)
        ids_second = await coordinator.process_prediction(prediction)

        # The first call fires rules; the second must not fire the same
        # rules again (they are within cooldown).
        # Allow 0 on second call for all rules that have a cooldown.
        assert len(ids_first) >= 1
        # Cooldown prevents firing again immediately.
        assert len(ids_second) == 0, (
            "Expected cooldown to suppress second identical prediction"
        )

    async def test_healthy_prediction_creates_no_incidents(
        self,
        coordinator: IncidentCoordinator,
        db_pool: asyncpg.Pool,  # type: ignore[type-arg]
    ) -> None:
        """A healthy prediction with high RUL must produce no incidents."""
        machine_id = f"ENG_HLTH_{uuid.uuid4().hex[:6]}"
        prediction = _healthy_prediction(machine_id=machine_id)

        incident_ids = await coordinator.process_prediction(prediction)

        assert incident_ids == [], (
            f"Expected no incidents for healthy prediction, "
            f"got: {incident_ids}"
        )

    async def test_incident_status_starts_as_open(
        self,
        coordinator: IncidentCoordinator,
        db_pool: asyncpg.Pool,  # type: ignore[type-arg]
    ) -> None:
        """Newly created incidents must have status='open'."""
        machine_id = f"ENG_OPEN_{uuid.uuid4().hex[:6]}"
        prediction = _critical_prediction(machine_id=machine_id)

        incident_ids = await coordinator.process_prediction(prediction)
        assert len(incident_ids) >= 1

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT status FROM alerting.incidents "
                "WHERE incident_id = ANY($1::text[])",
                incident_ids,
            )

        for row in rows:
            assert row["status"] == "open"

    async def test_incident_details_contain_rul_cycles(
        self,
        coordinator: IncidentCoordinator,
        db_pool: asyncpg.Pool,  # type: ignore[type-arg]
    ) -> None:
        """Persisted incident details must contain rul_cycles snapshot."""
        machine_id = f"ENG_DET_{uuid.uuid4().hex[:6]}"
        prediction = _critical_prediction(machine_id=machine_id)
        prediction = prediction.model_copy(update={"rul_cycles": 3.0})

        incident_ids = await coordinator.process_prediction(prediction)
        assert len(incident_ids) >= 1

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT details FROM alerting.incidents "
                "WHERE incident_id = $1",
                incident_ids[0],
            )

        assert row is not None
        details = (
            row["details"]
            if isinstance(row["details"], dict)
            else json.loads(row["details"])
        )
        assert "rul_cycles" in details
        assert details["rul_cycles"] == pytest.approx(3.0)
