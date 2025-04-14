"""Integration test fixtures using testcontainers.

Provides real Kafka and PostgreSQL/TimescaleDB containers scoped to the
test session so that container startup overhead is paid only once.
Bootstrap server URLs and database connection pools are made available
as session-scoped fixtures so integration tests can share them.
"""

from __future__ import annotations

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.kafka import KafkaContainer
from testcontainers.postgres import PostgresContainer

# ---------------------------------------------------------------------------
# Container fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def kafka_container() -> KafkaContainer:  # type: ignore[misc]
    """Start a real Kafka broker for the entire test session."""
    with KafkaContainer("confluentinc/cp-kafka:7.6.0") as container:
        yield container


@pytest.fixture(scope="session")
def postgres_container() -> PostgresContainer:  # type: ignore[misc]
    """Start a real PostgreSQL container for the entire test session.

    Uses the standard postgres image since TimescaleDB requires an extra
    pull; the schema migrations that create hypertables are applied
    via the ``db_pool`` fixture below.
    """
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


# ---------------------------------------------------------------------------
# Derived connection fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def kafka_bootstrap_servers(kafka_container: KafkaContainer) -> str:
    """Return the external bootstrap server address for the test Kafka."""
    return kafka_container.get_bootstrap_server()


@pytest.fixture(scope="session")
def postgres_url(postgres_container: PostgresContainer) -> str:
    """Return the raw psycopg2-style URL from the container."""
    return postgres_container.get_connection_url()


@pytest.fixture(scope="session")
def async_postgres_url(postgres_url: str) -> str:
    """Convert the sync URL to an asyncpg-compatible format."""
    # testcontainers returns a SQLAlchemy-style URL starting with
    # 'postgresql+psycopg2://...' or 'postgresql://...'; asyncpg
    # needs the plain 'postgresql://...' form.
    url = postgres_url.replace("postgresql+psycopg2://", "postgresql://")
    url = url.replace("postgresql://", "postgres://")
    return url


@pytest_asyncio.fixture
async def db_pool(
    async_postgres_url: str,
) -> asyncpg.Pool:  # type: ignore[type-arg]
    """Create a function-scoped asyncpg connection pool.

    Applies a minimal schema so the test database has the tables the
    stream processor and alerting service expect.  Function scope ensures
    the pool is created on the same event loop as the test that uses it,
    avoiding cross-loop errors with asyncpg.
    """
    pool = await asyncpg.create_pool(
        async_postgres_url,
        min_size=2,
        max_size=5,
    )
    assert pool is not None

    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id          BIGSERIAL PRIMARY KEY,
                machine_id  TEXT        NOT NULL,
                recorded_at TIMESTAMPTZ NOT NULL,
                cycle       INTEGER,
                readings    JSONB       NOT NULL DEFAULT '{}',
                operational JSONB       NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS sensor_readings_machine_ts
                ON sensor_readings (machine_id, recorded_at);

            CREATE TABLE IF NOT EXISTS computed_features (
                id              BIGSERIAL PRIMARY KEY,
                machine_id      TEXT        NOT NULL,
                computed_at     TIMESTAMPTZ NOT NULL,
                window_s        INTEGER     NOT NULL,
                temporal        JSONB       NOT NULL DEFAULT '{}',
                spectral        JSONB       NOT NULL DEFAULT '{}',
                correlation     JSONB       NOT NULL DEFAULT '{}',
                feature_version TEXT        NOT NULL DEFAULT 'v1'
            );

            CREATE TABLE IF NOT EXISTS incidents (
                id          BIGSERIAL PRIMARY KEY,
                machine_id  TEXT        NOT NULL,
                rule_id     TEXT        NOT NULL,
                severity    TEXT        NOT NULL,
                title       TEXT        NOT NULL,
                details     JSONB       NOT NULL DEFAULT '{}',
                triggered_at TIMESTAMPTZ NOT NULL,
                acknowledged BOOLEAN    NOT NULL DEFAULT FALSE,
                acknowledged_at TIMESTAMPTZ
            );
            """
        )

    yield pool
    await pool.close()
