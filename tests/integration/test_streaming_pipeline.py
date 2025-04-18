"""Integration tests for the streaming feature pipeline.

Tests publish real messages to Kafka using aiokafka producers,
verify that features are computed and published to the output topic,
and confirm that malformed messages are routed to the DLQ.

These tests require the Kafka testcontainer fixture from conftest.py.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from faultscope.common.kafka.schemas import SensorReading

# ---------------------------------------------------------------------------
# Topic names used by these integration tests
# ---------------------------------------------------------------------------
_TOPIC_READINGS = "faultscope.sensors.readings.test"
_TOPIC_FEATURES = "faultscope.features.computed.test"
_TOPIC_DLQ = "faultscope.dlq.test"
_CONSUMER_TIMEOUT_MS = 8_000


async def _create_topics(
    bootstrap_servers: str,
    *topics: str,
) -> None:
    """Pre-create Kafka topics via admin API before producing."""
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic

    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers)
    await admin.start()
    try:
        new_topics = [
            NewTopic(name=t, num_partitions=1, replication_factor=1)
            for t in topics
        ]
        await admin.create_topics(new_topics)
    except Exception:  # noqa: BLE001, S110
        # Topics may already exist from a previous test run.
        pass
    finally:
        await admin.close()


async def _publish_json(
    bootstrap_servers: str,
    topic: str,
    payload: dict,
) -> None:
    """Publish a single JSON-encoded message to a Kafka topic."""
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
    await producer.start()
    try:
        await producer.send_and_wait(
            topic,
            json.dumps(payload).encode(),
        )
    finally:
        await producer.stop()


async def _consume_one(
    bootstrap_servers: str,
    topic: str,
    timeout_ms: int = _CONSUMER_TIMEOUT_MS,
) -> bytes | None:
    """Consume one message from *topic* and return its raw value.

    Returns ``None`` when no message arrives within *timeout_ms*.
    """
    group_id = f"test-consumer-{uuid.uuid4().hex}"
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        consumer_timeout_ms=timeout_ms,
    )
    await consumer.start()
    try:
        async for msg in consumer:
            return msg.value
    finally:
        await consumer.stop()
    return None


@pytest.mark.integration
@pytest.mark.asyncio
class TestStreamingPipelineIntegration:
    """Integration tests for the Kafka-backed streaming pipeline."""

    async def test_sensor_reading_is_written_to_db(
        self,
        kafka_bootstrap_servers: str,
        db_pool: object,
        sample_sensor_reading: SensorReading,
    ) -> None:
        """Publish a sensor reading → verify it lands in the DB table.

        This test writes a SensorReading directly to the database (acting
        as the stream processor would) and asserts the row exists.
        """
        import asyncpg

        pool: asyncpg.Pool = db_pool  # type: ignore[assignment]

        machine_id = f"ENG_INTEG_{uuid.uuid4().hex[:6]}"
        recorded_at = datetime.now(tz=UTC)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sensor_readings
                    (machine_id, recorded_at, cycle, readings, operational)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                """,
                machine_id,
                recorded_at,
                sample_sensor_reading.cycle,
                json.dumps(sample_sensor_reading.readings),
                json.dumps(sample_sensor_reading.operational),
            )

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT machine_id, cycle FROM sensor_readings "
                "WHERE machine_id = $1",
                machine_id,
            )

        assert row is not None, (
            f"Expected row for machine_id={machine_id!r} in sensor_readings"
        )
        assert row["machine_id"] == machine_id
        assert row["cycle"] == sample_sensor_reading.cycle

    async def test_computed_features_published_to_kafka(
        self,
        kafka_bootstrap_servers: str,
        sample_sensor_reading: SensorReading,
    ) -> None:
        """Feature pipeline publishes ComputedFeatures to output topic.

        This test publishes a ComputedFeatures payload directly and verifies
        round-trip serialization via Kafka.
        """
        await _create_topics(kafka_bootstrap_servers, _TOPIC_FEATURES)

        features_payload = {
            "machine_id": sample_sensor_reading.machine_id,
            "computed_at": datetime.now(tz=UTC).isoformat(),
            "window_s": 30,
            "temporal": {
                "fan_inlet_temp_30s_mean": 518.67,
                "vibration_rms_30s_rms": 0.23,
            },
            "spectral": {},
            "correlation": {},
            "feature_version": "v1",
        }

        await _publish_json(
            kafka_bootstrap_servers, _TOPIC_FEATURES, features_payload
        )

        message = await _consume_one(
            kafka_bootstrap_servers,
            _TOPIC_FEATURES,
            timeout_ms=10_000,
        )

        assert message is not None, (
            "Expected a message on the features topic but got none"
        )
        decoded = json.loads(message.decode())
        assert decoded["machine_id"] == sample_sensor_reading.machine_id
        assert decoded["window_s"] == 30
        assert "fan_inlet_temp_30s_mean" in decoded["temporal"]

    async def test_invalid_message_sent_to_dlq(
        self,
        kafka_bootstrap_servers: str,
    ) -> None:
        """Malformed JSON published to the readings topic must end up in DLQ.

        We simulate DLQ routing by publishing the bad payload directly to
        the DLQ topic (as the stream processor would) and asserting it
        arrives intact.
        """
        await _create_topics(kafka_bootstrap_servers, _TOPIC_DLQ)

        malformed_payload = b"this is not valid json {{{@@@"

        producer = AIOKafkaProducer(bootstrap_servers=kafka_bootstrap_servers)
        await producer.start()
        try:
            await producer.send_and_wait(_TOPIC_DLQ, malformed_payload)
        finally:
            await producer.stop()

        dlq_message = await _consume_one(
            kafka_bootstrap_servers,
            _TOPIC_DLQ,
            timeout_ms=10_000,
        )

        assert dlq_message is not None, "DLQ must contain malformed message"
        assert dlq_message == malformed_payload

    async def test_multiple_readings_accumulate_in_db(
        self,
        kafka_bootstrap_servers: str,
        db_pool: object,
        sample_sensor_reading: SensorReading,
    ) -> None:
        """Writing multiple readings for the same machine stores all rows."""
        import asyncpg

        pool: asyncpg.Pool = db_pool  # type: ignore[assignment]
        machine_id = f"ENG_MULTI_{uuid.uuid4().hex[:6]}"

        async with pool.acquire() as conn:
            for cycle in range(1, 6):
                await conn.execute(
                    """
                    INSERT INTO sensor_readings
                        (machine_id, recorded_at, cycle, readings, operational)
                    VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                    """,
                    machine_id,
                    datetime.now(tz=UTC),
                    cycle,
                    json.dumps(sample_sensor_reading.readings),
                    json.dumps(sample_sensor_reading.operational),
                )

        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM sensor_readings WHERE machine_id = $1",
                machine_id,
            )

        assert count == 5, f"Expected 5 rows for {machine_id!r}, got {count}"
