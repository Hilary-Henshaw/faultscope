"""Async Kafka producer wrapper for FaultScope services.

Provides ``EventPublisher``, an ``aiokafka``-backed producer that
serialises Pydantic models to JSON bytes and publishes them to Kafka
topics.  Transient network errors are retried automatically using
``tenacity`` with exponential back-off.

Usage::

    async with EventPublisher(bootstrap_servers="localhost:9092") as pub:
        await pub.publish(
            topic="faultscope.sensors.readings",
            payload=reading,
            key=reading.machine_id,
        )
"""

from __future__ import annotations

import asyncio

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from faultscope.common.exceptions import KafkaPublishError
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

# Retry configuration constants.
_MAX_ATTEMPTS: int = 3
_WAIT_MIN_S: float = 0.5
_WAIT_MAX_S: float = 4.0


class EventPublisher:
    """Async Kafka producer with automatic JSON serialisation and retry.

    Wraps ``AIOKafkaProducer`` and adds:

    - Automatic serialisation of Pydantic ``BaseModel`` instances.
    - Per-publish structured logging at ``DEBUG`` level.
    - Transparent retry with exponential back-off (3 attempts).
    - Async context-manager support for clean lifecycle management.

    Parameters
    ----------
    bootstrap_servers:
        Comma-separated list of Kafka broker addresses, e.g.
        ``"broker1:9092,broker2:9092"``.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        """Initialise the publisher.

        The underlying ``AIOKafkaProducer`` is *not* started here;
        call ``start`` (or use as an async context manager) before
        publishing.

        Parameters
        ----------
        bootstrap_servers:
            Comma-separated ``host:port`` broker list.
        """
        self._bootstrap_servers: str = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the underlying Kafka producer.

        Must be called before any ``publish`` calls when *not* using
        the async context manager.

        Raises
        ------
        KafkaPublishError
            If the producer cannot connect to the Kafka cluster.
        """
        _log.info(
            "kafka_producer_starting",
            bootstrap_servers=self._bootstrap_servers,
        )
        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                # Encode string keys to bytes automatically.
                key_serializer=lambda k: (
                    k.encode("utf-8") if k is not None else None
                ),
                # Highest durability: wait for all in-sync replicas.
                acks="all",
                # Enable idempotent delivery to avoid duplicates on
                # retries at the broker level.
                enable_idempotence=True,
                compression_type="gzip",
                request_timeout_ms=30_000,
                retry_backoff_ms=500,
            )
            await self._producer.start()
        except Exception as exc:
            _log.error(
                "kafka_producer_start_failed",
                bootstrap_servers=self._bootstrap_servers,
                error=str(exc),
            )
            raise KafkaPublishError(
                f"Failed to start Kafka producer: {exc}",
                context={
                    "bootstrap_servers": self._bootstrap_servers,
                    "error": str(exc),
                },
            ) from exc

        _log.info(
            "kafka_producer_started",
            bootstrap_servers=self._bootstrap_servers,
        )

    async def stop(self) -> None:
        """Flush pending messages and stop the Kafka producer.

        Safe to call even if ``start`` was never called.
        """
        if self._producer is None:
            return
        _log.info("kafka_producer_stopping")
        try:
            await self._producer.stop()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "kafka_producer_stop_error",
                error=str(exc),
            )
        finally:
            self._producer = None
        _log.info("kafka_producer_stopped")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(
        self,
        topic: str,
        payload: BaseModel,
        key: str | None = None,
    ) -> None:
        """Serialise ``payload`` to JSON bytes and send it to ``topic``.

        The message key is used by Kafka for partition assignment.
        Passing the ``machine_id`` as the key ensures all events for a
        given machine land on the same partition (and therefore are
        consumed in order).

        Retries up to ``_MAX_ATTEMPTS`` times on transient errors with
        exponential back-off.

        Parameters
        ----------
        topic:
            Target Kafka topic name.
        payload:
            Pydantic model instance to serialise.  The ``model_dump_json``
            method is used to produce the UTF-8 JSON bytes.
        key:
            Optional string key (typically the ``machine_id``).  Encoded
            to UTF-8 bytes before being sent.

        Raises
        ------
        KafkaPublishError
            If the message cannot be published after all retry attempts.
        """
        if self._producer is None:
            raise KafkaPublishError(
                "EventPublisher.start() must be called before publish()",
                context={"topic": topic, "key": key},
            )

        value_bytes: bytes = payload.model_dump_json().encode("utf-8")

        machine_id: str | None = getattr(payload, "machine_id", None)
        _log.debug(
            "kafka_publish_attempt",
            topic=topic,
            key=key,
            machine_id=machine_id,
            payload_bytes=len(value_bytes),
        )

        await self._publish_with_retry(
            topic=topic,
            key=key,
            value=value_bytes,
            machine_id=machine_id,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=1,
            min=_WAIT_MIN_S,
            max=_WAIT_MAX_S,
        ),
        reraise=False,
    )
    async def _publish_with_retry(
        self,
        topic: str,
        key: str | None,
        value: bytes,
        machine_id: str | None,
    ) -> None:
        """Internal publish implementation wrapped by tenacity retry.

        Parameters
        ----------
        topic:
            Target Kafka topic.
        key:
            Optional string message key.
        value:
            Pre-serialised JSON bytes.
        machine_id:
            Used for structured log context.

        Raises
        ------
        KafkaPublishError
            Re-raised after all retry attempts are exhausted.
        """
        if self._producer is None:
            raise KafkaPublishError(
                "Producer is not started.",
                context={"topic": topic, "key": key},
            )
        try:
            await self._producer.send_and_wait(
                topic=topic,
                value=value,
                key=key,
            )
            _log.debug(
                "kafka_publish_success",
                topic=topic,
                key=key,
                machine_id=machine_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning(
                "kafka_publish_error",
                topic=topic,
                key=key,
                machine_id=machine_id,
                error=str(exc),
            )
            raise KafkaPublishError(
                f"Failed to publish to topic '{topic}': {exc}",
                context={
                    "topic": topic,
                    "key": key,
                    "machine_id": machine_id,
                    "error": str(exc),
                },
            ) from exc

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> EventPublisher:
        """Start the producer and return ``self``."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the producer on context exit."""
        await self.stop()
