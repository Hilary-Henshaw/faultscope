"""Async Kafka consumer wrapper for FaultScope services.

Provides ``EventSubscriber``, an ``aiokafka``-backed consumer that
deserialises JSON messages into typed Pydantic models.  Messages that
cannot be parsed are forwarded to the dead-letter queue (DLQ) rather
than crashing the consumer loop.

Usage::

    async with EventSubscriber(
        bootstrap_servers="localhost:9092",
        group_id="faultscope-inference",
        topics=["faultscope.features.computed"],
    ) as sub:
        async for features in sub.stream(ComputedFeatures):
            await handle(features)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import TypeVar

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import BaseModel, ValidationError

from faultscope.common.exceptions import KafkaConsumeError
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Name of the dead-letter queue topic.
_DLQ_TOPIC: str = "faultscope.dlq"


class EventSubscriber:
    """Async Kafka consumer with typed message deserialisation.

    Each unconsumed or unparseable message is forwarded to the DLQ so
    that no data is silently discarded.  The consumer commits offsets
    only after the caller's ``async for`` iteration step completes,
    providing at-least-once delivery semantics.

    Parameters
    ----------
    bootstrap_servers:
        Comma-separated list of Kafka broker addresses.
    group_id:
        Consumer group identifier.  All replicas of a service should
        share the same group so that partitions are distributed.
    topics:
        List of topic names to subscribe to.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: list[str],
    ) -> None:
        """Initialise the subscriber.

        The underlying ``AIOKafkaConsumer`` is not started here; call
        ``start`` or use as an async context manager.

        Parameters
        ----------
        bootstrap_servers:
            Comma-separated ``host:port`` broker list.
        group_id:
            Kafka consumer group ID.
        topics:
            Topics to subscribe to.
        """
        self._bootstrap_servers: str = bootstrap_servers
        self._group_id: str = group_id
        self._topics: list[str] = topics
        self._consumer: AIOKafkaConsumer | None = None
        # Minimal producer used exclusively to forward bad messages to
        # the DLQ.  Started lazily on first DLQ write.
        self._dlq_producer: AIOKafkaProducer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Kafka consumer and subscribe to configured topics.

        Raises
        ------
        KafkaConsumeError
            If the consumer cannot connect to the broker or subscribe.
        """
        _log.info(
            "kafka_consumer_starting",
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            topics=self._topics,
        )
        try:
            self._consumer = AIOKafkaConsumer(
                *self._topics,
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._group_id,
                # Manual commit lets us ack only after the caller
                # finishes processing (at-least-once semantics).
                enable_auto_commit=False,
                auto_offset_reset="latest",
                request_timeout_ms=30_000,
                session_timeout_ms=30_000,
                heartbeat_interval_ms=10_000,
            )
            await self._consumer.start()
        except Exception as exc:
            _log.error(
                "kafka_consumer_start_failed",
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._group_id,
                topics=self._topics,
                error=str(exc),
            )
            raise KafkaConsumeError(
                f"Failed to start Kafka consumer: {exc}",
                context={
                    "bootstrap_servers": self._bootstrap_servers,
                    "group_id": self._group_id,
                    "topics": self._topics,
                    "error": str(exc),
                },
            ) from exc

        _log.info(
            "kafka_consumer_started",
            group_id=self._group_id,
            topics=self._topics,
        )

    async def stop(self) -> None:
        """Commit pending offsets and stop the consumer.

        Safe to call even if ``start`` was never called.
        """
        if self._dlq_producer is not None:
            try:
                await self._dlq_producer.stop()
            except Exception as exc:  # noqa: BLE001
                _log.warning("kafka_dlq_producer_stop_error", error=str(exc))
            finally:
                self._dlq_producer = None

        if self._consumer is None:
            return

        _log.info("kafka_consumer_stopping", group_id=self._group_id)
        try:
            await self._consumer.stop()
        except Exception as exc:  # noqa: BLE001
            _log.warning("kafka_consumer_stop_error", error=str(exc))
        finally:
            self._consumer = None
        _log.info("kafka_consumer_stopped")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        schema: type[T],
    ) -> AsyncIterator[T]:
        """Yield typed messages deserialised from the subscribed topics.

        Messages that cannot be decoded (invalid JSON or schema
        validation failure) are forwarded to the DLQ topic and skipped.
        Offsets are committed after each successfully yielded message.

        Parameters
        ----------
        schema:
            The Pydantic ``BaseModel`` subclass to deserialise each
            message into.

        Yields
        ------
        T
            A deserialised, validated Pydantic model instance.

        Raises
        ------
        KafkaConsumeError
            If ``start()`` has not been called.
        """
        if self._consumer is None:
            raise KafkaConsumeError(
                "EventSubscriber.start() must be called before stream()",
                context={
                    "group_id": self._group_id,
                    "topics": self._topics,
                },
            )

        async for raw_msg in self._consumer:
            topic: str = raw_msg.topic
            partition: int = raw_msg.partition
            offset: int = raw_msg.offset
            raw_value: bytes = raw_msg.value or b""

            _log.debug(
                "kafka_message_received",
                topic=topic,
                partition=partition,
                offset=offset,
                bytes=len(raw_value),
            )

            parsed: T | None = await self._parse_message(
                raw_value=raw_value,
                schema=schema,
                topic=topic,
                partition=partition,
                offset=offset,
            )

            if parsed is None:
                # Undeserializable – already forwarded to DLQ.
                await self._consumer.commit()
                continue

            yield parsed

            # Commit the offset after the caller's processing step.
            await self._consumer.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _parse_message(
        self,
        raw_value: bytes,
        schema: type[T],
        topic: str,
        partition: int,
        offset: int,
    ) -> T | None:
        """Attempt to deserialise ``raw_value`` into ``schema``.

        On failure, forward the raw bytes to the DLQ and return
        ``None``.

        Parameters
        ----------
        raw_value:
            Raw bytes from the Kafka message.
        schema:
            Target Pydantic model class.
        topic:
            Source topic (for DLQ metadata).
        partition:
            Source partition (for DLQ metadata).
        offset:
            Source offset (for DLQ metadata).

        Returns
        -------
        T | None
            Parsed model on success, ``None`` on any parse failure.
        """
        try:
            data: object = json.loads(raw_value)
            return schema.model_validate(data)
        except json.JSONDecodeError as exc:
            _log.error(
                "kafka_message_json_decode_error",
                topic=topic,
                partition=partition,
                offset=offset,
                error=str(exc),
            )
        except ValidationError as exc:
            _log.error(
                "kafka_message_validation_error",
                topic=topic,
                partition=partition,
                offset=offset,
                schema=schema.__name__,
                errors=exc.errors(),
            )
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "kafka_message_unexpected_parse_error",
                topic=topic,
                partition=partition,
                offset=offset,
                error=str(exc),
            )

        await self._send_to_dlq(
            raw_value=raw_value,
            source_topic=topic,
            partition=partition,
            offset=offset,
        )
        return None

    async def _send_to_dlq(
        self,
        raw_value: bytes,
        source_topic: str,
        partition: int,
        offset: int,
    ) -> None:
        """Forward an unparseable message to the dead-letter queue.

        The DLQ message wraps the original bytes in a JSON envelope
        that carries provenance metadata.

        Parameters
        ----------
        raw_value:
            The original, unparseable message bytes.
        source_topic:
            The topic from which the bad message was read.
        partition:
            The partition from which the bad message was read.
        offset:
            The offset of the bad message.
        """
        dlq_producer = await self._get_dlq_producer()

        envelope: bytes = json.dumps(
            {
                "source_topic": source_topic,
                "partition": partition,
                "offset": offset,
                "group_id": self._group_id,
                "raw_payload": raw_value.decode("utf-8", errors="replace"),
            }
        ).encode("utf-8")

        try:
            await dlq_producer.send_and_wait(
                topic=_DLQ_TOPIC,
                value=envelope,
            )
            _log.warning(
                "kafka_message_sent_to_dlq",
                source_topic=source_topic,
                partition=partition,
                offset=offset,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # DLQ failures must not crash the consumer loop.
            _log.error(
                "kafka_dlq_send_failed",
                source_topic=source_topic,
                partition=partition,
                offset=offset,
                error=str(exc),
            )

    async def _get_dlq_producer(self) -> AIOKafkaProducer:
        """Return the shared DLQ producer, starting it lazily.

        Returns
        -------
        AIOKafkaProducer
            The started DLQ producer instance.
        """
        if self._dlq_producer is None:
            self._dlq_producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                acks=1,
                compression_type="gzip",
            )
            await self._dlq_producer.start()
            _log.info("kafka_dlq_producer_started")
        return self._dlq_producer

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> EventSubscriber:
        """Start the consumer and return ``self``."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the consumer on context exit."""
        await self.stop()
