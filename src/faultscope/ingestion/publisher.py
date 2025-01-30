"""Async Kafka publisher for the FaultScope ingestion service.

:class:`SensorPublisher` wraps the lower-level
:class:`~faultscope.common.kafka.producer.EventPublisher` and adds
ingestion-specific concerns:

- Partitioning by ``machine_id`` for strict per-machine ordering.
- Throughput counters logged every 1 000 messages.
- A clean async context-manager interface.

The class is intentionally thin — all retry logic, serialisation, and
producer lifecycle management live in ``EventPublisher``.
"""

from __future__ import annotations

import structlog

from faultscope.common.exceptions import KafkaPublishError
from faultscope.common.kafka.producer import EventPublisher
from faultscope.common.kafka.schemas import SensorReading
from faultscope.common.logging import get_logger

log: structlog.stdlib.BoundLogger = get_logger(__name__)

_THROUGHPUT_LOG_INTERVAL: int = 1_000


class SensorPublisher:
    """Publishes :class:`~faultscope.common.kafka.schemas.SensorReading`
    events to a Kafka topic.

    Wraps :class:`~faultscope.common.kafka.producer.EventPublisher` with
    ingestion-specific logic:

    - Uses ``reading.machine_id`` as the Kafka message key so that all
      readings for a given machine land on the same partition and are
      consumed in the order they were produced.
    - Emits a throughput log line every
      :data:`_THROUGHPUT_LOG_INTERVAL` messages.

    Parameters
    ----------
    bootstrap_servers:
        Comma-separated ``host:port`` Kafka broker list.
    topic:
        Target topic name (typically
        ``"faultscope.sensors.readings"``).
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
    ) -> None:
        if not bootstrap_servers:
            raise ValueError("bootstrap_servers must not be empty")
        if not topic:
            raise ValueError("topic must not be empty")
        self._topic = topic
        self._publisher = EventPublisher(bootstrap_servers=bootstrap_servers)
        self._sent: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the underlying Kafka producer.

        Must be called before :meth:`send_reading` when not using the
        async context manager.

        Raises
        ------
        KafkaPublishError
            If the producer cannot connect to the broker.
        """
        log.info(
            "sensor_publisher.starting",
            topic=self._topic,
        )
        await self._publisher.start()
        log.info(
            "sensor_publisher.started",
            topic=self._topic,
        )

    async def stop(self) -> None:
        """Flush and stop the underlying Kafka producer.

        Safe to call even if :meth:`start` was never invoked.
        """
        log.info("sensor_publisher.stopping", sent_total=self._sent)
        await self._publisher.stop()
        log.info("sensor_publisher.stopped", sent_total=self._sent)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def send_reading(self, reading: SensorReading) -> None:
        """Serialise and publish one sensor reading to Kafka.

        Uses ``reading.machine_id`` as the Kafka partition key to
        guarantee that all events for a single machine are consumed in
        order.

        Parameters
        ----------
        reading:
            The sensor reading to publish.

        Raises
        ------
        KafkaPublishError
            If the message cannot be delivered after retry attempts.
        """
        try:
            await self._publisher.publish(
                topic=self._topic,
                payload=reading,
                key=reading.machine_id,
            )
        except KafkaPublishError:
            log.error(
                "sensor_publisher.send_failed",
                machine_id=reading.machine_id,
                topic=self._topic,
                cycle=reading.cycle,
            )
            raise

        self._sent += 1
        if self._sent % _THROUGHPUT_LOG_INTERVAL == 0:
            log.info(
                "sensor_publisher.throughput",
                sent_total=self._sent,
                topic=self._topic,
            )

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> SensorPublisher:
        """Start the publisher and return ``self``."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the publisher on context exit."""
        await self.stop()
