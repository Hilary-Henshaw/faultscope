"""Real-time feature engineering pipeline orchestrator.

``FeaturePipeline`` ties together:

* an ``AIOKafkaConsumer`` reading from ``faultscope.sensors.readings``
* a ``DataQualityChecker`` that validates / cleans each message
* a ``TemporalFeatureExtractor`` computing rolling statistics
* a ``SpectralFeatureExtractor`` computing FFT-based features
* a ``CrossSensorCorrelator`` computing inter-sensor correlations
* a ``TimeSeriesWriter`` batching inserts into TimescaleDB
* an ``AIOKafkaProducer`` publishing ``ComputedFeatures`` to
  ``faultscope.features.computed``
* an ``AIOKafkaProducer`` routing rejected messages to
  ``faultscope.dlq``

Prometheus metrics exported on ``config.metrics_port``:

* ``faultscope_stream_messages_total`` – counter with label
  ``status`` in ``{processed, rejected, dlq}``
* ``faultscope_stream_processing_latency_ms`` – histogram of
  end-to-end processing latency per message in milliseconds
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from datetime import UTC, datetime

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, KafkaError
from prometheus_client import Counter, Histogram

from faultscope.common.exceptions import (
    KafkaConsumeError,
    KafkaPublishError,
)
from faultscope.common.logging import get_logger
from faultscope.streaming.config import StreamingConfig
from faultscope.streaming.features.correlation import (
    CrossSensorCorrelator,
)
from faultscope.streaming.features.spectral import (
    SpectralFeatureExtractor,
)
from faultscope.streaming.features.temporal import (
    TemporalFeatureExtractor,
)
from faultscope.streaming.models import ComputedFeatures, SensorReading
from faultscope.streaming.quality import DataQualityChecker
from faultscope.streaming.writer import TimeSeriesWriter

log = get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────

_MESSAGES_TOTAL = Counter(
    "faultscope_stream_messages_total",
    "Total messages processed by the streaming pipeline",
    ["status"],
)
_LATENCY_MS = Histogram(
    "faultscope_stream_processing_latency_ms",
    "End-to-end message processing latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

# Pre-create label instances to avoid repeated dict lookups.
_CTR_PROCESSED = _MESSAGES_TOTAL.labels(status="processed")
_CTR_REJECTED = _MESSAGES_TOTAL.labels(status="rejected")
_CTR_DLQ = _MESSAGES_TOTAL.labels(status="dlq")


def _build_sensor_pairs(
    sensors: list[str],
) -> list[tuple[str, str]]:
    """Return all unique unordered pairs from *sensors*."""
    return list(itertools.combinations(sorted(sensors), 2))


class FeaturePipeline:
    """Orchestrates the real-time feature engineering pipeline.

    Parameters
    ----------
    config:
        Fully populated ``StreamingConfig`` instance.
    """

    def __init__(self, config: StreamingConfig) -> None:
        self._cfg = config

        self._quality = DataQualityChecker(
            max_null_fraction=config.max_null_fraction,
            max_future_drift_s=config.max_future_drift_s,
            min_sensor_count=config.min_sensor_count,
        )

        self._temporal = TemporalFeatureExtractor(
            window_sizes_s=config.rolling_windows_s,
            sampling_rate_hz=config.fft_sampling_rate_hz,
        )

        self._spectral = SpectralFeatureExtractor(
            sampling_rate_hz=config.fft_sampling_rate_hz,
            fft_sensors=config.fft_sensors,
            min_samples=32,
        )

        sensor_pairs = _build_sensor_pairs(config.fft_sensors)
        self._correlator = CrossSensorCorrelator(
            sensor_pairs=sensor_pairs,
            min_samples=10,
        )

        self._writer = TimeSeriesWriter(
            db_url=config.db_async_url,
            batch_size=config.batch_size,
            flush_interval_s=config.flush_interval_s,
            pool_size=config.db_pool_size,
        )

        # Last accepted reading per machine (for forward-fill).
        self._last_reading: dict[str, SensorReading] = {}

        # Kafka clients are created during start().
        self._consumer: AIOKafkaConsumer | None = None
        self._producer: AIOKafkaProducer | None = None
        self._running: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise all I/O clients and the DB writer."""
        await self._writer.start()

        self._consumer = AIOKafkaConsumer(
            self._cfg.topic_sensor_readings,
            bootstrap_servers=self._cfg.kafka_bootstrap_servers,
            group_id=self._cfg.kafka_consumer_group,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            session_timeout_ms=30_000,
            heartbeat_interval_ms=10_000,
        )

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._cfg.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            enable_idempotence=True,
            compression_type="lz4",
        )

        try:
            await self._consumer.start()
        except KafkaConnectionError as exc:
            raise KafkaConsumeError(
                "Failed to connect Kafka consumer",
                context={
                    "servers": self._cfg.kafka_bootstrap_servers,
                    "error": str(exc),
                },
            ) from exc

        try:
            await self._producer.start()
        except KafkaConnectionError as exc:
            raise KafkaPublishError(
                "Failed to connect Kafka producer",
                context={
                    "servers": self._cfg.kafka_bootstrap_servers,
                    "error": str(exc),
                },
            ) from exc

        self._running = True
        log.info(
            "pipeline.started",
            input_topic=self._cfg.topic_sensor_readings,
            output_topic=self._cfg.topic_computed_features,
            dlq_topic=self._cfg.topic_dlq,
        )

    async def stop(self) -> None:
        """Drain in-flight work and shut down all clients."""
        self._running = False

        if self._consumer is not None:
            await self._consumer.stop()

        if self._producer is not None:
            await self._producer.stop()

        await self._writer.stop()
        log.info("pipeline.stopped")

    async def run(self) -> None:
        """Main processing loop.  Runs until cancelled or ``stop()`` is
        called.

        Each iteration pulls a batch of messages from Kafka (up to the
        consumer's internal fetch limit), processes them one-by-one,
        and commits offsets automatically.

        Raises
        ------
        KafkaConsumeError
            On unrecoverable Kafka consumer errors.
        """
        if self._consumer is None or self._producer is None:
            raise RuntimeError("Pipeline is not started; call start() first")

        log.info("pipeline.run_loop_started")
        try:
            async for msg in self._consumer:
                if not self._running:
                    break
                await self._handle_message(msg.value)
        except KafkaError as exc:
            raise KafkaConsumeError(
                "Kafka consumer error in run loop",
                context={"error": str(exc)},
            ) from exc
        except asyncio.CancelledError:
            log.info("pipeline.run_cancelled")
            raise

    # ── Message handling ──────────────────────────────────────────────

    async def _handle_message(self, raw: dict[object, object]) -> None:
        """Process one raw Kafka message.

        Parameters
        ----------
        raw:
            Deserialised JSON payload from the Kafka message value.
        """
        t0 = time.monotonic()

        try:
            reading = SensorReading.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "pipeline.parse_error",
                error=str(exc),
                raw_keys=list(raw.keys()) if isinstance(raw, dict) else None,
            )
            await self._send_dlq(raw, reason="parse_error")
            _CTR_DLQ.inc()
            return

        previous = self._last_reading.get(reading.machine_id)
        quality = self._quality.check(reading, previous)

        if quality.rejected:
            log.info(
                "pipeline.rejected",
                machine_id=reading.machine_id,
                flags=quality.flag_names,
            )
            await self._send_dlq(
                raw,
                reason="quality_rejected",
                flags=quality.flag_names,
            )
            _CTR_REJECTED.inc()
            _CTR_DLQ.inc()
            return

        # Replace readings dict with the filled version.
        clean_reading = reading.model_copy(
            update={"readings": quality.filled_readings}
        )

        # Persist the raw (cleaned) reading.
        await self._writer.buffer_reading(clean_reading, quality.flag_names)

        # Update rolling windows.
        self._temporal.update(
            machine_id=clean_reading.machine_id,
            readings=clean_reading.readings,
            timestamp=clean_reading.recorded_at,
        )

        # Extract and publish features for every configured window.
        now_utc = datetime.now(tz=UTC)
        for window_s in self._cfg.rolling_windows_s:
            temporal_feats = self._temporal.extract(
                clean_reading.machine_id,
                clean_reading.recorded_at,
            )

            window_vals = self._temporal.window_values(
                clean_reading.machine_id, window_s
            )
            spectral_feats = self._spectral.extract(
                clean_reading.machine_id, window_vals
            )
            corr_feats = self._correlator.extract(window_vals)

            if not temporal_feats and not spectral_feats:
                continue

            features = ComputedFeatures(
                machine_id=clean_reading.machine_id,
                computed_at=now_utc,
                window_s=window_s,
                temporal=temporal_feats,
                spectral=spectral_feats,
                correlation=corr_feats,
            )

            await self._writer.buffer_features(features)
            await self._publish_features(features)

        self._last_reading[clean_reading.machine_id] = clean_reading
        _CTR_PROCESSED.inc()

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        _LATENCY_MS.observe(elapsed_ms)

        log.debug(
            "pipeline.message_processed",
            machine_id=clean_reading.machine_id,
            latency_ms=round(elapsed_ms, 2),
            flags=quality.flag_names,
        )

    async def _publish_features(self, features: ComputedFeatures) -> None:
        """Serialise and publish a ``ComputedFeatures`` message.

        Parameters
        ----------
        features:
            Feature vector to publish.

        Raises
        ------
        KafkaPublishError
            When the producer cannot deliver the message.
        """
        if self._producer is None:
            raise KafkaPublishError(
                "Producer is not initialised",
                context={"machine_id": features.machine_id},
            )

        payload = features.model_dump(mode="json")
        try:
            await self._producer.send_and_wait(
                self._cfg.topic_computed_features,
                value=payload,
                key=features.machine_id.encode("utf-8"),
            )
        except KafkaError as exc:
            raise KafkaPublishError(
                "Failed to publish computed features",
                context={
                    "topic": self._cfg.topic_computed_features,
                    "machine_id": features.machine_id,
                    "error": str(exc),
                },
            ) from exc

    async def _send_dlq(
        self,
        raw: object,
        *,
        reason: str,
        flags: list[str] | None = None,
    ) -> None:
        """Route a message to the dead-letter queue topic.

        Parameters
        ----------
        raw:
            Original raw payload (dict from Kafka deserialization).
        reason:
            Short string describing why the message was rejected.
        flags:
            Optional list of quality flag names attached to the DLQ
            envelope.
        """
        if self._producer is None:
            log.error(
                "pipeline.dlq_no_producer",
                reason=reason,
            )
            return

        envelope: dict[str, object] = {
            "reason": reason,
            "flags": flags or [],
            "original": raw,
            "dlq_at": datetime.now(tz=UTC).isoformat(),
        }
        try:
            await self._producer.send_and_wait(
                self._cfg.topic_dlq,
                value=envelope,
            )
        except KafkaError as exc:
            # DLQ publish failures are non-fatal; log and continue.
            log.error(
                "pipeline.dlq_publish_failed",
                reason=reason,
                error=str(exc),
            )
