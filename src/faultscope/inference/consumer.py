"""Kafka consumer pipeline for automated inference.

``PredictionConsumer`` subscribes to ``faultscope.features.computed``,
deserialises each ``ComputedFeatures`` message, runs RUL and health
predictions via the ``PredictionEngine``, and publishes a
``RulPrediction`` result to ``faultscope.predictions.rul``.

It runs as a persistent background task alongside the FastAPI server,
started in the ``lifespan`` context or as a standalone process.

Error handling
--------------
- Deserialisable but prediction-failing messages are logged and skipped
  (the consumer does not crash on a single bad message).
- The underlying ``EventSubscriber`` handles JSON/validation errors by
  forwarding them to the DLQ.
- ``asyncio.CancelledError`` propagates cleanly through ``stop()``.

Usage::

    consumer = PredictionConsumer(
        config=InferenceConfig(),
        engine=prediction_engine,
        publisher=event_publisher,
    )
    task = asyncio.create_task(consumer.run())
    ...
    await consumer.stop()
    await task
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from faultscope.common.exceptions import (
    ModelLoadError,
    ValidationError,
)
from faultscope.common.kafka.consumer import EventSubscriber
from faultscope.common.kafka.producer import EventPublisher
from faultscope.common.kafka.schemas import ComputedFeatures, RulPrediction
from faultscope.common.logging import get_logger
from faultscope.inference.config import InferenceConfig
from faultscope.inference.engine.predictor import PredictionEngine

_log = get_logger(__name__)

_CONSUMER_GROUP: str = "faultscope-inference-consumer"
_FEATURES_TOPIC: str = "faultscope.features.computed"


class PredictionConsumer:
    """Subscribes to feature events, runs inference, publishes results.

    The consumer maintains a single ``EventSubscriber`` connection and
    processes messages sequentially.  For higher throughput, run
    multiple replicas in separate consumer-group members (Kafka will
    distribute partitions automatically).

    Parameters
    ----------
    config:
        Resolved ``InferenceConfig``.
    engine:
        Shared ``PredictionEngine`` instance (models must be loaded).
    publisher:
        Shared ``EventPublisher`` used to emit prediction results.
    """

    def __init__(
        self,
        config: InferenceConfig,
        engine: PredictionEngine,
        publisher: EventPublisher,
    ) -> None:
        self._config = config
        self._engine = engine
        self._publisher = publisher
        self._subscriber: EventSubscriber | None = None
        self._running = False

    async def run(self) -> None:
        """Start consuming and processing feature messages.

        Runs until ``stop()`` is called or the task is cancelled.
        Each ``ComputedFeatures`` message produces one ``RulPrediction``
        published to the configured output topic.
        """
        self._running = True
        self._subscriber = EventSubscriber(
            bootstrap_servers=self._config.kafka_bootstrap_servers,
            group_id=_CONSUMER_GROUP,
            topics=[_FEATURES_TOPIC],
        )
        await self._subscriber.start()

        _log.info(
            "prediction_consumer_started",
            group_id=_CONSUMER_GROUP,
            topic=_FEATURES_TOPIC,
        )

        try:
            async for features in self._subscriber.stream(ComputedFeatures):
                if not self._running:
                    break
                await self._process(features)
        except asyncio.CancelledError:
            _log.info("prediction_consumer_cancelled")
            raise
        finally:
            if self._subscriber is not None:
                await self._subscriber.stop()
                self._subscriber = None
            _log.info("prediction_consumer_stopped")

    async def stop(self) -> None:
        """Signal the consumer loop to exit after the current message."""
        self._running = False
        _log.info("prediction_consumer_stop_requested")

    async def _process(self, features: ComputedFeatures) -> None:
        """Run inference on a single feature message and publish result.

        Failures are caught and logged; they do not abort the loop.

        Parameters
        ----------
        features:
            Deserialised ``ComputedFeatures`` Kafka message.
        """
        machine_id = features.machine_id
        _log.debug(
            "prediction_consumer_processing",
            machine_id=machine_id,
            computed_at=features.computed_at.isoformat(),
        )

        # Build flat feature dict from all feature groups.
        flat_features: dict[str, float] = {}
        flat_features.update(features.temporal)
        flat_features.update(features.spectral)
        flat_features.update(features.correlation)

        if not flat_features:
            _log.warning(
                "prediction_consumer_empty_features",
                machine_id=machine_id,
            )
            return

        try:
            rul_result = await self._engine.predict_remaining_life(
                machine_id=machine_id,
                feature_sequence=[flat_features],
            )
            health_result = await self._engine.predict_health_status(
                machine_id=machine_id,
                features=flat_features,
            )
        except (ModelLoadError, ValidationError) as exc:
            _log.error(
                "prediction_consumer_inference_failed",
                machine_id=machine_id,
                error=str(exc),
            )
            return
        except Exception as exc:
            _log.error(
                "prediction_consumer_unexpected_error",
                machine_id=machine_id,
                error=str(exc),
                exc_info=True,
            )
            return

        prediction = RulPrediction(
            machine_id=machine_id,
            predicted_at=datetime.now(tz=UTC),
            rul_cycles=rul_result.rul_cycles,
            rul_hours=rul_result.rul_hours,
            rul_lower_bound=rul_result.rul_lower_bound,
            rul_upper_bound=rul_result.rul_upper_bound,
            health_label=health_result.health_label,  # type: ignore[arg-type]
            health_probabilities=health_result.probabilities,
            anomaly_score=max(
                0.0,
                min(1.0, 1.0 - rul_result.confidence),
            ),
            confidence=rul_result.confidence,
            rul_model_version=rul_result.model_version,
            health_model_version=health_result.model_version,
            latency_ms=rul_result.latency_ms,
        )

        try:
            await self._publisher.publish(
                topic=self._config.topic_rul_predictions,
                payload=prediction,
                key=machine_id,
            )
            _log.info(
                "prediction_consumer_published",
                machine_id=machine_id,
                rul_cycles=round(rul_result.rul_cycles, 2),
                health_label=health_result.health_label,
                topic=self._config.topic_rul_predictions,
            )
        except Exception as exc:
            _log.error(
                "prediction_consumer_publish_failed",
                machine_id=machine_id,
                error=str(exc),
            )
