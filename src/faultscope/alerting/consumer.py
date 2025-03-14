"""Kafka consumer for RUL prediction events.

``PredictionEventConsumer`` subscribes to the
``faultscope.predictions.rul`` topic and routes each message through
the ``IncidentCoordinator``.  It uses the shared ``EventSubscriber``
wrapper for at-least-once delivery with dead-letter queue support.

The consumer runs a tight ``async for`` loop; a ``stop()`` call sets a
cancellation flag that causes the loop to exit cleanly after the
current message finishes processing.

Usage::

    consumer = PredictionEventConsumer(config, coordinator)
    await consumer.run()        # blocks until stop() is called
    await consumer.stop()       # signal graceful shutdown
"""

from __future__ import annotations

import asyncio

from faultscope.alerting.config import AlertingConfig
from faultscope.alerting.coordinator import IncidentCoordinator
from faultscope.common.exceptions import DatabaseError, KafkaConsumeError
from faultscope.common.kafka.consumer import EventSubscriber
from faultscope.common.kafka.schemas import RulPrediction
from faultscope.common.logging import get_logger

_log = get_logger(__name__)


class PredictionEventConsumer:
    """Subscribe to ``faultscope.predictions.rul`` and process events.

    Each consumed ``RulPrediction`` message is forwarded to
    ``IncidentCoordinator.process_prediction``.  Messages that fail
    schema validation are forwarded to the DLQ by the underlying
    ``EventSubscriber``.

    Parameters
    ----------
    config:
        Resolved ``AlertingConfig`` with Kafka connection details.
    coordinator:
        The ``IncidentCoordinator`` instance to route events to.
    """

    def __init__(
        self,
        config: AlertingConfig,
        coordinator: IncidentCoordinator,
    ) -> None:
        self._config = config
        self._coordinator = coordinator
        self._subscriber = EventSubscriber(
            bootstrap_servers=config.kafka_bootstrap_servers,
            group_id=config.kafka_consumer_group,
            topics=[config.topic_rul_predictions],
        )
        self._stop_event: asyncio.Event = asyncio.Event()
        self._running: bool = False

    async def run(self) -> None:
        """Start consuming prediction events until ``stop()`` is called.

        Connects to Kafka, then processes messages in an ``async for``
        loop.  Transient database errors are logged and skipped; Kafka
        connectivity errors raise immediately.

        Raises
        ------
        KafkaConsumeError
            If the Kafka consumer cannot be started.
        """
        self._running = True
        self._stop_event.clear()

        _log.info(
            "prediction_consumer_starting",
            topic=self._config.topic_rul_predictions,
            group_id=self._config.kafka_consumer_group,
        )

        try:
            async with self._subscriber:
                async for prediction in self._subscriber.stream(RulPrediction):
                    if self._stop_event.is_set():
                        _log.info("prediction_consumer_stop_requested")
                        break
                    await self._handle(prediction)
        except KafkaConsumeError:
            _log.error(
                "prediction_consumer_kafka_error",
                topic=self._config.topic_rul_predictions,
            )
            raise
        except asyncio.CancelledError:
            _log.info("prediction_consumer_cancelled")
            raise
        finally:
            self._running = False
            _log.info("prediction_consumer_stopped")

    async def stop(self) -> None:
        """Signal the consumer loop to exit after the current message.

        Safe to call before ``run()`` is invoked; in that case it is a
        no-op.
        """
        _log.info("prediction_consumer_stop_signalled")
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    async def _handle(self, prediction: RulPrediction) -> None:
        """Process one prediction event through the coordinator.

        Errors from the coordinator (e.g. transient DB failures) are
        caught and logged so that the consumer loop continues.

        Parameters
        ----------
        prediction:
            Validated ``RulPrediction`` from Kafka.
        """
        _log.debug(
            "prediction_event_received",
            machine_id=prediction.machine_id,
            rul_cycles=prediction.rul_cycles,
            anomaly_score=prediction.anomaly_score,
            health_label=prediction.health_label,
        )

        try:
            incident_ids = await self._coordinator.process_prediction(
                prediction
            )
            if incident_ids:
                _log.info(
                    "prediction_event_processed",
                    machine_id=prediction.machine_id,
                    incidents_created=len(incident_ids),
                )
        except DatabaseError as exc:
            _log.error(
                "prediction_event_db_error",
                machine_id=prediction.machine_id,
                error=str(exc),
                context=exc.context,
            )
        except (TypeError, ValueError, KeyError) as exc:
            _log.error(
                "prediction_event_processing_error",
                machine_id=prediction.machine_id,
                error=str(exc),
            )
