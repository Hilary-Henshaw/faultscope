"""Rule evaluation engine for the FaultScope alerting service.

``RuleEvaluationEngine`` applies all configured ``DetectionRule``
instances to an incoming ``RulPrediction``, respects per-rule cooldown
windows, and returns the list of ``TriggeredIncident`` objects that
should be dispatched for notification.

Usage::

    from faultscope.alerting.engine.evaluator import (
        RuleEvaluationEngine,
        TriggeredIncident,
    )
    from faultscope.alerting.rules import DEFAULT_RULES, EvaluationContext

    engine = RuleEvaluationEngine(DEFAULT_RULES)
    ctx = EvaluationContext(previous_rul=42.0, active_sensor_count=2)
    incidents = engine.evaluate(prediction, ctx)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from faultscope.alerting.rules import (
    DetectionRule,
    EvaluationContext,
    Severity,
)
from faultscope.common.kafka.schemas import RulPrediction
from faultscope.common.logging import get_logger

_log = get_logger(__name__)


@dataclass
class TriggeredIncident:
    """A single rule evaluation that produced a positive match.

    Attributes
    ----------
    rule:
        The ``DetectionRule`` whose condition was satisfied.
    machine_id:
        Identifier of the machine that triggered the rule.
    title:
        Short human-readable title for the incident.
    details:
        Snapshot of the prediction values that caused the trigger.
    severity:
        Severity copied from the rule at evaluation time.
    triggered_at:
        UTC timestamp of evaluation.
    """

    rule: DetectionRule
    machine_id: str
    title: str
    details: dict[str, object]
    severity: Severity
    triggered_at: datetime


class RuleEvaluationEngine:
    """Evaluate all enabled detection rules against each prediction.

    Maintains an in-memory ``dict`` mapping
    ``(machine_id, rule_id) → last_trigger_utc`` to enforce per-rule
    cooldown periods.  Rules that are still within their cooldown window
    are skipped without producing an incident.

    Parameters
    ----------
    rules:
        List of ``DetectionRule`` instances to evaluate.  Disabled
        rules (``enabled=False``) are filtered out at construction time.
    """

    def __init__(self, rules: list[DetectionRule]) -> None:
        self._rules: list[DetectionRule] = [r for r in rules if r.enabled]
        # (machine_id, rule_id) → UTC datetime of last trigger
        self._last_trigger: dict[tuple[str, str], datetime] = {}

        _log.info(
            "rule_engine_initialized",
            total_rules=len(rules),
            enabled_rules=len(self._rules),
        )

    def evaluate(
        self,
        prediction: RulPrediction,
        context: EvaluationContext,
    ) -> list[TriggeredIncident]:
        """Evaluate all enabled rules against ``prediction``.

        Rules in cooldown for ``prediction.machine_id`` are skipped.
        Each rule that fires produces one ``TriggeredIncident``.

        Parameters
        ----------
        prediction:
            The incoming RUL prediction event.
        context:
            Runtime evaluation context (previous RUL, sensor count, …).

        Returns
        -------
        list[TriggeredIncident]
            Possibly empty list of triggered incidents.
        """
        now = datetime.now(tz=UTC)
        incidents: list[TriggeredIncident] = []

        for rule in self._rules:
            if self._is_in_cooldown(
                prediction.machine_id,
                rule.rule_id,
                rule.cooldown_s,
                now=now,
            ):
                _log.debug(
                    "rule_skipped_cooldown",
                    machine_id=prediction.machine_id,
                    rule_id=rule.rule_id,
                    cooldown_s=rule.cooldown_s,
                )
                continue

            try:
                fired = rule.evaluate(prediction, context)
            except (TypeError, KeyError, ValueError) as exc:
                _log.error(
                    "rule_evaluation_error",
                    rule_id=rule.rule_id,
                    machine_id=prediction.machine_id,
                    error=str(exc),
                )
                continue

            if not fired:
                continue

            self._record_trigger(prediction.machine_id, rule.rule_id, now=now)

            incident = TriggeredIncident(
                rule=rule,
                machine_id=prediction.machine_id,
                title=self._build_title(rule, prediction),
                details=self._build_details(rule, prediction, context),
                severity=rule.severity,
                triggered_at=now,
            )
            incidents.append(incident)

            _log.info(
                "rule_triggered",
                machine_id=prediction.machine_id,
                rule_id=rule.rule_id,
                severity=rule.severity.value,
                rul_cycles=prediction.rul_cycles,
                anomaly_score=prediction.anomaly_score,
                health_label=prediction.health_label,
            )

        return incidents

    # ------------------------------------------------------------------ #
    # Cooldown helpers
    # ------------------------------------------------------------------ #

    def _is_in_cooldown(
        self,
        machine_id: str,
        rule_id: str,
        cooldown_s: int,
        now: datetime | None = None,
    ) -> bool:
        """Return ``True`` if the rule is still within its cooldown.

        Parameters
        ----------
        machine_id:
            Machine under evaluation.
        rule_id:
            Rule being checked.
        cooldown_s:
            Required silence window in seconds.
        now:
            Current UTC time; defaults to ``datetime.now(utc)`` when
            ``None`` (injectable for testing).

        Returns
        -------
        bool
            ``True`` means the rule must be skipped this cycle.
        """
        key = (machine_id, rule_id)
        last = self._last_trigger.get(key)
        if last is None:
            return False
        effective_now = now or datetime.now(tz=UTC)
        elapsed = (effective_now - last).total_seconds()
        return elapsed < cooldown_s

    def _record_trigger(
        self,
        machine_id: str,
        rule_id: str,
        now: datetime | None = None,
    ) -> None:
        """Record that a rule has just fired for a machine.

        Parameters
        ----------
        machine_id:
            Machine that triggered the rule.
        rule_id:
            Rule that fired.
        now:
            Current UTC time; defaults to ``datetime.now(utc)``.
        """
        key = (machine_id, rule_id)
        self._last_trigger[key] = now or datetime.now(tz=UTC)

    # ------------------------------------------------------------------ #
    # Incident builders
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_title(
        rule: DetectionRule,
        prediction: RulPrediction,
    ) -> str:
        """Build a short incident title.

        Parameters
        ----------
        rule:
            The rule that fired.
        prediction:
            The triggering prediction event.

        Returns
        -------
        str
            Human-readable title combining machine and rule name.
        """
        return (
            f"[{rule.severity.value.upper()}] "
            f"{prediction.machine_id}: {rule.rule_name}"
        )

    @staticmethod
    def _build_details(
        rule: DetectionRule,
        prediction: RulPrediction,
        context: EvaluationContext,
    ) -> dict[str, object]:
        """Snapshot the relevant prediction values for this rule.

        Parameters
        ----------
        rule:
            The rule that fired.
        prediction:
            The triggering prediction.
        context:
            Evaluation context at trigger time.

        Returns
        -------
        dict[str, object]
            Key/value snapshot for storage and notification rendering.
        """
        return {
            "rule_id": rule.rule_id,
            "rule_name": rule.rule_name,
            "condition_type": rule.condition_type.value,
            "thresholds": dict(rule.thresholds),
            "rul_cycles": prediction.rul_cycles,
            "rul_hours": prediction.rul_hours,
            "anomaly_score": prediction.anomaly_score,
            "health_label": prediction.health_label,
            "confidence": prediction.confidence,
            "rul_model_version": prediction.rul_model_version,
            "health_model_version": prediction.health_model_version,
            "previous_rul": context.previous_rul,
            "active_sensor_count": context.active_sensor_count,
            "predicted_at": prediction.predicted_at.isoformat(),
        }
