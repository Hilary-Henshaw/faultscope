"""Typed detection rule definitions for the FaultScope alerting service.

Every rule is an immutable ``DetectionRule`` dataclass.  Rule logic is
expressed as typed Python comparisons dispatched by ``ConditionType`` —
there is no ``eval()``, no ``exec()``, and no dynamic expression
evaluation of any kind.

Usage::

    from faultscope.alerting.rules import DEFAULT_RULES, EvaluationContext
    from faultscope.common.kafka.schemas import RulPrediction

    ctx = EvaluationContext(previous_rul=45.0, active_sensor_count=4)
    for rule in DEFAULT_RULES:
        if rule.evaluate(prediction, ctx):
            print(f"Rule triggered: {rule.rule_name}")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faultscope.common.kafka.schemas import RulPrediction


class Severity(StrEnum):
    """Alert severity levels, ordered lowest → highest."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ConditionType(StrEnum):
    """Enumeration of all supported detection condition types."""

    RUL_BELOW = "rul_below"
    ANOMALY_SCORE_ABOVE = "anomaly_score_above"
    HEALTH_LABEL_IS = "health_label_is"
    RUL_DROP_RATE = "rul_drop_rate"
    MULTI_SENSOR = "multi_sensor"


@dataclass(frozen=True)
class DetectionRule:
    """Immutable specification of a single detection rule.

    Parameters
    ----------
    rule_id:
        Unique stable identifier used for cooldown tracking and
        deduplication (e.g. ``"rul_critical"``).
    rule_name:
        Human-readable display name.
    description:
        One-sentence explanation of what the rule detects.
    severity:
        Alert severity assigned when this rule fires.
    condition_type:
        Selects the evaluation branch inside ``evaluate``.
    thresholds:
        Typed parameter bag consumed by the evaluation logic.
        Keys and value types depend on ``condition_type``; see each
        branch in ``evaluate`` for the expected keys.
    cooldown_s:
        Minimum number of seconds that must elapse after a rule fires
        before the same (machine_id, rule_id) pair can fire again.
    enabled:
        When ``False`` the rule is loaded but never evaluated.
    """

    rule_id: str
    rule_name: str
    description: str
    severity: Severity
    condition_type: ConditionType
    thresholds: dict[str, float | str | int]
    cooldown_s: int
    enabled: bool = True

    def evaluate(
        self,
        prediction: RulPrediction,
        context: EvaluationContext,
    ) -> bool:
        """Return ``True`` if the rule condition is satisfied.

        Dispatch is performed by ``condition_type``; each branch
        contains only typed comparisons — no string evaluation.

        Parameters
        ----------
        prediction:
            The incoming RUL prediction event.
        context:
            Runtime state required for stateful rule types such as
            rate-of-change and multi-sensor checks.

        Returns
        -------
        bool
            ``True`` when the condition is met and the rule should
            trigger an incident.
        """
        if not self.enabled:
            return False

        if self.condition_type is ConditionType.RUL_BELOW:
            threshold = float(self.thresholds["threshold"])
            return prediction.rul_cycles < threshold

        if self.condition_type is ConditionType.ANOMALY_SCORE_ABOVE:
            threshold = float(self.thresholds["threshold"])
            return prediction.anomaly_score > threshold

        if self.condition_type is ConditionType.HEALTH_LABEL_IS:
            label = str(self.thresholds["label"])
            return prediction.health_label == label

        if self.condition_type is ConditionType.RUL_DROP_RATE:
            if context.previous_rul is None:
                return False
            cycles_per_hour = float(self.thresholds["cycles_per_hour"])
            drop = context.previous_rul - prediction.rul_cycles
            # Drop rate is expressed per hour; we normalise to 1-hour
            # window as a point-in-time rate estimate.
            return (drop / 1.0) > cycles_per_hour

        if self.condition_type is ConditionType.MULTI_SENSOR:
            min_sensors = int(self.thresholds["min_sensors"])
            return context.active_sensor_count >= min_sensors

        return False


@dataclass
class EvaluationContext:
    """Runtime context required by stateful rule evaluations.

    Attributes
    ----------
    previous_rul:
        The most recent ``rul_cycles`` value recorded for this machine
        prior to the current prediction.  Required by
        ``ConditionType.RUL_DROP_RATE``; ``None`` when no prior
        prediction exists (e.g. first event after service restart).
    active_sensor_count:
        Number of sensors that are currently reporting anomalous
        readings for this machine.  Required by
        ``ConditionType.MULTI_SENSOR``.
    machine_in_maintenance:
        ``True`` when the machine is in a scheduled maintenance window.
        Used by ``IncidentSuppressor`` but available here so that rule
        authors can also gate on it if needed.
    """

    previous_rul: float | None = None
    active_sensor_count: int = 0
    machine_in_maintenance: bool = False


# --------------------------------------------------------------------------- #
# Default rule registry — all 9 production rules
# --------------------------------------------------------------------------- #

DEFAULT_RULES: list[DetectionRule] = [
    DetectionRule(
        rule_id="rul_critical",
        rule_name="RUL Critical",
        description=(
            "Remaining useful life has fallen below 10 cycles; "
            "immediate intervention required."
        ),
        severity=Severity.CRITICAL,
        condition_type=ConditionType.RUL_BELOW,
        thresholds={"threshold": 10},
        cooldown_s=3600,
    ),
    DetectionRule(
        rule_id="rul_warning",
        rule_name="RUL Warning",
        description=(
            "Remaining useful life is below 30 cycles; "
            "schedule maintenance soon."
        ),
        severity=Severity.WARNING,
        condition_type=ConditionType.RUL_BELOW,
        thresholds={"threshold": 30},
        cooldown_s=7200,
    ),
    DetectionRule(
        rule_id="rul_info",
        rule_name="RUL Informational",
        description=(
            "Remaining useful life has dropped below 50 cycles; "
            "plan maintenance within the next service window."
        ),
        severity=Severity.INFO,
        condition_type=ConditionType.RUL_BELOW,
        thresholds={"threshold": 50},
        cooldown_s=14400,
    ),
    DetectionRule(
        rule_id="anomaly_critical",
        rule_name="Anomaly Critical",
        description=(
            "Anomaly score exceeds 0.9; severe deviation from "
            "normal operating behaviour detected."
        ),
        severity=Severity.CRITICAL,
        condition_type=ConditionType.ANOMALY_SCORE_ABOVE,
        thresholds={"threshold": 0.9},
        cooldown_s=1800,
    ),
    DetectionRule(
        rule_id="anomaly_warning",
        rule_name="Anomaly Warning",
        description=(
            "Anomaly score exceeds 0.7; notable deviation from "
            "baseline operating behaviour detected."
        ),
        severity=Severity.WARNING,
        condition_type=ConditionType.ANOMALY_SCORE_ABOVE,
        thresholds={"threshold": 0.7},
        cooldown_s=3600,
    ),
    DetectionRule(
        rule_id="health_imminent",
        rule_name="Imminent Failure",
        description=(
            "Health classifier predicts imminent_failure; "
            "immediate shutdown and inspection required."
        ),
        severity=Severity.CRITICAL,
        condition_type=ConditionType.HEALTH_LABEL_IS,
        thresholds={"label": "imminent_failure"},
        cooldown_s=1800,
    ),
    DetectionRule(
        rule_id="health_critical",
        rule_name="Critical Health",
        description=(
            "Health classifier predicts critical degradation; "
            "escalate to maintenance team."
        ),
        severity=Severity.WARNING,
        condition_type=ConditionType.HEALTH_LABEL_IS,
        thresholds={"label": "critical"},
        cooldown_s=3600,
    ),
    DetectionRule(
        rule_id="rapid_degradation",
        rule_name="Rapid RUL Degradation",
        description=(
            "RUL has dropped by more than 5 cycles since the "
            "previous prediction; accelerated wear detected."
        ),
        severity=Severity.CRITICAL,
        condition_type=ConditionType.RUL_DROP_RATE,
        thresholds={"cycles_per_hour": 5},
        cooldown_s=3600,
    ),
    DetectionRule(
        rule_id="multi_sensor_anomaly",
        rule_name="Multi-Sensor Anomaly",
        description=(
            "Three or more sensors are simultaneously reporting "
            "anomalous readings; systemic fault suspected."
        ),
        severity=Severity.CRITICAL,
        condition_type=ConditionType.MULTI_SENSOR,
        thresholds={"min_sensors": 3},
        cooldown_s=1800,
    ),
]

# Convenience index for O(1) lookup by rule_id.
RULES_BY_ID: dict[str, DetectionRule] = {r.rule_id: r for r in DEFAULT_RULES}
