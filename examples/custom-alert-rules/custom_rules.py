"""Custom alert rules example.

Shows how to define DetectionRule dataclasses and run them through the
RuleEvaluationEngine without any running infrastructure.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from faultscope.alerting.engine.evaluator import RuleEvaluationEngine
from faultscope.alerting.rules import (
    ConditionType,
    DetectionRule,
    EvaluationContext,
    Severity,
)
from faultscope.common.kafka.schemas import RulPrediction

UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# Define custom rules
# ---------------------------------------------------------------------------

CUSTOM_RULES: list[DetectionRule] = [
    # Built-in style: critical RUL threshold
    DetectionRule(
        rule_id="critical-rul",
        name="Critical RUL Threshold",
        description="Machine has fewer than 20 cycles remaining",
        condition_type=ConditionType.RUL_BELOW,
        threshold=20.0,
        severity=Severity.CRITICAL,
        cooldown_s=300,
        enabled=True,
    ),
    # Warning level: broader window
    DetectionRule(
        rule_id="warning-rul",
        name="Warning RUL Threshold",
        description="Machine has fewer than 50 cycles remaining",
        condition_type=ConditionType.RUL_BELOW,
        threshold=50.0,
        severity=Severity.WARNING,
        cooldown_s=600,
        enabled=True,
    ),
    # Custom: vibration anomaly detection
    DetectionRule(
        rule_id="vibration-anomaly",
        name="High Vibration Anomaly",
        description=(
            "Anomaly detector signals abnormal vibration pattern"
        ),
        condition_type=ConditionType.ANOMALY_SCORE_ABOVE,
        threshold=0.80,
        severity=Severity.WARNING,
        cooldown_s=180,
        enabled=True,
    ),
    # Custom: imminent failure label for pumps
    DetectionRule(
        rule_id="pump-imminent-failure",
        name="Pump Imminent Failure",
        description="Health classifier predicts imminent failure",
        condition_type=ConditionType.HEALTH_LABEL_IS,
        # threshold_label used for HEALTH_LABEL_IS rules
        threshold=0.0,
        severity=Severity.CRITICAL,
        cooldown_s=60,
        enabled=True,
    ),
]


# ---------------------------------------------------------------------------
# Build a test prediction
# ---------------------------------------------------------------------------

def make_test_prediction(
    machine_id: str,
    predicted_rul: float,
    anomaly_score: float,
    health_label: str,
) -> RulPrediction:
    """Construct a synthetic prediction for testing rules."""
    return RulPrediction(
        machine_id=machine_id,
        predicted_rul=predicted_rul,
        confidence_lower=predicted_rul * 0.8,
        confidence_upper=predicted_rul * 1.2,
        health_label=health_label,  # type: ignore[arg-type]
        anomaly_score=anomaly_score,
        model_version="test",
        predicted_at=dt.datetime.now(tz=UTC),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Custom Alert Rules Example ===\n")

    # The engine tracks cooldowns per (machine_id, rule_id)
    engine = RuleEvaluationEngine(rules=CUSTOM_RULES)

    machine_id = "pump-example-001"
    prediction = make_test_prediction(
        machine_id=machine_id,
        predicted_rul=18.0,        # Below both thresholds
        anomaly_score=0.91,        # Above vibration threshold
        health_label="imminent_failure",
    )

    context = EvaluationContext(
        prediction=prediction,
        machine_id=machine_id,
        anomalous_sensor_count=3,
    )

    print(f"Evaluating {len(CUSTOM_RULES)} rules against test prediction...")
    print(
        f"  machine_id={machine_id}, "
        f"RUL={prediction.predicted_rul}, "
        f"anomaly_score={prediction.anomaly_score}, "
        f"health={prediction.health_label}\n"
    )

    incidents = engine.evaluate(context)

    rule_map = {r.rule_id: r for r in CUSTOM_RULES}
    triggered_ids = {inc.rule_id for inc in incidents}

    for rule in CUSTOM_RULES:
        if rule.rule_id in triggered_ids:
            inc = next(i for i in incidents if i.rule_id == rule.rule_id)
            print(
                f"  [TRIGGERED] {rule.rule_id} "
                f"({rule.condition_type} threshold={rule.threshold})"
            )
            print(f"              severity={rule.severity}")
            print(f"              incident_id={inc.incident_id}")
        else:
            print(
                f"  [not triggered] {rule.rule_id} "
                f"({rule.condition_type})"
            )
        print()

    print(f"Total incidents triggered: {len(incidents)}")

    # ---------------------------------------------------------------------------
    # Demonstrate cooldown: evaluate the same prediction again immediately
    # ---------------------------------------------------------------------------
    print("\n--- Evaluating again (cooldown should suppress duplicates) ---\n")
    incidents_2 = engine.evaluate(context)

    if not incidents_2:
        print(
            "All rules suppressed by cooldown — "
            "no duplicate incidents generated."
        )
    else:
        print(f"WARNING: {len(incidents_2)} incidents not suppressed:")
        for inc in incidents_2:
            print(f"  {inc.rule_id}")

    # ---------------------------------------------------------------------------
    # Demonstrate a healthy prediction (no rules should trigger)
    # ---------------------------------------------------------------------------
    print("\n--- Evaluating a healthy machine ---\n")
    healthy_prediction = make_test_prediction(
        machine_id="pump-healthy-001",
        predicted_rul=110.0,
        anomaly_score=0.12,
        health_label="healthy",
    )
    healthy_context = EvaluationContext(
        prediction=healthy_prediction,
        machine_id="pump-healthy-001",
        anomalous_sensor_count=0,
    )
    healthy_incidents = engine.evaluate(healthy_context)
    if not healthy_incidents:
        print("No incidents — machine is healthy.")
    else:
        print(f"Unexpected incidents: {len(healthy_incidents)}")
        for inc in healthy_incidents:
            print(f"  {inc.rule_id}: {inc.severity}")


if __name__ == "__main__":
    main()
