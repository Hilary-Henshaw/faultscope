"""Unit tests for RuleEvaluationEngine.

Covers all ConditionType branches, cooldown enforcement, disabled
rules, simultaneous multi-rule firing, and no-fire scenarios.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from faultscope.alerting.engine.evaluator import (
    RuleEvaluationEngine,
    TriggeredIncident,
)
from faultscope.alerting.rules import (
    DEFAULT_RULES,
    ConditionType,
    DetectionRule,
    EvaluationContext,
    Severity,
)
from faultscope.common.kafka.schemas import RulPrediction


def _prediction(**overrides: object) -> RulPrediction:
    """Build a baseline healthy RulPrediction with optional field overrides."""
    base: dict[str, object] = {
        "machine_id": "ENG_001",
        "predicted_at": datetime.now(tz=UTC),
        "rul_cycles": 200.0,
        "rul_hours": 400.0,
        "rul_lower_bound": 180.0,
        "rul_upper_bound": 220.0,
        "health_label": "healthy",
        "health_probabilities": {
            "healthy": 0.90,
            "degrading": 0.07,
            "critical": 0.02,
            "imminent_failure": 0.01,
        },
        "anomaly_score": 0.05,
        "confidence": 0.95,
        "rul_model_version": "v1.0.0",
        "health_model_version": "v1.0.0",
    }
    base.update(overrides)
    return RulPrediction(**base)  # type: ignore[arg-type]


@pytest.mark.unit
class TestRuleEvaluationEngine:
    """Unit tests for RuleEvaluationEngine."""

    @pytest.fixture
    def engine(self) -> RuleEvaluationEngine:
        return RuleEvaluationEngine(DEFAULT_RULES)

    # ── RUL threshold rules ───────────────────────────────────────────

    def test_rul_critical_rule_fires_below_threshold(
        self,
        engine: RuleEvaluationEngine,
        sample_rul_prediction: RulPrediction,
    ) -> None:
        """RUL=5 cycles should trigger rul_critical rule."""
        prediction = sample_rul_prediction.model_copy(
            update={"rul_cycles": 5.0}
        )
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "rul_critical" in rule_ids

    def test_rul_warning_rule_fires_in_correct_range(
        self,
        engine: RuleEvaluationEngine,
        sample_rul_prediction: RulPrediction,
    ) -> None:
        """RUL=20 → rul_warning fires, but rul_critical must not fire."""
        prediction = sample_rul_prediction.model_copy(
            update={
                "rul_cycles": 20.0,
                "anomaly_score": 0.0,
                "health_label": "degrading",
            }
        )
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "rul_warning" in rule_ids
        assert "rul_critical" not in rule_ids

    def test_rul_critical_does_not_fire_above_threshold(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """RUL=15 is above the 10-cycle critical threshold → no critical."""
        prediction = _prediction(
            rul_cycles=15.0,
            anomaly_score=0.0,
            health_label="healthy",
        )
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "rul_critical" not in rule_ids

    # ── Anomaly score rules ───────────────────────────────────────────

    def test_anomaly_critical_fires_at_threshold(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """anomaly_score=0.95 (> 0.9 threshold) triggers anomaly_critical."""
        prediction = _prediction(
            rul_cycles=200.0,
            anomaly_score=0.95,
            health_label="healthy",
        )
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "anomaly_critical" in rule_ids

    def test_anomaly_critical_does_not_fire_below_threshold(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """anomaly_score=0.85 is below 0.9 threshold → no anomaly_critical."""
        prediction = _prediction(
            rul_cycles=200.0,
            anomaly_score=0.85,
            health_label="healthy",
        )
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "anomaly_critical" not in rule_ids

    # ── Health label rule ─────────────────────────────────────────────

    def test_health_imminent_failure_fires_for_correct_label(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """health_label=imminent_failure triggers health_imminent rule."""
        prediction = _prediction(
            rul_cycles=3.0,
            anomaly_score=0.5,
            health_label="imminent_failure",
            health_probabilities={
                "healthy": 0.0,
                "degrading": 0.0,
                "critical": 0.1,
                "imminent_failure": 0.9,
            },
        )
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "health_imminent" in rule_ids

    def test_health_imminent_does_not_fire_for_healthy_label(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """health_label=healthy must not trigger health_imminent rule."""
        prediction = _prediction(health_label="healthy", anomaly_score=0.0)
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "health_imminent" not in rule_ids

    # ── Cooldown enforcement ──────────────────────────────────────────

    def test_cooldown_prevents_duplicate_triggers(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """Same rule for same machine must not fire again within cooldown."""
        prediction = _prediction(rul_cycles=5.0)
        ctx = EvaluationContext()

        # First evaluation — should fire.
        incidents_1 = engine.evaluate(prediction, ctx)
        ids_1 = [i.rule.rule_id for i in incidents_1]
        assert "rul_critical" in ids_1

        # Immediate re-evaluation — must be suppressed by cooldown.
        incidents_2 = engine.evaluate(prediction, ctx)
        ids_2 = [i.rule.rule_id for i in incidents_2]
        assert "rul_critical" not in ids_2

    def test_cooldown_expires_and_allows_retriggering(self) -> None:
        """After cooldown elapses the rule fires again."""
        short_cooldown_rule = DetectionRule(
            rule_id="test_short_cooldown",
            rule_name="Short Cooldown Test",
            description="Rule with 1 second cooldown for testing.",
            severity=Severity.WARNING,
            condition_type=ConditionType.RUL_BELOW,
            thresholds={"threshold": 10},
            cooldown_s=1,
        )
        engine = RuleEvaluationEngine([short_cooldown_rule])
        prediction = _prediction(rul_cycles=5.0)
        ctx = EvaluationContext()

        # First fire.
        i1 = engine.evaluate(prediction, ctx)
        assert any(i.rule.rule_id == "test_short_cooldown" for i in i1)

        # Manually advance the last_trigger time to simulate expiry.
        key = (prediction.machine_id, "test_short_cooldown")
        engine._last_trigger[key] = datetime.now(tz=UTC) - timedelta(seconds=2)

        # Second fire after simulated expiry.
        i2 = engine.evaluate(prediction, ctx)
        assert any(i.rule.rule_id == "test_short_cooldown" for i in i2)

    # ── Disabled rule ─────────────────────────────────────────────────

    def test_disabled_rule_never_fires(self) -> None:
        """A rule with enabled=False must never produce an incident."""
        disabled_rule = DetectionRule(
            rule_id="disabled_test",
            rule_name="Disabled Rule",
            description="Should never fire.",
            severity=Severity.CRITICAL,
            condition_type=ConditionType.RUL_BELOW,
            thresholds={"threshold": 9999},
            cooldown_s=0,
            enabled=False,
        )
        engine = RuleEvaluationEngine([disabled_rule])
        prediction = _prediction(rul_cycles=0.0)
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        assert incidents == []

    # ── Multi-rule simultaneous firing ───────────────────────────────

    def test_multiple_rules_can_fire_simultaneously(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """Both rul_critical and anomaly_critical can fire for same event."""
        prediction = _prediction(
            rul_cycles=5.0,
            anomaly_score=0.95,
            health_label="imminent_failure",
        )
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = {i.rule.rule_id for i in incidents}
        assert "rul_critical" in rule_ids
        assert "anomaly_critical" in rule_ids

    # ── Healthy high-RUL machine ──────────────────────────────────────

    def test_no_rules_fire_for_healthy_machine_high_rul(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """A healthy machine with RUL=200 must generate zero incidents."""
        prediction = _prediction(
            rul_cycles=200.0,
            anomaly_score=0.01,
            health_label="healthy",
        )
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        assert incidents == []

    # ── Incident shape ────────────────────────────────────────────────

    def test_triggered_incident_title_contains_machine_id(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """TriggeredIncident.title must include the machine_id."""
        prediction = _prediction(
            machine_id="TEST_MACHINE",
            rul_cycles=5.0,
        )
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        critical_incidents = [
            i for i in incidents if i.rule.rule_id == "rul_critical"
        ]
        assert len(critical_incidents) == 1
        assert "TEST_MACHINE" in critical_incidents[0].title

    def test_triggered_incident_details_contains_rul_cycles(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """TriggeredIncident.details must carry rul_cycles snapshot."""
        prediction = _prediction(rul_cycles=5.0)
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        critical = next(
            i for i in incidents if i.rule.rule_id == "rul_critical"
        )
        assert critical.details["rul_cycles"] == 5.0

    # ── RUL drop rate rule ────────────────────────────────────────────

    def test_rapid_degradation_fires_when_rul_drops_fast(self) -> None:
        """RUL drop > 5 cycles triggers rapid_degradation rule."""
        engine = RuleEvaluationEngine(DEFAULT_RULES)
        prediction = _prediction(rul_cycles=40.0)
        # previous_rul=50, current=40 → drop=10 which exceeds threshold=5.
        ctx = EvaluationContext(previous_rul=50.0)
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "rapid_degradation" in rule_ids

    def test_rapid_degradation_skipped_with_no_previous_rul(self) -> None:
        """Without previous_rul the drop-rate rule cannot fire."""
        engine = RuleEvaluationEngine(DEFAULT_RULES)
        prediction = _prediction(rul_cycles=40.0)
        ctx = EvaluationContext(previous_rul=None)
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "rapid_degradation" not in rule_ids

    # ── Multi-sensor rule ─────────────────────────────────────────────

    def test_multi_sensor_anomaly_fires_when_threshold_met(self) -> None:
        """active_sensor_count >= 3 triggers multi_sensor_anomaly rule."""
        engine = RuleEvaluationEngine(DEFAULT_RULES)
        prediction = _prediction(rul_cycles=200.0, anomaly_score=0.0)
        ctx = EvaluationContext(active_sensor_count=4)
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "multi_sensor_anomaly" in rule_ids

    def test_multi_sensor_anomaly_does_not_fire_below_threshold(
        self,
    ) -> None:
        """active_sensor_count=1 must not trigger multi_sensor_anomaly."""
        engine = RuleEvaluationEngine(DEFAULT_RULES)
        prediction = _prediction(rul_cycles=200.0, anomaly_score=0.0)
        ctx = EvaluationContext(active_sensor_count=1)
        incidents = engine.evaluate(prediction, ctx)
        rule_ids = [i.rule.rule_id for i in incidents]
        assert "multi_sensor_anomaly" not in rule_ids

    # ── Engine construction ───────────────────────────────────────────

    def test_engine_filters_disabled_rules_at_init(self) -> None:
        """Disabled rules must be excluded from _rules at construction time."""
        disabled = DetectionRule(
            rule_id="disabled_init_test",
            rule_name="Disabled",
            description="Disabled rule.",
            severity=Severity.INFO,
            condition_type=ConditionType.RUL_BELOW,
            thresholds={"threshold": 9999},
            cooldown_s=0,
            enabled=False,
        )
        engine = RuleEvaluationEngine([disabled])
        assert len(engine._rules) == 0

    def test_severity_of_critical_incident_is_critical(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """rul_critical incident must have Severity.CRITICAL."""
        prediction = _prediction(rul_cycles=5.0)
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        critical = next(
            (i for i in incidents if i.rule.rule_id == "rul_critical"), None
        )
        assert critical is not None
        assert critical.severity == Severity.CRITICAL

    def test_triggered_incident_is_dataclass_instance(
        self,
        engine: RuleEvaluationEngine,
    ) -> None:
        """evaluate() returns TriggeredIncident instances."""
        prediction = _prediction(rul_cycles=5.0)
        ctx = EvaluationContext()
        incidents = engine.evaluate(prediction, ctx)
        for incident in incidents:
            assert isinstance(incident, TriggeredIncident)
