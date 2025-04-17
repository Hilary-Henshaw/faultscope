"""Unit tests for IncidentAggregator.

Tests verify grouping by machine, flush semantics, buffer clearing,
highest-severity preservation, and rule-id deduplication.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from faultscope.alerting.engine.aggregator import IncidentAggregator
from faultscope.alerting.engine.evaluator import TriggeredIncident
from faultscope.alerting.rules import (
    ConditionType,
    DetectionRule,
    Severity,
)


def _rule(
    rule_id: str = "test_rule",
    severity: Severity = Severity.WARNING,
) -> DetectionRule:
    return DetectionRule(
        rule_id=rule_id,
        rule_name=f"Rule {rule_id}",
        description="Test rule.",
        severity=severity,
        condition_type=ConditionType.RUL_BELOW,
        thresholds={"threshold": 10},
        cooldown_s=60,
    )


def _incident(
    machine_id: str = "ENG_001",
    rule_id: str = "test_rule",
    severity: Severity = Severity.WARNING,
) -> TriggeredIncident:
    rule = _rule(rule_id=rule_id, severity=severity)
    return TriggeredIncident(
        rule=rule,
        machine_id=machine_id,
        title=f"[{severity.value.upper()}] {machine_id}: {rule.rule_name}",
        details={"rul_cycles": 5.0},
        severity=severity,
        triggered_at=datetime.now(tz=UTC),
    )


@pytest.mark.unit
class TestIncidentAggregator:
    """Unit tests for IncidentAggregator."""

    def test_flush_returns_empty_when_no_incidents_added(self) -> None:
        """flush() with nothing added must return an empty list."""
        agg = IncidentAggregator(aggregation_window_s=300)
        groups = agg.flush()
        assert groups == []

    def test_single_incident_is_returned_in_flush(self) -> None:
        """An added incident must be present after flush."""
        agg = IncidentAggregator(aggregation_window_s=300)
        agg.add(_incident())
        groups = agg.flush()
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0].machine_id == "ENG_001"

    def test_incidents_within_window_are_grouped_by_machine(self) -> None:
        """Incidents for the same machine must appear in the same group."""
        agg = IncidentAggregator(aggregation_window_s=300)
        agg.add(_incident("ENG_001", "rule_a", Severity.WARNING))
        agg.add(_incident("ENG_001", "rule_b", Severity.INFO))
        groups = agg.flush()
        # All incidents for ENG_001 must end up in one group.
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_incidents_for_different_machines_in_separate_groups(
        self,
    ) -> None:
        """Incidents for different machines must be in separate groups."""
        agg = IncidentAggregator(aggregation_window_s=300)
        agg.add(_incident("ENG_001", "rule_x", Severity.WARNING))
        agg.add(_incident("ENG_002", "rule_x", Severity.WARNING))
        groups = agg.flush()
        machine_ids = {groups[0][0].machine_id, groups[1][0].machine_id}
        assert machine_ids == {"ENG_001", "ENG_002"}

    def test_flush_clears_buffer(self) -> None:
        """Second flush after the first must return empty list."""
        agg = IncidentAggregator(aggregation_window_s=300)
        agg.add(_incident())
        agg.flush()  # drains buffer
        second_flush = agg.flush()
        assert second_flush == []

    def test_highest_severity_preserved_in_group(self) -> None:
        """When INFO + CRITICAL arrive for same rule → CRITICAL is kept."""
        agg = IncidentAggregator(aggregation_window_s=300)
        agg.add(_incident("ENG_001", "rule_a", Severity.INFO))
        # CRITICAL should replace INFO for the same key.
        agg.add(_incident("ENG_001", "rule_a", Severity.CRITICAL))
        groups = agg.flush()
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0].severity == Severity.CRITICAL

    def test_lower_severity_does_not_downgrade_existing(self) -> None:
        """CRITICAL → INFO for same rule must keep CRITICAL in bucket."""
        agg = IncidentAggregator(aggregation_window_s=300)
        agg.add(_incident("ENG_001", "rule_a", Severity.CRITICAL))
        agg.add(_incident("ENG_001", "rule_a", Severity.INFO))
        groups = agg.flush()
        assert groups[0][0].severity == Severity.CRITICAL

    def test_duplicate_rule_id_deduplicated(self) -> None:
        """Same rule_id for same machine must not produce duplicate rows."""
        agg = IncidentAggregator(aggregation_window_s=300)
        for _ in range(5):
            agg.add(_incident("ENG_001", "rule_a", Severity.WARNING))
        groups = agg.flush()
        # Only one deduplicated entry per rule_id.
        rule_ids = [i.rule.rule_id for i in groups[0]]
        assert rule_ids.count("rule_a") == 1

    def test_multiple_distinct_rule_ids_all_retained(self) -> None:
        """Each distinct rule_id keeps its own deduplicated entry."""
        agg = IncidentAggregator(aggregation_window_s=300)
        rule_ids = [f"rule_{i}" for i in range(4)]
        for rid in rule_ids:
            agg.add(_incident("ENG_001", rid, Severity.WARNING))
        groups = agg.flush()
        assert len(groups) == 1
        returned_ids = {i.rule.rule_id for i in groups[0]}
        assert returned_ids == set(rule_ids)

    def test_flush_groups_sorted_critical_first(self) -> None:
        """Flush output must sort incidents: CRITICAL > WARNING > INFO."""
        agg = IncidentAggregator(aggregation_window_s=300)
        agg.add(_incident("ENG_001", "rule_info", Severity.INFO))
        agg.add(_incident("ENG_001", "rule_warn", Severity.WARNING))
        agg.add(_incident("ENG_001", "rule_crit", Severity.CRITICAL))
        groups = agg.flush()
        severities = [i.severity for i in groups[0]]
        assert severities[0] == Severity.CRITICAL
        assert severities[-1] == Severity.INFO

    def test_add_after_flush_starts_fresh_bucket(self) -> None:
        """Adding incidents after flush must re-populate the buffer."""
        agg = IncidentAggregator(aggregation_window_s=300)
        agg.add(_incident())
        agg.flush()
        agg.add(_incident())
        groups = agg.flush()
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_total_groups_equals_distinct_machine_count(self) -> None:
        """Number of groups must equal the number of distinct machines."""
        agg = IncidentAggregator(aggregation_window_s=300)
        machines = ["ENG_001", "ENG_002", "ENG_003"]
        for mid in machines:
            agg.add(_incident(mid, "rule_a", Severity.WARNING))
        groups = agg.flush()
        assert len(groups) == 3
