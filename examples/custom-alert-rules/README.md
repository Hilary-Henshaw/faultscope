# Custom Alert Rules Example

Demonstrates how to define and integrate custom detection rules into the
FaultScope alerting engine.

## Overview

FaultScope ships with 9 built-in rules covering the most common failure
patterns. You can add your own rules by:

1. Defining a `DetectionRule` dataclass with the appropriate `ConditionType`
2. Adding it to `DEFAULT_RULES` (or loading it from the database)
3. Optionally implementing a new `ConditionType` for logic not covered by the
   built-in condition types

## Built-in condition types

| ConditionType | Triggers when |
|---|---|
| `RUL_BELOW` | Predicted RUL drops below a threshold |
| `ANOMALY_SCORE_ABOVE` | Anomaly score exceeds a threshold |
| `HEALTH_LABEL_IS` | Health label matches a specific value |
| `RUL_DROP_RATE` | RUL decreases faster than N cycles per prediction |
| `MULTI_SENSOR` | N or more sensors simultaneously exceed their thresholds |

## Run

```bash
python examples/custom-alert-rules/custom_rules.py
```

No running services required — this example runs the rule evaluator in-memory.

## Expected output

```
=== Custom Alert Rules Example ===

Evaluating 4 rules against test prediction...

  [PASS] critical-rul (RUL_BELOW ≤ 20)
         Reason: RUL=18.0 is below threshold 20

  [SKIP] warning-rul (RUL_BELOW ≤ 50)
         Reason: Within cooldown window (same machine+rule seen recently)

  [PASS] vibration-anomaly (ANOMALY_SCORE_ABOVE > 0.80)
         Reason: anomaly_score=0.91 exceeds threshold 0.80

  [PASS] pump-imminent-failure (HEALTH_LABEL_IS = imminent_failure)
         Reason: health_label matches 'imminent_failure'

Triggered incidents: 3
  INC-001: critical-rul  [critical]
  INC-002: vibration-anomaly  [warning]
  INC-003: pump-imminent-failure  [critical]
```

## Adding rules to the database

Rules can also be loaded from the `detection_rules` table. Add a row:

```sql
INSERT INTO detection_rules (
    rule_id, name, description,
    condition_type, threshold, severity,
    cooldown_s, enabled
) VALUES (
    'high-temperature',
    'High Temperature Alert',
    'Thermal sensor exceeds safe operating range',
    'ANOMALY_SCORE_ABOVE',
    0.75,
    'warning',
    600,
    true
);
```

The alerting service loads rules from the database at startup and can be
refreshed without a restart via `POST /api/v1/rules/reload`.
