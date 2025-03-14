"""Detection rules REST endpoints for the FaultScope alerting service.

Routes
------
GET  /api/v1/rules          List all configured detection rules
GET  /api/v1/rules/{id}     Retrieve a single rule by ID
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from faultscope.alerting.rules import DEFAULT_RULES, RULES_BY_ID
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/rules", tags=["Rules"])


def _rule_to_dict(rule: object) -> dict[str, object]:
    """Serialise a ``DetectionRule`` to a plain dict.

    Parameters
    ----------
    rule:
        A ``DetectionRule`` instance.

    Returns
    -------
    dict[str, object]
        JSON-serialisable representation of the rule.
    """
    from faultscope.alerting.rules import DetectionRule

    assert isinstance(rule, DetectionRule)
    return {
        "rule_id": rule.rule_id,
        "rule_name": rule.rule_name,
        "description": rule.description,
        "severity": rule.severity.value,
        "condition_type": rule.condition_type.value,
        "thresholds": dict(rule.thresholds),
        "cooldown_s": rule.cooldown_s,
        "enabled": rule.enabled,
    }


@router.get(
    "",
    summary="List all configured detection rules",
)
async def list_rules() -> list[dict[str, object]]:
    """Return all detection rules loaded at startup.

    The list is sorted by severity (critical → warning → info) then
    by rule ID alphabetically within each severity band.

    Returns
    -------
    list[dict[str, object]]
        Serialised rule objects.
    """
    _severity_order = {"critical": 0, "warning": 1, "info": 2}
    sorted_rules = sorted(
        DEFAULT_RULES,
        key=lambda r: (
            _severity_order.get(r.severity.value, 99),
            r.rule_id,
        ),
    )
    return [_rule_to_dict(r) for r in sorted_rules]


@router.get(
    "/{rule_id}",
    summary="Retrieve a single detection rule by ID",
)
async def get_rule(rule_id: str) -> dict[str, object]:
    """Return a single rule by its stable ``rule_id``.

    Parameters
    ----------
    rule_id:
        Unique rule identifier, e.g. ``"rul_critical"``.

    Returns
    -------
    dict[str, object]
        Serialised rule object.

    Raises
    ------
    HTTPException
        404 when no rule with ``rule_id`` is registered.
    """
    rule = RULES_BY_ID.get(rule_id)
    if rule is None:
        _log.warning("rule_not_found", rule_id=rule_id)
        raise HTTPException(
            status_code=404,
            detail=f"Rule '{rule_id}' not found.",
        )
    return _rule_to_dict(rule)
