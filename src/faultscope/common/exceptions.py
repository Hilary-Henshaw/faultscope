"""FaultScope exception hierarchy.

All service-specific exceptions inherit from ``FaultScopeError`` so
callers can catch the base type or narrow to a specific failure mode.
Every exception accepts an optional ``context`` mapping that carries
structured metadata (machine IDs, topic names, etc.) useful for
structured log entries.
"""

from __future__ import annotations


class FaultScopeError(Exception):
    """Base exception for all FaultScope errors.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    context:
        Arbitrary key/value pairs attached to the error for structured
        logging (e.g. ``{"machine_id": "M001", "topic": "sensors"}``).
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, object] = context or {}

    def __repr__(self) -> str:
        """Return a detailed string representation including context."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"context={self.context!r})"
        )


class KafkaPublishError(FaultScopeError):
    """Raised when an event cannot be published to a Kafka topic.

    Parameters
    ----------
    message:
        Human-readable description of the publish failure.
    context:
        Structured details, e.g. ``{"topic": "...", "key": "..."}``.
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)


class KafkaConsumeError(FaultScopeError):
    """Raised when messages cannot be consumed from a Kafka topic.

    Parameters
    ----------
    message:
        Human-readable description of the consume failure.
    context:
        Structured details, e.g. ``{"topic": "...", "group_id": "..."}``.
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)


class DatabaseError(FaultScopeError):
    """Raised when a database operation fails.

    Parameters
    ----------
    message:
        Human-readable description of the database failure.
    context:
        Structured details, e.g. ``{"operation": "INSERT", "table": "..."}``.
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)


class ModelLoadError(FaultScopeError):
    """Raised when an ML model artifact cannot be loaded.

    Parameters
    ----------
    message:
        Human-readable description of the load failure.
    context:
        Structured details, e.g. ``{"path": "...", "version": "..."}``.
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)


class ValidationError(FaultScopeError):
    """Raised when input data fails schema or business-rule validation.

    Parameters
    ----------
    message:
        Human-readable description of the validation failure.
    context:
        Structured details, e.g. ``{"field": "rul_cycles", "value": -1}``.
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)


class ConfigurationError(FaultScopeError):
    """Raised when required configuration is missing or invalid.

    Parameters
    ----------
    message:
        Human-readable description of the configuration problem.
    context:
        Structured details, e.g. ``{"env_var": "FAULTSCOPE_DB_HOST"}``.
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
