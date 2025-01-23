"""FaultScope configuration models.

All settings are read from environment variables (prefixed per class)
and optionally from a ``.env`` file.  ``FaultScopeSettings`` is the
single root object services should instantiate; the sub-settings
objects can also be instantiated in isolation for testing.

Usage::

    from faultscope.common.config import FaultScopeSettings

    settings = FaultScopeSettings()
    dsn = settings.database.async_url
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class KafkaSettings(BaseSettings):
    """Kafka connectivity and topic configuration.

    All environment variables are prefixed with ``FAULTSCOPE_KAFKA_``.
    """

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_KAFKA_",
        env_file=".env",
        extra="ignore",
    )

    bootstrap_servers: str
    """Comma-separated ``host:port`` list, e.g. ``localhost:9092``."""

    consumer_group: str
    """Consumer group ID shared by all service replicas."""

    topic_sensor_readings: str = "faultscope.sensors.readings"
    """Raw sensor readings ingested from machines."""

    topic_computed_features: str = "faultscope.features.computed"
    """Engineered feature vectors produced by the streaming service."""

    topic_rul_predictions: str = "faultscope.predictions.rul"
    """RUL and health-label predictions from the inference service."""

    topic_incidents: str = "faultscope.incidents.triggered"
    """Triggered maintenance incidents from the alerting service."""

    topic_dlq: str = "faultscope.dlq"
    """Dead-letter queue for messages that could not be processed."""


class DatabaseSettings(BaseSettings):
    """TimescaleDB / PostgreSQL connectivity configuration.

    All environment variables are prefixed with ``FAULTSCOPE_DB_``.
    """

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_DB_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "localhost"
    """Database server hostname or IP address."""

    port: int = 5432
    """Database server port."""

    name: str = "faultscope"
    """Database name."""

    user: str = "faultscope"
    """Database login user."""

    password: SecretStr
    """Database password (never logged)."""

    pool_size: int = 10
    """Number of persistent connections in the async connection pool."""

    max_overflow: int = 20
    """Maximum extra connections allowed beyond ``pool_size``."""

    @property
    def async_url(self) -> str:
        """Return an asyncpg-compatible SQLAlchemy DSN.

        Returns
        -------
        str
            URL of the form
            ``postgresql+asyncpg://user:password@host:port/dbname``.
        """
        pwd = self.password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class LoggingSettings(BaseSettings):
    """Structured-logging configuration.

    All environment variables are prefixed with ``FAULTSCOPE_LOG_``.
    """

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_LOG_",
        env_file=".env",
        extra="ignore",
    )

    level: str = "INFO"
    """Log level string: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``."""

    format: str = "json"
    """Renderer to use: ``json`` for production, ``console`` for dev."""


class FaultScopeSettings(BaseSettings):
    """Root settings object for FaultScope services.

    Composes ``KafkaSettings``, ``DatabaseSettings``, and
    ``LoggingSettings``.  Services should instantiate this class once
    at startup and pass sub-settings to the components that need them.

    Example::

        settings = FaultScopeSettings()
        publisher = EventPublisher(settings.kafka.bootstrap_servers)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    kafka: KafkaSettings = KafkaSettings()
    database: DatabaseSettings = DatabaseSettings()
    logging: LoggingSettings = LoggingSettings()
