"""Alerting service configuration.

All settings are read from environment variables prefixed with
``FAULTSCOPE_`` or from a ``.env`` file in the working directory.
No credentials are hardcoded anywhere in this module.

Example::

    from faultscope.alerting.config import AlertingConfig

    cfg = AlertingConfig()
    pool = await asyncpg.create_pool(cfg.db_async_url)
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlertingConfig(BaseSettings):
    """Runtime configuration for the FaultScope alerting service.

    All fields without defaults **must** be supplied via environment
    variables or a ``.env`` file.  ``SecretStr`` fields are never
    included in log output or repr strings.

    Environment variables
    ---------------------
    FAULTSCOPE_KAFKA_BOOTSTRAP_SERVERS
        Comma-separated Kafka broker list, e.g. ``kafka:29092``.
    FAULTSCOPE_DB_PASSWORD
        PostgreSQL password for the ``faultscope`` database user.
    """

    # ------------------------------------------------------------------ #
    # HTTP server
    # ------------------------------------------------------------------ #
    host: str = "0.0.0.0"  # noqa: S104  # nosec B104
    port: int = 8001

    # ------------------------------------------------------------------ #
    # Kafka
    # ------------------------------------------------------------------ #
    kafka_bootstrap_servers: str
    kafka_consumer_group: str = "faultscope-alerting"
    topic_rul_predictions: str = "faultscope.predictions.rul"
    topic_incidents: str = "faultscope.incidents.triggered"

    # ------------------------------------------------------------------ #
    # PostgreSQL / TimescaleDB
    # ------------------------------------------------------------------ #
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "faultscope"
    db_user: str = "faultscope"
    db_password: SecretStr

    # ------------------------------------------------------------------ #
    # Alert aggregation
    # ------------------------------------------------------------------ #
    aggregation_window_s: int = 300

    # ------------------------------------------------------------------ #
    # Email (SMTP via aiosmtplib)
    # ------------------------------------------------------------------ #
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: SecretStr = SecretStr("")
    email_from: str = ""
    email_recipients: list[str] = []

    # ------------------------------------------------------------------ #
    # Slack (incoming webhook)
    # ------------------------------------------------------------------ #
    slack_webhook_url: SecretStr = SecretStr("")
    slack_channel: str = "#equipment-alerts"
    slack_mention: str = "@on-call"

    # ------------------------------------------------------------------ #
    # Generic webhook
    # ------------------------------------------------------------------ #
    webhook_url: str = ""

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    log_level: str = "INFO"
    log_format: str = "json"
    otel_enabled: bool = False
    otel_endpoint: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def db_async_url(self) -> str:
        """Return an asyncpg-compatible DSN for this service.

        Returns
        -------
        str
            URL of the form
            ``postgresql://user:password@host:port/dbname``.
        """
        pwd = self.db_password.get_secret_value()
        return (
            f"postgresql://{self.db_user}:{pwd}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
