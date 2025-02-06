"""Streaming service configuration.

All settings are read from environment variables prefixed with
``FAULTSCOPE_`` and optionally from a ``.env`` file.

Usage::

    from faultscope.streaming.config import StreamingConfig

    cfg = StreamingConfig()
    print(cfg.kafka_bootstrap_servers)
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class StreamingConfig(BaseSettings):
    """Configuration for the real-time feature engineering service.

    Reads all values from environment variables prefixed with
    ``FAULTSCOPE_``.  Sensitive values (``db_password``) are handled
    via ``pydantic.SecretStr`` and are never included in log output.
    """

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_",
        env_file=".env",
        extra="ignore",
    )

    # ── Kafka ─────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str
    """Comma-separated ``host:port`` list, e.g. ``localhost:9092``.
    Set via ``FAULTSCOPE_KAFKA_BOOTSTRAP_SERVERS``."""

    kafka_consumer_group: str = "faultscope-streaming"
    """Kafka consumer group shared by all streaming replicas."""

    topic_sensor_readings: str = "faultscope.sensors.readings"
    """Input topic: raw sensor readings produced by the ingestion
    service."""

    topic_computed_features: str = "faultscope.features.computed"
    """Output topic: engineered feature vectors published after each
    window computation."""

    topic_dlq: str = "faultscope.dlq"
    """Dead-letter queue topic for messages that fail quality checks."""

    # ── Database ──────────────────────────────────────────────────────
    db_host: str = "localhost"
    """TimescaleDB hostname."""

    db_port: int = 5432
    """TimescaleDB port."""

    db_name: str = "faultscope"
    """Database name."""

    db_user: str = "faultscope"
    """Database login user."""

    db_password: SecretStr
    """Database password.  Set via ``FAULTSCOPE_DB_PASSWORD``."""

    db_pool_size: int = 10
    """Number of persistent connections kept in the asyncpg pool."""

    # ── Feature extraction ────────────────────────────────────────────
    rolling_windows_s: list[int] = [10, 30, 60, 300]
    """Window durations (seconds) for temporal feature computation.
    Override via ``FAULTSCOPE_STREAM_ROLLING_WINDOWS_S`` as a
    JSON-encoded list, e.g. ``[10,30,60]``."""

    fft_sampling_rate_hz: float = 100.0
    """Assumed sensor sampling rate used for FFT frequency bins."""

    fft_sensors: list[str] = [
        "vibration_rms",
        "vibration_x",
        "vibration_y",
    ]
    """Sensor names for which spectral features are computed."""

    batch_size: int = 100
    """Maximum records buffered before a DB flush is triggered."""

    flush_interval_s: float = 5.0
    """Maximum seconds between DB flushes regardless of batch size."""

    # ── Data quality ──────────────────────────────────────────────────
    max_null_fraction: float = 0.3
    """Fraction of null sensor values above which a reading is
    rejected to the DLQ."""

    max_future_drift_s: float = 300.0
    """Seconds in the future beyond which a timestamp is rejected."""

    min_sensor_count: int = 5
    """Minimum number of sensor keys required for a valid reading."""

    # ── Observability ─────────────────────────────────────────────────
    metrics_port: int = 8002
    """Port on which the Prometheus metrics HTTP server listens."""

    log_level: str = "INFO"
    """Log level: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``."""

    log_format: str = "json"
    """Log renderer: ``json`` (production) or ``console`` (dev)."""

    @property
    def db_async_url(self) -> str:
        """Return a raw asyncpg DSN (without SQLAlchemy driver prefix).

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
