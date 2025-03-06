"""Configuration for the FaultScope inference service.

All settings are loaded from environment variables prefixed with
``FAULTSCOPE_INFERENCE_`` and optionally from a ``.env`` file.

Example::

    from faultscope.inference.config import InferenceConfig

    cfg = InferenceConfig()
    print(cfg.db_async_url)
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceConfig(BaseSettings):
    """Full configuration for the inference service.

    Attributes
    ----------
    host:
        Network interface the uvicorn server binds to.
    port:
        TCP port for the FastAPI application.
    workers:
        Number of uvicorn worker processes.
    api_key:
        Static API key validated by ``ApiKeyMiddleware``.
        Set via ``FAULTSCOPE_INFERENCE_API_KEY``.
    rate_limit_per_minute:
        Maximum requests per minute per IP address.
    model_reload_interval_s:
        Interval in seconds between MLflow Registry polls for new
        production model versions.
    mlflow_tracking_uri:
        URI of the MLflow tracking server.
    mlflow_rul_model_name:
        Registered model name for the LSTM RUL predictor.
    mlflow_health_model_name:
        Registered model name for the RF health classifier.
    kafka_bootstrap_servers:
        Comma-separated Kafka broker addresses.
    topic_rul_predictions:
        Topic where RUL predictions are published.
    db_host:
        TimescaleDB hostname.
    db_port:
        TimescaleDB port.
    db_name:
        Database name.
    db_user:
        Database login user.
    db_password:
        Database password (never logged).
    otel_enabled:
        Enable OpenTelemetry tracing when ``True``.
    """

    host: str = "0.0.0.0"  # noqa: S104  # nosec B104
    port: int = 8000
    workers: int = 4
    api_key: SecretStr
    rate_limit_per_minute: int = 100
    model_reload_interval_s: int = 60
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_rul_model_name: str = "faultscope-lifespan-predictor"
    mlflow_health_model_name: str = "faultscope-condition-classifier"
    kafka_bootstrap_servers: str
    topic_rul_predictions: str = "faultscope.predictions.rul"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "faultscope"
    db_user: str = "faultscope"
    db_password: SecretStr

    otel_enabled: bool = False

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_INFERENCE_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def db_async_url(self) -> str:
        """Return the asyncpg-compatible SQLAlchemy DSN.

        Returns
        -------
        str
            ``postgresql+asyncpg://user:password@host:port/dbname``
        """
        pwd = self.db_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.db_user}:{pwd}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
