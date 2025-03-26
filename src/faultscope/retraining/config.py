"""Configuration for the FaultScope retraining (MLOps) pipeline.

All settings are read from environment variables prefixed with
``FAULTSCOPE_`` and optionally from a ``.env`` file.

Example::

    from faultscope.retraining.config import RetrainingConfig

    cfg = RetrainingConfig()
    print(cfg.mlflow_tracking_uri)
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrainingConfig(BaseSettings):
    """Full configuration for the retraining pipeline.

    Attributes
    ----------
    mlflow_tracking_uri:
        URI of the MLflow tracking server.
    mlflow_experiment_name:
        MLflow experiment name where retraining runs are logged.
    drift_ks_p_value_threshold:
        KS-test p-value below which a feature is flagged as drifted.
        Lower threshold = more sensitive detector.
    drift_error_increase_threshold:
        Fractional increase in MAE that triggers concept-drift alert.
        0.20 means a 20% increase over the baseline MAE.
    retrain_schedule_cron:
        Cron expression for the scheduled retraining job.
    comparison_significance:
        Alpha level for paired t-test when comparing models.
    auto_promote:
        When ``True``, winning challenger is promoted automatically.
        When ``False`` (default), promotion requires manual approval.
    db_host:
        PostgreSQL / TimescaleDB hostname.
    db_port:
        PostgreSQL port.
    db_name:
        Database name.
    db_user:
        Database login user.
    db_password:
        Database password (never logged).
    """

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "faultscope-production"
    drift_ks_p_value_threshold: float = 0.05
    drift_error_increase_threshold: float = 0.20
    retrain_schedule_cron: str = "0 2 * * 0"
    comparison_significance: float = 0.05
    auto_promote: bool = False
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "faultscope"
    db_user: str = "faultscope"
    db_password: SecretStr

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_",
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
