"""Feature pipeline configuration.

All settings are read from environment variables with the
``FAULTSCOPE_`` prefix, or from a ``.env`` file in the working
directory.

Usage::

    from faultscope.features.config import FeaturesConfig

    cfg = FeaturesConfig()
    db_url = cfg.db_url
"""

from __future__ import annotations

from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeaturesConfig(BaseSettings):
    """Runtime configuration for the offline feature pipeline.

    Environment variables
    ---------------------
    FAULTSCOPE_DB_HOST
        TimescaleDB hostname (default: ``localhost``).
    FAULTSCOPE_DB_PORT
        TimescaleDB port (default: ``5432``).
    FAULTSCOPE_DB_NAME
        Database name (default: ``faultscope``).
    FAULTSCOPE_DB_USER
        Database login user (default: ``faultscope``).
    FAULTSCOPE_DB_PASSWORD
        Database password — never logged.
    FAULTSCOPE_MLFLOW_TRACKING_URI
        MLflow tracking server URI.
    FAULTSCOPE_DATASET_VERSION
        Dataset version tag written to ``feature_snapshots``.
    FAULTSCOPE_MAX_RUL_CYCLES
        Hard cap applied to computed RUL labels.
    FAULTSCOPE_HEALTH_LABEL_THRESHOLDS
        JSON dict mapping label name to lower-bound RUL threshold.
    FAULTSCOPE_TRAIN_FRACTION
        Fraction of machines assigned to the training split.
    FAULTSCOPE_VAL_FRACTION
        Fraction of machines assigned to the validation split.
    """

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_",
        env_file=".env",
        extra="ignore",
    )

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "faultscope"
    db_user: str = "faultscope"
    db_password: SecretStr

    mlflow_tracking_uri: str = "http://localhost:5000"
    dataset_version: str = "v1"
    max_rul_cycles: int = 125

    health_label_thresholds: dict[str, int] = {
        "healthy": 100,
        "degrading": 50,
        "critical": 20,
        "imminent_failure": 0,
    }

    train_fraction: float = 0.7
    val_fraction: float = 0.15
    # Remaining fraction (1 - train - val) goes to test.

    @computed_field  # type: ignore[misc]
    @property
    def db_url(self) -> str:
        """Return an asyncpg-compatible SQLAlchemy connection URL.

        Returns
        -------
        str
            URL of the form
            ``postgresql+asyncpg://user:pass@host:port/name``.
        """
        pwd = self.db_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.db_user}:{pwd}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
