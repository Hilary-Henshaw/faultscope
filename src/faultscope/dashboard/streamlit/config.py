"""Dashboard configuration loaded from environment variables.

All settings are read with the ``FAULTSCOPE_`` prefix so they coexist
with the other service configurations in the same ``.env`` file.

Example::

    from faultscope.dashboard.streamlit.config import DashboardConfig

    cfg = DashboardConfig()
    print(cfg.inference_base_url)
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardConfig(BaseSettings):
    """Runtime configuration for the Streamlit dashboard.

    Attributes
    ----------
    inference_base_url:
        Base URL of the FaultScope inference API (port 8000).
    alerting_base_url:
        Base URL of the FaultScope alerting API (port 8001).
    api_key:
        Bearer token sent in ``Authorization`` headers to the inference
        API.  Sourced from ``FAULTSCOPE_INFERENCE_API_KEY``.
    refresh_interval_s:
        How often the dashboard polls for fresh data (seconds).
    db_host:
        TimescaleDB hostname (used for direct queries if needed).
    db_port:
        TimescaleDB port.
    db_name:
        TimescaleDB database name.
    db_user:
        TimescaleDB login user.
    db_password:
        TimescaleDB password (never logged).
    mlflow_tracking_uri:
        URI for the MLflow tracking server.
    """

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_",
        env_file=".env",
        extra="ignore",
    )

    inference_base_url: str = "http://localhost:8000"
    alerting_base_url: str = "http://localhost:8001"
    api_key: str = ""
    refresh_interval_s: int = 5
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "faultscope"
    db_user: str = "faultscope"
    db_password: SecretStr = SecretStr("")
    mlflow_tracking_uri: str = "http://localhost:5000"
