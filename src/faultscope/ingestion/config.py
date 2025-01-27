"""Ingestion service configuration.

All settings are read from environment variables with the
``FAULTSCOPE_`` prefix, or from a ``.env`` file in the working
directory.  No defaults contain credentials.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    """Runtime configuration for the FaultScope ingestion service.

    Environment variables
    ---------------------
    FAULTSCOPE_KAFKA_BOOTSTRAP_SERVERS
        Comma-separated Kafka broker addresses, e.g.
        ``localhost:9092`` or ``kafka:29092``.
    FAULTSCOPE_INGEST_MACHINES
        Number of simulated machines to run concurrently.
    FAULTSCOPE_INGEST_INTERVAL_S
        Wall-clock seconds to wait between emission cycles.
    FAULTSCOPE_DEGRADATION_SEED
        NumPy RNG seed for reproducible degradation curves.
    FAULTSCOPE_ENABLE_CMAPSS
        Set to ``true`` to stream the NASA C-MAPSS dataset
        instead of running the synthetic simulator.
    FAULTSCOPE_CMAPSS_DATA_PATH
        Filesystem path to the directory that contains the
        C-MAPSS ``.txt`` files (``train_FD001.txt``, etc.).
    """

    kafka_bootstrap_servers: str
    topic_sensor_readings: str = "faultscope.sensors.readings"
    num_machines: int = 8
    emit_interval_s: float = 1.0
    degradation_seed: int = 42
    enable_cmapss: bool = False
    cmapss_data_path: str = "data/cmapss"

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_",
        env_file=".env",
        extra="ignore",
    )
