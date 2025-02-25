"""Training pipeline configuration.

All settings are read from environment variables prefixed with
``FAULTSCOPE_TRAINING_``, or from a ``.env`` file in the working
directory.

Usage::

    from faultscope.training.config import TrainingConfig

    cfg = TrainingConfig()
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    """Runtime configuration for the FaultScope model training pipeline.

    Environment variables
    ---------------------
    FAULTSCOPE_TRAINING_MLFLOW_TRACKING_URI
        MLflow tracking server URI.
    FAULTSCOPE_TRAINING_MLFLOW_EXPERIMENT_NAME
        Name of the MLflow experiment used for all training runs.
    FAULTSCOPE_TRAINING_SEQUENCE_LENGTH
        Number of time-steps per LSTM input window.
    FAULTSCOPE_TRAINING_BATCH_SIZE
        Mini-batch size for LSTM training.
    FAULTSCOPE_TRAINING_MAX_EPOCHS
        Maximum number of training epochs.
    FAULTSCOPE_TRAINING_EARLY_STOP_PATIENCE
        Epochs without improvement before early stopping fires.
    FAULTSCOPE_TRAINING_LSTM_LAYERS
        JSON list of hidden units per LSTM layer, e.g. [128,64,32].
    FAULTSCOPE_TRAINING_LSTM_DROPOUT
        Dropout rate applied after each LSTM layer.
    FAULTSCOPE_TRAINING_ATTENTION_UNITS
        Units in the multi-head attention layer.
    FAULTSCOPE_TRAINING_DENSE_LAYERS
        JSON list of units per Dense layer after the LSTM stack.
    FAULTSCOPE_TRAINING_DENSE_DROPOUT
        Dropout rate applied after each Dense layer.
    FAULTSCOPE_TRAINING_LEARNING_RATE
        Initial Adam learning rate.
    FAULTSCOPE_TRAINING_RF_N_ESTIMATORS
        Number of trees in the Random Forest.
    FAULTSCOPE_TRAINING_RF_MAX_DEPTH
        Maximum tree depth in the Random Forest.
    FAULTSCOPE_TRAINING_RF_CLASS_WEIGHT
        Class weight strategy (``"balanced"`` or ``"balanced_subsample"``).
    FAULTSCOPE_TRAINING_ENABLE_TUNING
        Set to ``true`` to run Optuna hyperparameter search first.
    FAULTSCOPE_TRAINING_N_TUNING_TRIALS
        Number of Optuna trials when tuning is enabled.
    """

    model_config = SettingsConfigDict(
        env_prefix="FAULTSCOPE_TRAINING_",
        env_file=".env",
        extra="ignore",
    )

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "faultscope-production"

    # ── LSTM / sequence model ─────────────────────────────────────
    sequence_length: int = 50
    batch_size: int = 32
    max_epochs: int = 100
    early_stop_patience: int = 20

    lstm_layers: list[int] = [128, 64, 32]
    lstm_dropout: float = 0.2
    attention_units: int = 64
    dense_layers: list[int] = [64, 32, 16]
    dense_dropout: float = 0.3
    learning_rate: float = 1e-3

    # ── Random Forest ─────────────────────────────────────────────
    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    rf_class_weight: str = "balanced"

    # ── Optuna tuning ─────────────────────────────────────────────
    enable_tuning: bool = False
    n_tuning_trials: int = 30
