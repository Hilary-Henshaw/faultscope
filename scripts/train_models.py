#!/usr/bin/env python3
"""Train FaultScope ML models.

Pulls feature snapshots from TimescaleDB, trains both the LSTM
remaining-useful-life regressor and the Random Forest health
classifier, logs every run to MLflow, and prints a training summary
with key metrics.

Usage::

    python scripts/train_models.py
    python scripts/train_models.py --dataset-version=v1
    python scripts/train_models.py --dataset-version=v1 --force-retrain
    python scripts/train_models.py --experiment=faultscope-dev --no-tuning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from faultscope.common.config import DatabaseSettings
from faultscope.common.logging import get_logger
from faultscope.training.config import TrainingConfig

_log = get_logger(__name__)

# Ordered health-label classes matching the DB CHECK constraint.
HEALTH_LABELS: list[str] = [
    "healthy",
    "degrading",
    "critical",
    "imminent_failure",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

async def _load_snapshots(
    db_settings: DatabaseSettings,
    dataset_version: str,
) -> pd.DataFrame:
    """Pull feature snapshots from TimescaleDB into a DataFrame."""
    import asyncpg

    pwd = db_settings.password.get_secret_value()
    dsn = (
        f"postgresql://{db_settings.user}:{pwd}"
        f"@{db_settings.host}:{db_settings.port}"
        f"/{db_settings.name}"
    )

    _log.info(
        "train.loading_snapshots",
        dataset_version=dataset_version,
        host=db_settings.host,
    )

    conn: asyncpg.Connection = await asyncpg.connect(dsn=dsn)  # type: ignore[type-arg]
    try:
        rows = await conn.fetch(
            """
            SELECT
                snapshot_at,
                machine_id,
                feature_vector,
                rul_cycles,
                health_label,
                split
            FROM feature_snapshots
            WHERE dataset_version = $1
            ORDER BY snapshot_at ASC
            """,
            dataset_version,
        )
    finally:
        await conn.close()

    if not rows:
        raise RuntimeError(
            f"No feature snapshots found for version "
            f"'{dataset_version}'.  Run 'make seed' first."
        )

    records = []
    for row in rows:
        vec: dict[str, Any] = json.loads(row["feature_vector"])
        records.append(
            {
                "snapshot_at": row["snapshot_at"],
                "machine_id": row["machine_id"],
                "rul_cycles": row["rul_cycles"],
                "health_label": row["health_label"],
                "split": row["split"],
                **vec,
            }
        )

    df = pd.DataFrame(records)
    _log.info(
        "train.snapshots_loaded",
        rows=len(df),
        machines=df["machine_id"].nunique(),
    )
    return df


def _split_features(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return (X_train, X_val, y_rul_train, y_rul_val,
    y_health_train, y_health_val).
    """
    meta_cols = {"snapshot_at", "machine_id", "rul_cycles",
                 "health_label", "split"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    # Fall back to random split when split column not populated
    if val_df.empty:
        _log.warning(
            "train.no_validation_split",
            hint="Falling back to 20%% random hold-out",
        )
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42
        )

    X_train = train_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_rul_train = train_df["rul_cycles"].fillna(0).to_numpy(
        dtype=np.float32
    )
    y_rul_val = val_df["rul_cycles"].fillna(0).to_numpy(dtype=np.float32)
    y_health_train = train_df["health_label"].fillna("healthy").to_numpy()
    y_health_val = val_df["health_label"].fillna("healthy").to_numpy()

    return (
        X_train, X_val,
        y_rul_train, y_rul_val,
        y_health_train, y_health_val,
    )


# ---------------------------------------------------------------------------
# LSTM training (stubbed to numpy ops when TF unavailable)
# ---------------------------------------------------------------------------

def _train_lstm(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainingConfig,
) -> dict[str, float]:
    """Train the LSTM RUL regressor and return validation metrics."""
    _log.info(
        "train.lstm_start",
        samples=len(X_train),
        sequence_length=cfg.sequence_length,
        layers=cfg.lstm_layers,
        epochs=cfg.max_epochs,
    )

    try:
        import tensorflow as tf
        from tensorflow import keras

        seq_len = cfg.sequence_length
        n_features = X_train.shape[1]

        # Reshape flat vectors into sequences; pad/truncate as needed
        def _to_sequences(
            X: np.ndarray, y: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            seqs, targets = [], []
            for i in range(seq_len, len(X) + 1):
                seqs.append(X[i - seq_len: i])
                targets.append(y[i - 1])
            if not seqs:
                # Fewer samples than seq_len — pad with zeros
                padded = np.zeros(
                    (1, seq_len, n_features), dtype=np.float32
                )
                padded[0, -len(X):] = X
                return padded, y[:1]
            return np.array(seqs, dtype=np.float32), np.array(targets)

        Xs_train, ys_train = _to_sequences(X_train, y_train)
        Xs_val, ys_val = _to_sequences(X_val, y_val)

        inputs = keras.Input(shape=(seq_len, n_features))
        x = inputs
        for i, units in enumerate(cfg.lstm_layers):
            return_seq = i < len(cfg.lstm_layers) - 1
            x = keras.layers.LSTM(
                units,
                return_sequences=return_seq,
                dropout=cfg.lstm_dropout,
                recurrent_dropout=0.0,
            )(x)
        for units in cfg.dense_layers:
            x = keras.layers.Dense(
                units, activation="relu"
            )(x)
            x = keras.layers.Dropout(cfg.dense_dropout)(x)
        output = keras.layers.Dense(1, activation="relu")(x)

        model = keras.Model(inputs, output)
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=cfg.learning_rate
            ),
            loss="huber",
            metrics=["mae"],
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
        )
        history = model.fit(
            Xs_train, ys_train,
            validation_data=(Xs_val, ys_val),
            epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

        preds = model.predict(Xs_val, verbose=0).flatten()
        mae = float(np.mean(np.abs(preds - ys_val)))
        rmse = float(np.sqrt(np.mean((preds - ys_val) ** 2)))
        ss_res = float(np.sum((ys_val - preds) ** 2))
        ss_tot = float(np.sum((ys_val - np.mean(ys_val)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        epochs_trained = len(history.history["loss"])
        _log.info(
            "train.lstm_complete",
            mae=round(mae, 3),
            rmse=round(rmse, 3),
            r2=round(r2, 4),
            epochs=epochs_trained,
        )
        return {"lstm_mae": mae, "lstm_rmse": rmse, "lstm_r2": r2,
                "lstm_epochs": float(epochs_trained)}

    except ImportError:
        _log.warning(
            "train.tensorflow_not_available",
            hint="Falling back to linear baseline for LSTM metrics",
        )
        # Baseline: predict mean RUL
        baseline_pred = np.full_like(y_val, fill_value=y_train.mean())
        mae = float(np.mean(np.abs(baseline_pred - y_val)))
        rmse = float(np.sqrt(np.mean((baseline_pred - y_val) ** 2)))
        return {
            "lstm_mae": mae,
            "lstm_rmse": rmse,
            "lstm_r2": 0.0,
            "lstm_epochs": 0.0,
        }


# ---------------------------------------------------------------------------
# Random Forest training
# ---------------------------------------------------------------------------

def _train_random_forest(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainingConfig,
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Train the Random Forest health classifier."""
    _log.info(
        "train.rf_start",
        samples=len(X_train),
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
    )

    le = LabelEncoder()
    le.classes_ = np.array(HEALTH_LABELS)
    y_enc_train = le.transform(y_train)
    y_enc_val = le.transform(y_val)

    clf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        class_weight=cfg.rf_class_weight,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_enc_train)

    preds = clf.predict(X_val)
    acc = float(accuracy_score(y_enc_val, preds))
    f1 = float(
        f1_score(y_enc_val, preds, average="weighted", zero_division=0)
    )
    report = classification_report(
        y_enc_val, preds,
        target_names=HEALTH_LABELS,
        zero_division=0,
    )
    _log.info(
        "train.rf_complete",
        accuracy=round(acc, 4),
        f1_weighted=round(f1, 4),
    )
    print("\nRandom Forest — Classification Report\n" + report)
    return clf, {"rf_accuracy": acc, "rf_f1_weighted": f1}


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def _print_summary(
    dataset_version: str,
    lstm_metrics: dict[str, float],
    rf_metrics: dict[str, float],
    run_id: str,
    elapsed: float,
) -> None:
    all_metrics = {**lstm_metrics, **rf_metrics}
    print(
        f"\nTraining Summary\n"
        f"{'=' * 50}\n"
        f"  Dataset version : {dataset_version}\n"
        f"  MLflow run ID   : {run_id}\n"
        f"  Elapsed         : {elapsed:.1f}s\n"
    )
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-' * 37}")
    for k, v in sorted(all_metrics.items()):
        print(f"  {k:<25} {v:>10.4f}")
    print()


async def main(
    dataset_version: str,
    force_retrain: bool,
    experiment: str,
    no_tuning: bool,
) -> None:
    """Orchestrate training of both models."""
    cfg = TrainingConfig()  # type: ignore[call-arg]
    cfg = TrainingConfig(  # type: ignore[call-arg]
        mlflow_experiment_name=experiment,
        enable_tuning=cfg.enable_tuning and not no_tuning,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
        sequence_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        early_stop_patience=cfg.early_stop_patience,
        lstm_layers=cfg.lstm_layers,
        lstm_dropout=cfg.lstm_dropout,
        attention_units=cfg.attention_units,
        dense_layers=cfg.dense_layers,
        dense_dropout=cfg.dense_dropout,
        learning_rate=cfg.learning_rate,
        rf_n_estimators=cfg.rf_n_estimators,
        rf_max_depth=cfg.rf_max_depth,
        rf_class_weight=cfg.rf_class_weight,
        n_tuning_trials=cfg.n_tuning_trials,
    )

    db_settings = DatabaseSettings()  # type: ignore[call-arg]

    df = await _load_snapshots(db_settings, dataset_version)
    (
        X_train, X_val,
        y_rul_train, y_rul_val,
        y_health_train, y_health_val,
    ) = _split_features(df)

    _log.info(
        "train.data_ready",
        train_rows=len(X_train),
        val_rows=len(X_val),
        features=X_train.shape[1],
    )

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    start = time.perf_counter()
    run_tags = {
        "dataset_version": dataset_version,
        "force_retrain": str(force_retrain),
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    with mlflow.start_run(tags=run_tags) as run:
        run_id = run.info.run_id

        # Log training config
        mlflow.log_params(
            {
                "dataset_version": dataset_version,
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "n_features": X_train.shape[1],
                "sequence_length": cfg.sequence_length,
                "lstm_layers": str(cfg.lstm_layers),
                "lstm_dropout": cfg.lstm_dropout,
                "batch_size": cfg.batch_size,
                "max_epochs": cfg.max_epochs,
                "learning_rate": cfg.learning_rate,
                "rf_n_estimators": cfg.rf_n_estimators,
                "rf_max_depth": cfg.rf_max_depth,
            }
        )

        # Train LSTM
        lstm_metrics = _train_lstm(
            X_train, X_val, y_rul_train, y_rul_val, cfg
        )
        mlflow.log_metrics(lstm_metrics)

        # Train Random Forest
        rf_model, rf_metrics = _train_random_forest(
            X_train, X_val, y_health_train, y_health_val, cfg
        )
        mlflow.log_metrics(rf_metrics)

        # Log RF model artifact
        mlflow.sklearn.log_model(
            rf_model,
            artifact_path="condition_classifier",
            registered_model_name="faultscope-condition-classifier",
        )

    elapsed = time.perf_counter() - start
    _print_summary(
        dataset_version, lstm_metrics, rf_metrics, run_id, elapsed
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FaultScope LSTM and Random Forest models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-version",
        default="v1",
        help="Feature snapshot version tag to train on",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        default=False,
        help=(
            "Force retraining even if a production-stage model "
            "already exists in MLflow"
        ),
    )
    parser.add_argument(
        "--experiment",
        default="faultscope-production",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        default=False,
        help="Disable Optuna hyperparameter search",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import asyncio

    args = _parse_args()
    try:
        asyncio.run(
            main(
                dataset_version=args.dataset_version,
                force_retrain=args.force_retrain,
                experiment=args.experiment,
                no_tuning=args.no_tuning,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        _log.exception("train.failed", error=str(exc))
        sys.exit(1)
