"""Model training demo.

Generates synthetic sensor data and trains both FaultScope models
(LifespanPredictor LSTM + ConditionClassifier RandomForest) without
requiring a running database or Kafka cluster.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------
MAX_RUL = 125


def generate_dataset(
    n_machines: int,
    n_cycles: int,
    n_sensors: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic degradation dataset.

    Each machine starts healthy (RUL = n_cycles) and degrades linearly
    to failure (RUL = 0), with Gaussian noise on each sensor.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for machine_idx in range(n_machines):
        machine_id = f"demo-machine-{machine_idx:03d}"
        # Vary degradation rate per machine
        noise_scale = rng.uniform(0.5, 2.0)

        for cycle in range(1, n_cycles + 1):
            rul = n_cycles - cycle
            # Sensors degrade smoothly toward failure
            degradation = 1.0 - (rul / n_cycles)

            readings: dict[str, Any] = {
                "machine_id": machine_id,
                "cycle": cycle,
                "rul": min(rul, MAX_RUL),
            }

            for s in range(n_sensors):
                baseline = 500.0 + s * 30.0
                trend = degradation * (20.0 + s * 5.0)
                noise = rng.normal(0.0, noise_scale * (1.0 + s * 0.1))
                readings[f"sensor_{s + 1}"] = baseline + trend + noise

            rows.append(readings)

    return pd.DataFrame(rows)


def add_health_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign health labels based on RUL thresholds."""
    conditions = [
        df["rul"] < 25,
        df["rul"] < 50,
        df["rul"] < 80,
    ]
    labels = ["imminent_failure", "critical", "degrading"]
    df["health_label"] = np.select(conditions, labels, default="healthy")
    return df


def train_val_test_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Machine-level stratified split (no shuffle, preserves time order)."""
    machines = df["machine_id"].unique()
    n = len(machines)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_machines = set(machines[-n_test:])
    val_machines = set(machines[-(n_test + n_val) : -n_test])

    test_df = df[df["machine_id"].isin(test_machines)]
    val_df = df[df["machine_id"].isin(val_machines)]
    train_df = df[
        ~df["machine_id"].isin(test_machines | val_machines)
    ]
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Sequence preparation for LSTM
# ---------------------------------------------------------------------------
def prepare_sequences(
    df: pd.DataFrame,
    sensor_cols: list[str],
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences grouped by machine."""
    X_seqs: list[np.ndarray] = []
    y_vals: list[float] = []

    for _, group in df.groupby("machine_id", sort=False):
        features = group[sensor_cols].values
        targets = group["rul"].values

        for i in range(len(features) - sequence_length + 1):
            X_seqs.append(features[i : i + sequence_length])
            y_vals.append(targets[i + sequence_length - 1])

    return np.array(X_seqs, dtype=np.float32), np.array(
        y_vals, dtype=np.float32
    )


# ---------------------------------------------------------------------------
# LifespanPredictor training (simplified LSTM)
# ---------------------------------------------------------------------------
def train_lifespan_predictor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensor_cols: list[str],
    sequence_length: int,
    epochs: int,
    batch_size: int,
    output_dir: pathlib.Path,
) -> dict[str, float]:
    """Train the LSTM-based RUL predictor."""
    try:
        import tensorflow as tf  # type: ignore[import-untyped]
        from tensorflow import keras  # type: ignore[import-untyped]
    except ImportError:
        print(
            "  TensorFlow not installed — skipping LifespanPredictor. "
            "Install with: pip install tensorflow"
        )
        return {}

    print("Training LifespanPredictor (LSTM)...")

    X_train, y_train = prepare_sequences(
        train_df, sensor_cols, sequence_length
    )
    X_val, y_val = prepare_sequences(val_df, sensor_cols, sequence_length)
    X_test, y_test = prepare_sequences(test_df, sensor_cols, sequence_length)

    # Normalize
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    n_features = len(sensor_cols)
    inputs = keras.Input(shape=(sequence_length, n_features))
    x = keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="huber",
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    # NASA PHM08 score
    diff = y_pred - y_test
    nasa = float(
        np.sum(
            np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
        )
    )

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "nasa_score": nasa}

    print("\n  Test metrics:")
    print(f"    MAE:        {mae:.1f} cycles")
    print(f"    RMSE:       {rmse:.1f} cycles")
    print(f"    R²:          {r2:.2f}")
    print(f"    NASA score: {nasa:.1f}")

    model_dir = output_dir / "lifespan_predictor"
    model.save(str(model_dir))
    np.save(str(output_dir / "scaler_mean.npy"), mean)
    np.save(str(output_dir / "scaler_std.npy"), std)
    print(f"  Saved to {model_dir}")

    return metrics


# ---------------------------------------------------------------------------
# ConditionClassifier training
# ---------------------------------------------------------------------------
def train_condition_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensor_cols: list[str],
    output_dir: pathlib.Path,
) -> dict[str, float]:
    """Train the RandomForest health classifier."""
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        accuracy_score,
        f1_score,
        recall_score,
    )
    from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]

    print("Training ConditionClassifier (Random Forest)...")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["health_label"])
    y_test = label_encoder.transform(test_df["health_label"])

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        oob_score=True,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(train_df[sensor_cols].values, y_train)

    print(f"  OOB score: {clf.oob_score_:.3f}")

    y_pred = clf.predict(test_df[sensor_cols].values)
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # Imminent failure recall (highest priority class)
    imm_idx = list(label_encoder.classes_).index("imminent_failure")
    imm_recall = float(recall_score(y_test, y_pred, labels=[imm_idx], average="micro"))

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "imminent_failure_recall": imm_recall,
    }

    print("\n  Test metrics:")
    print(f"    Accuracy:                {accuracy:.3f}")
    print(f"    Macro F1:                {macro_f1:.3f}")
    print(f"    Weighted F1:             {weighted_f1:.3f}")
    print(f"    Imminent failure recall: {imm_recall:.3f}")

    # Save
    model_path = output_dir / "condition_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "label_encoder": label_encoder}, f)
    print(f"  Saved to {model_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FaultScope model training demo"
    )
    parser.add_argument(
        "--machines", type=int, default=15, help="Number of machines"
    )
    parser.add_argument(
        "--cycles", type=int, default=150, help="Cycles per machine"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="LSTM training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="LSTM input sequence length",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/faultscope-demo-models"),
        help="Directory to save trained models",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== FaultScope Model Training Demo ===\n")
    print("Generating synthetic dataset...")

    df = generate_dataset(
        n_machines=args.machines,
        n_cycles=args.cycles,
    )
    df = add_health_labels(df)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    train_df, val_df, test_df = train_val_test_split(df)

    print(f"  Machines: {args.machines}, Cycles: {args.cycles}")
    print(f"  Total samples: {len(df)}")
    print(
        f"  Training: {len(train_df)} | "
        f"Validation: {len(val_df)} | "
        f"Test: {len(test_df)}"
    )
    print()

    # Train LSTM
    lstm_metrics = train_lifespan_predictor(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        sensor_cols=sensor_cols,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    print()

    # Train Random Forest
    rf_metrics = train_condition_classifier(
        train_df=train_df,
        test_df=test_df,
        sensor_cols=sensor_cols,
        output_dir=args.output_dir,
    )

    print(f"\nModels saved to {args.output_dir}/")
    print("  lifespan_predictor/   (TensorFlow SavedModel)")
    print("  condition_classifier.pkl")
    print(
        "\nTo use these models with the inference service, "
        "register them in MLflow and set the Production stage."
    )


if __name__ == "__main__":
    main()
