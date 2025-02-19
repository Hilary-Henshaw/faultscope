"""LSTM model for Remaining Useful Life regression.

``LifespanPredictor`` implements a multi-layer LSTM with a
multi-head attention mechanism, configurable dense head, and
Monte Carlo Dropout for prediction-interval estimation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from tensorflow import keras

from faultscope.common.exceptions import ModelLoadError, ValidationError

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def _cosine_decay_with_warmup(
    learning_rate: float,
    warmup_steps: int = 500,
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Return a cosine-decay schedule with linear warm-up."""
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10_000,
        warmup_steps=warmup_steps,
        warmup_target=learning_rate,
    )


class LifespanPredictor:
    """LSTM with multi-head attention for RUL regression.

    Architecture::

        Input (seq_len, n_features)
          → LSTM 128 (return_sequences=True) + Dropout
          → MultiHeadAttention (64 units, 4 heads)
          → LSTM 64  (return_sequences=True) + Dropout
          → LSTM 32  (return_sequences=False) + Dropout
          → Dense 64 (ReLU) + Dropout
          → Dense 32 (ReLU) + Dropout
          → Dense 16 (ReLU)
          → Dense 1  (Linear)  →  RUL (cycles)

    The model uses Huber loss (delta=1.0) which is robust to the
    outlier RUL values present early in a machine's lifecycle.

    Parameters
    ----------
    sequence_length:
        Number of consecutive time-steps in each input window.
    n_features:
        Number of feature columns per time-step.
    lstm_layers:
        List of hidden-unit counts, one integer per LSTM layer.
    lstm_dropout:
        Dropout rate applied after each LSTM layer.
    attention_units:
        Output dimensionality of the multi-head attention projection.
    dense_layers:
        List of unit counts for the feed-forward head layers.
    dense_dropout:
        Dropout rate applied after each Dense layer.
    learning_rate:
        Initial Adam learning rate.
    """

    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        lstm_layers: list[int],
        lstm_dropout: float,
        attention_units: int,
        dense_layers: list[int],
        dense_dropout: float,
        learning_rate: float,
    ) -> None:
        if not lstm_layers:
            raise ValidationError(
                "lstm_layers must contain at least one element",
                context={"lstm_layers": lstm_layers},
            )
        if not dense_layers:
            raise ValidationError(
                "dense_layers must contain at least one element",
                context={"dense_layers": dense_layers},
            )

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.attention_units = attention_units
        self.dense_layers = dense_layers
        self.dense_dropout = dense_dropout
        self.learning_rate = learning_rate

        self._model: keras.Model | None = None
        self._history: keras.callbacks.History | None = None

    def build(self) -> keras.Model:
        """Construct and compile the Keras model.

        Returns
        -------
        keras.Model
            The compiled model, also stored on ``self._model``.
        """
        inputs = keras.Input(
            shape=(self.sequence_length, self.n_features),
            name="sequence_input",
        )

        # First LSTM layer — return sequences for attention.
        x = keras.layers.LSTM(
            self.lstm_layers[0],
            return_sequences=True,
            name="lstm_0",
        )(inputs)
        x = keras.layers.Dropout(self.lstm_dropout, name="drop_lstm_0")(x)

        # Multi-head attention on first LSTM output.
        n_heads = 4
        key_dim = max(1, self.attention_units // n_heads)
        attn_out = keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=key_dim,
            name="multi_head_attention",
        )(x, x)
        x = keras.layers.Add(name="attention_residual")([x, attn_out])
        x = keras.layers.LayerNormalization(name="attention_norm")(x)

        # Remaining LSTM layers.
        for idx, units in enumerate(self.lstm_layers[1:], start=1):
            return_seq = idx < len(self.lstm_layers) - 1
            x = keras.layers.LSTM(
                units,
                return_sequences=return_seq,
                name=f"lstm_{idx}",
            )(x)
            x = keras.layers.Dropout(
                self.lstm_dropout, name=f"drop_lstm_{idx}"
            )(x)

        # Dense head.
        for idx, units in enumerate(self.dense_layers):
            x = keras.layers.Dense(
                units, activation="relu", name=f"dense_{idx}"
            )(x)
            x = keras.layers.Dropout(
                self.dense_dropout, name=f"drop_dense_{idx}"
            )(x)

        outputs = keras.layers.Dense(1, name="rul_output")(x)

        model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="lifespan_predictor",
        )

        schedule = _cosine_decay_with_warmup(self.learning_rate)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=schedule),
            loss=keras.losses.Huber(delta=1.0),
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )

        self._model = model
        log.info(
            "lifespan_predictor_built",
            total_params=model.count_params(),
            sequence_length=self.sequence_length,
            n_features=self.n_features,
        )
        return model

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "rul_cycles",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a flat DataFrame into (X, y) sequences.

        Groups by ``machine_id`` and creates overlapping sliding
        windows of length ``sequence_length``.  The label for each
        window is the RUL at the *last* time-step.

        Parameters
        ----------
        df:
            Feature DataFrame sorted by ``machine_id``, ``cycle``.
        feature_cols:
            Ordered list of column names to include in X.
        label_col:
            Column name containing the regression target.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``X`` with shape ``(n_samples, sequence_length, n_features)``
            and ``y`` with shape ``(n_samples,)``.

        Raises
        ------
        ValidationError
            If required columns are absent or no windows can be formed.
        """
        missing = set(feature_cols + [label_col, "machine_id"]) - set(
            df.columns
        )
        if missing:
            raise ValidationError(
                "DataFrame missing columns required for sequence preparation",
                context={"missing_columns": sorted(missing)},
            )

        all_X: list[np.ndarray] = []
        all_y: list[float] = []

        for machine_id, group in df.groupby("machine_id", sort=False):
            group_sorted = (
                group.sort_values("cycle")
                if "cycle" in group.columns
                else group
            )
            features = group_sorted[feature_cols].values.astype(np.float32)
            labels = group_sorted[label_col].values.astype(np.float32)
            n = len(features)
            if n < self.sequence_length:
                log.debug(
                    "skipping_machine_too_few_cycles",
                    machine_id=machine_id,
                    n_cycles=n,
                    required=self.sequence_length,
                )
                continue
            for i in range(n - self.sequence_length + 1):
                all_X.append(features[i : i + self.sequence_length])
                all_y.append(float(labels[i + self.sequence_length - 1]))

        if not all_X:
            raise ValidationError(
                "No sequences could be formed; all machines have fewer "
                "cycles than sequence_length",
                context={"sequence_length": self.sequence_length},
            )

        X = np.stack(all_X, axis=0)
        y = np.array(all_y, dtype=np.float32)

        log.info(
            "sequences_prepared",
            n_samples=X.shape[0],
            sequence_length=X.shape[1],
            n_features=X.shape[2],
        )
        return X, y

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> keras.callbacks.History:
        """Train the model with early stopping and LR reduction.

        Parameters
        ----------
        X_train, y_train:
            Training sequences and targets.
        X_val, y_val:
            Validation sequences and targets.

        Returns
        -------
        keras.callbacks.History
            Keras training history object.

        Raises
        ------
        ValidationError
            If the model has not been built first.
        """
        if self._model is None:
            self.build()

        assert self._model is not None  # noqa: S101

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=20,
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_mae",
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        history = self._model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        self._history = history
        final_val_mae = float(history.history["val_mae"][-1])
        final_val_rmse = float(history.history["val_rmse"][-1])
        log.info(
            "lifespan_predictor_trained",
            epochs_completed=len(history.history["loss"]),
            val_mae=round(final_val_mae, 4),
            val_rmse=round(final_val_rmse, 4),
        )
        return history

    def predict(
        self,
        X: np.ndarray,
        mc_passes: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict RUL with Monte Carlo Dropout uncertainty estimates.

        Runs ``mc_passes`` stochastic forward passes with dropout
        active.  Returns the mean prediction and an empirical 90 %
        prediction interval (5th and 95th percentiles).

        Parameters
        ----------
        X:
            Input array of shape ``(n_samples, sequence_length,
            n_features)``.
        mc_passes:
            Number of stochastic forward passes.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(mean_rul, lower_bound, upper_bound)`` each of shape
            ``(n_samples,)``.

        Raises
        ------
        ValidationError
            If the model has not been fitted.
        """
        if self._model is None:
            raise ValidationError(
                "Model must be built and fitted before calling predict",
                context={},
            )

        mc_samples: list[np.ndarray] = []
        for _ in range(mc_passes):
            # training=True keeps dropout active for MC sampling.
            preds: np.ndarray = (
                self._model(X, training=True).numpy().squeeze(-1)
            )
            mc_samples.append(preds)

        stacked = np.stack(mc_samples, axis=0)  # (mc_passes, n_samples)
        mean_rul = stacked.mean(axis=0)
        lower = np.percentile(stacked, 5, axis=0)
        upper = np.percentile(stacked, 95, axis=0)
        return mean_rul, lower, upper

    def save(self, path: str) -> None:
        """Save the Keras model and constructor hyperparameters.

        Saves the Keras model in SavedModel format at ``<path>/model``
        and pickles the constructor kwargs at ``<path>/hparams.pkl``.

        Parameters
        ----------
        path:
            Directory path where the model will be saved.

        Raises
        ------
        ValidationError
            If the model has not been built.
        """
        if self._model is None:
            raise ValidationError(
                "Nothing to save — model has not been built",
                context={"path": path},
            )

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save(str(save_dir / "model"))

        hparams: dict[str, Any] = {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "lstm_layers": self.lstm_layers,
            "lstm_dropout": self.lstm_dropout,
            "attention_units": self.attention_units,
            "dense_layers": self.dense_layers,
            "dense_dropout": self.dense_dropout,
            "learning_rate": self.learning_rate,
        }
        with open(save_dir / "hparams.pkl", "wb") as fh:
            pickle.dump(hparams, fh)

        log.info("lifespan_predictor_saved", path=path)

    @classmethod
    def load(cls, path: str) -> LifespanPredictor:
        """Load a previously saved ``LifespanPredictor``.

        Parameters
        ----------
        path:
            Directory path written by ``save``.

        Returns
        -------
        LifespanPredictor
            Reconstructed predictor with loaded weights.

        Raises
        ------
        ModelLoadError
            If the model or hparams file cannot be loaded.
        """
        save_dir = Path(path)
        hparams_path = save_dir / "hparams.pkl"
        model_path = save_dir / "model"

        if not hparams_path.exists():
            raise ModelLoadError(
                "hparams.pkl not found; the path may be incorrect",
                context={"path": path},
            )
        if not model_path.exists():
            raise ModelLoadError(
                "SavedModel directory not found",
                context={"path": path},
            )

        try:
            with open(hparams_path, "rb") as fh:
                hparams: dict[str, Any] = pickle.load(fh)  # noqa: S301  # nosec B301
        except Exception as exc:
            raise ModelLoadError(
                "Failed to load hparams.pkl",
                context={"path": path, "error": str(exc)},
            ) from exc

        try:
            keras_model = keras.models.load_model(str(model_path))
        except Exception as exc:
            raise ModelLoadError(
                "Failed to load Keras SavedModel",
                context={"path": path, "error": str(exc)},
            ) from exc

        predictor = cls(**hparams)
        predictor._model = keras_model  # noqa: SLF001
        log.info("lifespan_predictor_loaded", path=path)
        return predictor
