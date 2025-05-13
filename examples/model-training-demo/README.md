# Model Training Demo

Demonstrates the FaultScope training pipeline using synthetic data. You do not
need a running database or Kafka cluster to run this example — it generates
data in-memory and trains both models locally.

## What it trains

1. **LifespanPredictor** (LSTM + attention) — predicts RUL in cycles
2. **ConditionClassifier** (Random Forest) — classifies health state

Results are logged to MLflow if `FAULTSCOPE_TRAINING_MLFLOW_URI` is set,
otherwise metrics are printed to stdout.

## Prerequisites

- FaultScope installed with ML dependencies: `pip install -e ".[dev]"`
- TensorFlow 2.x
- scikit-learn

## Optional: MLflow tracking

Start a local MLflow server to track experiments:

```bash
mlflow server --backend-store-uri sqlite:///mlruns.db \
              --default-artifact-root ./mlartifacts \
              --host 0.0.0.0 --port 5000 &
export FAULTSCOPE_TRAINING_MLFLOW_URI=http://localhost:5000
```

## Run

```bash
python examples/model-training-demo/train_demo.py
```

Options:

```bash
python examples/model-training-demo/train_demo.py \
  --machines 20 \
  --cycles 200 \
  --epochs 5 \
  --output-dir /tmp/faultscope-demo-models
```

## Expected output

```
Generating synthetic dataset...
  Machines: 15, Cycles: 150
  Total samples: 2250
  Training set: 1575 | Validation: 337 | Test: 338

Training LifespanPredictor (LSTM)...
  Epoch 1/5: loss=1842.3 val_loss=1791.2
  Epoch 5/5: loss=312.4 val_loss=388.1

  Test metrics:
    MAE:        28.4 cycles
    RMSE:       41.2 cycles
    R²:          0.71
    NASA score: -4821.3

Training ConditionClassifier (Random Forest)...
  OOB score: 0.834

  Test metrics:
    Accuracy:              0.841
    Macro F1:              0.792
    Weighted F1:           0.847
    Imminent failure recall: 0.923

Models saved to /tmp/faultscope-demo-models/
  lifespan_predictor/  (SavedModel format)
  condition_classifier.pkl
```

## Notes

- Training on synthetic data is fast (< 5 minutes on CPU for default settings)
- The LSTM trains on sequences of 30 cycles; adjust with `--sequence-length`
- For production training, use the full `TrainingOrchestrator` which pulls real
  feature snapshots from TimescaleDB: `make train`
