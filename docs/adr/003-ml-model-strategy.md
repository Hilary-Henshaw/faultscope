# ADR 003: Dual-Model ML Strategy (LSTM + Random Forest)

**Status**: Accepted
**Date**: 2026-02-01
**Deciders**: FaultScope core team

## Context

FaultScope must solve two distinct predictive maintenance problems:

1. **Remaining Useful Life (RUL) regression** — Predict how many operational cycles remain before failure. Output is a continuous value.

2. **Health classification** — Assign a discrete health state (`healthy`, `degrading`, `critical`, `imminent_failure`). Output is a categorical label with per-class probabilities.

Several model strategies were considered:

**Option A: Single deep learning model** — One LSTM network with dual output heads (regression + classification). Simpler deployment, but regression and classification losses often conflict during training.

**Option B: Two separate models** — One model per task, each optimized independently.

**Option C: Classical models only** — Random Forest for both tasks. Fast, interpretable, no GPU required.

**Option D: Transformer-based model** — State-of-the-art on many time-series benchmarks. Very high computational cost; requires large datasets to outperform LSTM.

## Decision

Use two specialized models:

1. **`LifespanPredictor`**: Stacked LSTM (128→64→32 units) with multi-head attention and Monte Carlo (MC) Dropout for uncertainty quantification. Trained with Huber loss and cosine decay learning rate schedule.

2. **`ConditionClassifier`**: Random Forest with balanced class weights and out-of-bag error estimation. Trained on flattened feature snapshots.

## Rationale

### Why LSTM for RUL?

RUL prediction is fundamentally a sequential problem — the degradation trajectory over the last N cycles matters more than any single snapshot. LSTM captures temporal dependencies that tabular models miss.

Multi-head attention allows the model to weight different time steps independently, improving performance on non-monotonic degradation patterns (e.g., oscillating faults that temporarily recover).

MC Dropout provides calibrated uncertainty estimates at inference time: 10 stochastic forward passes with dropout active yield a distribution of RUL predictions. The 5th and 95th percentiles form the 90% confidence interval reported in the API. This is operationally valuable — operators can see not just "50 cycles remaining" but "between 38 and 64 cycles, 90% confidence."

### Why Random Forest for health classification?

Health classification requires:
- **Interpretability**: Maintenance teams need to understand why a machine was flagged
- **Robustness to imbalance**: `imminent_failure` samples are rare; RF with `class_weight='balanced'` handles this without oversampling
- **Feature importance**: MDI (mean decrease impurity) scores identify which sensors drive the prediction
- **Fast training**: RF trains in seconds on the feature snapshot; LSTM requires minutes to hours

The alternative — a classification head on the LSTM — would tie both tasks to the same training schedule and make it harder to retrain the classifier when new health state definitions are needed.

### Why not a transformer?

Transformer architectures achieve state-of-the-art on many time-series benchmarks, but:

- They require significantly more training data than CMAPSS provides (hundreds of millions of parameters vs. tens of millions for LSTM)
- Inference latency is higher (critical for p95 < 100 ms target)
- The marginal accuracy improvement over LSTM-attention on 21-sensor turbofan data does not justify the operational complexity

A transformer-based upgrade path is noted in the roadmap for when larger datasets are available.

### Separation of concerns

Running two independent models provides operational flexibility:

- The RF classifier can be retrained daily (fast) without touching the LSTM
- Drift detection applies independently: a new sensor pattern might affect classification without changing RUL trajectory
- Model rollback is isolated: if an LSTM update degrades RUL accuracy, the classifier continues serving correct health labels

## Consequences

**Positive**:
- Each model is optimized for its task (Huber loss for robust RUL regression; cross-entropy with balanced weights for classification)
- MC Dropout uncertainty is a first-class API feature, not an afterthought
- RF feature importances provide explainability for audit and debugging
- Independent retraining cycles via the `RetrainingOrchestrator`
- Both models are versioned in MLflow; rollback is a single `promote_to_production` call

**Negative**:
- Two models to maintain, version, and monitor
- `ModelVersionStore` must manage two independent hot-swap slots
- Inference latency is the sum of both model forward passes (mitigated by async parallel execution and Redis caching)
- Feature preprocessing must be consistent between training and inference (resolved by storing the scaler with the model artifact)

**Model Evaluation Criteria**:

| Model | Primary metric | Secondary metrics |
|---|---|---|
| `LifespanPredictor` | MAE (cycles) | RMSE, NASA PHM08 score, R² |
| `ConditionClassifier` | Macro-F1 | Imminent-failure recall (≥ 0.95 required), accuracy, weighted-F1 |

The imminent-failure recall constraint reflects operational priority: missing a true imminent failure (false negative) is far more costly than a false alarm.
