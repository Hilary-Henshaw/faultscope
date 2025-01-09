# Changelog

All notable changes to FaultScope are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.0] - 2026-03-31

### Added

- **Async ingestion service** — `SensorPublisher` wraps aiokafka to stream raw sensor readings
  to `faultscope.sensors.readings` at 10 000+ messages/second, keyed by `machine_id` for strict
  per-machine ordering.

- **Machine simulator** — `MachineSimulator` generates realistic multi-sensor data for four
  equipment types (turbofan, pump, compressor, motor) with four configurable degradation curve
  shapes: linear, exponential, step, and oscillating.

- **NASA C-MAPSS dataset loader** — `CMAPSSLoader` streams the 21-sensor turbofan dataset from
  disk and publishes it through the same ingestion pipeline as simulated data, enabling
  model training on real-world degradation trajectories.

- **Streaming feature-engineering pipeline** — rolling-window computation of temporal features
  (mean, std, RMS, kurtosis, peak-to-peak), spectral features (dominant frequency, spectral
  entropy, band power via FFT), and cross-sensor Pearson correlation coefficients, all updated
  in real time on every incoming reading batch.

- **TimescaleDB schema** — 11-table schema with hypertables for `sensor_readings`,
  `computed_features`, `feature_snapshots`, and `model_predictions`; continuous aggregate
  materialized views for 1-minute and 1-hour bucketed sensor averages; configurable data
  retention policies (90 days for raw readings, 365 days for predictions).

- **LSTM remaining-useful-life regressor** — stacked LSTM architecture (128 → 64 → 32 units)
  with multi-head attention, dense layers (64 → 32 → 16), Huber loss, and Adam optimiser with
  early stopping; trained on sequences of engineered feature vectors.

- **Random Forest health classifier** — 200-tree ensemble with class-weight balancing to handle
  imbalanced health-label distributions; classifies equipment into four states: healthy,
  degrading, critical, and imminent\_failure.

- **Monte Carlo Dropout uncertainty quantification** — inference service runs T=30 stochastic
  forward passes with dropout active and reports mean, 5th-percentile lower bound, and
  95th-percentile upper bound for every RUL prediction.

- **MLflow experiment tracking** — every training run logs parameters, metrics (MAE, RMSE, R²
  for LSTM; accuracy and weighted F1 for RF), model artifacts, and a classification report;
  models are registered in the MLflow Model Registry under `faultscope-lifespan-predictor` and
  `faultscope-condition-classifier`.

- **Zero-downtime model hot-swap** — inference service polls the MLflow registry for new
  production-stage models and swaps them into the active request path atomically without
  restarting or dropping in-flight requests; configurable polling interval.

- **FastAPI inference service** (port 8000) — REST endpoints for RUL prediction, health
  classification, batch prediction, and model introspection; X-API-Key authentication;
  `slowapi` rate limiting; OpenAPI / Swagger documentation auto-generated.

- **Alert engine service** (port 8001) — evaluates nine detection rules on every prediction
  event: three RUL threshold rules (info/warning/critical), two anomaly score rules, two
  health-label rules, a rapid-degradation rate rule, and a multi-sensor anomaly rule; per-rule
  cooldown windows prevent alert floods.

- **Incident lifecycle management** — incidents persist to TimescaleDB with full state
  transitions (open → acknowledged → closed); REST endpoints for listing, acknowledging, and
  closing incidents; acknowledgement audit trail with `acknowledged_by` and timestamp.

- **Multi-channel notification dispatch** — simultaneous delivery to Email (aiosmtplib + STARTTLS),
  Slack (slack-sdk incoming webhooks), and arbitrary HTTP webhooks; configurable per-channel
  severity filter; retry with exponential back-off on transient failures.

- **Statistical drift detection** — `DriftMonitor` runs two-sample KS tests on each feature
  column to detect covariate shift and a one-sided Welch's t-test on absolute prediction errors
  to detect concept drift; drift events are logged to TimescaleDB and trigger automated
  retraining jobs.

- **Streamlit operator dashboard** (port 8501) — live fleet health overview, per-machine RUL
  trend charts with confidence bands, active incident inbox, and prediction history;
  configurable auto-refresh interval.

- **Prometheus + Grafana observability** — all services export metrics to `/metrics`; pre-built
  Grafana dashboards covering throughput, error rates, inference latency percentiles, model
  prediction distributions, and incident volume over time.

- **OpenTelemetry distributed tracing** — end-to-end trace propagation across ingestion,
  streaming, inference, and alerting services; compatible with any OTLP-capable backend
  (Jaeger, Tempo, etc.).

- **Structured logging** — `structlog` with configurable JSON renderer (production) or
  colour-console renderer (development); every log event carries `service`, `machine_id`, and
  `trace_id` context fields.

- **Full Docker Compose stack** — single `compose.yml` brings up Kafka, Zookeeper, TimescaleDB,
  Redis, MinIO, MLflow, Prometheus, Grafana, and all application services; topic creation and
  schema initialisation run automatically on first start.

- **Developer tooling** — `Makefile` with targets for setup, lint, typecheck, unit tests,
  integration tests, e2e tests, coverage, seeding, training, and health checks; pre-commit hooks
  for ruff formatting and mypy.

---

## Links

[Unreleased]: https://github.com/your-org/faultscope/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/faultscope/releases/tag/v1.0.0
