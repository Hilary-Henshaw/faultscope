# FaultScope Architecture

## Overview

FaultScope is an event-driven microservices platform for real-time predictive maintenance of industrial equipment. Sensor telemetry flows through a Kafka backbone, is enriched by a stream processor, stored in TimescaleDB, and fed to ML models that predict remaining useful life (RUL) and equipment health. An alerting engine translates predictions into operational incidents, while Streamlit and Grafana dashboards surface the results.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SENSOR LAYER                                  │
│  Physical Sensors / CMAPSS Dataset / MachineSimulator               │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ SensorReading (JSON)
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION SERVICE                                  │
│  MachineSimulator (turbofan/pump/compressor) or CmapssLoader        │
│  SensorPublisher → EventPublisher (aiokafka, acks=all)              │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ faultscope.sensors.readings
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAMING SERVICE                                  │
│  EventSubscriber → DataQualityChecker → TemporalFeatureExtractor    │
│  SpectralFeatureExtractor → CrossSensorCorrelator                   │
│  TimeSeriesWriter → TimescaleDB (sensor_readings, computed_features)│
└──────────┬──────────────────────────────────────────────────────────┘
           │ faultscope.features.computed
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE SERVICE  (port 8000)                    │
│  ModelVersionStore (hot-swap via asyncio.Lock)                      │
│  PredictionEngine: LSTM→RUL + RandomForest→HealthLabel              │
│  FastAPI REST API with ApiKeyMiddleware + slowapi rate limiting      │
└──────────┬──────────────────────────────────────────────────────────┘
           │ faultscope.predictions.rul
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ALERTING SERVICE  (port 8001)                     │
│  RuleEvaluationEngine → IncidentAggregator → IncidentSuppressor     │
│  IncidentCoordinator: EmailNotifier/SlackNotifier/WebhookNotifier   │
│  FastAPI REST API (incidents CRUD + maintenance mode)               │
└─────────────────────────────────────────────────────────────────────┘
           │ faultscope.incidents.triggered
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DASHBOARD LAYER                                    │
│  Streamlit (port 8501): fleet overview, equipment detail, incidents  │
│  Grafana (port 3000): time-series panels, Prometheus metrics        │
└─────────────────────────────────────────────────────────────────────┘

Supporting Infrastructure:
  TimescaleDB ──────────────────── persistent time-series store
  Kafka (KRaft) ────────────────── event backbone (no ZooKeeper)
  Prometheus ───────────────────── metrics scraping
  MLflow + MinIO ───────────────── model registry & artifact store
  Redis ────────────────────────── inference result cache
```

## Kafka Topics

| Topic | Partitions | Retention | Schema |
|---|---|---|---|
| `faultscope.sensors.readings` | 6 | 7 days | `SensorReading` |
| `faultscope.features.computed` | 6 | 7 days | `ComputedFeatures` |
| `faultscope.predictions.rul` | 3 | 14 days | `RulPrediction` |
| `faultscope.incidents.triggered` | 3 | 30 days | Incident JSON |
| `faultscope.dlq` | 1 | 30 days | DLQ envelope |

Partition key is always `machine_id`, ensuring all messages for a given machine land on the same partition and preserve ordering.

## Services

### Ingestion (`faultscope.ingestion`)

Produces synthetic sensor readings using one of three modes:

- **simulation**: `MachineSimulator` with turbofan (21 sensors), pump (9), or compressor (9) profiles. Failed machines are automatically replaced.
- **cmapss**: Streams the NASA C-MAPSS FD001–FD004 dataset on a loop.
- **mixed**: Combines simulation and CMAPSS in parallel.

Key classes: `MachineSimulator`, `CmapssLoader`, `SensorPublisher`.

### Streaming (`faultscope.streaming`)

Consumes raw readings, applies quality checks, extracts three feature families, and persists to TimescaleDB:

| Feature Family | Examples |
|---|---|
| Temporal | `sensor_2_60s_mean`, `sensor_2_60s_rms` |
| Spectral | `sensor_2_dominant_freq_hz`, `sensor_2_spectral_entropy` |
| Correlation | `sensor_2_x_sensor_11_pearson` |

Bad messages (schema errors, quality failures) are forwarded to the DLQ with a JSON envelope containing the original payload and error reason.

### Inference (`faultscope.inference`)

Serves ML predictions over a FastAPI REST API:

- `POST /api/v1/predict/remaining-life` — LSTM with MC Dropout (10 passes), returns RUL estimate + 90% CI
- `POST /api/v1/predict/health-status` — RandomForest, returns label + per-class probabilities
- `POST /api/v1/predict/batch` — up to 100 samples
- `GET /api/v1/models` — active model versions
- `POST /api/v1/models/refresh` — force hot-swap from MLflow

`ModelVersionStore` polls MLflow every 60 s and atomically swaps model weights via `asyncio.Lock`, with zero-downtime shadow loading.

### Alerting (`faultscope.alerting`)

Evaluates 9 built-in detection rules against each prediction:

| Rule | Condition Type |
|---|---|
| Critical RUL | `RUL_BELOW` (threshold 20 cycles) |
| Warning RUL | `RUL_BELOW` (threshold 50 cycles) |
| Rapid degradation | `RUL_DROP_RATE` |
| High anomaly score | `ANOMALY_SCORE_ABOVE` |
| Imminent failure | `HEALTH_LABEL_IS` |
| Multi-sensor fault | `MULTI_SENSOR` |

Per-(machine_id, rule_id) cooldown prevents alert storms. `IncidentSuppressor` respects maintenance windows and quiet hours. Notifications are dispatched concurrently via Email, Slack, and/or Webhook.

### Feature Store (`faultscope.features`)

Offline pipeline run before training:

1. Pull raw features from `computed_features` table
2. Apply RUL labels (capped at `max_rul_cycles=125`)
3. Apply health labels (thresholds: healthy ≥ 80, degrading 50–79, critical 25–49, imminent_failure < 25)
4. Machine-level stratified train/val/test split (no shuffle)
5. Persist to `feature_snapshots` as JSONB

### Training (`faultscope.training`)

`TrainingOrchestrator` runs `LifespanPredictor` (LSTM) and `ConditionClassifier` (RandomForest) in CPU-bound executors:

- Walk-forward cross-validation via `TimeSeriesCrossValidator`
- Metrics logged to MLflow (MAE, RMSE, R², NASA PHM08 score for LSTM; accuracy, macro-F1, imminent-failure recall for RF)
- Best model promoted to MLflow Production stage
- Model card written to `reports/model_card.json`

### Retraining (`faultscope.retraining`)

`RetrainingOrchestrator` implements a 10-step MLOps loop:

1. Fetch recent predictions from DB
2. KS-test for data drift on each feature
3. Detect concept drift (prediction error trend)
4. If drift detected, trigger training
5. Compare challenger vs incumbent (paired t-test)
6. Promote if challenger is significantly better (p < 0.05)
7. Update `model_catalog` table
8. Persist `drift_events` record

## Database Schema

TimescaleDB with 11 tables:

| Table | Type | Description |
|---|---|---|
| `machines` | regular | Machine registry with metadata |
| `sensor_readings` | hypertable | Raw sensor telemetry |
| `computed_features` | hypertable | Streaming-extracted features (JSONB) |
| `feature_snapshots` | hypertable | Versioned training datasets |
| `model_predictions` | hypertable | All inference results with CI |
| `incidents` | regular | Active and resolved alerts |
| `detection_rules` | regular | Rule configuration |
| `service_records` | regular | Maintenance history |
| `model_catalog` | regular | Active model versions per type |
| `training_jobs` | regular | Training run history |
| `drift_events` | regular | Drift detection results |

Hypertables use 1-day chunks. `sensor_readings` and `model_predictions` have 90-day retention policies. Continuous aggregates provide hourly statistics without query overhead.

## Security

- Inference API: `X-API-Key` header verified via `secrets.compare_digest`
- Rate limiting: `slowapi` (100 req/min per IP by default)
- Request tracing: `RequestIdMiddleware` injects `X-Request-ID` on every response
- Alert rules: condition evaluation uses dataclass dispatch, not `eval()`
- Secrets: all credentials via environment variables, never hardcoded

## Observability

- **Logs**: structlog → JSON on all services; dev uses console renderer
- **Traces**: OpenTelemetry with OTLP export (configured via `OTEL_EXPORTER_OTLP_ENDPOINT`)
- **Metrics**: Prometheus scraped from `/metrics` on each service
  - `faultscope_stream_messages_total{status}` — streaming throughput
  - `faultscope_stream_latency_ms` — end-to-end feature extraction latency
  - `faultscope_inference_requests_total{endpoint,status}` — inference call counts
  - `faultscope_model_version{model_type}` — active model versions
