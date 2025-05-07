# Configuration Reference

FaultScope uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) v2. Every service reads its configuration from environment variables at startup. No configuration files are required — copy `.env.example` to `.env` and edit as needed.

## Loading Order

pydantic-settings reads variables in this precedence (highest first):

1. Environment variables set in the shell
2. `.env` file in the working directory
3. Default values in the `BaseSettings` subclass

## Common Settings (`FaultScopeSettings`)

Shared by all services. Prefix: none.

### Kafka (`KafkaSettings`)

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Comma-separated broker list |
| `FAULTSCOPE_KAFKA_SECURITY_PROTOCOL` | `PLAINTEXT` | `PLAINTEXT`, `SSL`, `SASL_SSL` |
| `FAULTSCOPE_KAFKA_SASL_MECHANISM` | — | `PLAIN`, `SCRAM-SHA-256`, `SCRAM-SHA-512` |
| `FAULTSCOPE_KAFKA_SASL_USERNAME` | — | SASL username |
| `FAULTSCOPE_KAFKA_SASL_PASSWORD` | — | SASL password |
| `FAULTSCOPE_KAFKA_TOPIC_READINGS` | `faultscope.sensors.readings` | Raw sensor topic |
| `FAULTSCOPE_KAFKA_TOPIC_FEATURES` | `faultscope.features.computed` | Computed features topic |
| `FAULTSCOPE_KAFKA_TOPIC_PREDICTIONS` | `faultscope.predictions.rul` | RUL predictions topic |
| `FAULTSCOPE_KAFKA_TOPIC_INCIDENTS` | `faultscope.incidents.triggered` | Triggered incidents topic |
| `FAULTSCOPE_KAFKA_TOPIC_DLQ` | `faultscope.dlq` | Dead-letter queue topic |
| `FAULTSCOPE_KAFKA_CONSUMER_GROUP` | `faultscope-consumers` | Consumer group ID |

### Database (`DatabaseSettings`)

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_DB_HOST` | `localhost` | TimescaleDB hostname |
| `FAULTSCOPE_DB_PORT` | `5432` | TimescaleDB port |
| `FAULTSCOPE_DB_NAME` | `faultscope` | Database name |
| `FAULTSCOPE_DB_USER` | `faultscope` | Database user |
| `FAULTSCOPE_DB_PASSWORD` | *(required)* | Database password |
| `FAULTSCOPE_DB_POOL_MIN` | `2` | Minimum asyncpg pool connections |
| `FAULTSCOPE_DB_POOL_MAX` | `10` | Maximum asyncpg pool connections |

The `async_url` property returns `postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}`.

### Logging (`LoggingSettings`)

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `FAULTSCOPE_LOG_FORMAT` | `json` | `json` (production) or `console` (development) |

## Ingestion Service (`IngestionConfig`)

Prefix: `FAULTSCOPE_INGESTION_`

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_INGESTION_MODE` | `simulation` | `simulation`, `cmapss`, or `mixed` |
| `FAULTSCOPE_INGESTION_MACHINE_COUNT` | `10` | Number of simulated machines |
| `FAULTSCOPE_INGESTION_PUBLISH_INTERVAL_MS` | `500` | Milliseconds between readings per machine |
| `FAULTSCOPE_INGESTION_CMAPSS_DATA_DIR` | `data/cmapss` | Path to CMAPSS dataset files |
| `FAULTSCOPE_INGESTION_CMAPSS_SUBSET` | `FD001` | `FD001`, `FD002`, `FD003`, or `FD004` |

## Streaming Service (`StreamingConfig`)

Prefix: `FAULTSCOPE_STREAMING_`

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_STREAMING_METRICS_PORT` | `8002` | Prometheus metrics HTTP port |
| `FAULTSCOPE_STREAMING_WINDOW_SIZES_S` | `[30, 60, 300]` | Feature window sizes in seconds |
| `FAULTSCOPE_STREAMING_WRITE_BUFFER_SIZE` | `500` | Rows buffered before DB flush |
| `FAULTSCOPE_STREAMING_FLUSH_INTERVAL_S` | `5` | Periodic flush interval in seconds |
| `FAULTSCOPE_STREAMING_QUALITY_NULL_THRESHOLD` | `0.2` | Max allowed null fraction |
| `FAULTSCOPE_STREAMING_QUALITY_OUTLIER_ZSCORE` | `5.0` | Z-score threshold for outlier detection |
| `FAULTSCOPE_STREAMING_FUTURE_TIMESTAMP_TOLERANCE_S` | `30` | Seconds ahead considered valid |

## Inference Service (`InferenceConfig`)

Prefix: `FAULTSCOPE_INFERENCE_`

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_INFERENCE_API_KEY` | *(required)* | API key for `X-API-Key` header |
| `FAULTSCOPE_INFERENCE_HOST` | `0.0.0.0` | Bind address |
| `FAULTSCOPE_INFERENCE_PORT` | `8000` | HTTP port |
| `FAULTSCOPE_INFERENCE_WORKERS` | `1` | Uvicorn worker processes |
| `FAULTSCOPE_INFERENCE_DB_HOST` | `localhost` | DB host (may differ from common) |
| `FAULTSCOPE_INFERENCE_DB_NAME` | `faultscope` | DB name |
| `FAULTSCOPE_INFERENCE_DB_USER` | `faultscope` | DB user |
| `FAULTSCOPE_INFERENCE_DB_PASSWORD` | *(required)* | DB password |
| `FAULTSCOPE_INFERENCE_MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `FAULTSCOPE_INFERENCE_MODEL_POLL_INTERVAL_S` | `60` | Model version check frequency |
| `FAULTSCOPE_INFERENCE_RATE_LIMIT_PER_MIN` | `100` | Requests per minute per IP |
| `FAULTSCOPE_INFERENCE_REDIS_URL` | `redis://redis:6379/0` | Cache backend URL |
| `FAULTSCOPE_INFERENCE_CACHE_TTL_S` | `30` | Prediction cache TTL in seconds |

## Alerting Service (`AlertingConfig`)

Prefix: `FAULTSCOPE_ALERTING_`

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_ALERTING_HOST` | `0.0.0.0` | Bind address |
| `FAULTSCOPE_ALERTING_PORT` | `8001` | HTTP port |
| `FAULTSCOPE_ALERTING_COOLDOWN_S` | `300` | Per-(machine, rule) cooldown in seconds |
| `FAULTSCOPE_ALERTING_AGGREGATION_WINDOW_S` | `60` | Incident aggregation window |

### Email Notifier

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_ALERTING_SMTP_HOST` | — | SMTP server hostname |
| `FAULTSCOPE_ALERTING_SMTP_PORT` | `587` | SMTP port |
| `FAULTSCOPE_ALERTING_SMTP_USER` | — | SMTP login username |
| `FAULTSCOPE_ALERTING_SMTP_PASSWORD` | — | SMTP login password |
| `FAULTSCOPE_ALERTING_SMTP_FROM` | — | Sender address |
| `FAULTSCOPE_ALERTING_ALERT_RECIPIENTS` | — | Comma-separated recipient addresses |

### Slack Notifier

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_ALERTING_SLACK_WEBHOOK_URL` | — | Slack Incoming Webhook URL |

### Webhook Notifier

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_ALERTING_WEBHOOK_URL` | — | Generic webhook endpoint |
| `FAULTSCOPE_ALERTING_WEBHOOK_SECRET` | — | HMAC signing secret |

## Dashboard Service

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_INFERENCE_URL` | `http://inference:8000` | Inference API base URL |
| `FAULTSCOPE_ALERTING_URL` | `http://alerting:8001` | Alerting API base URL |
| `FAULTSCOPE_DASHBOARD_API_KEY` | — | API key for calling Inference API |

## Training / Feature Store

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_TRAINING_MLFLOW_URI` | `http://mlflow:5000` | MLflow tracking server |
| `FAULTSCOPE_TRAINING_EXPERIMENT_NAME` | `faultscope-v1` | MLflow experiment name |
| `FAULTSCOPE_TRAINING_MAX_RUL_CYCLES` | `125` | RUL cap for label clipping |
| `FAULTSCOPE_TRAINING_SEQUENCE_LENGTH` | `30` | LSTM input window length |
| `FAULTSCOPE_TRAINING_BATCH_SIZE` | `256` | Training batch size |
| `FAULTSCOPE_TRAINING_MAX_EPOCHS` | `50` | Maximum training epochs |

## Observability

| Variable | Default | Description |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTLP gRPC endpoint |
| `OTEL_SERVICE_NAME` | *(set per service)* | Service name in traces |
| `FAULTSCOPE_TELEMETRY_ENABLED` | `true` | Enable OpenTelemetry export |

## MinIO / Object Store

| Variable | Default | Description |
|---|---|---|
| `FAULTSCOPE_MINIO_ENDPOINT` | `minio:9000` | MinIO API endpoint |
| `FAULTSCOPE_MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `FAULTSCOPE_MINIO_SECRET_KEY` | *(required)* | MinIO secret key |
| `FAULTSCOPE_MINIO_BUCKET` | `faultscope-artifacts` | Default bucket name |

## Security Notes

- **Never commit secrets** to version control. Use `.env` (which is `.gitignore`-d) or a secrets manager (Vault, AWS Secrets Manager, etc.)
- The `.env.example` file contains only placeholder values — copy it, fill in real credentials, and keep the result outside the repository.
- For production, prefer injecting secrets via the container orchestrator (Kubernetes Secrets, Docker Swarm secrets) rather than `.env` files.
