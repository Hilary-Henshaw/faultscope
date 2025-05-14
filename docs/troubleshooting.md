# Troubleshooting

## Quick Diagnostic

Run the health check script to get a summary of all service states:

```bash
make health
# or
python scripts/health_check.py
```

Check logs for a specific service:

```bash
docker compose logs --tail=100 inference
docker compose logs --tail=100 streaming
```

## Kafka Issues

### Consumer not receiving messages

**Symptom**: Streaming or alerting service starts but processes no messages.

**Diagnosis**:
```bash
# Check topic exists and has messages
docker compose exec kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --describe --topic faultscope.sensors.readings

# Check consumer group lag
docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group faultscope-consumers
```

**Common causes**:
- Topics not created yet — run `make run-infra` and wait for `kafka-init` to complete
- Consumer group offset reset needed — if the topic was recreated, the stored offset is stale:
  ```bash
  docker compose exec kafka kafka-consumer-groups.sh \
    --bootstrap-server localhost:9092 \
    --group faultscope-consumers \
    --reset-offsets --to-latest --all-topics --execute
  ```
- Wrong `FAULTSCOPE_KAFKA_BOOTSTRAP_SERVERS` — inside containers use `kafka:9092`, outside use `localhost:9092`

### Producer failing with `KafkaPublishError`

**Symptom**: Ingestion service logs `KafkaPublishError` with connection refused.

**Diagnosis**:
```bash
docker compose ps kafka
docker compose logs kafka | tail -50
```

**Fix**: Kafka may still be starting. The ingestion service uses tenacity retries (3 attempts, exponential backoff). If it fails all retries, restart it after Kafka is healthy:
```bash
docker compose restart ingestion
```

### High consumer lag

**Symptom**: `faultscope_stream_latency_ms` p95 is high; Grafana shows growing lag.

**Fix**: Scale up the streaming service (each replica handles a subset of partitions):
```bash
docker compose up -d --scale streaming=3
```

Ensure `FAULTSCOPE_KAFKA_CONSUMER_GROUP` is the same across all replicas.

## Database Issues

### Connection pool exhausted

**Symptom**: `asyncpg.exceptions.TooManyConnectionsError` in logs.

**Fix**:
1. Reduce `FAULTSCOPE_DB_POOL_MAX` if running many service replicas
2. Increase `max_connections` in PostgreSQL config
3. Add PgBouncer as a connection pooler in front of TimescaleDB

### TimescaleDB not accepting connections

**Symptom**: Services start but immediately log `DatabaseError: connection refused`.

**Diagnosis**:
```bash
docker compose exec timescaledb pg_isready -U faultscope -d faultscope
docker compose logs timescaledb | tail -30
```

**Common causes**:
- DB still initializing — wait for the health check to pass (up to 60 s on first run)
- Wrong password — verify `FAULTSCOPE_DB_PASSWORD` matches what was used when the volume was first created
- Corrupted volume — delete and recreate: `docker compose down -v && make run-infra`

### Hypertable chunks missing

**Symptom**: Queries return no rows even though data was inserted.

**Diagnosis**:
```sql
SELECT * FROM timescaledb_information.hypertables;
SELECT * FROM timescaledb_information.chunks ORDER BY range_start DESC LIMIT 10;
```

If hypertables are missing, the `init.sql` may not have run. Check:
```bash
docker compose logs timescaledb | grep "init.sql"
```

Re-initialize by dropping and recreating the volume.

## Inference Service Issues

### 503 Service Unavailable on prediction endpoints

**Symptom**: All prediction requests return 503.

**Cause**: `ModelVersionStore` has not yet loaded a model from MLflow.

**Diagnosis**:
```bash
curl http://localhost:8000/ready
# Returns {"status": "loading"} if models not yet loaded
docker compose logs inference | grep -E "model|MLflow|load"
```

**Fix**:
1. Ensure MLflow is running: `docker compose ps mlflow`
2. Ensure a trained model exists: `make train` or `python scripts/train_models.py`
3. Force a refresh: `POST http://localhost:8000/api/v1/models/refresh` with the API key

### 401 Unauthorized

**Symptom**: API returns `{"detail": "Invalid or missing API key"}`.

**Fix**: Include the `X-API-Key` header with the value of `FAULTSCOPE_INFERENCE_API_KEY`:
```bash
curl -H "X-API-Key: $FAULTSCOPE_INFERENCE_API_KEY" \
  http://localhost:8000/api/v1/predict/remaining-life \
  -d '{"machine_id": "M-001", "sensor_readings": {...}}'
```

### 429 Too Many Requests

**Symptom**: API returns 429 after many rapid requests.

**Fix**: The default rate limit is 100 requests/minute per IP. Increase via `FAULTSCOPE_INFERENCE_RATE_LIMIT_PER_MIN` or use request batching (`/api/v1/predict/batch`).

### Model predictions seem wrong / constant

**Symptom**: RUL predictions are always the same value regardless of input.

**Cause**: Model may be loaded from a bad checkpoint, or input features are being normalized incorrectly.

**Diagnosis**:
```bash
# Check which model version is active
curl -H "X-API-Key: $FAULTSCOPE_INFERENCE_API_KEY" \
  http://localhost:8000/api/v1/models
```

**Fix**: Retrain the model with `make train` and trigger a refresh.

## Alerting Issues

### No notifications received

**Symptom**: Incidents appear in the alerting API but no emails/Slack messages.

**Diagnosis**:
```bash
docker compose logs alerting | grep -E "notify|email|slack|webhook|error"
```

**Common causes**:
- SMTP credentials not set — `FAULTSCOPE_ALERTING_SMTP_HOST` is empty, notifier silently skipped
- Slack webhook URL expired — regenerate in Slack app settings
- Cooldown active — per-(machine, rule) cooldown (`FAULTSCOPE_ALERTING_COOLDOWN_S`, default 300 s) prevents repeated notifications

### Rules not triggering

**Symptom**: Predictions look critical but no incidents are created.

**Diagnosis**:
```bash
# Check that predictions are reaching the alerting consumer
docker compose logs alerting | grep "prediction"

# Check rule configuration
curl http://localhost:8001/api/v1/rules
```

**Common causes**:
- Consumer group lag — alerting service has not caught up to latest predictions
- Rule disabled — check `enabled` field in rule config
- Suppressor active — maintenance mode or quiet hours may be suppressing alerts:
  ```bash
  curl http://localhost:8001/api/v1/machines/M-001/maintenance
  ```

## Training Issues

### MLflow experiment not found

**Symptom**: `TrainingOrchestrator` raises `MlflowException`.

**Fix**: Ensure MLflow tracking server is running and accessible:
```bash
docker compose ps mlflow
curl http://localhost:5000/health
```

Set `FAULTSCOPE_TRAINING_MLFLOW_URI=http://localhost:5000` for local runs outside Docker.

### Training OOM (out of memory)

**Symptom**: Training process is killed by the OS or Docker OOM killer.

**Fix**:
1. Reduce `FAULTSCOPE_TRAINING_BATCH_SIZE` (try 64 or 32)
2. Reduce `FAULTSCOPE_TRAINING_SEQUENCE_LENGTH` (try 15)
3. Increase Docker memory limit in Docker Desktop settings

### Feature store empty

**Symptom**: `TrainingOrchestrator` logs "no feature snapshots found".

**Fix**: Run the feature pipeline to populate `feature_snapshots`:
```bash
python -m faultscope.features
```
This requires that the streaming service has been running and has populated `computed_features`.

## Dashboard Issues

### Streamlit shows "Service unavailable" badges

**Symptom**: Overview page shows red badges for Inference and/or Alerting.

**Fix**: The dashboard makes HTTP health checks to the API services. Ensure they are running:
```bash
docker compose ps inference alerting
```

### Grafana panels show "No data"

**Symptom**: Grafana panels are empty or show "No data" after startup.

**Causes**:
1. **Prometheus not scraping yet** — wait 30 s and refresh
2. **Wrong datasource UID** — verify datasources in Grafana Settings → Data sources
3. **TimescaleDB password not set** — Grafana uses `${FAULTSCOPE_DB_PASSWORD}` from the container environment; confirm it is passed through in `compose.yml`

## Performance Tuning

### High streaming latency

| Metric | Target | Action if exceeded |
|---|---|---|
| `faultscope_stream_latency_ms` p95 | < 200 ms | Scale streaming replicas |
| Kafka consumer lag | < 1000 messages | Scale streaming replicas |
| DB write throughput | > 5000 rows/s | Increase `FAULTSCOPE_STREAMING_WRITE_BUFFER_SIZE` |

### High inference latency

| Metric | Target | Action if exceeded |
|---|---|---|
| `/predict/remaining-life` p95 | < 100 ms | Reduce MC Dropout passes; add Redis caching |
| `/predict/batch` (100 samples) p95 | < 500 ms | Increase `FAULTSCOPE_INFERENCE_WORKERS` |

## Getting More Help

1. Check the logs: `docker compose logs <service>`
2. Enable debug logging: `FAULTSCOPE_LOG_LEVEL=DEBUG`
3. Enable console log format for development: `FAULTSCOPE_LOG_FORMAT=console`
4. Open an issue at the project repository with the output of `make health` and relevant log excerpts.
