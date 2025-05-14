# Deployment Guide

## Docker Compose (Quickstart)

The fastest way to run FaultScope is with Docker Compose. All infrastructure and application services are defined in `compose.yml`.

### Prerequisites

- Docker Engine ≥ 24.0
- Docker Compose plugin ≥ 2.20
- 8 GB RAM available to Docker
- Ports 3000, 5000, 5432, 8000, 8001, 8002, 8501, 9000, 9090, 9092 free

### Step 1 — Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   FAULTSCOPE_DB_PASSWORD
#   FAULTSCOPE_INFERENCE_API_KEY
#   FAULTSCOPE_INFERENCE_DB_PASSWORD
#   FAULTSCOPE_MINIO_SECRET_KEY
```

### Step 2 — Start infrastructure

```bash
make run-infra
# Starts: timescaledb, kafka, prometheus, grafana, mlflow, minio, redis
# Wait for all health checks to pass (≈ 30 s)
```

### Step 3 — Start application services

```bash
make run-all
# Starts: ingestion, streaming, inference, alerting, dashboard
```

### Step 4 — Verify

```bash
make health
# Expected: all services show OK
```

Service endpoints after startup:

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| Grafana | http://localhost:3000 (admin/admin) |
| Inference API | http://localhost:8000/docs |
| Alerting API | http://localhost:8001/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| MinIO Console | http://localhost:9001 |

### Stopping

```bash
make stop          # stop containers, preserve volumes
docker compose down -v  # stop and delete all volumes (destroys data)
```

## Environment Variables

See [configuration.md](configuration.md) for the complete reference. At minimum you need:

```dotenv
FAULTSCOPE_DB_PASSWORD=changeme
FAULTSCOPE_INFERENCE_API_KEY=changeme
FAULTSCOPE_INFERENCE_DB_PASSWORD=changeme
FAULTSCOPE_MINIO_SECRET_KEY=changeme
```

## Production Considerations

### Resource Sizing

| Service | CPU | RAM | Notes |
|---|---|---|---|
| TimescaleDB | 2–4 cores | 4–8 GB | Tune `shared_buffers`, `work_mem` |
| Kafka | 2–4 cores | 4 GB | 3-node cluster for HA |
| Inference | 4–8 cores | 8 GB | TensorFlow benefits from many cores |
| Streaming | 1–2 cores | 1 GB | Scales horizontally |
| Alerting | 1 core | 512 MB | Stateless, easily replicated |

### Horizontal Scaling

**Streaming service**: Run multiple replicas sharing the same consumer group (`FAULTSCOPE_KAFKA_CONSUMER_GROUP`). Kafka will distribute partitions automatically.

**Inference service**: Run behind a load balancer. Each replica loads its own model copy. `FAULTSCOPE_INFERENCE_WORKERS` controls Uvicorn workers per process.

**Alerting service**: Stateless with respect to notifications. Each replica independently evaluates rules against the shared DB state.

### Database Tuning

```sql
-- Recommended postgresql.conf overrides for TimescaleDB
shared_buffers = '2GB'
work_mem = '64MB'
maintenance_work_mem = '512MB'
max_connections = 200
wal_compression = on
timescaledb.max_background_workers = 8
```

Add a `timescaledb.conf` file and mount it into the container:

```yaml
# compose.yml override
timescaledb:
  volumes:
    - ./infra/timescaledb/timescaledb.conf:/etc/postgresql/postgresql.conf
  command: postgres -c config_file=/etc/postgresql/postgresql.conf
```

### TLS / HTTPS

For the Inference and Alerting APIs, terminate TLS at a reverse proxy (nginx, Caddy, or a cloud load balancer). Example nginx config fragment:

```nginx
server {
    listen 443 ssl;
    server_name inference.example.com;
    ssl_certificate /etc/letsencrypt/live/inference.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/inference.example.com/privkey.pem;

    location / {
        proxy_pass http://inference:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Kafka Security

For production Kafka, enable SASL/SCRAM authentication:

```dotenv
FAULTSCOPE_KAFKA_SECURITY_PROTOCOL=SASL_SSL
FAULTSCOPE_KAFKA_SASL_MECHANISM=SCRAM-SHA-256
FAULTSCOPE_KAFKA_SASL_USERNAME=faultscope
FAULTSCOPE_KAFKA_SASL_PASSWORD=changeme
```

### Secrets Management

Prefer a dedicated secrets manager over `.env` files:

**HashiCorp Vault**:
```bash
vault kv put secret/faultscope \
  db_password="..." \
  inference_api_key="..." \
  minio_secret_key="..."
```

**AWS Secrets Manager** (ECS/EKS):
```json
{
  "secrets": [
    {"name": "FAULTSCOPE_DB_PASSWORD", "valueFrom": "arn:aws:..."}
  ]
}
```

**Kubernetes Secrets**:
```bash
kubectl create secret generic faultscope-secrets \
  --from-literal=FAULTSCOPE_DB_PASSWORD=changeme \
  --from-literal=FAULTSCOPE_INFERENCE_API_KEY=changeme
```

### Monitoring

- Prometheus scrapes all services on their `/metrics` endpoints (configured in `infra/monitoring/prometheus.yml`)
- Grafana dashboards are provisioned automatically from `src/faultscope/dashboard/grafana/`
- Set up alerting rules in Prometheus or Grafana for critical system metrics (Kafka consumer lag, DB connection pool saturation, model prediction latency p95)

### Backup

**TimescaleDB**:
```bash
pg_dump -Fc -h localhost -U faultscope faultscope > faultscope_$(date +%Y%m%d).dump
```

**MLflow artifacts** (stored in MinIO): use `mc mirror` or enable MinIO bucket replication to a secondary bucket.

### Upgrading

1. Pull new images: `docker compose pull`
2. Apply DB migrations if any: check `CHANGELOG.md` for migration notes
3. Restart services one at a time to avoid downtime: `docker compose up -d --no-deps inference`
4. Verify health: `make health`
