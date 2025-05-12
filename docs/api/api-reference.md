# API Reference

FaultScope exposes two HTTP APIs:

- **Inference API** — `http://localhost:8000` — ML predictions
- **Alerting API** — `http://localhost:8001` — incident management

Interactive documentation (Swagger UI) is available at `/docs` on each service.

## Authentication

The Inference API requires an API key on all endpoints except `/health`, `/ready`, and `/metrics`.

```
X-API-Key: <value of FAULTSCOPE_INFERENCE_API_KEY>
```

The Alerting API is unauthenticated by default (intended for internal network use). Add an API gateway or network policy in production.

---

## Inference API (port 8000)

### Health

#### `GET /health`

Returns service liveness. No authentication required.

**Response 200**:
```json
{"status": "ok", "service": "faultscope-inference"}
```

#### `GET /ready`

Returns readiness (models loaded). No authentication required.

**Response 200** (ready):
```json
{"status": "ready", "models_loaded": true}
```

**Response 503** (models loading):
```json
{"status": "loading", "models_loaded": false}
```

---

### Predictions

#### `POST /api/v1/predict/remaining-life`

Predict remaining useful life (RUL) in cycles for a single sample.

**Request body**:
```json
{
  "machine_id": "M-001",
  "sensor_readings": {
    "sensor_2": 641.82,
    "sensor_3": 1589.7,
    "sensor_4": 1400.6,
    "sensor_7": 554.36,
    "sensor_8": 2388.02,
    "sensor_9": 9046.19,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.1,
    "sensor_14": 8138.62,
    "sensor_15": 8.4195,
    "sensor_17": 392.0,
    "sensor_20": 38.83,
    "sensor_21": 23.4190
  },
  "operational_setting_1": -0.0007,
  "operational_setting_2": -0.0004,
  "operational_setting_3": 100.0
}
```

**Fields**:

| Field | Type | Required | Description |
|---|---|---|---|
| `machine_id` | string | yes | Machine identifier |
| `sensor_readings` | object | yes | Map of sensor name to float value |
| `operational_setting_1` | float | no | Altitude setting |
| `operational_setting_2` | float | no | TRA setting |
| `operational_setting_3` | float | no | Throttle setting |

**Response 200**:
```json
{
  "machine_id": "M-001",
  "predicted_rul": 87.3,
  "confidence_lower": 72.1,
  "confidence_upper": 103.8,
  "confidence_level": 0.9,
  "model_version": "3",
  "predicted_at": "2026-03-31T10:00:00Z"
}
```

**Error responses**:

| Code | Cause |
|---|---|
| 401 | Missing or invalid API key |
| 422 | Invalid request body (missing required fields, wrong types) |
| 503 | Models not yet loaded |

---

#### `POST /api/v1/predict/health-status`

Classify current equipment health.

**Request body**:
```json
{
  "machine_id": "M-001",
  "sensor_readings": {
    "sensor_2": 641.82,
    "sensor_3": 1589.7
  }
}
```

**Response 200**:
```json
{
  "machine_id": "M-001",
  "health_label": "degrading",
  "probabilities": {
    "healthy": 0.12,
    "degrading": 0.71,
    "critical": 0.14,
    "imminent_failure": 0.03
  },
  "model_version": "2",
  "predicted_at": "2026-03-31T10:00:00Z"
}
```

Health labels in order of severity: `healthy` → `degrading` → `critical` → `imminent_failure`.

---

#### `POST /api/v1/predict/batch`

Predict RUL and health status for multiple samples in one request.

**Request body**:
```json
{
  "samples": [
    {
      "machine_id": "M-001",
      "sensor_readings": {"sensor_2": 641.82}
    },
    {
      "machine_id": "M-002",
      "sensor_readings": {"sensor_2": 700.0}
    }
  ]
}
```

Maximum 100 samples per request.

**Response 200**:
```json
{
  "results": [
    {
      "machine_id": "M-001",
      "predicted_rul": 87.3,
      "health_label": "degrading",
      "confidence_lower": 72.1,
      "confidence_upper": 103.8,
      "model_version": "3",
      "predicted_at": "2026-03-31T10:00:00Z"
    }
  ],
  "batch_size": 2,
  "processed_at": "2026-03-31T10:00:00Z"
}
```

**Error responses**:

| Code | Cause |
|---|---|
| 422 | More than 100 samples in batch |

---

### Model Catalog

#### `GET /api/v1/models`

Return active model versions.

**Response 200**:
```json
{
  "lifespan_predictor": {
    "version": "3",
    "stage": "Production",
    "loaded_at": "2026-03-31T09:00:00Z"
  },
  "condition_classifier": {
    "version": "2",
    "stage": "Production",
    "loaded_at": "2026-03-31T09:00:00Z"
  }
}
```

#### `POST /api/v1/models/refresh`

Force immediate model version check and hot-swap from MLflow registry.

**Response 200**:
```json
{"status": "refresh_triggered"}
```

---

## Alerting API (port 8001)

### Health

#### `GET /health`

**Response 200**:
```json
{"status": "ok", "service": "faultscope-alerting"}
```

#### `GET /ready`

**Response 200**:
```json
{"status": "ready", "kafka_connected": true, "db_connected": true}
```

---

### Predictions (Evaluation)

#### `POST /api/v1/evaluate`

Evaluate a prediction payload against all enabled rules and return any triggered incidents.

**Request body**: same schema as `RulPrediction` Kafka message.

```json
{
  "machine_id": "M-001",
  "predicted_rul": 18.0,
  "health_label": "critical",
  "anomaly_score": 0.91,
  "confidence_lower": 10.0,
  "confidence_upper": 26.0,
  "predicted_at": "2026-03-31T10:00:00Z"
}
```

**Response 200**:
```json
{
  "incidents_triggered": 2,
  "incidents": [
    {
      "incident_id": "INC-20260331-0042",
      "machine_id": "M-001",
      "rule_id": "critical-rul-threshold",
      "severity": "critical",
      "triggered_at": "2026-03-31T10:00:00Z"
    }
  ]
}
```

---

### Incidents

#### `GET /api/v1/incidents`

List incidents with optional filters.

**Query parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `machine_id` | string | — | Filter by machine |
| `severity` | string | — | `info`, `warning`, `critical` |
| `status` | string | `open` | `open`, `acknowledged`, `closed` |
| `limit` | integer | 50 | Max results (1–500) |
| `offset` | integer | 0 | Pagination offset |

**Response 200**:
```json
{
  "incidents": [
    {
      "incident_id": "INC-20260331-0042",
      "machine_id": "M-001",
      "rule_id": "critical-rul-threshold",
      "rule_name": "Critical RUL Threshold",
      "severity": "critical",
      "status": "open",
      "triggered_at": "2026-03-31T10:00:00Z",
      "acknowledged_at": null,
      "closed_at": null,
      "context": {
        "predicted_rul": 18.0,
        "threshold": 20
      }
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

#### `POST /api/v1/incidents/{incident_id}/acknowledge`

Acknowledge an incident.

**Response 200**:
```json
{
  "incident_id": "INC-20260331-0042",
  "status": "acknowledged",
  "acknowledged_at": "2026-03-31T10:05:00Z"
}
```

**Error responses**:

| Code | Cause |
|---|---|
| 404 | Incident not found |
| 409 | Incident already closed |

#### `POST /api/v1/incidents/{incident_id}/close`

Close a resolved incident.

**Request body** (optional):
```json
{"resolution_note": "Replaced bearing assembly."}
```

**Response 200**:
```json
{
  "incident_id": "INC-20260331-0042",
  "status": "closed",
  "closed_at": "2026-03-31T10:15:00Z"
}
```

---

### Alert Rules

#### `GET /api/v1/rules`

List all configured detection rules.

**Response 200**:
```json
{
  "rules": [
    {
      "rule_id": "critical-rul-threshold",
      "name": "Critical RUL Threshold",
      "description": "RUL below 20 cycles",
      "condition_type": "RUL_BELOW",
      "threshold": 20,
      "severity": "critical",
      "cooldown_s": 300,
      "enabled": true
    }
  ]
}
```

#### `GET /api/v1/rules/{rule_id}`

Get a single rule by ID.

**Response 404** if not found:
```json
{"detail": "Rule 'unknown-rule' not found"}
```

---

### Maintenance Mode

#### `GET /api/v1/machines/{machine_id}/maintenance`

Check whether a machine is in maintenance mode.

**Response 200**:
```json
{"machine_id": "M-001", "maintenance_mode": false}
```

#### `POST /api/v1/machines/{machine_id}/maintenance`

Enable maintenance mode (suppresses all alerts for this machine).

**Response 200**:
```json
{"machine_id": "M-001", "maintenance_mode": true}
```

#### `DELETE /api/v1/machines/{machine_id}/maintenance`

Disable maintenance mode.

**Response 200**:
```json
{"machine_id": "M-001", "maintenance_mode": false}
```

---

## Error Format

All error responses use the standard FastAPI format:

```json
{
  "detail": "Human-readable error message"
}
```

Validation errors (422) include field-level details:

```json
{
  "detail": [
    {
      "loc": ["body", "samples"],
      "msg": "List must have at most 100 items",
      "type": "too_long"
    }
  ]
}
```

## Rate Limiting

The Inference API enforces a rate limit of 100 requests per minute per IP address (configurable via `FAULTSCOPE_INFERENCE_RATE_LIMIT_PER_MIN`). When exceeded:

**Response 429**:
```json
{"error": "Rate limit exceeded: 100 per 1 minute"}
```

The response includes standard rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1743415260
Retry-After: 47
```

## Request Tracing

Every response includes an `X-Request-ID` header for distributed tracing:
```
X-Request-ID: 01HX7VGQ3K5T8N9P2MBRJ4WQEZ
```

Pass this ID in bug reports or when correlating logs.
