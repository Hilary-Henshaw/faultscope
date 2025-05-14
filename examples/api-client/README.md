# API Client Example

Demonstrates calling the FaultScope Inference API from Python using `httpx`.

Covers:
- Single RUL prediction
- Health status classification
- Batch prediction (multiple machines in one request)
- Error handling (auth errors, validation errors, service unavailable)

## Prerequisites

- FaultScope Inference API running: `docker compose up -d inference` (or `make run-all`)
- `httpx` installed: `pip install httpx`

## Environment

```bash
export FAULTSCOPE_INFERENCE_URL=http://localhost:8000
export FAULTSCOPE_INFERENCE_API_KEY=your-api-key-here
```

## Run

```bash
python examples/api-client/predict_rul.py
```

## Expected output

```
=== FaultScope Inference API Client Example ===

[RUL Prediction]
Machine: M-001
Predicted RUL: 87.3 cycles
90% CI: [72.1, 103.8]
Model version: 3

[Health Status]
Machine: M-001
Health label: degrading
Probabilities:
  healthy:           12.1%
  degrading:         70.8%
  critical:          14.2%
  imminent_failure:   2.9%

[Batch Prediction] 3 machines
  M-001: RUL=87.3, health=degrading
  M-002: RUL=42.1, health=critical
  M-003: RUL=112.6, health=healthy
```

## Key patterns

**Authentication**: Pass `X-API-Key` header on every request.

**Retry on 503**: The inference service may return 503 during startup while models load.
The example shows how to implement a simple retry with exponential backoff.

**Batch requests**: Use `/api/v1/predict/batch` when you need predictions for multiple
machines — it's significantly faster than N individual requests.
