# Development Guide

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- Docker + Docker Compose plugin
- GNU Make

## Setup

```bash
git clone <repo-url> faultscope
cd faultscope
make setup       # creates .venv with uv and installs all extras
make run-infra   # starts TimescaleDB, Kafka, MLflow, MinIO, Prometheus, Grafana
```

The `setup` target runs:
```bash
uv venv .venv
uv pip install -e ".[dev,lint]"
pre-commit install
```

## Running Services Locally

Each service is a Python module with a `__main__.py` entry point. Run them outside Docker with:

```bash
# Terminal 1: ingestion (simulation mode)
python -m faultscope.ingestion --mode simulation --log-format console

# Terminal 2: streaming processor
python -m faultscope.streaming

# Terminal 3: inference API
python -m faultscope.inference

# Terminal 4: alerting service
python -m faultscope.alerting
```

Ensure your shell has the required environment variables set (source `.env` or use `make env`).

The Streamlit dashboard runs separately:

```bash
streamlit run src/faultscope/dashboard/streamlit/app.py
```

## Code Style

All Python code must pass:

```bash
make lint      # ruff check + ruff format --check
make typecheck # mypy src/faultscope
```

Key rules enforced:

| Rule | Value |
|---|---|
| Line length | 79 characters |
| Import sort | ruff (I) |
| Type annotations | Required everywhere (mypy strict) |
| Docstrings | Google style, required on public APIs |
| No `TODO` comments | Use GitHub issues instead |
| No hardcoded secrets | Always use env vars via pydantic-settings |

Fix auto-correctable issues with:

```bash
make lint-fix  # ruff check --fix + ruff format
```

## Testing

### Unit Tests

```bash
make test-unit
# or with coverage report
make coverage
```

Unit tests live in `tests/unit/`. They mock all I/O (Kafka, DB). Marked with `@pytest.mark.unit`.

### Integration Tests

```bash
make test-integration
```

Integration tests use `testcontainers` to spin up real Kafka and TimescaleDB instances. Marked with `@pytest.mark.integration`. They require Docker to be running.

### End-to-End Tests

```bash
make test-e2e
```

E2E tests run against a live Docker Compose stack. Start the full stack first with `make run-all`.

### Writing Tests

Follow existing patterns:

```python
# tests/unit/streaming/test_example.py
import pytest
from faultscope.streaming.features.temporal import TemporalFeatureExtractor


@pytest.mark.unit
class TestTemporalFeatureExtractor:
    def test_mean_feature(self) -> None:
        extractor = TemporalFeatureExtractor(window_sizes_s=[60])
        # ... arrange, act, assert
```

Use `pytest-asyncio` for async tests:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_method() -> None:
    result = await some_async_function()
    assert result is not None
```

Use `hypothesis` for property-based testing:

```python
from hypothesis import given, strategies as st

@pytest.mark.unit
@given(st.floats(min_value=0.0, max_value=1000.0))
def test_property(value: float) -> None:
    ...
```

### Coverage

The CI requires 80% line coverage. Run locally:

```bash
pytest tests/unit -m unit --cov=src/faultscope --cov-report=term-missing
```

## Adding a New Sensor

1. **Update sensor schema** in `src/faultscope/common/kafka/schemas.py`:
   ```python
   class SensorReading(BaseModel):
       # ... existing fields ...
       new_sensor: float | None = None
   ```

2. **Update machine profiles** in `src/faultscope/ingestion/simulator/engine.py`:
   ```python
   MachineProfile(
       name="turbofan",
       sensors=["sensor_1", ..., "new_sensor"],
       # ...
   )
   ```

3. **Update CMAPSS sensor map** in `src/faultscope/ingestion/cmapss/sensor_map.py` if applicable.

4. **Add feature definitions** in `src/faultscope/streaming/features/temporal.py` — the `TemporalFeatureExtractor` automatically extracts features for all sensors in the reading; no changes needed for standard temporal features.

5. **Update DB schema** — add a migration or update `infra/timescaledb/init.sql` if the sensor needs its own column (most sensors are stored as JSONB in `computed_features`).

6. **Retrain models** — new sensors change the feature space; existing models must be retrained:
   ```bash
   make train
   ```

## Adding a New Alert Rule

1. **Define the rule** in `src/faultscope/alerting/rules.py`:
   ```python
   DetectionRule(
       rule_id="custom-vibration-threshold",
       name="High Vibration",
       description="Vibration exceeds 85 g RMS",
       condition_type=ConditionType.ANOMALY_SCORE_ABOVE,
       threshold=0.85,
       severity=Severity.WARNING,
       cooldown_s=300,
       enabled=True,
   )
   ```

2. **Add to DEFAULT_RULES** list in the same file.

3. **Add a DB seed row** in `infra/timescaledb/init.sql` (for fresh installs).

4. **Write a unit test** in `tests/unit/alerting/test_rule_evaluator.py`.

For rules requiring custom logic not covered by existing `ConditionType` values:

1. Add the new type to `ConditionType(StrEnum)`.
2. Add an `elif` branch in `DetectionRule.evaluate()`.
3. Update `EvaluationContext` if new context fields are needed.

## Adding a New Notifier

1. Create `src/faultscope/alerting/notifiers/mynotifier.py`:
   ```python
   from faultscope.alerting.notifiers.base import BaseNotifier, NotificationPayload

   class MyNotifier(BaseNotifier):
       channel_name = "mynotifier"

       async def send(self, payload: NotificationPayload) -> None:
           # implement delivery logic
           ...
   ```

2. Register in `src/faultscope/alerting/coordinator.py` inside `_build_notifiers()`.

3. Add configuration fields to `AlertingConfig` in the config module.

## Dependency Management

Add a new dependency:

```bash
uv pip install some-package
# Then add to pyproject.toml [project.dependencies] manually
```

Add a new dev/test dependency:

```bash
# Add to [project.optional-dependencies] dev group in pyproject.toml
uv pip install ".[dev]"
```

## Project Layout

```
faultscope/
├── src/faultscope/
│   ├── common/           # shared: config, logging, Kafka client, DB engine
│   ├── ingestion/        # sensor simulator + CMAPSS loader + publisher
│   ├── streaming/        # feature extraction + quality checks + DB writer
│   ├── features/         # offline feature store: labeling, versioning
│   ├── training/         # LSTM + RandomForest training + MLflow tracking
│   ├── retraining/       # drift detection + auto-retraining pipeline
│   ├── inference/        # FastAPI prediction service + model hot-swap
│   ├── alerting/         # rule engine + incident management + notifications
│   └── dashboard/        # Streamlit app + Grafana dashboards
├── tests/
│   ├── unit/             # fast, isolated tests (no I/O)
│   ├── integration/      # testcontainers-based (real Kafka/DB)
│   └── e2e/              # full-stack tests against Docker Compose
├── docker/               # per-service Dockerfiles
├── infra/                # TimescaleDB init.sql, Kafka scripts, Prometheus config
├── scripts/              # seed_demo_data, train_models, health_check
├── docs/                 # this documentation
└── examples/             # runnable code examples
```

## Pre-commit Hooks

The following hooks run on every `git commit`:

| Hook | What it checks |
|---|---|
| `trailing-whitespace` | No trailing spaces |
| `end-of-file-fixer` | Files end with newline |
| `check-yaml` | Valid YAML syntax |
| `check-json` | Valid JSON syntax |
| `check-merge-conflict` | No merge conflict markers |
| `ruff` | Python linting |
| `ruff-format` | Python formatting |
| `mypy` | Type checking |
| `gitleaks` | No secrets in code |

Run all hooks manually:

```bash
pre-commit run --all-files
```

## Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add oscillating degradation pattern to simulator
fix: handle zero-variance sensors in cross-correlation
docs: document alert rule configuration
test: add property-based tests for spectral features
refactor: extract model loading logic into ModelVersionStore
chore: upgrade TensorFlow to 2.16
```

Breaking changes must include `BREAKING CHANGE:` in the footer.
