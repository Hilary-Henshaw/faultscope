# Basic Simulation Example

This example shows how to use the `MachineSimulator` directly to generate synthetic
sensor readings and publish them to Kafka.

## Prerequisites

- FaultScope installed: `pip install -e ".[dev]"` from the repo root
- Kafka running: `make run-infra` (or provide your own broker)

## Environment

```bash
export FAULTSCOPE_KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export FAULTSCOPE_DB_PASSWORD=changeme
```

## Run

```bash
python examples/basic-simulation/run_simulation.py
```

Or with custom options:

```bash
python examples/basic-simulation/run_simulation.py \
  --machine-type turbofan \
  --count 5 \
  --interval-ms 200 \
  --duration-s 60
```

## What it does

1. Creates `--count` machines using the specified profile (turbofan/pump/compressor)
2. Advances each machine's degradation state by one cycle
3. Publishes a `SensorReading` to `faultscope.sensors.readings`
4. Sleeps `--interval-ms` milliseconds
5. Repeats for `--duration-s` seconds (or indefinitely if 0)

## Machine profiles

| Profile | Sensors | Failure modes |
|---|---|---|
| `turbofan` | 21 sensors | LINEAR, EXPONENTIAL, STEP, OSCILLATING |
| `pump` | 9 sensors | LINEAR, EXPONENTIAL |
| `compressor` | 9 sensors | LINEAR, STEP |

Degradation patterns are stochastic — each machine gets random noise amplitude,
drift rate, and failure threshold on construction.

## Expected output

```
2026-03-31T10:00:00Z [info] simulator_started machines=5 type=turbofan
2026-03-31T10:00:00Z [info] reading_published machine_id=M-001 rul=124 sensors=21
2026-03-31T10:00:00Z [info] reading_published machine_id=M-002 rul=87 sensors=21
...
2026-03-31T10:01:00Z [info] machine_failed machine_id=M-003 replacing=True
2026-03-31T10:01:00Z [info] reading_published machine_id=M-003-v2 rul=125 sensors=21
```

## Next steps

- Connect the **streaming service** to consume these readings and extract features
- Run `make run-all` for the full pipeline
