# ADR 001: Event-Driven Architecture with Apache Kafka

**Status**: Accepted
**Date**: 2026-01-15
**Deciders**: FaultScope core team

## Context

FaultScope must ingest high-frequency sensor telemetry from potentially hundreds of machines simultaneously, route that data through a processing pipeline (quality checks, feature extraction), and make the results available to multiple downstream consumers (inference service, feature store, dashboard). The design must handle:

- Burst traffic from synchronized maintenance cycles causing simultaneous sensor reconnects
- Independent scaling of each processing stage
- Replay of historical sensor data for model retraining
- Resilience to consumer downtime (data must not be lost if the inference service restarts)
- Zero coupling between producers and consumers

Two approaches were evaluated:

**Option A: REST polling** — Each downstream service polls an HTTP endpoint on a schedule. Simple to implement.

**Option B: Event streaming with Kafka** — Producers publish messages to topics; consumers subscribe independently.

## Decision

Use Apache Kafka as the event backbone for all inter-service communication.

## Rationale

| Criterion | REST polling | Kafka |
|---|---|---|
| Decoupling | Low (consumer must know producer URL) | High (consumers subscribe to topics) |
| Burst handling | Poor (consumer must poll frequently or miss data) | Good (Kafka buffers messages in topic) |
| Replay | Not possible | Built-in (seek to any offset) |
| Consumer independence | Low (producer is unavailable → data lost) | High (consumer lag is recoverable) |
| Fan-out | Requires multiple HTTP calls | Natural (multiple consumer groups) |
| Throughput | Limited by HTTP overhead | Millions of messages/second |
| Operational complexity | Low | Medium (broker management) |

Kafka wins on every criterion that matters for a production industrial IoT system. The operational overhead is acceptable given that a single-broker KRaft-mode Kafka (no ZooKeeper) is straightforward to run in Docker Compose.

Additional design choices that flow from this decision:

- **At-least-once delivery**: Consumers use manual offset commit after successful DB write. Duplicate messages are idempotent for time-series inserts (upsert by `(machine_id, timestamp)`).
- **Partition key = `machine_id`**: Guarantees ordering per machine across all topics.
- **Dead-letter queue (`faultscope.dlq`)**: Messages that cannot be processed (schema validation failure, repeated parse errors) are forwarded with a JSON envelope containing the error reason. This prevents one bad message from blocking an entire partition.
- **Topic retention**: Raw sensor data retained 7 days (allows replay of up to one week). Predictions and incidents retained 14–30 days.

## Consequences

**Positive**:
- Each service can be restarted, scaled, or deployed independently without affecting others
- Message replay enables retraining on historical data without re-running ingestion
- Grafana Kafka lag panel provides a clear SLI for pipeline health
- The DLQ gives operators visibility into data quality issues

**Negative**:
- Adds Kafka as a required infrastructure dependency
- Developers must understand consumer group semantics, offset management, and partition assignment
- Local development requires Docker (or a local Kafka installation)
- End-to-end latency increases by approximately 10–50 ms compared to direct HTTP calls (acceptable for maintenance prediction, where decisions are made over minutes to hours)

**Mitigations**:
- The `Makefile` provides `make run-infra` to start all infrastructure with one command
- Integration tests use `testcontainers` to spin up a real Kafka broker automatically, no manual setup needed
- The `EventPublisher` and `EventSubscriber` wrappers hide `aiokafka` complexity from service code
