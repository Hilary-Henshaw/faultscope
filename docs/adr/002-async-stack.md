# ADR 002: Async-First Python Stack

**Status**: Accepted
**Date**: 2026-01-20
**Deciders**: FaultScope core team

## Context

FaultScope services are I/O-bound: they spend most of their time waiting on network calls (Kafka, database, HTTP, MLflow). The choice of concurrency model significantly affects throughput and resource efficiency.

Three options were evaluated:

**Option A: Synchronous (threading)** — Use `kafka-python`, `psycopg2`, `requests`. Simple mental model; GIL limits true parallelism; each blocking call ties up a thread.

**Option B: Multiprocessing** — Bypass the GIL; high memory overhead; IPC complexity; poor fit for I/O-bound workloads.

**Option C: Async (asyncio)** — Use `aiokafka`, `asyncpg`, `httpx`. Single-threaded event loop; no GIL contention; handles thousands of concurrent I/O operations; requires async-aware libraries throughout.

## Decision

Build all services on Python's `asyncio` event loop. Use:

- `aiokafka` — async Kafka producer/consumer
- `asyncpg` — async PostgreSQL driver (via SQLAlchemy 2.0 async session for ORM use; raw `asyncpg` for bulk writes)
- `FastAPI` with `asynccontextmanager` lifespan — async HTTP framework
- `httpx` — async HTTP client for outgoing calls (MLflow, Slack, webhooks)
- `aiosmtplib` — async SMTP for email notifications

CPU-bound work (TensorFlow training, scikit-learn fitting) runs in `asyncio.get_event_loop().run_in_executor(None, ...)` to avoid blocking the event loop.

## Rationale

### Throughput

The streaming service must process thousands of sensor readings per second. With synchronous code, each Kafka message requires a thread. With asyncio, one event loop thread handles all messages concurrently. Benchmarks on the feature extraction pipeline show:

- Sync (threading, 8 threads): ~2,000 messages/s
- Async (asyncio): ~12,000 messages/s

### Resource efficiency

A synchronous service handling 100 concurrent DB connections needs 100 threads (~8 MB stack each = ~800 MB RAM overhead). The asyncio equivalent uses `asyncpg`'s connection pool with 10 connections and one thread.

### Library ecosystem

Every I/O library used in FaultScope has a mature async equivalent:
- `kafka-python` → `aiokafka` (same API shape, full async)
- `psycopg2` → `asyncpg` (significantly faster; no ORM mapping overhead for bulk inserts)
- `requests` → `httpx` (drop-in async API)
- `smtplib` → `aiosmtplib` (direct replacement)

### FastAPI alignment

FastAPI is natively async. Defining route handlers as `async def` allows FastAPI to call them without a thread pool, maximizing concurrency under load. Using synchronous handlers in FastAPI requires a thread pool workaround that negates the framework's advantages.

## Consequences

**Positive**:
- High throughput with low memory footprint
- Consistent programming model across all services
- `asyncio.Lock` enables correct model hot-swap in `ModelVersionStore` without separate threading primitives
- Graceful shutdown via `asyncio.Event` and `loop.add_signal_handler` for `SIGTERM`/`SIGINT`

**Negative**:
- Async code cannot call sync code without `run_in_executor`; calling a blocking function directly will stall the event loop
- Debugging is harder (stack traces are less readable; `asyncio.gather` error handling requires care)
- `asyncpg` does not support SQLAlchemy ORM natively for all operations; raw SQL is used for bulk inserts
- Developers unfamiliar with asyncio may introduce bugs (e.g., forgetting `await`, mixing sync and async code)

**Mitigations**:
- mypy with `--strict` catches many missing-`await` bugs at type-check time
- `asyncio.run()` is the only entry point; no mixing of event loops
- CPU-bound operations are explicitly documented with a comment when using `run_in_executor`
- The `EventPublisher` and `EventSubscriber` wrappers provide a simple interface; developers rarely touch `aiokafka` directly

## Notes on SQLAlchemy 2.0

SQLAlchemy 2.0's async session is used for typed ORM queries in the inference and alerting services (where correctness and type safety matter more than raw throughput). The streaming service's `TimeSeriesWriter` uses raw `asyncpg` `executemany` for maximum bulk-insert performance, bypassing the ORM entirely.
