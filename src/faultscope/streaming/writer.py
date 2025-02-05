"""Async TimescaleDB writer with in-memory batching.

Records are accumulated in memory and flushed to the database either
when the buffer reaches ``batch_size`` or when ``flush_interval_s``
seconds have elapsed since the last flush, whichever occurs first.

The writer uses a raw ``asyncpg`` connection pool and ``executemany``
for high-throughput batch inserts into ``sensor_readings`` and
``computed_features``.

Usage::

    writer = TimeSeriesWriter(db_url, batch_size=100)
    await writer.start()
    await writer.buffer_reading(reading, ["NULL_FRACTION_LOW"])
    await writer.buffer_features(features)
    await writer.stop()  # flushes remainder and closes pool
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import asyncpg

from faultscope.common.exceptions import DatabaseError
from faultscope.common.logging import get_logger
from faultscope.streaming.models import ComputedFeatures, SensorReading

log = get_logger(__name__)

# SQL for sensor_readings batch insert.
_INSERT_SENSOR_READINGS = """
    INSERT INTO sensor_readings
        (recorded_at, machine_id, cycle, readings,
         operational, quality_flags, ingested_at)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT DO NOTHING
"""

# SQL for computed_features batch insert.
_INSERT_COMPUTED_FEATURES = """
    INSERT INTO computed_features
        (computed_at, machine_id, window_s,
         temporal, spectral, correlation, feature_version)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT DO NOTHING
"""

# Type aliases for the raw row tuples sent to asyncpg.
_ReadingRow = tuple[
    datetime,  # recorded_at
    str,  # machine_id
    int | None,  # cycle
    str,  # readings (JSON)
    str,  # operational (JSON)
    list[str],  # quality_flags
    datetime,  # ingested_at
]

_FeatureRow = tuple[
    datetime,  # computed_at
    str,  # machine_id
    int,  # window_s
    str,  # temporal (JSON)
    str,  # spectral (JSON)
    str,  # correlation (JSON)
    str,  # feature_version
]


class TimeSeriesWriter:
    """Batched async writer for sensor readings and computed features.

    Parameters
    ----------
    db_url:
        Raw asyncpg DSN, e.g.
        ``postgresql://user:pass@host:5432/faultscope``.
    batch_size:
        Number of records that trigger an immediate flush.
    flush_interval_s:
        Maximum seconds between flushes.
    pool_size:
        Number of connections in the asyncpg pool.
    """

    def __init__(
        self,
        db_url: str,
        batch_size: int = 100,
        flush_interval_s: float = 5.0,
        pool_size: int = 10,
    ) -> None:
        self._db_url = db_url
        self._batch_size = batch_size
        self._flush_interval = flush_interval_s
        self._pool_size = pool_size

        self._pool: asyncpg.Pool[Any] | None = None
        self._reading_buf: list[_ReadingRow] = []
        self._feature_buf: list[_FeatureRow] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Create the asyncpg connection pool and start the flush timer.

        Must be called before any ``buffer_*`` methods.
        """
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._db_url,
                min_size=2,
                max_size=self._pool_size,
                command_timeout=30,
            )
        except (
            asyncpg.PostgresConnectionFailedError,
            OSError,
        ) as exc:
            raise DatabaseError(
                "Failed to create asyncpg connection pool",
                context={"db_url": self._db_url, "error": str(exc)},
            ) from exc

        self._flush_task = asyncio.create_task(
            self._periodic_flush(), name="writer_flush_timer"
        )
        log.info(
            "writer.started",
            pool_size=self._pool_size,
            flush_interval_s=self._flush_interval,
        )

    async def stop(self) -> None:
        """Flush buffered records and close the connection pool gracefully."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self.flush()

        if self._pool is not None:
            await self._pool.close()
            self._pool = None

        log.info("writer.stopped")

    # ── Buffer API ────────────────────────────────────────────────────

    async def buffer_reading(
        self,
        reading: SensorReading,
        quality_flags: list[str],
    ) -> None:
        """Append a sensor reading to the write buffer.

        Triggers an immediate flush when the buffer reaches
        ``batch_size``.

        Parameters
        ----------
        reading:
            Validated and cleaned sensor reading.
        quality_flags:
            List of flag name strings, e.g. ``["NULL_FRACTION_LOW"]``.
        """
        recorded = reading.recorded_at
        if recorded.tzinfo is None:
            recorded = recorded.replace(tzinfo=UTC)

        row: _ReadingRow = (
            recorded,
            reading.machine_id,
            reading.cycle,
            json.dumps(reading.readings),
            json.dumps(reading.operational),
            quality_flags,
            datetime.now(tz=UTC),
        )

        async with self._lock:
            self._reading_buf.append(row)
            should_flush = (
                len(self._reading_buf) + len(self._feature_buf)
                >= self._batch_size
            )

        if should_flush:
            await self.flush()

    async def buffer_features(
        self,
        features: ComputedFeatures,
    ) -> None:
        """Append computed features to the write buffer.

        Triggers an immediate flush when the combined buffer reaches
        ``batch_size``.

        Parameters
        ----------
        features:
            Feature vector to persist.
        """
        computed = features.computed_at
        if computed.tzinfo is None:
            computed = computed.replace(tzinfo=UTC)

        row: _FeatureRow = (
            computed,
            features.machine_id,
            features.window_s,
            json.dumps(features.temporal),
            json.dumps(features.spectral),
            json.dumps(features.correlation),
            features.feature_version,
        )

        async with self._lock:
            self._feature_buf.append(row)
            should_flush = (
                len(self._reading_buf) + len(self._feature_buf)
                >= self._batch_size
            )

        if should_flush:
            await self.flush()

    # ── Flush ─────────────────────────────────────────────────────────

    async def flush(self) -> None:
        """Write all buffered records to TimescaleDB.

        Uses ``executemany`` on the asyncpg connection for efficient
        bulk inserts.  Buffers are swapped under lock to minimise
        contention with concurrent ``buffer_*`` calls.

        Raises
        ------
        DatabaseError
            When the pool is unavailable or the INSERT fails.
        """
        async with self._lock:
            reading_rows = self._reading_buf[:]
            feature_rows = self._feature_buf[:]
            self._reading_buf.clear()
            self._feature_buf.clear()

        if not reading_rows and not feature_rows:
            return

        if self._pool is None:
            raise DatabaseError(
                "Writer pool is not initialised; call start() first",
                context={},
            )

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    if reading_rows:
                        await conn.executemany(
                            _INSERT_SENSOR_READINGS,
                            reading_rows,
                        )
                    if feature_rows:
                        await conn.executemany(
                            _INSERT_COMPUTED_FEATURES,
                            feature_rows,
                        )
        except asyncpg.PostgresError as exc:
            raise DatabaseError(
                "Batch insert failed",
                context={
                    "readings": len(reading_rows),
                    "features": len(feature_rows),
                    "error": str(exc),
                },
            ) from exc

        log.debug(
            "writer.flushed",
            readings=len(reading_rows),
            features=len(feature_rows),
        )

    # ── Private helpers ───────────────────────────────────────────────

    async def _periodic_flush(self) -> None:
        """Background task that flushes the buffer on a fixed interval."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except DatabaseError as exc:
                log.error(
                    "writer.periodic_flush_error",
                    error=str(exc),
                    context=exc.context,
                )
