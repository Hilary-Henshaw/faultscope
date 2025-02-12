"""Feature extractor: reads TimescaleDB and assembles training datasets.

The ``FeatureExtractor`` queries the ``computed_features`` hypertable,
flattens JSONB feature columns into a wide DataFrame, and provides
cycle sequences needed for RUL label assignment.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from faultscope.common.exceptions import DatabaseError

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_METADATA_COLUMNS: frozenset[str] = frozenset(
    {"machine_id", "computed_at", "window_s", "feature_version"}
)


class FeatureExtractor:
    """Reads ``computed_features`` table and assembles training dataset.

    Pulls features for all machines within a date range, joining
    metadata.  JSONB feature columns (``temporal``, ``spectral``,
    ``correlation``) are flattened into top-level DataFrame columns
    using the pattern ``<group>__<name>`` (e.g.
    ``temporal__vibration_mean``).

    Parameters
    ----------
    db_url:
        asyncpg-compatible SQLAlchemy URL, e.g.
        ``postgresql+asyncpg://user:pass@host:5432/faultscope``.
    """

    def __init__(self, db_url: str) -> None:
        self._engine: AsyncEngine = create_async_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )

    async def extract_window(
        self,
        start: datetime,
        end: datetime,
        machine_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a wide DataFrame of computed features.

        One row per ``(machine_id, computed_at)`` pair, columns equal
        feature names plus ``machine_id``, ``computed_at``,
        ``window_s``, and ``feature_version``.

        Parameters
        ----------
        start:
            Inclusive start of the time window (UTC).
        end:
            Inclusive end of the time window (UTC).
        machine_ids:
            Optional allow-list of machine identifiers.  When ``None``
            all machines in the window are returned.

        Returns
        -------
        pd.DataFrame
            Wide feature DataFrame sorted by ``machine_id``,
            ``computed_at``.

        Raises
        ------
        DatabaseError
            If the query fails or returns an empty result set.
        """
        log.info(
            "extracting_features",
            start=start.isoformat(),
            end=end.isoformat(),
            machine_filter=machine_ids,
        )

        params: dict[str, object] = {
            "start": start,
            "end": end,
        }
        base_sql = (
            "SELECT machine_id, computed_at, window_s, feature_version,"
            " temporal, spectral, correlation"
            " FROM computed_features"
            " WHERE computed_at >= :start"
            " AND computed_at <= :end"
        )
        if machine_ids:
            base_sql = base_sql + " AND machine_id = ANY(:machine_ids)"
            params["machine_ids"] = machine_ids
        base_sql = base_sql + " ORDER BY machine_id, computed_at"
        query = text(base_sql)

        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(query, params)
                rows = result.fetchall()
        except Exception as exc:
            raise DatabaseError(
                "Failed to query computed_features",
                context={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "error": str(exc),
                },
            ) from exc

        if not rows:
            raise DatabaseError(
                "No computed features found for the requested window",
                context={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "machine_ids": machine_ids,
                },
            )

        records: list[dict[str, object]] = []
        for row in rows:
            record: dict[str, object] = {
                "machine_id": row.machine_id,
                "computed_at": row.computed_at,
                "window_s": row.window_s,
                "feature_version": row.feature_version,
            }
            for group_name in ("temporal", "spectral", "correlation"):
                raw = getattr(row, group_name)
                group: dict[str, float] = (
                    raw if isinstance(raw, dict) else json.loads(raw or "{}")
                )
                for feat_name, feat_val in group.items():
                    record[f"{group_name}__{feat_name}"] = feat_val
            records.append(record)

        df = pd.DataFrame(records)
        df["computed_at"] = pd.to_datetime(df["computed_at"], utc=True)

        log.info(
            "features_extracted",
            n_rows=len(df),
            n_machines=df["machine_id"].nunique(),
            n_features=len(df.columns) - len(_METADATA_COLUMNS),
        )
        return df

    async def get_machine_cycles(
        self,
        machine_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return the cycle sequence for a single machine.

        Used by ``RulLabeler`` to determine each machine's total
        observed cycle count within the extraction window.

        Parameters
        ----------
        machine_id:
            The machine whose readings are queried.
        start:
            Inclusive start of the window (UTC).
        end:
            Inclusive end of the window (UTC).

        Returns
        -------
        pd.DataFrame
            Columns: ``machine_id``, ``recorded_at``, ``cycle``.
            Rows sorted by ``recorded_at`` ascending.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        log.debug(
            "fetching_machine_cycles",
            machine_id=machine_id,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        query = text(
            """
            SELECT
                machine_id,
                recorded_at,
                cycle
            FROM sensor_readings
            WHERE machine_id = :machine_id
              AND recorded_at >= :start
              AND recorded_at <= :end
              AND cycle IS NOT NULL
            ORDER BY recorded_at
            """
        )

        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(
                    query,
                    {
                        "machine_id": machine_id,
                        "start": start,
                        "end": end,
                    },
                )
                rows = result.fetchall()
        except Exception as exc:
            raise DatabaseError(
                "Failed to query sensor_readings cycles",
                context={
                    "machine_id": machine_id,
                    "error": str(exc),
                },
            ) from exc

        if not rows:
            return pd.DataFrame(columns=["machine_id", "recorded_at", "cycle"])

        df = pd.DataFrame(rows, columns=["machine_id", "recorded_at", "cycle"])
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
        df["cycle"] = df["cycle"].astype(int)
        return df

    async def close(self) -> None:
        """Dispose the async engine and release all connections."""
        await self._engine.dispose()
