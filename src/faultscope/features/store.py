"""Versioned feature snapshot store backed by TimescaleDB.

``VersionedFeatureStore`` saves labelled feature DataFrames to the
``feature_snapshots`` hypertable and reloads them by version + split,
providing a reproducible offline dataset for model training.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from faultscope.common.exceptions import DatabaseError, ValidationError

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_VALID_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})

# Columns that are stored as dedicated table columns, not inside the
# feature_vector JSONB blob.
_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {
        "machine_id",
        "computed_at",
        "window_s",
        "feature_version",
        "rul_cycles",
        "health_label",
        "split",
        "dataset_version",
        "snapshot_at",
        "cycle",
    }
)


class VersionedFeatureStore:
    """Saves and loads versioned training dataset snapshots.

    Snapshots are persisted in the ``feature_snapshots`` TimescaleDB
    hypertable.  Each row stores the full feature vector as a JSONB
    blob alongside ``rul_cycles``, ``health_label``, ``split``, and
    ``dataset_version`` metadata.

    Parameters
    ----------
    db_url:
        asyncpg-compatible SQLAlchemy URL.
    dataset_version:
        Default dataset version tag used when none is supplied to
        ``save_snapshot`` / ``load_snapshot``.
    """

    def __init__(
        self,
        db_url: str,
        dataset_version: str,
    ) -> None:
        self._engine: AsyncEngine = create_async_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self._dataset_version = dataset_version

    async def save_snapshot(
        self,
        df: pd.DataFrame,
        split: str,
    ) -> None:
        """Insert all rows of *df* into ``feature_snapshots``.

        The ``split`` value determines the dataset partition
        (``train``, ``validation``, or ``test``).  The feature
        columns are serialised as a JSONB blob in
        ``feature_vector``.

        Parameters
        ----------
        df:
            Labelled feature DataFrame.  Must contain at minimum
            ``machine_id``, ``rul_cycles``, and ``health_label``.
        split:
            Dataset partition label: one of ``train``,
            ``validation``, ``test``.

        Raises
        ------
        ValidationError
            If *split* is not a recognised partition name, or if
            required columns are absent.
        DatabaseError
            If the INSERT statement fails.
        """
        if split not in _VALID_SPLITS:
            raise ValidationError(
                f"Invalid split '{split}'. "
                f"Must be one of {sorted(_VALID_SPLITS)}",
                context={"split": split},
            )
        missing = {"machine_id", "rul_cycles", "health_label"} - set(
            df.columns
        )
        if missing:
            raise ValidationError(
                "DataFrame missing required label columns",
                context={"missing_columns": sorted(missing)},
            )

        feature_cols = self.get_feature_columns(df)
        snapshot_at = datetime.now(tz=UTC)

        insert_stmt = text(
            """
            INSERT INTO feature_snapshots
                (snapshot_at, machine_id, feature_vector,
                 rul_cycles, health_label, dataset_version, split)
            VALUES
                (:snapshot_at, :machine_id, :feature_vector,
                 :rul_cycles, :health_label, :dataset_version, :split)
            """
        )

        records: list[dict[str, object]] = []
        for _, row in df.iterrows():
            fv: dict[str, object] = {
                col: (
                    row[col].item()  # type: ignore[union-attr]
                    if hasattr(row[col], "item")
                    else row[col]
                )
                for col in feature_cols
            }
            records.append(
                {
                    "snapshot_at": snapshot_at,
                    "machine_id": str(row["machine_id"]),
                    "feature_vector": json.dumps(fv),
                    "rul_cycles": int(row["rul_cycles"]),
                    "health_label": str(row["health_label"]),
                    "dataset_version": self._dataset_version,
                    "split": split,
                }
            )

        try:
            async with self._engine.begin() as conn:
                await conn.execute(insert_stmt, records)
        except Exception as exc:
            raise DatabaseError(
                "Failed to insert feature snapshots",
                context={
                    "split": split,
                    "n_rows": len(df),
                    "dataset_version": self._dataset_version,
                    "error": str(exc),
                },
            ) from exc

        log.info(
            "snapshot_saved",
            split=split,
            n_rows=len(df),
            dataset_version=self._dataset_version,
            snapshot_at=snapshot_at.isoformat(),
        )

    async def load_snapshot(
        self,
        split: str,
        dataset_version: str | None = None,
    ) -> pd.DataFrame:
        """Load a saved snapshot from the database.

        Returns a wide DataFrame with one column per feature, plus
        ``machine_id``, ``snapshot_at``, ``rul_cycles``,
        ``health_label``, ``split``, and ``dataset_version``.

        Parameters
        ----------
        split:
            Dataset partition to load (``train``, ``validation``,
            ``test``).
        dataset_version:
            Version tag to load.  Defaults to the version the store
            was initialised with.

        Returns
        -------
        pd.DataFrame
            Wide feature DataFrame.

        Raises
        ------
        ValidationError
            If *split* is not a recognised partition name.
        DatabaseError
            If the query fails or returns no rows.
        """
        if split not in _VALID_SPLITS:
            raise ValidationError(
                f"Invalid split '{split}'. "
                f"Must be one of {sorted(_VALID_SPLITS)}",
                context={"split": split},
            )
        version = dataset_version or self._dataset_version

        query = text(
            """
            SELECT
                snapshot_at,
                machine_id,
                feature_vector,
                rul_cycles,
                health_label,
                dataset_version,
                split
            FROM feature_snapshots
            WHERE split = :split
              AND dataset_version = :version
            ORDER BY machine_id, snapshot_at
            """
        )

        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(
                    query, {"split": split, "version": version}
                )
                rows = result.fetchall()
        except Exception as exc:
            raise DatabaseError(
                "Failed to load feature snapshots",
                context={
                    "split": split,
                    "dataset_version": version,
                    "error": str(exc),
                },
            ) from exc

        if not rows:
            raise DatabaseError(
                "No feature snapshots found for the requested "
                "split/version combination",
                context={"split": split, "dataset_version": version},
            )

        records: list[dict[str, object]] = []
        for row in rows:
            fv: dict[str, object] = (
                row.feature_vector
                if isinstance(row.feature_vector, dict)
                else json.loads(row.feature_vector)
            )
            record: dict[str, object] = {
                "snapshot_at": row.snapshot_at,
                "machine_id": row.machine_id,
                "rul_cycles": row.rul_cycles,
                "health_label": row.health_label,
                "dataset_version": row.dataset_version,
                "split": row.split,
                **fv,
            }
            records.append(record)

        df = pd.DataFrame(records)
        df["snapshot_at"] = pd.to_datetime(df["snapshot_at"], utc=True)

        log.info(
            "snapshot_loaded",
            split=split,
            dataset_version=version,
            n_rows=len(df),
            n_features=len(self.get_feature_columns(df)),
        )
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return column names that are model features.

        Excludes all metadata, label, and administrative columns
        defined in ``_NON_FEATURE_COLS``.

        Parameters
        ----------
        df:
            Any feature DataFrame produced by this pipeline.

        Returns
        -------
        list[str]
            Sorted list of feature column names.
        """
        return sorted(
            col for col in df.columns if col not in _NON_FEATURE_COLS
        )

    async def close(self) -> None:
        """Dispose the async engine."""
        await self._engine.dispose()
