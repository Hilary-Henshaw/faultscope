"""NASA C-MAPSS turbofan dataset loader.

Loads one or more of the four FD00x sub-datasets from space-separated
text files, normalises sensor values using per-dataset min-max scaling,
and yields :class:`~faultscope.common.kafka.schemas.SensorReading`
objects in engine / cycle order.

If the raw files are not present locally, the loader raises a clear
:exc:`FileNotFoundError` rather than attempting a network download so
that credentials or network policies never need to be embedded here.
The caller (or a CI step) is responsible for obtaining the data and
placing it under ``data_path``.

Dataset format
--------------
Each ``.txt`` file has 26 space-separated columns:

    engine_id  cycle  op1  op2  op3  s1 … s21

with no header row.  Column semantics are documented in
:mod:`faultscope.ingestion.cmapss.sensor_map`.

References
----------
A. Saxena, K. Goebel, D. Simon, and N. Eklund,
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure
Simulation", *PHM '08*, 2008.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import structlog

from faultscope.common.kafka.schemas import SensorReading
from faultscope.ingestion.cmapss.sensor_map import (
    CMAPSS_SENSOR_COLUMNS,
    OPERATIONAL_COLUMNS,
    SENSOR_COLUMNS,
)

log: structlog.BoundLogger = structlog.get_logger(__name__)

#: Supported sub-dataset identifiers.
_VALID_DATASET_IDS: frozenset[str] = frozenset(
    {"FD001", "FD002", "FD003", "FD004"}
)

#: Wall-clock start epoch used when constructing ``recorded_at``
#: timestamps from cycle numbers (avoids ``datetime.now()`` non-
#: determinism; one cycle = one second).
_EPOCH: datetime = datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)


class CmapssLoader:
    """Loads and streams the NASA C-MAPSS turbofan degradation dataset.

    The loader expects the raw space-separated text files to be present
    at ``data_path``.  It does **not** download data automatically.

    Parameters
    ----------
    data_path:
        Directory that contains files named ``train_FD001.txt`` …
        ``train_FD004.txt``.
    dataset_ids:
        Which sub-datasets to use.  Defaults to all four
        (``["FD001", "FD002", "FD003", "FD004"]``).

    Raises
    ------
    ValueError
        If any entry in ``dataset_ids`` is not one of the four
        recognised identifiers.
    FileNotFoundError
        If ``data_path`` does not exist.
    """

    def __init__(
        self,
        data_path: str,
        dataset_ids: list[str] | None = None,
    ) -> None:
        resolved = Path(data_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"C-MAPSS data directory not found: {resolved}"
            )
        self._data_path = resolved

        ids: list[str] = dataset_ids or list(_VALID_DATASET_IDS)
        invalid = set(ids) - _VALID_DATASET_IDS
        if invalid:
            raise ValueError(
                f"Unknown dataset IDs: {sorted(invalid)}. "
                f"Valid IDs: {sorted(_VALID_DATASET_IDS)}"
            )
        self._dataset_ids: list[str] = sorted(ids)

        log.info(
            "cmapss_loader.init",
            data_path=str(resolved),
            dataset_ids=self._dataset_ids,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a single C-MAPSS sub-dataset and normalise sensor values.

        Reads the training split (``train_FDxxx.txt``), assigns human-
        readable column names, and applies min-max normalisation
        independently to each sensor column so that values lie in
        ``[0.0, 1.0]``.  Operational settings are **not** normalised
        because their interpretation varies by operating regime.

        Parameters
        ----------
        dataset_id:
            One of ``"FD001"``, ``"FD002"``, ``"FD003"``, ``"FD004"``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns matching
            :data:`~faultscope.ingestion.cmapss.sensor_map.CMAPSS_SENSOR_COLUMNS`
            plus an additional ``"rul"`` column containing the remaining
            useful life at each row (cycles until the engine's last
            observed cycle).

        Raises
        ------
        ValueError
            If ``dataset_id`` is not recognised.
        FileNotFoundError
            If the expected file is absent from ``data_path``.
        RuntimeError
            If the file is malformed (wrong number of columns).
        """
        if dataset_id not in _VALID_DATASET_IDS:
            raise ValueError(
                f"Unknown dataset_id {dataset_id!r}. "
                f"Expected one of {sorted(_VALID_DATASET_IDS)}"
            )

        file_path = self._data_path / f"train_{dataset_id}.txt"
        if not file_path.exists():
            raise FileNotFoundError(f"C-MAPSS file not found: {file_path}")

        log.info(
            "cmapss_loader.loading",
            dataset_id=dataset_id,
            file=str(file_path),
        )

        try:
            df = pd.read_csv(
                file_path,
                sep=r"\s+",
                header=None,
                names=CMAPSS_SENSOR_COLUMNS,
                engine="python",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse C-MAPSS file {file_path}: {exc}"
            ) from exc

        if df.shape[1] != len(CMAPSS_SENSOR_COLUMNS):
            raise RuntimeError(
                f"Expected {len(CMAPSS_SENSOR_COLUMNS)} columns in "
                f"{file_path}, got {df.shape[1]}"
            )

        # Compute RUL: for each engine, RUL = max_cycle - current_cycle.
        max_cycles: pd.Series = df.groupby("engine_id")["cycle"].transform(
            "max"
        )
        df["rul"] = (max_cycles - df["cycle"]).astype(int)

        # Min-max normalise sensor columns only.
        for col in SENSOR_COLUMNS:
            col_min = df[col].min()
            col_max = df[col].max()
            span = col_max - col_min
            if span > 0:
                df[col] = (df[col] - col_min) / span
            else:
                # Constant sensor — leave as-is (typically 0.0).
                df[col] = 0.0

        log.info(
            "cmapss_loader.loaded",
            dataset_id=dataset_id,
            rows=len(df),
            engines=int(df["engine_id"].nunique()),
        )
        return df

    def iter_readings(
        self,
        dataset_id: str,
    ) -> Iterator[SensorReading]:
        """Yield one :class:`~faultscope.common.kafka.schemas.SensorReading`
        per row, in ascending engine-id / cycle order.

        Each engine is labelled ``"CMAPSS-{dataset_id}-{engine_id:04d}"``
        so that machine IDs are globally unique when multiple sub-datasets
        are used simultaneously.

        The ``recorded_at`` timestamp is synthesised from a fixed epoch
        plus ``cycle`` seconds so that consumers see strictly monotonic
        timestamps per engine.

        Parameters
        ----------
        dataset_id:
            Which sub-dataset to stream.

        Yields
        ------
        SensorReading
            One reading per data row.

        Raises
        ------
        ValueError
            If ``dataset_id`` is not recognised.
        FileNotFoundError
            If the underlying file is missing.
        RuntimeError
            If the file cannot be parsed.
        """
        df = self.load_dataset(dataset_id)
        df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

        for row in df.itertuples(index=False):
            engine_id: int = int(row.engine_id)
            cycle: int = int(row.cycle)

            machine_id = f"CMAPSS-{dataset_id}-{engine_id:04d}"
            recorded_at = _EPOCH + timedelta(seconds=cycle)

            readings: dict[str, float] = {
                col: round(float(getattr(row, col)), 6)
                for col in SENSOR_COLUMNS
            }
            operational: dict[str, float] = {
                col: round(float(getattr(row, col)), 6)
                for col in OPERATIONAL_COLUMNS
            }

            yield SensorReading(
                machine_id=machine_id,
                recorded_at=recorded_at,
                cycle=cycle,
                readings=readings,
                operational=operational,
            )
