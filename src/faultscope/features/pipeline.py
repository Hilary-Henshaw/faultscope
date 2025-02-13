"""Offline feature generation pipeline orchestrator.

``FeaturePipelineRunner`` chains together extraction, labelling,
train/val/test splitting, and persistence in a single ``run`` call.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd
import structlog

from faultscope.common.exceptions import ValidationError
from faultscope.features.config import FeaturesConfig
from faultscope.features.extractor import FeatureExtractor
from faultscope.features.labeler import HealthLabeler, RulLabeler
from faultscope.features.store import VersionedFeatureStore

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def _stratified_machine_split(
    machine_ids: list[str],
    train_frac: float,
    val_frac: float,
) -> tuple[list[str], list[str], list[str]]:
    """Split machine identifiers into train / validation / test sets.

    Splitting is performed at the machine level so that no machine
    appears in more than one split (prevents temporal data leakage
    between the partitions).

    Parameters
    ----------
    machine_ids:
        Complete list of unique machine identifiers.
    train_frac:
        Fraction of machines assigned to training.
    val_frac:
        Fraction of machines assigned to validation.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(train_ids, val_ids, test_ids)`` lists.

    Raises
    ------
    ValidationError
        If fractions are invalid or no machines remain for any split.
    """
    if train_frac + val_frac >= 1.0:
        raise ValidationError(
            "train_fraction + val_fraction must be less than 1.0",
            context={
                "train_fraction": train_frac,
                "val_fraction": val_frac,
            },
        )
    if not machine_ids:
        raise ValidationError(
            "Cannot split an empty machine list",
            context={},
        )

    # Sort for determinism; do not shuffle so results are reproducible
    # without a random seed.
    sorted_ids = sorted(machine_ids)
    n = len(sorted_ids)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = n - n_train - n_val

    if n_test < 1:
        raise ValidationError(
            "Not enough machines to populate all three splits",
            context={
                "n_machines": n,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
            },
        )

    train_ids = sorted_ids[:n_train]
    val_ids = sorted_ids[n_train : n_train + n_val]
    test_ids = sorted_ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


class FeaturePipelineRunner:
    """Orchestrates end-to-end offline feature generation.

    Steps executed by ``run``:

    1. Extract feature windows from TimescaleDB.
    2. Fetch per-machine cycle sequences and merge into the feature
       DataFrame.
    3. Assign RUL labels via ``RulLabeler``.
    4. Assign health status labels via ``HealthLabeler``.
    5. Split at the machine level into train / validation / test.
    6. Persist each split to ``feature_snapshots`` via
       ``VersionedFeatureStore``.

    Parameters
    ----------
    config:
        Fully populated ``FeaturesConfig`` instance.
    """

    def __init__(self, config: FeaturesConfig) -> None:
        self._cfg = config
        self._extractor = FeatureExtractor(config.db_url)
        self._rul_labeler = RulLabeler(config.max_rul_cycles)
        self._health_labeler = HealthLabeler(config.health_label_thresholds)
        self._store = VersionedFeatureStore(
            config.db_url, config.dataset_version
        )

    async def run(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        """Run the full feature pipeline.

        Parameters
        ----------
        start:
            Inclusive start of the extraction window (UTC).
        end:
            Inclusive end of the extraction window (UTC).

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping with keys ``"train"``, ``"validation"``,
            ``"test"``, each holding the labelled feature DataFrame
            for that split.

        Raises
        ------
        ValidationError
            If the extracted dataset is too small to split.
        DatabaseError
            If any DB operation fails.
        """
        log.info(
            "feature_pipeline_started",
            start=start.isoformat(),
            end=end.isoformat(),
            dataset_version=self._cfg.dataset_version,
        )

        # Step 1: extract features.
        feature_df = await self._extractor.extract_window(start, end)

        machine_ids: list[str] = feature_df["machine_id"].unique().tolist()

        # Step 2: fetch cycle sequences for each machine concurrently
        # and merge into the feature DataFrame.
        cycle_tasks = [
            self._extractor.get_machine_cycles(mid, start, end)
            for mid in machine_ids
        ]
        cycle_dfs: list[pd.DataFrame] = await asyncio.gather(*cycle_tasks)
        cycles_combined = pd.concat(
            [c for c in cycle_dfs if not c.empty], ignore_index=True
        )

        if cycles_combined.empty:
            raise ValidationError(
                "No cycle data found for any machine in the extraction "
                "window; cannot compute RUL labels",
                context={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )

        # Map computed_at → nearest cycle by merging on machine_id and
        # choosing the closest recorded_at.  We use an asof merge
        # (time-ordered) so each feature row gets the cycle count of
        # the most recent sensor reading at or before its computed_at.
        feature_df = feature_df.sort_values(["machine_id", "computed_at"])
        cycles_combined = cycles_combined.sort_values(
            ["machine_id", "recorded_at"]
        )
        merged = pd.merge_asof(
            feature_df.rename(columns={"computed_at": "_computed_at"}),
            cycles_combined.rename(columns={"recorded_at": "_computed_at"})[
                ["machine_id", "_computed_at", "cycle"]
            ],
            on="_computed_at",
            by="machine_id",
            direction="backward",
        ).rename(columns={"_computed_at": "computed_at"})

        # Drop rows where no cycle was found (edge of window).
        merged = merged.dropna(subset=["cycle"])
        merged["cycle"] = merged["cycle"].astype(int)

        # Step 3: RUL labels.
        labelled = self._rul_labeler.assign_rul(merged)

        # Step 4: health labels.
        labelled = self._health_labeler.assign_health(labelled)

        # Step 5: machine-level train/val/test split.
        train_ids, val_ids, test_ids = _stratified_machine_split(
            machine_ids,
            self._cfg.train_fraction,
            self._cfg.val_fraction,
        )

        splits: dict[str, pd.DataFrame] = {
            "train": labelled[labelled["machine_id"].isin(train_ids)].copy(),
            "validation": labelled[
                labelled["machine_id"].isin(val_ids)
            ].copy(),
            "test": labelled[labelled["machine_id"].isin(test_ids)].copy(),
        }

        log.info(
            "feature_pipeline_splits",
            n_train=len(splits["train"]),
            n_validation=len(splits["validation"]),
            n_test=len(splits["test"]),
            train_machines=len(train_ids),
            val_machines=len(val_ids),
            test_machines=len(test_ids),
        )

        # Step 6: persist all three splits.
        for split_name, split_df in splits.items():
            await self._store.save_snapshot(split_df, split_name)

        log.info(
            "feature_pipeline_complete",
            dataset_version=self._cfg.dataset_version,
            total_rows=len(labelled),
        )
        return splits

    async def close(self) -> None:
        """Release all database connections."""
        await self._extractor.close()
        await self._store.close()
