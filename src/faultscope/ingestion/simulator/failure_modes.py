"""Degradation pattern implementations for the FaultScope simulator.

Each :class:`DegradationCurve` instance encapsulates a single
pattern that maps a cycle number to a normalised wear signal in
``[0.0, 1.0]`` where ``0.0`` is perfectly healthy and ``1.0``
represents end-of-life.

Patterns
--------
LINEAR
    Straight-line increase from 0 to 1 over ``total_cycles``.
EXPONENTIAL
    Slow start, rapid acceleration toward failure using a
    re-scaled exponential.
STEP
    Mostly stable with sudden jumps at randomly chosen cycle
    thresholds; each jump adds to a running total.
OSCILLATING
    Cyclic signal (sine wave) layered on an upward-trending
    baseline so both the amplitude and the mean increase with
    wear.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import numpy.typing as npt


class DegradationPattern(StrEnum):
    """Supported degradation curve shapes."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    OSCILLATING = "oscillating"


class DegradationCurve:
    """Generates a normalised degradation signal ``[0.0 .. 1.0]``.

    ``0.0`` means the machine is healthy; ``1.0`` means it has
    reached end-of-life.

    Parameters
    ----------
    pattern:
        Which curve shape to use.
    total_cycles:
        The number of cycles after which the machine is considered
        failed (degradation reaches 1.0).
    rng:
        A seeded NumPy random generator used for stochastic
        components so that runs are reproducible.

    Raises
    ------
    ValueError
        If ``total_cycles`` is not a positive integer.
    """

    def __init__(
        self,
        pattern: DegradationPattern,
        total_cycles: int,
        rng: np.random.Generator,
    ) -> None:
        if total_cycles < 1:
            raise ValueError(f"total_cycles must be >= 1, got {total_cycles}")
        self._pattern = pattern
        self._total = total_cycles
        self._rng = rng

        # Pre-compute pattern-specific parameters once so that
        # repeated calls to sample() are cheap and deterministic.
        self._params: dict[str, float] = self._init_params()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_params(self) -> dict[str, float]:
        """Compute stochastic parameters from the RNG at init time."""
        if self._pattern == DegradationPattern.LINEAR:
            return {}

        if self._pattern == DegradationPattern.EXPONENTIAL:
            # k controls how steeply the curve rises;
            # larger k => slower start then rapid surge.
            return {"k": float(self._rng.uniform(4.0, 8.0))}

        if self._pattern == DegradationPattern.STEP:
            # Choose 3–6 step positions (as fractions of total_cycles)
            n_steps: int = int(self._rng.integers(3, 7))
            # Positions are sorted ascending; the last step can be at
            # most 90 % of the way through the lifecycle so there is
            # still some time at the degraded level.
            positions: npt.NDArray[np.float64] = np.sort(
                self._rng.uniform(0.05, 0.90, size=n_steps)
            )
            # Heights of each step sum to ≈ 1.0.
            heights_raw = self._rng.uniform(0.05, 0.3, size=n_steps)
            heights: npt.NDArray[np.float64] = heights_raw / heights_raw.sum()
            return {
                "n_steps": float(n_steps),
                **{f"pos_{i}": float(positions[i]) for i in range(n_steps)},
                **{f"h_{i}": float(heights[i]) for i in range(n_steps)},
            }

        if self._pattern == DegradationPattern.OSCILLATING:
            # frequency in cycles/total_cycles (2–6 full oscillations)
            return {
                "freq": float(self._rng.uniform(2.0, 6.0)),
                # damping of the oscillation amplitude near end-of-life
                "amp_scale": float(self._rng.uniform(0.08, 0.18)),
            }

        raise ValueError(  # pragma: no cover
            f"Unknown DegradationPattern: {self._pattern}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, cycle: int) -> float:
        """Return the degradation level at *cycle*.

        The returned value is always clamped to ``[0.0, 1.0]``.

        Parameters
        ----------
        cycle:
            The zero-based cycle index (0 = brand new).

        Returns
        -------
        float
            Normalised degradation level in ``[0.0, 1.0]``.
        """
        t: float = min(cycle / self._total, 1.0)

        if self._pattern == DegradationPattern.LINEAR:
            return float(np.clip(t, 0.0, 1.0))

        if self._pattern == DegradationPattern.EXPONENTIAL:
            k = self._params["k"]
            # Rescaled so that sample(0) ≈ 0 and sample(total) ≈ 1.
            raw = (np.exp(k * t) - 1.0) / (np.exp(k) - 1.0)
            return float(np.clip(raw, 0.0, 1.0))

        if self._pattern == DegradationPattern.STEP:
            n = int(self._params["n_steps"])
            level = 0.0
            for i in range(n):
                if t >= self._params[f"pos_{i}"]:
                    level += self._params[f"h_{i}"]
            return float(np.clip(level, 0.0, 1.0))

        if self._pattern == DegradationPattern.OSCILLATING:
            freq = self._params["freq"]
            amp = self._params["amp_scale"]
            # Upward trend component
            trend = t
            # Oscillation with amplitude that grows with degradation
            oscillation = amp * (1.0 + t) * np.sin(2.0 * np.pi * freq * t)
            raw = trend + oscillation
            return float(np.clip(raw, 0.0, 1.0))

        raise RuntimeError(  # pragma: no cover
            f"Unhandled DegradationPattern: {self._pattern}"
        )
