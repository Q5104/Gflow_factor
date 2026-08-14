"""Pure median/IQR stability assessment for learned-normalizer calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np


CALIBRATION_STABILITY_SCHEMA = "factor_gfn.calibration_stability.v1"
CalibrationStabilityStatus = Literal[
    "collecting", "stable", "continue_to_hard_limit", "fail_closed"
]


@dataclass(frozen=True, slots=True)
class CalibrationStabilityConfig:
    """Frozen absolute-only engineering precheck for formal no-anchor runs."""

    minimum_valid_samples: int = 64
    maximum_requested_slots: int = 128
    comparison_window: int = 16
    median_absolute_tolerance: float = 0.25
    iqr_absolute_tolerance: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "minimum_valid_samples",
            "maximum_requested_slots",
            "comparison_window",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_valid_samples != 64:
            raise ValueError("formal no-anchor minimum_valid_samples must be 64")
        if self.maximum_requested_slots != 128:
            raise ValueError("formal no-anchor maximum_requested_slots must be 128")
        if self.comparison_window != 16:
            raise ValueError("formal no-anchor comparison_window must be 16")
        if self.median_absolute_tolerance != 0.25:
            raise ValueError("formal median_absolute_tolerance must be 0.25 logZ")
        if self.iqr_absolute_tolerance != 0.50:
            raise ValueError("formal iqr_absolute_tolerance must be 0.50 logZ")


@dataclass(frozen=True, slots=True)
class CalibrationStabilityResult:
    status: CalibrationStabilityStatus
    requested_slots: int
    valid_samples: int
    median: float | None
    iqr: float | None
    previous_window_median: float | None
    recent_window_median: float | None
    previous_window_iqr: float | None
    recent_window_iqr: float | None
    median_shift: float | None
    iqr_shift: float | None
    reason: str


def _median_iqr(values: np.ndarray) -> tuple[float, float]:
    quantiles = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return float(quantiles[1]), float(quantiles[2] - quantiles[0])


def assess_calibration_stability(
    observations: Iterable[float],
    *,
    requested_slots: int,
    config: CalibrationStabilityConfig,
) -> CalibrationStabilityResult:
    if isinstance(requested_slots, bool) or not isinstance(requested_slots, int):
        raise TypeError("requested_slots must be an integer")
    if requested_slots < 0 or requested_slots > config.maximum_requested_slots:
        raise ValueError("requested_slots is outside the configured budget")
    values = np.asarray(tuple(observations), dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("calibration observations must be a finite 1-D sequence")
    if values.size > requested_slots:
        raise ValueError("valid observations cannot exceed requested slots")

    common = {
        "requested_slots": requested_slots,
        "valid_samples": int(values.size),
        "median": None,
        "iqr": None,
        "previous_window_median": None,
        "recent_window_median": None,
        "previous_window_iqr": None,
        "recent_window_iqr": None,
        "median_shift": None,
        "iqr_shift": None,
    }
    if values.size < config.minimum_valid_samples:
        exhausted = requested_slots >= config.maximum_requested_slots
        return CalibrationStabilityResult(
            status="fail_closed" if exhausted else "collecting",
            reason=(
                "hard request budget exhausted before 64 valid samples"
                if exhausted
                else "fewer than 64 valid samples"
            ),
            **common,
        )

    median, iqr = _median_iqr(values)
    is_recheck_count = (
        (values.size - config.minimum_valid_samples) % config.comparison_window == 0
    )
    if not is_recheck_count and requested_slots < config.maximum_requested_slots:
        return CalibrationStabilityResult(
            status="continue_to_hard_limit",
            requested_slots=requested_slots,
            valid_samples=int(values.size),
            median=median,
            iqr=iqr,
            previous_window_median=None,
            recent_window_median=None,
            previous_window_iqr=None,
            recent_window_iqr=None,
            median_shift=None,
            iqr_shift=None,
            reason="awaiting the next 16-valid-sample stability checkpoint",
        )
    width = config.comparison_window
    previous = values[-2 * width : -width]
    recent = values[-width:]
    previous_median, previous_iqr = _median_iqr(previous)
    recent_median, recent_iqr = _median_iqr(recent)
    median_shift = abs(recent_median - previous_median)
    iqr_shift = abs(recent_iqr - previous_iqr)
    stable = (
        median_shift <= config.median_absolute_tolerance
        and iqr_shift <= config.iqr_absolute_tolerance
    )
    if stable:
        status: CalibrationStabilityStatus = "stable"
        reason = "median and IQR are stable across the two latest windows"
    elif requested_slots >= config.maximum_requested_slots:
        status = "fail_closed"
        reason = "median or IQR remains unstable at the hard request budget"
    else:
        status = "continue_to_hard_limit"
        reason = "minimum sufficiency reached but median or IQR is unstable"
    return CalibrationStabilityResult(
        status=status,
        requested_slots=requested_slots,
        valid_samples=int(values.size),
        median=median,
        iqr=iqr,
        previous_window_median=previous_median,
        recent_window_median=recent_median,
        previous_window_iqr=previous_iqr,
        recent_window_iqr=recent_iqr,
        median_shift=median_shift,
        iqr_shift=iqr_shift,
        reason=reason,
    )


__all__ = [
    "CALIBRATION_STABILITY_SCHEMA",
    "CalibrationStabilityConfig",
    "CalibrationStabilityResult",
    "CalibrationStabilityStatus",
    "assess_calibration_stability",
]
