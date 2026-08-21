"""Causal rolling-ICIR weights for the five-trading-day OOS calendar."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True, slots=True)
class RollingICIRConfig:
    window_observations: int = 150
    min_observations: int = 100
    update_every_periods: int = 4
    maturity_lag_periods: int = 2
    shrinkage_to_equal: float = 0.5
    max_weight: float = 0.03
    winsor_quantile: float = 0.95
    min_cross_section_count: int = 20
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.window_observations < 2:
            raise ValueError("window_observations must be at least 2")
        if not 2 <= self.min_observations <= self.window_observations:
            raise ValueError("min_observations must be within the rolling window")
        if self.update_every_periods < 1 or self.maturity_lag_periods < 1:
            raise ValueError("update and maturity lags must be positive")
        if not 0.0 <= self.shrinkage_to_equal <= 1.0:
            raise ValueError("shrinkage_to_equal must be in [0, 1]")
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0.0 < self.winsor_quantile <= 1.0:
            raise ValueError("winsor_quantile must be in (0, 1]")
        if self.min_cross_section_count < 2:
            raise ValueError("min_cross_section_count must be at least 2")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_observations": self.window_observations,
            "min_observations": self.min_observations,
            "update_every_periods": self.update_every_periods,
            "maturity_lag_periods": self.maturity_lag_periods,
            "shrinkage_to_equal": self.shrinkage_to_equal,
            "max_weight": self.max_weight,
            "winsor_quantile": self.winsor_quantile,
            "min_cross_section_count": self.min_cross_section_count,
            "epsilon": self.epsilon,
            "ic_frequency": "every_5_trading_days",
            "label_formula": "open[t+6] / open[t+1] - 1",
            "annualized": False,
            "std_ddof": 1,
            "positive_clipping": True,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RollingICIRConfig":
        fields = {
            key: value[key]
            for key in (
                "window_observations",
                "min_observations",
                "update_every_periods",
                "maturity_lag_periods",
                "shrinkage_to_equal",
                "max_weight",
                "winsor_quantile",
                "min_cross_section_count",
                "epsilon",
            )
        }
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class RollingICIRResult:
    scores: npt.NDArray[np.float64]
    weights_by_update: pd.DataFrame
    diagnostics_by_update: pd.DataFrame


def cross_sectional_spearman(
    values: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    min_count: int,
    epsilon: float,
) -> float:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < min_count:
        return math.nan
    x = x[valid]
    y = y[valid]
    if np.ptp(x) <= epsilon or np.ptp(y) <= epsilon:
        return math.nan
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if not math.isfinite(denominator) or denominator <= epsilon:
        return math.nan
    return float(np.dot(x_centered, y_centered) / denominator)


def periodic_ic_matrix(
    dates: npt.ArrayLike,
    values: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    min_cross_section_count: int,
    epsilon: float,
) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]:
    date_array = np.asarray(dates).astype("datetime64[D]")
    feature_array = np.asarray(values, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    if feature_array.ndim != 2 or feature_array.shape[0] != date_array.size:
        raise ValueError("rolling IC features and dates are misaligned")
    if label_array.shape != date_array.shape:
        raise ValueError("rolling IC labels and dates are misaligned")
    unique_dates = np.unique(date_array)
    result = np.full((unique_dates.size, feature_array.shape[1]), np.nan)
    for date_index, date in enumerate(unique_dates):
        mask = date_array == date
        for factor_index in range(feature_array.shape[1]):
            result[date_index, factor_index] = cross_sectional_spearman(
                feature_array[mask, factor_index],
                label_array[mask],
                min_count=min_cross_section_count,
                epsilon=epsilon,
            )
    return unique_dates, result


def _redistribute_cap(weights: npt.NDArray[np.float64], cap: float) -> npt.NDArray[np.float64]:
    result = np.asarray(weights, dtype=np.float64).copy()
    for _ in range(result.size + 1):
        over = result > cap + 1e-15
        if not over.any():
            break
        excess = float(np.sum(result[over] - cap))
        result[over] = cap
        under = ~over
        room = np.maximum(cap - result[under], 0.0)
        room_total = float(room.sum())
        if room_total <= 0:
            break
        result[under] += excess * room / room_total
    result /= float(result.sum())
    return result


def estimate_rolling_weights(
    ic_history: npt.ArrayLike,
    config: RollingICIRConfig,
    *,
    previous_weights: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    history = np.asarray(ic_history, dtype=np.float64)
    if history.ndim != 2 or history.shape[1] == 0:
        raise ValueError("IC history must be a nonempty date-by-factor matrix")
    history = history[-config.window_observations :]
    factor_count = history.shape[1]
    equal = np.full(factor_count, 1.0 / factor_count)
    finite_counts = np.isfinite(history).sum(axis=0)
    candidates = np.zeros(factor_count, dtype=np.float64)
    raw_icir = np.full(factor_count, np.nan)
    for factor_index in range(factor_count):
        finite = history[np.isfinite(history[:, factor_index]), factor_index]
        if finite.size < config.min_observations:
            continue
        standard_deviation = float(np.std(finite, ddof=1))
        if not math.isfinite(standard_deviation) or standard_deviation <= config.epsilon:
            continue
        raw_icir[factor_index] = float(np.mean(finite) / standard_deviation)
        candidates[factor_index] = max(raw_icir[factor_index], 0.0)

    positive = candidates[candidates > 0]
    fallback = positive.size == 0 or float(candidates.sum()) <= config.epsilon
    if fallback:
        if previous_weights is None:
            weights = equal
            fallback_reason = "all_nonpositive_or_invalid_icir"
        else:
            weights = np.asarray(previous_weights, dtype=np.float64).copy()
            weights /= float(weights.sum())
            fallback_reason = "all_nonpositive_or_invalid_icir_keep_previous"
    else:
        winsor_cap = float(np.quantile(positive, config.winsor_quantile))
        clipped = np.minimum(candidates, winsor_cap)
        dynamic = clipped / float(clipped.sum())
        weights = config.shrinkage_to_equal * equal + (1.0 - config.shrinkage_to_equal) * dynamic
        effective_cap = max(config.max_weight, 1.0 / factor_count)
        weights = _redistribute_cap(weights, effective_cap)
        fallback_reason = None

    diagnostics = {
        "history_period_count": int(history.shape[0]),
        "finite_observation_min": int(finite_counts.min()),
        "finite_observation_median": float(np.median(finite_counts)),
        "finite_observation_max": int(finite_counts.max()),
        "positive_dynamic_factor_count": int(np.sum(candidates > 0)),
        "fallback_status": fallback,
        "fallback_reason": fallback_reason,
        "max_weight": float(weights.max()),
        "min_weight": float(weights.min()),
        "weight_hhi": float(np.dot(weights, weights)),
        "raw_icir": raw_icir,
    }
    return weights, diagnostics


def generate_causal_rolling_scores(
    dates: npt.ArrayLike,
    values: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    aliases: tuple[str, ...],
    seed_dates: npt.ArrayLike,
    seed_ic_values: npt.ArrayLike,
    config: RollingICIRConfig,
) -> RollingICIRResult:
    date_array = np.asarray(dates).astype("datetime64[D]")
    feature_array = np.asarray(values, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    if feature_array.shape != (date_array.size, len(aliases)):
        raise ValueError("rolling score feature identity mismatch")
    if label_array.shape != date_array.shape:
        raise ValueError("rolling score labels are misaligned")
    unique_dates, oos_ic = periodic_ic_matrix(
        date_array,
        feature_array,
        label_array,
        min_cross_section_count=config.min_cross_section_count,
        epsilon=config.epsilon,
    )
    seed_date_array = np.asarray(seed_dates).astype("datetime64[D]")
    seed_values = np.asarray(seed_ic_values, dtype=np.float64)
    if seed_values.shape != (seed_date_array.size, len(aliases)):
        raise ValueError("rolling IC seed identity mismatch")
    if seed_date_array.size and np.any(seed_date_array[1:] <= seed_date_array[:-1]):
        raise ValueError("rolling IC seed dates must be strictly increasing")
    if seed_date_array.size and unique_dates.size and seed_date_array[-1] >= unique_dates[0]:
        raise ValueError("rolling IC seed must end before OOS")

    scores = np.full(date_array.size, np.nan)
    weight_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    weights: npt.NDArray[np.float64] | None = None
    for date_index, date in enumerate(unique_dates):
        if date_index % config.update_every_periods == 0:
            last_mature_index = date_index - config.maturity_lag_periods
            matured_count = max(last_mature_index + 1, 0)
            combined_dates = np.concatenate((seed_date_array, unique_dates[:matured_count]))
            combined_ic = np.vstack((seed_values, oos_ic[:matured_count]))
            if combined_dates.size > config.window_observations:
                combined_dates = combined_dates[-config.window_observations :]
                combined_ic = combined_ic[-config.window_observations :]
            weights, diagnostics = estimate_rolling_weights(
                combined_ic,
                config,
                previous_weights=weights,
            )
            for factor_index, alias in enumerate(aliases):
                weight_rows.append(
                    {
                        "update_date": pd.Timestamp(date),
                        "factor_alias": alias,
                        "weight": float(weights[factor_index]),
                    }
                )
            diagnostic_rows.append(
                {
                    "update_date": pd.Timestamp(date),
                    "history_start": pd.Timestamp(combined_dates[0]) if combined_dates.size else pd.NaT,
                    "history_end": pd.Timestamp(combined_dates[-1]) if combined_dates.size else pd.NaT,
                    "latest_mature_oos_date": (
                        pd.Timestamp(unique_dates[matured_count - 1]) if matured_count else pd.NaT
                    ),
                    **{key: value for key, value in diagnostics.items() if key != "raw_icir"},
                }
            )
        if weights is None:
            raise RuntimeError("rolling weights were not initialized")
        mask = date_array == date
        scores[mask] = feature_array[mask] @ weights
    if not np.isfinite(scores).all():
        raise ValueError("rolling ICIR scores contain nonfinite values")
    return RollingICIRResult(
        scores=scores,
        weights_by_update=pd.DataFrame(weight_rows),
        diagnostics_by_update=pd.DataFrame(diagnostic_rows),
    )


__all__ = [
    "RollingICIRConfig",
    "RollingICIRResult",
    "cross_sectional_spearman",
    "estimate_rolling_weights",
    "generate_causal_rolling_scores",
    "periodic_ic_matrix",
]
