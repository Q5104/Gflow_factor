"""Independent Barra long-short series used by the reward penalty.

Stocks are equally weighted within each top/bottom leg. The five style factors
are never equally weighted or otherwise combined into a synthetic exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
import numpy.typing as npt

from factor_gfn.evaluator.cross_section import clean_factor_cross_sections

from .config import BarraConfig, DEFAULT_BARRA_CONFIG
from .factors import BarraFactorSet, STYLE_NAMES


@dataclass(frozen=True, slots=True)
class LongShortSeries:
    long_return: npt.NDArray[np.float64]
    short_return: npt.NDArray[np.float64]
    long_short_return: npt.NDArray[np.float64]
    universe_count: npt.NDArray[np.int64]
    leg_count: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LongShortSummary:
    mean_period_return: float
    annualized_return: float
    annualized_ir: float
    period_std: float
    valid_periods: int


@dataclass(frozen=True, slots=True)
class BarraPenaltyResult:
    barra_ts_corr: float
    correlations: dict[str, float]
    valid_periods: dict[str, int]


def equal_weight_long_short(
    exposure: npt.ArrayLike,
    forward_returns: npt.ArrayLike,
    rebalance_indices: npt.ArrayLike,
    config: BarraConfig = DEFAULT_BARRA_CONFIG,
    *,
    universe_mask: npt.ArrayLike | None = None,
) -> LongShortSeries:
    factor = np.asarray(exposure, dtype=np.float64)
    returns = np.asarray(forward_returns, dtype=np.float64)
    if factor.ndim != 2 or returns.shape != factor.shape:
        raise ValueError("exposure 与 forward_returns 必须是同形 (date, stock) 矩阵")
    indices = np.asarray(rebalance_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("rebalance_indices 必须是一维整数数组")
    indices = indices.astype(np.int64, copy=False)
    if indices.size and ((indices < 0).any() or (indices >= factor.shape[0]).any()):
        raise IndexError("rebalance_indices 包含越界日期")
    if np.unique(indices).size != indices.size:
        raise ValueError("rebalance_indices 不能重复")
    factor = clean_factor_cross_sections(
        factor,
        universe_mask,
        row_indices=indices,
        config=config.cleaning,
    )

    date_count = factor.shape[0]
    long_return = np.full(date_count, np.nan)
    short_return = np.full(date_count, np.nan)
    long_short_return = np.full(date_count, np.nan)
    universe_count = np.zeros(date_count, dtype=np.int64)
    leg_count = np.zeros(date_count, dtype=np.int64)

    for date_index in indices:
        valid = np.isfinite(factor[date_index]) & np.isfinite(returns[date_index])
        count = int(valid.sum())
        universe_count[date_index] = count
        if count < config.min_cross_section_count:
            continue
        values = factor[date_index, valid]
        if np.ptp(values) <= np.finfo(np.float64).eps:
            continue
        period_returns = returns[date_index, valid]
        selected_count = min(max(1, ceil(count * config.long_short_quantile)), count // 2)
        order = np.argsort(values, kind="stable")
        bottom = order[:selected_count]
        top = order[-selected_count:]
        leg_count[date_index] = selected_count
        long_return[date_index] = float(period_returns[top].mean())
        short_return[date_index] = float(period_returns[bottom].mean())
        long_short_return[date_index] = long_return[date_index] - short_return[date_index]

    return LongShortSeries(
        long_return=long_return,
        short_return=short_return,
        long_short_return=long_short_return,
        universe_count=universe_count,
        leg_count=leg_count,
    )


def build_barra_long_short_returns(
    factor_set: BarraFactorSet,
    forward_returns: npt.ArrayLike,
    rebalance_indices: npt.ArrayLike,
    config: BarraConfig = DEFAULT_BARRA_CONFIG,
    *,
    universe_mask: npt.ArrayLike | None = None,
) -> dict[str, LongShortSeries]:
    """Return five separate style long-short series; never combine styles."""
    return {
        name: equal_weight_long_short(
            factor_set.exposures[name],
            forward_returns,
            rebalance_indices,
            config,
            universe_mask=universe_mask,
        )
        for name in STYLE_NAMES
    }


def summarize_long_short(
    values: npt.ArrayLike,
    annualization: float = 252.0 / 5.0,
    ddof: int = DEFAULT_BARRA_CONFIG.performance_ddof,
) -> LongShortSummary:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values 必须是一维收益序列")
    valid = array[np.isfinite(array)]
    mean = float(valid.mean()) if valid.size else np.nan
    std = float(valid.std(ddof=ddof)) if valid.size > ddof else np.nan
    ir = mean / std * sqrt(annualization) if np.isfinite(std) and std > 0 else np.nan
    return LongShortSummary(
        mean_period_return=mean,
        annualized_return=float(mean * annualization) if np.isfinite(mean) else np.nan,
        annualized_ir=float(ir),
        period_std=std,
        valid_periods=int(valid.size),
    )


def cumulative_return(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values 必须是一维收益序列")
    result = np.full(array.shape, np.nan)
    wealth = 1.0
    for index, value in enumerate(array):
        if np.isfinite(value):
            wealth *= 1.0 + value
            result[index] = wealth - 1.0
    return result


def calculate_barra_ts_corr(
    candidate_long_short: npt.ArrayLike,
    barra_long_short: dict[str, LongShortSeries],
    min_periods: int = DEFAULT_BARRA_CONFIG.min_common_periods,
) -> BarraPenaltyResult:
    """Return ``max_k abs(corr(candidate_ls, barra_k_ls))`` across five styles."""
    if min_periods < 2:
        raise ValueError("min_periods 至少为 2")
    candidate = np.asarray(candidate_long_short, dtype=np.float64)
    if candidate.ndim != 1:
        raise ValueError("candidate_long_short 必须是一维收益序列")
    correlations: dict[str, float] = {}
    valid_periods: dict[str, int] = {}
    for name in STYLE_NAMES:
        reference = np.asarray(barra_long_short[name].long_short_return, dtype=np.float64)
        if reference.shape != candidate.shape:
            raise ValueError(f"{name} 收益序列长度与候选序列不一致")
        valid = np.isfinite(candidate) & np.isfinite(reference)
        count = int(valid.sum())
        valid_periods[name] = count
        if count < min_periods or np.std(candidate[valid]) <= 0 or np.std(reference[valid]) <= 0:
            correlations[name] = np.nan
        else:
            correlations[name] = float(np.corrcoef(candidate[valid], reference[valid])[0, 1])
    finite = [abs(value) for value in correlations.values() if np.isfinite(value)]
    return BarraPenaltyResult(
        barra_ts_corr=max(finite) if finite else np.nan,
        correlations=correlations,
        valid_periods=valid_periods,
    )


__all__ = [
    "BarraPenaltyResult",
    "LongShortSeries",
    "LongShortSummary",
    "build_barra_long_short_returns",
    "calculate_barra_ts_corr",
    "cumulative_return",
    "equal_weight_long_short",
    "summarize_long_short",
]
