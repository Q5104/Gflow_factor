"""NumPy implementation of five manually constructed Barra-style exposures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .config import BarraConfig, DEFAULT_BARRA_CONFIG


FloatMatrix = npt.NDArray[np.float64]
STYLE_NAMES = ("market_beta", "size", "momentum", "volatility", "liquidity")


@dataclass(frozen=True, slots=True)
class BarraFactorSet:
    exposures: dict[str, FloatMatrix]
    market_return: npt.NDArray[np.float64]
    stock_return: FloatMatrix | None = None

    def __post_init__(self) -> None:
        missing = set(STYLE_NAMES).difference(self.exposures)
        extra = set(self.exposures).difference(STYLE_NAMES)
        if missing or extra:
            raise ValueError(f"Barra exposure 名称不一致；缺少={sorted(missing)}，多余={sorted(extra)}")


def _matrix(values: npt.ArrayLike, name: str) -> FloatMatrix:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or 0 in result.shape:
        raise ValueError(f"{name} 必须是非空 (date, stock) 二维矩阵")
    return result


def _same_shape(reference: FloatMatrix, **matrices: npt.ArrayLike) -> dict[str, FloatMatrix]:
    result: dict[str, FloatMatrix] = {}
    for name, values in matrices.items():
        matrix = _matrix(values, name)
        if matrix.shape != reference.shape:
            raise ValueError(f"{name} 形状 {matrix.shape} 与基准 {reference.shape} 不一致")
        result[name] = matrix
    return result


def daily_close_returns(adjusted_close: npt.ArrayLike) -> FloatMatrix:
    close = _matrix(adjusted_close, "adjusted_close")
    result = np.full(close.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[1:] > 0) & (close[:-1] > 0)
    target = result[1:]
    np.divide(close[1:], close[:-1], out=target, where=valid)
    target[valid] -= 1.0
    target[~valid] = np.nan
    return result


def market_cap_weighted_return(
    stock_return: npt.ArrayLike,
    market_cap: npt.ArrayLike,
    universe_mask: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    returns = _matrix(stock_return, "stock_return")
    aligned = _same_shape(returns, market_cap=market_cap)
    cap = aligned["market_cap"]
    universe = np.asarray(universe_mask, dtype=bool)
    if universe.shape != returns.shape:
        raise ValueError("universe_mask 形状必须与收益矩阵一致")

    result = np.full(returns.shape[0], np.nan, dtype=np.float64)
    if returns.shape[0] < 2:
        return result
    lagged_cap = cap[:-1]
    valid = (
        np.isfinite(returns[1:])
        & np.isfinite(lagged_cap)
        & (lagged_cap > 0)
        & universe[1:]
        & universe[:-1]
    )
    weights = np.where(valid, lagged_cap, 0.0)
    denominators = weights.sum(axis=1)
    numerators = (weights * np.where(valid, returns[1:], 0.0)).sum(axis=1)
    usable = denominators > 0
    target = result[1:]
    target[usable] = numerators[usable] / denominators[usable]
    return result


def _window_sums(values: FloatMatrix, window: int) -> tuple[FloatMatrix, FloatMatrix, npt.NDArray[np.int64]]:
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    clean_sq = clean * clean
    prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(clean, axis=0)])
    prefix_sq = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(clean_sq, axis=0)])
    prefix_count = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.int64), np.cumsum(valid, axis=0, dtype=np.int64)]
    )
    ends = np.arange(1, values.shape[0] + 1)
    starts = np.maximum(0, ends - window)
    return prefix[ends] - prefix[starts], prefix_sq[ends] - prefix_sq[starts], prefix_count[
        ends
    ] - prefix_count[starts]


def rolling_mean_strict(values: npt.ArrayLike, window: int) -> FloatMatrix:
    matrix = _matrix(values, "values")
    result = np.full(matrix.shape, np.nan, dtype=np.float64)
    if window <= 0:
        raise ValueError("window 必须为正整数")
    if window > matrix.shape[0]:
        return result
    sums, _, counts = _window_sums(matrix, window)
    valid = counts == window
    np.divide(sums, counts, out=result, where=valid)
    result[~valid] = np.nan
    return result


def rolling_std_strict(
    values: npt.ArrayLike,
    window: int,
    min_periods: int | None = None,
) -> FloatMatrix:
    matrix = _matrix(values, "values")
    result = np.full(matrix.shape, np.nan, dtype=np.float64)
    if window <= 0:
        raise ValueError("window 必须为正整数")
    required = window if min_periods is None else int(min_periods)
    if not 2 <= required <= window:
        raise ValueError("min_periods 必须位于 [2, window] 内")
    sums, sums_sq, counts = _window_sums(matrix, window)
    valid = counts >= required
    means = np.zeros_like(sums)
    np.divide(sums, counts, out=means, where=valid)
    second_moment = np.zeros_like(sums_sq)
    np.divide(sums_sq, counts, out=second_moment, where=valid)
    variance = np.maximum(second_moment - means * means, 0.0)
    result[valid] = np.sqrt(variance[valid])
    return result


def rolling_beta_strict(
    stock_return: npt.ArrayLike,
    market_return: npt.ArrayLike,
    window: int,
    stock_chunk_size: int = DEFAULT_BARRA_CONFIG.stock_chunk_size,
    min_periods: int | None = None,
) -> FloatMatrix:
    stocks = _matrix(stock_return, "stock_return")
    market = np.asarray(market_return, dtype=np.float64)
    if market.ndim != 1 or market.shape[0] != stocks.shape[0]:
        raise ValueError("market_return 必须是一维且 date 轴与 stock_return 一致")
    if window <= 0 or stock_chunk_size <= 0:
        raise ValueError("window 和 stock_chunk_size 必须为正整数")
    required = window if min_periods is None else int(min_periods)
    if not 2 <= required <= window:
        raise ValueError("min_periods 必须位于 [2, window] 内")
    result = np.full(stocks.shape, np.nan, dtype=np.float64)

    eps = np.finfo(np.float64).eps
    for start in range(0, stocks.shape[1], stock_chunk_size):
        stop = min(start + stock_chunk_size, stocks.shape[1])
        x = stocks[:, start:stop]
        y = np.broadcast_to(market[:, None], x.shape)
        pair_valid = np.isfinite(x) & np.isfinite(y)

        def window_sum(values: FloatMatrix) -> FloatMatrix:
            prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
            ends = np.arange(1, values.shape[0] + 1)
            starts = np.maximum(0, ends - window)
            return prefix[ends] - prefix[starts]

        clean_x = np.where(pair_valid, x, 0.0)
        clean_y = np.where(pair_valid, y, 0.0)
        counts = window_sum(pair_valid.astype(np.float64))
        sum_x = window_sum(clean_x)
        sum_y = window_sum(clean_y)
        sum_xy = window_sum(clean_x * clean_y)
        sum_y2 = window_sum(clean_y * clean_y)
        valid = counts >= required
        mean_x = np.zeros_like(sum_x)
        mean_y = np.zeros_like(sum_y)
        mean_xy = np.zeros_like(sum_xy)
        mean_y2 = np.zeros_like(sum_y2)
        np.divide(sum_x, counts, out=mean_x, where=valid)
        np.divide(sum_y, counts, out=mean_y, where=valid)
        np.divide(sum_xy, counts, out=mean_xy, where=valid)
        np.divide(sum_y2, counts, out=mean_y2, where=valid)
        covariance = mean_xy - mean_x * mean_y
        variance = np.maximum(mean_y2 - mean_y * mean_y, 0.0)
        usable = valid & (variance > eps)
        target = result[:, start:stop]
        np.divide(covariance, variance, out=target, where=usable)
        target[~usable] = np.nan
    return result


def momentum_exposure(
    adjusted_close: npt.ArrayLike,
    lookback: int,
    skip: int,
) -> FloatMatrix:
    close = _matrix(adjusted_close, "adjusted_close")
    if not 0 <= skip < lookback:
        raise ValueError("skip 必须位于 [0, lookback) 内")
    result = np.full(close.shape, np.nan, dtype=np.float64)
    if lookback >= close.shape[0]:
        return result
    recent = close[lookback - skip : close.shape[0] - skip] if skip else close[lookback:]
    old = close[: close.shape[0] - lookback]
    valid = np.isfinite(recent) & np.isfinite(old) & (recent > 0) & (old > 0)
    target = result[lookback:]
    np.divide(recent, old, out=target, where=valid)
    target[valid] -= 1.0
    target[~valid] = np.nan
    return result


def calculate_barra_factors(
    adjusted_close: npt.ArrayLike,
    volume: npt.ArrayLike,
    float_market_cap: npt.ArrayLike,
    total_market_cap: npt.ArrayLike,
    list_a_shares: npt.ArrayLike,
    universe_mask: npt.ArrayLike,
    config: BarraConfig = DEFAULT_BARRA_CONFIG,
) -> BarraFactorSet:
    close = _matrix(adjusted_close, "adjusted_close")
    matrices = _same_shape(
        close,
        volume=volume,
        float_market_cap=float_market_cap,
        total_market_cap=total_market_cap,
        list_a_shares=list_a_shares,
    )
    universe = np.asarray(universe_mask, dtype=bool)
    if universe.shape != close.shape:
        raise ValueError("universe_mask 形状必须与行情矩阵一致")

    stock_return = daily_close_returns(close)
    selected_cap = matrices[f"{config.market_cap_type}_market_cap"]
    market_return = market_cap_weighted_return(stock_return, selected_cap, universe)
    beta = rolling_beta_strict(
        stock_return,
        market_return,
        config.beta_window,
        config.stock_chunk_size,
        config.beta_min_periods,
    )

    size = np.full(close.shape, np.nan, dtype=np.float64)
    valid_cap = np.isfinite(selected_cap) & (selected_cap > 0)
    size[valid_cap] = np.log(selected_cap[valid_cap])
    momentum = momentum_exposure(close, config.momentum_lookback, config.momentum_skip)
    volatility = rolling_std_strict(
        stock_return,
        config.volatility_window,
        config.volatility_min_periods,
    )

    turnover = np.full(close.shape, np.nan, dtype=np.float64)
    valid_turnover = (
        np.isfinite(matrices["volume"])
        & (matrices["volume"] >= 0)
        & np.isfinite(matrices["list_a_shares"])
        & (matrices["list_a_shares"] > 0)
    )
    np.divide(
        matrices["volume"],
        matrices["list_a_shares"],
        out=turnover,
        where=valid_turnover,
    )
    liquidity = rolling_mean_strict(turnover, config.liquidity_window)

    exposures = {
        "market_beta": beta,
        "size": size,
        "momentum": momentum,
        "volatility": volatility,
        "liquidity": liquidity,
    }
    for values in exposures.values():
        values[~universe] = np.nan
        values[~np.isfinite(values)] = np.nan
    return BarraFactorSet(exposures=exposures, market_return=market_return, stock_return=stock_return)


__all__ = [
    "STYLE_NAMES",
    "BarraFactorSet",
    "calculate_barra_factors",
    "daily_close_returns",
    "market_cap_weighted_return",
    "momentum_exposure",
    "rolling_beta_strict",
    "rolling_mean_strict",
    "rolling_std_strict",
]
