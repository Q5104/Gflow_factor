"""非时序算子的 NumPy 基准实现。

所有公开算子接收形状为 ``(date, stock)`` 的二维数组，返回新的 ``float64``
数组，不修改输入。输入中的非有限值和计算产生的非有限值统一记为 NaN。
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

from factor_gfn.grammar.operators import (
    BINARY_OPERATORS,
    CROSS_SECTIONAL_OPERATORS,
    TS_BINARY_OPERATORS,
    TS_UNARY_OPERATORS,
    UNARY_OPERATORS,
)
from factor_gfn.grammar.tokens import WINDOWS


FloatMatrix = NDArray[np.float64]
OperatorFunction = Callable[..., FloatMatrix]

EPSILON = 1e-12
LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95
MIN_CROSS_SECTIONAL_COUNT = 2


def _matrix(values: ArrayLike, label: str = "输入") -> FloatMatrix:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{label}必须是二维 (date, stock) 数组，实际形状为 {array.shape}")
    result = array.copy()
    result[~np.isfinite(result)] = np.nan
    return result


def _binary_matrices(x: ArrayLike, y: ArrayLike) -> tuple[FloatMatrix, FloatMatrix]:
    left = _matrix(x, "左输入")
    right = _matrix(y, "右输入")
    if left.shape != right.shape:
        raise ValueError(f"二元算子输入形状必须一致：{left.shape} != {right.shape}")
    return left, right


def _finite_or_nan(values: ArrayLike) -> FloatMatrix:
    result = np.asarray(values, dtype=np.float64).copy()
    result[~np.isfinite(result)] = np.nan
    return result


def abs(x: ArrayLike) -> FloatMatrix:
    return np.abs(_matrix(x))


def neg(x: ArrayLike) -> FloatMatrix:
    return -_matrix(x)


def sign(x: ArrayLike) -> FloatMatrix:
    return np.sign(_matrix(x))


def log(x: ArrayLike) -> FloatMatrix:
    values = _matrix(x)
    with np.errstate(all="ignore"):
        return _finite_or_nan(np.log(np.abs(values) + EPSILON))


def inv(x: ArrayLike) -> FloatMatrix:
    values = _matrix(x)
    result = np.full_like(values, np.nan)
    valid = np.isfinite(values) & (np.abs(values) > EPSILON)
    with np.errstate(all="ignore"):
        np.divide(1.0, values, out=result, where=valid)
    return _finite_or_nan(result)


def sqrt(x: ArrayLike) -> FloatMatrix:
    values = _matrix(x)
    with np.errstate(all="ignore"):
        return _finite_or_nan(np.sqrt(np.abs(values)))


def tanh(x: ArrayLike) -> FloatMatrix:
    return np.tanh(_matrix(x))


def relu(x: ArrayLike) -> FloatMatrix:
    return np.maximum(_matrix(x), 0.0)


def softsign(x: ArrayLike) -> FloatMatrix:
    values = _matrix(x)
    with np.errstate(all="ignore"):
        return _finite_or_nan(values / (1.0 + np.abs(values)))


def signed_power2(x: ArrayLike) -> FloatMatrix:
    values = _matrix(x)
    with np.errstate(all="ignore"):
        return _finite_or_nan(np.sign(values) * np.abs(values) ** 2)


def signed_power3(x: ArrayLike) -> FloatMatrix:
    values = _matrix(x)
    with np.errstate(all="ignore"):
        return _finite_or_nan(np.sign(values) * np.abs(values) ** 3)


def signed_log1p(x: ArrayLike) -> FloatMatrix:
    values = _matrix(x)
    with np.errstate(all="ignore"):
        return _finite_or_nan(np.sign(values) * np.log1p(np.abs(values)))


def add(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    with np.errstate(all="ignore"):
        return _finite_or_nan(left + right)


def sub(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    with np.errstate(all="ignore"):
        return _finite_or_nan(left - right)


def mul(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    with np.errstate(all="ignore"):
        return _finite_or_nan(left * right)


def div(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    result = np.full_like(left, np.nan)
    valid = np.isfinite(left) & np.isfinite(right) & (np.abs(right) > EPSILON)
    with np.errstate(all="ignore"):
        np.divide(left, right, out=result, where=valid)
    return _finite_or_nan(result)


def max2(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    return np.maximum(left, right)


def min2(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    return np.minimum(left, right)


def greater(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    result = np.full_like(left, np.nan)
    valid = np.isfinite(left) & np.isfinite(right)
    result[valid] = (left[valid] > right[valid]).astype(np.float64)
    return result


def less(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    result = np.full_like(left, np.nan)
    valid = np.isfinite(left) & np.isfinite(right)
    result[valid] = (left[valid] < right[valid]).astype(np.float64)
    return result


def signed_ratio(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    denominator = np.abs(left) + np.abs(right) + EPSILON
    with np.errstate(all="ignore"):
        return _finite_or_nan((left - right) / denominator)


def log_ratio(x: ArrayLike, y: ArrayLike) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    numerator = left + EPSILON
    denominator = right + EPSILON
    ratio = np.full_like(left, np.nan)
    valid_denominator = np.isfinite(denominator) & (denominator != 0.0)
    with np.errstate(all="ignore"):
        np.divide(numerator, denominator, out=ratio, where=valid_denominator)
    result = np.full_like(left, np.nan)
    valid_ratio = np.isfinite(ratio) & (ratio > 0.0)
    with np.errstate(all="ignore"):
        np.log(ratio, out=result, where=valid_ratio)
    return _finite_or_nan(result)


def _average_zero_based_rank(values: FloatMatrix) -> FloatMatrix:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _cross_sectional(
    x: ArrayLike,
    transform: Callable[[FloatMatrix], FloatMatrix],
) -> FloatMatrix:
    values = _matrix(x)
    result = np.full_like(values, np.nan)
    for date_index in range(values.shape[0]):
        valid = np.isfinite(values[date_index])
        if int(valid.sum()) < MIN_CROSS_SECTIONAL_COUNT:
            continue
        transformed = np.asarray(transform(values[date_index, valid]), dtype=np.float64)
        if transformed.shape != (int(valid.sum()),):
            raise RuntimeError("截面变换必须保持当日有效股票数量")
        transformed[~np.isfinite(transformed)] = np.nan
        result[date_index, valid] = transformed
    return result


def cs_rank(x: ArrayLike) -> FloatMatrix:
    return _cross_sectional(
        x,
        lambda values: (_average_zero_based_rank(values) + 0.5) / values.size,
    )


def cs_zscore(x: ArrayLike) -> FloatMatrix:
    def transform(values: FloatMatrix) -> FloatMatrix:
        standard_deviation = np.std(values, ddof=0)
        if standard_deviation <= EPSILON:
            return np.full_like(values, np.nan)
        return (values - np.mean(values)) / standard_deviation

    return _cross_sectional(x, transform)


def cs_demean(x: ArrayLike) -> FloatMatrix:
    return _cross_sectional(x, lambda values: values - np.mean(values))


def cs_scale(x: ArrayLike) -> FloatMatrix:
    def transform(values: FloatMatrix) -> FloatMatrix:
        denominator = np.sum(np.abs(values))
        if denominator <= EPSILON:
            return np.full_like(values, np.nan)
        return values / denominator

    return _cross_sectional(x, transform)


def cs_normalize(x: ArrayLike) -> FloatMatrix:
    def transform(values: FloatMatrix) -> FloatMatrix:
        minimum = np.min(values)
        value_range = np.max(values) - minimum
        return (values - minimum) / (value_range + EPSILON)

    return _cross_sectional(x, transform)


def _quantile_bounds(values: FloatMatrix) -> tuple[float, float]:
    lower, upper = np.quantile(
        values,
        (LOWER_QUANTILE, UPPER_QUANTILE),
        method="linear",
    )
    return float(lower), float(upper)


def cs_winsorize(x: ArrayLike) -> FloatMatrix:
    def transform(values: FloatMatrix) -> FloatMatrix:
        lower, upper = _quantile_bounds(values)
        return np.clip(values, lower, upper)

    return _cross_sectional(x, transform)


def cs_truncate(x: ArrayLike) -> FloatMatrix:
    def transform(values: FloatMatrix) -> FloatMatrix:
        lower, upper = _quantile_bounds(values)
        result = values.copy()
        result[(result < lower) | (result > upper)] = np.nan
        return result

    return _cross_sectional(x, transform)


def cs_quantile(x: ArrayLike) -> FloatMatrix:
    def transform(values: FloatMatrix) -> FloatMatrix:
        return _average_zero_based_rank(values) / (values.size - 1)

    return _cross_sectional(x, transform)


def cs_rank_gauss(x: ArrayLike) -> FloatMatrix:
    def transform(values: FloatMatrix) -> FloatMatrix:
        probability = (_average_zero_based_rank(values) + 0.5) / values.size
        return ndtri(probability)

    return _cross_sectional(x, transform)


def _window(window: int) -> int:
    if not isinstance(window, (int, np.integer)) or isinstance(window, (bool, np.bool_)):
        raise TypeError("window 必须是整数")
    window = int(window)
    if window not in WINDOWS:
        raise ValueError(f"window 必须属于 {WINDOWS}，实际为 {window}")
    return window


def _rolling_unary(
    x: ArrayLike,
    window: int,
    transform: Callable[[FloatMatrix], FloatMatrix],
) -> FloatMatrix:
    values = _matrix(x)
    window = _window(window)
    result = np.full_like(values, np.nan)
    for date_index in range(window - 1, values.shape[0]):
        sample = values[date_index - window + 1 : date_index + 1]
        valid_stocks = np.isfinite(sample).all(axis=0)
        if not valid_stocks.any():
            continue
        transformed = np.asarray(transform(sample[:, valid_stocks]), dtype=np.float64)
        expected_shape = (int(valid_stocks.sum()),)
        if transformed.shape != expected_shape:
            raise RuntimeError(
                f"时序一元变换必须返回形状 {expected_shape}，实际为 {transformed.shape}"
            )
        transformed[~np.isfinite(transformed)] = np.nan
        result[date_index, valid_stocks] = transformed
    return result


def _rolling_binary(
    x: ArrayLike,
    y: ArrayLike,
    window: int,
    transform: Callable[[FloatMatrix, FloatMatrix], FloatMatrix],
) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    window = _window(window)
    result = np.full_like(left, np.nan)
    for date_index in range(window - 1, left.shape[0]):
        left_sample = left[date_index - window + 1 : date_index + 1]
        right_sample = right[date_index - window + 1 : date_index + 1]
        valid_stocks = np.isfinite(left_sample).all(axis=0) & np.isfinite(
            right_sample
        ).all(axis=0)
        if not valid_stocks.any():
            continue
        transformed = np.asarray(
            transform(left_sample[:, valid_stocks], right_sample[:, valid_stocks]),
            dtype=np.float64,
        )
        expected_shape = (int(valid_stocks.sum()),)
        if transformed.shape != expected_shape:
            raise RuntimeError(
                f"时序二元变换必须返回形状 {expected_shape}，实际为 {transformed.shape}"
            )
        transformed[~np.isfinite(transformed)] = np.nan
        result[date_index, valid_stocks] = transformed
    return result


def ts_mean(x: ArrayLike, window: int) -> FloatMatrix:
    return _rolling_unary(x, window, lambda sample: np.mean(sample, axis=0))


def ts_std(x: ArrayLike, window: int) -> FloatMatrix:
    return _rolling_unary(x, window, lambda sample: np.std(sample, axis=0, ddof=0))


def ts_max(x: ArrayLike, window: int) -> FloatMatrix:
    return _rolling_unary(x, window, lambda sample: np.max(sample, axis=0))


def ts_min(x: ArrayLike, window: int) -> FloatMatrix:
    return _rolling_unary(x, window, lambda sample: np.min(sample, axis=0))


def ts_rank(x: ArrayLike, window: int) -> FloatMatrix:
    def transform(sample: FloatMatrix) -> FloatMatrix:
        output = np.empty(sample.shape[1], dtype=np.float64)
        for stock_index in range(sample.shape[1]):
            ranks = _average_zero_based_rank(sample[:, stock_index])
            output[stock_index] = (ranks[-1] + 0.5) / sample.shape[0]
        return output

    return _rolling_unary(x, window, transform)


def ts_delay(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    window = _window(window)
    result = np.full_like(values, np.nan)
    if values.shape[0] > window:
        result[window:] = values[:-window]
    return result


def ts_delta(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    window = _window(window)
    result = np.full_like(values, np.nan)
    if values.shape[0] <= window:
        return result
    current = values[window:]
    delayed = values[:-window]
    valid = np.isfinite(current) & np.isfinite(delayed)
    difference = np.full_like(current, np.nan)
    difference[valid] = current[valid] - delayed[valid]
    result[window:] = difference
    return _finite_or_nan(result)


def ts_sum(x: ArrayLike, window: int) -> FloatMatrix:
    return _rolling_unary(x, window, lambda sample: np.sum(sample, axis=0))


def ts_argmax(x: ArrayLike, window: int) -> FloatMatrix:
    def transform(sample: FloatMatrix) -> FloatMatrix:
        maximum = np.max(sample, axis=0)
        # 翻转后首次出现的位置对应原窗口中最近一次极值。
        return sample.shape[0] - 1 - np.argmax(sample[::-1] == maximum, axis=0)

    return _rolling_unary(x, window, transform)


def ts_argmin(x: ArrayLike, window: int) -> FloatMatrix:
    def transform(sample: FloatMatrix) -> FloatMatrix:
        minimum = np.min(sample, axis=0)
        return sample.shape[0] - 1 - np.argmax(sample[::-1] == minimum, axis=0)

    return _rolling_unary(x, window, transform)


def ts_wma(x: ArrayLike, window: int) -> FloatMatrix:
    window = _window(window)
    weights = np.arange(1, window + 1, dtype=np.float64)
    weights /= np.sum(weights)
    return _rolling_unary(x, window, lambda sample: weights @ sample)


def ts_ema(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    window = _window(window)
    result = np.full_like(values, np.nan)
    alpha = 2.0 / (window + 1.0)

    for stock_index in range(values.shape[1]):
        ema = np.nan
        consecutive_valid = 0
        for date_index in range(values.shape[0]):
            value = values[date_index, stock_index]
            if not np.isfinite(value):
                ema = np.nan
                consecutive_valid = 0
                continue
            if consecutive_valid == 0:
                ema = value
            else:
                ema = alpha * value + (1.0 - alpha) * ema
            consecutive_valid += 1
            if consecutive_valid >= window:
                result[date_index, stock_index] = ema
    return result


def _time_regression(sample: FloatMatrix) -> tuple[FloatMatrix, FloatMatrix]:
    time = np.arange(sample.shape[0], dtype=np.float64)
    centered_time = time - np.mean(time)
    denominator = np.sum(centered_time**2)
    centered_sample = sample - np.mean(sample, axis=0)
    slope = np.sum(centered_time[:, None] * centered_sample, axis=0) / denominator
    intercept = np.mean(sample, axis=0) - slope * np.mean(time)
    return intercept, slope


def ts_slope(x: ArrayLike, window: int) -> FloatMatrix:
    return _rolling_unary(x, window, lambda sample: _time_regression(sample)[1])


def ts_residual(x: ArrayLike, window: int) -> FloatMatrix:
    def transform(sample: FloatMatrix) -> FloatMatrix:
        intercept, slope = _time_regression(sample)
        fitted_current = intercept + slope * (sample.shape[0] - 1)
        return sample[-1] - fitted_current

    return _rolling_unary(x, window, transform)


def ts_zscore(x: ArrayLike, window: int) -> FloatMatrix:
    def transform(sample: FloatMatrix) -> FloatMatrix:
        mean = np.mean(sample, axis=0)
        standard_deviation = np.std(sample, axis=0, ddof=0)
        output = np.full(sample.shape[1], np.nan, dtype=np.float64)
        valid = standard_deviation > EPSILON
        output[valid] = (sample[-1, valid] - mean[valid]) / standard_deviation[valid]
        return output

    return _rolling_unary(x, window, transform)


def ts_position(x: ArrayLike, window: int) -> FloatMatrix:
    def transform(sample: FloatMatrix) -> FloatMatrix:
        minimum = np.min(sample, axis=0)
        value_range = np.max(sample, axis=0) - minimum
        return (sample[-1] - minimum) / (value_range + EPSILON)

    return _rolling_unary(x, window, transform)


def ts_range(x: ArrayLike, window: int) -> FloatMatrix:
    return _rolling_unary(
        x,
        window,
        lambda sample: np.max(sample, axis=0) - np.min(sample, axis=0),
    )


def ts_corr(x: ArrayLike, y: ArrayLike, window: int) -> FloatMatrix:
    def transform(left: FloatMatrix, right: FloatMatrix) -> FloatMatrix:
        left_centered = left - np.mean(left, axis=0)
        right_centered = right - np.mean(right, axis=0)
        left_scale = np.sqrt(np.mean(left_centered**2, axis=0))
        right_scale = np.sqrt(np.mean(right_centered**2, axis=0))
        output = np.full(left.shape[1], np.nan, dtype=np.float64)
        valid = (left_scale > EPSILON) & (right_scale > EPSILON)
        covariance = np.mean(left_centered * right_centered, axis=0)
        output[valid] = covariance[valid] / (left_scale[valid] * right_scale[valid])
        return output

    return _rolling_binary(x, y, window, transform)


def ts_cov(x: ArrayLike, y: ArrayLike, window: int) -> FloatMatrix:
    def transform(left: FloatMatrix, right: FloatMatrix) -> FloatMatrix:
        left_centered = left - np.mean(left, axis=0)
        right_centered = right - np.mean(right, axis=0)
        return np.mean(left_centered * right_centered, axis=0)

    return _rolling_binary(x, y, window, transform)


def _cross_regression(
    dependent: FloatMatrix,
    predictor: FloatMatrix,
) -> tuple[FloatMatrix, FloatMatrix]:
    predictor_mean = np.mean(predictor, axis=0)
    dependent_mean = np.mean(dependent, axis=0)
    predictor_centered = predictor - predictor_mean
    dependent_centered = dependent - dependent_mean
    predictor_variance = np.mean(predictor_centered**2, axis=0)
    beta = np.full(dependent.shape[1], np.nan, dtype=np.float64)
    valid = predictor_variance > EPSILON
    covariance = np.mean(dependent_centered * predictor_centered, axis=0)
    beta[valid] = covariance[valid] / predictor_variance[valid]
    intercept = dependent_mean - beta * predictor_mean
    return intercept, beta


def ts_beta(x: ArrayLike, y: ArrayLike, window: int) -> FloatMatrix:
    return _rolling_binary(x, y, window, lambda left, right: _cross_regression(left, right)[1])


def ts_orth(x: ArrayLike, y: ArrayLike, window: int) -> FloatMatrix:
    def transform(left: FloatMatrix, right: FloatMatrix) -> FloatMatrix:
        intercept, beta = _cross_regression(left, right)
        return left[-1] - intercept - beta * right[-1]

    return _rolling_binary(x, y, window, transform)


NON_TS_OPERATOR_FUNCTIONS = MappingProxyType(
    {
        "abs": abs,
        "neg": neg,
        "sign": sign,
        "log": log,
        "inv": inv,
        "sqrt": sqrt,
        "tanh": tanh,
        "relu": relu,
        "softsign": softsign,
        "signed_power2": signed_power2,
        "signed_power3": signed_power3,
        "signed_log1p": signed_log1p,
        "add": add,
        "sub": sub,
        "mul": mul,
        "div": div,
        "max2": max2,
        "min2": min2,
        "greater": greater,
        "less": less,
        "signed_ratio": signed_ratio,
        "log_ratio": log_ratio,
        "cs_rank": cs_rank,
        "cs_zscore": cs_zscore,
        "cs_demean": cs_demean,
        "cs_scale": cs_scale,
        "cs_normalize": cs_normalize,
        "cs_winsorize": cs_winsorize,
        "cs_truncate": cs_truncate,
        "cs_quantile": cs_quantile,
        "cs_rank_gauss": cs_rank_gauss,
    }
)

TS_OPERATOR_FUNCTIONS = MappingProxyType(
    {
        "ts_mean": ts_mean,
        "ts_std": ts_std,
        "ts_max": ts_max,
        "ts_min": ts_min,
        "ts_rank": ts_rank,
        "ts_delay": ts_delay,
        "ts_delta": ts_delta,
        "ts_sum": ts_sum,
        "ts_argmax": ts_argmax,
        "ts_argmin": ts_argmin,
        "ts_wma": ts_wma,
        "ts_ema": ts_ema,
        "ts_slope": ts_slope,
        "ts_residual": ts_residual,
        "ts_zscore": ts_zscore,
        "ts_position": ts_position,
        "ts_range": ts_range,
        "ts_corr": ts_corr,
        "ts_cov": ts_cov,
        "ts_beta": ts_beta,
        "ts_orth": ts_orth,
    }
)

ALL_OPERATOR_FUNCTIONS = MappingProxyType(
    {**NON_TS_OPERATOR_FUNCTIONS, **TS_OPERATOR_FUNCTIONS}
)


def _validate_registry_coverage() -> None:
    expected_non_ts = {
        operator.name
        for operator in (
            *UNARY_OPERATORS,
            *BINARY_OPERATORS,
            *CROSS_SECTIONAL_OPERATORS,
        )
    }
    actual_non_ts = set(NON_TS_OPERATOR_FUNCTIONS)
    if actual_non_ts != expected_non_ts:
        raise RuntimeError(
            "非时序算子实现与文法注册表不一致："
            f"missing={sorted(expected_non_ts - actual_non_ts)}, "
            f"extra={sorted(actual_non_ts - expected_non_ts)}"
        )
    expected_ts = {
        operator.name for operator in (*TS_UNARY_OPERATORS, *TS_BINARY_OPERATORS)
    }
    actual_ts = set(TS_OPERATOR_FUNCTIONS)
    if actual_ts != expected_ts:
        raise RuntimeError(
            f"时序算子实现与文法注册表不一致：missing={sorted(expected_ts - actual_ts)}, "
            f"extra={sorted(actual_ts - expected_ts)}"
        )
    if len(ALL_OPERATOR_FUNCTIONS) != 52:
        raise RuntimeError(f"全部非叶子算子实现应为 52，实际为 {len(ALL_OPERATOR_FUNCTIONS)}")


_validate_registry_coverage()


__all__ = [
    "EPSILON",
    "ALL_OPERATOR_FUNCTIONS",
    "LOWER_QUANTILE",
    "MIN_CROSS_SECTIONAL_COUNT",
    "NON_TS_OPERATOR_FUNCTIONS",
    "TS_OPERATOR_FUNCTIONS",
    "UPPER_QUANTILE",
    *ALL_OPERATOR_FUNCTIONS.keys(),
]
