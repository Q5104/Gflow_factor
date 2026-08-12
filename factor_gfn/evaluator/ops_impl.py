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

from .numba_kernels import (
    BINARY_BETA,
    BINARY_CORR,
    BINARY_COV,
    BINARY_ORTH,
    ROLLING_ARGMAX,
    ROLLING_ARGMIN,
    ROLLING_MAX,
    ROLLING_MIN,
    ROLLING_POSITION,
    ROLLING_RANGE,
    ROLLING_RANK,
    UNARY_MEAN,
    UNARY_STD,
    UNARY_ZSCORE,
    WEIGHTED_RESIDUAL,
    WEIGHTED_SLOPE,
    WEIGHTED_WMA,
    cross_sectional_average_rank_kernel,
    ema_kernel,
    rolling_binary_moment_kernel,
    rolling_sum_kernel,
    rolling_unary_moment_kernel,
    rolling_weighted_kernel,
    rolling_window_kernel,
)


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
    if np.isinf(array).any():
        result = array.copy()
        result[~np.isfinite(result)] = np.nan
        return result
    return array


def _binary_matrices(x: ArrayLike, y: ArrayLike) -> tuple[FloatMatrix, FloatMatrix]:
    left = _matrix(x, "左输入")
    right = _matrix(y, "右输入")
    if left.shape != right.shape:
        raise ValueError(f"二元算子输入形状必须一致：{left.shape} != {right.shape}")
    return left, right


def _finite_or_nan(values: ArrayLike) -> FloatMatrix:
    array = np.asarray(values, dtype=np.float64)
    result = array if array.flags.owndata and array.flags.writeable else array.copy()
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
    ranks, counts = cross_sectional_average_rank_kernel(_matrix(x))
    denominator = np.maximum(counts, 1)
    return (ranks + 0.5) / denominator[:, None]


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
    ranks, counts = cross_sectional_average_rank_kernel(_matrix(x))
    denominator = np.maximum(counts - 1, 1)
    return ranks / denominator[:, None]


def cs_rank_gauss(x: ArrayLike) -> FloatMatrix:
    ranks, counts = cross_sectional_average_rank_kernel(_matrix(x))
    probability = (ranks + 0.5) / np.maximum(counts, 1)[:, None]
    return ndtri(probability)


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


def _rolling_sum_complete(values: FloatMatrix, window: int) -> FloatMatrix:
    """完整窗口滑动和；只保留股票轴状态，避免逐窗口切片。"""

    result = np.full_like(values, np.nan)
    running_sum = np.zeros(values.shape[1], dtype=np.float64)
    missing_count = np.zeros(values.shape[1], dtype=np.int32)
    for date_index in range(values.shape[0]):
        current = values[date_index]
        current_valid = np.isfinite(current)
        np.add(running_sum, current, out=running_sum, where=current_valid)
        missing_count += ~current_valid
        if date_index >= window:
            expired = values[date_index - window]
            expired_valid = np.isfinite(expired)
            np.subtract(running_sum, expired, out=running_sum, where=expired_valid)
            missing_count -= ~expired_valid
        if date_index >= window - 1:
            complete = missing_count == 0
            result[date_index, complete] = running_sum[complete]
    return result


def _rolling_unary_moment(
    values: FloatMatrix,
    window: int,
    mode: str,
) -> FloatMatrix:
    """用平移后的滑动一、二阶矩计算 mean/std/zscore。"""

    result = np.full_like(values, np.nan)
    origin = np.zeros(values.shape[1], dtype=np.float64)
    has_origin = np.zeros(values.shape[1], dtype=bool)
    running_sum = np.zeros(values.shape[1], dtype=np.float64)
    running_square_sum = np.zeros(values.shape[1], dtype=np.float64)
    missing_count = np.zeros(values.shape[1], dtype=np.int32)
    centered = np.zeros(values.shape[1], dtype=np.float64)
    for date_index in range(values.shape[0]):
        current = values[date_index]
        current_valid = np.isfinite(current)
        new_origin = current_valid & ~has_origin
        origin[new_origin] = current[new_origin]
        has_origin |= current_valid
        centered.fill(0.0)
        np.subtract(current, origin, out=centered, where=current_valid)
        np.add(running_sum, centered, out=running_sum)
        np.add(running_square_sum, centered * centered, out=running_square_sum)
        missing_count += ~current_valid
        if date_index >= window:
            expired = values[date_index - window]
            expired_valid = np.isfinite(expired)
            centered.fill(0.0)
            np.subtract(expired, origin, out=centered, where=expired_valid)
            np.subtract(running_sum, centered, out=running_sum)
            np.subtract(
                running_square_sum,
                centered * centered,
                out=running_square_sum,
            )
            missing_count -= ~expired_valid
        if date_index < window - 1:
            continue
        complete = missing_count == 0
        if mode == "mean":
            result[date_index, complete] = (
                origin[complete] + running_sum[complete] / window
            )
            continue
        variance = (
            running_square_sum[complete]
            - running_sum[complete] * running_sum[complete] / window
        ) / window
        variance = np.maximum(variance, 0.0)
        standard_deviation = np.sqrt(variance)
        if mode == "std":
            result[date_index, complete] = standard_deviation
        elif mode == "zscore":
            eligible = standard_deviation > EPSILON
            complete_indices = np.flatnonzero(complete)
            selected = complete_indices[eligible]
            result[date_index, selected] = (
                values[date_index, selected]
                - origin[selected]
                - running_sum[selected] / window
            ) / standard_deviation[eligible]
        else:
            raise ValueError(f"未知滑动矩模式：{mode}")
    return result


def _rolling_weighted_statistic(
    values: FloatMatrix,
    window: int,
    mode: str,
) -> FloatMatrix:
    """用滑动和及线性加权和计算 WMA、趋势斜率和当前残差。"""

    result = np.full_like(values, np.nan)
    origin = np.zeros(values.shape[1], dtype=np.float64)
    has_origin = np.zeros(values.shape[1], dtype=bool)
    running_sum = np.zeros(values.shape[1], dtype=np.float64)
    weighted_sum = np.zeros(values.shape[1], dtype=np.float64)
    missing_count = np.zeros(values.shape[1], dtype=np.int32)
    centered = np.zeros(values.shape[1], dtype=np.float64)
    denominator = window * (window**2 - 1) / 12.0
    mean_time = (window - 1) / 2.0
    weight_total = window * (window + 1) / 2.0
    for date_index in range(values.shape[0]):
        current = values[date_index]
        current_valid = np.isfinite(current)
        new_origin = current_valid & ~has_origin
        origin[new_origin] = current[new_origin]
        has_origin |= current_valid
        centered.fill(0.0)
        np.subtract(current, origin, out=centered, where=current_valid)
        if date_index < window:
            running_sum += centered
            weighted_sum += (date_index + 1) * centered
            missing_count += ~current_valid
        else:
            previous_sum = running_sum.copy()
            expired = values[date_index - window]
            expired_valid = np.isfinite(expired)
            expired_centered = np.zeros_like(centered)
            np.subtract(expired, origin, out=expired_centered, where=expired_valid)
            running_sum += centered - expired_centered
            weighted_sum += window * centered - previous_sum
            missing_count += (~current_valid).astype(np.int32)
            missing_count -= (~expired_valid).astype(np.int32)
        if date_index < window - 1:
            continue
        complete = missing_count == 0
        if mode == "wma":
            result[date_index, complete] = (
                origin[complete]
                + weighted_sum[complete] / weight_total
            )
            continue
        zero_based_weighted = weighted_sum[complete] - running_sum[complete]
        slope = (
            zero_based_weighted - mean_time * running_sum[complete]
        ) / denominator
        if mode == "slope":
            result[date_index, complete] = slope
        elif mode == "residual":
            complete_indices = np.flatnonzero(complete)
            mean = origin[complete] + running_sum[complete] / window
            result[date_index, complete] = (
                values[date_index, complete_indices]
                - mean
                - slope * (window - 1 - mean_time)
            )
        else:
            raise ValueError(f"未知线性滑动统计模式：{mode}")
    return result


def _rolling_binary_moment(
    left: FloatMatrix,
    right: FloatMatrix,
    window: int,
    mode: str,
) -> FloatMatrix:
    """用平移后的滑动联合矩计算 cov/corr/beta/orth。"""

    result = np.full_like(left, np.nan)
    left_origin = np.zeros(left.shape[1], dtype=np.float64)
    right_origin = np.zeros(right.shape[1], dtype=np.float64)
    has_left_origin = np.zeros(left.shape[1], dtype=bool)
    has_right_origin = np.zeros(right.shape[1], dtype=bool)
    sum_left = np.zeros(left.shape[1], dtype=np.float64)
    sum_right = np.zeros(left.shape[1], dtype=np.float64)
    sum_left_square = np.zeros(left.shape[1], dtype=np.float64)
    sum_right_square = np.zeros(left.shape[1], dtype=np.float64)
    sum_product = np.zeros(left.shape[1], dtype=np.float64)
    missing_count = np.zeros(left.shape[1], dtype=np.int32)
    left_centered = np.zeros(left.shape[1], dtype=np.float64)
    right_centered = np.zeros(left.shape[1], dtype=np.float64)

    def update(date_index: int, sign_: float) -> NDArray[np.bool_]:
        left_row = left[date_index]
        right_row = right[date_index]
        left_valid = np.isfinite(left_row)
        right_valid = np.isfinite(right_row)
        left_new_origin = left_valid & ~has_left_origin
        right_new_origin = right_valid & ~has_right_origin
        left_origin[left_new_origin] = left_row[left_new_origin]
        right_origin[right_new_origin] = right_row[right_new_origin]
        has_left_origin[:] |= left_valid
        has_right_origin[:] |= right_valid
        joint_valid = left_valid & right_valid
        left_centered.fill(0.0)
        right_centered.fill(0.0)
        np.subtract(left_row, left_origin, out=left_centered, where=joint_valid)
        np.subtract(right_row, right_origin, out=right_centered, where=joint_valid)
        sum_left[:] += sign_ * left_centered
        sum_right[:] += sign_ * right_centered
        sum_left_square[:] += sign_ * left_centered * left_centered
        sum_right_square[:] += sign_ * right_centered * right_centered
        sum_product[:] += sign_ * left_centered * right_centered
        return joint_valid

    for date_index in range(left.shape[0]):
        current_valid = update(date_index, 1.0)
        missing_count += ~current_valid
        if date_index >= window:
            expired_valid = update(date_index - window, -1.0)
            missing_count -= ~expired_valid
        if date_index < window - 1:
            continue
        complete = missing_count == 0
        complete_indices = np.flatnonzero(complete)
        covariance = (
            sum_product[complete]
            - sum_left[complete] * sum_right[complete] / window
        ) / window
        if mode == "cov":
            result[date_index, complete] = covariance
            continue
        left_variance = (
            sum_left_square[complete]
            - sum_left[complete] * sum_left[complete] / window
        ) / window
        right_variance = (
            sum_right_square[complete]
            - sum_right[complete] * sum_right[complete] / window
        ) / window
        left_variance = np.maximum(left_variance, 0.0)
        right_variance = np.maximum(right_variance, 0.0)
        if mode == "corr":
            eligible = (np.sqrt(left_variance) > EPSILON) & (
                np.sqrt(right_variance) > EPSILON
            )
            selected = complete_indices[eligible]
            result[date_index, selected] = covariance[eligible] / np.sqrt(
                left_variance[eligible] * right_variance[eligible]
            )
            continue
        eligible = right_variance > EPSILON
        selected = complete_indices[eligible]
        beta = covariance[eligible] / right_variance[eligible]
        if mode == "beta":
            result[date_index, selected] = beta
        elif mode == "orth":
            left_mean = left_origin[selected] + sum_left[selected] / window
            right_mean = right_origin[selected] + sum_right[selected] / window
            result[date_index, selected] = (
                left[date_index, selected]
                - left_mean
                - beta * (right[date_index, selected] - right_mean)
            )
        else:
            raise ValueError(f"未知二元滑动矩模式：{mode}")
    return result


def ts_mean(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_unary_moment_kernel(values, _window(window), UNARY_MEAN, EPSILON)


def ts_std(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_unary_moment_kernel(values, _window(window), UNARY_STD, EPSILON)


def ts_max(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_window_kernel(values, _window(window), ROLLING_MAX, EPSILON)


def ts_min(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_window_kernel(values, _window(window), ROLLING_MIN, EPSILON)


def ts_rank(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_window_kernel(values, _window(window), ROLLING_RANK, EPSILON)


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
    values = _matrix(x)
    return rolling_sum_kernel(values, _window(window))


def ts_argmax(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_window_kernel(values, _window(window), ROLLING_ARGMAX, EPSILON)


def ts_argmin(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_window_kernel(values, _window(window), ROLLING_ARGMIN, EPSILON)


def ts_wma(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_weighted_kernel(values, _window(window), WEIGHTED_WMA)


def ts_ema(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return ema_kernel(values, _window(window))


def _time_regression(sample: FloatMatrix) -> tuple[FloatMatrix, FloatMatrix]:
    time = np.arange(sample.shape[0], dtype=np.float64)
    centered_time = time - np.mean(time)
    denominator = np.sum(centered_time**2)
    centered_sample = sample - np.mean(sample, axis=0)
    slope = np.sum(centered_time[:, None] * centered_sample, axis=0) / denominator
    intercept = np.mean(sample, axis=0) - slope * np.mean(time)
    return intercept, slope


def ts_slope(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_weighted_kernel(values, _window(window), WEIGHTED_SLOPE)


def ts_residual(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_weighted_kernel(values, _window(window), WEIGHTED_RESIDUAL)


def ts_zscore(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_unary_moment_kernel(values, _window(window), UNARY_ZSCORE, EPSILON)


def ts_position(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_window_kernel(values, _window(window), ROLLING_POSITION, EPSILON)


def ts_range(x: ArrayLike, window: int) -> FloatMatrix:
    values = _matrix(x)
    return rolling_window_kernel(values, _window(window), ROLLING_RANGE, EPSILON)


def ts_corr(x: ArrayLike, y: ArrayLike, window: int) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    return rolling_binary_moment_kernel(
        left, right, _window(window), BINARY_CORR, EPSILON
    )


def ts_cov(x: ArrayLike, y: ArrayLike, window: int) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    return rolling_binary_moment_kernel(
        left, right, _window(window), BINARY_COV, EPSILON
    )


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
    left, right = _binary_matrices(x, y)
    return rolling_binary_moment_kernel(
        left, right, _window(window), BINARY_BETA, EPSILON
    )


def ts_orth(x: ArrayLike, y: ArrayLike, window: int) -> FloatMatrix:
    left, right = _binary_matrices(x, y)
    return rolling_binary_moment_kernel(
        left, right, _window(window), BINARY_ORTH, EPSILON
    )


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
