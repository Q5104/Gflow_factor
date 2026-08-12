"""循环型时序算子的 Numba 内核。"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import numpy.typing as npt
from numba import njit


NUMBA_KERNEL_SCHEMA = "factor_gfn.numba_cpu_loops.v2"

ROLLING_MAX = 0
ROLLING_MIN = 1
ROLLING_RANK = 2
ROLLING_ARGMAX = 3
ROLLING_ARGMIN = 4
ROLLING_POSITION = 5
ROLLING_RANGE = 6

UNARY_MEAN = 0
UNARY_STD = 1
UNARY_ZSCORE = 2

WEIGHTED_WMA = 0
WEIGHTED_SLOPE = 1
WEIGHTED_RESIDUAL = 2

BINARY_COV = 0
BINARY_CORR = 1
BINARY_BETA = 2
BINARY_ORTH = 3


@njit(cache=True)
def cross_sectional_average_rank_kernel(
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Average zero-based ranks for every finite row, preserving NaNs."""

    date_count, stock_count = values.shape
    ranks = np.full((date_count, stock_count), np.nan, dtype=np.float64)
    counts = np.zeros(date_count, dtype=np.int64)
    finite_values = np.empty(stock_count, dtype=np.float64)
    finite_indices = np.empty(stock_count, dtype=np.int64)
    for date_index in range(date_count):
        count = 0
        for stock_index in range(stock_count):
            value = values[date_index, stock_index]
            if np.isfinite(value):
                finite_values[count] = value
                finite_indices[count] = stock_index
                count += 1
        counts[date_index] = count
        if count < 2:
            continue
        order = np.argsort(finite_values[:count])
        start = 0
        while start < count:
            stop = start + 1
            sorted_value = finite_values[order[start]]
            while stop < count and finite_values[order[stop]] == sorted_value:
                stop += 1
            average_rank = (start + stop - 1) / 2.0
            for position in range(start, stop):
                original = finite_indices[order[position]]
                ranks[date_index, original] = average_rank
            start = stop
    return ranks, counts


@njit(cache=True)
def _average_ranks_from_order(
    values: npt.NDArray[np.float64],
    order: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    count = values.size
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        stop = start + 1
        sorted_value = values[order[start]]
        while stop < count and values[order[stop]] == sorted_value:
            stop += 1
        average_rank = (start + stop - 1) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    return ranks


@njit(cache=True)
def rank_ic_values_kernel(
    factor: npt.NDArray[np.float64],
    forward_returns: npt.NDArray[np.float64],
    min_count: int,
) -> npt.NDArray[np.float64]:
    """Compute per-row Spearman RankIC without materializing rank panels."""

    date_count, stock_count = factor.shape
    ic_values = np.full(date_count, np.nan, dtype=np.float64)
    factor_values = np.empty(stock_count, dtype=np.float64)
    return_values = np.empty(stock_count, dtype=np.float64)
    for date_index in range(date_count):
        count = 0
        for stock_index in range(stock_count):
            factor_value = factor[date_index, stock_index]
            return_value = forward_returns[date_index, stock_index]
            if np.isfinite(factor_value) and np.isfinite(return_value):
                factor_values[count] = factor_value
                return_values[count] = return_value
                count += 1
        if count < min_count:
            continue
        scores = factor_values[:count]
        returns = return_values[:count]
        factor_order = np.argsort(scores, kind="mergesort")
        return_order = np.argsort(returns, kind="mergesort")
        factor_ranks = _average_ranks_from_order(scores, factor_order)
        return_ranks = _average_ranks_from_order(returns, return_order)
        rank_mean = (count - 1.0) / 2.0
        covariance_sum = 0.0
        factor_square_sum = 0.0
        return_square_sum = 0.0
        for index in range(count):
            factor_centered = factor_ranks[index] - rank_mean
            return_centered = return_ranks[index] - rank_mean
            covariance_sum += factor_centered * return_centered
            factor_square_sum += factor_centered * factor_centered
            return_square_sum += return_centered * return_centered
        denominator = np.sqrt(factor_square_sum * return_square_sum)
        if denominator > 0.0:
            ic_values[date_index] = covariance_sum / denominator
    return ic_values


@njit(cache=True)
def cleaned_portfolio_series_kernel(
    factor: npt.NDArray[np.float64],
    forward_returns: npt.NDArray[np.float64],
    direction: int,
    min_count: int,
    long_quantile: float,
    epsilon: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute directed long excess and raw LS while sharing stable order."""

    date_count, stock_count = factor.shape
    long_excess = np.full(date_count, np.nan, dtype=np.float64)
    long_short = np.full(date_count, np.nan, dtype=np.float64)
    factor_values = np.empty(stock_count, dtype=np.float64)
    return_values = np.empty(stock_count, dtype=np.float64)
    for date_index in range(date_count):
        count = 0
        for stock_index in range(stock_count):
            factor_value = factor[date_index, stock_index]
            return_value = forward_returns[date_index, stock_index]
            if np.isfinite(factor_value) and np.isfinite(return_value):
                factor_values[count] = factor_value
                return_values[count] = return_value
                count += 1
        if count < min_count:
            continue
        scores = factor_values[:count]
        returns = return_values[:count]
        factor_order = np.argsort(scores, kind="mergesort")
        minimum = scores[factor_order[0]]
        maximum = scores[factor_order[count - 1]]
        if maximum - minimum <= epsilon:
            continue
        selected_count = max(1, int(np.ceil(count * long_quantile)))
        benchmark_sum = 0.0
        for index in range(count):
            benchmark_sum += returns[index]
        benchmark_mean = benchmark_sum / count
        directed_order = factor_order
        if direction == -1:
            directed_order = np.argsort(-scores, kind="mergesort")
        long_sum = 0.0
        for position in range(count - selected_count, count):
            long_sum += returns[directed_order[position]]
        long_excess[date_index] = long_sum / selected_count - benchmark_mean

        leg_count = min(selected_count, count // 2)
        top_sum = 0.0
        bottom_sum = 0.0
        for position in range(leg_count):
            bottom_sum += returns[factor_order[position]]
            top_sum += returns[factor_order[count - leg_count + position]]
        long_short[date_index] = (top_sum - bottom_sum) / leg_count
    return long_excess, long_short


@njit(cache=True)
def rolling_sum_kernel(
    values: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]:
    date_count, stock_count = values.shape
    result = np.full((date_count, stock_count), np.nan, dtype=np.float64)
    for stock_index in range(stock_count):
        running_sum = 0.0
        missing_count = 0
        for date_index in range(date_count):
            current = values[date_index, stock_index]
            if np.isfinite(current):
                running_sum += current
            else:
                missing_count += 1
            if date_index >= window:
                expired = values[date_index - window, stock_index]
                if np.isfinite(expired):
                    running_sum -= expired
                else:
                    missing_count -= 1
            if date_index >= window - 1 and missing_count == 0:
                result[date_index, stock_index] = running_sum
    return result


@njit(cache=True)
def rolling_unary_moment_kernel(
    values: npt.NDArray[np.float64],
    window: int,
    mode: int,
    epsilon: float,
) -> npt.NDArray[np.float64]:
    date_count, stock_count = values.shape
    result = np.full((date_count, stock_count), np.nan, dtype=np.float64)
    for stock_index in range(stock_count):
        origin = 0.0
        has_origin = False
        running_sum = 0.0
        running_square_sum = 0.0
        missing_count = 0
        for date_index in range(date_count):
            current = values[date_index, stock_index]
            current_valid = np.isfinite(current)
            if current_valid and not has_origin:
                origin = current
                has_origin = True
            if current_valid:
                centered = current - origin
                running_sum += centered
                running_square_sum += centered * centered
            else:
                missing_count += 1
            if date_index >= window:
                expired = values[date_index - window, stock_index]
                if np.isfinite(expired):
                    centered = expired - origin
                    running_sum -= centered
                    running_square_sum -= centered * centered
                else:
                    missing_count -= 1
            if date_index < window - 1 or missing_count != 0:
                continue
            mean = origin + running_sum / window
            if mode == UNARY_MEAN:
                result[date_index, stock_index] = mean
                continue
            variance = (running_square_sum - running_sum * running_sum / window) / window
            if variance < 0.0:
                variance = 0.0
            standard_deviation = np.sqrt(variance)
            if mode == UNARY_STD:
                result[date_index, stock_index] = standard_deviation
            elif mode == UNARY_ZSCORE and standard_deviation > epsilon:
                result[date_index, stock_index] = (current - mean) / standard_deviation
    return result


@njit(cache=True)
def rolling_weighted_kernel(
    values: npt.NDArray[np.float64],
    window: int,
    mode: int,
) -> npt.NDArray[np.float64]:
    date_count, stock_count = values.shape
    result = np.full((date_count, stock_count), np.nan, dtype=np.float64)
    denominator = window * (window * window - 1.0) / 12.0
    mean_time = (window - 1.0) / 2.0
    weight_total = window * (window + 1.0) / 2.0
    for stock_index in range(stock_count):
        origin = 0.0
        has_origin = False
        running_sum = 0.0
        weighted_sum = 0.0
        missing_count = 0
        for date_index in range(date_count):
            current = values[date_index, stock_index]
            current_valid = np.isfinite(current)
            if current_valid and not has_origin:
                origin = current
                has_origin = True
            centered = current - origin if current_valid else 0.0
            if date_index < window:
                running_sum += centered
                weighted_sum += (date_index + 1.0) * centered
                if not current_valid:
                    missing_count += 1
            else:
                previous_sum = running_sum
                expired = values[date_index - window, stock_index]
                expired_valid = np.isfinite(expired)
                expired_centered = expired - origin if expired_valid else 0.0
                running_sum += centered - expired_centered
                weighted_sum += window * centered - previous_sum
                if not current_valid:
                    missing_count += 1
                if not expired_valid:
                    missing_count -= 1
            if date_index < window - 1 or missing_count != 0:
                continue
            if mode == WEIGHTED_WMA:
                result[date_index, stock_index] = origin + weighted_sum / weight_total
                continue
            zero_based_weighted = weighted_sum - running_sum
            slope = (zero_based_weighted - mean_time * running_sum) / denominator
            if mode == WEIGHTED_SLOPE:
                result[date_index, stock_index] = slope
            elif mode == WEIGHTED_RESIDUAL:
                mean = origin + running_sum / window
                result[date_index, stock_index] = (
                    current - mean - slope * (window - 1.0 - mean_time)
                )
    return result


@njit(cache=True)
def rolling_binary_moment_kernel(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    window: int,
    mode: int,
    epsilon: float,
) -> npt.NDArray[np.float64]:
    date_count, stock_count = left.shape
    result = np.full((date_count, stock_count), np.nan, dtype=np.float64)
    for stock_index in range(stock_count):
        left_origin = 0.0
        right_origin = 0.0
        has_left_origin = False
        has_right_origin = False
        sum_left = 0.0
        sum_right = 0.0
        sum_left_square = 0.0
        sum_right_square = 0.0
        sum_product = 0.0
        missing_count = 0
        for date_index in range(date_count):
            left_value = left[date_index, stock_index]
            right_value = right[date_index, stock_index]
            left_valid = np.isfinite(left_value)
            right_valid = np.isfinite(right_value)
            if left_valid and not has_left_origin:
                left_origin = left_value
                has_left_origin = True
            if right_valid and not has_right_origin:
                right_origin = right_value
                has_right_origin = True
            joint_valid = left_valid and right_valid
            if joint_valid:
                left_centered = left_value - left_origin
                right_centered = right_value - right_origin
                sum_left += left_centered
                sum_right += right_centered
                sum_left_square += left_centered * left_centered
                sum_right_square += right_centered * right_centered
                sum_product += left_centered * right_centered
            else:
                missing_count += 1
            if date_index >= window:
                expired_left = left[date_index - window, stock_index]
                expired_right = right[date_index - window, stock_index]
                expired_valid = np.isfinite(expired_left) and np.isfinite(expired_right)
                if expired_valid:
                    left_centered = expired_left - left_origin
                    right_centered = expired_right - right_origin
                    sum_left -= left_centered
                    sum_right -= right_centered
                    sum_left_square -= left_centered * left_centered
                    sum_right_square -= right_centered * right_centered
                    sum_product -= left_centered * right_centered
                else:
                    missing_count -= 1
            if date_index < window - 1 or missing_count != 0:
                continue
            covariance = (sum_product - sum_left * sum_right / window) / window
            if mode == BINARY_COV:
                result[date_index, stock_index] = covariance
                continue
            left_variance = (
                sum_left_square - sum_left * sum_left / window
            ) / window
            right_variance = (
                sum_right_square - sum_right * sum_right / window
            ) / window
            if left_variance < 0.0:
                left_variance = 0.0
            if right_variance < 0.0:
                right_variance = 0.0
            if mode == BINARY_CORR:
                left_std = np.sqrt(left_variance)
                right_std = np.sqrt(right_variance)
                if left_std > epsilon and right_std > epsilon:
                    result[date_index, stock_index] = covariance / (left_std * right_std)
                continue
            if right_variance <= epsilon:
                continue
            beta = covariance / right_variance
            if mode == BINARY_BETA:
                result[date_index, stock_index] = beta
            elif mode == BINARY_ORTH:
                left_mean = left_origin + sum_left / window
                right_mean = right_origin + sum_right / window
                result[date_index, stock_index] = (
                    left_value - left_mean - beta * (right_value - right_mean)
                )
    return result


@njit(cache=True)
def rolling_window_kernel(
    values: npt.NDArray[np.float64],
    window: int,
    mode: int,
    epsilon: float,
) -> npt.NDArray[np.float64]:
    """完整窗口 Rank 或单调队列极值；并列极值选择最近位置。"""

    date_count, stock_count = values.shape
    result = np.full((date_count, stock_count), np.nan, dtype=np.float64)
    if mode == ROLLING_RANK:
        for date_index in range(window - 1, date_count):
            start = date_index - window + 1
            for stock_index in range(stock_count):
                current = values[date_index, stock_index]
                if not np.isfinite(current):
                    continue
                less_count = 0
                equal_count = 0
                complete = True
                for offset in range(window):
                    value = values[start + offset, stock_index]
                    if not np.isfinite(value):
                        complete = False
                        break
                    if value < current:
                        less_count += 1
                    elif value == current:
                        equal_count += 1
                if complete:
                    result[date_index, stock_index] = (
                        less_count + 0.5 * equal_count
                    ) / window
        return result

    maximum_queue = np.empty((stock_count, window), dtype=np.int64)
    minimum_queue = np.empty((stock_count, window), dtype=np.int64)
    maximum_head = np.zeros(stock_count, dtype=np.int64)
    minimum_head = np.zeros(stock_count, dtype=np.int64)
    maximum_size = np.zeros(stock_count, dtype=np.int64)
    minimum_size = np.zeros(stock_count, dtype=np.int64)
    consecutive_valid = np.zeros(stock_count, dtype=np.int64)
    for date_index in range(date_count):
        for stock_index in range(stock_count):
            value = values[date_index, stock_index]
            if not np.isfinite(value):
                maximum_head[stock_index] = 0
                minimum_head[stock_index] = 0
                maximum_size[stock_index] = 0
                minimum_size[stock_index] = 0
                consecutive_valid[stock_index] = 0
                continue
            consecutive_valid[stock_index] += 1

            expired_boundary = date_index - window
            while maximum_size[stock_index] > 0:
                maximum_index = maximum_queue[
                    stock_index, maximum_head[stock_index]
                ]
                if maximum_index > expired_boundary:
                    break
                maximum_head[stock_index] = (
                    maximum_head[stock_index] + 1
                ) % window
                maximum_size[stock_index] -= 1
            while minimum_size[stock_index] > 0:
                minimum_index = minimum_queue[
                    stock_index, minimum_head[stock_index]
                ]
                if minimum_index > expired_boundary:
                    break
                minimum_head[stock_index] = (
                    minimum_head[stock_index] + 1
                ) % window
                minimum_size[stock_index] -= 1

            while maximum_size[stock_index] > 0:
                back = (
                    maximum_head[stock_index] + maximum_size[stock_index] - 1
                ) % window
                previous_index = maximum_queue[stock_index, back]
                if value < values[previous_index, stock_index]:
                    break
                maximum_size[stock_index] -= 1
            maximum_tail = (
                maximum_head[stock_index] + maximum_size[stock_index]
            ) % window
            maximum_queue[stock_index, maximum_tail] = date_index
            maximum_size[stock_index] += 1

            while minimum_size[stock_index] > 0:
                back = (
                    minimum_head[stock_index] + minimum_size[stock_index] - 1
                ) % window
                previous_index = minimum_queue[stock_index, back]
                if value > values[previous_index, stock_index]:
                    break
                minimum_size[stock_index] -= 1
            minimum_tail = (
                minimum_head[stock_index] + minimum_size[stock_index]
            ) % window
            minimum_queue[stock_index, minimum_tail] = date_index
            minimum_size[stock_index] += 1
            if consecutive_valid[stock_index] < window:
                continue

            maximum_index = maximum_queue[stock_index, maximum_head[stock_index]]
            minimum_index = minimum_queue[stock_index, minimum_head[stock_index]]
            maximum = values[maximum_index, stock_index]
            minimum = values[minimum_index, stock_index]
            start = date_index - window + 1
            if mode == ROLLING_MAX:
                result[date_index, stock_index] = maximum
            elif mode == ROLLING_MIN:
                result[date_index, stock_index] = minimum
            elif mode == ROLLING_ARGMAX:
                result[date_index, stock_index] = maximum_index - start
            elif mode == ROLLING_ARGMIN:
                result[date_index, stock_index] = minimum_index - start
            elif mode == ROLLING_POSITION:
                result[date_index, stock_index] = (
                    value - minimum
                ) / (maximum - minimum + epsilon)
            elif mode == ROLLING_RANGE:
                result[date_index, stock_index] = maximum - minimum
    return result


@njit(cache=True)
def ema_kernel(
    values: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]:
    """``adjust=False`` EMA；缺失后重置并重新等待完整连续窗口。"""

    date_count, stock_count = values.shape
    result = np.full((date_count, stock_count), np.nan, dtype=np.float64)
    alpha = 2.0 / (window + 1.0)
    for stock_index in range(stock_count):
        ema = np.nan
        consecutive_valid = 0
        for date_index in range(date_count):
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


def warm_numba_kernels() -> float:
    """预热两类签名，返回本次调用耗时；不改变任何研究数据。"""

    started = perf_counter()
    sample = np.arange(12, dtype=np.float64).reshape(6, 2)
    rolling_window_kernel(sample, 5, ROLLING_RANK, 1e-12)
    ema_kernel(sample, 5)
    cross_sectional_average_rank_kernel(sample)
    rank_ic_values_kernel(sample, sample + 1.0, 2)
    cleaned_portfolio_series_kernel(sample, sample + 1.0, 1, 2, 0.1, 1e-12)
    rolling_sum_kernel(sample, 5)
    rolling_unary_moment_kernel(sample, 5, UNARY_STD, 1e-12)
    rolling_weighted_kernel(sample, 5, WEIGHTED_SLOPE)
    rolling_binary_moment_kernel(sample, sample + 1.0, 5, BINARY_CORR, 1e-12)
    return float(perf_counter() - started)


__all__ = [
    "NUMBA_KERNEL_SCHEMA",
    "ROLLING_ARGMAX",
    "ROLLING_ARGMIN",
    "ROLLING_MAX",
    "ROLLING_MIN",
    "ROLLING_POSITION",
    "ROLLING_RANGE",
    "ROLLING_RANK",
    "BINARY_BETA",
    "BINARY_CORR",
    "BINARY_COV",
    "BINARY_ORTH",
    "UNARY_MEAN",
    "UNARY_STD",
    "UNARY_ZSCORE",
    "WEIGHTED_RESIDUAL",
    "WEIGHTED_SLOPE",
    "WEIGHTED_WMA",
    "cross_sectional_average_rank_kernel",
    "cleaned_portfolio_series_kernel",
    "ema_kernel",
    "rolling_binary_moment_kernel",
    "rolling_sum_kernel",
    "rolling_unary_moment_kernel",
    "rolling_weighted_kernel",
    "rank_ic_values_kernel",
    "rolling_window_kernel",
    "warm_numba_kernels",
]
