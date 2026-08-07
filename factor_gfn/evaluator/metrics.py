"""5日因子评价指标的 NumPy/SciPy 基准实现。

这里提供 reward 与最终绩效统计共同依赖的基础指标。候选因子进入 IC、组合或
因子相关性评价前统一执行逐日 1%/99% 缩尾、申万一级行业中性化与
z-score，但不改写解释器原始输出。
本模块不定义最终 reward 公式。
收益标签时点属于项目复现假设，不代表研报披露的内部实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from .cross_section import (
    DEFAULT_CLEANING_CONFIG,
    CrossSectionalCleaningConfig,
    NeutralizationDiagnostics,
    clean_candidate_factor_cross_sections,
    clean_factor_cross_sections,
)


FloatMatrix = npt.NDArray[np.float64]
FloatVector = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """集中管理第一版5日评价口径。"""

    horizon: int = 5
    entry_lag: int = 1
    rebalance_interval: int = 5
    rebalance_offset: int = 0
    annualization: float = 252.0 / 5.0
    long_quantile: float = 0.10
    min_cross_section_count: int = 20
    performance_ddof: int = 1

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon 必须为正整数")
        if self.entry_lag < 0:
            raise ValueError("entry_lag 不能为负数")
        if self.rebalance_interval <= 0:
            raise ValueError("rebalance_interval 必须为正整数")
        if not 0 <= self.rebalance_offset < self.rebalance_interval:
            raise ValueError("rebalance_offset 必须位于 [0, rebalance_interval) 内")
        if not np.isfinite(self.annualization) or self.annualization <= 0:
            raise ValueError("annualization 必须是有限正数")
        if not 0.0 < self.long_quantile < 1.0:
            raise ValueError("long_quantile 必须位于 (0, 1) 内")
        if self.min_cross_section_count < 2:
            raise ValueError("min_cross_section_count 至少为 2")
        if self.performance_ddof < 0:
            raise ValueError("performance_ddof 不能为负数")


@dataclass(frozen=True, slots=True)
class CrossSectionalSeries:
    """逐日截面指标及其联合有效样本覆盖。"""

    values: FloatVector
    sample_count: IntVector
    coverage: FloatVector


@dataclass(frozen=True, slots=True)
class ICSummary:
    mean: float
    std: float
    icir: float
    valid_periods: int
    total_periods: int


@dataclass(frozen=True, slots=True)
class ICEvaluation:
    """逐日重叠5日 RankIC 与每5日非重叠 reward RankIC。"""

    daily: CrossSectionalSeries
    rebalance_indices: IntVector
    rebalance_values: FloatVector
    rebalance_summary: ICSummary


@dataclass(frozen=True, slots=True)
class LongPortfolioSeries:
    long_return: FloatVector
    benchmark_return: FloatVector
    excess_return: FloatVector
    universe_count: IntVector
    long_count: IntVector


@dataclass(frozen=True, slots=True)
class LongShortPortfolioSeries:
    """候选因子原始方向的 Top-Bottom 等权收益序列。"""

    long_return: FloatVector
    short_return: FloatVector
    long_short_return: FloatVector
    universe_count: IntVector
    leg_count: IntVector


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    mean_period_return: float
    annualized_return: float
    annualized_ir: float
    std: float
    valid_periods: int
    total_periods: int


@dataclass(frozen=True, slots=True)
class CorrelationSummary:
    mean: float
    mean_absolute: float
    valid_periods: int
    total_periods: int


DEFAULT_CONFIG = EvaluationConfig()


def _matrix(values: npt.ArrayLike, label: str) -> FloatMatrix:
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须能够转换为 float64 数组") from exc
    if result.ndim != 2:
        raise ValueError(f"{label} 必须是 (date, stock) 二维矩阵")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{label} 的 date 轴和 stock 轴均不能为空")
    result[~np.isfinite(result)] = np.nan
    return result


def _same_shape(left: FloatMatrix, right: FloatMatrix) -> None:
    if left.shape != right.shape:
        raise ValueError(f"两个矩阵形状必须一致，实际为 {left.shape} 和 {right.shape}")


def _clean_candidate(
    factor: FloatMatrix,
    industry_labels: npt.ArrayLike | None,
    universe_mask: npt.ArrayLike | None,
    row_indices: npt.ArrayLike | None,
    cleaning_config: CrossSectionalCleaningConfig,
    neutralize_industry: bool,
    neutralization_diagnostics: NeutralizationDiagnostics | None = None,
) -> FloatMatrix:
    """选择显式的候选因子清洗口径，禁止静默跳过行业中性化。"""

    if neutralize_industry:
        if industry_labels is None:
            raise ValueError("启用行业中性化时必须提供 industry_labels")
        return clean_candidate_factor_cross_sections(
            factor,
            industry_labels,
            universe_mask,
            row_indices=row_indices,
            config=cleaning_config,
            diagnostics=neutralization_diagnostics,
        )
    return clean_factor_cross_sections(
        factor,
        universe_mask,
        row_indices=row_indices,
        config=cleaning_config,
    )


def build_forward_returns(
    open_prices: npt.ArrayLike,
    config: EvaluationConfig = DEFAULT_CONFIG,
) -> FloatMatrix:
    """构造 ``open[t+1+horizon] / open[t+1] - 1`` 同形标签矩阵。

    默认值对应已确认的复现假设：``open[t+6] / open[t+1] - 1``。无法取得
    完整标签、价格缺失或价格非正时返回 NaN。
    """

    prices = _matrix(open_prices, "open_prices")
    result = np.full_like(prices, np.nan)
    entry = config.entry_lag
    exit_ = entry + config.horizon
    if exit_ >= prices.shape[0]:
        return result

    entry_prices = prices[entry : prices.shape[0] - config.horizon]
    exit_prices = prices[exit_:]
    valid = (
        np.isfinite(entry_prices)
        & np.isfinite(exit_prices)
        & (entry_prices > 0.0)
        & (exit_prices > 0.0)
    )
    target = result[: prices.shape[0] - exit_]
    np.divide(exit_prices, entry_prices, out=target, where=valid)
    target[valid] -= 1.0
    target[~valid] = np.nan
    return result


def _pearson(left: FloatVector, right: FloatVector) -> float:
    if left.size < 2:
        return np.nan
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(
        np.dot(left_centered, left_centered)
        * np.dot(right_centered, right_centered)
    )
    if denominator <= np.finfo(np.float64).eps:
        return np.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def rank_ic_series(
    factor: npt.ArrayLike,
    forward_returns: npt.ArrayLike,
    min_count: int = DEFAULT_CONFIG.min_cross_section_count,
    *,
    industry_labels: npt.ArrayLike | None = None,
    universe_mask: npt.ArrayLike | None = None,
    cleaning_config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
    neutralize_industry: bool = True,
    row_indices: npt.ArrayLike | None = None,
    neutralization_diagnostics: NeutralizationDiagnostics | None = None,
) -> CrossSectionalSeries:
    """计算清洗后候选因子的逐日截面 Spearman RankIC。"""

    if min_count < 2:
        raise ValueError("min_count 至少为 2")
    raw_factor = _matrix(factor, "factor")
    if row_indices is None:
        rows = np.arange(raw_factor.shape[0], dtype=np.int64)
    else:
        raw_rows = np.asarray(row_indices)
        if raw_rows.ndim != 1 or not np.issubdtype(raw_rows.dtype, np.integer):
            raise ValueError("row_indices 必须是一维整数数组")
        rows = raw_rows.astype(np.int64, copy=False)
        if rows.size and ((rows < 0).any() or (rows >= raw_factor.shape[0]).any()):
            raise IndexError("row_indices 包含越界日期")
        if np.unique(rows).size != rows.size:
            raise ValueError("row_indices 不能包含重复日期")
    factor_matrix = _clean_candidate(
        raw_factor,
        industry_labels,
        universe_mask,
        rows,
        cleaning_config,
        neutralize_industry,
        neutralization_diagnostics,
    )
    return_matrix = _matrix(forward_returns, "forward_returns")
    _same_shape(factor_matrix, return_matrix)

    date_count, stock_count = factor_matrix.shape
    values = np.full(date_count, np.nan)
    sample_count = np.zeros(date_count, dtype=np.int64)
    universe = (
        np.ones(raw_factor.shape, dtype=bool)
        if universe_mask is None
        else np.asarray(universe_mask, dtype=bool)
    )
    for date_index in rows:
        raw_valid = (
            universe[date_index]
            & np.isfinite(raw_factor[date_index])
            & np.isfinite(return_matrix[date_index])
        )
        sample_count[date_index] = int(raw_valid.sum())
        valid = np.isfinite(factor_matrix[date_index]) & np.isfinite(
            return_matrix[date_index]
        )
        count = int(valid.sum())
        if count < min_count:
            continue
        factor_values = factor_matrix[date_index, valid]
        return_values = return_matrix[date_index, valid]
        values[date_index] = _pearson(
            rankdata(factor_values, method="average"),
            rankdata(return_values, method="average"),
        )
    return CrossSectionalSeries(
        values=values,
        sample_count=sample_count,
        coverage=sample_count.astype(np.float64) / stock_count,
    )


def select_rebalance_indices(
    sample_count: npt.ArrayLike,
    config: EvaluationConfig = DEFAULT_CONFIG,
) -> IntVector:
    """从首个满足最小样本数的日期起按固定交易日间隔选择评价日。"""

    counts = np.asarray(sample_count)
    if counts.ndim != 1:
        raise ValueError("sample_count 必须是一维数组")
    candidates = np.flatnonzero(counts >= config.min_cross_section_count)
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    first = int(candidates[0]) + config.rebalance_offset
    if first >= counts.size:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, counts.size, config.rebalance_interval, dtype=np.int64)


def summarize_ic(
    values: npt.ArrayLike,
    ddof: int = DEFAULT_CONFIG.performance_ddof,
) -> ICSummary:
    """汇总 IC；绩效统计使用样本标准差，与时序算子 ``ddof=0`` 独立。"""

    if ddof < 0:
        raise ValueError("ddof 不能为负数")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("IC values 必须是一维数组")
    valid = array[np.isfinite(array)]
    mean = float(valid.mean()) if valid.size else np.nan
    std = float(valid.std(ddof=ddof)) if valid.size > ddof else np.nan
    icir = mean / std if np.isfinite(std) and std > 0.0 else np.nan
    return ICSummary(
        mean=mean,
        std=std,
        icir=float(icir),
        valid_periods=int(valid.size),
        total_periods=int(array.size),
    )


def evaluate_rank_ic(
    factor: npt.ArrayLike,
    forward_returns: npt.ArrayLike,
    config: EvaluationConfig = DEFAULT_CONFIG,
    *,
    industry_labels: npt.ArrayLike | None = None,
    universe_mask: npt.ArrayLike | None = None,
    cleaning_config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
    neutralize_industry: bool = True,
    rebalance_indices: npt.ArrayLike | None = None,
    neutralization_diagnostics: NeutralizationDiagnostics | None = None,
) -> ICEvaluation:
    """同时返回分析用逐日 IC 与 reward 默认使用的5日非重叠 IC。"""

    fixed_indices: npt.NDArray[np.int64] | None = None
    if rebalance_indices is not None:
        raw_indices = np.asarray(rebalance_indices)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("rebalance_indices 必须是一维整数数组")
        fixed_indices = raw_indices.astype(np.int64, copy=False)
        if fixed_indices.size and (
            (fixed_indices < 0).any() or (fixed_indices >= np.asarray(factor).shape[0]).any()
        ):
            raise IndexError("rebalance_indices 包含越界日期")
        if np.unique(fixed_indices).size != fixed_indices.size:
            raise ValueError("rebalance_indices 不能包含重复日期")

    daily = rank_ic_series(
        factor,
        forward_returns,
        min_count=config.min_cross_section_count,
        industry_labels=industry_labels,
        universe_mask=universe_mask,
        cleaning_config=cleaning_config,
        neutralize_industry=neutralize_industry,
        row_indices=fixed_indices,
        neutralization_diagnostics=neutralization_diagnostics,
    )
    indices = (
        select_rebalance_indices(daily.sample_count, config)
        if fixed_indices is None
        else fixed_indices.copy()
    )
    rebalance_values = daily.values[indices].copy()
    return ICEvaluation(
        daily=daily,
        rebalance_indices=indices,
        rebalance_values=rebalance_values,
        rebalance_summary=summarize_ic(
            rebalance_values, ddof=config.performance_ddof
        ),
    )


def infer_long_direction(train_ic_mean: float) -> int:
    """由训练集 IC 方向生成多头排序方向，不修改任何因子值。"""

    if not np.isfinite(train_ic_mean) or train_ic_mean == 0.0:
        raise ValueError("train_ic_mean 必须是非零有限值")
    return 1 if train_ic_mean > 0.0 else -1


def long_portfolio_series(
    factor: npt.ArrayLike,
    forward_returns: npt.ArrayLike,
    rebalance_indices: npt.ArrayLike,
    direction: int,
    config: EvaluationConfig = DEFAULT_CONFIG,
    *,
    industry_labels: npt.ArrayLike | None = None,
    universe_mask: npt.ArrayLike | None = None,
    cleaning_config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
    neutralize_industry: bool = True,
    neutralization_diagnostics: NeutralizationDiagnostics | None = None,
) -> LongPortfolioSeries:
    """构造前10%等权多头、有效股票池等权基准及两者超额序列。"""

    if direction not in (-1, 1):
        raise ValueError("direction 只能为 -1 或 1")
    raw_factor = _matrix(factor, "factor")
    return_matrix = _matrix(forward_returns, "forward_returns")
    _same_shape(raw_factor, return_matrix)
    indices = np.asarray(rebalance_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("rebalance_indices 必须是一维整数数组")
    indices = indices.astype(np.int64, copy=False)
    if indices.size and ((indices < 0).any() or (indices >= raw_factor.shape[0]).any()):
        raise IndexError("rebalance_indices 包含越界日期")
    if np.unique(indices).size != indices.size:
        raise ValueError("rebalance_indices 不能包含重复日期")
    factor_matrix = _clean_candidate(
        raw_factor,
        industry_labels,
        universe_mask,
        indices,
        cleaning_config,
        neutralize_industry,
        neutralization_diagnostics,
    )

    date_count = factor_matrix.shape[0]
    long_return = np.full(date_count, np.nan)
    benchmark_return = np.full(date_count, np.nan)
    excess_return = np.full(date_count, np.nan)
    universe_count = np.zeros(date_count, dtype=np.int64)
    long_count = np.zeros(date_count, dtype=np.int64)

    for date_index in indices:
        valid = np.isfinite(factor_matrix[date_index]) & np.isfinite(
            return_matrix[date_index]
        )
        count = int(valid.sum())
        universe_count[date_index] = count
        if count < config.min_cross_section_count:
            continue
        scores = direction * factor_matrix[date_index, valid]
        if np.ptp(scores) <= np.finfo(np.float64).eps:
            continue
        returns = return_matrix[date_index, valid]
        selected_count = max(1, ceil(count * config.long_quantile))
        order = np.argsort(scores, kind="stable")
        selected = order[-selected_count:]
        long_count[date_index] = selected_count
        long_return[date_index] = float(returns[selected].mean())
        benchmark_return[date_index] = float(returns.mean())
        excess_return[date_index] = (
            long_return[date_index] - benchmark_return[date_index]
        )

    return LongPortfolioSeries(
        long_return=long_return,
        benchmark_return=benchmark_return,
        excess_return=excess_return,
        universe_count=universe_count,
        long_count=long_count,
    )


def long_short_portfolio_series(
    factor: npt.ArrayLike,
    forward_returns: npt.ArrayLike,
    rebalance_indices: npt.ArrayLike,
    config: EvaluationConfig = DEFAULT_CONFIG,
    *,
    industry_labels: npt.ArrayLike | None = None,
    universe_mask: npt.ArrayLike | None = None,
    cleaning_config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
    neutralize_industry: bool = True,
    neutralization_diagnostics: NeutralizationDiagnostics | None = None,
) -> LongShortPortfolioSeries:
    """构造候选因子原始方向的 Top 10% 减 Bottom 10% 收益序列。"""

    raw_factor = _matrix(factor, "factor")
    return_matrix = _matrix(forward_returns, "forward_returns")
    _same_shape(raw_factor, return_matrix)
    indices = np.asarray(rebalance_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("rebalance_indices 必须是一维整数数组")
    indices = indices.astype(np.int64, copy=False)
    if indices.size and ((indices < 0).any() or (indices >= raw_factor.shape[0]).any()):
        raise IndexError("rebalance_indices 包含越界日期")
    if np.unique(indices).size != indices.size:
        raise ValueError("rebalance_indices 不能包含重复日期")
    cleaned = _clean_candidate(
        raw_factor,
        industry_labels,
        universe_mask,
        indices,
        cleaning_config,
        neutralize_industry,
        neutralization_diagnostics,
    )

    date_count = raw_factor.shape[0]
    long_return = np.full(date_count, np.nan)
    short_return = np.full(date_count, np.nan)
    long_short_return = np.full(date_count, np.nan)
    universe_count = np.zeros(date_count, dtype=np.int64)
    leg_count = np.zeros(date_count, dtype=np.int64)
    for date_index in indices:
        valid = np.isfinite(cleaned[date_index]) & np.isfinite(return_matrix[date_index])
        count = int(valid.sum())
        universe_count[date_index] = count
        if count < config.min_cross_section_count:
            continue
        scores = cleaned[date_index, valid]
        if np.ptp(scores) <= np.finfo(np.float64).eps:
            continue
        returns = return_matrix[date_index, valid]
        selected_count = min(max(1, ceil(count * config.long_quantile)), count // 2)
        order = np.argsort(scores, kind="stable")
        bottom = order[:selected_count]
        top = order[-selected_count:]
        leg_count[date_index] = selected_count
        long_return[date_index] = float(returns[top].mean())
        short_return[date_index] = float(returns[bottom].mean())
        long_short_return[date_index] = long_return[date_index] - short_return[date_index]
    return LongShortPortfolioSeries(
        long_return=long_return,
        short_return=short_return,
        long_short_return=long_short_return,
        universe_count=universe_count,
        leg_count=leg_count,
    )


def summarize_excess_returns(
    excess_returns: npt.ArrayLike,
    config: EvaluationConfig = DEFAULT_CONFIG,
) -> PerformanceSummary:
    """汇总5日超额收益，IR 使用 ``sqrt(annualization)`` 年化。"""

    array = np.asarray(excess_returns, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("excess_returns 必须是一维数组")
    valid = array[np.isfinite(array)]
    mean = float(valid.mean()) if valid.size else np.nan
    std = (
        float(valid.std(ddof=config.performance_ddof))
        if valid.size > config.performance_ddof
        else np.nan
    )
    annualized_return = mean * config.annualization if np.isfinite(mean) else np.nan
    annualized_ir = (
        mean / std * sqrt(config.annualization)
        if np.isfinite(std) and std > 0.0
        else np.nan
    )
    return PerformanceSummary(
        mean_period_return=mean,
        annualized_return=float(annualized_return),
        annualized_ir=float(annualized_ir),
        std=std,
        valid_periods=int(valid.size),
        total_periods=int(array.size),
    )


def factor_cross_sectional_correlation(
    left_factor: npt.ArrayLike,
    right_factor: npt.ArrayLike,
    min_count: int = DEFAULT_CONFIG.min_cross_section_count,
    *,
    industry_labels: npt.ArrayLike | None = None,
    universe_mask: npt.ArrayLike | None = None,
    cleaning_config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
    neutralize_industry: bool = True,
) -> CrossSectionalSeries:
    """计算两个因子的逐日截面 Spearman 相关性。"""
    left = _clean_candidate(
        _matrix(left_factor, "left_factor"),
        industry_labels,
        universe_mask,
        None,
        cleaning_config,
        neutralize_industry,
    )
    right = _clean_candidate(
        _matrix(right_factor, "right_factor"),
        industry_labels,
        universe_mask,
        None,
        cleaning_config,
        neutralize_industry,
    )
    _same_shape(left, right)
    date_count, stock_count = left.shape
    values = np.full(date_count, np.nan)
    sample_count = np.zeros(date_count, dtype=np.int64)
    for date_index in range(date_count):
        valid = np.isfinite(left[date_index]) & np.isfinite(right[date_index])
        count = int(valid.sum())
        sample_count[date_index] = count
        if count < min_count:
            continue
        values[date_index] = _pearson(
            rankdata(left[date_index, valid], method="average"),
            rankdata(right[date_index, valid], method="average"),
        )
    return CrossSectionalSeries(
        values=values,
        sample_count=sample_count,
        coverage=sample_count.astype(np.float64) / stock_count,
    )


def summarize_correlation(values: npt.ArrayLike) -> CorrelationSummary:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("correlation values 必须是一维数组")
    valid = array[np.isfinite(array)]
    return CorrelationSummary(
        mean=float(valid.mean()) if valid.size else np.nan,
        mean_absolute=float(np.abs(valid).mean()) if valid.size else np.nan,
        valid_periods=int(valid.size),
        total_periods=int(array.size),
    )


def excess_return_correlation(
    left_excess: npt.ArrayLike,
    right_excess: npt.ArrayLike,
    min_periods: int = 2,
) -> float:
    """计算两个多头超额收益序列在共同有效调仓期上的 Pearson 相关性。"""

    if min_periods < 2:
        raise ValueError("min_periods 至少为 2")
    left = np.asarray(left_excess, dtype=np.float64)
    right = np.asarray(right_excess, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("收益序列必须是一维数组")
    if left.shape != right.shape:
        raise ValueError("两个收益序列长度必须一致")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < min_periods:
        return np.nan
    return _pearson(left[valid], right[valid])


__all__ = [
    "DEFAULT_CONFIG",
    "CorrelationSummary",
    "CrossSectionalSeries",
    "EvaluationConfig",
    "ICEvaluation",
    "ICSummary",
    "LongPortfolioSeries",
    "LongShortPortfolioSeries",
    "PerformanceSummary",
    "build_forward_returns",
    "evaluate_rank_ic",
    "excess_return_correlation",
    "factor_cross_sectional_correlation",
    "infer_long_direction",
    "long_portfolio_series",
    "long_short_portfolio_series",
    "rank_ic_series",
    "select_rebalance_indices",
    "summarize_correlation",
    "summarize_excess_returns",
    "summarize_ic",
]
