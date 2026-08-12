"""阶段四 Reward 计算、数值稳定化和进程内缓存。

原始奖励严格保持研报结构；``reward_floor`` 只作用于 TB 所需的正值奖励，
不会把缺失指标或无效候选因子伪装成低奖励有效样本。
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np
import numpy.typing as npt

from factor_gfn.barra import (
    STYLE_NAMES,
    BarraPenaltyResult,
    LongShortSeries,
    calculate_barra_ts_corr,
)
from factor_gfn.evaluator import (
    DEFAULT_CONFIG,
    EncodedIndustryPanel,
    EvaluationConfig,
    NeutralizationDiagnostics,
    cleaned_portfolio_returns_from_cleaned,
    clean_candidate_factor_cross_sections,
    clean_factor_cross_sections,
    encode_industry_panel,
    evaluate_rank_ic,
    infer_long_direction,
    long_portfolio_series,
    long_portfolio_series_from_cleaned,
    long_short_portfolio_series,
    long_short_portfolio_series_from_cleaned,
    rank_ic_values_from_cleaned,
    summarize_excess_returns,
    summarize_ic,
)

from .config import RewardConfig


@dataclass(frozen=True, slots=True)
class RewardResult:
    expression_hash: str
    valid: bool
    invalid_reason: str | None
    train_ic: float
    train_long_ir: float
    barra_ts_corr: float
    barra_correlations: dict[str, float]
    barra_valid_periods: dict[str, int]
    dominant_barra_factor: str | None
    dominant_barra_correlation: float
    raw_reward: float
    reward: float
    log_reward: float
    floor_applied: bool
    long_direction: int | None
    ic_valid_periods: int
    long_ir_valid_periods: int
    industry_neutralized: bool
    neutralization_skipped_dates: tuple[str, ...]
    neutralization_skipped_rate: float
    neutralization_skipped_details: tuple[dict[str, int | str], ...]


@dataclass(frozen=True, slots=True)
class RewardEvaluation:
    result: RewardResult
    cache_hit: bool


class RewardCache:
    """有容量上限的 LRU 进程内缓存；缓存值在读写时复制。"""

    def __init__(self, max_entries: int = 50_000) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries 必须是正整数")
        self.max_entries = max_entries
        self._values: OrderedDict[str, RewardResult] = OrderedDict()

    def get(self, key: str) -> RewardResult | None:
        value = self._values.get(key)
        if value is None:
            return None
        self._values.move_to_end(key)
        return deepcopy(value)

    def put(self, key: str, value: RewardResult) -> None:
        self._values[key] = deepcopy(value)
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


def _dominant_barra(correlations: Mapping[str, float]) -> tuple[str | None, float]:
    finite = [(name, float(value)) for name, value in correlations.items() if np.isfinite(value)]
    if not finite:
        return None, np.nan
    return max(finite, key=lambda item: abs(item[1]))


def _invalid_result(
    expression_hash: str,
    reason: str,
    *,
    train_ic: float = np.nan,
    train_long_ir: float = np.nan,
    penalty: BarraPenaltyResult | None = None,
    long_direction: int | None = None,
    ic_valid_periods: int = 0,
    long_ir_valid_periods: int = 0,
    industry_neutralized: bool,
    neutralization_skipped_dates: tuple[str, ...] = (),
    neutralization_skipped_rate: float = 0.0,
    neutralization_skipped_details: tuple[dict[str, int | str], ...] = (),
) -> RewardResult:
    correlations = (
        {name: float(penalty.correlations[name]) for name in STYLE_NAMES}
        if penalty is not None
        else {name: np.nan for name in STYLE_NAMES}
    )
    periods = (
        {name: int(penalty.valid_periods[name]) for name in STYLE_NAMES}
        if penalty is not None
        else {name: 0 for name in STYLE_NAMES}
    )
    dominant_name, dominant_corr = _dominant_barra(correlations)
    return RewardResult(
        expression_hash=expression_hash,
        valid=False,
        invalid_reason=reason,
        train_ic=float(train_ic),
        train_long_ir=float(train_long_ir),
        barra_ts_corr=float(penalty.barra_ts_corr) if penalty is not None else np.nan,
        barra_correlations=correlations,
        barra_valid_periods=periods,
        dominant_barra_factor=dominant_name,
        dominant_barra_correlation=float(dominant_corr),
        raw_reward=np.nan,
        reward=np.nan,
        log_reward=np.nan,
        floor_applied=False,
        long_direction=long_direction,
        ic_valid_periods=ic_valid_periods,
        long_ir_valid_periods=long_ir_valid_periods,
        industry_neutralized=industry_neutralized,
        neutralization_skipped_dates=neutralization_skipped_dates,
        neutralization_skipped_rate=float(neutralization_skipped_rate),
        neutralization_skipped_details=deepcopy(neutralization_skipped_details),
    )


def combine_reward_components(
    expression_hash: str,
    train_ic: float,
    train_long_ir: float,
    penalty: BarraPenaltyResult,
    config: RewardConfig = RewardConfig(),
    *,
    long_direction: int | None = None,
    ic_valid_periods: int = 0,
    long_ir_valid_periods: int = 0,
    neutralization_skipped_dates: tuple[str, ...] = (),
    neutralization_skipped_rate: float = 0.0,
    neutralization_skipped_details: tuple[dict[str, int | str], ...] = (),
) -> RewardResult:
    """应用冻结公式并同时保留 raw reward 与 TB 正值 reward。"""

    if not expression_hash:
        raise ValueError("expression_hash 不能为空")
    neutralized = config.candidate_industry_neutralization
    if not np.isfinite(train_ic):
        return _invalid_result(
            expression_hash,
            "train_ic 非有限值",
            train_ic=train_ic,
            train_long_ir=train_long_ir,
            penalty=penalty,
            long_direction=long_direction,
            ic_valid_periods=ic_valid_periods,
            long_ir_valid_periods=long_ir_valid_periods,
            industry_neutralized=neutralized,
            neutralization_skipped_dates=neutralization_skipped_dates,
            neutralization_skipped_rate=neutralization_skipped_rate,
            neutralization_skipped_details=neutralization_skipped_details,
        )
    if not np.isfinite(train_long_ir):
        return _invalid_result(
            expression_hash,
            "train_long_ir 非有限值",
            train_ic=train_ic,
            train_long_ir=train_long_ir,
            penalty=penalty,
            long_direction=long_direction,
            ic_valid_periods=ic_valid_periods,
            long_ir_valid_periods=long_ir_valid_periods,
            industry_neutralized=neutralized,
            neutralization_skipped_dates=neutralization_skipped_dates,
            neutralization_skipped_rate=neutralization_skipped_rate,
            neutralization_skipped_details=neutralization_skipped_details,
        )
    if not np.isfinite(penalty.barra_ts_corr):
        return _invalid_result(
            expression_hash,
            "barra_ts_corr 非有限值或共同有效期不足",
            train_ic=train_ic,
            train_long_ir=train_long_ir,
            penalty=penalty,
            long_direction=long_direction,
            ic_valid_periods=ic_valid_periods,
            long_ir_valid_periods=long_ir_valid_periods,
            industry_neutralized=neutralized,
            neutralization_skipped_dates=neutralization_skipped_dates,
            neutralization_skipped_rate=neutralization_skipped_rate,
            neutralization_skipped_details=neutralization_skipped_details,
        )

    clipped_ir = float(np.clip(train_long_ir, 0.0, config.long_ir_cap))
    clipped_barra = float(np.clip(penalty.barra_ts_corr, 0.0, 1.0))
    raw_reward = (
        abs(float(train_ic))
        * (1.0 + config.long_ir_lambda * clipped_ir)
        * (1.0 - config.barra_ts_penalty_mu * clipped_barra)
    )
    if not math.isfinite(raw_reward) or raw_reward < 0.0:
        return _invalid_result(
            expression_hash,
            "reward 公式产生无效值",
            train_ic=train_ic,
            train_long_ir=train_long_ir,
            penalty=penalty,
            long_direction=long_direction,
            ic_valid_periods=ic_valid_periods,
            long_ir_valid_periods=long_ir_valid_periods,
            industry_neutralized=neutralized,
            neutralization_skipped_dates=neutralization_skipped_dates,
            neutralization_skipped_rate=neutralization_skipped_rate,
            neutralization_skipped_details=neutralization_skipped_details,
        )
    reward = max(raw_reward, config.reward_floor)
    correlations = {name: float(penalty.correlations[name]) for name in STYLE_NAMES}
    periods = {name: int(penalty.valid_periods[name]) for name in STYLE_NAMES}
    dominant_name, dominant_corr = _dominant_barra(correlations)
    return RewardResult(
        expression_hash=expression_hash,
        valid=True,
        invalid_reason=None,
        train_ic=float(train_ic),
        train_long_ir=float(train_long_ir),
        barra_ts_corr=float(penalty.barra_ts_corr),
        barra_correlations=correlations,
        barra_valid_periods=periods,
        dominant_barra_factor=dominant_name,
        dominant_barra_correlation=float(dominant_corr),
        raw_reward=float(raw_reward),
        reward=float(reward),
        log_reward=float(math.log(reward)),
        floor_applied=raw_reward < config.reward_floor,
        long_direction=long_direction,
        ic_valid_periods=ic_valid_periods,
        long_ir_valid_periods=long_ir_valid_periods,
        industry_neutralized=neutralized,
        neutralization_skipped_dates=neutralization_skipped_dates,
        neutralization_skipped_rate=float(neutralization_skipped_rate),
        neutralization_skipped_details=deepcopy(neutralization_skipped_details),
    )


class RewardEvaluator:
    """把候选因子矩阵评估为可供 TB 使用的正值奖励。"""

    def __init__(
        self,
        forward_returns: npt.ArrayLike,
        barra_long_short: Mapping[str, LongShortSeries],
        *,
        data_fingerprint: str,
        evaluation_config: EvaluationConfig = DEFAULT_CONFIG,
        reward_config: RewardConfig = RewardConfig(),
        universe_mask: npt.ArrayLike | None = None,
        industry_labels: npt.ArrayLike | None = None,
        industry_fingerprint: str | None = None,
        rebalance_indices: npt.ArrayLike | None = None,
        evaluation_dates: npt.ArrayLike | None = None,
        cache: RewardCache | None = None,
    ) -> None:
        returns = np.asarray(forward_returns, dtype=np.float64)
        if returns.ndim != 2 or 0 in returns.shape:
            raise ValueError("forward_returns 必须是非空 (date, stock) 矩阵")
        if not data_fingerprint.strip():
            raise ValueError("data_fingerprint 不能为空")
        missing = set(STYLE_NAMES).difference(barra_long_short)
        if missing:
            raise ValueError(f"barra_long_short 缺少风格：{sorted(missing)}")
        if reward_config.candidate_industry_neutralization:
            if industry_labels is None:
                raise ValueError("启用候选因子行业中性化时必须提供 industry_labels")
            if not industry_fingerprint or not industry_fingerprint.strip():
                raise ValueError("启用行业中性化时必须提供 industry_fingerprint")
            if evaluation_dates is None:
                raise ValueError("启用行业中性化时必须提供 evaluation_dates")
        if universe_mask is not None and np.asarray(universe_mask).shape != returns.shape:
            raise ValueError("universe_mask 必须与 forward_returns 同形")
        fixed_indices: npt.NDArray[np.int64] | None = None
        if rebalance_indices is not None:
            raw_indices = np.asarray(rebalance_indices)
            if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
                raise ValueError("rebalance_indices 必须是一维整数数组")
            fixed_indices = raw_indices.astype(np.int64, copy=True)
            if fixed_indices.size == 0:
                raise ValueError("固定 rebalance_indices 不能为空")
            if (fixed_indices < 0).any() or (fixed_indices >= returns.shape[0]).any():
                raise IndexError("rebalance_indices 包含越界日期")
            if np.unique(fixed_indices).size != fixed_indices.size:
                raise ValueError("rebalance_indices 不能包含重复日期")
            fixed_indices.setflags(write=False)

        fixed_dates: npt.NDArray[np.datetime64] | None = None
        if evaluation_dates is not None:
            try:
                fixed_dates = np.asarray(evaluation_dates, dtype="datetime64[D]").copy()
            except (TypeError, ValueError) as exc:
                raise ValueError("evaluation_dates 必须能够转换为日期数组") from exc
            if fixed_dates.ndim != 1 or fixed_dates.size != returns.shape[0]:
                raise ValueError("evaluation_dates 必须是一维且与 forward_returns 日期轴等长")
            if np.isnat(fixed_dates).any():
                raise ValueError("evaluation_dates 不能包含 NaT")
            if np.unique(fixed_dates).size != fixed_dates.size:
                raise ValueError("evaluation_dates 不能包含重复日期")
            if fixed_dates.size > 1 and not np.all(fixed_dates[:-1] < fixed_dates[1:]):
                raise ValueError("evaluation_dates 必须严格升序")
            fixed_dates.setflags(write=False)

        self.forward_returns = returns
        self.barra_long_short = dict(barra_long_short)
        self.evaluation_config = evaluation_config
        self.reward_config = reward_config
        self.universe_mask = universe_mask
        self.industry_labels = industry_labels
        self.rebalance_indices = fixed_indices
        self.evaluation_dates = fixed_dates
        self.cache = cache if cache is not None else RewardCache()
        self._compact_forward_returns: npt.NDArray[np.float64] | None = None
        self._compact_universe_mask: npt.NDArray[np.bool_] | None = None
        self._compact_industry_labels: npt.NDArray | None = None
        self._compact_encoded_industries: EncodedIndustryPanel | None = None
        self._compact_barra_long_short: dict[str, LongShortSeries] | None = None
        if fixed_indices is not None:
            self._compact_forward_returns = returns[fixed_indices]
            self._compact_universe_mask = (
                None
                if universe_mask is None
                else np.asarray(universe_mask, dtype=bool)[fixed_indices]
            )
            if industry_labels is not None:
                labels = np.asarray(industry_labels)
                if labels.ndim == 1 and labels.shape[0] == returns.shape[1]:
                    self._compact_industry_labels = labels
                elif labels.shape == returns.shape:
                    self._compact_industry_labels = labels[fixed_indices]
                else:
                    raise ValueError(
                        "industry_labels 必须是一维股票标签或与 forward_returns 同形"
                    )
                self._compact_encoded_industries = encode_industry_panel(
                    self._compact_industry_labels,
                    self._compact_forward_returns.shape,
                )
            compact_references: dict[str, LongShortSeries] = {}
            for name in STYLE_NAMES:
                series = barra_long_short[name]
                if np.asarray(series.long_short_return).shape != (returns.shape[0],):
                    raise ValueError(f"{name} Barra 收益序列必须与日期轴等长")
                compact_references[name] = LongShortSeries(
                    long_return=np.asarray(series.long_return)[fixed_indices],
                    short_return=np.asarray(series.short_return)[fixed_indices],
                    long_short_return=np.asarray(series.long_short_return)[fixed_indices],
                    universe_count=np.asarray(series.universe_count)[fixed_indices],
                    leg_count=np.asarray(series.leg_count)[fixed_indices],
                )
            self._compact_barra_long_short = compact_references
        manifest = {
            "schema": "factor_gfn.reward_context.v3",
            "data_fingerprint": data_fingerprint,
            "industry_fingerprint": industry_fingerprint,
            "evaluation_config": asdict(evaluation_config),
            "reward_config": asdict(reward_config),
            "styles": list(STYLE_NAMES),
            "rebalance_indices": (
                fixed_indices.tolist() if fixed_indices is not None else None
            ),
            "evaluation_dates_sha256": (
                hashlib.sha256(fixed_dates.astype("int64").tobytes()).hexdigest()
                if fixed_dates is not None
                else None
            ),
            "industry_neutralization_policy": {
                "schema": "factor_gfn.strict_industry_neutralization.v1",
                "failed_date_action": "exclude_entire_candidate_cross_section",
                "unknown_industry_stock_action": "exclude_stock",
                "calendar_action": "keep_global_phase_without_backfill",
            },
            "reward_panel": {
                "mode": (
                    "fixed_rebalance_compact"
                    if fixed_indices is not None
                    else "dynamic_full_axis"
                ),
                "row_count": (
                    int(fixed_indices.size)
                    if fixed_indices is not None
                    else int(returns.shape[0])
                ),
                "candidate_cleaning_calls_per_evaluation": (
                    1 if fixed_indices is not None else 3
                ),
            },
        }
        payload = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        self.context_fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _neutralization_summary(
        self,
        diagnostics: NeutralizationDiagnostics,
        rebalance_indices: npt.NDArray[np.int64],
        *,
        compact_rows: bool = False,
    ) -> tuple[
        tuple[str, ...],
        float,
        tuple[dict[str, int | str], ...],
    ]:
        if not self.reward_config.candidate_industry_neutralization:
            return (), 0.0, ()
        if self.evaluation_dates is None:
            raise RuntimeError("行业中性化诊断缺少 evaluation_dates")
        if compact_rows:
            local_rows = sorted(
                set(range(rebalance_indices.size)).intersection(diagnostics.skipped_rows)
            )
            row_pairs = [
                (local_index, int(rebalance_indices[local_index]))
                for local_index in local_rows
            ]
        else:
            active_rows = {int(index) for index in rebalance_indices}
            row_pairs = [
                (index, index)
                for index in sorted(active_rows.intersection(diagnostics.skipped_rows))
            ]
        skipped_dates = tuple(
            str(self.evaluation_dates[original_index])
            for _, original_index in row_pairs
        )
        rate = len(row_pairs) / len(rebalance_indices) if len(rebalance_indices) else 0.0
        details = tuple(
            {
                "date": str(self.evaluation_dates[original_index]),
                "row_index": int(original_index),
                "factor_valid_count": int(
                    diagnostics.skipped_details[local_index].factor_valid_count
                ),
                "known_industry_count": int(
                    diagnostics.skipped_details[local_index].known_industry_count
                ),
                "industry_count": int(
                    diagnostics.skipped_details[local_index].industry_count
                ),
                "required_regression_count": int(
                    diagnostics.skipped_details[local_index].required_regression_count
                ),
                "reason": diagnostics.skipped_details[local_index].reason,
            }
            for local_index, original_index in row_pairs
        )
        return skipped_dates, float(rate), details

    def _evaluate_fixed_panel(
        self,
        expression_hash: str,
        values: npt.NDArray[np.float64],
    ) -> RewardResult:
        """在冻结调仓截面上清洗一次，并复用同一矩阵计算全部 Reward 分量。"""

        if self.rebalance_indices is None or self._compact_forward_returns is None:
            raise RuntimeError("固定调仓面板尚未初始化")
        compact_values = values[self.rebalance_indices]
        neutralize = self.reward_config.candidate_industry_neutralization
        diagnostics = NeutralizationDiagnostics()
        if neutralize:
            if self._compact_industry_labels is None:
                raise RuntimeError("固定调仓面板缺少行业标签")
            cleaned = clean_candidate_factor_cross_sections(
                compact_values,
                self._compact_industry_labels,
                self._compact_universe_mask,
                diagnostics=diagnostics,
                encoded_industries=self._compact_encoded_industries,
            )
        else:
            cleaned = clean_factor_cross_sections(
                compact_values,
                self._compact_universe_mask,
            )

        ic_values = rank_ic_values_from_cleaned(
            cleaned,
            self._compact_forward_returns,
            self.evaluation_config.min_cross_section_count,
        )
        ic_summary = summarize_ic(
            ic_values,
            ddof=self.evaluation_config.performance_ddof,
        )
        train_ic = ic_summary.mean
        skipped_dates, skipped_rate, skipped_details = self._neutralization_summary(
            diagnostics,
            self.rebalance_indices,
            compact_rows=True,
        )
        if not np.isfinite(train_ic) or train_ic == 0.0:
            penalty = BarraPenaltyResult(
                barra_ts_corr=np.nan,
                correlations={name: np.nan for name in STYLE_NAMES},
                valid_periods={name: 0 for name in STYLE_NAMES},
            )
            return _invalid_result(
                expression_hash,
                "train_ic 为零或非有限值，无法确定多头方向",
                train_ic=train_ic,
                penalty=penalty,
                ic_valid_periods=ic_summary.valid_periods,
                industry_neutralized=neutralize,
                neutralization_skipped_dates=skipped_dates,
                neutralization_skipped_rate=skipped_rate,
                neutralization_skipped_details=skipped_details,
            )

        direction = infer_long_direction(train_ic)
        long_excess, candidate_long_short = cleaned_portfolio_returns_from_cleaned(
            cleaned,
            self._compact_forward_returns,
            direction,
            self.evaluation_config,
        )
        long_summary = summarize_excess_returns(
            long_excess,
            self.evaluation_config,
        )
        if self._compact_barra_long_short is None:
            raise RuntimeError("固定调仓面板缺少 Barra 参考序列")
        penalty = calculate_barra_ts_corr(
            candidate_long_short,
            self._compact_barra_long_short,
            min_periods=self.reward_config.barra_min_common_periods,
        )
        return combine_reward_components(
            expression_hash,
            train_ic,
            long_summary.annualized_ir,
            penalty,
            self.reward_config,
            long_direction=direction,
            ic_valid_periods=ic_summary.valid_periods,
            long_ir_valid_periods=long_summary.valid_periods,
            neutralization_skipped_dates=skipped_dates,
            neutralization_skipped_rate=skipped_rate,
            neutralization_skipped_details=skipped_details,
        )

    def evaluate(self, expression_hash: str, factor: npt.ArrayLike) -> RewardEvaluation:
        if not expression_hash:
            raise ValueError("expression_hash 不能为空")
        key = f"{self.context_fingerprint}:{expression_hash}"
        cached = self.cache.get(key)
        if cached is not None:
            return RewardEvaluation(result=cached, cache_hit=True)

        values = np.asarray(factor, dtype=np.float64)
        if values.shape != self.forward_returns.shape:
            raise ValueError("factor 必须与 forward_returns 同形")
        if self.rebalance_indices is not None:
            result = self._evaluate_fixed_panel(expression_hash, values)
            self.cache.put(key, result)
            return RewardEvaluation(result=result, cache_hit=False)

        neutralize = self.reward_config.candidate_industry_neutralization
        diagnostics = NeutralizationDiagnostics()
        ic = evaluate_rank_ic(
            values,
            self.forward_returns,
            self.evaluation_config,
            industry_labels=self.industry_labels,
            universe_mask=self.universe_mask,
            neutralize_industry=neutralize,
            rebalance_indices=self.rebalance_indices,
            neutralization_diagnostics=diagnostics,
        )
        train_ic = ic.rebalance_summary.mean
        if not np.isfinite(train_ic) or train_ic == 0.0:
            skipped_dates, skipped_rate, skipped_details = self._neutralization_summary(
                diagnostics,
                ic.rebalance_indices,
            )
            penalty = BarraPenaltyResult(
                barra_ts_corr=np.nan,
                correlations={name: np.nan for name in STYLE_NAMES},
                valid_periods={name: 0 for name in STYLE_NAMES},
            )
            result = _invalid_result(
                expression_hash,
                "train_ic 为零或非有限值，无法确定多头方向",
                train_ic=train_ic,
                penalty=penalty,
                ic_valid_periods=ic.rebalance_summary.valid_periods,
                industry_neutralized=neutralize,
                neutralization_skipped_dates=skipped_dates,
                neutralization_skipped_rate=skipped_rate,
                neutralization_skipped_details=skipped_details,
            )
            self.cache.put(key, result)
            return RewardEvaluation(result=result, cache_hit=False)

        direction = infer_long_direction(train_ic)
        long_series = long_portfolio_series(
            values,
            self.forward_returns,
            ic.rebalance_indices,
            direction,
            self.evaluation_config,
            industry_labels=self.industry_labels,
            universe_mask=self.universe_mask,
            neutralize_industry=neutralize,
            neutralization_diagnostics=diagnostics,
        )
        long_summary = summarize_excess_returns(
            long_series.excess_return[ic.rebalance_indices],
            self.evaluation_config,
        )
        candidate_ls = long_short_portfolio_series(
            values,
            self.forward_returns,
            ic.rebalance_indices,
            self.evaluation_config,
            industry_labels=self.industry_labels,
            universe_mask=self.universe_mask,
            neutralize_industry=neutralize,
            neutralization_diagnostics=diagnostics,
        )
        penalty = calculate_barra_ts_corr(
            candidate_ls.long_short_return,
            self.barra_long_short,
            min_periods=self.reward_config.barra_min_common_periods,
        )
        skipped_dates, skipped_rate, skipped_details = self._neutralization_summary(
            diagnostics,
            ic.rebalance_indices,
        )
        result = combine_reward_components(
            expression_hash,
            train_ic,
            long_summary.annualized_ir,
            penalty,
            self.reward_config,
            long_direction=direction,
            ic_valid_periods=ic.rebalance_summary.valid_periods,
            long_ir_valid_periods=long_summary.valid_periods,
            neutralization_skipped_dates=skipped_dates,
            neutralization_skipped_rate=skipped_rate,
            neutralization_skipped_details=skipped_details,
        )
        self.cache.put(key, result)
        return RewardEvaluation(result=result, cache_hit=False)


__all__ = [
    "RewardCache",
    "RewardEvaluation",
    "RewardEvaluator",
    "RewardResult",
    "combine_reward_components",
]
