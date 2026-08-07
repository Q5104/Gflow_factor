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
    EvaluationConfig,
    evaluate_rank_ic,
    infer_long_direction,
    long_portfolio_series,
    long_short_portfolio_series,
    summarize_excess_returns,
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
        if universe_mask is not None and np.asarray(universe_mask).shape != returns.shape:
            raise ValueError("universe_mask 必须与 forward_returns 同形")

        self.forward_returns = returns
        self.barra_long_short = dict(barra_long_short)
        self.evaluation_config = evaluation_config
        self.reward_config = reward_config
        self.universe_mask = universe_mask
        self.industry_labels = industry_labels
        self.cache = cache if cache is not None else RewardCache()
        manifest = {
            "schema": "factor_gfn.reward_context.v1",
            "data_fingerprint": data_fingerprint,
            "industry_fingerprint": industry_fingerprint,
            "evaluation_config": asdict(evaluation_config),
            "reward_config": asdict(reward_config),
            "styles": list(STYLE_NAMES),
        }
        payload = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        self.context_fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
        neutralize = self.reward_config.candidate_industry_neutralization
        ic = evaluate_rank_ic(
            values,
            self.forward_returns,
            self.evaluation_config,
            industry_labels=self.industry_labels,
            universe_mask=self.universe_mask,
            neutralize_industry=neutralize,
        )
        train_ic = ic.rebalance_summary.mean
        if not np.isfinite(train_ic) or train_ic == 0.0:
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
        )
        penalty = calculate_barra_ts_corr(
            candidate_ls.long_short_return,
            self.barra_long_short,
            min_periods=self.reward_config.barra_min_common_periods,
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
