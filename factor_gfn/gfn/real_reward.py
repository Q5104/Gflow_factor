"""Real-data RewardProvider connecting expressions to the existing Trainer."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from factor_gfn.evaluator import FactorInterpreter, SUBEXPRESSION_CACHE_SCHEMA
from factor_gfn.evaluator.numba_kernels import NUMBA_KERNEL_SCHEMA, warm_numba_kernels
from factor_gfn.grammar import Expression

from .config import RewardConfig
from .real_data import RealRewardDataContext
from .reward import RewardCache, RewardEvaluator, RewardResult
from .trainer import RewardAssignment, RewardProvider


REAL_REWARD_PROVIDER_SCHEMA = "factor_gfn.real_reward_provider.v7"
# The first real 50-step profile produced only 3 hits for 6,425 lookups while
# evicting 6,417 full-history matrices.  Keep the optional implementation for
# targeted workloads, but do not pay its hashing and memory-churn cost in the
# default candidate search.
DEFAULT_SUBEXPRESSION_CACHE_MAX_BYTES = 0
DEFAULT_REAL_REWARD_CONFIG = RewardConfig(candidate_industry_neutralization=True)


@dataclass(frozen=True, slots=True)
class RealRewardEvaluationRecord:
    """One actual factor calculation; provider cache hits do not add records."""

    expression_hash: str
    formula: str
    prefix_token_ids: tuple[int, ...]
    node_count: int
    depth: int
    finite_universe_count: int
    universe_count: int
    finite_universe_coverage: float
    factor_seconds: float
    reward_seconds: float
    result: RewardResult


class RealRewardProvider(RewardProvider):
    """Evaluate complete expressions on the frozen real-data context.

    The provider-level LRU is checked before invoking ``FactorInterpreter``.
    This is deliberately separate from ``RewardEvaluator``'s result cache,
    because the latter receives an already-computed factor matrix.
    """

    def __init__(
        self,
        context: RealRewardDataContext,
        reward_config: RewardConfig = DEFAULT_REAL_REWARD_CONFIG,
        *,
        cache_max_entries: int = 50_000,
        subexpression_cache_max_bytes: int = DEFAULT_SUBEXPRESSION_CACHE_MAX_BYTES,
    ) -> None:
        if not isinstance(context, RealRewardDataContext):
            raise TypeError("context 必须是 RealRewardDataContext")
        if not isinstance(reward_config, RewardConfig):
            raise TypeError("reward_config 必须是 RewardConfig")
        if not reward_config.candidate_industry_neutralization:
            raise ValueError("真实 RewardProvider 必须启用候选因子行业中性化")
        if (
            isinstance(cache_max_entries, bool)
            or not isinstance(cache_max_entries, int)
            or cache_max_entries < 1
        ):
            raise ValueError("cache_max_entries 必须是正整数")
        if (
            isinstance(subexpression_cache_max_bytes, bool)
            or not isinstance(subexpression_cache_max_bytes, int)
            or subexpression_cache_max_bytes < 0
        ):
            raise ValueError("subexpression_cache_max_bytes 必须是非负整数")
        if context.rebalance_indices.size == 0:
            raise ValueError("真实 Reward 上下文的全局调仓日历不能为空")

        rows = np.asarray(context.evaluation_factor_rows, dtype=np.int64)
        if rows.ndim != 1 or rows.size != context.forward_returns.shape[0]:
            raise ValueError("evaluation_factor_rows 与 Reward 日期轴不一致")
        if (rows < 0).any() or (rows >= context.factor_tensor.shape[0]).any():
            raise IndexError("evaluation_factor_rows 包含越界日期")
        if np.unique(rows).size != rows.size or not np.all(rows[:-1] < rows[1:]):
            raise ValueError("evaluation_factor_rows 必须唯一且严格升序")
        if context.universe_mask.shape != context.forward_returns.shape:
            raise ValueError("universe_mask 与 forward_returns 未对齐")
        if context.industry_labels.shape != context.forward_returns.shape:
            raise ValueError("industry_labels 与 forward_returns 未对齐")

        self.context = context
        self.reward_config = reward_config
        self.cache_max_entries = cache_max_entries
        self.subexpression_cache_max_bytes = subexpression_cache_max_bytes
        self._interpreter = FactorInterpreter(
            context.factor_tensor,
            subexpression_cache_max_bytes=subexpression_cache_max_bytes,
        )
        self._numba_warmup_seconds = warm_numba_kernels()
        industry_fingerprint = str(
            context.manifest["sources"]["industry_metadata_sha256"]
        )
        self._evaluator = RewardEvaluator(
            context.forward_returns,
            context.barra_long_short,
            data_fingerprint=context.fingerprint,
            evaluation_config=context.config.evaluation,
            reward_config=reward_config,
            universe_mask=context.universe_mask,
            industry_labels=context.industry_labels,
            industry_fingerprint=industry_fingerprint,
            rebalance_indices=context.rebalance_indices,
            evaluation_dates=context.evaluation_dates,
            cache=RewardCache(max_entries=cache_max_entries),
        )
        self._cache: OrderedDict[str, RewardAssignment] = OrderedDict()
        self._records: list[RealRewardEvaluationRecord] = []
        self._request_count = 0
        self._cache_hit_count = 0
        self._interpreter_evaluation_count = 0

        calendar_hash = hashlib.sha256(
            np.asarray(context.rebalance_indices, dtype=np.int64).tobytes()
        ).hexdigest()
        self._manifest: dict[str, Any] = {
            "schema": REAL_REWARD_PROVIDER_SCHEMA,
            "context_fingerprint": context.fingerprint,
            "reward_evaluator_context_fingerprint": self._evaluator.context_fingerprint,
            "evaluation_config": asdict(context.config.evaluation),
            "reward_config": asdict(reward_config),
            "calendar": {
                "sha256": calendar_hash,
                "periods": int(context.rebalance_indices.size),
                "first_date": str(
                    context.evaluation_dates[context.rebalance_indices[0]]
                ),
                "last_date": str(
                    context.evaluation_dates[context.rebalance_indices[-1]]
                ),
            },
            "reward_panel": {
                "mode": "fixed_rebalance_compact",
                "history_rows_interpreted": int(context.forward_returns.shape[0]),
                "evaluation_rows": int(context.rebalance_indices.size),
                "candidate_cleaning_calls_per_evaluation": 1,
            },
            "interpreter": {
                "numeric_kernel_schema": NUMBA_KERNEL_SCHEMA,
                "numba_kernel_schema": NUMBA_KERNEL_SCHEMA,
                "numba_pre_warmed": True,
                "input_storage_mode": (
                    "borrowed_read_only_float64"
                    if self._interpreter.borrows_input_data
                    else "owned_normalized_copy"
                ),
                "leaf_views_are_read_only": True,
                "returned_factor_is_independent": True,
            },
            "industry_neutralization": {
                "enabled": True,
                "policy_schema": "factor_gfn.strict_industry_neutralization.v1",
                "encoding_schema": "factor_gfn.point_in_time_industry_codes.v1",
                "projection": "group_mean_equivalent_to_intercept_plus_full_dummies",
                "failed_date_action": "exclude_entire_candidate_cross_section",
                "unknown_industry_stock_action": "exclude_stock",
                "calendar_action": "keep_global_phase_without_backfill",
                "audit_fields": [
                    "date",
                    "row_index",
                    "factor_valid_count",
                    "known_industry_count",
                    "industry_count",
                    "required_regression_count",
                    "reason",
                ],
            },
            "cache": {
                "scope": "provider_context_plus_expression_structural_hash",
                "max_entries": cache_max_entries,
                "factor_matrix_cached": False,
                "subexpression": {
                    "schema": SUBEXPRESSION_CACHE_SCHEMA,
                    "scope": "interpreter_instance_plus_expression_structural_hash",
                    "max_bytes": subexpression_cache_max_bytes,
                    "enabled": subexpression_cache_max_bytes > 0,
                    "eviction": "lru",
                    "leaf_cached": False,
                    "root_cached": False,
                    "cached_arrays_read_only": True,
                    "checkpointed": False,
                    "resume_state": "empty",
                },
            },
        }
        payload = json.dumps(
            self._manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return deepcopy(self._manifest)

    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def cache_hit_count(self) -> int:
        return self._cache_hit_count

    @property
    def interpreter_evaluation_count(self) -> int:
        return self._interpreter_evaluation_count

    @property
    def evaluation_records(self) -> tuple[RealRewardEvaluationRecord, ...]:
        return tuple(deepcopy(self._records))

    def cache_info(self) -> dict[str, Any]:
        return {
            "entries": len(self._cache),
            "max_entries": self.cache_max_entries,
            "requests": self._request_count,
            "hits": self._cache_hit_count,
            "interpreter_evaluations": self._interpreter_evaluation_count,
            "subexpression": self._interpreter.subexpression_cache_info(),
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        self._evaluator.cache.clear()
        self._interpreter.clear_subexpression_cache()

    @staticmethod
    def _with_cache_flag(
        assignment: RewardAssignment, *, cache_hit: bool
    ) -> RewardAssignment:
        metadata = deepcopy(assignment.metadata) if assignment.metadata is not None else {}
        metadata["provider_cache_hit"] = cache_hit
        return RewardAssignment(
            valid=assignment.valid,
            reward=assignment.reward,
            log_reward=assignment.log_reward,
            rejection_reason=assignment.rejection_reason,
            metadata=metadata,
        )

    def _put_cache(self, key: str, assignment: RewardAssignment) -> None:
        self._cache[key] = deepcopy(assignment)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_max_entries:
            self._cache.popitem(last=False)

    def _evaluation_factor_view(self, values: np.ndarray) -> np.ndarray:
        rows = self.context.evaluation_factor_rows
        if rows.size and np.all(np.diff(rows) == 1):
            return values[int(rows[0]) : int(rows[-1]) + 1]
        return values[rows]

    def evaluate(self, expression: Expression) -> RewardAssignment:
        if not isinstance(expression, Expression):
            raise TypeError("expression 必须是 Expression")
        self._request_count += 1
        expression_hash = expression.structural_hash()
        cached = self._cache.get(expression_hash)
        if cached is not None:
            self._cache.move_to_end(expression_hash)
            self._cache_hit_count += 1
            return self._with_cache_flag(cached, cache_hit=True)

        subexpression_before = self._interpreter.subexpression_cache_info()
        factor_started = perf_counter()
        full_factor = self._interpreter.evaluate(expression)
        self._interpreter_evaluation_count += 1
        factor = self._evaluation_factor_view(full_factor)
        if factor.shape != self.context.forward_returns.shape:
            raise ValueError("解释器因子矩阵与真实 Reward 日期/股票轴不一致")
        factor_seconds = perf_counter() - factor_started
        subexpression_after = self._interpreter.subexpression_cache_info()

        universe = np.asarray(self.context.universe_mask, dtype=bool)
        universe_count = int(universe.sum())
        finite_universe_count = int(np.sum(universe & np.isfinite(factor)))
        coverage = (
            finite_universe_count / universe_count if universe_count else math.nan
        )

        reward_started = perf_counter()
        evaluation = self._evaluator.evaluate(expression_hash, factor)
        reward_seconds = perf_counter() - reward_started
        result = evaluation.result

        metadata: dict[str, Any] = {
            "schema": "factor_gfn.real_reward_assignment.v2",
            "provider_fingerprint": self._fingerprint,
            "provider_cache_hit": False,
            "reward_evaluator_cache_hit": evaluation.cache_hit,
            "expression_hash": expression_hash,
            "formula": expression.to_formula(),
            "prefix_token_ids": list(expression.to_prefix()),
            "node_count": expression.stats.node_count,
            "depth": expression.stats.depth,
            "finite_universe_count": finite_universe_count,
            "universe_count": universe_count,
            "finite_universe_coverage": coverage,
            "factor_seconds": factor_seconds,
            "reward_seconds": reward_seconds,
            "subexpression_cache_hits": (
                subexpression_after["hits"] - subexpression_before["hits"]
            ),
            "subexpression_cache_misses": (
                subexpression_after["misses"] - subexpression_before["misses"]
            ),
            "subexpression_cache_evictions": (
                subexpression_after["evictions"]
                - subexpression_before["evictions"]
            ),
            "subexpression_cache_entries": subexpression_after["entries"],
            "subexpression_cache_current_bytes": subexpression_after[
                "current_bytes"
            ],
            "subexpression_cache_max_bytes": subexpression_after["max_bytes"],
            "reward_result": asdict(result),
        }
        if result.valid:
            assignment = RewardAssignment(
                valid=True,
                reward=result.reward,
                log_reward=result.log_reward,
                metadata=metadata,
            )
        else:
            assignment = RewardAssignment(
                valid=False,
                rejection_reason=result.invalid_reason,
                metadata=metadata,
            )
        self._records.append(
            RealRewardEvaluationRecord(
                expression_hash=expression_hash,
                formula=expression.to_formula(),
                prefix_token_ids=expression.to_prefix(),
                node_count=expression.stats.node_count,
                depth=expression.stats.depth,
                finite_universe_count=finite_universe_count,
                universe_count=universe_count,
                finite_universe_coverage=coverage,
                factor_seconds=factor_seconds,
                reward_seconds=reward_seconds,
                result=deepcopy(result),
            )
        )
        self._put_cache(expression_hash, assignment)
        return self._with_cache_flag(assignment, cache_hit=False)


__all__ = [
    "DEFAULT_SUBEXPRESSION_CACHE_MAX_BYTES",
    "DEFAULT_REAL_REWARD_CONFIG",
    "REAL_REWARD_PROVIDER_SCHEMA",
    "RealRewardEvaluationRecord",
    "RealRewardProvider",
]
