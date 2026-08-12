import unittest
import warnings

import numpy as np

from factor_gfn.barra import STYLE_NAMES, BarraConfig, LongShortSeries
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.gfn import (
    GFNConfig,
    GFNTrainer,
    RealRewardDataConfig,
    RealRewardDataContext,
    RealRewardProvider,
    RewardConfig,
)
from factor_gfn.grammar import Expression, get_action_id


def _series(values: np.ndarray) -> LongShortSeries:
    zeros = np.zeros(values.shape, dtype=np.int64)
    return LongShortSeries(
        long_return=np.full(values.shape, np.nan),
        short_return=np.full(values.shape, np.nan),
        long_short_return=values.copy(),
        universe_count=zeros.copy(),
        leg_count=zeros.copy(),
    )


def _context() -> RealRewardDataContext:
    rng = np.random.default_rng(20260807)
    date_count, stock_count = 80, 30
    dates = np.arange(
        np.datetime64("2010-01-01"),
        np.datetime64("2010-03-22"),
        dtype="datetime64[D]",
    )
    stocks = np.array([f"{index:06d}" for index in range(stock_count)])
    groups = np.repeat(np.arange(3), 10)
    within = np.tile(np.linspace(-1.0, 1.0, 10), 3)
    day = np.arange(date_count, dtype=np.float64)[:, None]
    close = 100.0 + groups[None, :] * 20.0 + within[None, :] * 5.0 + day * 0.01
    tensor = np.empty((date_count, 6, stock_count), dtype=np.float64)
    for feature in range(6):
        tensor[:, feature, :] = close + feature * 0.1

    scale = 0.015 + 0.005 * np.sin(np.arange(date_count) / 4.0)
    returns = within[None, :] * scale[:, None]
    returns += rng.normal(0.0, 0.001, returns.shape)
    universe = np.ones(returns.shape, dtype=bool)
    industries = np.broadcast_to(
        np.array([801010, 801020, 801030], dtype=np.int32).repeat(10),
        returns.shape,
    ).copy()
    rebalance = np.arange(10, 75, 5, dtype=np.int64)
    references = {
        name: _series(rng.normal(0.0, 0.02, date_count)) for name in STYLE_NAMES
    }
    evaluation = EvaluationConfig(
        rebalance_interval=5,
        min_cross_section_count=10,
        long_quantile=0.10,
    )
    barra = BarraConfig(
        long_short_quantile=0.10,
        min_cross_section_count=10,
        min_common_periods=5,
    )
    config = RealRewardDataConfig(
        train_start="2010-01-01",
        train_end="2010-03-21",
        evaluation=evaluation,
        barra=barra,
    )
    return RealRewardDataContext(
        factor_tensor=tensor,
        config=config,
        history_dates=dates,
        evaluation_factor_rows=np.arange(date_count, dtype=np.int64),
        evaluation_dates=dates,
        stocks=stocks,
        forward_returns=returns,
        universe_mask=universe,
        industry_labels=industries,
        rebalance_indices=rebalance,
        barra_long_short=references,
        fingerprint="a" * 64,
        manifest={
            "sources": {"industry_metadata_sha256": "b" * 64},
        },
    )


class RealRewardProviderTests(unittest.TestCase):
    def _provider(
        self,
        *,
        cache_max_entries: int = 8,
        subexpression_cache_max_bytes: int = 512 * 1024**2,
    ) -> RealRewardProvider:
        return RealRewardProvider(
            _context(),
            RewardConfig(
                barra_min_common_periods=5,
                candidate_industry_neutralization=True,
            ),
            cache_max_entries=cache_max_entries,
            subexpression_cache_max_bytes=subexpression_cache_max_bytes,
        )

    def test_valid_expression_returns_full_decomposition(self) -> None:
        provider = self._provider()
        expression = Expression.from_prefix([get_action_id("close")])

        assignment = provider.evaluate(expression)

        self.assertTrue(assignment.valid)
        self.assertGreater(assignment.reward, 0.0)
        self.assertFalse(assignment.metadata["provider_cache_hit"])
        result = assignment.metadata["reward_result"]
        self.assertTrue(result["industry_neutralized"])
        self.assertEqual(result["neutralization_skipped_dates"], ())
        self.assertEqual(result["neutralization_skipped_rate"], 0.0)
        self.assertEqual(result["neutralization_skipped_details"], ())
        self.assertEqual(set(result["barra_correlations"]), set(STYLE_NAMES))
        self.assertEqual(result["ic_valid_periods"], 13)
        self.assertEqual(provider.interpreter_evaluation_count, 1)
        self.assertEqual(len(provider.evaluation_records), 1)
        policy = provider.manifest()["industry_neutralization"]
        self.assertEqual(
            policy["policy_schema"],
            "factor_gfn.strict_industry_neutralization.v1",
        )
        self.assertEqual(
            policy["failed_date_action"],
            "exclude_entire_candidate_cross_section",
        )
        self.assertEqual(policy["unknown_industry_stock_action"], "exclude_stock")
        self.assertIn("reason", policy["audit_fields"])
        reward_panel = provider.manifest()["reward_panel"]
        self.assertEqual(reward_panel["mode"], "fixed_rebalance_compact")
        self.assertEqual(reward_panel["history_rows_interpreted"], 80)
        self.assertEqual(reward_panel["evaluation_rows"], 13)
        self.assertEqual(reward_panel["candidate_cleaning_calls_per_evaluation"], 1)
        interpreter = provider.manifest()["interpreter"]
        self.assertEqual(
            interpreter["numeric_kernel_schema"],
            "factor_gfn.numba_cpu_loops.v2",
        )
        self.assertEqual(
            interpreter["numba_kernel_schema"],
            "factor_gfn.numba_cpu_loops.v2",
        )
        self.assertTrue(interpreter["numba_pre_warmed"])
        self.assertEqual(interpreter["input_storage_mode"], "owned_normalized_copy")
        self.assertTrue(interpreter["leaf_views_are_read_only"])
        self.assertTrue(interpreter["returned_factor_is_independent"])
        subexpression = provider.manifest()["cache"]["subexpression"]
        self.assertEqual(
            subexpression["schema"],
            "factor_gfn.subexpression_lru.v1",
        )
        self.assertEqual(subexpression["max_bytes"], 512 * 1024**2)
        self.assertTrue(subexpression["enabled"])
        self.assertEqual(subexpression["eviction"], "lru")
        self.assertFalse(subexpression["leaf_cached"])
        self.assertFalse(subexpression["root_cached"])
        self.assertFalse(subexpression["checkpointed"])

    def test_shared_subexpression_cache_is_audited_per_candidate(self) -> None:
        provider = self._provider()
        shared = get_action_id("ts_mean", 5)
        first = Expression.from_prefix(
            (
                get_action_id("add"),
                shared,
                get_action_id("close"),
                get_action_id("high"),
            )
        )
        second = Expression.from_prefix(
            (
                get_action_id("sub"),
                shared,
                get_action_id("close"),
                get_action_id("low"),
            )
        )

        first_assignment = provider.evaluate(first)
        second_assignment = provider.evaluate(second)

        self.assertEqual(first_assignment.metadata["subexpression_cache_hits"], 0)
        self.assertEqual(first_assignment.metadata["subexpression_cache_misses"], 1)
        self.assertEqual(second_assignment.metadata["subexpression_cache_hits"], 1)
        self.assertEqual(second_assignment.metadata["subexpression_cache_misses"], 0)
        self.assertGreater(
            second_assignment.metadata["subexpression_cache_current_bytes"],
            0,
        )
        self.assertEqual(
            provider.cache_info()["subexpression"]["hits"],
            1,
        )

    def test_provider_cache_prevents_second_interpreter_call(self) -> None:
        provider = self._provider()
        expression = Expression.from_prefix([get_action_id("close")])

        first = provider.evaluate(expression)
        second = provider.evaluate(expression)

        self.assertFalse(first.metadata["provider_cache_hit"])
        self.assertTrue(second.metadata["provider_cache_hit"])
        self.assertEqual(first.reward, second.reward)
        self.assertEqual(provider.request_count, 2)
        self.assertEqual(provider.cache_hit_count, 1)
        self.assertEqual(provider.interpreter_evaluation_count, 1)
        self.assertEqual(len(provider.evaluation_records), 1)

    def test_failed_dates_persist_complete_neutralization_audit(self) -> None:
        context = _context()
        stock_count = context.industry_labels.shape[1]
        context.industry_labels[:] = np.arange(stock_count, dtype=np.int32)
        provider = RealRewardProvider(
            context,
            RewardConfig(
                barra_min_common_periods=5,
                candidate_industry_neutralization=True,
            ),
        )
        expression = Expression.from_prefix([get_action_id("close")])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assignment = provider.evaluate(expression)

        result = assignment.metadata["reward_result"]
        self.assertFalse(assignment.valid)
        self.assertEqual(
            len(result["neutralization_skipped_details"]),
            context.rebalance_indices.size,
        )
        detail = result["neutralization_skipped_details"][0]
        self.assertEqual(
            detail["date"],
            str(context.evaluation_dates[context.rebalance_indices[0]]),
        )
        self.assertEqual(detail["factor_valid_count"], stock_count)
        self.assertEqual(detail["known_industry_count"], stock_count)
        self.assertEqual(detail["industry_count"], stock_count)
        self.assertEqual(detail["required_regression_count"], stock_count + 1)
        self.assertEqual(
            detail["reason"],
            "insufficient_industry_regression_samples",
        )

    def test_divergent_lru_orders_do_not_break_evaluation(self) -> None:
        provider = self._provider(cache_max_entries=2)
        close = Expression.from_prefix([get_action_id("close")])
        opened = Expression.from_prefix([get_action_id("open")])
        high = Expression.from_prefix([get_action_id("high")])

        provider.evaluate(close)
        provider.evaluate(opened)
        provider.evaluate(close)  # 只刷新 Provider LRU 顺序
        provider.evaluate(high)
        repeated = provider.evaluate(opened)

        self.assertFalse(repeated.metadata["provider_cache_hit"])
        self.assertTrue(repeated.metadata["reward_evaluator_cache_hit"])
        self.assertEqual(provider.interpreter_evaluation_count, 4)

    def test_constant_expression_is_rejected_without_reward_floor(self) -> None:
        provider = self._provider()
        close = get_action_id("close")
        expression = Expression.from_prefix([get_action_id("sub"), close, close])

        assignment = provider.evaluate(expression)

        self.assertFalse(assignment.valid)
        self.assertIsNone(assignment.reward)
        self.assertIsNone(assignment.log_reward)
        self.assertIn("train_ic", assignment.rejection_reason)
        self.assertFalse(assignment.metadata["reward_result"]["valid"])

    def test_real_provider_refuses_disabled_industry_neutralization(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须启用"):
            RealRewardProvider(
                _context(),
                RewardConfig(candidate_industry_neutralization=False),
            )

    def test_fingerprint_changes_with_reward_configuration(self) -> None:
        context = _context()
        first = RealRewardProvider(
            context,
            RewardConfig(
                long_ir_lambda=0.3,
                barra_min_common_periods=5,
                candidate_industry_neutralization=True,
            ),
        )
        second = RealRewardProvider(
            context,
            RewardConfig(
                long_ir_lambda=0.4,
                barra_min_common_periods=5,
                candidate_industry_neutralization=True,
            ),
        )
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_trainer_rejects_mismatched_reward_configuration(self) -> None:
        provider = self._provider()
        with self.assertRaisesRegex(ValueError, "reward_config 不一致"):
            GFNTrainer(GFNConfig(), provider)


if __name__ == "__main__":
    unittest.main()
