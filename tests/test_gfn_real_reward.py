import unittest

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
    def _provider(self, *, cache_max_entries: int = 8) -> RealRewardProvider:
        return RealRewardProvider(
            _context(),
            RewardConfig(
                barra_min_common_periods=5,
                candidate_industry_neutralization=True,
            ),
            cache_max_entries=cache_max_entries,
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
        self.assertEqual(set(result["barra_correlations"]), set(STYLE_NAMES))
        self.assertEqual(result["ic_valid_periods"], 13)
        self.assertEqual(provider.interpreter_evaluation_count, 1)
        self.assertEqual(len(provider.evaluation_records), 1)

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
