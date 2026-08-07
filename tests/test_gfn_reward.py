import unittest

import numpy as np

from factor_gfn.barra import BarraPenaltyResult, LongShortSeries, STYLE_NAMES
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.gfn import (
    RewardConfig,
    RewardEvaluator,
    combine_reward_components,
)


def _penalty(correlations: dict[str, float]) -> BarraPenaltyResult:
    return BarraPenaltyResult(
        barra_ts_corr=max(abs(value) for value in correlations.values()),
        correlations=correlations,
        valid_periods={name: 80 for name in STYLE_NAMES},
    )


def _series(values: np.ndarray) -> LongShortSeries:
    nan_values = np.full(values.shape, np.nan)
    zeros = np.zeros(values.shape, dtype=np.int64)
    return LongShortSeries(
        long_return=nan_values.copy(),
        short_return=nan_values.copy(),
        long_short_return=values.copy(),
        universe_count=zeros.copy(),
        leg_count=zeros.copy(),
    )


class RewardFormulaTests(unittest.TestCase):
    def test_formula_caps_ir_and_retains_signed_barra_correlations(self):
        correlations = {
            "market_beta": -0.40,
            "size": 0.10,
            "momentum": 0.20,
            "volatility": -0.05,
            "liquidity": 0.30,
        }
        result = combine_reward_components(
            "expr",
            train_ic=-0.05,
            train_long_ir=3.0,
            penalty=_penalty(correlations),
        )
        expected = 0.05 * (1.0 + 0.3 * 2.0) * (1.0 - 0.2 * 0.4)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.raw_reward, expected)
        self.assertAlmostEqual(result.reward, expected)
        self.assertEqual(result.barra_correlations, correlations)
        self.assertEqual(result.dominant_barra_factor, "market_beta")
        self.assertAlmostEqual(result.dominant_barra_correlation, -0.40)

    def test_floor_only_stabilizes_valid_zero_reward(self):
        correlations = {name: 0.0 for name in STYLE_NAMES}
        result = combine_reward_components(
            "zero",
            train_ic=0.0,
            train_long_ir=0.0,
            penalty=_penalty(correlations),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.raw_reward, 0.0)
        self.assertEqual(result.reward, 1e-8)
        self.assertTrue(result.floor_applied)
        self.assertAlmostEqual(result.log_reward, np.log(1e-8))

    def test_invalid_barra_sample_is_not_converted_to_floor(self):
        penalty = BarraPenaltyResult(
            barra_ts_corr=np.nan,
            correlations={name: np.nan for name in STYLE_NAMES},
            valid_periods={name: 20 for name in STYLE_NAMES},
        )
        result = combine_reward_components("invalid", 0.05, 1.0, penalty)
        self.assertFalse(result.valid)
        self.assertTrue(np.isnan(result.reward))
        self.assertFalse(result.floor_applied)


class RewardEvaluatorTests(unittest.TestCase):
    def test_explicit_no_industry_mode_runs_and_cache_hits(self):
        rng = np.random.default_rng(2026)
        date_count, stock_count = 24, 30
        factor = np.tile(np.linspace(-1.0, 1.0, stock_count), (date_count, 1))
        scales = np.linspace(0.01, 0.04, date_count)[:, None]
        returns = factor * scales + rng.normal(0.0, 0.005, factor.shape)
        references = {
            name: _series(rng.normal(0.0, 0.02, date_count)) for name in STYLE_NAMES
        }
        evaluator = RewardEvaluator(
            returns,
            references,
            data_fingerprint="synthetic-v1",
            evaluation_config=EvaluationConfig(
                rebalance_interval=1,
                min_cross_section_count=10,
            ),
            reward_config=RewardConfig(
                barra_min_common_periods=5,
                candidate_industry_neutralization=False,
            ),
        )

        first = evaluator.evaluate("factor-hash", factor)
        second = evaluator.evaluate("factor-hash", factor)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertTrue(first.result.valid)
        self.assertFalse(first.result.industry_neutralized)
        self.assertEqual(set(first.result.barra_correlations), set(STYLE_NAMES))
        self.assertEqual(first.result, second.result)

    def test_industry_mode_requires_labels_and_fingerprint(self):
        values = np.ones((10, 20))
        references = {name: _series(np.ones(10)) for name in STYLE_NAMES}
        with self.assertRaisesRegex(ValueError, "industry_labels"):
            RewardEvaluator(
                values,
                references,
                data_fingerprint="data-v1",
                reward_config=RewardConfig(candidate_industry_neutralization=True),
            )


if __name__ == "__main__":
    unittest.main()
