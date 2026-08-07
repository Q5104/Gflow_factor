import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factor_gfn.evaluator import (
    EvaluationConfig,
    build_forward_returns,
    evaluate_rank_ic,
    excess_return_correlation,
    factor_cross_sectional_correlation,
    infer_long_direction,
    long_portfolio_series,
    rank_ic_series,
    select_rebalance_indices,
    summarize_correlation,
    summarize_excess_returns,
    summarize_ic,
)


def _one_industry(values: np.ndarray) -> np.ndarray:
    return np.full(values.shape[-1], "测试行业", dtype=object)


class ForwardReturnTests(unittest.TestCase):
    def test_confirmed_open_t1_to_open_t6_label(self):
        prices = np.arange(1.0, 13.0)[:, None] * np.asarray([[1.0, 2.0]])
        original = prices.copy()
        result = build_forward_returns(prices)

        expected = np.full_like(prices, np.nan)
        expected[:6] = prices[6:] / prices[1:7] - 1.0
        np.testing.assert_allclose(result, expected, equal_nan=True)
        np.testing.assert_array_equal(prices, original)

    def test_invalid_and_incomplete_prices_return_nan(self):
        prices = np.ones((10, 2))
        prices[1, 0] = 0.0
        prices[6, 1] = np.nan
        result = build_forward_returns(prices)
        self.assertTrue(np.isnan(result[0]).all())
        self.assertTrue(np.isnan(result[-6:]).all())


class RankICTests(unittest.TestCase):
    def test_explicit_rebalance_indices_are_not_shifted_by_factor_warmup(self):
        stock_count = 30
        factor = np.tile(np.linspace(-1.0, 1.0, stock_count), (12, 1))
        returns = factor.copy()
        factor[:4] = np.nan
        fixed = np.array([1, 6, 11], dtype=np.int64)

        result = evaluate_rank_ic(
            factor,
            returns,
            EvaluationConfig(min_cross_section_count=10),
            neutralize_industry=False,
            rebalance_indices=fixed,
        )

        np.testing.assert_array_equal(result.rebalance_indices, fixed)
        self.assertTrue(np.isnan(result.rebalance_values[0]))
        self.assertAlmostEqual(result.rebalance_values[1], 1.0)
        self.assertAlmostEqual(result.rebalance_values[2], 1.0)

    def setUp(self):
        self.config = EvaluationConfig(min_cross_section_count=3)

    def test_positive_negative_constant_and_insufficient_cross_sections(self):
        factor = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, np.nan, np.nan, 4.0],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        returns = np.asarray(
            [
                [10.0, 20.0, 30.0, 40.0],
                [40.0, 30.0, 20.0, 10.0],
                [10.0, 20.0, 30.0, 40.0],
                [10.0, 20.0, 30.0, 40.0],
                [10.0, 20.0, 30.0, 40.0],
            ]
        )
        result = rank_ic_series(
            factor,
            returns,
            min_count=3,
            industry_labels=_one_industry(factor),
        )

        np.testing.assert_allclose(result.values[:2], [1.0, -1.0])
        self.assertTrue(np.isnan(result.values[2:]).all())
        np.testing.assert_array_equal(result.sample_count, [4, 4, 4, 2, 0])
        np.testing.assert_allclose(result.coverage, [1.0, 1.0, 1.0, 0.5, 0.0])

    def test_matches_scipy_and_pandas_reference(self):
        rng = np.random.default_rng(17)
        factor = rng.normal(size=(12, 30))
        returns = rng.normal(size=(12, 30))
        factor[3, :4] = np.nan
        result = rank_ic_series(
            factor,
            returns,
            min_count=10,
            industry_labels=_one_industry(factor),
        )

        for date_index in range(12):
            valid = np.isfinite(factor[date_index]) & np.isfinite(returns[date_index])
            scipy_value = spearmanr(
                factor[date_index, valid], returns[date_index, valid]
            ).statistic
            pandas_value = pd.Series(factor[date_index]).corr(
                pd.Series(returns[date_index]), method="spearman"
            )
            self.assertAlmostEqual(result.values[date_index], scipy_value)
            self.assertAlmostEqual(result.values[date_index], pandas_value)

    def test_future_factor_mutation_cannot_change_past_ic(self):
        rng = np.random.default_rng(23)
        factor = rng.normal(size=(20, 30))
        returns = rng.normal(size=(20, 30))
        before = rank_ic_series(
            factor,
            returns,
            min_count=20,
            industry_labels=_one_industry(factor),
        ).values
        factor[12:] = rng.normal(loc=1_000.0, size=factor[12:].shape)
        after = rank_ic_series(
            factor,
            returns,
            min_count=20,
            industry_labels=_one_industry(factor),
        ).values
        np.testing.assert_allclose(before[:12], after[:12], equal_nan=True)

    def test_rebalance_offset_is_relative_to_first_evaluable_date(self):
        counts = np.asarray([0, 2, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20])
        config = EvaluationConfig(min_cross_section_count=20)
        np.testing.assert_array_equal(select_rebalance_indices(counts, config), [3, 8])

        shifted = EvaluationConfig(min_cross_section_count=20, rebalance_offset=1)
        np.testing.assert_array_equal(
            select_rebalance_indices(counts, shifted), [4, 9]
        )

    def test_reward_ic_is_non_overlapping_but_daily_ic_is_retained(self):
        rng = np.random.default_rng(29)
        factor = rng.normal(size=(16, 25))
        returns = factor + rng.normal(scale=0.1, size=factor.shape)
        result = evaluate_rank_ic(
            factor,
            returns,
            self.config,
            industry_labels=_one_industry(factor),
        )

        self.assertEqual(result.daily.values.shape, (16,))
        np.testing.assert_array_equal(result.rebalance_indices, [0, 5, 10, 15])
        np.testing.assert_allclose(
            result.rebalance_values, result.daily.values[[0, 5, 10, 15]]
        )

    def test_ic_summary_uses_sample_standard_deviation(self):
        values = np.asarray([0.1, 0.2, 0.4, np.nan])
        result = summarize_ic(values, ddof=1)
        self.assertAlmostEqual(result.std, np.std([0.1, 0.2, 0.4], ddof=1))
        self.assertAlmostEqual(result.icir, np.mean([0.1, 0.2, 0.4]) / result.std)
        self.assertEqual(result.valid_periods, 3)


class LongPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.config = EvaluationConfig(min_cross_section_count=10)

    def test_direction_is_training_only_and_does_not_modify_factor(self):
        factor = np.tile(np.arange(20.0), (6, 1))
        returns = np.tile(np.arange(20.0)[::-1] / 100.0, (6, 1))
        original = factor.copy()
        direction = infer_long_direction(-0.2)
        result = long_portfolio_series(
            factor,
            returns,
            np.asarray([0, 5]),
            direction,
            self.config,
            industry_labels=_one_industry(factor),
        )

        self.assertEqual(direction, -1)
        self.assertEqual(result.long_count[0], 2)
        self.assertAlmostEqual(result.long_return[0], np.mean([0.19, 0.18]))
        self.assertAlmostEqual(result.benchmark_return[0], returns[0].mean())
        self.assertAlmostEqual(
            result.excess_return[0],
            result.long_return[0] - result.benchmark_return[0],
        )
        np.testing.assert_array_equal(factor, original)

    def test_constant_factor_and_missing_period_return_nan(self):
        factor = np.ones((6, 20))
        returns = np.ones((6, 20))
        result = long_portfolio_series(
            factor,
            returns,
            np.asarray([0, 5]),
            1,
            self.config,
            industry_labels=_one_industry(factor),
        )
        self.assertTrue(np.isnan(result.excess_return[[0, 5]]).all())

    def test_annualized_ir_uses_sqrt_periods_per_year_and_ddof_one(self):
        excess = np.asarray([0.01, 0.02, 0.03, np.nan])
        result = summarize_excess_returns(excess, self.config)
        expected_std = np.std([0.01, 0.02, 0.03], ddof=1)
        self.assertAlmostEqual(result.std, expected_std)
        self.assertAlmostEqual(result.annualized_return, 0.02 * (252.0 / 5.0))
        self.assertAlmostEqual(
            result.annualized_ir,
            0.02 / expected_std * np.sqrt(252.0 / 5.0),
        )


class CorrelationTests(unittest.TestCase):
    def test_factor_cross_sectional_correlation_and_summary(self):
        left = np.tile(np.arange(20.0), (3, 1))
        right = np.vstack((left[0], -left[0], np.ones(20)))
        series = factor_cross_sectional_correlation(
            left,
            right,
            min_count=10,
            industry_labels=_one_industry(left),
        )
        np.testing.assert_allclose(series.values[:2], [1.0, -1.0])
        self.assertTrue(np.isnan(series.values[2]))
        summary = summarize_correlation(series.values)
        self.assertAlmostEqual(summary.mean, 0.0)
        self.assertAlmostEqual(summary.mean_absolute, 1.0)

    def test_excess_return_correlation_uses_joint_finite_periods(self):
        left = np.asarray([0.1, np.nan, 0.2, 0.3])
        right = np.asarray([0.2, 0.8, 0.4, 0.6])
        self.assertAlmostEqual(excess_return_correlation(left, right), 1.0)
        self.assertTrue(
            np.isnan(excess_return_correlation(left, right, min_periods=4))
        )


if __name__ == "__main__":
    unittest.main()
