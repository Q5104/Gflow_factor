import unittest
import warnings
from unittest.mock import patch

import numpy as np

from factor_gfn.barra import BarraPenaltyResult, LongShortSeries, STYLE_NAMES
from factor_gfn.evaluator import (
    EvaluationConfig,
    evaluate_rank_ic,
    infer_long_direction,
    long_portfolio_series,
    long_short_portfolio_series,
    summarize_excess_returns,
)
from factor_gfn.barra import calculate_barra_ts_corr
from factor_gfn.evaluator.cross_section import clean_candidate_factor_cross_sections
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
    def test_fixed_calendar_cleans_only_compact_rows_once_and_matches_reference(self):
        rng = np.random.default_rng(20260810)
        date_count, stock_count = 10, 30
        factor = rng.normal(size=(date_count, stock_count))
        groups = np.repeat(np.arange(3), 10)
        factor += groups[None, :] * 0.25
        returns = 0.02 * factor + rng.normal(0.0, 0.01, factor.shape)
        industries = np.broadcast_to(groups, factor.shape).copy()
        universe = np.ones(factor.shape, dtype=bool)
        rebalance = np.array([1, 4, 7, 9], dtype=np.int64)
        dates = np.arange(
            np.datetime64("2010-01-01"),
            np.datetime64("2010-01-11"),
            dtype="datetime64[D]",
        )
        references = {
            name: _series(rng.normal(0.0, 0.02, date_count)) for name in STYLE_NAMES
        }
        config = EvaluationConfig(
            rebalance_interval=1,
            min_cross_section_count=10,
            long_quantile=0.10,
        )
        reward_config = RewardConfig(
            barra_min_common_periods=2,
            candidate_industry_neutralization=True,
        )
        evaluator = RewardEvaluator(
            returns,
            references,
            data_fingerprint="compact-panel-v1",
            evaluation_config=config,
            reward_config=reward_config,
            universe_mask=universe,
            industry_labels=industries,
            industry_fingerprint="industry-v1",
            rebalance_indices=rebalance,
            evaluation_dates=dates,
        )

        with patch(
            "factor_gfn.gfn.reward.clean_candidate_factor_cross_sections",
            wraps=clean_candidate_factor_cross_sections,
        ) as cleaner:
            optimized = evaluator.evaluate("compact-factor", factor).result

        self.assertEqual(cleaner.call_count, 1)
        self.assertEqual(cleaner.call_args.args[0].shape, (rebalance.size, stock_count))

        legacy_ic = evaluate_rank_ic(
            factor,
            returns,
            config,
            industry_labels=industries,
            universe_mask=universe,
            neutralize_industry=True,
            rebalance_indices=rebalance,
        )
        direction = infer_long_direction(legacy_ic.rebalance_summary.mean)
        legacy_long = long_portfolio_series(
            factor,
            returns,
            rebalance,
            direction,
            config,
            industry_labels=industries,
            universe_mask=universe,
            neutralize_industry=True,
        )
        legacy_long_summary = summarize_excess_returns(
            legacy_long.excess_return[rebalance],
            config,
        )
        legacy_ls = long_short_portfolio_series(
            factor,
            returns,
            rebalance,
            config,
            industry_labels=industries,
            universe_mask=universe,
            neutralize_industry=True,
        )
        legacy_penalty = calculate_barra_ts_corr(
            legacy_ls.long_short_return,
            references,
            min_periods=2,
        )

        self.assertAlmostEqual(optimized.train_ic, legacy_ic.rebalance_summary.mean)
        self.assertAlmostEqual(optimized.train_long_ir, legacy_long_summary.annualized_ir)
        self.assertAlmostEqual(optimized.barra_ts_corr, legacy_penalty.barra_ts_corr)
        self.assertEqual(optimized.ic_valid_periods, legacy_ic.rebalance_summary.valid_periods)
        self.assertEqual(optimized.long_ir_valid_periods, legacy_long_summary.valid_periods)
        for name in STYLE_NAMES:
            self.assertAlmostEqual(
                optimized.barra_correlations[name],
                legacy_penalty.correlations[name],
            )

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
        self.assertEqual(first.result.neutralization_skipped_dates, ())
        self.assertEqual(first.result.neutralization_skipped_rate, 0.0)
        self.assertEqual(first.result.neutralization_skipped_details, ())
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

        with self.assertRaisesRegex(ValueError, "evaluation_dates"):
            RewardEvaluator(
                values,
                references,
                data_fingerprint="data-v1",
                industry_labels=np.zeros(values.shape, dtype=np.int32),
                industry_fingerprint="industry-v1",
                reward_config=RewardConfig(candidate_industry_neutralization=True),
            )

    def test_neutralization_skips_are_deduplicated_and_persisted(self):
        date_count, stock_count = 6, 4
        factor = np.tile(np.arange(stock_count, dtype=np.float64), (date_count, 1))
        returns = factor * np.arange(1.0, date_count + 1.0)[:, None]
        reference_values = np.arange(date_count, dtype=np.float64)
        references = {name: _series(reference_values) for name in STYLE_NAMES}
        dates = np.arange(
            np.datetime64("2010-01-01"),
            np.datetime64("2010-01-07"),
            dtype="datetime64[D]",
        )
        rebalance = np.array([0, 2, 4, 5], dtype=np.int64)
        evaluator = RewardEvaluator(
            returns,
            references,
            data_fingerprint="synthetic-industry-v1",
            evaluation_config=EvaluationConfig(
                rebalance_interval=1,
                min_cross_section_count=2,
                long_quantile=0.25,
            ),
            reward_config=RewardConfig(
                barra_min_common_periods=2,
                candidate_industry_neutralization=True,
            ),
            industry_labels=np.arange(stock_count),
            industry_fingerprint="industry-v1",
            rebalance_indices=rebalance,
            evaluation_dates=dates,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = evaluator.evaluate("factor-industry", factor)
            second = evaluator.evaluate("factor-industry", factor)
            invalid = evaluator.evaluate("constant-industry", np.ones_like(factor))

        self.assertEqual(
            first.result.neutralization_skipped_dates,
            ("2010-01-01", "2010-01-03", "2010-01-05", "2010-01-06"),
        )
        self.assertEqual(first.result.neutralization_skipped_rate, 1.0)
        self.assertEqual(
            tuple(item["row_index"] for item in first.result.neutralization_skipped_details),
            (0, 2, 4, 5),
        )
        first_detail = first.result.neutralization_skipped_details[0]
        self.assertEqual(first_detail["date"], "2010-01-01")
        self.assertEqual(first_detail["factor_valid_count"], 4)
        self.assertEqual(first_detail["known_industry_count"], 4)
        self.assertEqual(first_detail["industry_count"], 4)
        self.assertEqual(first_detail["required_regression_count"], 5)
        self.assertEqual(
            first_detail["reason"],
            "insufficient_industry_regression_samples",
        )
        self.assertEqual(first.result, second.result)
        self.assertTrue(second.cache_hit)
        self.assertFalse(invalid.result.valid)
        self.assertEqual(
            invalid.result.neutralization_skipped_dates,
            first.result.neutralization_skipped_dates,
        )
        self.assertEqual(invalid.result.neutralization_skipped_rate, 1.0)
        self.assertEqual(
            invalid.result.neutralization_skipped_details,
            first.result.neutralization_skipped_details,
        )


if __name__ == "__main__":
    unittest.main()
