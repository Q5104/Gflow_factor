from __future__ import annotations

import unittest

import numpy as np

from factor_gfn.backtest.rolling_icir import (
    RollingICIRConfig,
    estimate_rolling_weights,
    generate_causal_rolling_scores,
)


class RollingICIRTests(unittest.TestCase):
    def test_weight_estimator_shrinks_caps_and_normalizes(self) -> None:
        history = np.column_stack(
            (
                np.linspace(0.01, 0.10, 10),
                np.linspace(0.01, 0.02, 10),
                np.linspace(-0.10, -0.01, 10),
            )
        )
        config = RollingICIRConfig(
            window_observations=10,
            min_observations=5,
            shrinkage_to_equal=0.5,
            max_weight=0.6,
            min_cross_section_count=2,
        )
        weights, diagnostics = estimate_rolling_weights(history, config)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue((weights >= 0).all())
        self.assertLessEqual(float(weights.max()), 0.6 + 1e-12)
        self.assertGreater(float(weights[2]), 0.0)  # equal-weight shrinkage remains
        self.assertFalse(diagnostics["fallback_status"])

    def test_future_labels_cannot_change_earlier_scores(self) -> None:
        stock_count = 20
        oos_date_count = 8
        unique_dates = np.arange(
            np.datetime64("2021-01-04"),
            np.datetime64("2021-01-04") + oos_date_count,
        )
        dates = np.repeat(unique_dates, stock_count)
        cross_section = np.tile(np.arange(stock_count, dtype=float), oos_date_count)
        values = np.column_stack((cross_section, -cross_section))
        labels = cross_section.copy()
        seed_dates = np.arange(np.datetime64("2020-12-27"), np.datetime64("2021-01-04"))
        seed_ic = np.column_stack((np.linspace(0.1, 0.2, 8), np.linspace(-0.2, -0.1, 8)))
        config = RollingICIRConfig(
            window_observations=8,
            min_observations=4,
            update_every_periods=2,
            maturity_lag_periods=2,
            shrinkage_to_equal=0.0,
            max_weight=1.0,
            min_cross_section_count=20,
        )
        original = generate_causal_rolling_scores(
            dates,
            values,
            labels,
            aliases=("factor_000", "factor_001"),
            seed_dates=seed_dates,
            seed_ic_values=seed_ic,
            config=config,
        )
        changed_labels = labels.copy()
        changed_labels[dates == unique_dates[-1]] *= -1.0
        changed = generate_causal_rolling_scores(
            dates,
            values,
            changed_labels,
            aliases=("factor_000", "factor_001"),
            seed_dates=seed_dates,
            seed_ic_values=seed_ic,
            config=config,
        )
        np.testing.assert_allclose(original.scores, changed.scores)
        np.testing.assert_allclose(
            original.weights_by_update["weight"],
            changed.weights_by_update["weight"],
        )

    def test_maturity_lag_and_window_boundary_are_explicit(self) -> None:
        stock_count = 20
        unique_dates = np.arange(np.datetime64("2021-01-10"), np.datetime64("2021-01-15"))
        dates = np.repeat(unique_dates, stock_count)
        cross_section = np.tile(np.arange(stock_count, dtype=float), unique_dates.size)
        values = np.column_stack((cross_section, -cross_section))
        labels = cross_section.copy()
        seed_dates = np.arange(np.datetime64("2021-01-06"), np.datetime64("2021-01-10"))
        seed_ic = np.column_stack((np.full(4, 0.1), np.full(4, -0.1)))
        config = RollingICIRConfig(
            window_observations=4,
            min_observations=2,
            update_every_periods=2,
            maturity_lag_periods=2,
            shrinkage_to_equal=0.5,
            max_weight=0.8,
            min_cross_section_count=20,
        )
        result = generate_causal_rolling_scores(
            dates,
            values,
            labels,
            aliases=("factor_000", "factor_001"),
            seed_dates=seed_dates,
            seed_ic_values=seed_ic,
            config=config,
        )
        diagnostics = result.diagnostics_by_update.reset_index(drop=True)
        self.assertTrue(np.isnat(diagnostics.loc[0, "latest_mature_oos_date"].to_datetime64()))
        self.assertEqual(
            diagnostics.loc[1, "latest_mature_oos_date"].to_datetime64(),
            unique_dates[0],
        )
        self.assertTrue((diagnostics["history_period_count"] == 4).all())


if __name__ == "__main__":
    unittest.main()
