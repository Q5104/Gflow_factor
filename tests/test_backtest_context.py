import unittest

import numpy as np

from factor_gfn.backtest.context import (
    Stage5DataConfig,
    build_stage5_context_from_arrays,
)
from factor_gfn.barra import STYLE_NAMES, BarraConfig
from factor_gfn.evaluator import EvaluationConfig


class Stage5ContextTests(unittest.TestCase):
    def _build_context(self):
        dates = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-02-15"),
            dtype="datetime64[D]",
        )
        stocks = np.array(["000001", "000002", "600000"])
        day = np.arange(dates.size, dtype=np.float64)[:, None]
        stock = np.arange(stocks.size, dtype=np.float64)[None, :]
        tensor = np.empty((dates.size, 2, stocks.size), dtype=np.float64)
        tensor[:, 0, :] = 10.0 + day + stock
        tensor[:, 1, :] = 20.0 + day + stock
        universe = np.ones((dates.size, stocks.size), dtype=bool)
        industry = np.tile(np.array([1, 1, 2], dtype=np.int32), (dates.size, 1))
        exposures = {
            name: np.tile(np.arange(stocks.size, dtype=np.float64), (dates.size, 1))
            for name in STYLE_NAMES
        }
        exposures["market_beta"][:2] = np.nan
        config = Stage5DataConfig(
            train_start="2020-01-01",
            train_end="2020-01-15",
            validation_start="2020-01-16",
            validation_end="2020-01-30",
            oos_start="2020-01-31",
            oos_end="2020-02-14",
            evaluation=EvaluationConfig(
                min_cross_section_count=2,
                long_quantile=0.25,
            ),
            barra=BarraConfig(
                beta_window=3,
                beta_min_periods=2,
                momentum_lookback=3,
                momentum_skip=1,
                volatility_window=3,
                volatility_min_periods=2,
                liquidity_window=2,
                long_short_quantile=0.25,
                min_cross_section_count=2,
                min_common_periods=2,
            ),
        )
        return build_stage5_context_from_arrays(
            dates=dates,
            stocks=stocks,
            factor_tensor=tensor,
            universe_mask=universe,
            industry_labels=industry,
            barra_exposures=exposures,
            config=config,
            source_manifest={"test_data": "v1"},
        )

    def test_shared_calendar_keeps_phase_and_records_exclusions(self) -> None:
        context = self._build_context()
        rows = np.array([entry.signal_row for entry in context.calendar])
        np.testing.assert_array_equal(np.diff(rows), np.full(rows.size - 1, 5))
        self.assertEqual(context.calendar[0].signal_row, 2)
        crossing = [
            entry for entry in context.calendar
            if "label_crosses_split" in entry.exclusion_reasons
        ]
        self.assertEqual([entry.signal_row for entry in crossing], [12, 27, 42])
        self.assertTrue(all(not entry.included for entry in crossing))
        self.assertEqual(context.manifest["calendar"]["anchor_date"], "2020-01-03")

    def test_labels_and_feature_history_are_split_isolated(self) -> None:
        context = self._build_context()
        train = context.get_split_data("train")
        validation = context.get_split_data("validation")
        self.assertTrue(np.isnan(train.forward_returns[-6:]).all())
        self.assertTrue(np.isnan(validation.forward_returns[-6:]).all())
        self.assertEqual(train.factor_tensor.shape[0], 15)
        self.assertEqual(validation.factor_tensor.shape[0], 30)
        np.testing.assert_array_equal(
            validation.evaluation_factor_rows, np.arange(15, 30)
        )
        self.assertEqual(str(validation.history_dates[0]), "2020-01-01")
        self.assertEqual(str(validation.evaluation_dates[0]), "2020-01-16")

    def test_oos_is_locked_until_selection_fingerprint_is_supplied(self) -> None:
        context = self._build_context()
        with self.assertRaises(PermissionError):
            context.get_split_data("oos")
        oos = context.get_split_data(
            "oos", frozen_selection_fingerprint="a" * 64
        )
        self.assertEqual(oos.boundary.name, "oos")
        self.assertEqual(context.manifest["oos"]["candidate_evaluation_count"], 0)

    def test_config_rejects_overlapping_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "互不重叠"):
            Stage5DataConfig(
                train_end="2020-01-10",
                validation_start="2020-01-10",
            )


if __name__ == "__main__":
    unittest.main()
