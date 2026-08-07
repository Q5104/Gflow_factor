import json
import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factor_gfn.barra import STYLE_NAMES, BarraConfig
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.gfn.real_data import (
    RealRewardDataConfig,
    RealRewardDataPaths,
    build_global_rebalance_indices,
    build_real_reward_data_context,
)


class GlobalRewardCalendarTests(unittest.TestCase):
    def test_calendar_starts_when_all_styles_are_evaluable(self) -> None:
        returns = np.ones((20, 4), dtype=np.float64)
        universe = np.ones_like(returns, dtype=bool)
        exposures = {
            name: np.tile(np.arange(4, dtype=np.float64), (20, 1))
            for name in STYLE_NAMES
        }
        exposures["market_beta"][:3] = np.nan
        config = RealRewardDataConfig(
            train_start="2020-01-01",
            train_end="2020-12-31",
            evaluation=EvaluationConfig(
                rebalance_interval=5,
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

        indices = build_global_rebalance_indices(
            exposures, returns, universe, config
        )

        np.testing.assert_array_equal(indices, np.array([3, 8, 13, 18]))
        self.assertFalse(indices.flags.writeable)

    def test_calendar_requires_exactly_five_styles(self) -> None:
        values = np.ones((10, 3))
        with self.assertRaisesRegex(ValueError, "五个第一版 Barra"):
            build_global_rebalance_indices(
                {"size": values}, values, values.astype(bool)
            )


class RealRewardDataContextTests(unittest.TestCase):
    def _prepare_files(self, root: Path) -> tuple[RealRewardDataPaths, RealRewardDataConfig]:
        processed = root / "processed"
        barra = processed / "barra"
        processed.mkdir()
        barra.mkdir()

        dates = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-02-15"),
            dtype="datetime64[D]",
        )
        stocks = np.array(["000001", "000002", "600000", "600001"])
        day = np.arange(dates.size, dtype=np.float64)[:, None]
        stock = np.arange(1, stocks.size + 1, dtype=np.float64)[None, :]
        open_prices = 20.0 + 0.15 * day + stock * (1.0 + 0.002 * day * day)
        tensor = np.empty((dates.size, 6, stocks.size), dtype=np.float32)
        for feature in range(6):
            tensor[:, feature, :] = open_prices + feature * 0.01
        universe = np.ones((dates.size, stocks.size), dtype=bool)
        np.save(processed / "data_tensor.npy", tensor, allow_pickle=False)
        np.save(processed / "universe_mask.npy", universe, allow_pickle=False)
        np.save(processed / "date_list.npy", dates, allow_pickle=False)
        np.save(processed / "stock_list.npy", stocks, allow_pickle=False)
        (processed / "metadata.json").write_text(
            json.dumps({"schema": "test.processed.v1"}), encoding="utf-8"
        )

        for style_index, name in enumerate(STYLE_NAMES):
            exposure = (
                stock
                + 0.4 * np.sin(day / (2.0 + style_index)) * ((-1.0) ** stock)
            ).astype(np.float32)
            if name == "market_beta":
                exposure[:3] = np.nan
            np.save(barra / f"{name}.npy", exposure, allow_pickle=False)
        np.save(barra / "market_return.npy", np.linspace(0.0, 0.01, dates.size))
        (barra / "metadata.json").write_text(
            json.dumps({"schema": "test.barra.v1"}), encoding="utf-8"
        )

        industry_rows = []
        for date in dates:
            for stock_code, code in zip(stocks, [801780, 801780, 801120, 801120]):
                industry_rows.append(
                    {
                        "trade_date": pd.Timestamp(date),
                        "stock_code": stock_code,
                        "sw_code_1": str(code),
                    }
                )
        industry_path = processed / "industry_sw_daily.parquet"
        pd.DataFrame(industry_rows).to_parquet(industry_path, index=False)
        industry_metadata = processed / "industry_sw_daily_metadata.json"
        industry_metadata.write_text(
            json.dumps({"schema": "test.industry.v1"}), encoding="utf-8"
        )

        paths = RealRewardDataPaths(
            processed_dir=processed,
            industry_path=industry_path,
            industry_metadata_path=industry_metadata,
        )
        config = RealRewardDataConfig(
            train_start="2020-01-01",
            train_end="2020-02-09",
            evaluation=EvaluationConfig(
                rebalance_interval=5,
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
        return paths, config

    def test_context_is_aligned_global_and_excludes_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, config = self._prepare_files(Path(temporary))

            context = build_real_reward_data_context(config, paths)

            self.assertEqual(context.factor_tensor.shape, (40, 6, 4))
            self.assertEqual(context.forward_returns.shape, (40, 4))
            self.assertEqual(str(context.evaluation_dates[-1]), "2020-02-09")
            self.assertEqual(int(context.rebalance_indices[0]), 3)
            self.assertEqual(int(context.rebalance_indices[-1]), 33)
            self.assertTrue(
                np.isfinite(
                    context.forward_returns[context.rebalance_indices]
                ).any(axis=1).all()
            )
            self.assertEqual(set(context.barra_long_short), set(STYLE_NAMES))
            self.assertTrue(context.manifest["calendar"]["rule"].startswith("global_"))
            self.assertEqual(
                context.manifest["industry"]["universe_missing_count"], 0
            )
            self.assertEqual(len(context.fingerprint), 64)
            self.assertTrue(np.isnan(context.forward_returns[-6:]).all())
            del context
            gc.collect()

    def test_fingerprint_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, config = self._prepare_files(Path(temporary))
            first = build_real_reward_data_context(config, paths)
            second = build_real_reward_data_context(config, paths)
            self.assertEqual(first.fingerprint, second.fingerprint)
            del first, second
            gc.collect()


if __name__ == "__main__":
    unittest.main()
