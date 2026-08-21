import gc
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factor_gfn.barra import STYLE_NAMES, BarraConfig
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.gfn.real_data import (
    ExpressionFeatureSpec,
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
            json.dumps(
                {
                    "schema": "test.processed.v1",
                    "feature_order": [
                        "open", "high", "low", "close", "vwap", "volume"
                    ],
                }
            ),
            encoding="utf-8",
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

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    def _with_derived_tensor(
        self,
        paths: RealRewardDataPaths,
        tensor: np.ndarray,
        *,
        directory_name: str = "daily_derived_v1",
    ) -> RealRewardDataPaths:
        derived = paths.processed_dir / directory_name
        derived.mkdir()
        np.save(derived / "data_tensor.npy", tensor.astype(np.float32), allow_pickle=False)
        spec = ExpressionFeatureSpec.daily_derived(derived)
        metadata = {
            "status": "completed",
            "feature_space_id": spec.feature_space_id,
            "feature_order": list(spec.ordered_feature_names),
            "feature_count": len(spec.ordered_feature_names),
            "shape": list(tensor.shape),
            "builder_schema_fingerprint": spec.expected_schema_fingerprint,
            "axes": {
                "date": {"sha256": self._sha256(paths.date_list_path)},
                "stock": {"sha256": self._sha256(paths.stock_list_path)},
            },
        }
        (derived / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return RealRewardDataPaths(
            processed_dir=paths.processed_dir,
            industry_path=paths.industry_path,
            industry_metadata_path=paths.industry_metadata_path,
            expression_features=spec,
        )

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

    def test_derived_expression_tensor_keeps_raw_labels_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, config = self._prepare_files(Path(temporary))
            dates = np.load(paths.date_list_path, allow_pickle=False)
            stocks = np.load(paths.stock_list_path, allow_pickle=False)
            first_tensor = np.arange(
                dates.size * 16 * stocks.size, dtype=np.float32
            ).reshape(dates.size, 16, stocks.size)
            first_tensor[5, 1, 2] = np.nan
            second_tensor = np.full_like(first_tensor, -123.0)
            first_paths = self._with_derived_tensor(paths, first_tensor)
            second_paths = self._with_derived_tensor(
                paths, second_tensor, directory_name="daily_derived_v1_second"
            )

            raw = build_real_reward_data_context(config, paths)
            first = build_real_reward_data_context(config, first_paths)
            second = build_real_reward_data_context(config, second_paths)

            np.testing.assert_array_equal(first.forward_returns, raw.forward_returns)
            np.testing.assert_array_equal(second.forward_returns, raw.forward_returns)
            self.assertEqual(first.factor_tensor.shape, (40, 16, 4))
            np.testing.assert_equal(first.factor_tensor[5, 1, 2], np.nan)
            self.assertEqual(
                first.ordered_feature_names[1],
                "ret_cc1",
            )
            self.assertEqual(first.expression_feature_space_id, "daily_derived_v1")
            self.assertNotEqual(first.fingerprint, raw.fingerprint)
            del raw, first, second
            gc.collect()

    def test_derived_schema_and_axes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths, config = self._prepare_files(Path(temporary))
            dates = np.load(paths.date_list_path, allow_pickle=False)
            stocks = np.load(paths.stock_list_path, allow_pickle=False)
            tensor = np.ones((dates.size, 16, stocks.size), dtype=np.float32)
            derived_paths = self._with_derived_tensor(paths, tensor)
            metadata_path = derived_paths.expression_metadata_path
            original = json.loads(metadata_path.read_text(encoding="utf-8"))

            wrong_order = dict(original)
            wrong_order["feature_order"] = list(reversed(original["feature_order"]))
            metadata_path.write_text(json.dumps(wrong_order), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature order"):
                build_real_reward_data_context(config, derived_paths)

            wrong_axis = dict(original)
            wrong_axis["axes"] = {
                **original["axes"],
                "date": {"sha256": "0" * 64},
            }
            metadata_path.write_text(json.dumps(wrong_axis), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "date axis fingerprint"):
                build_real_reward_data_context(config, derived_paths)

            wrong_fingerprint = dict(original)
            wrong_fingerprint["builder_schema_fingerprint"] = "0" * 64
            metadata_path.write_text(json.dumps(wrong_fingerprint), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema fingerprint"):
                build_real_reward_data_context(config, derived_paths)

            metadata_path.write_text(json.dumps(original), encoding="utf-8")
            wrong_count = np.ones((dates.size, 15, stocks.size), dtype=np.float32)
            np.save(derived_paths.expression_tensor_path, wrong_count, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "feature count"):
                build_real_reward_data_context(config, derived_paths)

            wrong_date_axis = np.ones(
                (dates.size - 1, 16, stocks.size), dtype=np.float32
            )
            date_metadata = dict(original)
            date_metadata["shape"] = list(wrong_date_axis.shape)
            metadata_path.write_text(json.dumps(date_metadata), encoding="utf-8")
            np.save(
                derived_paths.expression_tensor_path,
                wrong_date_axis,
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "Raw date/stock"):
                build_real_reward_data_context(config, derived_paths)


if __name__ == "__main__":
    unittest.main()
