from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import factor_gfn.data.daily_derived_artifact as artifact_module
from factor_gfn.data import (
    DAILY_DERIVED_FEATURE_NAMES,
    DailyDerivedArtifactConfig,
    build_daily_derived_artifact,
    build_daily_derived_features,
    daily_derived_schema_contract,
    daily_derived_schema_fingerprint,
    inspect_daily_derived_artifact,
    inspect_daily_derived_inputs,
)


class DailyDerivedArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dates = np.arange(
            np.datetime64("2024-01-02"), np.datetime64("2024-01-10"), dtype="datetime64[D]"
        )
        self.stocks = np.array(["000001", "000002"])
        np.save(self.root / "date_list.npy", self.dates, allow_pickle=False)
        np.save(self.root / "stock_list.npy", self.stocks, allow_pickle=False)
        np.save(
            self.root / "universe_mask.npy",
            np.ones((self.dates.size, self.stocks.size), dtype=bool),
            allow_pickle=False,
        )
        self.shares = np.array(
            [
                [np.nan, np.nan],
                [10_000.0, np.nan],
                [10_000.0, np.nan],
                [20_000.0, np.nan],
                [20_000.0, np.nan],
                [20_000.0, np.nan],
                [20_000.0, np.nan],
                [20_000.0, np.nan],
            ],
            dtype=np.float32,
        )
        np.save(self.root / "list_a_shares.npy", self.shares, allow_pickle=False)
        (self.root / "barra_metadata.json").write_text(
            json.dumps(
                {
                    "schema": "factor_gfn.barra_five_style.v1",
                    "shape": [int(self.dates.size), int(self.stocks.size)],
                }
            ),
            encoding="utf-8",
        )

        market_rows = []
        raw_close_rows = []
        for date_index, date in enumerate(self.dates):
            for stock_index, stock in enumerate(self.stocks):
                close = 20.0 + date_index + 5.0 * stock_index
                open_ = close - 0.5
                high = close + 1.0
                low = close - 1.0
                volume = 1_000.0 + 100.0 * date_index + 50.0 * stock_index
                close_raw = close / 2.0
                amount_raw = volume * close_raw
                market_rows.append(
                    {
                        "trade_date": pd.Timestamp(date),
                        "stock_code": stock,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "amount": amount_raw,
                    }
                )
                raw_close_rows.append(
                    {
                        "trade_date": pd.Timestamp(date),
                        "stock_code": stock,
                        "close": close_raw,
                    }
                )
        self.market = pd.DataFrame(market_rows).sample(frac=1.0, random_state=7)
        self.raw_close = pd.DataFrame(raw_close_rows).sample(frac=1.0, random_state=11)
        self.market.to_parquet(self.root / "market.parquet", index=False)
        self.raw_close.to_parquet(self.root / "raw_close.parquet", index=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self, output_name: str, block_size: int = 2) -> DailyDerivedArtifactConfig:
        return DailyDerivedArtifactConfig(
            market_path=self.root / "market.parquet",
            raw_close_path=self.root / "raw_close.parquet",
            date_list_path=self.root / "date_list.npy",
            stock_list_path=self.root / "stock_list.npy",
            universe_mask_path=self.root / "universe_mask.npy",
            shares_path=self.root / "list_a_shares.npy",
            shares_metadata_path=self.root / "barra_metadata.json",
            output_dir=self.root / output_name,
            date_block_size=block_size,
        )

    def _direct_inputs(self) -> dict[str, np.ndarray]:
        ordered = self.market.sort_values(["trade_date", "stock_code"])
        shape = (self.dates.size, self.stocks.size)

        def matrix(column: str) -> np.ndarray:
            return ordered[column].to_numpy(dtype=np.float64).reshape(shape)

        return {
            "open_adjusted": matrix("open"),
            "high_adjusted": matrix("high"),
            "low_adjusted": matrix("low"),
            "close_adjusted": matrix("close"),
            "vwap_adjusted": matrix("close"),
            "volume_raw": matrix("volume"),
            "amount_raw": matrix("amount"),
            "list_a_shares": self.shares.astype(np.float64),
        }

    def test_preflight_validates_reused_shares_axes_and_missing_semantics(self) -> None:
        summary = inspect_daily_derived_inputs(self._config("preflight"))
        self.assertEqual(summary["shares_shape"], [8, 2])
        self.assertEqual(summary["shares_finite_count"], 7)
        self.assertEqual(summary["shares_missing_count"], 9)
        self.assertIn("change_date <= trade_date", summary["shares_reuse_contract"])

    def test_shuffled_real_rows_align_to_authority_axes_and_builder(self) -> None:
        config = self._config("aligned", block_size=2)
        metadata = build_daily_derived_artifact(config)
        actual = np.load(config.tensor_path, mmap_mode="r", allow_pickle=False)
        expected = build_daily_derived_features(**self._direct_inputs()).astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6, equal_nan=True)
        self.assertEqual(actual.shape, (8, 16, 2))
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(metadata["feature_order"], list(DAILY_DERIVED_FEATURE_NAMES))

    def test_reused_shares_preserve_effective_date_boundaries_and_missing(self) -> None:
        config = self._config("shares", block_size=3)
        build_daily_derived_artifact(config)
        tensor = np.load(config.tensor_path, mmap_mode="r", allow_pickle=False)
        turnover = tensor[:, DAILY_DERIVED_FEATURE_NAMES.index("turnover"), :]
        volume = self._direct_inputs()["volume_raw"]
        self.assertTrue(np.isnan(turnover[0, 0]))
        self.assertAlmostEqual(turnover[1, 0], volume[1, 0] / 10_000.0)
        self.assertAlmostEqual(turnover[2, 0], volume[2, 0] / 10_000.0)
        self.assertAlmostEqual(turnover[3, 0], volume[3, 0] / 20_000.0)
        self.assertTrue(np.isnan(turnover[:, 1]).all())

    def test_adjusted_vwap_and_raw_amount_are_preserved(self) -> None:
        config = self._config("amount", block_size=4)
        build_daily_derived_artifact(config)
        actual = np.load(config.tensor_path, mmap_mode="r", allow_pickle=False)
        expected = build_daily_derived_features(**self._direct_inputs()).astype(np.float32)
        for name in ("ret_close_vwap", "ret_open_vwap", "illiq", "ret_amt_chg5"):
            index = DAILY_DERIVED_FEATURE_NAMES.index(name)
            np.testing.assert_allclose(actual[:, index], expected[:, index], equal_nan=True)

    def test_date_blocks_with_overlap_equal_single_block(self) -> None:
        blocked = self._config("blocked", block_size=2)
        single = self._config("single", block_size=99)
        build_daily_derived_artifact(blocked)
        build_daily_derived_artifact(single)
        left = np.load(blocked.tensor_path, mmap_mode="r", allow_pickle=False)
        right = np.load(single.tensor_path, mmap_mode="r", allow_pickle=False)
        np.testing.assert_array_equal(left, right)

    def test_metadata_and_fingerprint_are_stable_and_contract_sensitive(self) -> None:
        first = daily_derived_schema_fingerprint()
        self.assertEqual(first, daily_derived_schema_fingerprint())
        contract = daily_derived_schema_contract()
        contract["feature_order"] = list(reversed(contract["feature_order"]))
        self.assertNotEqual(first, daily_derived_schema_fingerprint(contract))
        contract = daily_derived_schema_contract()
        contract["lag_contract"] = "different lag"
        self.assertNotEqual(first, daily_derived_schema_fingerprint(contract))
        contract = daily_derived_schema_contract()
        contract["shares_pit_contract"] = "different PIT"
        self.assertNotEqual(first, daily_derived_schema_fingerprint(contract))

        config = self._config("metadata")
        build_daily_derived_artifact(config)
        loaded = inspect_daily_derived_artifact(config)
        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(loaded["builder_schema_fingerprint"], first)
        self.assertIn("qa", loaded)

    def test_atomic_failure_leaves_no_formal_or_completed_artifact(self) -> None:
        config = self._config("failure", block_size=2)
        original = artifact_module.build_daily_derived_features
        calls = 0

        def fail_on_second_block(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic failure")
            return original(**kwargs)

        with mock.patch.object(
            artifact_module, "build_daily_derived_features", side_effect=fail_on_second_block
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                build_daily_derived_artifact(config)
        self.assertFalse(config.tensor_path.exists())
        self.assertFalse(config.metadata_path.exists())
        self.assertFalse((config.output_dir / "data_tensor.npy.tmp").exists())
        self.assertFalse((config.output_dir / "metadata.json.tmp").exists())

    def test_existing_formal_artifact_is_never_overwritten(self) -> None:
        config = self._config("no_overwrite")
        build_daily_derived_artifact(config)
        before = config.metadata_path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "不覆盖"):
            build_daily_derived_artifact(config)
        self.assertEqual(config.metadata_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
