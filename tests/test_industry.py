import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factor_gfn.data.industry import (
    SWIND_INPUT_COLUMNS,
    IndustryBuildConfig,
    build_sw_industry_daily,
    load_sw_industry_panel,
)


def _industry_row(
    trade_date: str,
    stock_code: str,
    stock_name: str,
    suffix: str = "SZ",
) -> dict[str, str]:
    return {
        "TradingDay": trade_date,
        "StockCode": f"{stock_code}.{suffix}",
        "StockName": stock_name,
        "SWCode1": "801780.SI",
        "SWName1": "银行",
        "SWCode2": "801192.SI",
        "SWName2": "银行Ⅱ",
        "SWCode3": "851911.SI",
        "SWName3": "银行Ⅲ",
    }


def _write_source(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows, columns=SWIND_INPUT_COLUMNS).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


class PointInTimeIndustryPipelineTests(unittest.TestCase):
    def _config(self, root: Path) -> IndustryBuildConfig:
        source_dir = root / "swind"
        source_dir.mkdir()
        market = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]
                ),
                "stock_code": ["000001", "000004", "000001", "600000"],
            }
        )
        market_path = root / "daily_clean.parquet"
        market.to_parquet(market_path, index=False)
        return IndustryBuildConfig(
            source_dir=source_dir,
            market_keys_path=market_path,
            output_path=root / "industry_sw_daily.parquet",
            metadata_path=root / "industry_sw_daily_metadata.json",
        )

    def test_build_aligns_to_market_keys_and_keeps_all_three_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            _write_source(
                config.source_dir / "swind_20200102.csv",
                [
                    _industry_row("2020-01-02", "000001", "平安银行"),
                    _industry_row("2020-01-02", "000999", "范围外股票"),
                ],
            )
            _write_source(
                config.source_dir / "swind_20200103.csv",
                [
                    _industry_row("2020-01-03", "000001", "平安银行"),
                    _industry_row("2020-01-03", "600000", "浦发银行", suffix="SH"),
                ],
            )

            metadata = build_sw_industry_daily(config)
            output = pd.read_parquet(config.output_path)

            self.assertEqual(len(output), 4)
            self.assertEqual(int(output.duplicated(["trade_date", "stock_code"]).sum()), 0)
            self.assertEqual(output["stock_code"].tolist(), ["000001", "000004", "000001", "600000"])
            self.assertEqual(output.loc[0, "sw_code_1"], "801780")
            self.assertEqual(output.loc[0, "sw_code_2"], "801192")
            self.assertEqual(output.loc[0, "sw_code_3"], "851911")
            self.assertTrue(pd.isna(output.loc[1, "sw_code_1"]))
            self.assertEqual(metadata["output_summary"]["row_count"], 4)
            self.assertEqual(metadata["output_summary"]["missing_level1"], 1)

            panel1 = load_sw_industry_panel(
                np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
                np.array(["000001", "000004", "600000"]),
                path=config.output_path,
            )
            panel2 = load_sw_industry_panel(
                np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
                np.array(["000001", "000004", "600000"]),
                level=2,
                path=config.output_path,
            )
            np.testing.assert_array_equal(
                panel1,
                np.array([[801780, -1, -1], [801780, -1, 801780]], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                panel2,
                np.array([[801192, -1, -1], [801192, -1, 801192]], dtype=np.int32),
            )

    def test_duplicate_normalized_source_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            _write_source(
                config.source_dir / "swind_20200102.csv",
                [
                    _industry_row("2020-01-02", "000001", "平安银行", suffix="SZ"),
                    _industry_row("2020-01-02", "000001", "重复", suffix="SH"),
                ],
            )
            _write_source(
                config.source_dir / "swind_20200103.csv",
                [_industry_row("2020-01-03", "000001", "平安银行")],
            )

            with self.assertRaisesRegex(ValueError, "duplicate_normalized_keys"):
                build_sw_industry_daily(config)

    def test_invalid_stock_suffix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            bad = _industry_row("2020-01-02", "000001", "平安银行")
            bad["StockCode"] = "000001.XX"
            _write_source(config.source_dir / "swind_20200102.csv", [bad])
            _write_source(
                config.source_dir / "swind_20200103.csv",
                [_industry_row("2020-01-03", "000001", "平安银行")],
            )

            with self.assertRaisesRegex(ValueError, "invalid_stock_code"):
                build_sw_industry_daily(config)


if __name__ == "__main__":
    unittest.main()
