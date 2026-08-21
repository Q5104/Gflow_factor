from __future__ import annotations

import unittest

import numpy as np

from factor_gfn.data import (
    CLV_TOL,
    DAILY_DERIVED_FEATURE_NAMES,
    build_daily_derived_features,
)


FEATURE_INDEX = {name: index for index, name in enumerate(DAILY_DERIVED_FEATURE_NAMES)}


def _valid_inputs(dtype: np.dtype = np.dtype(np.float64)) -> dict[str, np.ndarray]:
    close = np.array(
        [
            [10.0, 20.0],
            [11.0, 21.0],
            [12.0, 22.0],
            [13.0, 23.0],
            [14.0, 24.0],
            [15.0, 25.0],
            [16.0, 26.0],
        ],
        dtype=dtype,
    )
    open_ = close - 1.0
    high = close + 1.0
    low = open_ - 1.0
    vwap = (open_ + close) / 2.0
    volume = np.array(
        [
            [100.0, 200.0],
            [110.0, 220.0],
            [120.0, 240.0],
            [130.0, 260.0],
            [140.0, 280.0],
            [150.0, 300.0],
            [160.0, 320.0],
        ],
        dtype=dtype,
    )
    amount = volume * 10.0
    shares = np.broadcast_to(np.array([10_000.0, 20_000.0], dtype=dtype), close.shape).copy()
    return {
        "open_adjusted": open_,
        "high_adjusted": high,
        "low_adjusted": low,
        "close_adjusted": close,
        "vwap_adjusted": vwap,
        "volume_raw": volume,
        "amount_raw": amount,
        "list_a_shares": shares,
    }


class DailyDerivedBuilderTests(unittest.TestCase):
    def test_feature_order_shape_and_float64_output(self) -> None:
        expected = (
            "ret_gap",
            "ret_cc1",
            "ret_co",
            "ret_hl",
            "ret_range",
            "ret_body",
            "ret_upper_shadow",
            "ret_lower_shadow",
            "ret_close_vwap",
            "ret_open_vwap",
            "ret_vol_chg1",
            "ret_vol_chg5",
            "turnover",
            "illiq",
            "ret_amt_chg5",
            "clv",
        )
        self.assertEqual(DAILY_DERIVED_FEATURE_NAMES, expected)

        result = build_daily_derived_features(**_valid_inputs(np.dtype(np.float32)))
        self.assertEqual(result.shape, (7, 16, 2))
        self.assertEqual(result.dtype, np.float64)

    def test_all_sixteen_formulas_match_hand_calculation(self) -> None:
        result = build_daily_derived_features(**_valid_inputs())
        expected = np.array(
            [
                14.0 / 14.0 - 1.0,
                15.0 / 14.0 - 1.0,
                15.0 / 14.0 - 1.0,
                16.0 / 13.0 - 1.0,
                3.0 / 14.0,
                1.0 / 14.0,
                1.0 / 14.0,
                1.0 / 14.0,
                15.0 / 14.5 - 1.0,
                14.0 / 14.5 - 1.0,
                150.0 / 140.0 - 1.0,
                150.0 / 100.0 - 1.0,
                150.0 / 10_000.0,
                1e8 * abs(15.0 / 14.0 - 1.0) / 1_500.0,
                1_500.0 / 1_000.0 - 1.0,
                (2.0 * 15.0 - 16.0 - 13.0) / (16.0 - 13.0),
            ]
        )
        np.testing.assert_allclose(result[5, :, 0], expected, rtol=1e-14, atol=1e-14)

    def test_lags_use_exact_axis_positions_and_do_not_skip_nan(self) -> None:
        inputs = _valid_inputs()
        inputs["close_adjusted"][2, 0] = np.nan
        inputs["volume_raw"][0, 0] = np.nan
        result = build_daily_derived_features(**inputs)

        lag1 = ["ret_gap", "ret_cc1", "ret_vol_chg1"]
        for name in lag1:
            self.assertTrue(np.isnan(result[0, FEATURE_INDEX[name], 0]))
        for name in ("ret_vol_chg5", "ret_amt_chg5"):
            self.assertTrue(np.isnan(result[:5, FEATURE_INDEX[name], 0]).all())

        self.assertTrue(np.isnan(result[3, FEATURE_INDEX["ret_gap"], 0]))
        self.assertTrue(np.isnan(result[3, FEATURE_INDEX["ret_cc1"], 0]))
        self.assertTrue(np.isfinite(result[4, FEATURE_INDEX["ret_cc1"], 0]))
        self.assertTrue(np.isnan(result[1, FEATURE_INDEX["ret_vol_chg1"], 0]))
        self.assertTrue(np.isfinite(result[2, FEATURE_INDEX["ret_vol_chg1"], 0]))
        self.assertTrue(np.isnan(result[5, FEATURE_INDEX["ret_vol_chg5"], 0]))

    def test_nan_is_feature_specific_for_each_input_family(self) -> None:
        cases = (
            ("open_adjusted", ("ret_gap", "ret_co", "ret_body"), ("ret_cc1", "ret_hl")),
            ("close_adjusted", ("ret_cc1", "ret_co", "ret_close_vwap", "illiq"), ("ret_gap", "ret_hl")),
            ("volume_raw", ("ret_vol_chg1", "ret_vol_chg5", "turnover"), ("ret_cc1", "ret_amt_chg5")),
            ("amount_raw", ("illiq", "ret_amt_chg5"), ("ret_cc1", "turnover")),
            ("list_a_shares", ("turnover",), ("ret_cc1", "illiq")),
            ("vwap_adjusted", ("ret_close_vwap", "ret_open_vwap"), ("ret_cc1", "turnover")),
        )
        for input_name, affected, unaffected in cases:
            with self.subTest(input_name=input_name):
                inputs = _valid_inputs()
                inputs[input_name][5, 0] = np.nan
                result = build_daily_derived_features(**inputs)
                for feature_name in affected:
                    self.assertTrue(np.isnan(result[5, FEATURE_INDEX[feature_name], 0]))
                for feature_name in unaffected:
                    self.assertTrue(np.isfinite(result[5, FEATURE_INDEX[feature_name], 0]))
                self.assertFalse(np.isnan(result[5, :, 0]).all())

    def test_infinite_dependencies_propagate_to_nan(self) -> None:
        inputs = _valid_inputs()
        inputs["vwap_adjusted"][5, 0] = np.inf
        inputs["list_a_shares"][5, 1] = -np.inf
        result = build_daily_derived_features(**inputs)
        self.assertTrue(np.isnan(result[5, FEATURE_INDEX["ret_close_vwap"], 0]))
        self.assertTrue(np.isnan(result[5, FEATURE_INDEX["ret_open_vwap"], 0]))
        self.assertTrue(np.isnan(result[5, FEATURE_INDEX["turnover"], 1]))
        self.assertTrue(np.isfinite(result[5, FEATURE_INDEX["ret_cc1"], :]).all())

    def test_nonpositive_denominators_fail_closed(self) -> None:
        cases = (
            ("open_adjusted", 5, "ret_co"),
            ("low_adjusted", 5, "ret_hl"),
            ("vwap_adjusted", 5, "ret_close_vwap"),
            ("vwap_adjusted", 5, "ret_open_vwap"),
            ("close_adjusted", 4, "ret_cc1"),
            ("close_adjusted", 4, "ret_gap"),
            ("volume_raw", 4, "ret_vol_chg1"),
            ("volume_raw", 0, "ret_vol_chg5"),
            ("list_a_shares", 5, "turnover"),
            ("amount_raw", 5, "illiq"),
            ("amount_raw", 0, "ret_amt_chg5"),
        )
        for input_name, row, feature_name in cases:
            with self.subTest(input_name=input_name, row=row, feature_name=feature_name):
                inputs = _valid_inputs()
                inputs[input_name][row, 0] = 0.0
                result = build_daily_derived_features(**inputs)
                self.assertTrue(np.isnan(result[5, FEATURE_INDEX[feature_name], 0]))

    def test_nonpositive_price_numerators_and_geometry_inputs_fail_closed(self) -> None:
        numerator_cases = (
            ("open_adjusted", ("ret_gap", "ret_open_vwap")),
            ("close_adjusted", ("ret_cc1", "ret_co", "ret_close_vwap", "illiq")),
            ("high_adjusted", ("ret_hl",)),
        )
        for input_name, feature_names in numerator_cases:
            for invalid_value in (0.0, -1.0):
                with self.subTest(
                    input_name=input_name,
                    invalid_value=invalid_value,
                ):
                    inputs = _valid_inputs()
                    inputs[input_name][5, 0] = invalid_value
                    result = build_daily_derived_features(**inputs)
                    for feature_name in feature_names:
                        self.assertTrue(np.isnan(result[5, FEATURE_INDEX[feature_name], 0]))

        geometry_features = (
            "ret_range",
            "ret_body",
            "ret_upper_shadow",
            "ret_lower_shadow",
            "clv",
        )
        for input_name in (
            "open_adjusted",
            "high_adjusted",
            "low_adjusted",
            "close_adjusted",
        ):
            with self.subTest(geometry_input=input_name):
                inputs = _valid_inputs()
                inputs[input_name][5, 0] = 0.0
                result = build_daily_derived_features(**inputs)
                for feature_name in geometry_features:
                    self.assertTrue(np.isnan(result[5, FEATURE_INDEX[feature_name], 0]))

    def test_kline_identity_holds_for_valid_geometry(self) -> None:
        result = build_daily_derived_features(**_valid_inputs())
        combined = (
            result[:, FEATURE_INDEX["ret_body"], :]
            + result[:, FEATURE_INDEX["ret_upper_shadow"], :]
            + result[:, FEATURE_INDEX["ret_lower_shadow"], :]
        )
        np.testing.assert_allclose(
            result[1:, FEATURE_INDEX["ret_range"], :],
            combined[1:],
            rtol=1e-14,
            atol=1e-14,
        )

    def test_invalid_geometry_only_fails_geometry_features(self) -> None:
        geometry_features = (
            "ret_range",
            "ret_body",
            "ret_upper_shadow",
            "ret_lower_shadow",
            "clv",
        )
        for invalid_field, invalid_value in (("high_adjusted", 14.0), ("low_adjusted", 14.5)):
            with self.subTest(invalid_field=invalid_field):
                inputs = _valid_inputs()
                inputs[invalid_field][5, 0] = invalid_value
                result = build_daily_derived_features(**inputs)
                for feature_name in geometry_features:
                    self.assertTrue(np.isnan(result[5, FEATURE_INDEX[feature_name], 0]))
                self.assertTrue(np.isfinite(result[5, FEATURE_INDEX["ret_cc1"], 0]))
                self.assertTrue(np.isfinite(result[5, FEATURE_INDEX["turnover"], 0]))

    def test_clv_boundaries_flat_range_and_rounding_tolerance(self) -> None:
        high = np.array([[12.0, 12.0, 12.0, 10.0, 0.67]])
        low = np.array([[10.0, 10.0, 10.0, 10.0, 0.03]])
        close = np.array([[12.0, 10.0, 11.0, 10.0, 0.03]])
        open_ = np.array([[11.0, 11.0, 11.0, 10.0, 0.59]])
        shape = high.shape
        raw_rounding_value = (2.0 * close[0, 4] - high[0, 4] - low[0, 4]) / (
            high[0, 4] - low[0, 4]
        )
        self.assertLess(raw_rounding_value, -1.0)
        self.assertLessEqual(abs(raw_rounding_value), 1.0 + CLV_TOL)

        result = build_daily_derived_features(
            open_adjusted=open_,
            high_adjusted=high,
            low_adjusted=low,
            close_adjusted=close,
            vwap_adjusted=np.full(shape, 10.0),
            volume_raw=np.full(shape, 100.0),
            amount_raw=np.full(shape, 1_000.0),
            list_a_shares=np.full(shape, 10_000.0),
        )
        clv = result[0, FEATURE_INDEX["clv"], :]
        np.testing.assert_array_equal(clv[:3], np.array([1.0, -1.0, 0.0]))
        self.assertTrue(np.isnan(clv[3]))
        self.assertEqual(clv[4], -1.0)

    def test_turnover_and_illiq_use_frozen_scale(self) -> None:
        result = build_daily_derived_features(**_valid_inputs())
        self.assertAlmostEqual(result[5, FEATURE_INDEX["turnover"], 0], 0.015)
        expected = 1e8 * abs(15.0 / 14.0 - 1.0) / 1_500.0
        self.assertAlmostEqual(result[5, FEATURE_INDEX["illiq"], 0], expected)

    def test_future_inputs_cannot_change_current_or_past_features(self) -> None:
        inputs = _valid_inputs()
        baseline = build_daily_derived_features(**inputs)
        changed = {name: values.copy() for name, values in inputs.items()}
        for values in changed.values():
            values[4:] *= 7.0
        altered = build_daily_derived_features(**changed)
        np.testing.assert_allclose(baseline[:4], altered[:4], equal_nan=True)

    def test_misaligned_or_non_matrix_input_fails_fast(self) -> None:
        inputs = _valid_inputs()
        inputs["amount_raw"] = np.ones((6, 2))
        with self.assertRaisesRegex(ValueError, "shape 必须一致"):
            build_daily_derived_features(**inputs)

        inputs = _valid_inputs()
        inputs["open_adjusted"] = np.ones(7)
        with self.assertRaisesRegex(ValueError, "必须是二维"):
            build_daily_derived_features(**inputs)


if __name__ == "__main__":
    unittest.main()
