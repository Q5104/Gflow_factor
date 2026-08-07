import unittest

import numpy as np
import pandas as pd

from factor_gfn.data.masks import (
    FEATURE_COLUMNS,
    apply_feature_valid_mask,
    build_feature_validity,
    build_universe_eligibility,
    combine_masks,
    is_current_st_name,
)


class CurrentStMaskTests(unittest.TestCase):
    def test_only_supported_st_prefixes_are_matched(self):
        names = pd.Series(["ST甲", "*ST乙", "S*ST丙", "SST丁", "STAR科技", "正常股份"])
        result = is_current_st_name(names).tolist()
        self.assertEqual(result, [True, True, True, True, False, False])

    def test_current_st_and_listing_age_are_separate_eligibility_reasons(self):
        keys = pd.DataFrame(
            {
                "trade_date": [
                    "2020-06-28",  # 距 2020-01-01 179 天
                    "2020-06-29",  # 正好 180 天
                    "2020-06-29",
                    "2020-06-29",
                    "2020-06-29",
                ],
                "stock_code": ["1", "1", "2", "3", "4"],
            }
        )
        stocks = pd.DataFrame(
            {
                "stock_code": ["000001", "000002", "000003", "000004"],
                "short_name": ["正常股份", "ST风险", "缺日期", pd.NA],
                "list_date": ["2020-01-01", "2010-01-01", pd.NaT, "2010-01-01"],
            }
        )

        result = build_universe_eligibility(keys, stocks, min_listing_days=180)
        self.assertEqual(result["listing_age_eligible"].tolist(), [False, True, True, False, True])
        self.assertEqual(result["is_current_st"].tolist(), [False, False, True, False, False])
        self.assertEqual(result["universe_eligible"].tolist(), [False, True, False, False, False])


class FeatureValidityTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-02", periods=5),
                "stock_code": ["000001"] * 5,
                "open": [10.0, 10.0, 10.0, 10.0, 10.0],
                "high": [11.0, 11.0, 10.4, 11.0, np.inf],
                "low": [9.0, 9.0, 9.0, 9.0, 9.0],
                "close": [10.5, 10.5, 10.5, 10.5, 10.5],
                "vwap": [10.2, 10.2, 10.2, 11.5, 10.2],
                "volume": [100.0, 0.0, 100.0, 100.0, 100.0],
            }
        )

    def test_invalid_reasons_and_common_feature_mask(self):
        validity = build_feature_validity(self.frame)
        self.assertEqual(validity["feature_valid"].tolist(), [True, False, False, False, False])
        self.assertFalse(validity.loc[1, "volume_positive"])
        self.assertFalse(validity.loc[2, "ohlc_consistent"])
        self.assertFalse(validity.loc[3, "vwap_in_range"])
        self.assertFalse(validity.loc[4, "prices_finite"])

        cleaned = apply_feature_valid_mask(self.frame, validity)
        self.assertTrue(cleaned.loc[0, FEATURE_COLUMNS].notna().all())
        self.assertTrue(cleaned.loc[1:, FEATURE_COLUMNS].isna().all(axis=None))

    def test_universe_ineligibility_does_not_clear_valid_features(self):
        one_row = self.frame.iloc[[0]].copy()
        validity = build_feature_validity(one_row)
        stocks = pd.DataFrame(
            {
                "stock_code": ["000001"],
                "short_name": ["ST风险"],
                "list_date": ["2000-01-01"],
            }
        )
        eligibility = build_universe_eligibility(one_row, stocks)
        combined = combine_masks(validity, eligibility)

        self.assertTrue(combined.loc[0, "feature_valid"])
        self.assertFalse(combined.loc[0, "universe_eligible"])
        self.assertFalse(combined.loc[0, "usable_mask"])
        self.assertTrue(one_row[FEATURE_COLUMNS].notna().all(axis=None))


if __name__ == "__main__":
    unittest.main()

