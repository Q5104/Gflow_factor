import unittest
import warnings

import numpy as np

from factor_gfn.evaluator.cross_section import (
    IndustryNeutralizationWarning,
    clean_candidate_factor_cross_sections,
    clean_factor_cross_sections,
)


class CrossSectionalCleaningTests(unittest.TestCase):
    def test_winsorize_then_zscore_preserves_nan_and_mask(self):
        factor = np.array([[1.0, 2.0, 3.0, 1_000.0, np.nan]])
        universe = np.array([[True, True, True, True, False]])
        original = factor.copy()

        cleaned = clean_factor_cross_sections(factor, universe)

        self.assertTrue(np.isnan(cleaned[0, 4]))
        self.assertAlmostEqual(float(np.nanmean(cleaned[0])), 0.0)
        self.assertAlmostEqual(float(np.nanstd(cleaned[0], ddof=0)), 1.0)
        np.testing.assert_allclose(factor, original, equal_nan=True)

    def test_constant_cross_section_becomes_nan(self):
        cleaned = clean_factor_cross_sections(np.ones((2, 10)))
        self.assertTrue(np.isnan(cleaned).all())

    def test_candidate_cleaning_removes_industry_group_means(self):
        factor = np.array([[1.0, 3.0, 100.0, 110.0]])
        industries = np.array(["银行", "银行", "计算机", "计算机"], dtype=object)

        cleaned = clean_candidate_factor_cross_sections(factor, industries)

        self.assertAlmostEqual(float(cleaned[0, :2].mean()), 0.0)
        self.assertAlmostEqual(float(cleaned[0, 2:].mean()), 0.0)
        self.assertAlmostEqual(float(np.nanstd(cleaned[0], ddof=0)), 1.0)

    def test_missing_industry_keeps_stock_in_final_zscore(self):
        factor = np.array([[1.0, 3.0, 20.0]])
        industries = np.array(["银行", "银行", None], dtype=object)

        cleaned = clean_candidate_factor_cross_sections(factor, industries)

        self.assertTrue(np.isfinite(cleaned).all())

    def test_single_stock_industry_has_zero_residual(self):
        factor = np.array([[5.0, 1.0, 3.0]])
        industries = np.array(["单股票行业", "银行", "银行"], dtype=object)

        cleaned = clean_candidate_factor_cross_sections(factor, industries)

        self.assertAlmostEqual(float(cleaned[0, 0]), 0.0, places=12)

    def test_insufficient_industry_regression_warns_and_skips(self):
        factor = np.array([[1.0, 2.0]])
        industries = np.array(["银行", "计算机"], dtype=object)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned = clean_candidate_factor_cross_sections(factor, industries)

        expected = clean_factor_cross_sections(factor)
        np.testing.assert_allclose(cleaned, expected)
        self.assertTrue(
            any(item.category is IndustryNeutralizationWarning for item in caught)
        )


if __name__ == "__main__":
    unittest.main()
