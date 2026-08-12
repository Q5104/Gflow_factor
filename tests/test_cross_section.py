import unittest
import warnings
from unittest.mock import patch

import numpy as np

from factor_gfn.evaluator.cross_section import (
    IndustryNeutralizationWarning,
    NeutralizationDiagnostics,
    clean_candidate_factor_cross_sections,
    clean_factor_cross_sections,
    encode_industry_panel,
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

    def test_missing_industry_stock_is_excluded_from_final_zscore(self):
        factor = np.array([[1.0, 3.0, 20.0]])
        industries = np.array(["银行", "银行", None], dtype=object)

        cleaned = clean_candidate_factor_cross_sections(factor, industries)

        self.assertTrue(np.isfinite(cleaned[0, :2]).all())
        self.assertTrue(np.isnan(cleaned[0, 2]))

    def test_integer_point_in_time_industries_use_negative_one_as_missing(self):
        factor = np.array([[1.0, 3.0, 20.0], [2.0, 5.0, 30.0]])
        industries = np.array(
            [[801780, 801780, -1], [801780, 801150, 801150]],
            dtype=np.int32,
        )

        cleaned = clean_candidate_factor_cross_sections(factor, industries)

        self.assertTrue(np.isfinite(cleaned[0, :2]).all())
        self.assertTrue(np.isnan(cleaned[0, 2]))
        self.assertTrue(np.isfinite(cleaned[1]).all())

    def test_single_stock_industry_has_zero_residual(self):
        factor = np.array([[5.0, 1.0, 3.0]])
        industries = np.array(["单股票行业", "银行", "银行"], dtype=object)

        cleaned = clean_candidate_factor_cross_sections(factor, industries)

        self.assertAlmostEqual(float(cleaned[0, 0]), 0.0, places=12)

    def test_insufficient_industry_regression_warns_and_excludes_date(self):
        factor = np.array([[1.0, 2.0]])
        industries = np.array(["银行", "计算机"], dtype=object)
        diagnostics = NeutralizationDiagnostics()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned = clean_candidate_factor_cross_sections(
                factor,
                industries,
                diagnostics=diagnostics,
            )

        self.assertTrue(np.isnan(cleaned).all())
        self.assertTrue(
            any(item.category is IndustryNeutralizationWarning for item in caught)
        )
        self.assertEqual(diagnostics.skipped_rows, {0})
        detail = diagnostics.skipped_details[0]
        self.assertEqual(detail.factor_valid_count, 2)
        self.assertEqual(detail.known_industry_count, 2)
        self.assertEqual(detail.industry_count, 2)
        self.assertEqual(detail.required_regression_count, 3)
        self.assertEqual(detail.reason, "insufficient_industry_regression_samples")

    def test_no_known_industries_excludes_date_with_full_audit(self):
        diagnostics = NeutralizationDiagnostics()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = clean_candidate_factor_cross_sections(
                np.array([[1.0, 2.0, 3.0]]),
                np.array([None, None, None], dtype=object),
                diagnostics=diagnostics,
            )

        self.assertTrue(np.isnan(cleaned).all())
        detail = diagnostics.skipped_details[0]
        self.assertEqual(detail.factor_valid_count, 3)
        self.assertEqual(detail.known_industry_count, 0)
        self.assertEqual(detail.industry_count, 0)
        self.assertEqual(detail.reason, "no_known_industry_labels")

    def test_neutralization_kernel_failure_excludes_date_instead_of_using_raw_factor(self):
        diagnostics = NeutralizationDiagnostics()
        with patch(
            "factor_gfn.evaluator.cross_section._industry_group_residuals",
            side_effect=np.linalg.LinAlgError("synthetic"),
        ), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = clean_candidate_factor_cross_sections(
                np.array([[1.0, 2.0, 3.0, 4.0]]),
                np.array(["银行", "银行", "计算机", "计算机"], dtype=object),
                diagnostics=diagnostics,
            )

        self.assertTrue(np.isnan(cleaned).all())
        self.assertEqual(
            diagnostics.skipped_details[0].reason,
            "industry_ols_failure",
        )

    def test_preencoded_industries_match_rank_deficient_ols_reference(self):
        rng = np.random.default_rng(20260811)
        factor = rng.normal(size=(7, 40))
        factor[1, ::7] = np.nan
        universe = rng.random(factor.shape) > 0.08
        industries = rng.integers(-1, 6, size=factor.shape, dtype=np.int32)
        encoded = encode_industry_panel(industries, factor.shape)

        optimized = clean_candidate_factor_cross_sections(
            factor,
            industries,
            universe,
            encoded_industries=encoded,
        )

        reference = np.full_like(factor, np.nan)
        for date_index in range(factor.shape[0]):
            valid = universe[date_index] & np.isfinite(factor[date_index])
            raw = factor[date_index, valid]
            lower, upper = np.quantile(raw, [0.01, 0.99])
            clipped = np.clip(raw, lower, upper)
            labels = industries[date_index, valid]
            known = labels >= 0
            categories = np.unique(labels[known])
            if int(known.sum()) < categories.size + 1:
                continue
            design = np.column_stack(
                [np.ones(int(known.sum()))]
                + [(labels[known] == category).astype(float) for category in categories]
            )
            coefficients = np.linalg.lstsq(design, clipped[known], rcond=None)[0]
            residuals = clipped[known] - design @ coefficients
            residuals[np.abs(residuals) <= 1e-12] = 0.0
            std = residuals.std(ddof=0)
            if std <= 1e-12:
                continue
            positions = np.flatnonzero(valid)[known]
            reference[date_index, positions] = (residuals - residuals.mean()) / std

        np.testing.assert_array_equal(np.isnan(optimized), np.isnan(reference))
        np.testing.assert_allclose(optimized, reference, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
