import unittest

import numpy as np
import pandas as pd
from scipy.special import ndtri

from factor_gfn.evaluator import ops_impl as ops
from factor_gfn.grammar.operators import (
    BINARY_OPERATORS,
    CROSS_SECTIONAL_OPERATORS,
    UNARY_OPERATORS,
)


class OperatorRegistryCoverageTests(unittest.TestCase):
    def test_all_31_non_ts_operators_are_implemented_once(self):
        expected = {
            operator.name
            for operator in (
                *UNARY_OPERATORS,
                *BINARY_OPERATORS,
                *CROSS_SECTIONAL_OPERATORS,
            )
        }
        self.assertEqual(len(expected), 31)
        self.assertEqual(set(ops.NON_TS_OPERATOR_FUNCTIONS), expected)

    def test_all_operator_outputs_are_new_float64_matrices(self):
        left = np.asarray([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])
        right = np.asarray([[2.0, 1.0, 3.0], [1.0, 2.0, 3.0]])
        original_left = left.copy()
        original_right = right.copy()

        for operator in (*UNARY_OPERATORS, *CROSS_SECTIONAL_OPERATORS):
            with self.subTest(operator=operator.name):
                result = ops.NON_TS_OPERATOR_FUNCTIONS[operator.name](left)
                self.assertEqual(result.shape, left.shape)
                self.assertEqual(result.dtype, np.dtype(np.float64))
                self.assertFalse(np.isinf(result).any())
                self.assertFalse(np.shares_memory(result, left))
        for operator in BINARY_OPERATORS:
            with self.subTest(operator=operator.name):
                result = ops.NON_TS_OPERATOR_FUNCTIONS[operator.name](left, right)
                self.assertEqual(result.shape, left.shape)
                self.assertEqual(result.dtype, np.dtype(np.float64))
                self.assertFalse(np.isinf(result).any())

        np.testing.assert_equal(left, original_left)
        np.testing.assert_equal(right, original_right)

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            ops.abs(np.asarray([1.0, 2.0]))
        with self.assertRaises(ValueError):
            ops.add(np.ones((2, 2)), np.ones((2, 3)))


class UnaryOperatorTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray([[-4.0, -1.0, 0.0, 1.0, 4.0, np.nan]])

    def test_basic_unary_transforms(self):
        np.testing.assert_allclose(
            ops.abs(self.x), [[4.0, 1.0, 0.0, 1.0, 4.0, np.nan]], equal_nan=True
        )
        np.testing.assert_allclose(
            ops.neg(self.x), [[4.0, 1.0, -0.0, -1.0, -4.0, np.nan]], equal_nan=True
        )
        np.testing.assert_allclose(
            ops.sign(self.x), [[-1.0, -1.0, 0.0, 1.0, 1.0, np.nan]], equal_nan=True
        )
        np.testing.assert_allclose(
            ops.sqrt(self.x), np.sqrt(np.abs(self.x)), equal_nan=True
        )
        np.testing.assert_allclose(
            ops.relu(self.x), [[0.0, 0.0, 0.0, 1.0, 4.0, np.nan]], equal_nan=True
        )

    def test_protected_log_inverse_and_soft_transforms(self):
        np.testing.assert_allclose(
            ops.log(self.x), np.log(np.abs(self.x) + ops.EPSILON), equal_nan=True
        )
        expected_inverse = np.asarray([[-0.25, -1.0, np.nan, 1.0, 0.25, np.nan]])
        np.testing.assert_allclose(ops.inv(self.x), expected_inverse, equal_nan=True)
        np.testing.assert_allclose(ops.tanh(self.x), np.tanh(self.x), equal_nan=True)
        np.testing.assert_allclose(
            ops.softsign(self.x), self.x / (1.0 + np.abs(self.x)), equal_nan=True
        )

    def test_signed_power_and_log1p(self):
        np.testing.assert_allclose(
            ops.signed_power2(self.x),
            np.sign(self.x) * np.abs(self.x) ** 2,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            ops.signed_power3(self.x),
            np.sign(self.x) * np.abs(self.x) ** 3,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            ops.signed_log1p(self.x),
            np.sign(self.x) * np.log1p(np.abs(self.x)),
            equal_nan=True,
        )

    def test_input_infinity_is_treated_as_missing(self):
        result = ops.neg(np.asarray([[np.inf, -np.inf, 1.0]]))
        np.testing.assert_allclose(result, [[np.nan, np.nan, -1.0]], equal_nan=True)


class BinaryOperatorTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray([[2.0, -2.0, 0.0, np.nan]])
        self.y = np.asarray([[4.0, -4.0, 0.0, 1.0]])

    def test_arithmetic_and_nan_propagation(self):
        np.testing.assert_allclose(
            ops.add(self.x, self.y), [[6.0, -6.0, 0.0, np.nan]], equal_nan=True
        )
        np.testing.assert_allclose(
            ops.sub(self.x, self.y), [[-2.0, 2.0, 0.0, np.nan]], equal_nan=True
        )
        np.testing.assert_allclose(
            ops.mul(self.x, self.y), [[8.0, 8.0, 0.0, np.nan]], equal_nan=True
        )
        np.testing.assert_allclose(
            ops.div(self.x, self.y), [[0.5, 0.5, np.nan, np.nan]], equal_nan=True
        )
        np.testing.assert_allclose(
            ops.max2(self.x, self.y), [[4.0, -2.0, 0.0, np.nan]], equal_nan=True
        )
        np.testing.assert_allclose(
            ops.min2(self.x, self.y), [[2.0, -4.0, 0.0, np.nan]], equal_nan=True
        )

    def test_comparisons_return_float_and_keep_missing(self):
        np.testing.assert_allclose(
            ops.greater(self.x, self.y), [[0.0, 1.0, 0.0, np.nan]], equal_nan=True
        )
        np.testing.assert_allclose(
            ops.less(self.x, self.y), [[1.0, 0.0, 0.0, np.nan]], equal_nan=True
        )

    def test_confirmed_ratio_definitions(self):
        expected_signed = (self.x - self.y) / (
            np.abs(self.x) + np.abs(self.y) + ops.EPSILON
        )
        np.testing.assert_allclose(
            ops.signed_ratio(self.x, self.y), expected_signed, equal_nan=True
        )

        expected_log = np.log(
            (self.x[:, :3] + ops.EPSILON) / (self.y[:, :3] + ops.EPSILON)
        )
        np.testing.assert_allclose(
            ops.log_ratio(self.x, self.y)[:, :3], expected_log, equal_nan=True
        )
        self.assertTrue(np.isnan(ops.log_ratio(self.x, self.y)[0, 3]))

    def test_log_ratio_rejects_non_positive_ratio(self):
        result = ops.log_ratio(
            np.asarray([[1.0, -1.0, 1.0]]),
            np.asarray([[-1.0, 1.0, 0.0]]),
        )
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertTrue(np.isnan(result[0, 1]))
        self.assertTrue(np.isfinite(result[0, 2]))


class CrossSectionalOperatorTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray(
            [
                [1.0, 2.0, 100.0, np.nan],
                [5.0, 5.0, 5.0, 5.0],
                [np.nan, np.nan, 1.0, np.nan],
            ]
        )

    def test_rank_quantile_and_rank_gauss_tie_rules(self):
        expected_rank = np.asarray(
            [
                [1.0 / 6.0, 0.5, 5.0 / 6.0, np.nan],
                [0.5, 0.5, 0.5, 0.5],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        expected_quantile = np.asarray(
            [
                [0.0, 0.5, 1.0, np.nan],
                [0.5, 0.5, 0.5, 0.5],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        np.testing.assert_allclose(ops.cs_rank(self.x), expected_rank, equal_nan=True)
        np.testing.assert_allclose(
            ops.cs_quantile(self.x), expected_quantile, equal_nan=True
        )
        np.testing.assert_allclose(
            ops.cs_rank_gauss(self.x), ndtri(expected_rank), equal_nan=True
        )

    def test_demean_zscore_scale_and_minmax_normalize(self):
        first = self.x[[0]]
        valid = np.asarray([1.0, 2.0, 100.0])
        np.testing.assert_allclose(
            ops.cs_demean(first)[0, :3], valid - valid.mean(), equal_nan=True
        )
        np.testing.assert_allclose(
            ops.cs_zscore(first)[0, :3],
            (valid - valid.mean()) / valid.std(ddof=0),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            ops.cs_scale(first)[0, :3], valid / np.abs(valid).sum(), equal_nan=True
        )
        np.testing.assert_allclose(
            ops.cs_normalize(first)[0, :3],
            (valid - valid.min()) / (valid.max() - valid.min() + ops.EPSILON),
            equal_nan=True,
        )

        self.assertTrue(np.isnan(ops.cs_zscore(self.x)[1]).all())
        np.testing.assert_allclose(ops.cs_demean(self.x)[1], np.zeros(4))
        np.testing.assert_allclose(ops.cs_normalize(self.x)[1], np.zeros(4))

    def test_winsorize_clips_and_truncate_sets_outliers_to_nan(self):
        first = self.x[[0]]
        valid = np.asarray([1.0, 2.0, 100.0])
        lower, upper = np.quantile(valid, [0.05, 0.95], method="linear")

        winsorized = ops.cs_winsorize(first)
        np.testing.assert_allclose(
            winsorized[0, :3], np.clip(valid, lower, upper), equal_nan=True
        )

        truncated = ops.cs_truncate(first)
        self.assertTrue(np.isnan(truncated[0, 0]))
        self.assertEqual(truncated[0, 1], 2.0)
        self.assertTrue(np.isnan(truncated[0, 2]))
        self.assertTrue(np.isnan(truncated[0, 3]))

    def test_cross_section_never_uses_other_dates(self):
        baseline = ops.cs_zscore(self.x)
        changed = self.x.copy()
        changed[1] = [1_000.0, -1_000.0, 50.0, 20.0]
        result = ops.cs_zscore(changed)
        np.testing.assert_allclose(result[0], baseline[0], equal_nan=True)
        np.testing.assert_allclose(result[2], baseline[2], equal_nan=True)

    def test_pandas_reference_on_random_sample(self):
        rng = np.random.default_rng(20260804)
        sample = rng.normal(size=(5, 10))
        sample[0, [1, 7]] = np.nan
        sample[2, 3:5] = 0.25
        frame = pd.DataFrame(sample)
        count = frame.notna().sum(axis=1)
        rank = frame.rank(axis=1, method="average")

        expected_rank = rank.sub(0.5).div(count, axis=0)
        expected_quantile = rank.sub(1.0).div(count.sub(1.0), axis=0)
        mean = frame.mean(axis=1)
        std = frame.std(axis=1, ddof=0)
        expected_zscore = frame.sub(mean, axis=0).div(std, axis=0)

        np.testing.assert_allclose(
            ops.cs_rank(sample), expected_rank.to_numpy(), equal_nan=True
        )
        np.testing.assert_allclose(
            ops.cs_quantile(sample), expected_quantile.to_numpy(), equal_nan=True
        )
        np.testing.assert_allclose(
            ops.cs_zscore(sample), expected_zscore.to_numpy(), equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()
