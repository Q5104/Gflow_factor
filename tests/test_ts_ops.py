import unittest

import numpy as np
import pandas as pd

from factor_gfn.evaluator import ops_impl as ops
from factor_gfn.evaluator.numba_kernels import (
    ema_kernel,
    rolling_window_kernel,
    warm_numba_kernels,
)
from factor_gfn.grammar import WINDOWS
from factor_gfn.grammar.operators import TS_BINARY_OPERATORS, TS_UNARY_OPERATORS


class TimeSeriesCoverageTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(65, dtype=np.float64)[:, None]
        stock = np.arange(3, dtype=np.float64)[None, :]
        self.x = np.sin(time / 7.0 + stock) + 0.03 * time + 0.1 * stock
        self.y = np.cos(time / 9.0 + stock / 2.0) + 0.02 * time - 0.2 * stock

    def test_all_21_ts_operators_are_implemented_once(self):
        expected = {
            operator.name
            for operator in (*TS_UNARY_OPERATORS, *TS_BINARY_OPERATORS)
        }
        self.assertEqual(len(expected), 21)
        self.assertEqual(set(ops.TS_OPERATOR_FUNCTIONS), expected)
        self.assertEqual(len(ops.ALL_OPERATOR_FUNCTIONS), 52)

    def test_every_operator_runs_for_all_five_windows(self):
        for operator in TS_UNARY_OPERATORS:
            function = ops.TS_OPERATOR_FUNCTIONS[operator.name]
            for window in WINDOWS:
                with self.subTest(operator=operator.name, window=window):
                    result = function(self.x, window)
                    self.assertEqual(result.shape, self.x.shape)
                    self.assertEqual(result.dtype, np.dtype(np.float64))
                    self.assertFalse(np.isinf(result).any())
        for operator in TS_BINARY_OPERATORS:
            function = ops.TS_OPERATOR_FUNCTIONS[operator.name]
            for window in WINDOWS:
                with self.subTest(operator=operator.name, window=window):
                    result = function(self.x, self.y, window)
                    self.assertEqual(result.shape, self.x.shape)
                    self.assertEqual(result.dtype, np.dtype(np.float64))
                    self.assertFalse(np.isinf(result).any())

    def test_invalid_windows_are_rejected(self):
        for invalid in (0, 4, 6, 1.5, True):
            with self.subTest(window=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    ops.ts_mean(self.x, invalid)

    def test_inputs_are_not_modified(self):
        original_x = self.x.copy()
        original_y = self.y.copy()
        for operator in TS_UNARY_OPERATORS:
            ops.TS_OPERATOR_FUNCTIONS[operator.name](self.x, 5)
        for operator in TS_BINARY_OPERATORS:
            ops.TS_OPERATOR_FUNCTIONS[operator.name](self.x, self.y, 5)
        np.testing.assert_array_equal(self.x, original_x)
        np.testing.assert_array_equal(self.y, original_y)


class TimeSeriesPandasReferenceTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(12, dtype=np.float64)
        self.x = np.column_stack((time**1.2 + 1.0, np.sin(time / 2.0) + time))
        self.y = np.column_stack((0.5 * time + 2.0, np.cos(time / 3.0) - time / 4.0))
        self.window = 5
        self.frame = pd.DataFrame(self.x)
        self.other = pd.DataFrame(self.y)

    def assert_frame_close(self, actual, expected):
        np.testing.assert_allclose(
            actual,
            np.asarray(expected, dtype=np.float64),
            rtol=1e-11,
            atol=1e-11,
            equal_nan=True,
        )

    def test_basic_rolling_operators_match_pandas(self):
        rolling = self.frame.rolling(self.window, min_periods=self.window)
        self.assert_frame_close(ops.ts_mean(self.x, 5), rolling.mean())
        self.assert_frame_close(ops.ts_std(self.x, 5), rolling.std(ddof=0))
        self.assert_frame_close(ops.ts_max(self.x, 5), rolling.max())
        self.assert_frame_close(ops.ts_min(self.x, 5), rolling.min())
        self.assert_frame_close(ops.ts_sum(self.x, 5), rolling.sum())
        self.assert_frame_close(
            ops.ts_range(self.x, 5),
            rolling.max() - rolling.min(),
        )

    def test_delay_delta_wma_and_ema_match_pandas(self):
        self.assert_frame_close(ops.ts_delay(self.x, 5), self.frame.shift(5))
        self.assert_frame_close(
            ops.ts_delta(self.x, 5), self.frame - self.frame.shift(5)
        )
        weights = np.arange(1, self.window + 1, dtype=np.float64)
        weights /= weights.sum()
        expected_wma = self.frame.rolling(5, min_periods=5).apply(
            lambda values: float(values @ weights), raw=True
        )
        self.assert_frame_close(ops.ts_wma(self.x, 5), expected_wma)
        expected_ema = self.frame.ewm(
            alpha=2.0 / 6.0,
            adjust=False,
            min_periods=5,
        ).mean()
        self.assert_frame_close(ops.ts_ema(self.x, 5), expected_ema)

    def test_rank_position_and_arg_operators_match_pandas_reference(self):
        rolling = self.frame.rolling(5, min_periods=5)
        expected_rank = rolling.apply(
            lambda values: (pd.Series(values).rank(method="average").iloc[-1] - 0.5)
            / len(values),
            raw=False,
        )
        expected_position = rolling.apply(
            lambda values: (values.iloc[-1] - values.min())
            / (values.max() - values.min() + ops.EPSILON),
            raw=False,
        )
        expected_argmax = rolling.apply(
            lambda values: len(values) - 1 - np.argmax(values.to_numpy()[::-1] == values.max()),
            raw=False,
        )
        expected_argmin = rolling.apply(
            lambda values: len(values) - 1 - np.argmax(values.to_numpy()[::-1] == values.min()),
            raw=False,
        )
        self.assert_frame_close(ops.ts_rank(self.x, 5), expected_rank)
        self.assert_frame_close(ops.ts_position(self.x, 5), expected_position)
        self.assert_frame_close(ops.ts_argmax(self.x, 5), expected_argmax)
        self.assert_frame_close(ops.ts_argmin(self.x, 5), expected_argmin)

    def test_covariance_and_correlation_match_pandas(self):
        expected_cov = np.full_like(self.x, np.nan)
        expected_corr = np.full_like(self.x, np.nan)
        for stock_index in range(self.x.shape[1]):
            expected_cov[:, stock_index] = (
                self.frame[stock_index]
                .rolling(5, min_periods=5)
                .cov(self.other[stock_index], ddof=0)
                .to_numpy()
            )
            expected_corr[:, stock_index] = (
                self.frame[stock_index]
                .rolling(5, min_periods=5)
                .corr(self.other[stock_index], ddof=0)
                .to_numpy()
            )
        self.assert_frame_close(ops.ts_cov(self.x, self.y, 5), expected_cov)
        self.assert_frame_close(ops.ts_corr(self.x, self.y, 5), expected_corr)

    def test_cumulative_kernels_match_window_reference_with_missing_values(self):
        rng = np.random.default_rng(20260810)
        left = rng.normal(size=(45, 7))
        right = 0.4 * left + rng.normal(size=left.shape)
        left[[3, 17, 31], [1, 4, 6]] = np.nan
        right[[8, 17, 29], [2, 4, 0]] = np.nan

        for window in WINDOWS:
            weights = np.arange(1, window + 1, dtype=np.float64)
            weights /= weights.sum()
            unary_cases = {
                "ts_mean": lambda sample: np.mean(sample, axis=0),
                "ts_std": lambda sample: np.std(sample, axis=0, ddof=0),
                "ts_sum": lambda sample: np.sum(sample, axis=0),
                "ts_wma": lambda sample: weights @ sample,
                "ts_slope": lambda sample: ops._time_regression(sample)[1],
                "ts_residual": lambda sample: sample[-1]
                - (
                    ops._time_regression(sample)[0]
                    + ops._time_regression(sample)[1] * (sample.shape[0] - 1)
                ),
                "ts_zscore": lambda sample: np.divide(
                    sample[-1] - np.mean(sample, axis=0),
                    np.std(sample, axis=0, ddof=0),
                    out=np.full(sample.shape[1], np.nan),
                    where=np.std(sample, axis=0, ddof=0) > ops.EPSILON,
                ),
            }
            for name, transform in unary_cases.items():
                with self.subTest(operator=name, window=window):
                    expected = ops._rolling_unary(left, window, transform)
                    actual = ops.TS_OPERATOR_FUNCTIONS[name](left, window)
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=1e-10,
                        atol=1e-10,
                        equal_nan=True,
                    )

            binary_cases = {
                "ts_cov": lambda x, y: np.mean(
                    (x - np.mean(x, axis=0)) * (y - np.mean(y, axis=0)),
                    axis=0,
                ),
                "ts_corr": lambda x, y: np.divide(
                    np.mean(
                        (x - np.mean(x, axis=0)) * (y - np.mean(y, axis=0)),
                        axis=0,
                    ),
                    np.std(x, axis=0, ddof=0) * np.std(y, axis=0, ddof=0),
                    out=np.full(x.shape[1], np.nan),
                    where=(np.std(x, axis=0, ddof=0) > ops.EPSILON)
                    & (np.std(y, axis=0, ddof=0) > ops.EPSILON),
                ),
                "ts_beta": lambda x, y: ops._cross_regression(x, y)[1],
                "ts_orth": lambda x, y: x[-1]
                - ops._cross_regression(x, y)[0]
                - ops._cross_regression(x, y)[1] * y[-1],
            }
            for name, transform in binary_cases.items():
                with self.subTest(operator=name, window=window):
                    expected = ops._rolling_binary(left, right, window, transform)
                    actual = ops.TS_OPERATOR_FUNCTIONS[name](left, right, window)
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=1e-10,
                        atol=1e-10,
                        equal_nan=True,
                    )

    def test_centered_moments_remain_stable_for_large_levels(self):
        time = np.arange(40, dtype=np.float64)[:, None]
        stock = np.arange(4, dtype=np.float64)[None, :]
        left = 1.0e12 + time * (0.5 + stock * 0.1) + np.sin(time / 3.0)
        right = -2.0e11 + time * (0.2 + stock * 0.05) + np.cos(time / 5.0)

        expected_std = ops._rolling_unary(
            left,
            20,
            lambda sample: np.std(sample, axis=0, ddof=0),
        )
        expected_cov = ops._rolling_binary(
            left,
            right,
            20,
            lambda x, y: np.mean(
                (x - np.mean(x, axis=0)) * (y - np.mean(y, axis=0)),
                axis=0,
            ),
        )
        np.testing.assert_allclose(
            ops.ts_std(left, 20), expected_std, rtol=1e-5, atol=1e-5, equal_nan=True
        )
        np.testing.assert_allclose(
            ops.ts_cov(left, right, 20),
            expected_cov,
            rtol=1e-4,
            atol=1e-4,
            equal_nan=True,
        )

    def test_numba_loop_kernels_match_numpy_reference_for_all_windows(self):
        rng = np.random.default_rng(20260811)
        values = rng.normal(size=(75, 9))
        values[10:13, 2] = np.nan
        values[31, 5] = np.nan
        values[40:45, 7] = 3.0

        warm_numba_kernels()
        self.assertTrue(rolling_window_kernel.signatures)
        self.assertTrue(ema_kernel.signatures)

        def rank_reference(sample):
            output = np.empty(sample.shape[1], dtype=np.float64)
            for stock_index in range(sample.shape[1]):
                ranks = ops._average_zero_based_rank(sample[:, stock_index])
                output[stock_index] = (ranks[-1] + 0.5) / sample.shape[0]
            return output

        for window in WINDOWS:
            numpy_cases = {
                "ts_max": lambda sample: np.max(sample, axis=0),
                "ts_min": lambda sample: np.min(sample, axis=0),
                "ts_rank": rank_reference,
                "ts_argmax": lambda sample: sample.shape[0]
                - 1
                - np.argmax(sample[::-1] == np.max(sample, axis=0), axis=0),
                "ts_argmin": lambda sample: sample.shape[0]
                - 1
                - np.argmax(sample[::-1] == np.min(sample, axis=0), axis=0),
                "ts_position": lambda sample: (
                    sample[-1] - np.min(sample, axis=0)
                )
                / (np.max(sample, axis=0) - np.min(sample, axis=0) + ops.EPSILON),
                "ts_range": lambda sample: np.max(sample, axis=0)
                - np.min(sample, axis=0),
            }
            for name, transform in numpy_cases.items():
                with self.subTest(operator=name, window=window):
                    expected = ops._rolling_unary(values, window, transform)
                    actual = ops.TS_OPERATOR_FUNCTIONS[name](values, window)
                    np.testing.assert_allclose(actual, expected, equal_nan=True)

            expected_ema = np.full_like(values, np.nan)
            alpha = 2.0 / (window + 1.0)
            for stock_index in range(values.shape[1]):
                ema = np.nan
                consecutive_valid = 0
                for date_index in range(values.shape[0]):
                    value = values[date_index, stock_index]
                    if not np.isfinite(value):
                        ema = np.nan
                        consecutive_valid = 0
                        continue
                    ema = value if consecutive_valid == 0 else (
                        alpha * value + (1.0 - alpha) * ema
                    )
                    consecutive_valid += 1
                    if consecutive_valid >= window:
                        expected_ema[date_index, stock_index] = ema
            np.testing.assert_allclose(
                ops.ts_ema(values, window),
                expected_ema,
                equal_nan=True,
            )


class TimeSeriesBoundaryTests(unittest.TestCase):
    def test_complete_window_and_ema_reset_after_missing(self):
        values = np.arange(12, dtype=np.float64)[:, None]
        values[4, 0] = np.nan

        mean = ops.ts_mean(values, 5)
        self.assertTrue(np.isnan(mean[:9]).all())
        self.assertTrue(np.isfinite(mean[9, 0]))

        ema = ops.ts_ema(values, 5)
        self.assertTrue(np.isnan(ema[:9]).all())
        self.assertTrue(np.isfinite(ema[9, 0]))

        delay = ops.ts_delay(values, 5)
        self.assertTrue(np.isnan(delay[:5]).all())
        self.assertEqual(delay[5, 0], values[0, 0])
        self.assertTrue(np.isnan(delay[9, 0]))

        delta = ops.ts_delta(values, 5)
        self.assertEqual(delta[5, 0], values[5, 0] - values[0, 0])
        self.assertTrue(np.isnan(delta[9, 0]))

    def test_all_nan_inputs_stay_nan(self):
        values = np.full((12, 2), np.nan)
        for operator in TS_UNARY_OPERATORS:
            result = ops.TS_OPERATOR_FUNCTIONS[operator.name](values, 5)
            self.assertTrue(np.isnan(result).all(), operator.name)
        for operator in TS_BINARY_OPERATORS:
            result = ops.TS_OPERATOR_FUNCTIONS[operator.name](values, values, 5)
            self.assertTrue(np.isnan(result).all(), operator.name)

    def test_constant_and_zero_variance_rules(self):
        values = np.ones((10, 2), dtype=np.float64)
        np.testing.assert_allclose(ops.ts_std(values, 5)[4:], 0.0)
        np.testing.assert_allclose(ops.ts_cov(values, values, 5)[4:], 0.0)
        self.assertTrue(np.isnan(ops.ts_corr(values, values, 5)[4:]).all())
        self.assertTrue(np.isnan(ops.ts_zscore(values, 5)[4:]).all())
        np.testing.assert_allclose(ops.ts_position(values, 5)[4:], 0.0)
        np.testing.assert_allclose(ops.ts_slope(values, 5)[4:], 0.0, atol=1e-14)
        np.testing.assert_allclose(ops.ts_residual(values, 5)[4:], 0.0, atol=1e-14)
        np.testing.assert_allclose(ops.ts_range(values, 5)[4:], 0.0)
        np.testing.assert_allclose(ops.ts_rank(values, 5)[4:], 0.5)
        np.testing.assert_allclose(ops.ts_argmax(values, 5)[4:], 4.0)
        np.testing.assert_allclose(ops.ts_argmin(values, 5)[4:], 4.0)
        self.assertTrue(np.isnan(ops.ts_beta(values, values, 5)[4:]).all())
        self.assertTrue(np.isnan(ops.ts_orth(values, values, 5)[4:]).all())

    def test_rank_and_arg_ties_use_confirmed_rules(self):
        values = np.asarray([[1.0], [3.0], [3.0], [2.0], [3.0]])
        self.assertAlmostEqual(ops.ts_rank(values, 5)[4, 0], 0.7)
        self.assertEqual(ops.ts_argmax(values, 5)[4, 0], 4.0)
        self.assertEqual(ops.ts_argmin(values, 5)[4, 0], 0.0)

    def test_regression_direction_and_current_residual(self):
        time = np.arange(12, dtype=np.float64)[:, None]
        trend = 2.0 + 3.0 * time
        np.testing.assert_allclose(ops.ts_slope(trend, 5)[4:], 3.0, atol=1e-12)
        np.testing.assert_allclose(ops.ts_residual(trend, 5)[4:], 0.0, atol=1e-12)

        predictor = time**2 + 1.0
        dependent = 2.0 + 3.0 * predictor
        np.testing.assert_allclose(
            ops.ts_beta(dependent, predictor, 5)[4:], 3.0, atol=1e-12
        )
        np.testing.assert_allclose(
            ops.ts_orth(dependent, predictor, 5)[4:], 0.0, atol=1e-11
        )
        reverse_beta = ops.ts_beta(predictor, dependent, 5)[4:]
        np.testing.assert_allclose(reverse_beta, 1.0 / 3.0, atol=1e-12)


class TimeSeriesCausalityTests(unittest.TestCase):
    def test_future_changes_never_change_past_outputs(self):
        time = np.arange(20, dtype=np.float64)[:, None]
        stock = np.arange(3, dtype=np.float64)[None, :]
        left = np.sin(time / 3.0 + stock) + time * (0.1 + stock * 0.01)
        right = np.cos(time / 4.0 + stock) - time * (0.03 + stock * 0.02)
        changed_left = left.copy()
        changed_right = right.copy()
        changed_left[10:] = changed_left[10:] * -100.0 + 999.0
        changed_right[10:] = changed_right[10:] * 50.0 - 777.0

        for operator in TS_UNARY_OPERATORS:
            function = ops.TS_OPERATOR_FUNCTIONS[operator.name]
            baseline = function(left, 5)
            changed = function(changed_left, 5)
            with self.subTest(operator=operator.name):
                np.testing.assert_allclose(
                    changed[:10], baseline[:10], equal_nan=True
                )

        for operator in TS_BINARY_OPERATORS:
            function = ops.TS_OPERATOR_FUNCTIONS[operator.name]
            baseline = function(left, right, 5)
            changed = function(changed_left, changed_right, 5)
            with self.subTest(operator=operator.name):
                np.testing.assert_allclose(
                    changed[:10], baseline[:10], equal_nan=True
                )


if __name__ == "__main__":
    unittest.main()
