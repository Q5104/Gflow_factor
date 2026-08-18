import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from factor_gfn.backtest.context import (
    Stage5DataConfig,
    build_stage5_context_from_arrays,
)
from factor_gfn.backtest.expression_compatibility import ACCEPTED_REGISTRY_SCHEMA
from factor_gfn.backtest.stage6_evaluation import (
    STAGE6_EVALUATION_RESULT_SCHEMA,
    Stage6CandidateEvaluator,
    Stage6EvaluationConfig,
    _load_npy_row_prefix,
    _stable_hash,
    build_stage6_evaluation_context_from_arrays,
)
from factor_gfn.barra import STYLE_NAMES, BarraConfig
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.grammar import Expression, get_action_id


class Stage6EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-03-16"),
            dtype="datetime64[D]",
        )
        self.stocks = np.asarray([f"S{index:02d}" for index in range(8)])
        day = np.arange(self.dates.size, dtype=np.float64)[:, None]
        stock = np.arange(self.stocks.size, dtype=np.float64)[None, :]
        tensor = np.empty((self.dates.size, 6, self.stocks.size), dtype=np.float64)
        tensor[:, 0, :] = 10.0 + stock + day * (0.05 + 0.004 * stock**2)
        for feature in range(1, 6):
            tensor[:, feature, :] = (
                20.0 + feature + 0.1 * day + (feature + 1.0) * stock
            )
        self.tensor = tensor
        self.universe = np.ones((self.dates.size, self.stocks.size), dtype=bool)
        self.industry = np.tile(
            np.repeat(np.arange(4, dtype=np.int32), 2),
            (self.dates.size, 1),
        )
        self.exposures = {
            name: np.broadcast_to(
                stock + 0.01 * day * (style_index + 1),
                (self.dates.size, self.stocks.size),
            ).copy()
            for style_index, name in enumerate(STYLE_NAMES)
        }
        evaluation = EvaluationConfig(
            min_cross_section_count=4,
            long_quantile=0.25,
        )
        barra = BarraConfig(
            beta_window=3,
            beta_min_periods=2,
            momentum_lookback=3,
            momentum_skip=1,
            volatility_window=3,
            volatility_min_periods=2,
            liquidity_window=2,
            long_short_quantile=0.25,
            min_cross_section_count=4,
            min_common_periods=2,
        )
        self.stage6_config = Stage6EvaluationConfig(
            train_start="2020-01-01",
            train_end="2020-01-31",
            validation_start="2020-02-01",
            validation_end="2020-02-29",
            evaluation=evaluation,
            barra=barra,
        )
        self.context = build_stage6_evaluation_context_from_arrays(
            dates=self.dates,
            stocks=self.stocks,
            factor_tensor=self.tensor,
            universe_mask=self.universe,
            industry_labels=self.industry,
            barra_exposures=self.exposures,
            config=self.stage6_config,
            source_manifest={"fixture": "v1"},
        )

    def _candidate(self) -> dict:
        expression = Expression.from_prefix([get_action_id("close")])
        return {
            "schema": ACCEPTED_REGISTRY_SCHEMA,
            "current_structural_hash": expression.structural_hash(),
            "source_claimed_structural_hash": expression.structural_hash(),
            "formula": expression.to_formula(),
            "prefix_token_ids": list(expression.to_prefix()),
            "node_count": expression.stats.node_count,
            "depth": expression.stats.depth,
            "origin_ids": ["origin-1"],
            "source_ids": ["source-1"],
            "compatibility_record_fingerprint": "c" * 64,
            "historical_metric_reuse": "forbidden",
            "stage6_metric_recompute_required": True,
        }

    def _factor_from_returns(self, train_sign=1.0, validation_sign=1.0):
        factor = np.zeros((self.context.dates.size, self.stocks.size), dtype=np.float64)
        for split_name, sign in (("train", train_sign), ("validation", validation_sign)):
            split = self.context.get_split_data(split_name)
            factor[split.global_rebalance_rows] = sign * split.forward_returns
        return factor

    def _evaluate(self, factor, candidate=None):
        evaluator = Stage6CandidateEvaluator(
            self.context,
            compatibility_audit_fingerprint="a" * 64,
            accepted_registry_fingerprint="b" * 64,
        )
        with patch.object(evaluator._interpreter, "evaluate", return_value=factor):
            return evaluator.evaluate(self._candidate() if candidate is None else candidate)

    def test_context_truncates_at_actual_validation_trade_date_and_rejects_oos(self):
        self.assertEqual(str(self.context.dates[-1]), "2020-02-29")
        self.assertEqual(
            self.context.manifest["requested_validation_end"], "2020-02-29"
        )
        self.assertEqual(
            self.context.manifest["actual_latest_loaded_trade_date"], "2020-02-29"
        )
        self.assertFalse(self.context.manifest["oos"]["loaded"])
        self.assertEqual(self.context.manifest["oos"]["candidate_evaluation_count"], 0)
        with self.assertRaises(PermissionError):
            self.context.get_split_data("oos")

    def test_real_loader_primitive_maps_only_requested_matrix_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npy"
            values = np.arange(60, dtype=np.float64).reshape(10, 2, 3)
            np.save(path, values, allow_pickle=False)
            prefix = _load_npy_row_prefix(path, 6)
            self.assertEqual(prefix.shape, (6, 2, 3))
            np.testing.assert_array_equal(prefix, values[:6])
            self.assertFalse(prefix.flags.writeable)
            with self.assertRaises(IndexError):
                _ = prefix[6]
            prefix._mmap.close()

    def test_requested_end_and_actual_latest_trade_date_are_distinct(self):
        config = Stage6EvaluationConfig(
            train_start="2020-01-01",
            train_end="2020-01-31",
            validation_start="2020-02-01",
            validation_end="2020-03-01",
            evaluation=self.stage6_config.evaluation,
            barra=self.stage6_config.barra,
        )
        context = build_stage6_evaluation_context_from_arrays(
            dates=self.dates[self.dates <= np.datetime64("2020-02-29")],
            stocks=self.stocks,
            factor_tensor=self.tensor[:60],
            universe_mask=self.universe[:60],
            industry_labels=self.industry[:60],
            barra_exposures={name: values[:60] for name, values in self.exposures.items()},
            config=config,
        )
        self.assertEqual(context.manifest["requested_validation_end"], "2020-03-01")
        self.assertEqual(
            context.manifest["actual_latest_loaded_trade_date"], "2020-02-29"
        )

    def test_calendar_is_exact_stage5_prefix_and_validation_keeps_train_warmup(self):
        stage5 = build_stage5_context_from_arrays(
            dates=self.dates,
            stocks=self.stocks,
            factor_tensor=self.tensor,
            universe_mask=self.universe,
            industry_labels=self.industry,
            barra_exposures=self.exposures,
            config=Stage5DataConfig(
                train_start="2020-01-01",
                train_end="2020-01-31",
                validation_start="2020-02-01",
                validation_end="2020-02-29",
                oos_start="2020-03-01",
                oos_end="2020-03-15",
                evaluation=self.stage6_config.evaluation,
                barra=self.stage6_config.barra,
            ),
        )
        expected = [
            (entry.signal_row, entry.included, entry.exclusion_reasons)
            for entry in stage5.calendar
            if entry.signal_row <= self.context.splits["validation"].end_row
        ]
        actual = [
            (entry.signal_row, entry.included, entry.exclusion_reasons)
            for entry in self.context.calendar
        ]
        self.assertEqual(actual, expected)
        validation = self.context.get_split_data("validation")
        self.assertGreater(validation.global_rebalance_rows[0], 0)
        self.assertEqual(self.context.factor_tensor.shape[0], self.context.dates.size)

    def test_context_arrays_and_split_arrays_are_read_only(self):
        self.assertFalse(self.context.factor_tensor.flags.writeable)
        self.assertFalse(self.context.dates.flags.writeable)
        split = self.context.get_split_data("train")
        self.assertFalse(split.forward_returns.flags.writeable)
        self.assertFalse(split.industry_labels.flags.writeable)
        with self.assertRaises(ValueError):
            split.forward_returns[0, 0] = 0.0

    def test_train_direction_is_frozen_and_validation_ic_keeps_raw_sign(self):
        result = self._evaluate(self._factor_from_returns(1.0, -1.0))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.train_direction, 1)
        self.assertGreater(result.train["ic"]["mean"], 0.0)
        self.assertLess(result.validation["ic"]["mean"], 0.0)
        self.assertEqual(result.validation["long"]["direction"], 1)

    def test_negative_train_ic_freezes_negative_direction(self):
        result = self._evaluate(self._factor_from_returns(-1.0, -1.0))
        self.assertEqual(result.train_direction, -1)
        self.assertLess(result.train["ic"]["mean"], 0.0)
        self.assertEqual(result.validation["long"]["direction"], -1)

    def test_invalid_train_direction_still_computes_validation_ic_and_both_barra(self):
        factor = self._factor_from_returns(1.0, 1.0)
        train = self.context.get_split_data("train")
        factor[train.global_rebalance_rows] = 1.0
        result = self._evaluate(factor)
        self.assertEqual(result.status, "completed_invalid")
        self.assertEqual(result.invalid_reasons, ("train_direction_unavailable",))
        self.assertIsNone(result.train_direction)
        self.assertIsNone(result.train["long"]["excess_series"]["values"])
        self.assertGreater(result.validation["ic"]["valid_periods"], 0)
        self.assertEqual(
            set(result.train["barra"]["common_valid_periods"]), set(STYLE_NAMES)
        )
        self.assertEqual(
            set(result.validation["barra"]["common_valid_periods"]), set(STYLE_NAMES)
        )

    def test_metric_valid_periods_and_long_series_are_separate_and_aligned(self):
        result = self._evaluate(self._factor_from_returns())
        for split_name in ("train", "validation"):
            split = result.to_dict()[split_name]
            self.assertIn("valid_periods", split["ic"])
            self.assertIn("valid_periods", split["long"])
            self.assertEqual(
                len(split["long"]["excess_series"]["dates"]),
                len(split["long"]["excess_series"]["values"]),
            )
            self.assertEqual(
                set(split["barra"]["common_valid_periods"]), set(STYLE_NAMES)
            )

    def test_result_fingerprint_excludes_timings_source_and_paths(self):
        candidate = self._candidate()
        first = self._evaluate(self._factor_from_returns(), candidate)
        candidate["origin_ids"] = ["different-origin"]
        candidate["source_ids"] = ["different-source"]
        second = self._evaluate(self._factor_from_returns(), candidate)
        self.assertEqual(first.result_fingerprint, second.result_fingerprint)
        payload = first.deterministic_payload()
        self.assertEqual(first.result_fingerprint, _stable_hash(payload))
        encoded = json.dumps(payload, allow_nan=False)
        for forbidden in (
            "factor_seconds",
            "train_evaluation_seconds",
            "validation_evaluation_seconds",
            "total_seconds",
            "created_at",
            "output_path",
            "source_identity",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_result_serialization_contains_no_nan_or_infinity(self):
        factor = np.ones((self.context.dates.size, self.stocks.size))
        result = self._evaluate(factor)
        self.assertEqual(result.schema, STAGE6_EVALUATION_RESULT_SCHEMA)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_candidate_identity_mismatch_and_historical_metrics_fail_closed(self):
        candidate = self._candidate()
        candidate["formula"] = "open"
        with self.assertRaisesRegex(ValueError, "identity"):
            self._evaluate(self._factor_from_returns(), candidate)
        candidate = self._candidate()
        candidate["reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "historical metrics"):
            self._evaluate(self._factor_from_returns(), candidate)


if __name__ == "__main__":
    unittest.main()
