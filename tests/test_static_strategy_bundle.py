from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import numpy as np

from factor_gfn.backtest.baseline_factor_pool import (
    OOS_UNTOUCHED,
    FrozenBaselineFactorRecord,
    VerifiedFrozenBaselineFactorPool,
)
from factor_gfn.backtest.development_factor_matrix import (
    DEVELOPMENT_FACTOR_MATRIX_MANIFEST_FILENAME,
    DevelopmentFactorMatrices,
    DevelopmentSplitMatrix,
    FeaturesOnlyFactorMatrix,
    FrozenFactorFeature,
    build_development_factor_matrices,
    freeze_development_factor_matrices,
    impute_strategy_factor_nonfinite,
    load_verified_development_factor_matrices,
    strategy_matrix_base_eligibility,
)
from factor_gfn.backtest.stage6_evaluation import (
    Stage6EvaluationConfig,
    _stable_hash,
    build_stage6_evaluation_context_from_arrays,
)
from factor_gfn.backtest.static_strategy_bundle import (
    FIXED_ICIR_FILENAME,
    LIGHTGBM_MODEL_FILENAME,
    STRATEGY_BUNDLE_MANIFEST_FILENAME,
    STRATEGY_OOS_LOCKED,
    StrategyBundleIntegrityError,
    build_equal_weight_strategy,
    build_fixed_icir_strategy,
    build_static_strategy_bundle,
    equal_date_sample_weights,
    freeze_static_strategy_bundle,
    load_verified_strategy_bundle,
    score_api_accepts_labels,
    score_frozen_strategy,
)
from factor_gfn.backtest.strategy_input import VerifiedStrategyInput
from factor_gfn.barra import STYLE_NAMES
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.evaluator.cross_section import encode_industry_panel
from factor_gfn.grammar import ACTIONS, Expression, get_action_id


def _readonly(values):
    array = np.asarray(values)
    array.setflags(write=False)
    return array


def _record(expression: Expression, rank: int, direction: int) -> FrozenBaselineFactorRecord:
    return FrozenBaselineFactorRecord(
        provisional_rank=rank,
        stage6_sorted_rank=rank,
        structural_hash=expression.structural_hash(),
        formula=expression.to_formula(),
        prefix_token_ids=expression.to_prefix(),
        node_count=expression.stats.node_count,
        depth=expression.stats.depth,
        train_direction=direction,
        train_metrics=MappingProxyType({}),
        validation_metrics=MappingProxyType({}),
        selection_status=MappingProxyType(
            {"hard_filter_pass": True, "decorrelation_status": "retained"}
        ),
        result_identity=MappingProxyType({}),
        source_identity=MappingProxyType({"source_ids": (), "origin_ids": ()}),
    )


def _synthetic_pool_and_context(root: Path):
    dates = np.arange(
        np.datetime64("2020-01-01"), np.datetime64("2020-02-20")
    ).astype("datetime64[D]")
    stocks = np.asarray([f"S{index:03d}" for index in range(24)])
    date_grid = np.arange(dates.size, dtype=np.float64)[:, None]
    stock_grid = np.arange(stocks.size, dtype=np.float64)[None, :]
    open_values = (
        10.0
        + 0.02 * date_grid
        + 0.05 * stock_grid
        + 0.03 * np.sin(date_grid / 3.0 + stock_grid)
    )
    close_values = open_values * (
        1.0 + 0.004 * np.sin(date_grid / 2.0 + stock_grid / 3.0)
    )
    high = np.maximum(open_values, close_values) + 0.1
    low = np.minimum(open_values, close_values) - 0.1
    vwap = (open_values + close_values) / 2.0
    volume = 1000.0 + 5.0 * date_grid + 7.0 * stock_grid + np.cos(stock_grid)
    tensor = np.stack(
        [open_values, high, low, close_values, vwap, volume], axis=1
    ).astype(np.float64)
    universe = np.ones((dates.size, stocks.size), dtype=bool)
    industries = np.broadcast_to(
        (np.arange(stocks.size) % 3).astype(np.int32), universe.shape
    ).copy()
    industries[25, 0] = -1
    barra = {
        name: np.sin(stock_grid / (index + 2.0)) + 0.001 * date_grid
        for index, name in enumerate(STYLE_NAMES)
    }
    config = Stage6EvaluationConfig(
        train_start="2020-01-01",
        train_end="2020-01-25",
        validation_start="2020-01-26",
        validation_end="2020-02-19",
        evaluation=EvaluationConfig(min_cross_section_count=20),
    )
    context = build_stage6_evaluation_context_from_arrays(
        dates=dates,
        stocks=stocks,
        factor_tensor=tensor,
        universe_mask=universe,
        industry_labels=industries,
        barra_exposures=barra,
        config=config,
        source_manifest={"provider_fingerprint": "a" * 64},
    )
    rolling_close = Expression.from_prefix(
        [get_action_id("ts_mean", 5), get_action_id("close")]
    )
    volume_expression = Expression.from_prefix([get_action_id("volume")])
    records = (
        _record(rolling_close, 1, 1),
        _record(volume_expression, 2, -1),
    )
    fingerprint = "b" * 64
    manifest_path = root / "synthetic_pool" / fingerprint / "baseline_factor_pool_manifest.json"
    pool = VerifiedFrozenBaselineFactorPool(
        manifest_path=manifest_path,
        records_path=manifest_path.with_name("baseline_factor_pool.jsonl"),
        baseline_factor_pool_fingerprint=fingerprint,
        manifest=MappingProxyType({}),
        records=records,
        ordered_structural_hashes=tuple(record.structural_hash for record in records),
        frozen_train_directions=tuple(record.train_direction for record in records),
        upstream_provenance=MappingProxyType({}),
        oos_status=OOS_UNTOUCHED,
    )
    return pool, context


def _synthetic_top100_pool_and_input(
    root: Path,
    context,
) -> tuple[VerifiedFrozenBaselineFactorPool, VerifiedStrategyInput]:
    leaf = get_action_id("close")
    unary_expressions = [
        Expression.from_prefix([action_id, leaf])
        for action_id, action in enumerate(ACTIONS)
        if action.arity == 1
    ][:100]
    binary_expressions = [
        Expression.from_prefix([action_id, get_action_id("close"), get_action_id("volume")])
        for action_id, action in enumerate(ACTIONS)
        if action.arity == 2
    ][:20]
    expressions = tuple(unary_expressions + binary_expressions)
    if len(expressions) != 120:
        raise AssertionError("synthetic Top100 fixture requires 120 expressions")
    records = tuple(
        _record(expression, rank, 1 if rank % 2 else -1)
        for rank, expression in enumerate(expressions, start=1)
    )
    fingerprint = "c" * 64
    manifest_path = root / "large_pool" / fingerprint / "baseline_factor_pool_manifest.json"
    pool = VerifiedFrozenBaselineFactorPool(
        manifest_path=manifest_path,
        records_path=manifest_path.with_name("baseline_factor_pool.jsonl"),
        baseline_factor_pool_fingerprint=fingerprint,
        manifest=MappingProxyType({}),
        records=records,
        ordered_structural_hashes=tuple(record.structural_hash for record in records),
        frozen_train_directions=tuple(record.train_direction for record in records),
        upstream_provenance=MappingProxyType({}),
        oos_status=OOS_UNTOUCHED,
    )
    input_manifest_path = root / "strategy_input" / ("d" * 64) / "strategy_input_manifest.json"
    strategy_input = VerifiedStrategyInput(
        manifest_path=input_manifest_path,
        strategy_input_fingerprint="d" * 64,
        factor_pool_manifest_path=pool.manifest_path,
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        records=pool.records[:100],
        ordered_structural_hashes=pool.ordered_structural_hashes[:100],
        frozen_train_directions=pool.frozen_train_directions[:100],
        top_k=100,
        manifest=MappingProxyType({}),
        oos_status=OOS_UNTOUCHED,
    )
    return pool, strategy_input


class DevelopmentMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pool, self.context = _synthetic_pool_and_context(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_matrix_preserves_order_direction_complete_case_and_warmup(self):
        matrices = build_development_factor_matrices(self.pool, self.context)
        self.assertEqual(
            tuple(item.structural_hash for item in matrices.feature_mapping),
            self.pool.ordered_structural_hashes,
        )
        self.assertEqual(
            tuple(item.train_direction for item in matrices.feature_mapping),
            (1, -1),
        )
        validation = matrices.splits["validation"].features
        self.assertIn(np.datetime64("2020-01-26"), validation.dates)
        keys = set(zip(validation.dates.astype(str), validation.symbols))
        self.assertNotIn(("2020-01-26", "S000"), keys)
        self.assertTrue(np.isfinite(validation.values).all())
        self.assertEqual(
            matrices.contract["label"]["formula"],
            "open[t+6] / open[t+1] - 1",
        )
        self.assertEqual(matrices.contract["oos"], "not_loaded_and_not_accepted")

    def test_strategy_imputation_preserves_finite_and_excludes_base_ineligible(self):
        cleaned = np.asarray([[1.25, np.nan, np.inf, -2.0]], dtype=np.float64)
        universe = np.asarray([[True, True, True, False]])
        encoded = encode_industry_panel(
            np.asarray([[0, 1, -1, 2]], dtype=np.int32),
            cleaned.shape,
        )
        eligible = strategy_matrix_base_eligibility(universe, encoded)
        imputed = impute_strategy_factor_nonfinite(cleaned, eligible)

        np.testing.assert_array_equal(eligible, [[True, True, False, False]])
        self.assertEqual(imputed[0, 0], 1.25)
        self.assertEqual(imputed[0, 1], 0.0)
        self.assertTrue(np.isnan(imputed[0, 2]))
        self.assertTrue(np.isnan(imputed[0, 3]))

    def test_expression_identity_tamper_fails(self):
        bad_record = replace(self.pool.records[0], formula="close")
        bad_pool = replace(self.pool, records=(bad_record, self.pool.records[1]))
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            build_development_factor_matrices(bad_pool, self.context)

    def test_matrix_freeze_is_verified_idempotent_and_timestamp_independent(self):
        matrices = build_development_factor_matrices(self.pool, self.context)
        with patch(
            "factor_gfn.backtest.development_factor_matrix.load_verified_baseline_factor_pool",
            return_value=self.pool,
        ):
            first = freeze_development_factor_matrices(matrices, self.root / "runs")
            second = freeze_development_factor_matrices(matrices, self.root / "runs")
            loaded = load_verified_development_factor_matrices(first.manifest_path)
            self.assertTrue(second.reused_existing_artifact)
            self.assertEqual(first.fingerprint, loaded.fingerprint)
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            manifest["created_at_utc"] = "2099-01-01T00:00:00+00:00"
            first.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            reloaded = load_verified_development_factor_matrices(first.manifest_path)
            self.assertEqual(reloaded.fingerprint, first.fingerprint)

    def test_strategy_input_binds_matrix_to_exact_top100_prefix(self):
        pool, strategy_input = _synthetic_top100_pool_and_input(
            self.root,
            self.context,
        )
        matrices = build_development_factor_matrices(
            pool,
            self.context,
            strategy_input=strategy_input,
        )

        self.assertEqual(len(matrices.feature_mapping), 100)
        self.assertEqual(
            tuple(item.structural_hash for item in matrices.feature_mapping),
            pool.ordered_structural_hashes[:100],
        )
        self.assertNotIn(
            pool.ordered_structural_hashes[100],
            tuple(item.structural_hash for item in matrices.feature_mapping),
        )
        self.assertEqual(matrices.strategy_input_fingerprint, "d" * 64)
        self.assertEqual(matrices.contract["strategy_input"]["top_k"], 100)
        self.assertEqual(
            matrices.contract["missing"],
            "post_cleaning_factor_specific_nonfinite_to_zero",
        )
        for split_name in ("train", "validation"):
            split = self.context.get_split_data(split_name)
            encoded = encode_industry_panel(
                split.industry_labels,
                split.universe_mask.shape,
            )
            expected_rows = int(
                strategy_matrix_base_eligibility(split.universe_mask, encoded).sum()
            )
            values = matrices.splits[split_name].features.values
            self.assertEqual(values.shape, (expected_rows, 100))
            self.assertTrue(np.isfinite(values).all())

        with (
            patch(
                "factor_gfn.backtest.development_factor_matrix.load_verified_baseline_factor_pool",
                return_value=pool,
            ),
            patch(
                "factor_gfn.backtest.development_factor_matrix.load_verified_strategy_input",
                return_value=strategy_input,
            ),
        ):
            artifact = freeze_development_factor_matrices(matrices, self.root / "runs")
            loaded = load_verified_development_factor_matrices(artifact.manifest_path)

        self.assertEqual(len(loaded.feature_mapping), 100)
        self.assertEqual(loaded.strategy_input_fingerprint, "d" * 64)
        self.assertEqual(
            loaded.strategy_input_manifest_path,
            strategy_input.manifest_path.resolve(),
        )


def _manual_matrices() -> DevelopmentFactorMatrices:
    aliases = ("factor_000", "factor_001")
    hashes = ("1" * 64, "2" * 64)
    mapping = tuple(
        FrozenFactorFeature(
            alias=aliases[index],
            factor_index=index,
            structural_hash=hashes[index],
            formula="close",
            prefix_token_ids=(get_action_id("close"),),
            node_count=1,
            depth=0,
            train_direction=1,
        )
        for index in range(2)
    )
    split_results = {}
    patterns = (
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([1.0, 2.0, 4.0, 3.0]),
        np.asarray([1.0, 3.0, 4.0, 2.0]),
        np.asarray([2.0, 1.0, 4.0, 3.0]),
    )
    for split_index, name in enumerate(("train", "validation")):
        dates = np.repeat(
            np.asarray(
                [
                    np.datetime64(f"2020-01-{1 + split_index * 10:02d}"),
                    np.datetime64(f"2020-01-{6 + split_index * 10:02d}"),
                ]
            ),
            4,
        )
        symbols = np.tile(np.asarray(["A", "B", "C", "D"]), 2)
        first = np.concatenate(patterns[split_index * 2 : split_index * 2 + 2])
        values = np.column_stack([first, -first])
        labels = np.tile(np.asarray([1.0, 2.0, 3.0, 4.0]), 2)
        features = FeaturesOnlyFactorMatrix(
            split=name,
            dates=_readonly(dates),
            symbols=_readonly(symbols),
            values=_readonly(values),
            factor_pool_fingerprint="b" * 64,
            feature_aliases=aliases,
            ordered_structural_hashes=hashes,
            fingerprint=f"{split_index + 3}" * 64,
        )
        split_results[name] = DevelopmentSplitMatrix(
            features=features,
            forward_returns=_readonly(labels),
            label_fingerprint=f"{split_index + 5}" * 64,
            boundary=MappingProxyType({}),
        )
    return DevelopmentFactorMatrices(
        artifact_manifest_path=Path("synthetic_manifest.json"),
        artifact_fingerprint="e" * 64,
        factor_pool_manifest_path=Path("synthetic_pool.json"),
        factor_pool_fingerprint="b" * 64,
        context_fingerprint="c" * 64,
        calendar_fingerprint="d" * 64,
        feature_mapping=mapping,
        splits=MappingProxyType(split_results),
        contract=MappingProxyType(
            {
                "cleaning": MappingProxyType({"zscore_ddof": 0}),
                "evaluation_config": MappingProxyType(
                    {"min_cross_section_count": 2}
                ),
            }
        ),
        provenance=MappingProxyType({}),
        logical_fingerprint="f" * 64,
    )


class LinearStrategyTests(unittest.TestCase):
    def test_equal_arbitrary_k_exact_mean_and_label_free_api(self):
        matrices = _manual_matrices()
        strategy = build_equal_weight_strategy(matrices)
        self.assertEqual(strategy.weights, (0.5, 0.5))
        scores = score_frozen_strategy(strategy, matrices.splits["train"].features)
        np.testing.assert_allclose(scores.strategy_scores, 0.0)
        self.assertFalse(score_api_accepts_labels())

    def test_fixed_icir_ddof1_positive_clipping_and_frozen_score(self):
        matrices = _manual_matrices()
        strategy = build_fixed_icir_strategy(
            matrices, min_cross_section_count=2
        )
        self.assertFalse(strategy.metadata["fallback_status"])
        np.testing.assert_allclose(strategy.weights, [1.0, 0.0])
        per_factor = strategy.metadata["per_factor"]
        self.assertGreater(per_factor[0]["icir"], 0)
        self.assertLess(per_factor[1]["icir"], 0)
        self.assertEqual(per_factor[0]["observation_count"], 4)
        scores = score_frozen_strategy(strategy, matrices.splits["train"].features)
        np.testing.assert_allclose(
            scores.strategy_scores,
            matrices.splits["train"].features.values[:, 0],
        )

    def test_fixed_icir_all_nonpositive_falls_back_to_equal(self):
        matrices = _manual_matrices()
        swapped = {}
        for name, split in matrices.splits.items():
            values = -np.abs(split.features.values)
            features = replace(split.features, values=_readonly(values))
            swapped[name] = replace(split, features=features)
        fallback_input = replace(matrices, splits=MappingProxyType(swapped))
        strategy = build_fixed_icir_strategy(
            fallback_input, min_cross_section_count=2
        )
        self.assertTrue(strategy.metadata["fallback_status"])
        self.assertEqual(strategy.metadata["fallback_reason"], "all_nonpositive_or_invalid_icir")
        np.testing.assert_allclose(strategy.weights, [0.5, 0.5])

    def test_equal_date_sample_weights_have_equal_date_totals(self):
        dates = np.asarray(
            ["2020-01-01"] * 2 + ["2020-01-06"] * 5 + ["2020-01-11"] * 3,
            dtype="datetime64[D]",
        )
        weights = equal_date_sample_weights(dates)
        totals = [weights[dates == date].sum() for date in np.unique(dates)]
        np.testing.assert_allclose(totals, totals[0])


@unittest.skipUnless(
    importlib.util.find_spec("lightgbm") is not None,
    "lightgbm is not installed in the project interpreter",
)
class LightGBMAndBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.pool, cls.context = _synthetic_pool_and_context(cls.root)
        raw_matrices = build_development_factor_matrices(cls.pool, cls.context)
        with patch(
            "factor_gfn.backtest.development_factor_matrix.load_verified_baseline_factor_pool",
            return_value=cls.pool,
        ):
            matrix_artifact = freeze_development_factor_matrices(
                raw_matrices, cls.root / "runs"
            )
            cls.matrices = load_verified_development_factor_matrices(
                matrix_artifact.manifest_path
            )
        cls.bundle = build_static_strategy_bundle(cls.pool, cls.matrices)
        with cls._loader_patches():
            cls.artifact = freeze_static_strategy_bundle(
                cls.bundle, cls.root / "runs"
            )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _loader_patches(cls):
        pool_patch_a = patch(
            "factor_gfn.backtest.static_strategy_bundle.load_verified_baseline_factor_pool",
            return_value=cls.pool,
        )
        pool_patch_b = patch(
            "factor_gfn.backtest.development_factor_matrix.load_verified_baseline_factor_pool",
            return_value=cls.pool,
        )

        class _Both:
            def __enter__(self):
                pool_patch_a.start()
                pool_patch_b.start()

            def __exit__(self, exc_type, exc, traceback):
                pool_patch_b.stop()
                pool_patch_a.stop()

        return _Both()

    def _copied_bundle(self) -> Path:
        manifest = json.loads(self.artifact.manifest_path.read_text(encoding="utf-8"))
        fingerprint = manifest["strategy_bundle_fingerprint"]
        destination = self.root / f"copy-{next(tempfile._get_candidate_names())}" / fingerprint
        shutil.copytree(self.artifact.manifest_path.parent, destination)
        return destination / STRATEGY_BUNDLE_MANIFEST_FILENAME

    def test_lightgbm_training_flow_and_label_free_final_prediction(self):
        strategy = self.bundle.strategies["lightgbm"]
        self.assertGreaterEqual(strategy.metadata["best_iteration"], 1)
        self.assertEqual(
            strategy.metadata["final_n_estimators"],
            strategy.metadata["best_iteration"],
        )
        self.assertEqual(
            strategy.metadata["development_score_source"],
            "selection_model_not_final_refit_model",
        )
        self.assertEqual(
            strategy.metadata["missing_contract"],
            "post_cleaning_factor_specific_nonfinite_to_zero",
        )
        self.assertEqual(
            self.bundle.lightgbm_development_scores["train"].shape,
            (self.matrices.splits["train"].features.row_count,),
        )
        scores = score_frozen_strategy(
            strategy, self.matrices.splits["validation"].features
        )
        self.assertTrue(np.isfinite(scores.strategy_scores).all())

    def test_bundle_freeze_loader_idempotency_and_deep_immutability(self):
        with self._loader_patches():
            second = freeze_static_strategy_bundle(self.bundle, self.root / "runs")
            loaded = load_verified_strategy_bundle(self.artifact.manifest_path)
        self.assertTrue(second.reused_existing_artifact)
        self.assertEqual(loaded.bundle_fingerprint, self.artifact.bundle_fingerprint)
        self.assertEqual(loaded.oos_status, STRATEGY_OOS_LOCKED)
        with self.assertRaises(TypeError):
            loaded.manifest["oos_status"] = "changed"
        with self.assertRaises(TypeError):
            loaded.strategies["fixed_icir"].metadata["fallback_status"] = True

    def test_factor_pool_mismatch_fails_closed(self):
        wrong_pool = replace(
            self.pool, baseline_factor_pool_fingerprint="c" * 64
        )
        with (
            patch(
                "factor_gfn.backtest.static_strategy_bundle.load_verified_baseline_factor_pool",
                return_value=wrong_pool,
            ),
            patch(
                "factor_gfn.backtest.development_factor_matrix.load_verified_baseline_factor_pool",
                return_value=self.pool,
            ),
            self.assertRaises(StrategyBundleIntegrityError),
        ):
            load_verified_strategy_bundle(self.artifact.manifest_path)

    def test_timestamp_only_change_does_not_change_bundle_identity(self):
        manifest_path = self._copied_bundle()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["created_at_utc"] = "2099-01-01T00:00:00+00:00"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self._loader_patches():
            loaded = load_verified_strategy_bundle(manifest_path)
        self.assertEqual(loaded.bundle_fingerprint, self.artifact.bundle_fingerprint)

    def test_icir_model_best_iteration_order_and_direction_tamper_fail(self):
        mutations = (
            FIXED_ICIR_FILENAME,
            LIGHTGBM_MODEL_FILENAME,
            "lightgbm/metadata.json",
            STRATEGY_BUNDLE_MANIFEST_FILENAME,
        )
        for filename in mutations:
            manifest_path = self._copied_bundle()
            target = manifest_path.parent / filename
            if filename.endswith(".json"):
                value = json.loads(target.read_text(encoding="utf-8"))
                if filename == FIXED_ICIR_FILENAME:
                    value["weights"][0] += 0.01
                elif filename == "lightgbm/metadata.json":
                    value["metadata"]["best_iteration"] += 1
                else:
                    value["frozen_directions"][0] *= -1
                target.write_text(
                    json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            else:
                target.write_bytes(target.read_bytes() + b"tamper")
            with self._loader_patches(), self.assertRaises(StrategyBundleIntegrityError):
                load_verified_strategy_bundle(manifest_path)

    def test_conflicting_existing_artifact_fails_closed(self):
        target = self.artifact.manifest_path.parent / FIXED_ICIR_FILENAME
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"tamper")
            with self._loader_patches(), self.assertRaises(StrategyBundleIntegrityError):
                freeze_static_strategy_bundle(self.bundle, self.root / "runs")
        finally:
            target.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
