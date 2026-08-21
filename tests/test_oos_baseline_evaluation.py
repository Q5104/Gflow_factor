from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd

from factor_gfn.backtest.baseline_factor_pool import (
    OOS_UNTOUCHED,
    FrozenBaselineFactorRecord,
    VerifiedFrozenBaselineFactorPool,
)
from factor_gfn.backtest.context import (
    Stage5DataConfig,
    build_stage5_context_from_arrays,
)
from factor_gfn.backtest.development_factor_matrix import FeaturesOnlyFactorMatrix
from factor_gfn.backtest.oos_authority import (
    OOSAuthorityError,
    TestFactorMatrix,
    VerifiedTestLabels,
    VerifiedTestScoreArtifact,
    build_test_factor_matrix,
    freeze_test_score_artifact,
    generate_test_strategy_scores,
    load_verified_test_labels,
    load_verified_test_score_artifact,
    unlock_verified_test_features,
)
from factor_gfn.backtest.oos_evaluation import (
    MIN_DECILE_STOCK_COUNT,
    OOSBaselineEvaluationIntegrityError,
    annualized_ratio,
    compound_nav,
    deterministic_deciles,
    drift_adjusted_one_way_turnover,
    evaluate_oos_baselines,
    freeze_oos_baseline_evaluation,
    geometric_annualized_return,
    load_verified_oos_baseline_evaluation,
    max_drawdown_from_returns,
    spearman_average_ties,
)
from factor_gfn.backtest.stage6_evaluation import _stable_hash
from factor_gfn.backtest.static_strategy_bundle import (
    STRATEGY_IDS,
    STRATEGY_OOS_LOCKED,
    FrozenLinearStrategy,
    VerifiedFrozenStrategyBundle,
)
from factor_gfn.barra import STYLE_NAMES
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.evaluator.cross_section import DEFAULT_CLEANING_CONFIG
from factor_gfn.grammar import Expression, get_action_id
from factor_gfn.reporting import OOSReportRenderer, build_oos_report_data


def _readonly(values):
    array = np.asarray(values)
    array.setflags(write=False)
    return array


def _authorities(root: Path):
    expressions = (
        Expression.from_prefix([get_action_id("close")]),
        Expression.from_prefix([get_action_id("volume")]),
    )
    records = tuple(
        FrozenBaselineFactorRecord(
            provisional_rank=index + 1,
            stage6_sorted_rank=index + 1,
            structural_hash=expression.structural_hash(),
            formula=expression.to_formula(),
            prefix_token_ids=expression.to_prefix(),
            node_count=expression.stats.node_count,
            depth=expression.stats.depth,
            train_direction=1 if index == 0 else -1,
            train_metrics=MappingProxyType({}),
            validation_metrics=MappingProxyType({}),
            selection_status=MappingProxyType({}),
            result_identity=MappingProxyType({}),
            source_identity=MappingProxyType({}),
        )
        for index, expression in enumerate(expressions)
    )
    pool_fingerprint = "b" * 64
    pool = VerifiedFrozenBaselineFactorPool(
        manifest_path=root / "pool" / pool_fingerprint / "baseline_factor_pool_manifest.json",
        records_path=root / "pool" / pool_fingerprint / "baseline_factor_pool.jsonl",
        baseline_factor_pool_fingerprint=pool_fingerprint,
        manifest=MappingProxyType({}),
        records=records,
        ordered_structural_hashes=tuple(record.structural_hash for record in records),
        frozen_train_directions=(1, -1),
        upstream_provenance=MappingProxyType({}),
        oos_status=OOS_UNTOUCHED,
    )
    aliases = ("factor_000", "factor_001")
    strategies = {}
    for strategy_id in STRATEGY_IDS:
        strategies[strategy_id] = FrozenLinearStrategy(
            strategy_id=strategy_id,  # type: ignore[arg-type]
            factor_pool_fingerprint=pool_fingerprint,
            feature_aliases=aliases,
            ordered_structural_hashes=pool.ordered_structural_hashes,
            weights=(0.5, 0.5),
            metadata=MappingProxyType({"kind": "test"}),
            fingerprint=(str(len(strategies) + 1) * 64),
        )
    cleaning = {
        **asdict(DEFAULT_CLEANING_CONFIG),
        "order": [
            "winsorize",
            "point_in_time_industry_neutralize",
            "zscore",
            "factor_specific_nonfinite_to_zero",
        ],
        "industry": "SW_level_1_point_in_time",
        "neutralization_failure": "leave_nonfinite_then_strategy_impute_zero",
        "imputation_scope": "base_eligible_stocks_only",
        "base_eligibility": "universe_and_known_point_in_time_industry",
    }
    bundle_fingerprint = "c" * 64
    manifest_path = root / "bundle" / bundle_fingerprint / "strategy_bundle_manifest.json"
    bundle = VerifiedFrozenStrategyBundle(
        manifest_path=manifest_path,
        bundle_fingerprint=bundle_fingerprint,
        factor_pool_fingerprint=pool_fingerprint,
        development_matrix_fingerprint="d" * 64,
        feature_aliases=aliases,
        ordered_structural_hashes=pool.ordered_structural_hashes,
        frozen_directions=(1, -1),
        strategies=MappingProxyType(strategies),
        manifest=MappingProxyType(
            {
                "shared_contract": {
                    "cleaning_contract_fingerprint": _stable_hash(cleaning),
                    "missing": "post_cleaning_factor_specific_nonfinite_to_zero",
                },
                "created_at_utc": "2026-08-17T00:00:00+00:00",
            }
        ),
        oos_status=STRATEGY_OOS_LOCKED,
    )
    return pool, bundle


def _stage5_context():
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-03-16"))
    stocks = np.asarray([f"S{i:03d}" for i in range(24)])
    date_grid = np.arange(dates.size, dtype=float)[:, None]
    stock_grid = np.arange(stocks.size, dtype=float)[None, :]
    open_values = 10 + 0.02 * date_grid + 0.04 * stock_grid + 0.01 * np.sin(date_grid + stock_grid)
    close = open_values * (1 + 0.003 * np.cos(date_grid / 2 + stock_grid))
    tensor = np.stack(
        [
            open_values,
            np.maximum(open_values, close) + 0.1,
            np.minimum(open_values, close) - 0.1,
            close,
            (open_values + close) / 2,
            1000 + date_grid + 3 * stock_grid,
        ],
        axis=1,
    )
    universe = np.ones((dates.size, stocks.size), dtype=bool)
    industries = np.broadcast_to((np.arange(stocks.size) % 3)[None, :], universe.shape).astype(np.int32).copy()
    tensor[50:, 3, 0] = np.nan
    industries[50:, 1] = -1
    universe[50:, 2] = False
    barra = {name: np.broadcast_to(stock_grid + index, universe.shape) for index, name in enumerate(STYLE_NAMES)}
    config = Stage5DataConfig(
        train_start="2020-01-01",
        train_end="2020-01-25",
        validation_start="2020-01-26",
        validation_end="2020-02-19",
        oos_start="2020-02-20",
        oos_end="2020-03-15",
        evaluation=EvaluationConfig(min_cross_section_count=20),
    )
    return build_stage5_context_from_arrays(
        dates=dates,
        stocks=stocks,
        factor_tensor=tensor,
        universe_mask=universe,
        industry_labels=industries,
        barra_exposures=barra,
        config=config,
        source_manifest={"fixture": "authority_only"},
    )


class OOSAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pool, self.bundle = _authorities(self.root)
        self.context = _stage5_context()

    def tearDown(self):
        self.temporary.cleanup()

    def test_verified_pair_unlocks_features_only_and_matching_scores(self):
        feature_context = unlock_verified_test_features(self.context, self.pool, self.bundle)
        self.assertFalse(hasattr(feature_context, "forward_returns"))
        matrix = build_test_factor_matrix(feature_context, self.pool, self.bundle)
        self.assertTrue(np.isfinite(matrix.features.values).all())
        self.assertEqual(
            matrix.contract["missing"],
            "post_cleaning_factor_specific_nonfinite_to_zero",
        )
        test_keys = list(zip(matrix.features.dates.astype(str), matrix.features.symbols))
        self.assertTrue(any(symbol == "S000" for _, symbol in test_keys))
        self.assertFalse(any(symbol == "S001" for _, symbol in test_keys))
        self.assertFalse(any(symbol == "S002" for _, symbol in test_keys))
        s000_values = matrix.features.values[matrix.features.symbols == "S000"]
        self.assertTrue(s000_values.size)
        np.testing.assert_allclose(s000_values[:, 0], 0.0)
        self.assertTrue(np.isfinite(s000_values[:, 1]).all())
        generated = generate_test_strategy_scores(self.bundle, matrix)
        keys = [
            list(zip(generated[name].dates.astype(str), generated[name].symbols))
            for name in STRATEGY_IDS
        ]
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[1], keys[2])
        artifact = freeze_test_score_artifact(
            self.pool, self.bundle, matrix, generated, self.root / "runs"
        )
        loaded = load_verified_test_score_artifact(
            artifact.manifest_path, self.pool, self.bundle, matrix
        )
        labels = load_verified_test_labels(
            self.context, self.pool, self.bundle, matrix, loaded
        )
        self.assertEqual(labels.score_artifact_fingerprint, loaded.fingerprint)
        self.assertEqual(len(labels.forward_returns), matrix.features.row_count)

    def test_fake_or_mismatched_authority_and_early_label_access_fail(self):
        bad_pool = replace(self.pool, baseline_factor_pool_fingerprint="f" * 64)
        with self.assertRaises(OOSAuthorityError):
            unlock_verified_test_features(self.context, bad_pool, self.bundle)
        bad_bundle = replace(self.bundle, factor_pool_fingerprint="e" * 64)
        with self.assertRaises(OOSAuthorityError):
            unlock_verified_test_features(self.context, self.pool, bad_bundle)
        feature_context = unlock_verified_test_features(self.context, self.pool, self.bundle)
        matrix = build_test_factor_matrix(feature_context, self.pool, self.bundle)
        with self.assertRaises(PermissionError):
            load_verified_test_labels(self.context, self.pool, self.bundle, matrix, None)  # type: ignore[arg-type]

    def test_score_payload_tamper_fails_closed(self):
        feature_context = unlock_verified_test_features(self.context, self.pool, self.bundle)
        matrix = build_test_factor_matrix(feature_context, self.pool, self.bundle)
        generated = generate_test_strategy_scores(self.bundle, matrix)
        artifact = freeze_test_score_artifact(self.pool, self.bundle, matrix, generated, self.root / "runs")
        payload = artifact.manifest_path.parent / "strategy_scores_test.parquet"
        payload.write_bytes(payload.read_bytes() + b"tamper")
        with self.assertRaises(OOSAuthorityError):
            load_verified_test_score_artifact(artifact.manifest_path, self.pool, self.bundle, matrix)

    def test_common_score_key_mismatch_fails_closed_without_intersection(self):
        feature_context = unlock_verified_test_features(self.context, self.pool, self.bundle)
        matrix = build_test_factor_matrix(feature_context, self.pool, self.bundle)
        generated = dict(generate_test_strategy_scores(self.bundle, matrix))
        fixed = generated["fixed_icir"]
        generated["fixed_icir"] = replace(
            fixed, symbols=_readonly(np.roll(fixed.symbols, 1))
        )
        with self.assertRaisesRegex(OOSAuthorityError, "common Test score keys mismatch"):
            freeze_test_score_artifact(
                self.pool, self.bundle, matrix, generated, self.root / "runs"
            )


def _evaluation_inputs(root: Path):
    pool, bundle = _authorities(root)
    dates_unique = np.asarray(["2021-01-01", "2021-01-06", "2021-01-11", "2021-01-16"], dtype="datetime64[D]")
    symbols_unique = np.asarray([f"S{i:02d}" for i in range(20)])
    dates = np.repeat(dates_unique, 20)
    symbols = np.tile(symbols_unique, 4)
    values = np.column_stack([np.tile(np.arange(20), 4), np.tile(np.arange(20)[::-1], 4)]).astype(float)
    features = FeaturesOnlyFactorMatrix(
        split="test",
        dates=_readonly(dates),
        symbols=_readonly(symbols),
        values=_readonly(values),
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        feature_aliases=bundle.feature_aliases,
        ordered_structural_hashes=bundle.ordered_structural_hashes,
        fingerprint="7" * 64,
    )
    matrix = TestFactorMatrix(
        features=features,
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        strategy_bundle_fingerprint=bundle.bundle_fingerprint,
        feature_context_fingerprint="8" * 64,
        source_context_fingerprint="9" * 64,
        calendar_fingerprint="a" * 64,
        frozen_directions=bundle.frozen_directions,
        raw_universe_counts=MappingProxyType({str(date): 20 for date in dates_unique}),
        eligible_counts=MappingProxyType({str(date): 20 for date in dates_unique}),
        requested_boundary=("2021-01-01", "2021-01-31"),
        actual_boundary=("2021-01-01", "2021-01-31"),
        contract=MappingProxyType({}),
        fingerprint="6" * 64,
    )
    frames = []
    base = np.arange(20, dtype=float)
    for strategy_index, strategy_id in enumerate(STRATEGY_IDS):
        strategy_scores = np.tile(base if strategy_index != 1 else base[::-1], 4)
        if strategy_id == "lightgbm":
            strategy_scores = np.tile(np.roll(base, 3), 4)
        frames.append(pd.DataFrame({"date": dates, "symbol": symbols, "strategy_id": strategy_id, "strategy_score": strategy_scores}))
    score_frame = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol", "strategy_id"], kind="mergesort").reset_index(drop=True)
    scores = VerifiedTestScoreArtifact(
        manifest_path=root / "scores" / ("5" * 64) / "test_score_manifest.json",
        fingerprint="5" * 64,
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        strategy_bundle_fingerprint=bundle.bundle_fingerprint,
        test_factor_matrix_fingerprint=matrix.fingerprint,
        common_key_fingerprint="4" * 64,
        scores=score_frame,
        manifest=MappingProxyType({}),
    )
    labels_values = np.tile(np.linspace(-0.05, 0.08, 20), 4)
    labels_values[20] = np.nan  # middle date is jointly invalid; turnover chain resets.
    labels = VerifiedTestLabels(
        score_artifact_fingerprint=scores.fingerprint,
        source_context_fingerprint=matrix.source_context_fingerprint,
        calendar_fingerprint=matrix.calendar_fingerprint,
        dates=_readonly(dates.copy()),
        symbols=_readonly(symbols.copy()),
        forward_returns=_readonly(labels_values),
        fingerprint="3" * 64,
        first_access_evidence=MappingProxyType(
            {
                "event": "first_test_label_access_after_verified_score_freeze",
                "score_artifact_fingerprint": scores.fingerprint,
                "score_manifest_path": str(scores.manifest_path),
                "accessed_at_utc": "2026-08-17T00:00:00+00:00",
                "timestamp_excluded_from_label_fingerprint": True,
            }
        ),
    )
    return pool, bundle, matrix, scores, labels


def _lgb_diagnostics():
    row = {
        "mean_rank_ic": 0.1,
        "ic_std": 0.2,
        "icir": 0.5,
        "ic_positive_ratio": 0.6,
        "valid_ic_periods": 3,
        "mean_5d_g10_g1": 0.02,
    }
    return {
        "schema": "factor_gfn.lightgbm_split_effectiveness.v1",
        "rows": {
            "Train": {**row, "model_source": "selection_model", "interpretation": "development_in_sample", "is_true_oos": False},
            "Validation": {**row, "model_source": "selection_model", "interpretation": "development_early_stopping_holdout", "is_true_oos": False},
            "Test": {**row, "model_source": "final_train_validation_refit_frozen_model", "interpretation": "true_untouched_oos", "is_true_oos": True},
        },
        "development_diagnostics_source": "stored_selection_model_scores",
        "final_model_backfill_for_development": False,
    }


class EvaluationMathAndTurnoverTests(unittest.TestCase):
    def test_deciles_ties_sizes_and_semantics(self):
        scores = np.repeat([1.0, 2.0], 10)
        symbols = np.asarray([f"S{i:02d}" for i in range(20)])[::-1]
        groups = deterministic_deciles(scores, symbols)
        self.assertEqual(set(groups), set(range(1, 11)))
        self.assertLessEqual(max(np.bincount(groups)[1:]) - min(np.bincount(groups)[1:]), 1)
        self.assertTrue(np.all(scores[groups == 1] <= scores[groups == 10]))
        with self.assertRaises(ValueError):
            deterministic_deciles(np.arange(19), [str(i) for i in range(19)])
        self.assertAlmostEqual(spearman_average_ties([1, 1, 2], [1, 2, 3]), math.sqrt(3) / 2)

    def test_returns_nav_annualization_ratio_and_drawdown(self):
        returns = np.asarray([0.10, -0.05])
        np.testing.assert_allclose(compound_nav(returns), [1.1, 1.045])
        self.assertAlmostEqual(geometric_annualized_return(returns, 2), 0.045)
        expected_ratio = returns.mean() / returns.std(ddof=1) * math.sqrt(2)
        self.assertAlmostEqual(annualized_ratio(returns, 2), expected_ratio)
        self.assertAlmostEqual(max_drawdown_from_returns(returns), -0.05)

    def test_turnover_hand_calculations_union_replacement_and_invalid_drift(self):
        turnover, replacement, reason = drift_adjusted_one_way_turnover(
            ["A", "B"], [0.10, -0.10], ["A", "B"]
        )
        self.assertAlmostEqual(turnover, 0.05)
        self.assertEqual(replacement, 0.0)
        self.assertEqual(reason, "ok")
        turnover, replacement, _ = drift_adjusted_one_way_turnover(
            ["A", "B"], [0.0, 0.0], ["C", "D"]
        )
        self.assertAlmostEqual(turnover, 1.0)
        self.assertAlmostEqual(replacement, 1.0)
        turnover, replacement, _ = drift_adjusted_one_way_turnover(
            ["A", "B"], [0.0, 0.0], ["B", "C"]
        )
        self.assertAlmostEqual(turnover, 0.5)
        self.assertAlmostEqual(replacement, 0.5)
        turnover, _, reason = drift_adjusted_one_way_turnover(
            ["A", "B"], [0.0, np.nan], ["A", "B"]
        )
        self.assertTrue(np.isnan(turnover))
        self.assertEqual(reason, "invalid_previous_holding_drift_return")

    def test_common_invalid_period_and_drift_adjusted_turnover_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = _evaluation_inputs(Path(directory))
            with patch("factor_gfn.backtest.oos_evaluation._lightgbm_split_effectiveness", return_value=_lgb_diagnostics()):
                evaluation = evaluate_oos_baselines(*inputs, cost_rate=0.0)
            coverage = evaluation.coverage_by_date
            self.assertEqual(coverage["valid_period"].tolist(), [True, False, True, True])
            self.assertEqual(coverage.iloc[1]["eligible_stock_count"], 20)
            self.assertEqual(coverage.iloc[1]["label_eligible_count"], 19)
            for strategy_id in STRATEGY_IDS:
                turnover = evaluation.turnover_by_date.loc[evaluation.turnover_by_date["strategy_id"] == strategy_id]
                self.assertTrue(np.isnan(turnover.iloc[0]["one_way_turnover"]))
                self.assertTrue(np.isnan(turnover.iloc[1]["one_way_turnover"]))
                self.assertGreater(turnover.iloc[2]["one_way_turnover"], 0.0)
                self.assertEqual(turnover.iloc[2]["constituent_replacement_rate"], 0.0)
            benchmark_counts = evaluation.portfolio_returns.groupby("date")["benchmark_return"].nunique()
            self.assertTrue((benchmark_counts == 1).all())
            self.assertEqual(evaluation.contract["headline_performance"], "gross")


class EvaluationArtifactAndReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = _evaluation_inputs(self.root)
        with patch("factor_gfn.backtest.oos_evaluation._lightgbm_split_effectiveness", return_value=_lgb_diagnostics()):
            self.evaluation = evaluate_oos_baselines(*self.inputs)

    def tearDown(self):
        self.temporary.cleanup()

    def _freeze_and_load(self):
        first = freeze_oos_baseline_evaluation(self.evaluation, self.root / "runs")
        second = freeze_oos_baseline_evaluation(self.evaluation, self.root / "runs")
        self.assertTrue(second.reused_existing_artifact)
        loaded = load_verified_oos_baseline_evaluation(
            first.manifest_path,
            expected_factor_pool_fingerprint=self.evaluation.factor_pool_fingerprint,
            expected_strategy_bundle_fingerprint=self.evaluation.strategy_bundle_fingerprint,
            expected_test_score_artifact_fingerprint=self.evaluation.test_score_artifact_fingerprint,
        )
        self.assertEqual(first.fingerprint, loaded.fingerprint)
        return first, loaded

    def test_immutable_evaluation_and_tamper_detection(self):
        artifact, _ = self._freeze_and_load()
        with self.assertRaises(OOSBaselineEvaluationIntegrityError):
            load_verified_oos_baseline_evaluation(
                artifact.manifest_path,
                expected_factor_pool_fingerprint="f" * 64,
                expected_strategy_bundle_fingerprint=self.evaluation.strategy_bundle_fingerprint,
                expected_test_score_artifact_fingerprint=self.evaluation.test_score_artifact_fingerprint,
            )
        payload = artifact.manifest_path.parent / "coverage_by_date.parquet"
        payload.write_bytes(payload.read_bytes() + b"tamper")
        with self.assertRaises(OOSBaselineEvaluationIntegrityError):
            load_verified_oos_baseline_evaluation(
                artifact.manifest_path,
                expected_factor_pool_fingerprint=self.evaluation.factor_pool_fingerprint,
                expected_strategy_bundle_fingerprint=self.evaluation.strategy_bundle_fingerprint,
                expected_test_score_artifact_fingerprint=self.evaluation.test_score_artifact_fingerprint,
            )

    def test_eighteen_figures_eight_tables_and_notebook_contract(self):
        _, loaded = self._freeze_and_load()
        report = build_oos_report_data(loaded)
        split_payload = loaded.payloads["lightgbm_split_effectiveness"]
        self.assertEqual(split_payload["rows"]["Validation"]["model_source"], "selection_model")
        self.assertFalse(split_payload["rows"]["Validation"]["is_true_oos"])
        self.assertEqual(split_payload["rows"]["Test"]["model_source"], "final_train_validation_refit_frozen_model")
        self.assertTrue(split_payload["rows"]["Test"]["is_true_oos"])
        self.assertFalse(split_payload["final_model_backfill_for_development"])
        renderer = OOSReportRenderer(report, self.root / "outputs")
        manifest_path = renderer.render_all()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["figures"]), 20)
        self.assertEqual(len(manifest["tables"]), 8)
        self.assertEqual(len(list((manifest_path.parent / "figures").glob("*.png"))), 20)
        self.assertEqual(len(list((manifest_path.parent / "tables").glob("*.csv"))), 8)
        notebook_path = Path(__file__).parents[1] / "notebooks" / "oos_baseline_evaluation.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        headings = ["".join(cell.source) for cell in notebook.cells if cell.cell_type == "markdown"]
        required_headings = [
            "00｜参数与权威性校验",
            "01｜样本覆盖",
            "02｜策略评分有效性",
            "03｜十分位组合分析",
            "04｜G10 多头组合与基准",
            "05｜超额收益与多空收益",
            "06｜换手率",
            "07｜滚动 ICIR 权重诊断",
            "08｜LightGBM 跨阶段诊断",
            "09｜策略冻结与可复现信息",
            "10｜导出正式报告",
        ]
        positions = [
            next(index for index, text in enumerate(headings) if heading in text)
            for heading in required_headings
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(all("\ufffd" not in text for text in headings))
        code = "\n".join("".join(cell.source) for cell in notebook.cells if cell.cell_type == "code")
        self.assertNotIn("score_frozen_strategy", code)
        self.assertNotIn("deterministic_deciles", code)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code"))


if __name__ == "__main__":
    unittest.main()
