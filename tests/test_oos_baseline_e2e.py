from __future__ import annotations

import json
import re
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import nbformat
import numpy as np

from factor_gfn.backtest.baseline_factor_pool import (
    OOS_UNTOUCHED,
    FrozenBaselineFactorRecord,
    VerifiedFrozenBaselineFactorPool,
)
from factor_gfn.backtest.context import Stage5DataConfig, build_stage5_context_from_arrays
from factor_gfn.backtest.development_factor_matrix import (
    build_development_factor_matrices,
    freeze_development_factor_matrices,
    load_verified_development_factor_matrices,
)
from factor_gfn.backtest.oos_authority import (
    OOSAuthorityError,
    build_test_factor_matrix,
    freeze_test_score_artifact,
    generate_test_strategy_scores,
    load_verified_test_labels,
    load_verified_test_score_artifact,
    unlock_verified_test_features,
)
from factor_gfn.backtest.oos_evaluation import (
    evaluate_oos_baselines,
    freeze_oos_baseline_evaluation,
    load_verified_oos_baseline_evaluation,
)
from factor_gfn.backtest.stage6_evaluation import (
    Stage6EvaluationConfig,
    build_stage6_evaluation_context_from_arrays,
)
from factor_gfn.backtest.static_strategy_bundle import (
    STRATEGY_IDS,
    build_static_strategy_bundle,
    freeze_static_strategy_bundle,
    load_verified_strategy_bundle,
)
from factor_gfn.barra import STYLE_NAMES
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.grammar import Expression, get_action_id
from factor_gfn.reporting import OOSReportRenderer, build_oos_report_data


def _record(expression: Expression, rank: int) -> FrozenBaselineFactorRecord:
    return FrozenBaselineFactorRecord(
        provisional_rank=rank,
        stage6_sorted_rank=rank,
        structural_hash=expression.structural_hash(),
        formula=expression.to_formula(),
        prefix_token_ids=expression.to_prefix(),
        node_count=expression.stats.node_count,
        depth=expression.stats.depth,
        train_direction=1,
        train_metrics=MappingProxyType({"fixture": "positive_signal"}),
        validation_metrics=MappingProxyType({"fixture": "positive_signal"}),
        selection_status=MappingProxyType(
            {"hard_filter_pass": True, "decorrelation_status": "retained"}
        ),
        result_identity=MappingProxyType({"fixture": True}),
        source_identity=MappingProxyType({"source_ids": (), "origin_ids": ()}),
    )


def _synthetic_inputs(root: Path):
    dates = np.arange(
        np.datetime64("2020-01-01"), np.datetime64("2020-06-09")
    ).astype("datetime64[D]")
    stocks = np.asarray([f"S{index:03d}" for index in range(30)])
    date_grid = np.arange(dates.size, dtype=np.float64)[:, None]
    stock_grid = np.arange(stocks.size, dtype=np.float64)[None, :]
    daily_growth = 0.0002 + 0.000055 * stock_grid
    open_values = (8.0 + 0.08 * stock_grid) * np.power(1.0 + daily_growth, date_grid)
    close_values = open_values * (1.0 + 0.0005 * stock_grid)
    high_values = np.maximum(open_values, close_values) + 0.05
    low_values = np.minimum(open_values, close_values) - 0.05
    vwap_values = (open_values + close_values) / 2.0
    volume_values = 1000.0 + 25.0 * stock_grid + 0.2 * date_grid
    factor_tensor = np.stack(
        [
            open_values,
            high_values,
            low_values,
            close_values,
            vwap_values,
            volume_values,
        ],
        axis=1,
    ).astype(np.float64)
    universe = np.ones((dates.size, stocks.size), dtype=bool)
    industries = np.broadcast_to(
        (np.arange(stocks.size) % 3).astype(np.int32), universe.shape
    ).copy()
    industries[125, 0] = -1  # One Test feature-coverage loss; labels remain separate.
    barra = {
        name: np.broadcast_to(
            stock_grid / (index + 2.0) + 0.0001 * date_grid, universe.shape
        ).copy()
        for index, name in enumerate(STYLE_NAMES)
    }
    evaluation = EvaluationConfig(min_cross_section_count=20)
    stage6 = build_stage6_evaluation_context_from_arrays(
        dates=dates[:120],
        stocks=stocks,
        factor_tensor=factor_tensor[:120],
        universe_mask=universe[:120],
        industry_labels=industries[:120],
        barra_exposures={name: values[:120] for name, values in barra.items()},
        config=Stage6EvaluationConfig(
            train_start="2020-01-01",
            train_end="2020-03-20",
            validation_start="2020-03-21",
            validation_end="2020-04-29",
            evaluation=evaluation,
        ),
        source_manifest={"fixture": "E2_synthetic_development_only"},
    )
    stage5 = build_stage5_context_from_arrays(
        dates=dates,
        stocks=stocks,
        factor_tensor=factor_tensor,
        universe_mask=universe,
        industry_labels=industries,
        barra_exposures=barra,
        config=Stage5DataConfig(
            train_start="2020-01-01",
            train_end="2020-03-20",
            validation_start="2020-03-21",
            validation_end="2020-04-29",
            oos_start="2020-04-30",
            oos_end="2020-06-08",
            evaluation=evaluation,
        ),
        source_manifest={"fixture": "E2_synthetic_full_history"},
    )
    expressions = (
        Expression.from_prefix([get_action_id("close")]),
        Expression.from_prefix([get_action_id("volume")]),
    )
    records = tuple(_record(expression, index + 1) for index, expression in enumerate(expressions))
    fingerprint = "e" * 64
    manifest_path = (
        root
        / "synthetic_frozen_factor_pool"
        / fingerprint
        / "baseline_factor_pool_manifest.json"
    )
    pool = VerifiedFrozenBaselineFactorPool(
        manifest_path=manifest_path,
        records_path=manifest_path.with_name("baseline_factor_pool.jsonl"),
        baseline_factor_pool_fingerprint=fingerprint,
        manifest=MappingProxyType({"fixture": "E2"}),
        records=records,
        ordered_structural_hashes=tuple(record.structural_hash for record in records),
        frozen_train_directions=(1, 1),
        upstream_provenance=MappingProxyType({"fixture": "E2"}),
        oos_status=OOS_UNTOUCHED,
    )
    return pool, stage6, stage5


class FullSyntheticOOSBaselineE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.pool, stage6_context, cls.stage5_context = _synthetic_inputs(cls.root)
        cls.patch_stack = ExitStack()
        cls.patch_stack.enter_context(
            patch(
                "factor_gfn.backtest.development_factor_matrix.load_verified_baseline_factor_pool",
                return_value=cls.pool,
            )
        )
        cls.patch_stack.enter_context(
            patch(
                "factor_gfn.backtest.static_strategy_bundle.load_verified_baseline_factor_pool",
                return_value=cls.pool,
            )
        )

        matrices = build_development_factor_matrices(cls.pool, stage6_context)
        matrix_artifact = freeze_development_factor_matrices(
            matrices, cls.root / "runs"
        )
        cls.development_matrices = load_verified_development_factor_matrices(
            matrix_artifact.manifest_path
        )
        built_bundle = build_static_strategy_bundle(
            cls.pool, cls.development_matrices
        )
        bundle_artifact = freeze_static_strategy_bundle(
            built_bundle, cls.root / "runs"
        )
        cls.bundle = load_verified_strategy_bundle(bundle_artifact.manifest_path)

        feature_context = unlock_verified_test_features(
            cls.stage5_context, cls.pool, cls.bundle
        )
        cls.test_matrix = build_test_factor_matrix(
            feature_context, cls.pool, cls.bundle
        )
        cls.generated_scores = generate_test_strategy_scores(
            cls.bundle, cls.test_matrix
        )
        score_artifact = freeze_test_score_artifact(
            cls.pool,
            cls.bundle,
            cls.test_matrix,
            cls.generated_scores,
            cls.root / "runs",
        )
        cls.test_scores = load_verified_test_score_artifact(
            score_artifact.manifest_path,
            cls.pool,
            cls.bundle,
            cls.test_matrix,
        )
        cls.test_labels = load_verified_test_labels(
            cls.stage5_context,
            cls.pool,
            cls.bundle,
            cls.test_matrix,
            cls.test_scores,
        )
        cls.evaluation = evaluate_oos_baselines(
            cls.pool,
            cls.bundle,
            cls.test_matrix,
            cls.test_scores,
            cls.test_labels,
        )
        evaluation_artifact = freeze_oos_baseline_evaluation(
            cls.evaluation, cls.root / "runs"
        )
        cls.verified_evaluation = load_verified_oos_baseline_evaluation(
            evaluation_artifact.manifest_path,
            expected_factor_pool_fingerprint=cls.pool.baseline_factor_pool_fingerprint,
            expected_strategy_bundle_fingerprint=cls.bundle.bundle_fingerprint,
            expected_test_score_artifact_fingerprint=cls.test_scores.fingerprint,
        )
        cls.report = build_oos_report_data(cls.verified_evaluation)
        cls.output_dir = cls.root / "outputs" / cls.verified_evaluation.fingerprint
        cls.report_manifest_path = OOSReportRenderer(
            cls.report, cls.output_dir
        ).render_all()

    @classmethod
    def tearDownClass(cls):
        cls.patch_stack.close()
        cls.temporary.cleanup()

    def test_full_e1a_to_e1b_chain_has_expected_positive_signal_results(self):
        self.assertEqual(tuple(self.bundle.strategies), STRATEGY_IDS)
        lightgbm = self.bundle.strategies["lightgbm"]
        self.assertGreater(lightgbm.metadata["best_iteration"], 0)
        self.assertLess(lightgbm.metadata["best_iteration"], 1000)
        self.assertEqual(
            lightgbm.metadata["final_n_estimators"],
            lightgbm.metadata["best_iteration"],
        )
        self.assertEqual(lightgbm.metadata["early_stopping_patience"], 50)
        self.assertEqual(lightgbm.metadata["development_score_source"], "selection_model_not_final_refit_model")
        self.assertEqual(
            tuple(lightgbm.metadata["final_refit_splits"]), ("train", "validation")
        )

        performance = self.verified_evaluation.payloads["performance_summary"]
        deciles = self.report.decile_return_table.set_index("Strategy")
        nav = self.verified_evaluation.payloads["nav_series"]
        for strategy_id, display_name in zip(
            STRATEGY_IDS, ("Equal Weight", "Rolling ICIR", "LightGBM")
        ):
            self.assertGreater(performance[strategy_id]["mean_rank_ic"], 0.0)
            self.assertGreater(deciles.loc[display_name, "G10"], deciles.loc[display_name, "G1"])
            self.assertGreater(deciles.loc[display_name, "G10-G1"], 0.0)
            strategy_nav = nav.loc[nav["strategy_id"] == strategy_id].sort_values("date")
            self.assertEqual(strategy_nav.iloc[0]["g10_nav"], 1.0)
            self.assertGreater(strategy_nav.iloc[-1]["g10_nav"], 1.0)
            self.assertGreater(strategy_nav.iloc[-1]["long_short_nav"], 1.0)
            self.assertGreater(
                performance[strategy_id]["g10_long"]["geometric_annualized_return"],
                0.0,
            )
            self.assertGreater(
                performance[strategy_id]["turnover"]["turnover_observation_count"],
                0,
            )
            self.assertGreater(
                performance[strategy_id]["turnover"]["mean_one_way_turnover"],
                0.0,
            )

        coverage = self.verified_evaluation.payloads["coverage_by_date"]
        self.assertEqual(int(coverage["raw_universe_count"].max()), 30)
        self.assertEqual(int(coverage["eligible_stock_count"].min()), 29)
        self.assertLess(float(coverage["coverage_ratio"].min()), 1.0)
        split_rows = self.verified_evaluation.payloads[
            "lightgbm_split_effectiveness"
        ]["rows"]
        self.assertEqual(split_rows["Train"]["model_source"], "selection_model")
        self.assertEqual(split_rows["Validation"]["model_source"], "selection_model")
        self.assertEqual(
            split_rows["Test"]["model_source"],
            "final_train_validation_refit_frozen_model",
        )

    def test_artifacts_report_and_notebook_loader_are_complete(self):
        report_manifest = json.loads(
            self.report_manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(len(report_manifest["figures"]), 20)
        self.assertEqual(len(report_manifest["tables"]), 8)
        self.assertEqual(len(list((self.output_dir / "figures").glob("*.png"))), 20)
        self.assertEqual(len(list((self.output_dir / "tables").glob("*.csv"))), 8)
        self.assertIn(
            "01_oos_strategy_performance_comparison.png",
            report_manifest["figures"],
        )
        self.assertIn(
            "16_rolling_icir_excess_nav_drawdown.png",
            report_manifest["figures"],
        )
        self.assertIn(
            "05_rolling_icir_test_strategy_score_rankic.png",
            report_manifest["figures"],
        )
        renderer = OOSReportRenderer(self.report, self.output_dir)
        annual_figure = renderer.figure_annual_returns("fixed_icir")
        annual_year_count = self.report.portfolio_returns.loc[
            self.report.portfolio_returns["strategy_id"] == "fixed_icir", "date"
        ].dt.year.nunique()
        self.assertEqual(len(annual_figure.axes[0].patches), annual_year_count)
        self.assertIn("滚动 ICIR", annual_figure.axes[0].get_title())
        excess_figure = renderer.figure_excess_nav_drawdown("fixed_icir")
        self.assertEqual(len(excess_figure.axes), 2)
        self.assertEqual(excess_figure.axes[1].get_xlabel(), "调仓日期")
        rank_ic_figure = renderer.figure_rank_ic("fixed_icir")
        self.assertEqual(len(rank_ic_figure.axes), 2)
        self.assertIn("滚动 ICIR", rank_ic_figure.axes[0].get_title())
        self.assertEqual(
            len(rank_ic_figure.axes[0].patches),
            len(self.report.strategy_rank_ic_by_date.loc[
                self.report.strategy_rank_ic_by_date["strategy_id"] == "fixed_icir"
            ]),
        )

        notebook_path = Path(__file__).parents[1] / "notebooks" / "oos_baseline_evaluation.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        loader_source = next(
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code" and cell.get("id") == "e1b-00-load"
        )
        loader_source = re.sub(
            r"LATEST_POINTER = [\s\S]*?TEST_SCORE_ARTIFACT_FINGERPRINT = manifest_identity\['test_score_artifact_fingerprint'\]",
            "\n".join(
                (
                    f"EVALUATION_MANIFEST = Path(r{str(self.verified_evaluation.manifest_path)!r})",
                    f"FACTOR_POOL_FINGERPRINT = {self.pool.baseline_factor_pool_fingerprint!r}",
                    f"STRATEGY_BUNDLE_FINGERPRINT = {self.bundle.bundle_fingerprint!r}",
                    f"TEST_SCORE_ARTIFACT_FINGERPRINT = {self.test_scores.fingerprint!r}",
                )
            ),
            loader_source,
            count=1,
        )
        notebook_output = self.root / "notebook_outputs"
        loader_source = re.sub(
            r"OUTPUT_ROOT = Path\(r['\"]outputs/oos_baseline['\"]\)",
            f"OUTPUT_ROOT = Path(r{str(notebook_output)!r})",
            loader_source,
            count=1,
        )
        namespace: dict[str, object] = {}
        exec(compile(loader_source, str(notebook_path), "exec"), namespace)
        self.assertEqual(
            namespace["evaluation"].fingerprint,
            self.verified_evaluation.fingerprint,
        )

    def test_two_critical_fail_closed_cases(self):
        mismatched_pool = replace(
            self.pool, baseline_factor_pool_fingerprint="f" * 64
        )
        with self.assertRaises(OOSAuthorityError):
            unlock_verified_test_features(
                self.stage5_context, mismatched_pool, self.bundle
            )

        mismatched_scores = dict(self.generated_scores)
        fixed = mismatched_scores["fixed_icir"]
        mismatched_scores["fixed_icir"] = replace(
            fixed, symbols=np.roll(fixed.symbols, 1)
        )
        with self.assertRaisesRegex(OOSAuthorityError, "common Test score keys mismatch"):
            freeze_test_score_artifact(
                self.pool,
                self.bundle,
                self.test_matrix,
                mismatched_scores,
                self.root / "mismatch_runs",
            )


if __name__ == "__main__":
    unittest.main()
