from __future__ import annotations

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
from factor_gfn.backtest.context import Stage5DataConfig, build_stage5_context_from_arrays
from factor_gfn.backtest.development_factor_matrix import (
    DevelopmentFactorMatrixIntegrityError,
    build_development_factor_matrices,
    freeze_development_factor_matrices,
    load_verified_development_factor_matrices,
)
from factor_gfn.backtest.oos_authority import (
    build_test_factor_matrix,
    generate_test_strategy_scores,
    unlock_verified_test_features,
)
from factor_gfn.backtest.stage6_evaluation import (
    Stage6EvaluationConfig,
    build_stage6_evaluation_context_from_arrays,
)
from factor_gfn.backtest.static_strategy_bundle import (
    STRATEGY_IDS,
    STRATEGY_OOS_LOCKED,
    VerifiedFrozenStrategyBundle,
    build_static_strategy_bundle,
)
from factor_gfn.barra import STYLE_NAMES
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.feature_spaces import DAILY_DERIVED_FEATURE_NAMES
from factor_gfn.grammar import DAILY_DERIVED_ACTION_REGISTRY, Expression


def _vocabulary() -> MappingProxyType:
    registry = DAILY_DERIVED_ACTION_REGISTRY
    return MappingProxyType(
        {
            "feature_space_id": registry.feature_space.feature_space_id,
            "feature_space_fingerprint": registry.feature_space_fingerprint,
            "action_space_fingerprint": registry.fingerprint(),
        }
    )


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
        train_metrics=MappingProxyType({}),
        validation_metrics=MappingProxyType({}),
        selection_status=MappingProxyType(
            {"hard_filter_pass": True, "decorrelation_status": "retained"}
        ),
        result_identity=MappingProxyType({}),
        source_identity=MappingProxyType({"source_ids": (), "origin_ids": ()}),
    )


class DerivedStrategyPlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        dates = np.arange(
            np.datetime64("2020-01-01"), np.datetime64("2020-04-01")
        ).astype("datetime64[D]")
        stocks = np.asarray([f"S{index:03d}" for index in range(24)])
        day = np.arange(dates.size, dtype=np.float64)[:, None]
        stock = np.arange(stocks.size, dtype=np.float64)[None, :]
        raw_open = 10.0 + 0.03 * day + 0.04 * stock
        derived = np.stack(
            [
                np.sin(day / (feature_index + 2.0) + stock / 7.0)
                + 0.01 * feature_index * stock
                for feature_index in range(len(DAILY_DERIVED_FEATURE_NAMES))
            ],
            axis=1,
        )
        universe = np.ones((dates.size, stocks.size), dtype=bool)
        industry = np.broadcast_to(
            (np.arange(stocks.size) % 3).astype(np.int32), universe.shape
        ).copy()
        barra = {
            name: np.broadcast_to(
                stock / (index + 2.0) + 0.001 * day, universe.shape
            ).copy()
            for index, name in enumerate(STYLE_NAMES)
        }
        evaluation = EvaluationConfig(min_cross_section_count=20)
        cls.stage6 = build_stage6_evaluation_context_from_arrays(
            dates=dates[:61],
            stocks=stocks,
            factor_tensor=derived[:61],
            raw_open=raw_open[:61],
            ordered_feature_names=DAILY_DERIVED_FEATURE_NAMES,
            universe_mask=universe[:61],
            industry_labels=industry[:61],
            barra_exposures={name: values[:61] for name, values in barra.items()},
            config=Stage6EvaluationConfig(
                train_start="2020-01-01",
                train_end="2020-01-31",
                validation_start="2020-02-01",
                validation_end="2020-03-01",
                evaluation=evaluation,
            ),
        )
        cls.stage5 = build_stage5_context_from_arrays(
            dates=dates,
            stocks=stocks,
            factor_tensor=derived,
            raw_open=raw_open,
            ordered_feature_names=DAILY_DERIVED_FEATURE_NAMES,
            universe_mask=universe,
            industry_labels=industry,
            barra_exposures=barra,
            config=Stage5DataConfig(
                train_start="2020-01-01",
                train_end="2020-01-31",
                validation_start="2020-02-01",
                validation_end="2020-03-01",
                oos_start="2020-03-02",
                oos_end="2020-03-31",
                evaluation=evaluation,
            ),
        )
        registry = DAILY_DERIVED_ACTION_REGISTRY
        expressions = tuple(
            Expression.from_prefix(
                [registry.get_action_id(name)], action_registry=registry
            )
            for name in ("ret_cc1", "clv")
        )
        records = tuple(
            _record(expression, index + 1)
            for index, expression in enumerate(expressions)
        )
        fingerprint = "d" * 64
        manifest_path = root / "synthetic_pool" / "manifest.json"
        cls.pool = VerifiedFrozenBaselineFactorPool(
            manifest_path=manifest_path,
            records_path=manifest_path.with_name("records.jsonl"),
            baseline_factor_pool_fingerprint=fingerprint,
            manifest=MappingProxyType({"vocabulary": _vocabulary()}),
            records=records,
            ordered_structural_hashes=tuple(
                record.structural_hash for record in records
            ),
            frozen_train_directions=(1, 1),
            upstream_provenance=MappingProxyType({}),
            oos_status=OOS_UNTOUCHED,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_derived_dev_strategies_test_matrix_and_scores_assemble_without_labels(
        self,
    ) -> None:
        matrices = build_development_factor_matrices(self.pool, self.stage6)
        with patch(
            "factor_gfn.backtest.development_factor_matrix.load_verified_baseline_factor_pool",
            return_value=self.pool,
        ):
            artifact = freeze_development_factor_matrices(
                matrices, Path(self.temporary.name) / "runs"
            )
            matrices = load_verified_development_factor_matrices(
                artifact.manifest_path
            )
        built = build_static_strategy_bundle(self.pool, matrices)
        self.assertEqual(tuple(built.strategies), STRATEGY_IDS)
        bundle = VerifiedFrozenStrategyBundle(
            manifest_path=Path(self.temporary.name) / "synthetic_bundle.json",
            bundle_fingerprint=built.logical_fingerprint,
            factor_pool_fingerprint=built.factor_pool_fingerprint,
            development_matrix_fingerprint=built.development_matrix_fingerprint,
            feature_aliases=built.feature_aliases,
            ordered_structural_hashes=built.ordered_structural_hashes,
            frozen_directions=built.frozen_directions,
            strategies=built.strategies,
            manifest=MappingProxyType({"shared_contract": built.shared_contract}),
            oos_status=STRATEGY_OOS_LOCKED,
        )

        feature_context = unlock_verified_test_features(
            self.stage5, self.pool, bundle
        )
        self.assertEqual(
            feature_context.ordered_feature_names, DAILY_DERIVED_FEATURE_NAMES
        )
        self.assertIs(
            feature_context.factor_tensor, self.stage5.expression_feature_tensor
        )
        test_matrix = build_test_factor_matrix(feature_context, self.pool, bundle)
        scores = generate_test_strategy_scores(bundle, test_matrix)

        self.assertEqual(test_matrix.features.factor_count, 2)
        self.assertTrue(np.isfinite(test_matrix.features.values).all())
        self.assertEqual(tuple(scores), STRATEGY_IDS)
        self.assertTrue(
            all(np.isfinite(score.strategy_scores).all() for score in scores.values())
        )

    def test_unknown_vocabulary_and_raw_context_mismatch_fail_closed(self) -> None:
        unknown = replace(
            self.pool,
            manifest=MappingProxyType(
                {
                    "vocabulary": {
                        **dict(_vocabulary()),
                        "feature_space_id": "unknown",
                    }
                }
            ),
        )
        with self.assertRaisesRegex(
            DevelopmentFactorMatrixIntegrityError, "vocabulary is invalid"
        ):
            build_development_factor_matrices(unknown, self.stage6)

        raw_context = replace(
            self.stage6,
            ordered_feature_names=(
                "open",
                "high",
                "low",
                "close",
                "vwap",
                "volume",
            ),
        )
        with self.assertRaisesRegex(
            DevelopmentFactorMatrixIntegrityError, "does not match"
        ):
            build_development_factor_matrices(self.pool, raw_context)


if __name__ == "__main__":
    unittest.main()
