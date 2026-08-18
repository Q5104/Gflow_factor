from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from factor_gfn.backtest.baseline_factor_pool import (
    OOS_UNTOUCHED,
    FrozenBaselineFactorRecord,
    VerifiedFrozenBaselineFactorPool,
)
from factor_gfn.backtest.strategy_input import (
    STRATEGY_INPUT_TOP_K,
    StrategyInputIntegrityError,
    freeze_top100_strategy_input,
    load_verified_strategy_input,
)
from factor_gfn.backtest.static_strategy_bundle import (
    STRATEGY_OOS_LOCKED,
    VerifiedFrozenStrategyBundle,
)
from factor_gfn.backtest.oos_authority import (
    OOSAuthorityError,
    _strategy_records,
    _verify_authorities,
)


def _record(index: int) -> FrozenBaselineFactorRecord:
    return FrozenBaselineFactorRecord(
        provisional_rank=index + 1,
        stage6_sorted_rank=index + 1,
        structural_hash=f"{index + 1:064x}",
        formula=f"factor_{index:03d}",
        prefix_token_ids=(index + 1,),
        node_count=1,
        depth=0,
        train_direction=1 if index % 2 == 0 else -1,
        train_metrics=MappingProxyType({}),
        validation_metrics=MappingProxyType({}),
        selection_status=MappingProxyType(
            {"hard_filter_pass": True, "decorrelation_status": "retained"}
        ),
        result_identity=MappingProxyType({}),
        source_identity=MappingProxyType({"source_ids": (), "origin_ids": ()}),
    )


def _pool(root: Path, count: int = 120) -> VerifiedFrozenBaselineFactorPool:
    records = tuple(_record(index) for index in range(count))
    fingerprint = "a" * 64
    manifest_path = root / fingerprint / "baseline_factor_pool_manifest.json"
    return VerifiedFrozenBaselineFactorPool(
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


class StrategyInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pool = _pool(self.root)
        self.loader_patch = patch(
            "factor_gfn.backtest.strategy_input.load_verified_baseline_factor_pool",
            return_value=self.pool,
        )
        self.loader_patch.start()

    def tearDown(self) -> None:
        self.loader_patch.stop()
        self.temporary.cleanup()

    def test_freeze_is_exact_top100_prefix_and_idempotent(self) -> None:
        first = freeze_top100_strategy_input(self.pool, self.root / "runs")
        second = freeze_top100_strategy_input(self.pool, self.root / "runs")
        verified = load_verified_strategy_input(first.manifest_path)

        self.assertEqual(first.factor_count, STRATEGY_INPUT_TOP_K)
        self.assertTrue(second.reused_existing_artifact)
        self.assertEqual(
            verified.ordered_structural_hashes,
            self.pool.ordered_structural_hashes[:STRATEGY_INPUT_TOP_K],
        )
        self.assertEqual(
            verified.frozen_train_directions,
            self.pool.frozen_train_directions[:STRATEGY_INPUT_TOP_K],
        )
        self.assertEqual(verified.factor_pool_fingerprint, "a" * 64)

    def test_pool_smaller_than_top100_is_rejected(self) -> None:
        small = _pool(self.root / "small", count=99)
        with self.assertRaisesRegex(StrategyInputIntegrityError, "fewer than"):
            freeze_top100_strategy_input(small, self.root / "runs-small")

    def test_manifest_cannot_claim_a_reordered_prefix(self) -> None:
        artifact = freeze_top100_strategy_input(self.pool, self.root / "runs")
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        hashes = manifest["strategy_input"]["ordered_structural_hashes"]
        hashes[0], hashes[1] = hashes[1], hashes[0]
        artifact.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(StrategyInputIntegrityError):
            load_verified_strategy_input(artifact.manifest_path)

    def test_changed_parent_order_invalidates_existing_input(self) -> None:
        artifact = freeze_top100_strategy_input(self.pool, self.root / "runs")
        records = list(self.pool.records)
        records[0], records[1] = records[1], records[0]
        changed = replace(
            self.pool,
            records=tuple(records),
            ordered_structural_hashes=tuple(record.structural_hash for record in records),
            frozen_train_directions=tuple(record.train_direction for record in records),
        )
        with patch(
            "factor_gfn.backtest.strategy_input.load_verified_baseline_factor_pool",
            return_value=changed,
        ):
            with self.assertRaisesRegex(StrategyInputIntegrityError, "exact Top100"):
                load_verified_strategy_input(artifact.manifest_path)

    def test_oos_authority_uses_bound_top100_not_the_full_pool(self) -> None:
        artifact = freeze_top100_strategy_input(self.pool, self.root / "runs")
        strategy_input = load_verified_strategy_input(artifact.manifest_path)
        bundle = VerifiedFrozenStrategyBundle(
            manifest_path=self.root / "bundle_manifest.json",
            bundle_fingerprint="e" * 64,
            factor_pool_fingerprint=self.pool.baseline_factor_pool_fingerprint,
            development_matrix_fingerprint="f" * 64,
            feature_aliases=tuple(f"factor_{index:03d}" for index in range(100)),
            ordered_structural_hashes=strategy_input.ordered_structural_hashes,
            frozen_directions=strategy_input.frozen_train_directions,
            strategies=MappingProxyType(
                {
                    "equal_weight": None,
                    "fixed_icir": None,
                    "lightgbm": None,
                }
            ),
            manifest=MappingProxyType({}),
            oos_status=STRATEGY_OOS_LOCKED,
            strategy_input_manifest_path=strategy_input.manifest_path,
            strategy_input_fingerprint=strategy_input.strategy_input_fingerprint,
        )
        with patch(
            "factor_gfn.backtest.oos_authority.load_verified_strategy_input",
            return_value=strategy_input,
        ):
            _verify_authorities(self.pool, bundle)
            records = _strategy_records(self.pool, bundle)
            self.assertEqual(len(records), 100)
            self.assertEqual(
                tuple(record.structural_hash for record in records),
                self.pool.ordered_structural_hashes[:100],
            )
            bad_bundle = replace(
                bundle,
                ordered_structural_hashes=(
                    *bundle.ordered_structural_hashes[:-1],
                    self.pool.ordered_structural_hashes[100],
                ),
            )
            with self.assertRaises(OOSAuthorityError):
                _verify_authorities(self.pool, bad_bundle)


if __name__ == "__main__":
    unittest.main()
