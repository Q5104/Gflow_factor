import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from factor_gfn.barra import STYLE_NAMES
from factor_gfn.backtest.candidate_import import import_candidate_source_set
from factor_gfn.backtest.expression_compatibility import (
    audit_expression_compatibility,
)
from factor_gfn.backtest.sources import (
    CandidateSourceSpec,
    materialize_candidate_source,
    materialize_source_set,
)
from factor_gfn.backtest.stage6_mixed_evaluation import Stage6TrainReuseOverlay
from factor_gfn.backtest.stage6_train_reuse import (
    HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA,
    HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA,
    run_stage6_hybrid_train_reuse_overlay,
)
from factor_gfn.grammar import (
    DAILY_DERIVED_ACTION_REGISTRY,
    RAW_ACTION_REGISTRY,
    ActionRegistry,
    Expression,
)


def _stable_hash(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provider_manifest() -> dict:
    return {
        "schema": "factor_gfn.real_reward_provider.v8",
        "data_scope": "training_only",
        "context_fingerprint": "c" * 64,
        "reward_evaluator_context_fingerprint": "d" * 64,
        "evaluation_config": {"rebalance_interval": 5, "entry_lag": 1},
        "reward_config": {
            "barra_min_common_periods": 60,
            "candidate_industry_neutralization": True,
        },
        "calendar": {
            "periods": 2,
            "first_date": "2020-01-01",
            "last_date": "2020-01-06",
            "sha256": "e" * 64,
        },
        "reward_panel": {"mode": "fixed_rebalance_compact"},
        "interpreter": {"numeric_kernel_schema": "test-kernel"},
        "industry_neutralization": {"enabled": True},
    }


def _train_scope_projection(provider: dict) -> dict:
    reward_config = provider["reward_config"]
    return {
        "provider_schema": provider["schema"],
        "data_scope": provider["data_scope"],
        "context_fingerprint": provider["context_fingerprint"],
        "reward_evaluator_context_fingerprint": provider[
            "reward_evaluator_context_fingerprint"
        ],
        "evaluation_config": provider["evaluation_config"],
        "barra_metric_config": {
            "min_common_periods": reward_config["barra_min_common_periods"],
            "candidate_industry_neutralization": reward_config[
                "candidate_industry_neutralization"
            ],
        },
        "calendar": provider["calendar"],
        "reward_panel": provider["reward_panel"],
        "interpreter": provider["interpreter"],
        "industry_neutralization": provider["industry_neutralization"],
    }


def _current_implementation() -> dict:
    root = Path(__file__).resolve().parents[1] / "factor_gfn" / "gfn"
    return {
        "artifact_module_sha256": _sha256_file(root / "train_candidate_artifact.py"),
        "reward_module_sha256": _sha256_file(root / "reward.py"),
        "real_reward_module_sha256": _sha256_file(root / "real_reward.py"),
    }


def _write_json(path: Path, value) -> bytes:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(data)
    return data


class HybridStage6SourceAdapterTests(unittest.TestCase):
    def _make_run(
        self,
        root: Path,
        *,
        complete: bool = True,
        exact_n: bool = False,
        action_registry: ActionRegistry = RAW_ACTION_REGISTRY,
    ) -> tuple[Path, Expression]:
        run = root / "hybrid_test_run"
        run.mkdir()
        expression = (
            Expression.from_prefix(
                (action_registry.get_action_id(action_registry.leaf_names[0]),),
                action_registry=action_registry,
            )
            if exact_n
            else Expression.from_prefix(
                (
                    action_registry.get_action_id("add"),
                    action_registry.get_action_id(action_registry.leaf_names[0]),
                    action_registry.get_action_id(action_registry.leaf_names[1]),
                ),
                action_registry=action_registry,
            ).canonicalize()
        )
        long_excess_dates = [] if exact_n else ["2020-01-01", "2020-01-06"]
        long_excess_values = [] if exact_n else [0.01, None]
        long_valid_periods = 2 if exact_n else 1
        provider_manifest = _provider_manifest()
        provider_fingerprint = _stable_hash(provider_manifest)
        config = {
            "schema": "factor_gfn.hybrid_variance_runner.v1",
            "checkpoint_schema": "factor_gfn.checkpoint.hybrid_variance.v1",
            "objective_mode": "hybrid_variance",
            "config_fingerprint": "a" * 64,
            "reward_provider_fingerprint": provider_fingerprint,
            "train_candidate_artifact": {
                "enabled": True,
                "schema": "factor_gfn.stage5_train_candidate_artifact.v1",
                "filename": "train_candidate_artifact.json",
            },
        }
        config_bytes = _write_json(run / "hybrid_run_config.json", config)
        contract = {
            "schema": "factor_gfn.train_evaluation_contract.v1",
            "provider_fingerprint": provider_fingerprint,
            "train_scope_projection": _train_scope_projection(provider_manifest),
            "implementation": _current_implementation(),
        }
        contract_fingerprint = _stable_hash(contract)
        artifact = {
            "schema": "factor_gfn.stage5_train_candidate_artifact.v1",
            "source_run": {
                "run_directory_name": run.name,
                "hybrid_run_config_sha256": hashlib.sha256(
                    config_bytes
                ).hexdigest(),
            },
            "train_evaluation_contract": contract,
            "train_evaluation_contract_fingerprint": contract_fingerprint,
            "committed_optimizer_step": 3,
            "candidate_count": 1,
            "records": [
                {
                    "schema": "factor_gfn.stage5_train_candidate_record.v1",
                    "structural_hash": expression.structural_hash(),
                    "formula": expression.to_formula(),
                    "prefix_token_ids": list(expression.to_prefix()),
                    "node_count": expression.stats.node_count,
                    "depth": expression.stats.depth,
                    "train_evaluation_contract_fingerprint": contract_fingerprint,
                    "first_seen": {"optimizer_step": 1, "condition_N": 1},
                    "last_seen": {"optimizer_step": 3, "condition_N": 1},
                    "visit_count": 2,
                    "train_ic": 0.02,
                    "train_ic_valid_periods": 2,
                    "train_direction": 1,
                    "train_long_ir": 0.5,
                    "train_long_valid_periods": long_valid_periods,
                    "train_long_excess_dates": long_excess_dates,
                    "train_long_excess_values": long_excess_values,
                    "train_barra_ts_corr": 0.2,
                    "train_barra_correlations": {
                        name: 0.1 for name in STYLE_NAMES
                    },
                    "train_barra_valid_periods_by_style": {
                        name: 2 for name in STYLE_NAMES
                    },
                    "neutralization_diagnostics": {
                        "industry_neutralized": True,
                        "skipped_dates": [],
                        "skipped_rate": 0.0,
                        "details": [],
                    },
                }
            ],
        }
        if action_registry != RAW_ACTION_REGISTRY:
            vocabulary = {
                "feature_space_id": action_registry.feature_space.feature_space_id,
                "feature_space_fingerprint": (
                    action_registry.feature_space_fingerprint
                ),
                "action_space_fingerprint": action_registry.fingerprint(),
            }
            artifact["vocabulary"] = vocabulary
            artifact["records"][0]["vocabulary"] = vocabulary
        _write_json(run / "train_candidate_artifact.json", artifact)
        checkpoint = {
            "schema": "factor_gfn.checkpoint.hybrid_variance.v1",
            "objective_mode": "hybrid_variance",
            "config_fingerprint": "a" * 64,
            "reward_provider_fingerprint": provider_fingerprint,
            "global_optimizer_step": 3,
            "total_trajectories_seen": 48,
        }
        torch.save(checkpoint, run / "checkpoint_latest.pt")
        _write_json(
            run / "runner_state.json",
            {
                "schema": "factor_gfn.hybrid_variance_runner.v1",
                "global_optimizer_step": 3,
                "total_trajectories_seen": 48,
                "pending_assignment": None if complete else {"condition_N": 2},
                "complete": complete,
                "latest_checkpoint": str(run / "checkpoint_latest.pt"),
            },
        )
        return run, expression

    @staticmethod
    def _spec(run: Path, *, source_id: str = "hybrid") -> CandidateSourceSpec:
        return CandidateSourceSpec(
            source_id=source_id,
            source_type="hybrid_train_artifact",
            source_role="formal_discovery",
            source_path=run,
            approval_note="approved completed Hybrid Train artifact",
        )

    def test_incomplete_hybrid_run_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._make_run(root, complete=False)
            with self.assertRaisesRegex(ValueError, "尚未 complete"):
                materialize_candidate_source(self._spec(run), root / "snapshots")

    def test_completed_snapshot_binds_provenance_and_external_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._make_run(root)
            manifest_path = materialize_candidate_source(
                self._spec(run), root / "snapshots"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["snapshot_kind"], "completed_hybrid_train_artifact"
            )
            self.assertEqual(manifest["cutoff"]["committed_optimizer_step"], 3)
            self.assertEqual(manifest["cutoff"]["candidate_count"], 1)
            self.assertEqual(
                manifest["source_semantics"]["generation_config_fingerprint"],
                "a" * 64,
            )
            self.assertEqual(
                manifest["source_semantics"]["provider_fingerprint"],
                _stable_hash(_provider_manifest()),
            )
            self.assertEqual(
                manifest["source_semantics"]["context_fingerprint"], "c" * 64
            )
            self.assertEqual(len(manifest["external_artifacts"]), 1)
            external = manifest["external_artifacts"][0]
            self.assertEqual(external["name"], "checkpoint_latest.pt")
            self.assertFalse((manifest_path.parent / "checkpoint_latest.pt").exists())
            self.assertTrue((manifest_path.parent / "checkpoint_metadata.json").is_file())

    def test_completed_run_step_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._make_run(root)
            artifact_path = run / "train_candidate_artifact.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["committed_optimizer_step"] = 2
            _write_json(artifact_path, artifact)
            with self.assertRaisesRegex(ValueError, "optimizer step 不一致"):
                materialize_candidate_source(self._spec(run), root / "snapshots")

    def test_hybrid_import_preserves_identity_and_valid_is_schema_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, expression = self._make_run(root)
            source_set = materialize_source_set(
                [self._spec(run)], root / "snapshots"
            )
            import_manifest = import_candidate_source_set(
                source_set, root / "registries"
            )
            origin = json.loads(
                (import_manifest.parent / "normalized_candidate_origins.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                origin["source_claimed_structural_hash"], expression.structural_hash()
            )
            rebuilt = Expression.from_prefix(origin["prefix_token_ids"])
            self.assertEqual(rebuilt.structural_hash(), expression.structural_hash())
            self.assertEqual(
                origin["target_node_count_method"],
                "reported_by_hybrid_train_candidate_artifact",
            )
            self.assertTrue(origin["old_metric_audit"]["old_valid"])
            self.assertIsNone(origin["old_metric_audit"]["old_reward"])
            self.assertFalse(
                origin["old_metric_audit"]["reuse_for_stage6_selection"]
            )
            self.assertEqual(
                origin["old_metric_audit"]["valid_semantics"],
                "candidate_import_schema_compatibility_only",
            )
            self.assertNotIn("vocabulary", origin)

            compatibility = audit_expression_compatibility(
                import_manifest, source_set, root / "compatibility"
            )
            accepted = json.loads(
                (compatibility.parent / "auto_accepted_candidate_registry.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                accepted["current_structural_hash"], expression.structural_hash()
            )
            self.assertNotIn("valid", accepted)
            self.assertNotIn("reward", accepted)
            self.assertNotIn("vocabulary", accepted)
            self.assertTrue(accepted["stage6_metric_recompute_required"])

    def test_derived_vocabulary_survives_snapshot_import_and_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, expression = self._make_run(
                root,
                action_registry=DAILY_DERIVED_ACTION_REGISTRY,
            )
            source_set = materialize_source_set(
                [self._spec(run)], root / "snapshots"
            )
            source_set_payload = json.loads(source_set.read_text(encoding="utf-8"))
            source_manifest_path = Path(
                source_set_payload["source_manifests"][0]["snapshot_manifest"]
            )
            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
            vocabulary = source_manifest["source_semantics"]["vocabulary"]
            self.assertEqual(vocabulary["feature_space_id"], "daily_derived_v1")

            candidate_import = import_candidate_source_set(
                source_set, root / "registries"
            )
            origin = json.loads(
                (candidate_import.parent / "normalized_candidate_origins.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            group = json.loads(
                (candidate_import.parent / "candidate_registry.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(origin["vocabulary"], vocabulary)
            self.assertEqual(group["representations"][0]["vocabulary"], vocabulary)

            compatibility = audit_expression_compatibility(
                candidate_import, source_set, root / "compatibility"
            )
            accepted = json.loads(
                (compatibility.parent / "auto_accepted_candidate_registry.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(accepted["vocabulary"], vocabulary)
            rebuilt = Expression.from_prefix(
                accepted["prefix_token_ids"],
                action_registry=DAILY_DERIVED_ACTION_REGISTRY,
            )
            self.assertEqual(rebuilt.structural_hash(), expression.structural_hash())

    def test_hybrid_vocabulary_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._make_run(
                root,
                action_registry=DAILY_DERIVED_ACTION_REGISTRY,
            )
            artifact_path = run / "train_candidate_artifact.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["records"][0]["vocabulary"] = {
                **artifact["vocabulary"],
                "action_space_fingerprint": "0" * 64,
            }
            _write_json(artifact_path, artifact)
            with self.assertRaisesRegex(ValueError, "vocabulary 不一致"):
                materialize_candidate_source(self._spec(run), root / "snapshots")

    def test_exact_contract_artifact_builds_deterministic_v2_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, expression = self._make_run(root)
            source_set = materialize_source_set(
                [self._spec(run)], root / "snapshots"
            )
            candidate_import = import_candidate_source_set(
                source_set, root / "registries"
            )
            compatibility = audit_expression_compatibility(
                candidate_import, source_set, root / "compatibility"
            )
            compatibility_manifest = json.loads(
                compatibility.read_text(encoding="utf-8")
            )
            evaluator = SimpleNamespace(
                context=SimpleNamespace(fingerprint="1" * 64),
                evaluation_contract_fingerprint="2" * 64,
                compatibility_audit_fingerprint=compatibility_manifest[
                    "audit_fingerprint"
                ],
                accepted_registry_fingerprint=compatibility_manifest[
                    "accepted_registry_fingerprint"
                ],
            )
            provider = _provider_manifest()
            provider_fingerprint = _stable_hash(provider)
            first = run_stage6_hybrid_train_reuse_overlay(
                source_set_manifest_path=source_set,
                candidate_import_manifest_path=candidate_import,
                compatibility_manifest_path=compatibility,
                evaluator=evaluator,
                target_provider_manifest=provider,
                target_provider_fingerprint=provider_fingerprint,
                output_root=root / "overlays",
            )
            second = run_stage6_hybrid_train_reuse_overlay(
                source_set_manifest_path=source_set,
                candidate_import_manifest_path=candidate_import,
                compatibility_manifest_path=compatibility,
                evaluator=evaluator,
                target_provider_manifest=provider,
                target_provider_fingerprint=provider_fingerprint,
                output_root=root / "overlays",
            )
            self.assertEqual(first, second)
            manifest = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA)
            self.assertEqual(manifest["verification_mode"], "hybrid_exact_contract")
            self.assertEqual(manifest["counts"]["overlay_candidates"], 1)
            self.assertEqual(manifest["counts"]["numeric_verification_candidates"], 0)
            overlay = Stage6TrainReuseOverlay.load(first)
            record = overlay.get(expression.structural_hash())
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["schema"], HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA)
            self.assertEqual(record["train_metrics"]["train_ic"], 0.02)
            self.assertNotIn("train_long_excess_dates", record["train_metrics"])
            self.assertEqual(
                record["train_long_excess"]["dates"],
                ["2020-01-01", "2020-01-06"],
            )
            self.assertEqual(record["train_long_excess"]["values"], [0.01, None])

    def test_exact_n_missing_long_excess_remains_explicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, expression = self._make_run(root, exact_n=True)
            source_set = materialize_source_set(
                [self._spec(run)], root / "snapshots"
            )
            candidate_import = import_candidate_source_set(
                source_set, root / "registries"
            )
            compatibility = audit_expression_compatibility(
                candidate_import, source_set, root / "compatibility"
            )
            compatibility_manifest = json.loads(
                compatibility.read_text(encoding="utf-8")
            )
            evaluator = SimpleNamespace(
                context=SimpleNamespace(fingerprint="1" * 64),
                evaluation_contract_fingerprint="2" * 64,
                compatibility_audit_fingerprint=compatibility_manifest[
                    "audit_fingerprint"
                ],
                accepted_registry_fingerprint=compatibility_manifest[
                    "accepted_registry_fingerprint"
                ],
            )
            provider = _provider_manifest()
            overlay_path = run_stage6_hybrid_train_reuse_overlay(
                source_set_manifest_path=source_set,
                candidate_import_manifest_path=candidate_import,
                compatibility_manifest_path=compatibility,
                evaluator=evaluator,
                target_provider_manifest=provider,
                target_provider_fingerprint=_stable_hash(provider),
                output_root=root / "overlays",
            )
            record = Stage6TrainReuseOverlay.load(overlay_path).get(
                expression.structural_hash()
            )
            self.assertIsNotNone(record)
            assert record is not None
            self.assertIsNone(record["train_long_excess"])

    def test_train_contract_mismatch_preserves_candidates_for_full_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, expression = self._make_run(root)
            source_set = materialize_source_set(
                [self._spec(run)], root / "snapshots"
            )
            candidate_import = import_candidate_source_set(
                source_set, root / "registries"
            )
            compatibility = audit_expression_compatibility(
                candidate_import, source_set, root / "compatibility"
            )
            compatibility_manifest = json.loads(
                compatibility.read_text(encoding="utf-8")
            )
            evaluator = SimpleNamespace(
                context=SimpleNamespace(fingerprint="1" * 64),
                evaluation_contract_fingerprint="2" * 64,
                compatibility_audit_fingerprint=compatibility_manifest[
                    "audit_fingerprint"
                ],
                accepted_registry_fingerprint=compatibility_manifest[
                    "accepted_registry_fingerprint"
                ],
            )
            current_provider = _provider_manifest()
            current_provider["evaluation_config"] = {
                **current_provider["evaluation_config"],
                "entry_lag": 2,
            }
            manifest_path = run_stage6_hybrid_train_reuse_overlay(
                source_set_manifest_path=source_set,
                candidate_import_manifest_path=candidate_import,
                compatibility_manifest_path=compatibility,
                evaluator=evaluator,
                target_provider_manifest=current_provider,
                target_provider_fingerprint=_stable_hash(current_provider),
                output_root=root / "overlays",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["verification_mode"],
                "hybrid_full_fresh_train_fallback",
            )
            self.assertEqual(
                manifest["fresh_train_fallback_reason"],
                "current_train_contract_mismatch",
            )
            self.assertEqual(manifest["counts"]["accepted_candidates"], 1)
            self.assertEqual(manifest["counts"]["overlay_candidates"], 0)
            self.assertEqual(
                manifest["counts"]["fresh_train_fallback_candidates"], 1
            )
            overlay = Stage6TrainReuseOverlay.load(manifest_path)
            self.assertIsNone(overlay.get(expression.structural_hash()))
            verification = json.loads(
                (manifest_path.parent / "train_reuse_contract_verification.json")
                .read_text(encoding="utf-8")
            )
            self.assertFalse(verification["old_train_metrics_allowed"])
            self.assertEqual(verification["result"], "FULL_FRESH_TRAIN_FALLBACK")
            self.assertIn(
                "train_scope_projection.evaluation_config.entry_lag",
                verification["contract_difference_paths"],
            )
            self.assertNotIn("train_metrics", verification)

    def test_corrupt_n_gt_2_artifact_fails_before_contract_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._make_run(root)
            artifact_path = run / "train_candidate_artifact.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["records"][0]["train_long_excess_dates"] = []
            artifact["records"][0]["train_long_excess_values"] = []
            artifact["records"][0]["train_long_valid_periods"] = 0
            _write_json(artifact_path, artifact)
            source_set = materialize_source_set(
                [self._spec(run)], root / "snapshots"
            )
            candidate_import = import_candidate_source_set(
                source_set, root / "registries"
            )
            compatibility = audit_expression_compatibility(
                candidate_import, source_set, root / "compatibility"
            )
            compatibility_manifest = json.loads(
                compatibility.read_text(encoding="utf-8")
            )
            evaluator = SimpleNamespace(
                context=SimpleNamespace(fingerprint="1" * 64),
                evaluation_contract_fingerprint="2" * 64,
                compatibility_audit_fingerprint=compatibility_manifest[
                    "audit_fingerprint"
                ],
                accepted_registry_fingerprint=compatibility_manifest[
                    "accepted_registry_fingerprint"
                ],
            )
            current_provider = _provider_manifest()
            current_provider["evaluation_config"] = {
                **current_provider["evaluation_config"],
                "entry_lag": 2,
            }
            with self.assertRaisesRegex(RuntimeError, "N>2.*long-excess"):
                run_stage6_hybrid_train_reuse_overlay(
                    source_set_manifest_path=source_set,
                    candidate_import_manifest_path=candidate_import,
                    compatibility_manifest_path=compatibility,
                    evaluator=evaluator,
                    target_provider_manifest=current_provider,
                    target_provider_fingerprint=_stable_hash(current_provider),
                    output_root=root / "overlays",
                )

    def test_structural_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._make_run(root)
            artifact_path = run / "train_candidate_artifact.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["records"][0]["node_count"] += 1
            _write_json(artifact_path, artifact)
            source_set = materialize_source_set(
                [self._spec(run)], root / "snapshots"
            )
            candidate_import = import_candidate_source_set(
                source_set, root / "registries"
            )
            compatibility = audit_expression_compatibility(
                candidate_import, source_set, root / "compatibility"
            )
            compatibility_manifest = json.loads(
                compatibility.read_text(encoding="utf-8")
            )
            evaluator = SimpleNamespace(
                context=SimpleNamespace(fingerprint="1" * 64),
                evaluation_contract_fingerprint="2" * 64,
                compatibility_audit_fingerprint=compatibility_manifest[
                    "audit_fingerprint"
                ],
                accepted_registry_fingerprint=compatibility_manifest[
                    "accepted_registry_fingerprint"
                ],
            )
            provider = _provider_manifest()
            with self.assertRaisesRegex(
                RuntimeError, "not downstream eligible|structurally incompatible"
            ):
                run_stage6_hybrid_train_reuse_overlay(
                    source_set_manifest_path=source_set,
                    candidate_import_manifest_path=candidate_import,
                    compatibility_manifest_path=compatibility,
                    evaluator=evaluator,
                    target_provider_manifest=provider,
                    target_provider_fingerprint=_stable_hash(provider),
                    output_root=root / "overlays",
                )

    def test_hybrid_source_set_cannot_mix_with_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = self._make_run(root)
            second = root / "legacy_run"
            second.mkdir()
            legacy = CandidateSourceSpec(
                source_id="legacy",
                source_type="discovery_run",
                source_role="historical_discovery",
                source_path=second,
                approval_note="legacy source must remain isolated",
            )
            with self.assertRaisesRegex(ValueError, "只包含一个 Hybrid source"):
                materialize_source_set(
                    [self._spec(first), legacy],
                    root / "snapshots",
                )

    def test_external_checkpoint_mutation_blocks_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _ = self._make_run(root)
            source_set = materialize_source_set(
                [self._spec(run)], root / "snapshots"
            )
            with (run / "checkpoint_latest.pt").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(RuntimeError, "external.*指纹不符"):
                import_candidate_source_set(source_set, root / "registries")


if __name__ == "__main__":
    unittest.main()
