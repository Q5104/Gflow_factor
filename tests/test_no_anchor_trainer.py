import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from factor_gfn.gfn.config import ModelConfig, TrainingConfig
from factor_gfn.gfn.exhaustive import (
    ExhaustivePlanningConfig,
    ExhaustiveRegistry,
    resolve_exhaustive_plan,
)
from factor_gfn.gfn.no_anchor_config import (
    NoAnchorCalibrationConfig,
    NoAnchorComplexityConfig,
    NoAnchorGFNConfig,
)
from factor_gfn.gfn.targeted_calibration import (
    load_targeted_calibration_progress,
    save_targeted_calibration_progress,
    write_high_variance_engineering_artifact_from_progress,
    write_targeted_calibration_artifact,
)
from factor_gfn.gfn.trainer import GFNTrainer, SyntheticRewardProvider
from factor_gfn.grammar import Expression, SearchSpaceConfig


class CountingSyntheticRewardProvider(SyntheticRewardProvider):
    def __init__(self) -> None:
        super().__init__()
        self.evaluate_count = 0

    def evaluate(self, expression):
        self.evaluate_count += 1
        return super().evaluate(expression)


def _config(*, max_steps: int = 3) -> NoAnchorGFNConfig:
    return NoAnchorGFNConfig(
        search_space=SearchSpaceConfig(max_depth=2, max_nodes=2),
        model=ModelConfig(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            token_policy_mode="grammar_hierarchical",
        ),
        complexity=NoAnchorComplexityConfig(
            exact_normalizer_node_counts=(1,),
            exact_node_retry_budget=0,
        ),
        calibration=NoAnchorCalibrationConfig(
            enabled=True,
            target_node_counts=(2,),
        ),
        training=TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            log_z_learning_rate=1e-2,
            max_steps=max_steps,
            seed=20260813,
        ),
    )


def _build_registry(
    path: Path,
    provider: CountingSyntheticRewardProvider,
    *,
    node_counts: tuple[int, ...] = (1,),
) -> ExhaustiveRegistry:
    registry = ExhaustiveRegistry(path)
    plan = resolve_exhaustive_plan(
        SearchSpaceConfig(max_depth=2, max_nodes=2),
        ExhaustivePlanningConfig(),
    )
    registry.register_plan(
        plan,
        provider_fingerprint=provider.fingerprint(),
        context_fingerprint=provider.manifest()["context_fingerprint"],
    )
    for node_count in node_counts:
        for candidate in registry.pending_candidates(node_count):
            expression = Expression.from_prefix(candidate.prefix_token_ids)
            assignment = provider.evaluate(expression)
            registry.record_evaluation(
                candidate.structural_hash,
                valid=True,
                reward_details={
                    "reward_result": {
                        "raw_reward": assignment.reward,
                        "reward": assignment.reward,
                        "log_reward": assignment.log_reward,
                    }
                },
                target_mass=assignment.reward,
            )
        registry.compute_exact_masses(node_count, reward_floor=1e-8)
    provider.evaluate_count = 0
    return registry


def _write_historical_diagnostic(
    root: Path,
    trainer: GFNTrainer,
    registry: ExhaustiveRegistry,
    *,
    medians: dict[int, float],
    mutate_checkpoint=None,
) -> None:
    root.mkdir(parents=True)
    source_config_fingerprint = "approved-old-diagnostic-config"
    provider_manifest = trainer.reward_provider.manifest()
    provider_fingerprint = trainer.reward_provider.fingerprint()
    context_fingerprint = provider_manifest["context_fingerprint"]
    F = list(trainer.resolved_feasible_node_counts)
    E = list(trainer.resolved_exhaustive_node_counts)
    L = list(trainer.resolved_learned_node_counts)
    summary = {
        "schema": "factor_gfn.conditional_diagnostic.v1",
        "config_fingerprint": source_config_fingerprint,
        "resolved_F": F,
        "resolved_E": E,
        "resolved_S": L,
        "training_only": True,
        "validation_oos_not_loaded": True,
        "industry_neutralization_on": True,
        "exact_by_N": {
            str(node_count): asdict(registry.exact_mass_result(node_count))
            for node_count in E
        },
        "calibration_by_N": {
            str(node_count): {
                "node_count": node_count,
                "calibration_requested": 18,
                "calibration_valid": 17,
                "calibration_sampled_attempts": 20,
                "median": medians[node_count],
                "logmeanexp": medians[node_count] + 0.5,
                "p10": medians[node_count] - 1.0,
                "p25": medians[node_count] - 0.5,
                "p75": medians[node_count] + 0.5,
                "p90": medians[node_count] + 1.0,
                "iqr": 1.0,
                "median_implied_minus_exact_tb_log_z": None,
                "logmeanexp_implied_minus_exact_tb_log_z": None,
            }
            for node_count in L
        },
    }
    context = {
        "schema": "factor_gfn.conditional_diagnostic.v1",
        "config_fingerprint": source_config_fingerprint,
        "provider_fingerprint": provider_fingerprint,
        "context_fingerprint": context_fingerprint,
        "search_space": asdict(trainer.config.search_space),
        "resolved_complexity": {
            "resolved_feasible_node_counts": F,
            "resolved_exhaustive_node_counts": E,
            "resolved_discovery_node_counts": L,
        },
    }
    config_manifest = trainer.config.manifest()
    checkpoint = {
        "schema": "factor_gfn.checkpoint.v5",
        "config_fingerprint": source_config_fingerprint,
        "reward_provider_fingerprint": provider_fingerprint,
        "run_id": "approved-old-diagnostic-run",
        "run_metadata": {
            "config_manifest": config_manifest,
            "reward_provider": provider_manifest,
        },
        # These fields prove that the reader tolerates but never imports the old state.
        "model_state": {"forbidden": "old-model"},
        "optimizer_state": {"forbidden": "old-optimizer"},
        "anchor_state": {"forbidden": "old-anchor"},
    }
    if mutate_checkpoint is not None:
        mutate_checkpoint(checkpoint)
    (root / "diagnostic_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (root / "diagnostic_context.json").write_text(
        json.dumps(context), encoding="utf-8"
    )
    torch.save(checkpoint, root / "diagnostic_checkpoint.pt")


class NoAnchorTrainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.provider = CountingSyntheticRewardProvider()
        self.registry = _build_registry(
            Path(self.temporary.name) / "registry.sqlite3",
            self.provider,
        )

    def tearDown(self):
        self.registry.close()
        self.temporary.cleanup()

    def _configured_trainer(self) -> GFNTrainer:
        trainer = GFNTrainer(_config(), self.provider, device="cpu")
        semantics = trainer.target_exhaustive_reuse_semantics()
        proofs = trainer.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        self.assertEqual(set(proofs), {1})
        return trainer

    @staticmethod
    def _complete_constant_calibration(trainer: GFNTrainer) -> None:
        assert trainer.calibration is not None
        for _ in range(64):
            node_count = trainer.calibration.next_node_count()
            if node_count != 2:
                raise AssertionError("no-anchor calibration must only schedule L")
            trainer.calibration.record_slot(
                node_count,
                sampled_attempts=1,
                implied_log_z=1.5,
            )
        trainer._finalize_calibration()

    def test_scheduler_is_D_equals_F_but_calibration_is_L_only(self):
        trainer = self._configured_trainer()
        self.assertEqual(trainer.resolved_feasible_node_counts, (1, 2))
        self.assertEqual(trainer.resolved_discovery_node_counts, (1, 2))
        self.assertEqual(trainer.resolved_exhaustive_node_counts, (1,))
        self.assertEqual(trainer.resolved_learned_node_counts, (2,))
        self.assertEqual(trainer.calibration.node_counts, (2,))
        self.assertEqual(trainer.calibration.exhaustive_node_counts, ())
        self._complete_constant_calibration(trainer)
        self.assertEqual(set(trainer.calibration_report()), {2})
        self.assertTrue(bool(trainer.tb_loss.learned_log_z_initialized_mask[1]))
        self.assertFalse(bool(trainer.tb_loss.learned_log_z_initialized_mask[0]))

    def test_exact_discovery_uses_hash_lookup_and_mixed_batch_updates_correctly(self):
        trainer = self._configured_trainer()
        self._complete_constant_calibration(trainer)
        exact_buffer_before = trainer.tb_loss.exact_tb_log_z_by_node_count.clone()
        learned_before = trainer.tb_loss.log_z_by_node_count.detach().clone()
        policy_before = [item.detach().clone() for item in trainer.model.parameters()]
        stats = trainer.train_step()
        self.assertFalse(stats.skipped_update)
        self.assertEqual(stats.requested_count_by_N, {1: 1, 2: 1})
        # Only N=2 reaches the provider; N=1 is served from the proven registry.
        self.assertEqual(self.provider.evaluate_count, 1)
        self.assertTrue(
            torch.equal(
                exact_buffer_before,
                trainer.tb_loss.exact_tb_log_z_by_node_count,
            )
        )
        self.assertEqual(
            float(trainer.tb_loss.log_z_by_node_count[0]),
            float(learned_before[0]),
        )
        self.assertNotEqual(
            float(trainer.tb_loss.log_z_by_node_count[1]),
            float(learned_before[1]),
        )
        gradient = trainer.tb_loss.log_z_by_node_count.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(float(gradient[0]), 0.0)
        self.assertNotEqual(float(gradient[1]), 0.0)
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    policy_before,
                    trainer.model.parameters(),
                    strict=True,
                )
            )
        )

    def test_equivalence_is_single_initialization_and_training_fails_without_it(self):
        unconfigured = GFNTrainer(_config(), self.provider, device="cpu")
        unconfigured.register_exact_mass_result(self.registry.exact_mass_result(1))
        self._complete_constant_calibration(unconfigured)
        with self.assertRaisesRegex(RuntimeError, "verified registry lookup"):
            unconfigured.train_step()

        trainer = self._configured_trainer()
        with self.assertRaisesRegex(RuntimeError, "already verified"):
            semantics = trainer.target_exhaustive_reuse_semantics()
            trainer.configure_no_anchor_exhaustive_registry(
                self.registry,
                source_semantics_by_N={1: semantics},
            )

    def test_target_semantics_are_runtime_derived_and_mismatch_is_atomic(self):
        trainer = GFNTrainer(_config(), self.provider, device="cpu")
        target = trainer.target_exhaustive_reuse_semantics()
        changed_source = replace(
            target,
            interpreter_semantics_fingerprint="different-interpreter",
        )
        with self.assertRaisesRegex(ValueError, "semantics mismatch"):
            trainer.configure_no_anchor_exhaustive_registry(
                self.registry,
                source_semantics_by_N={1: changed_source},
            )
        self.assertEqual(trainer.registered_exact_masses_by_N, {})
        self.assertEqual(trainer.exhaustive_reward_lookups_by_N, {})

    def test_verified_historical_median_initializes_constants_without_old_state(self):
        config = replace(_config(), calibration=NoAnchorCalibrationConfig())
        trainer = GFNTrainer(config, self.provider, device="cpu")
        semantics = trainer.target_exhaustive_reuse_semantics()
        trainer.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        source = Path(self.temporary.name) / "historical_valid"
        _write_historical_diagnostic(
            source,
            trainer,
            self.registry,
            medians={2: 7.25},
        )
        model_before = {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        }
        record = trainer.initialize_verified_historical_log_z(source)
        self.assertEqual(record.learned_node_counts, (2,))
        self.assertEqual(record.median_log_z_by_N, {2: 7.25})
        self.assertAlmostEqual(float(trainer.tb_loss.log_z_by_node_count[1]), 7.25)
        self.assertEqual(trainer.optimizer.state, {})
        self.assertEqual(trainer.step, 0)
        self.assertEqual(trainer.optimizer_step, 0)
        self.assertTrue(all(
            torch.equal(model_before[key], value)
            for key, value in trainer.model.state_dict().items()
        ))
        self.assertEqual(self.provider.evaluate_count, 0)
        stats = trainer.train_step()
        self.assertFalse(stats.skipped_update)
        checkpoint = Path(self.temporary.name) / "historical_no_anchor.pt"
        trainer.save_checkpoint(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(
            payload["historical_log_z_initialization"]["provenance_fingerprint"],
            record.provenance_fingerprint,
        )
        resumed = GFNTrainer(config, self.provider, device="cpu")
        resumed_semantics = resumed.target_exhaustive_reuse_semantics()
        resumed.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: resumed_semantics},
        )
        resumed.load_checkpoint(checkpoint)
        self.assertEqual(
            resumed.historical_log_z_initialization.provenance_fingerprint,
            record.provenance_fingerprint,
        )
        with self.assertRaisesRegex(RuntimeError, "fresh training state"):
            trainer.initialize_verified_historical_log_z(source)

    def test_historical_semantics_mismatch_fails_before_any_log_z_write(self):
        config = replace(_config(), calibration=NoAnchorCalibrationConfig())
        trainer = GFNTrainer(config, self.provider, device="cpu")
        source = Path(self.temporary.name) / "historical_mismatch"

        def mutate(checkpoint):
            checkpoint["run_metadata"]["config_manifest"][
                "token_space_fingerprint"
            ] = "different-operator-space"

        _write_historical_diagnostic(
            source,
            trainer,
            self.registry,
            medians={2: 7.25},
            mutate_checkpoint=mutate,
        )
        before = trainer.tb_loss.log_z_by_node_count.detach().clone()
        with self.assertRaisesRegex(ValueError, "semantics mismatch"):
            trainer.initialize_verified_historical_log_z(source)
        self.assertTrue(torch.equal(before, trainer.tb_loss.log_z_by_node_count))
        self.assertIsNone(trainer.historical_log_z_initialization)
        self.assertFalse(bool(trainer.tb_loss.learned_log_z_initialized_mask[1]))

    def test_targeted_fallback_recalibrates_only_named_N_before_fresh_training(self):
        config = replace(
            _config(),
            search_space=SearchSpaceConfig(max_depth=2, max_nodes=3),
            calibration=NoAnchorCalibrationConfig(
                enabled=True,
                target_node_counts=(3,),
            ),
        )
        trainer = GFNTrainer(config, self.provider, device="cpu")
        semantics = trainer.target_exhaustive_reuse_semantics()
        trainer.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        source = Path(self.temporary.name) / "historical_targeted"
        _write_historical_diagnostic(
            source,
            trainer,
            self.registry,
            medians={2: 7.25, 3: 11.0},
        )
        record = trainer.initialize_verified_historical_log_z(source)
        self.assertEqual(record.learned_node_counts, (2,))
        self.assertEqual(trainer.calibration.node_counts, (3,))
        self.assertTrue(bool(trainer.tb_loss.learned_log_z_initialized_mask[1]))
        self.assertFalse(bool(trainer.tb_loss.learned_log_z_initialized_mask[2]))
        with self.assertRaisesRegex(RuntimeError, "calibration must complete"):
            trainer.train_step()
        for _ in range(64):
            self.assertEqual(trainer.calibration.next_node_count(), 3)
            trainer.calibration.record_slot(
                3,
                sampled_attempts=1,
                implied_log_z=9.5,
            )
        trainer._finalize_calibration()
        self.assertAlmostEqual(float(trainer.tb_loss.log_z_by_node_count[1]), 7.25)
        self.assertAlmostEqual(float(trainer.tb_loss.log_z_by_node_count[2]), 9.5)
        self.assertEqual(trainer.step, 0)
        self.assertEqual(trainer.optimizer_step, 0)
        self.assertEqual(trainer.optimizer.state, {})

    def _targeted_artifact_fixture(self):
        config = replace(
            _config(),
            search_space=SearchSpaceConfig(max_depth=2, max_nodes=3),
            complexity=NoAnchorComplexityConfig(
                exact_normalizer_node_counts=(1,),
                exact_node_retry_budget=3,
            ),
            calibration=NoAnchorCalibrationConfig(
                enabled=True,
                target_node_counts=(3,),
            ),
        )
        trainer = GFNTrainer(config, self.provider, device="cpu")
        semantics = trainer.target_exhaustive_reuse_semantics()
        trainer.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        source = Path(self.temporary.name) / "historical_artifact"
        _write_historical_diagnostic(
            source,
            trainer,
            self.registry,
            medians={2: 7.25, 3: 11.0},
        )
        trainer.initialize_verified_historical_log_z(source)
        return config, trainer, source

    def test_targeted_progress_resume_preserves_only_calibration_state(self):
        config, trainer, source = self._targeted_artifact_fixture()
        for _ in range(10):
            trainer.calibration.record_slot(
                3,
                sampled_attempts=1,
                implied_log_z=9.5,
            )
        progress = Path(self.temporary.name) / "targeted_progress.pt"
        save_targeted_calibration_progress(progress, trainer)
        payload = torch.load(progress, map_location="cpu", weights_only=False)
        self.assertNotIn("model_state", payload)
        self.assertNotIn("optimizer_state", payload)

        resumed = GFNTrainer(config, self.provider, device="cpu")
        semantics = resumed.target_exhaustive_reuse_semantics()
        resumed.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        resumed.initialize_verified_historical_log_z(source)
        load_targeted_calibration_progress(progress, resumed)
        self.assertEqual(resumed.calibration.state_dict(), trainer.calibration.state_dict())
        self.assertEqual(resumed.step, 0)
        self.assertEqual(resumed.optimizer_step, 0)
        self.assertEqual(resumed.optimizer.state, {})

    def test_completed_targeted_artifact_initializes_only_target_and_round_trips(self):
        config, trainer, source = self._targeted_artifact_fixture()
        for _ in range(64):
            trainer.calibration.record_slot(
                3,
                sampled_attempts=1,
                implied_log_z=9.5,
            )
        trainer._finalize_calibration()
        completed_progress = Path(self.temporary.name) / "completed_progress.pt"
        save_targeted_calibration_progress(completed_progress, trainer)
        artifact = Path(self.temporary.name) / "targeted_result.json"
        write_targeted_calibration_artifact(artifact, trainer)

        progress_resumed = GFNTrainer(config, self.provider, device="cpu")
        progress_semantics = progress_resumed.target_exhaustive_reuse_semantics()
        progress_resumed.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: progress_semantics},
        )
        progress_resumed.initialize_verified_historical_log_z(source)
        load_targeted_calibration_progress(completed_progress, progress_resumed)
        self.assertEqual(progress_resumed.calibration.status, "complete")
        self.assertAlmostEqual(
            float(progress_resumed.tb_loss.log_z_by_node_count[2]), 9.5
        )

        fresh = GFNTrainer(config, self.provider, device="cpu")
        semantics = fresh.target_exhaustive_reuse_semantics()
        fresh.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        fresh.initialize_verified_historical_log_z(source)
        model_before = {
            key: value.detach().clone() for key, value in fresh.model.state_dict().items()
        }
        record = fresh.initialize_verified_targeted_log_z(artifact)
        self.assertEqual(record.target_node_counts, (3,))
        self.assertAlmostEqual(float(fresh.tb_loss.log_z_by_node_count[1]), 7.25)
        self.assertAlmostEqual(float(fresh.tb_loss.log_z_by_node_count[2]), 9.5)
        self.assertEqual(fresh.step, 0)
        self.assertEqual(fresh.optimizer_step, 0)
        self.assertEqual(fresh.optimizer.state, {})
        for key, value in fresh.model.state_dict().items():
            self.assertTrue(torch.equal(value, model_before[key]), key)

        checkpoint = Path(self.temporary.name) / "targeted_initialized.pt"
        fresh.save_checkpoint(checkpoint)
        restored = GFNTrainer(config, self.provider, device="cpu")
        restored_semantics = restored.target_exhaustive_reuse_semantics()
        restored.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: restored_semantics},
        )
        restored.load_checkpoint(checkpoint)
        self.assertEqual(
            restored.targeted_log_z_initialization.artifact_fingerprint,
            record.artifact_fingerprint,
        )

    def test_targeted_artifact_context_mismatch_fails_before_log_z_write(self):
        config, trainer, source = self._targeted_artifact_fixture()
        for _ in range(64):
            trainer.calibration.record_slot(
                3,
                sampled_attempts=1,
                implied_log_z=9.5,
            )
        trainer._finalize_calibration()
        artifact = Path(self.temporary.name) / "targeted_mismatch.json"
        write_targeted_calibration_artifact(artifact, trainer)

        changed = replace(
            config,
            complexity=replace(config.complexity, exact_node_retry_budget=2),
        )
        fresh = GFNTrainer(changed, self.provider, device="cpu")
        semantics = fresh.target_exhaustive_reuse_semantics()
        fresh.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        fresh.initialize_verified_historical_log_z(source)
        before = fresh.tb_loss.log_z_by_node_count.detach().clone()
        with self.assertRaisesRegex(ValueError, "context mismatch"):
            fresh.initialize_verified_targeted_log_z(artifact)
        self.assertTrue(torch.equal(before, fresh.tb_loss.log_z_by_node_count))
        self.assertFalse(bool(fresh.tb_loss.learned_log_z_initialized_mask[2]))
        self.assertIsNone(fresh.targeted_log_z_initialization)

    def test_high_variance_engineering_artifact_is_explicit_and_importable(self):
        config, trainer, source = self._targeted_artifact_fixture()
        for index in range(127):
            trainer.calibration.record_slot(
                3,
                sampled_attempts=1,
                implied_log_z=0.0 if index < 111 else 10.0,
            )
        progress = Path(self.temporary.name) / "high_variance_progress.pt"
        save_targeted_calibration_progress(progress, trainer)
        artifact = Path(self.temporary.name) / "high_variance_result.json"
        payload = write_high_variance_engineering_artifact_from_progress(
            progress,
            artifact,
            approval_reason="human-approved health-run engineering initialization",
        )
        self.assertEqual(
            payload["initialization_status"],
            "high_variance_engineering_estimate",
        )
        self.assertEqual(payload["strict_stability_check"], "failed")
        self.assertEqual(payload["calibration_statistics_by_N"]["3"]["calibration_valid"], 127)
        self.assertNotEqual(
            payload["calibration_stability_by_N"]["3"]["status"], "stable"
        )

        fresh = GFNTrainer(config, self.provider, device="cpu")
        semantics = fresh.target_exhaustive_reuse_semantics()
        fresh.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        fresh.initialize_verified_historical_log_z(source)
        record = fresh.initialize_verified_targeted_log_z(artifact)
        self.assertEqual(
            record.initialization_status,
            "high_variance_engineering_estimate",
        )
        self.assertEqual(record.strict_stability_check, "failed")
        self.assertEqual(fresh.calibration.status, "complete")
        self.assertAlmostEqual(
            float(fresh.tb_loss.log_z_by_node_count[2]),
            float(payload["median_log_z_by_N"]["3"]),
        )

    def test_training_and_checkpoint_fail_until_all_learned_log_z_are_initialized(self):
        config = replace(_config(), calibration=NoAnchorCalibrationConfig())
        trainer = GFNTrainer(config, self.provider, device="cpu")
        semantics = trainer.target_exhaustive_reuse_semantics()
        trainer.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        with self.assertRaisesRegex(RuntimeError, "initialization is incomplete"):
            trainer.train_step()
        path = Path(self.temporary.name) / "must_not_exist.pt"
        with self.assertRaisesRegex(RuntimeError, "initialization is incomplete"):
            trainer.save_checkpoint(path)
        self.assertFalse(path.exists())

    def test_no_anchor_checkpoint_round_trip_is_deterministic(self):
        trainer = self._configured_trainer()
        self._complete_constant_calibration(trainer)
        trainer.train_step()
        path = Path(self.temporary.name) / "no_anchor.pt"
        trainer.save_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["schema"], "factor_gfn.checkpoint.no_anchor.v1")
        self.assertFalse(
            {"anchor_state", "anchor_optimizer_step", "total_policy_optimizer_step"}
            & set(payload)
        )
        expected_stats = trainer.train_step()
        expected_diagnostics = trainer.last_discovery_trajectory_diagnostics
        expected_model = {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        }
        expected_log_z = trainer.tb_loss.log_z_by_node_count.detach().clone()

        resumed = GFNTrainer(_config(), self.provider, device="cpu")
        semantics = resumed.target_exhaustive_reuse_semantics()
        resumed.configure_no_anchor_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics},
        )
        resumed.load_checkpoint(path)
        actual_stats = resumed.train_step()
        self.assertEqual(actual_stats, expected_stats)
        self.assertEqual(
            resumed.last_discovery_trajectory_diagnostics,
            expected_diagnostics,
        )
        self.assertTrue(torch.equal(resumed.tb_loss.log_z_by_node_count, expected_log_z))
        for key, expected in expected_model.items():
            self.assertTrue(torch.equal(resumed.model.state_dict()[key], expected), key)

    def test_no_anchor_rejects_legacy_schema_scalar_and_anchor_fields(self):
        source = self._configured_trainer()
        self._complete_constant_calibration(source)
        path = Path(self.temporary.name) / "schema_rejection.pt"
        source.save_checkpoint(path)
        clean = torch.load(path, map_location="cpu", weights_only=False)

        def resumed_trainer():
            resumed = GFNTrainer(_config(), self.provider, device="cpu")
            semantics = resumed.target_exhaustive_reuse_semantics()
            resumed.configure_no_anchor_exhaustive_registry(
                self.registry,
                source_semantics_by_N={1: semantics},
            )
            return resumed

        legacy = dict(clean)
        legacy["schema"] = "factor_gfn.checkpoint.v5"
        torch.save(legacy, path)
        with self.assertRaisesRegex(ValueError, "rejects legacy v1-v5"):
            resumed_trainer().load_checkpoint(path)

        anchored = dict(clean)
        anchored["anchor_state"] = {"configured": True}
        torch.save(anchored, path)
        with self.assertRaisesRegex(ValueError, "rejects anchor fields"):
            resumed_trainer().load_checkpoint(path)

        scalar = dict(clean)
        scalar["normalizer_manifest"] = {
            **clean["normalizer_manifest"],
            "mode": "legacy_scalar",
        }
        torch.save(scalar, path)
        with self.assertRaisesRegex(ValueError, "normalizer"):
            resumed_trainer().load_checkpoint(path)

    def test_N1_N2_equivalence_is_verified_once_with_complete_hash_sets(self):
        provider = CountingSyntheticRewardProvider()
        registry = _build_registry(
            Path(self.temporary.name) / "registry_n1_n2.sqlite3",
            provider,
            node_counts=(1, 2),
        )
        try:
            config = NoAnchorGFNConfig(
                search_space=SearchSpaceConfig(max_depth=6, max_nodes=20),
                model=ModelConfig(
                    d_model=16,
                    num_heads=4,
                    num_layers=1,
                    dim_feedforward=32,
                    token_policy_mode="grammar_hierarchical",
                ),
                training=TrainingConfig(max_steps=1, seed=7),
            )
            trainer = GFNTrainer(config, provider, device="cpu")
            semantics = trainer.target_exhaustive_reuse_semantics()
            proofs = trainer.configure_no_anchor_exhaustive_registry(
                registry,
                source_semantics_by_N={1: semantics, 2: semantics},
            )
            self.assertEqual(len(proofs[1].canonical_structural_hashes), 6)
            self.assertEqual(len(proofs[2].canonical_structural_hashes), 636)
            self.assertEqual(provider.evaluate_count, 0)
            self.assertEqual(set(trainer.exhaustive_reward_lookups_by_N), {1, 2})
        finally:
            registry.close()


if __name__ == "__main__":
    unittest.main()
