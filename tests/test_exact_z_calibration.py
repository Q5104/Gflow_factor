import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from factor_gfn.gfn import (
    ComplexitySchedulerConfig,
    ExactMassResult,
    ExhaustivePlanningConfig,
    ExhaustiveRegistry,
    GFNConfig,
    GFNTrainer,
    ModelConfig,
    NormalizerCalibration,
    NormalizerCalibrationConfig,
    SamplingConfig,
    SyntheticRewardProvider,
    TrainingConfig,
    resolve_exhaustive_plan,
)
from factor_gfn.grammar import SearchSpaceConfig


def _n1_plan():
    return resolve_exhaustive_plan(
        SearchSpaceConfig(max_depth=2, max_nodes=2),
        ExhaustivePlanningConfig(
            planned_real_reward_budget_seconds=5.0,
            max_budget_fraction=1.0,
        ),
    )


def _register_n1(
    registry: ExhaustiveRegistry,
    *,
    raw_rewards: list[float] | None,
    reward_floor: float,
    provider_fingerprint: str = "provider-v1",
    context_fingerprint: str = "context-v1",
) -> None:
    plan = _n1_plan()
    registry.register_plan(
        plan,
        provider_fingerprint=provider_fingerprint,
        context_fingerprint=context_fingerprint,
    )
    candidates = registry.pending_candidates(1)
    if raw_rewards is None:
        for index, candidate in enumerate(candidates):
            registry.record_evaluation(
                candidate.structural_hash,
                valid=False,
                rejection_reason="synthetic invalid",
                reward_details={"index": index},
            )
        return
    if len(raw_rewards) != len(candidates):
        raise ValueError("raw_rewards length must match N=1 candidates")
    for candidate, raw_reward in zip(candidates, raw_rewards, strict=True):
        reward = max(raw_reward, reward_floor)
        registry.record_evaluation(
            candidate.structural_hash,
            valid=True,
            reward_details={
                "reward_result": {
                    "raw_reward": raw_reward,
                    "reward": reward,
                    "log_reward": math.log(reward),
                }
            },
            target_mass=reward,
        )


class ExactRewardMassTests(unittest.TestCase):
    def test_raw_and_tb_masses_are_separate_and_use_stored_tb_log_reward(self):
        raw_rewards = [0.0, 0.05, 0.2, 0.3, 0.4, 0.5]
        floor = 0.1
        with tempfile.TemporaryDirectory() as directory:
            with ExhaustiveRegistry(Path(directory) / "registry.sqlite3") as registry:
                _register_n1(
                    registry,
                    raw_rewards=raw_rewards,
                    reward_floor=floor,
                )
                result = registry.compute_exact_masses(1, reward_floor=floor)
                repeated = registry.compute_exact_masses(1, reward_floor=floor)
        self.assertEqual(result, repeated)
        self.assertEqual(result.raw_reward_mass_status, "positive_mass")
        self.assertAlmostEqual(
            result.exact_raw_reward_log_mass,
            math.log(math.fsum(raw_rewards)),
            places=14,
        )
        self.assertAlmostEqual(
            result.exact_tb_log_z,
            math.log(math.fsum(max(value, floor) for value in raw_rewards)),
            places=14,
        )
        self.assertNotEqual(
            result.exact_raw_reward_log_mass,
            result.exact_tb_log_z,
        )

    def test_zero_raw_mass_is_none_but_tb_mass_remains_finite(self):
        floor = 0.25
        with tempfile.TemporaryDirectory() as directory:
            with ExhaustiveRegistry(Path(directory) / "registry.sqlite3") as registry:
                _register_n1(
                    registry,
                    raw_rewards=[0.0] * 6,
                    reward_floor=floor,
                )
                result = registry.compute_exact_masses(1, reward_floor=floor)
        self.assertIsNone(result.exact_raw_reward_log_mass)
        self.assertEqual(result.raw_reward_mass_status, "zero_mass")
        self.assertAlmostEqual(result.exact_tb_log_z, math.log(6 * floor), places=14)

    def test_all_invalid_fails_closed_without_exact_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExhaustiveRegistry(Path(directory) / "registry.sqlite3") as registry:
                _register_n1(registry, raw_rewards=None, reward_floor=0.1)
                with self.assertRaisesRegex(RuntimeError, "no valid exhaustive candidate"):
                    registry.compute_exact_masses(1, reward_floor=0.1)
                with self.assertRaisesRegex(RuntimeError, "no complete exact mass"):
                    registry.exact_mass_result(1)

    def test_reward_floor_or_log_reward_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExhaustiveRegistry(Path(directory) / "registry.sqlite3") as registry:
                plan = _n1_plan()
                registry.register_plan(
                    plan,
                    provider_fingerprint="provider-v1",
                    context_fingerprint="context-v1",
                )
                for index, candidate in enumerate(registry.pending_candidates(1)):
                    raw = 0.2
                    reward = 0.2
                    registry.record_evaluation(
                        candidate.structural_hash,
                        valid=True,
                        reward_details={
                            "raw_reward": raw,
                            "reward": reward,
                            "log_reward": math.log(reward) + (0.01 if index == 0 else 0.0),
                        },
                        target_mass=reward,
                    )
                with self.assertRaisesRegex(ValueError, "log_reward"):
                    registry.compute_exact_masses(1, reward_floor=0.1)


class CalibrationStatisticsTests(unittest.TestCase):
    def test_median_logmeanexp_quantiles_and_exact_diagnostics(self):
        calibration = NormalizerCalibration(
            node_counts=(1, 2),
            exhaustive_node_counts=(1,),
            minimum_valid_samples=2,
            maximum_requested_slots_per_N=4,
            seed=17,
        )
        values = {1: [1.0, 3.0], 2: [2.0, 6.0]}
        positions = {1: 0, 2: 0}
        while not calibration.ready_to_finalize:
            node_count = calibration.next_node_count()
            index = positions[node_count]
            calibration.record_slot(
                node_count,
                sampled_attempts=1,
                implied_log_z=values[node_count][index],
            )
            positions[node_count] += 1
        stats = calibration.finalize(exact_tb_log_z_by_N={1: 1.5})
        self.assertEqual(stats[1].median, 2.0)
        self.assertEqual(stats[2].median, 4.0)
        self.assertAlmostEqual(
            stats[1].logmeanexp,
            math.log((math.exp(1.0) + math.exp(3.0)) / 2.0),
        )
        self.assertEqual(stats[1].median_implied_minus_exact_tb_log_z, 0.5)
        self.assertIsNone(stats[2].median_implied_minus_exact_tb_log_z)
        self.assertAlmostEqual(stats[1].iqr, 1.0)

    def test_insufficient_valid_samples_fails_at_bounded_request_budget(self):
        calibration = NormalizerCalibration(
            node_counts=(2,),
            exhaustive_node_counts=(),
            minimum_valid_samples=2,
            maximum_requested_slots_per_N=2,
            seed=3,
        )
        node_count = calibration.next_node_count()
        calibration.record_slot(node_count, sampled_attempts=1, implied_log_z=None)
        node_count = calibration.next_node_count()
        with self.assertRaisesRegex(RuntimeError, "only 0 valid samples"):
            calibration.record_slot(
                node_count,
                sampled_attempts=1,
                implied_log_z=None,
            )
        self.assertEqual(calibration.status, "failed")

    def test_state_round_trip_preserves_scheduler_and_observations(self):
        source = NormalizerCalibration(
            node_counts=(1, 2, 3),
            exhaustive_node_counts=(1,),
            minimum_valid_samples=3,
            maximum_requested_slots_per_N=5,
            seed=29,
        )
        first = source.next_node_count()
        source.record_slot(first, sampled_attempts=2, implied_log_z=4.0)
        state = source.state_dict()
        expected = [source.next_node_count() for _ in range(8)]
        resumed = NormalizerCalibration(
            node_counts=(1, 2, 3),
            exhaustive_node_counts=(1,),
            minimum_valid_samples=3,
            maximum_requested_slots_per_N=5,
            seed=999,
        )
        resumed.load_state_dict(state)
        actual = [resumed.next_node_count() for _ in range(8)]
        self.assertEqual(actual, expected)
        self.assertEqual(resumed.implied_log_z_by_N[first], [4.0])
        self.assertEqual(resumed.sampled_attempts_by_N[first], 2)


def _calibration_config(*, minimum: int = 2, maximum: int = 4) -> GFNConfig:
    return GFNConfig(
        search_space=SearchSpaceConfig(max_depth=2, max_nodes=2),
        model=ModelConfig(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            token_policy_mode="grammar_hierarchical",
        ),
        sampling=SamplingConfig(),
        complexity_scheduler=ComplexitySchedulerConfig(
            enabled=True,
            exhaustive_node_counts=(1,),
            exact_node_retry_budget=0,
        ),
        calibration=NormalizerCalibrationConfig(
            enabled=True,
            minimum_valid_calibration_samples=minimum,
            maximum_requested_calibration_slots_per_N=maximum,
        ),
        training=TrainingConfig(
            batch_size=1,
            learning_rate=1e-3,
            log_z_learning_rate=1e-2,
            max_steps=3,
            seed=812,
        ),
    )


def _synthetic_exact_mass(provider: SyntheticRewardProvider) -> ExactMassResult:
    return ExactMassResult(
        node_count=1,
        valid_candidate_count=6,
        invalid_candidate_count=0,
        exact_raw_reward_log_mass=1.0,
        raw_reward_mass_status="positive_mass",
        exact_tb_log_z=1.25,
        reward_floor=1e-8,
        provider_fingerprint=provider.fingerprint(),
        context_fingerprint=provider.manifest()["context_fingerprint"],
        aggregation_fingerprint="synthetic-exact-n1",
    )


class TrainerCalibrationIntegrationTests(unittest.TestCase):
    def test_calibration_freezes_policy_and_discovery_scheduler_then_initializes_median(self):
        provider = SyntheticRewardProvider()
        trainer = GFNTrainer(_calibration_config(), provider, device="cpu")
        trainer.register_exact_mass_result(_synthetic_exact_mass(provider))
        policy_before = {
            key: value.detach().clone() for key, value in trainer.model.state_dict().items()
        }
        discovery_before = trainer.complexity_scheduler.state_dict()
        with self.assertRaisesRegex(RuntimeError, "must complete before training"):
            trainer.train_step()
        report = None
        while report is None:
            report = trainer.calibration_step()
        self.assertEqual(trainer.optimizer_step, 0)
        self.assertFalse(trainer.optimizer.state)
        self.assertEqual(trainer.complexity_scheduler.state_dict(), discovery_before)
        for key, before in policy_before.items():
            self.assertTrue(torch.equal(trainer.model.state_dict()[key], before), key)
        self.assertEqual(trainer.tb_loss.log_z_by_node_count.dtype, torch.float32)
        self.assertEqual(
            trainer.tb_loss.exact_tb_log_z_by_node_count.dtype,
            torch.float64,
        )
        self.assertTrue(bool(trainer.tb_loss.exact_log_z_mask[0]))
        self.assertFalse(bool(trainer.tb_loss.learned_log_z_initialized_mask[0]))
        self.assertTrue(bool(trainer.tb_loss.learned_log_z_initialized_mask[1]))
        expected = torch.tensor(report[2].median, dtype=torch.float32).item()
        self.assertEqual(float(trainer.tb_loss.log_z_by_node_count[1]), expected)
        self.assertAlmostEqual(
            report[1].median_implied_minus_exact_tb_log_z,
            report[1].median - 1.25,
        )
        stats = trainer.train_step()
        self.assertFalse(stats.skipped_update)
        self.assertEqual(trainer.optimizer_step, 1)

    def test_provider_without_explicit_training_only_scope_is_rejected(self):
        class UnsafeProvider(SyntheticRewardProvider):
            def manifest(self):
                manifest = super().manifest()
                manifest.pop("data_scope")
                return manifest

        with self.assertRaisesRegex(ValueError, "data_scope=training_only"):
            GFNTrainer(_calibration_config(), UnsafeProvider(), device="cpu")

    def test_initialization_is_forbidden_after_optimizer_step(self):
        provider = SyntheticRewardProvider()
        trainer = GFNTrainer(_calibration_config(), provider, device="cpu")
        trainer.register_exact_mass_result(_synthetic_exact_mass(provider))
        trainer.optimizer_step = 1
        with self.assertRaisesRegex(RuntimeError, "optimizer_step == 0"):
            trainer.calibration_step()


class CalibrationConfigTests(unittest.TestCase):
    def test_diagnostic_defaults_and_invalid_budget(self):
        config = NormalizerCalibrationConfig()
        self.assertEqual(config.minimum_valid_calibration_samples, 64)
        self.assertEqual(config.maximum_requested_calibration_slots_per_N, 128)
        with self.assertRaisesRegex(ValueError, "must be at least"):
            NormalizerCalibrationConfig(
                enabled=True,
                minimum_valid_calibration_samples=4,
                maximum_requested_calibration_slots_per_N=3,
            )


if __name__ == "__main__":
    unittest.main()
