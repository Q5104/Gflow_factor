import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from factor_gfn.gfn import (
    CUBLAS_WORKSPACE_CONFIG,
    GFNConfig,
    GFNTrainer,
    ModelConfig,
    RewardAssignment,
    SamplingConfig,
    SearchSpaceConfig,
    SyntheticRewardProvider,
    TrainingConfig,
    configure_cuda_determinism,
)


def _config(*, token_policy_mode="flat", **training_overrides) -> GFNConfig:
    training = {
        "batch_size": 4,
        "learning_rate": 1e-3,
        "log_z_learning_rate": 1e-2,
        "max_steps": 20,
        "model_gradient_clip_norm": 1.0,
        "log_z_gradient_clip_norm": 1.0,
        "seed": 2026,
    }
    training.update(training_overrides)
    return GFNConfig(
        search_space=SearchSpaceConfig(max_depth=2, max_nodes=5),
        model=ModelConfig(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            token_policy_mode=token_policy_mode,
        ),
        sampling=SamplingConfig(temperature=1.0, greedy=False),
        training=TrainingConfig(**training),
    )


def _assert_nested_equal(test: unittest.TestCase, left, right) -> None:
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            _assert_nested_equal(test, first, second)
    else:
        test.assertEqual(left, right)


class _RejectAllProvider:
    def evaluate(self, expression):
        return RewardAssignment(valid=False, rejection_reason="synthetic rejection")

    def manifest(self):
        return {"schema": "tests.reject_all.v1"}

    def fingerprint(self):
        return "reject-all-v1"


class TrainerLoopTests(unittest.TestCase):
    def test_grammar_hierarchical_training_records_all_levels(self):
        trainer = GFNTrainer(
            _config(token_policy_mode="grammar_hierarchical"),
            SyntheticRewardProvider(),
            device="cpu",
        )
        stats = trainer.train_step()
        category_probabilities = (
            stats.feature_category_probability_mean,
            stats.unary_category_probability_mean,
            stats.ts_unary_category_probability_mean,
            stats.binary_category_probability_mean,
            stats.ts_binary_category_probability_mean,
            stats.cross_sectional_category_probability_mean,
        )
        category_rates = (
            stats.feature_category_action_rate,
            stats.unary_category_action_rate,
            stats.ts_unary_category_action_rate,
            stats.binary_category_action_rate,
            stats.ts_binary_category_action_rate,
            stats.cross_sectional_category_action_rate,
        )
        window_probabilities = tuple(
            getattr(stats, f"window_{window}_probability_mean")
            for window in (5, 10, 20, 40, 60)
        )
        window_rates = tuple(
            getattr(stats, f"window_{window}_action_rate")
            for window in (5, 10, 20, 40, 60)
        )
        self.assertAlmostEqual(sum(category_probabilities), 1.0, places=6)
        self.assertAlmostEqual(sum(category_rates), 1.0, places=6)
        self.assertAlmostEqual(sum(window_probabilities), 1.0, places=6)
        self.assertAlmostEqual(sum(window_rates), 1.0, places=6)
        for value in (
            stats.grammar_category_entropy_mean,
            stats.grammar_category_entropy_normalized_mean,
            stats.operator_entropy_mean,
            stats.operator_entropy_normalized_mean,
            stats.window_entropy_mean,
            stats.window_entropy_normalized_mean,
            stats.temporal_operator_action_rate,
        ):
            self.assertIsNotNone(value)
            self.assertTrue(np.isfinite(value))

    def test_hierarchical_training_records_group_and_terminal_diagnostics(self):
        trainer = GFNTrainer(
            _config(token_policy_mode="arity_hierarchical"),
            SyntheticRewardProvider(),
            device="cpu",
        )
        stats = trainer.train_step()
        for value in (
            stats.group_entropy_mean,
            stats.group_entropy_normalized_mean,
            stats.leaf_group_probability_mean,
            stats.unary_group_probability_mean,
            stats.binary_group_probability_mean,
            stats.leaf_action_rate,
            stats.unary_action_rate,
            stats.binary_action_rate,
            stats.terminal_node_count_p50,
            stats.terminal_node_count_p90,
            stats.max_node_terminal_rate,
        ):
            self.assertIsNotNone(value)
            self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(
            stats.leaf_action_rate
            + stats.unary_action_rate
            + stats.binary_action_rate,
            1.0,
        )
        self.assertAlmostEqual(
            stats.leaf_group_probability_mean
            + stats.unary_group_probability_mean
            + stats.binary_group_probability_mean,
            1.0,
        )
        self.assertGreaterEqual(stats.max_node_terminal_rate, 0.0)
        self.assertLessEqual(stats.max_node_terminal_rate, 1.0)

    def test_initial_log_z_must_be_a_finite_non_boolean_real(self):
        for invalid in (True, float("inf"), float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "initial_log_z"):
                    TrainingConfig(initial_log_z=invalid)

    def test_configured_initial_log_z_is_used_and_fingerprinted(self):
        configured = _config(initial_log_z=3.5)
        baseline = _config(initial_log_z=0.0)
        trainer = GFNTrainer(configured, SyntheticRewardProvider(), device="cpu")
        self.assertEqual(float(trainer.tb_loss.log_z.detach()), 3.5)
        self.assertNotEqual(configured.fingerprint(), baseline.fingerprint())

    def test_clip_norms_must_be_finite_positive_and_are_fingerprinted(self):
        for name in ("model_gradient_clip_norm", "log_z_gradient_clip_norm"):
            for invalid in (0.0, -1.0, float("inf"), float("nan")):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, name):
                        TrainingConfig(**{name: invalid})
        baseline = _config(
            model_gradient_clip_norm=1.0,
            log_z_gradient_clip_norm=1.0,
        )
        changed = _config(
            model_gradient_clip_norm=5.0,
            log_z_gradient_clip_norm=5.0,
        )
        self.assertNotEqual(baseline.fingerprint(), changed.fingerprint())

    def test_cuda_determinism_sets_required_cublas_workspace(self):
        with patch.dict(os.environ, {}, clear=True):
            actual = configure_cuda_determinism(
                "cuda:0", deterministic_algorithms=True
            )
            self.assertEqual(actual, CUBLAS_WORKSPACE_CONFIG)
            self.assertEqual(
                os.environ["CUBLAS_WORKSPACE_CONFIG"], CUBLAS_WORKSPACE_CONFIG
            )
            self.assertIsNone(
                configure_cuda_determinism("cpu", deterministic_algorithms=True)
            )

    def test_cuda_determinism_rejects_conflicting_workspace(self):
        with patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "重启 Python/Jupyter Kernel"):
                configure_cuda_determinism(
                    "cuda:0", deterministic_algorithms=True
                )

    def test_synthetic_training_updates_transformer_and_log_z(self):
        trainer = GFNTrainer(_config(), SyntheticRewardProvider(), device="cpu")
        before_model = {
            name: value.detach().clone() for name, value in trainer.model.state_dict().items()
        }
        before_log_z = float(trainer.tb_loss.log_z.detach())
        stats = trainer.train(3)

        self.assertEqual(len(stats), 3)
        self.assertEqual(trainer.step, 3)
        self.assertEqual(trainer.optimizer_step, 3)
        self.assertNotEqual(float(trainer.tb_loss.log_z.detach()), before_log_z)
        self.assertTrue(
            any(
                not torch.equal(before_model[name], current)
                for name, current in trainer.model.state_dict().items()
            )
        )
        for item in stats:
            self.assertFalse(item.skipped_update)
            self.assertTrue(np.isfinite(item.loss))
            self.assertTrue(np.isfinite(item.gradient_norm))
            self.assertTrue(np.isfinite(item.tb_delta_mean))
            self.assertTrue(np.isfinite(item.tb_delta_std))
            self.assertTrue(np.isfinite(item.tb_delta_rms))
            self.assertAlmostEqual(
                item.loss,
                item.tb_delta_mean**2 + item.tb_delta_std**2,
                places=9,
            )
            self.assertAlmostEqual(item.tb_delta_rms**2, item.loss, places=9)
            self.assertAlmostEqual(
                item.tb_delta_mean_square_ratio
                + item.tb_delta_std_square_ratio,
                1.0,
                places=9,
            )
            self.assertTrue(np.isfinite(item.mean_log_pf))
            self.assertTrue(np.isfinite(item.mean_log_pb))
            self.assertTrue(np.isfinite(item.model_gradient_norm_before_clip))
            self.assertTrue(np.isfinite(item.log_z_gradient_before_clip))
            self.assertGreater(item.model_gradient_norm_before_clip, 0.0)
            self.assertGreater(item.model_gradient_clip_coefficient, 0.0)
            self.assertLessEqual(item.model_gradient_clip_coefficient, 1.0)
            self.assertGreater(item.log_z_gradient_clip_coefficient, 0.0)
            self.assertLessEqual(item.log_z_gradient_clip_coefficient, 1.0)
            self.assertIsNone(item.gradient_clip_coefficient)
            self.assertGreater(item.model_parameter_update_norm, 0.0)
            self.assertGreater(item.model_relative_update_norm, 0.0)
            self.assertNotEqual(item.log_z_update, 0.0)
            self.assertTrue(np.isfinite(item.policy_entropy_mean))
            self.assertTrue(np.isfinite(item.policy_entropy_normalized_mean))
            self.assertGreaterEqual(item.policy_entropy_normalized_mean, 0.0)
            self.assertLessEqual(item.policy_entropy_normalized_mean, 1.0)
            self.assertEqual(item.effective_batch_size, 4)
            self.assertEqual(item.batch_rejection_rate, 0.0)
            self.assertEqual(item.illegal_action_rate, 0.0)
            self.assertGreater(item.expression_unique_rate, 0.0)

    def test_model_and_log_z_gradients_are_clipped_independently(self):
        trainer = GFNTrainer(
            _config(
                model_gradient_clip_norm=1e-6,
                log_z_gradient_clip_norm=1e6,
            ),
            SyntheticRewardProvider(),
            device="cpu",
        )
        stats = trainer.train_step()
        self.assertLess(stats.model_gradient_clip_coefficient, 1.0)
        self.assertEqual(stats.log_z_gradient_clip_coefficient, 1.0)
        self.assertIsNone(stats.gradient_clip_coefficient)

    def test_rejection_monitoring_and_sampling_cap_skip_update(self):
        config = _config(batch_size=2, max_sampling_multiplier=3)
        trainer = GFNTrainer(config, _RejectAllProvider(), device="cpu")
        before = {
            name: value.detach().clone() for name, value in trainer.model.state_dict().items()
        }
        stats = trainer.train_step()
        self.assertTrue(stats.skipped_update)
        self.assertEqual(stats.sampled_count, 6)
        self.assertEqual(stats.invalid_reward_count, 6)
        self.assertEqual(stats.effective_batch_size, 0)
        self.assertEqual(stats.batch_rejection_rate, 1.0)
        self.assertEqual(stats.resample_rounds, 3)
        self.assertEqual(trainer.optimizer_step, 0)
        for name, value in trainer.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))

    def test_optimizer_eps_and_reward_floor_are_separate_metadata(self):
        trainer = GFNTrainer(_config(), SyntheticRewardProvider(), device="cpu")
        metadata = trainer.run_metadata()
        semantics = metadata["parameter_semantics"]
        self.assertEqual(semantics["optimizer_eps"]["value"], 1e-8)
        self.assertEqual(semantics["reward_floor"]["value"], 1e-8)
        self.assertNotEqual(
            semantics["optimizer_eps"]["meaning"],
            semantics["reward_floor"]["meaning"],
        )


if __name__ == "__main__":
    unittest.main()
