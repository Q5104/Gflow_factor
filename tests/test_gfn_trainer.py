import json
import os
import random
import tempfile
import unittest
from pathlib import Path
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
    SyntheticRewardConfig,
    SyntheticRewardProvider,
    TrainingConfig,
    configure_cuda_determinism,
    write_run_metadata,
)
from factor_gfn.gfn.checkpoint import (
    _cpu_byte_rng_state,
    _legacy_v3_config_fingerprint,
    _legacy_v4_config_fingerprint,
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


class CheckpointTests(unittest.TestCase):
    def test_retired_arity_checkpoint_keeps_direct_v4_read_compatibility(self):
        config = _config(token_policy_mode="arity_hierarchical")
        source = GFNTrainer(config, SyntheticRewardProvider(), device="cpu")
        source.train(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_v4_arity.pt"
            source.save_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["config_fingerprint"] = _legacy_v4_config_fingerprint(source)
            torch.save(payload, path)
            resumed = GFNTrainer(config, SyntheticRewardProvider(), device="cpu")
            resumed.load_checkpoint(path)
            grammar = GFNTrainer(
                _config(token_policy_mode="grammar_hierarchical"),
                SyntheticRewardProvider(),
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "GFNConfig"):
                grammar.load_checkpoint(path)
        self.assertEqual(resumed.step, 1)

    def test_flat_checkpoint_accepts_pre_hierarchical_v3_fingerprint(self):
        config = _config(token_policy_mode="flat")
        source = GFNTrainer(config, SyntheticRewardProvider(), device="cpu")
        source.train(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_v3_flat.pt"
            source.save_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["config_fingerprint"] = _legacy_v3_config_fingerprint(source)
            torch.save(payload, path)
            resumed = GFNTrainer(config, SyntheticRewardProvider(), device="cpu")
            resumed.load_checkpoint(path)
            hierarchical = GFNTrainer(
                _config(token_policy_mode="arity_hierarchical"),
                SyntheticRewardProvider(),
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "GFNConfig"):
                hierarchical.load_checkpoint(path)
        self.assertEqual(resumed.step, 1)
        self.assertIsNone(resumed.model.group_head)

    def test_hierarchical_checkpoint_restores_exact_continuation(self):
        config = _config(token_policy_mode="arity_hierarchical")
        provider = SyntheticRewardProvider()
        continuous = GFNTrainer(config, provider, device="cpu")
        continuous.train(3)
        split = GFNTrainer(config, provider, device="cpu")
        split.train(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hierarchical.pt"
            split.save_checkpoint(path)
            resumed = GFNTrainer(config, provider, device="cpu")
            resumed.load_checkpoint(path)
            resumed.train(2)
        self.assertEqual(continuous.history, resumed.history)
        for name, value in continuous.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), name)
        _assert_nested_equal(
            self,
            continuous.optimizer.state_dict(),
            resumed.optimizer.state_dict(),
        )

    def test_grammar_checkpoint_restores_exactly_and_rejects_legacy_policy(self):
        config = _config(token_policy_mode="grammar_hierarchical")
        provider = SyntheticRewardProvider()
        continuous = GFNTrainer(config, provider, device="cpu")
        continuous.train(3)
        split = GFNTrainer(config, provider, device="cpu")
        split.train(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grammar_hierarchical.pt"
            split.save_checkpoint(path)
            resumed = GFNTrainer(config, provider, device="cpu")
            resumed.load_checkpoint(path)
            resumed.train(2)
            legacy = GFNTrainer(
                _config(token_policy_mode="arity_hierarchical"),
                provider,
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "GFNConfig"):
                legacy.load_checkpoint(path)
        self.assertEqual(continuous.history, resumed.history)
        for name, value in continuous.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), name)
        _assert_nested_equal(
            self,
            continuous.optimizer.state_dict(),
            resumed.optimizer.state_dict(),
        )

    def test_rng_state_is_normalized_to_cpu_byte_tensor(self):
        state = torch.arange(8, dtype=torch.uint8)
        normalized = _cpu_byte_rng_state(state, label="test")
        self.assertEqual(normalized.dtype, torch.uint8)
        self.assertEqual(normalized.device.type, "cpu")
        self.assertTrue(normalized.is_contiguous())
        with self.assertRaisesRegex(TypeError, "ByteTensor"):
            _cpu_byte_rng_state(state.to(dtype=torch.int64), label="test")
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            _cpu_byte_rng_state([1, 2, 3], label="test")

    def test_atomic_checkpoint_restores_deterministic_continuation(self):
        config = _config()
        provider = SyntheticRewardProvider()

        continuous = GFNTrainer(config, provider, device="cpu")
        continuous.train(4)
        expected_python_rng = random.getstate()
        expected_numpy_rng = np.random.get_state()
        expected_torch_rng = torch.get_rng_state().clone()

        split = GFNTrainer(config, provider, device="cpu")
        split.train(2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "trainer.pt"
            metadata_path = Path(directory) / "run_metadata.json"
            split.save_checkpoint(checkpoint_path)
            self.assertTrue(checkpoint_path.exists())
            self.assertFalse(Path(str(checkpoint_path) + ".tmp").exists())
            write_run_metadata(metadata_path, split)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["config_fingerprint"], config.fingerprint())

            resumed = GFNTrainer(config, provider, device="cpu")
            resumed.load_checkpoint(checkpoint_path)
            resumed.train(2)

        self.assertEqual(continuous.step, resumed.step)
        self.assertEqual(continuous.optimizer_step, resumed.optimizer_step)
        self.assertEqual(continuous.history, resumed.history)
        for name, value in continuous.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), name)
        self.assertTrue(
            torch.equal(continuous.tb_loss.log_z.detach(), resumed.tb_loss.log_z.detach())
        )
        _assert_nested_equal(
            self,
            continuous.optimizer.state_dict(),
            resumed.optimizer.state_dict(),
        )
        self.assertEqual(random.getstate(), expected_python_rng)
        current_numpy_rng = np.random.get_state()
        self.assertEqual(current_numpy_rng[0], expected_numpy_rng[0])
        np.testing.assert_array_equal(current_numpy_rng[1], expected_numpy_rng[1])
        self.assertEqual(current_numpy_rng[2:], expected_numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_rng))

    def test_checkpoint_history_without_diagnostics_remains_loadable(self):
        trainer = GFNTrainer(_config(), SyntheticRewardProvider(), device="cpu")
        trainer.train(1)
        diagnostic_fields = {
            "tb_delta_mean",
            "tb_delta_std",
            "tb_delta_rms",
            "tb_delta_mean_square_ratio",
            "tb_delta_std_square_ratio",
            "mean_log_pf",
            "mean_log_pb",
            "model_gradient_norm_before_clip",
            "log_z_gradient_before_clip",
            "model_gradient_clip_coefficient",
            "log_z_gradient_clip_coefficient",
            "gradient_clip_coefficient",
            "model_parameter_update_norm",
            "model_relative_update_norm",
            "log_z_update",
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy_history.pt"
            trainer.save_checkpoint(checkpoint_path)
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            for item in payload["history"]:
                for field in diagnostic_fields:
                    item.pop(field, None)
            torch.save(payload, checkpoint_path)

            resumed = GFNTrainer(_config(), SyntheticRewardProvider(), device="cpu")
            resumed.load_checkpoint(checkpoint_path)

        self.assertEqual(resumed.step, 1)
        self.assertEqual(resumed.optimizer_step, 1)
        for field in diagnostic_fields:
            self.assertIsNone(getattr(resumed.history[0], field))

    def test_checkpoint_rejects_reward_provider_mismatch(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer.pt"
            source = GFNTrainer(config, SyntheticRewardProvider(), device="cpu")
            source.save_checkpoint(path)
            changed = GFNTrainer(
                config,
                SyntheticRewardProvider(SyntheticRewardConfig(close_bonus=0.5)),
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "Reward Provider"):
                changed.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
