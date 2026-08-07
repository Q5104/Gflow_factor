import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from factor_gfn.gfn import (
    GFNConfig,
    GFNTrainer,
    ModelConfig,
    RewardAssignment,
    SamplingConfig,
    SearchSpaceConfig,
    SyntheticRewardConfig,
    SyntheticRewardProvider,
    TrainingConfig,
    write_run_metadata,
)


def _config(**training_overrides) -> GFNConfig:
    training = {
        "batch_size": 4,
        "learning_rate": 1e-3,
        "log_z_learning_rate": 1e-2,
        "max_steps": 20,
        "gradient_clip_norm": 1.0,
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
            self.assertTrue(np.isfinite(item.policy_entropy_mean))
            self.assertTrue(np.isfinite(item.policy_entropy_normalized_mean))
            self.assertGreaterEqual(item.policy_entropy_normalized_mean, 0.0)
            self.assertLessEqual(item.policy_entropy_normalized_mean, 1.0)
            self.assertEqual(item.effective_batch_size, 4)
            self.assertEqual(item.batch_rejection_rate, 0.0)
            self.assertEqual(item.illegal_action_rate, 0.0)
            self.assertGreater(item.expression_unique_rate, 0.0)

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
