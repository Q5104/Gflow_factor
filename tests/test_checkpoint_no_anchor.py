import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from factor_gfn.gfn import (
    GFNConfig,
    GFNTrainer,
    ModelConfig,
    SyntheticRewardProvider,
    SearchSpaceConfig,
    TrainingConfig,
)


def _legacy_config() -> GFNConfig:
    return GFNConfig(
        search_space=SearchSpaceConfig(max_depth=1, max_nodes=2),
        model=ModelConfig(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            token_policy_mode="flat",
        ),
        training=TrainingConfig(
            batch_size=2,
            max_steps=3,
            learning_rate=1e-3,
            log_z_learning_rate=1e-2,
            seed=91,
        ),
    )


def _write_legacy_stage4_fixture(path: Path, trainer: GFNTrainer) -> None:
    """Test-only historical fixture writer; production has no legacy writer."""

    payload = {
        "schema": "factor_gfn.checkpoint.v4",
        "config_fingerprint": trainer.config.fingerprint(),
        "reward_provider_fingerprint": trainer.reward_provider.fingerprint(),
        "device_type": trainer.device.type,
        "model_state": trainer.model.state_dict(),
        "tb_loss_state": trainer.tb_loss.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "step": trainer.step,
        "optimizer_step": trainer.optimizer_step,
        "history": [asdict(item) for item in trainer.history],
        "run_id": trainer.run_id,
        "created_at_utc": trainer.created_at_utc,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,
        },
        "run_metadata": trainer.run_metadata(),
    }
    torch.save(payload, path)


class LegacyStage4ReadOnlyCheckpointTests(unittest.TestCase):
    def test_legacy_scalar_fixture_loads_but_legacy_save_is_forbidden(self):
        provider = SyntheticRewardProvider()
        source = GFNTrainer(_legacy_config(), provider, device="cpu")
        source.train_step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-stage4.pt"
            _write_legacy_stage4_fixture(path, source)
            resumed = GFNTrainer(_legacy_config(), provider, device="cpu")
            resumed.load_checkpoint(path)
            self.assertEqual(resumed.step, source.step)
            self.assertEqual(resumed.optimizer_step, source.optimizer_step)
            self.assertEqual(resumed.history, source.history)
            forbidden = Path(directory) / "new-legacy-write.pt"
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                resumed.save_checkpoint(forbidden)
            self.assertFalse(forbidden.exists())

    def test_authentic_pre_condition_projection_state_loads_read_only_and_runs(self):
        provider = SyntheticRewardProvider()
        source = GFNTrainer(_legacy_config(), provider, device="cpu")
        source.train_step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-before-condition-projection.pt"
            _write_legacy_stage4_fixture(path, source)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["schema"] = "factor_gfn.checkpoint.v1"
            payload["model_state"] = dict(payload["model_state"])
            payload["model_state"].pop("condition_projection.weight")
            optimizer_state = payload["optimizer_state"]
            names = [name for name, _ in source.model.named_parameters()]
            condition_index = names.index("condition_projection.weight")
            condition_id = optimizer_state["param_groups"][0]["params"].pop(
                condition_index
            )
            optimizer_state["state"].pop(condition_id, None)
            torch.save(payload, path)
            resumed = GFNTrainer(_legacy_config(), provider, device="cpu")
            resumed.load_checkpoint(path)
            self.assertEqual(resumed.step, source.step)
            self.assertTrue(
                torch.equal(
                    resumed.model.condition_projection.weight,
                    torch.zeros_like(resumed.model.condition_projection.weight),
                )
            )
            next_stats = resumed.train_step()
            self.assertFalse(next_stats.skipped_update)
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                resumed.save_checkpoint(Path(directory) / "forbidden.pt")

    def test_legacy_conditioned_checkpoint_is_not_in_read_only_scope(self):
        provider = SyntheticRewardProvider()
        trainer = GFNTrainer(_legacy_config(), provider, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-anchor-v5.pt"
            _write_legacy_stage4_fixture(path, trainer)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["schema"] = "factor_gfn.checkpoint.v5"
            payload["anchor_state"] = {"configured": True}
            payload["anchor_optimizer_step"] = 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "anchor checkpoint is rejected"):
                trainer.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
