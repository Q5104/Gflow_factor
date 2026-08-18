import copy
import unittest

import torch

from factor_gfn.gfn import (
    ComplexitySchedulerConfig,
    GFNConfig,
    GFNTrainer,
    ModelConfig,
    RewardAssignment,
    SamplingConfig,
    SearchSpaceConfig,
    SyntheticRewardProvider,
    TrainingConfig,
)


def _conditioned_config(*, retry_budget: int) -> GFNConfig:
    return GFNConfig(
        search_space=SearchSpaceConfig(max_depth=5, max_nodes=15),
        model=ModelConfig(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            token_policy_mode="grammar_hierarchical",
        ),
        sampling=SamplingConfig(temperature=1.0, greedy=False),
        complexity_scheduler=ComplexitySchedulerConfig(
            enabled=True,
            exhaustive_node_counts=(1, 2),
            exact_node_retry_budget=retry_budget,
        ),
        training=TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            log_z_learning_rate=1e-2,
            max_steps=2,
            seed=20260816,
        ),
    )


class _RejectFirstRoundProvider(SyntheticRewardProvider):
    def __init__(self, first_round_size: int) -> None:
        super().__init__()
        self.first_round_size = first_round_size
        self.calls = 0

    def evaluate(self, expression):
        self.calls += 1
        if self.calls <= self.first_round_size:
            return RewardAssignment(valid=False, rejection_reason="first round")
        return super().evaluate(expression)


class _RejectAllProvider(SyntheticRewardProvider):
    def evaluate(self, expression):
        return RewardAssignment(valid=False, rejection_reason="forced rejection")


class HybridSingleConditionBatchTests(unittest.TestCase):
    def test_fixed_N_and_configurable_K_ignore_legacy_batch_size(self):
        trainer = GFNTrainer(
            _conditioned_config(retry_budget=0),
            SyntheticRewardProvider(),
            device="cpu",
        )

        for K in (3, 5):
            with self.subTest(K=K):
                batch = trainer.collect_single_condition_batch(
                    condition_N=7,
                    trajectories_per_batch=K,
                )
                self.assertTrue(batch.complete)
                self.assertEqual(batch.requested_count, K)
                self.assertEqual(batch.accepted_count, K)
                self.assertEqual(batch.target_node_counts, (7,) * K)
                self.assertEqual(
                    {trajectory.target_node_count for trajectory in batch.trajectories},
                    {7},
                )
                self.assertEqual(
                    {trajectory.terminal_node_count for trajectory in batch.trajectories},
                    {7},
                )
                self.assertEqual(batch.sampled_count, K)
                self.assertEqual(batch.retry_count, 0)
                self.assertEqual(batch.retry_exhausted_count, 0)

    def test_rejected_slots_retry_same_condition_until_complete(self):
        K = 4
        provider = _RejectFirstRoundProvider(K)
        trainer = GFNTrainer(
            _conditioned_config(retry_budget=1),
            provider,
            device="cpu",
        )

        batch = trainer.collect_single_condition_batch(
            condition_N=7,
            trajectories_per_batch=K,
        )

        self.assertTrue(batch.complete)
        self.assertEqual(batch.accepted_count, K)
        self.assertEqual(batch.sampled_count, 2 * K)
        self.assertEqual(batch.invalid_count, K)
        self.assertEqual(batch.retry_count, K)
        self.assertEqual(batch.retry_exhausted_count, 0)
        self.assertEqual(batch.sampling_rounds, 2)
        self.assertTrue(
            all(
                trajectory.target_node_count == 7
                for trajectory in batch.trajectories
            )
        )

    def test_incomplete_batch_does_not_advance_scheduler_or_optimizer(self):
        K = 4
        trainer = GFNTrainer(
            _conditioned_config(retry_budget=1),
            _RejectAllProvider(),
            device="cpu",
        )
        scheduler_before = copy.deepcopy(trainer.complexity_scheduler.state_dict())
        optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
        model_before = {
            name: value.detach().clone()
            for name, value in trainer.model.state_dict().items()
        }

        batch = trainer.collect_single_condition_batch(
            condition_N=7,
            trajectories_per_batch=K,
        )

        self.assertFalse(batch.complete)
        self.assertEqual(batch.accepted_count, 0)
        self.assertEqual(batch.sampled_count, 2 * K)
        self.assertEqual(batch.invalid_count, 2 * K)
        self.assertEqual(batch.retry_count, K)
        self.assertEqual(batch.retry_exhausted_count, K)
        self.assertEqual(trainer.complexity_scheduler.state_dict(), scheduler_before)
        self.assertEqual(trainer.optimizer.state_dict(), optimizer_before)
        self.assertEqual(trainer.step, 0)
        self.assertEqual(trainer.optimizer_step, 0)
        for name, value in trainer.model.state_dict().items():
            self.assertTrue(torch.equal(value, model_before[name]), name)

    def test_invalid_condition_and_K_are_rejected_before_sampling(self):
        trainer = GFNTrainer(
            _conditioned_config(retry_budget=0),
            SyntheticRewardProvider(),
            device="cpu",
        )
        for invalid_N in (True, 0, 16):
            with self.subTest(condition_N=invalid_N):
                with self.assertRaisesRegex(ValueError, "condition_N"):
                    trainer.collect_single_condition_batch(
                        condition_N=invalid_N,
                        trajectories_per_batch=3,
                    )
        for invalid_K in (True, 0, 1):
            with self.subTest(K=invalid_K):
                with self.assertRaisesRegex(ValueError, "trajectories_per_batch"):
                    trainer.collect_single_condition_batch(
                        condition_N=7,
                        trajectories_per_batch=invalid_K,
                    )


if __name__ == "__main__":
    unittest.main()
