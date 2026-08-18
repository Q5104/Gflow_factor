import tempfile
import unittest
from pathlib import Path

import torch

from factor_gfn.gfn import (
    ComplexitySchedulerConfig,
    ForwardPolicyNetwork,
    GFNConfig,
    GFNTrainer,
    ModelConfig,
    SamplingConfig,
    StateAdapter,
    SyntheticRewardProvider,
    TrainingConfig,
    TrajectoryBalanceLoss,
    sample_trajectories,
)
from factor_gfn.grammar import SearchSpaceConfig


def _conditioned_batch(targets: list[int]):
    torch.manual_seed(812)
    search_space = SearchSpaceConfig(max_depth=2, max_nodes=5)
    adapter = StateAdapter(search_space)
    model = ForwardPolicyNetwork(
        ModelConfig(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            token_policy_mode="grammar_hierarchical",
        ),
        search_space,
    )
    trajectories = sample_trajectories(
        model,
        adapter,
        num_trajectories=len(targets),
        target_node_counts=targets,
        sampling_config=SamplingConfig(),
    )
    for trajectory in trajectories:
        trajectory.attach_reward(1.0, 0.0)
    return model, trajectories


def _conditioned_config(*, batch_size: int = 1) -> GFNConfig:
    return GFNConfig(
        search_space=SearchSpaceConfig(max_depth=2, max_nodes=5),
        model=ModelConfig(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            token_policy_mode="grammar_hierarchical",
        ),
        complexity_scheduler=ComplexitySchedulerConfig(
            enabled=True,
            exhaustive_node_counts=(1,),
        ),
        training=TrainingConfig(
            batch_size=batch_size,
            learning_rate=1e-3,
            log_z_learning_rate=1e-2,
            max_steps=8,
            seed=812,
            weight_decay=0.25,
        ),
    )


class ConditionalNormalizerMathTests(unittest.TestCase):
    def test_vector_and_buffers_follow_dynamic_max_nodes(self):
        for max_nodes in (3, 9):
            objective = TrajectoryBalanceLoss(
                initial_log_z=2.5,
                max_nodes=max_nodes,
                exact_node_counts=(1,),
            )
            self.assertEqual(objective.log_z_by_node_count.shape, (max_nodes,))
            self.assertEqual(objective.exact_tb_log_z_by_node_count.shape, (max_nodes,))
            self.assertEqual(objective.exact_log_z_mask.shape, (max_nodes,))
            self.assertFalse(bool(objective.exact_log_z_mask.any()))
            self.assertTrue(
                torch.equal(
                    objective.log_z_by_node_count.detach(),
                    torch.full((max_nodes,), 2.5),
                )
            )

    def test_mixed_n_selects_exact_buffer_and_n_minus_one_parameter(self):
        _, trajectories = _conditioned_batch([1, 2, 5])
        objective = TrajectoryBalanceLoss(
            max_nodes=5,
            exact_node_counts=(1,),
        )
        with torch.no_grad():
            objective.log_z_by_node_count.copy_(
                torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
            )
        with self.assertRaisesRegex(RuntimeError, "N=1"):
            objective(trajectories)
        objective.set_exact_log_z(1, -3.0)
        objective.set_exact_log_z(1, -3.0)
        with self.assertRaisesRegex(ValueError, "already registered"):
            objective.set_exact_log_z(1, -2.0)
        output = objective(trajectories)
        torch.testing.assert_close(
            output.log_z,
            torch.tensor([-3.0, 20.0, 50.0], dtype=torch.float64),
        )
        with self.assertRaises(ValueError):
            objective.set_exact_log_z(2, 7.0)
        with self.assertRaises(ValueError):
            objective.set_exact_log_z(0, 7.0)

    def test_exact_batch_trains_policy_without_parameter_gradient(self):
        model, trajectories = _conditioned_batch([1])
        objective = TrajectoryBalanceLoss(max_nodes=5, exact_node_counts=(1,))
        objective.set_exact_log_z(1, 0.75)
        exact_before = objective.exact_tb_log_z_by_node_count.detach().clone()
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters(), "weight_decay": 0.2},
                {"params": objective.parameters(), "weight_decay": 0.2},
            ],
            lr=1e-2,
        )
        optimizer.zero_grad(set_to_none=True)
        objective(trajectories).loss.backward()
        self.assertIsNone(objective.log_z_by_node_count.grad)
        self.assertTrue(any(
            parameter.grad is not None
            and bool((parameter.grad != 0).any())
            for parameter in model.parameters()
        ))
        optimizer.step()
        self.assertTrue(torch.equal(
            objective.exact_tb_log_z_by_node_count,
            exact_before,
        ))

    def test_non_exact_batch_only_has_gradient_at_selected_index(self):
        model, trajectories = _conditioned_batch([2])
        objective = TrajectoryBalanceLoss(max_nodes=5, exact_node_counts=(1,))
        output = objective(trajectories)
        output.loss.backward()
        gradient = objective.log_z_by_node_count.grad
        self.assertIsNotNone(gradient)
        self.assertNotEqual(float(gradient[1]), 0.0)
        self.assertTrue(torch.equal(
            gradient[[0, 2, 3, 4]],
            torch.zeros(4),
        ))
        self.assertTrue(any(
            parameter.grad is not None and bool((parameter.grad != 0).any())
            for parameter in model.parameters()
        ))


class ConditionalNormalizerTrainerTests(unittest.TestCase):
    def test_plain_sgd_updates_only_active_logz_and_records_per_n_update(self):
        trainer = GFNTrainer(
            _conditioned_config(batch_size=1),
            SyntheticRewardProvider(),
            device="cpu",
            normalizer_optimizer="sgd",
        )
        self.assertEqual(
            [group["name"] for group in trainer.optimizer.param_groups],
            ["policy", "normalizer"],
        )
        self.assertEqual(trainer.optimizer.param_groups[1]["momentum"], 0.0)
        self.assertEqual(trainer.optimizer.param_groups[1]["weight_decay"], 0.0)
        before = trainer.tb_loss.log_z_by_node_count.detach().clone()
        stats = trainer.train_step()
        after = trainer.tb_loss.log_z_by_node_count.detach().clone()
        changed = torch.nonzero(after != before, as_tuple=False).flatten().tolist()
        self.assertEqual(len(changed), 1)
        node_count = changed[0] + 1
        self.assertEqual(set(stats.log_z_update_by_N or {}), {node_count})
        self.assertAlmostEqual(
            stats.log_z_update_by_N[node_count],
            float(after[changed[0]] - before[changed[0]]),
        )
        self.assertEqual(trainer.optimizer_contract()["normalizer_optimizer"], "sgd")

    def test_optimizer_groups_are_separate_and_absent_scalars_do_not_move(self):
        trainer = GFNTrainer(
            _conditioned_config(batch_size=1),
            SyntheticRewardProvider(),
            device="cpu",
        )
        self.assertEqual(
            [group["name"] for group in trainer.optimizer.param_groups],
            ["policy", "normalizer"],
        )
        self.assertEqual(trainer.optimizer.param_groups[1]["weight_decay"], 0.0)
        before = trainer.tb_loss.log_z_by_node_count.detach().clone()
        previous_requested = dict(trainer.requested_count_by_N)
        trainer.train_step()
        assigned = next(
            node_count
            for node_count, count in trainer.requested_count_by_N.items()
            if count != previous_requested[node_count]
        )
        after_first = trainer.tb_loss.log_z_by_node_count.detach().clone()
        changed = torch.nonzero(after_first != before, as_tuple=False).flatten().tolist()
        self.assertEqual(changed, [assigned - 1])

        previous_requested = dict(trainer.requested_count_by_N)
        trainer.train_step()
        assigned_second = next(
            node_count
            for node_count, count in trainer.requested_count_by_N.items()
            if count != previous_requested[node_count]
        )
        after_second = trainer.tb_loss.log_z_by_node_count.detach().clone()
        self.assertEqual(
            float(after_second[assigned - 1]),
            float(after_first[assigned - 1]),
        )
        self.assertNotEqual(
            float(after_second[assigned_second - 1]),
            float(after_first[assigned_second - 1]),
        )


if __name__ == "__main__":
    unittest.main()
