import math
import unittest

import torch

from factor_gfn.gfn import (
    ForwardPolicyNetwork,
    ModelConfig,
    SamplingConfig,
    StateAdapter,
    Trajectory,
    TrajectoryBalanceLoss,
    TrajectoryStep,
    sample_trajectories,
)
from factor_gfn.grammar import Expression, SearchSpaceConfig, get_action_id


def _sample_batch(count: int = 3, *, greedy: bool = False):
    torch.manual_seed(317)
    search_space = SearchSpaceConfig(max_depth=3, max_nodes=7)
    adapter = StateAdapter(search_space)
    model = ForwardPolicyNetwork(
        ModelConfig(
            d_model=32,
            num_heads=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        ),
        search_space,
    )
    trajectories = sample_trajectories(
        model,
        adapter,
        num_trajectories=count,
        sampling_config=SamplingConfig(temperature=1.0, greedy=greedy),
    )
    return model, trajectories


def _long_synthetic_trajectory(step_count: int = 128) -> Trajectory:
    token_id = get_action_id("close")
    steps = []
    for index in range(step_count):
        log_p_slot = torch.tensor(-0.01, dtype=torch.float32, requires_grad=True)
        log_p_token = torch.tensor(-0.02, dtype=torch.float32, requires_grad=True)
        n_parents = index % 5 + 1
        steps.append(
            TrajectoryStep(
                state_hash=f"{index:064x}",
                selected_slot_index=0,
                selected_slot_path=(),
                selected_slot_orbit_key="synthetic",
                selected_token_id=token_id,
                log_p_slot=log_p_slot,
                log_p_token=log_p_token,
                log_pf=log_p_slot + log_p_token,
                child_state_hash=f"{index + 1:064x}",
                n_parents=n_parents,
                log_pb=-math.log(n_parents),
            )
        )
    return Trajectory(
        steps=steps,
        terminal_state_hash=f"{step_count:064x}",
        terminal_expression=Expression.from_prefix([token_id]),
        sampling_mode="stochastic",
    )


class TrajectoryRewardContractTests(unittest.TestCase):
    def test_attach_reward_is_validated_and_idempotent(self):
        _, trajectories = _sample_batch(1)
        trajectory = trajectories[0]
        trajectory.attach_reward(1e-8)
        trajectory.attach_reward(1e-8, math.log(1e-8))
        self.assertAlmostEqual(trajectory.log_reward, math.log(1e-8))
        with self.assertRaisesRegex(ValueError, "一致"):
            trajectory.attach_reward(1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "覆盖"):
            trajectory.attach_reward(2e-8)

    def test_nonpositive_and_nonfinite_rewards_are_rejected(self):
        _, trajectories = _sample_batch(1)
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    trajectories[0].attach_reward(value)


class TrajectoryBalanceLossTests(unittest.TestCase):
    def test_artificially_balanced_trajectory_has_zero_loss(self):
        _, trajectories = _sample_batch(1)
        trajectory = trajectories[0]
        balanced_log_reward = float(trajectory.sum_log_pf.detach()) - trajectory.sum_log_pb
        trajectory.attach_reward(math.exp(balanced_log_reward), balanced_log_reward)
        objective = TrajectoryBalanceLoss(initial_log_z=0.0)
        output = objective(trajectories)
        self.assertLess(abs(float(output.deltas[0].detach())), 1e-7)
        self.assertLess(float(output.loss.detach()), 1e-12)

    def test_transformer_and_log_z_receive_finite_gradients(self):
        model, trajectories = _sample_batch(4)
        for trajectory in trajectories:
            trajectory.attach_reward(1e-8)
        objective = TrajectoryBalanceLoss()
        output = objective(trajectories)
        output.loss.backward()

        self.assertIsNotNone(objective.log_z.grad)
        self.assertTrue(bool(torch.isfinite(objective.log_z.grad)))
        model_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(model_gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in model_gradients))
        self.assertTrue(any(bool((gradient != 0).any()) for gradient in model_gradients))
        self.assertFalse(output.mean_log_pb.requires_grad)

    def test_tiny_reward_and_long_trajectory_remain_finite(self):
        trajectory = _long_synthetic_trajectory()
        trajectory.attach_reward(1e-8)
        objective = TrajectoryBalanceLoss()
        output = objective([trajectory])
        self.assertEqual(output.deltas.dtype, torch.float64)
        self.assertTrue(bool(torch.isfinite(output.deltas).all()))
        self.assertTrue(bool(torch.isfinite(output.loss)))
        output.loss.backward()
        self.assertTrue(bool(torch.isfinite(objective.log_z.grad)))

    def test_empty_greedy_and_missing_reward_batches_are_rejected(self):
        objective = TrajectoryBalanceLoss()
        with self.assertRaisesRegex(ValueError, "空 batch"):
            objective([])

        _, greedy = _sample_batch(1, greedy=True)
        greedy[0].attach_reward(1.0)
        with self.assertRaisesRegex(ValueError, "greedy"):
            objective(greedy)

        _, stochastic = _sample_batch(1)
        with self.assertRaisesRegex(ValueError, "尚未挂载"):
            objective(stochastic)

    def test_log_z_is_global_parameter_and_round_trips_state_dict(self):
        objective = TrajectoryBalanceLoss(initial_log_z=0.0)
        self.assertEqual(dict(objective.named_parameters()).keys(), {"log_z"})
        self.assertAlmostEqual(float(objective.estimated_z), 1.0)
        with torch.no_grad():
            objective.log_z.fill_(-2.5)
        restored = TrajectoryBalanceLoss(initial_log_z=7.0)
        restored.load_state_dict(objective.state_dict())
        self.assertAlmostEqual(float(restored.log_z.detach()), -2.5)


if __name__ == "__main__":
    unittest.main()
