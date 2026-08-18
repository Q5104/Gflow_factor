from __future__ import annotations

from dataclasses import replace
import math
import unittest

import torch

from factor_gfn.gfn import (
    LogPartitionVarianceLoss,
    direct_log_partition_variance,
)
from factor_gfn.gfn.trajectory import (
    Trajectory,
    TrajectoryStep,
    target_condition_fingerprint,
)
from factor_gfn.grammar import Expression, SearchSpaceConfig, get_action_id


def _tensor_objective(zeta_values: list[float]):
    desired_zeta = torch.tensor(zeta_values, dtype=torch.float64)
    sum_log_pf = (-desired_zeta).clone().requires_grad_(True)
    sum_log_pb = torch.zeros_like(desired_zeta, requires_grad=True)
    log_reward = torch.zeros_like(desired_zeta, requires_grad=True)
    output = direct_log_partition_variance(
        sum_log_pf=sum_log_pf,
        sum_log_pb=sum_log_pb,
        log_reward=log_reward,
    )
    return output, sum_log_pf, sum_log_pb, log_reward


def _conditioned_trajectory(
    *,
    index: int,
    target_node_count: int = 3,
    log_pf_value: float = -1.0,
    stochastic: bool = True,
) -> tuple[Trajectory, torch.Tensor]:
    if target_node_count == 2:
        prefix = [get_action_id("neg"), get_action_id("close")]
    elif target_node_count == 3:
        prefix = [get_action_id("add"), get_action_id("close"), get_action_id("open")]
    elif target_node_count == 4:
        prefix = [
            get_action_id("neg"),
            get_action_id("add"),
            get_action_id("close"),
            get_action_id("open"),
        ]
    else:
        raise ValueError("test helper supports target N=2, N=3, or N=4")
    expression = Expression.from_prefix(prefix)
    log_p_slot = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    log_p_token = torch.tensor(
        log_pf_value,
        dtype=torch.float64,
        requires_grad=True,
    )
    step = TrajectoryStep(
        state_hash=f"{2 * index:064x}",
        selected_slot_index=0,
        selected_slot_path=(),
        selected_slot_orbit_key="lpv-test",
        selected_token_id=get_action_id("close"),
        log_p_slot=log_p_slot,
        log_p_token=log_p_token,
        log_pf=log_p_slot + log_p_token,
        child_state_hash=f"{2 * index + 1:064x}",
        n_parents=1,
        log_pb=0.0,
    )
    search_space = SearchSpaceConfig(max_depth=5, max_nodes=15)
    trajectory = Trajectory(
        steps=[step],
        terminal_state_hash=step.child_state_hash,
        terminal_expression=expression,
        sampling_mode="stochastic" if stochastic else "greedy",
        target_node_count=target_node_count,
        terminal_node_count=target_node_count,
        condition_fingerprint=target_condition_fingerprint(
            target_node_count,
            search_space.fingerprint(),
        ),
    )
    trajectory.attach_reward(1.0)
    return trajectory, log_p_token


class DirectLogPartitionVarianceTests(unittest.TestCase):
    def test_equal_zeta_has_zero_loss_and_zero_gradient(self):
        output, sum_log_pf, _, _ = _tensor_objective([5.0, 5.0, 5.0, 5.0])

        self.assertEqual(float(output.loss), 0.0)
        self.assertEqual(float(output.zeta_variance), 0.0)
        self.assertEqual(float(output.centered_zeta_rms), 0.0)
        output.loss.backward()
        torch.testing.assert_close(sum_log_pf.grad, torch.zeros(4, dtype=torch.float64))

    def test_direct_empirical_variance_value_and_analytic_gradient(self):
        zeta_values = torch.tensor([1.0, 3.0, 8.0, 10.0], dtype=torch.float64)
        output, sum_log_pf, _, _ = _tensor_objective(zeta_values.tolist())
        centered = zeta_values - zeta_values.mean()
        expected_loss = centered.square().mean()

        torch.testing.assert_close(output.zeta, zeta_values)
        torch.testing.assert_close(output.loss, expected_loss)
        torch.testing.assert_close(output.zeta_variance, expected_loss)
        torch.testing.assert_close(output.zeta_std, torch.sqrt(expected_loss))
        self.assertTrue(output.zeta_mean.requires_grad)

        output.loss.backward()
        expected_pf_gradient = -2.0 * centered / len(zeta_values)
        torch.testing.assert_close(sum_log_pf.grad, expected_pf_gradient)

    def test_reward_and_pb_are_detached_while_pf_keeps_gradient(self):
        output, sum_log_pf, sum_log_pb, log_reward = _tensor_objective(
            [1.0, 2.0, 4.0, 8.0]
        )

        output.loss.backward()
        self.assertIsNotNone(sum_log_pf.grad)
        self.assertIsNone(sum_log_pb.grad)
        self.assertIsNone(log_reward.grad)

    def test_requires_vector_inputs_with_k_at_least_two(self):
        with self.assertRaisesRegex(ValueError, "K >= 2"):
            direct_log_partition_variance(
                sum_log_pf=torch.tensor([-1.0]),
                sum_log_pb=torch.tensor([0.0]),
                log_reward=torch.tensor([0.0]),
            )
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            direct_log_partition_variance(
                sum_log_pf=torch.tensor([[-1.0, -2.0]]),
                sum_log_pb=torch.tensor([[0.0, 0.0]]),
                log_reward=torch.tensor([[0.0, 0.0]]),
            )
        with self.assertRaisesRegex(ValueError, "share one shape"):
            direct_log_partition_variance(
                sum_log_pf=torch.tensor([-1.0, -2.0]),
                sum_log_pb=torch.tensor([0.0, 0.0, 0.0]),
                log_reward=torch.tensor([0.0, 0.0]),
            )

    def test_trajectory_objective_preserves_pf_gradient_and_has_no_state(self):
        trajectories: list[Trajectory] = []
        log_p_tokens: list[torch.Tensor] = []
        for index, value in enumerate((-1.0, -2.0, -4.0, -8.0), start=1):
            trajectory, log_p_token = _conditioned_trajectory(
                index=index,
                log_pf_value=value,
            )
            trajectories.append(trajectory)
            log_p_tokens.append(log_p_token)

        objective = LogPartitionVarianceLoss()
        self.assertEqual(list(objective.parameters()), [])
        self.assertEqual(objective.state_dict(), {})
        output = objective(trajectories)
        self.assertEqual(output.zeta.dtype, torch.float64)
        output.loss.backward()
        self.assertTrue(all(value.grad is not None for value in log_p_tokens))

    def test_trajectory_objective_requires_one_lpv_condition(self):
        first, _ = _conditioned_trajectory(index=1, target_node_count=3)
        second, _ = _conditioned_trajectory(index=2, target_node_count=4)
        objective = LogPartitionVarianceLoss()

        with self.assertRaisesRegex(ValueError, "one fixed condition"):
            objective([first, second])

        exact_like, _ = _conditioned_trajectory(index=3, target_node_count=2)
        with self.assertRaisesRegex(ValueError, "N must be in 3..15"):
            objective([exact_like, exact_like])

    def test_greedy_and_missing_reward_trajectories_are_rejected(self):
        stochastic, _ = _conditioned_trajectory(index=1)
        greedy, _ = _conditioned_trajectory(index=2, stochastic=False)
        with self.assertRaisesRegex(ValueError, "禁止"):
            LogPartitionVarianceLoss()([stochastic, greedy])

        missing_reward = replace(stochastic, reward=None, log_reward=None)
        with self.assertRaisesRegex(ValueError, "Reward"):
            LogPartitionVarianceLoss()([missing_reward, missing_reward])

    def test_output_uses_population_variance_not_vargrad_normalization(self):
        output, _, _, _ = _tensor_objective([1.0, 3.0, 8.0, 10.0])
        zeta = output.zeta.detach()

        torch.testing.assert_close(output.loss.detach(), zeta.var(correction=0))
        self.assertFalse(
            math.isclose(
                float(output.loss.detach()),
                float(0.5 * zeta.var(correction=1)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
