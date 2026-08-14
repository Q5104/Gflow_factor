import math
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
    Trajectory,
    TrajectoryBalanceLoss,
    TrajectoryStep,
    sample_trajectories,
    state_hash,
    target_condition_fingerprint,
)
from factor_gfn.grammar import (
    DAGAction,
    ExactNodeGrammarState,
    SearchSpaceConfig,
)


TOY_SEARCH_SPACE = SearchSpaceConfig(max_depth=2, max_nodes=2)
TOY_MODEL_CONFIG = ModelConfig(
    d_model=8,
    num_heads=2,
    num_layers=1,
    dim_feedforward=16,
    dropout=0.0,
    token_policy_mode="grammar_hierarchical",
)


def _toy_model_and_adapter(seed: int = 517):
    torch.manual_seed(seed)
    adapter = StateAdapter(TOY_SEARCH_SPACE)
    model = ForwardPolicyNetwork(TOY_MODEL_CONFIG, TOY_SEARCH_SPACE)
    return model, adapter


def _all_n1_trajectories(
    model: ForwardPolicyNetwork,
    adapter: StateAdapter,
    rewards: tuple[float, ...],
) -> tuple[list[Trajectory], torch.Tensor]:
    source = ExactNodeGrammarState.source(
        target_node_count=1,
        search_space=TOY_SEARCH_SPACE,
    )
    batch = adapter.batch([source])
    output = model(batch)
    legal_token_ids = torch.nonzero(
        output.legal_token_mask[0, 0],
        as_tuple=False,
    ).flatten().tolist()
    if len(legal_token_ids) != 6 or len(rewards) != 6:
        raise AssertionError("N=1 toy stratum must contain exactly six terminals")
    trajectories: list[Trajectory] = []
    for token_id, reward in zip(legal_token_ids, rewards, strict=True):
        action = DAGAction((), token_id)
        child = source.step(action)
        log_p_slot = output.slot_log_probs[0, 0]
        log_p_token = output.token_log_probs[0, 0, token_id]
        trajectory = Trajectory(
            steps=[
                TrajectoryStep(
                    state_hash=state_hash(source),
                    selected_slot_index=0,
                    selected_slot_path=(),
                    selected_slot_orbit_key=source.open_slots()[0].orbit_key,
                    selected_token_id=token_id,
                    log_p_slot=log_p_slot,
                    log_p_token=log_p_token,
                    log_pf=log_p_slot + log_p_token,
                    child_state_hash=state_hash(child),
                    n_parents=child.count_parents(),
                    log_pb=child.log_backward_probability(),
                )
            ],
            terminal_state_hash=state_hash(child),
            terminal_expression=child.to_expression(),
            sampling_mode="stochastic",
            target_node_count=1,
            terminal_node_count=1,
            condition_fingerprint=target_condition_fingerprint(
                1,
                TOY_SEARCH_SPACE.fingerprint(),
            ),
        )
        trajectory.attach_reward(reward, math.log(reward))
        trajectories.append(trajectory)
    probabilities = torch.exp(
        torch.stack([trajectory.sum_log_pf for trajectory in trajectories])
    )
    return trajectories, probabilities


def _train_complete_n1_distribution(
    rewards: tuple[float, ...],
    *,
    exact_log_z: float | None = None,
    steps: int = 140,
):
    model, adapter = _toy_model_and_adapter()
    objective = TrajectoryBalanceLoss(
        initial_log_z=-1.5,
        max_nodes=TOY_SEARCH_SPACE.max_nodes,
        exact_node_counts=((1,) if exact_log_z is not None else ()),
    )
    if exact_log_z is not None:
        objective.set_exact_log_z(1, exact_log_z)
    # Break the neutral initialization so constant-Reward convergence is tested.
    with torch.no_grad():
        assert model.operator_head is not None
        model.operator_head.bias.copy_(
            torch.linspace(-1.5, 1.5, model.operator_head.bias.numel())
        )
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": 2e-2},
            {"params": objective.parameters(), "lr": 1.5e-1},
        ]
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        trajectories, _ = _all_n1_trajectories(model, adapter, rewards)
        objective(trajectories).loss.backward()
        optimizer.step()
    trajectories, probabilities = _all_n1_trajectories(model, adapter, rewards)
    output = objective(trajectories)
    return model, objective, probabilities.detach(), output


class ConditionalTBN1ConvergenceTests(unittest.TestCase):
    def test_constant_reward_converges_to_uniform_terminal_distribution(self):
        _, objective, probabilities, output = _train_complete_n1_distribution(
            (1.0,) * 6
        )
        target = torch.full((6,), 1.0 / 6.0)
        kl = torch.sum(target * torch.log(target / probabilities))
        self.assertLess(float(kl), 2e-4)
        self.assertLess(float(torch.max(torch.abs(probabilities - target))), 5e-3)
        self.assertLess(float(output.loss.detach()), 2e-4)
        self.assertAlmostEqual(
            float(objective.log_z_by_node_count[0].detach()),
            math.log(6.0),
            delta=2e-2,
        )

    def test_known_partition_learns_log_z_and_reward_proportional_policy(self):
        rewards = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        true_log_z = math.log(sum(rewards))
        _, objective, probabilities, output = _train_complete_n1_distribution(rewards)
        target = torch.tensor(rewards) / sum(rewards)
        self.assertAlmostEqual(
            float(objective.log_z_by_node_count[0].detach()),
            true_log_z,
            delta=2e-2,
        )
        self.assertLess(float(torch.max(torch.abs(probabilities - target))), 6e-3)
        self.assertLess(float(output.loss.detach()), 3e-4)

    def test_fixed_exact_z_trains_policy_without_moving_exact_or_parameter(self):
        rewards = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        true_log_z = math.log(sum(rewards))
        _, objective, probabilities, output = _train_complete_n1_distribution(
            rewards,
            exact_log_z=true_log_z,
        )
        target = torch.tensor(rewards) / sum(rewards)
        self.assertLess(float(torch.max(torch.abs(probabilities - target))), 6e-3)
        self.assertLess(float(output.loss.detach()), 3e-4)
        self.assertAlmostEqual(
            float(objective.exact_tb_log_z_by_node_count[0]),
            true_log_z,
            places=6,
        )
        self.assertEqual(float(objective.log_z_by_node_count[0].detach()), -1.5)


class ConditionalTBMixedNTests(unittest.TestCase):
    def test_mixed_n_checks_multistep_probability_sums_and_delta(self):
        model, adapter = _toy_model_and_adapter(seed=518)
        trajectories = sample_trajectories(
            model,
            adapter,
            num_trajectories=2,
            target_node_counts=[1, 2],
            sampling_config=SamplingConfig(),
        )
        rewards = (1.25, 2.5)
        for trajectory, reward in zip(trajectories, rewards, strict=True):
            trajectory.attach_reward(reward, math.log(reward))
        self.assertEqual([len(item.steps) for item in trajectories], [1, 2])

        manual_sum_pf: list[torch.Tensor] = []
        manual_sum_pb: list[float] = []
        for trajectory in trajectories:
            state = ExactNodeGrammarState.source(
                target_node_count=trajectory.target_node_count,
                search_space=TOY_SEARCH_SPACE,
            )
            step_log_pf: list[torch.Tensor] = []
            step_log_pb: list[float] = []
            for step in trajectory.steps:
                action = DAGAction(step.selected_slot_path, step.selected_token_id)
                child = state.step(action)
                expected_log_pf = step.log_p_slot + step.log_p_token
                torch.testing.assert_close(step.log_pf, expected_log_pf)
                expected_log_pb = -math.log(child.count_parents())
                self.assertAlmostEqual(step.log_pb, expected_log_pb, places=12)
                step_log_pf.append(expected_log_pf)
                step_log_pb.append(expected_log_pb)
                state = child
            summed_pf = torch.stack(step_log_pf).sum()
            summed_pb = math.fsum(step_log_pb)
            torch.testing.assert_close(trajectory.sum_log_pf, summed_pf)
            self.assertAlmostEqual(trajectory.sum_log_pb, summed_pb, places=12)
            manual_sum_pf.append(summed_pf)
            manual_sum_pb.append(summed_pb)

        objective = TrajectoryBalanceLoss(max_nodes=2)
        with torch.no_grad():
            objective.log_z_by_node_count.copy_(torch.tensor([0.35, -0.65]))
        output = objective(trajectories)
        torch.testing.assert_close(
            output.log_z,
            torch.tensor([0.35, -0.65], dtype=torch.float64),
        )
        expected_deltas = torch.stack(
            [
                objective.log_z_by_node_count[index].to(torch.float64)
                + manual_sum_pf[index].to(torch.float64)
                - math.log(rewards[index])
                - manual_sum_pb[index]
                for index in range(2)
            ]
        )
        torch.testing.assert_close(output.deltas, expected_deltas)
        output.loss.backward()
        self.assertTrue(bool((objective.log_z_by_node_count.grad != 0).all()))


if __name__ == "__main__":
    unittest.main()
