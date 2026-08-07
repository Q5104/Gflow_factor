import math
import unittest

import torch
from torch import nn

from factor_gfn.gfn import (
    ForwardPolicyNetwork,
    ModelConfig,
    PolicyOutput,
    SamplingConfig,
    SearchSpaceConfig,
    StateAdapter,
    sample_trajectories,
    sample_trajectory,
)
from factor_gfn.grammar import DAGAction, GrammarState, TOTAL_ACTIONS, get_action_id


class _CloseOnlyPolicy(nn.Module):
    """测试用确定性策略：始终在第一个可用槽位填 close。"""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch, *, temperature: float = 1.0) -> PolicyOutput:
        batch_size, slot_count = batch.slot_mask.shape
        device = self.anchor.device
        slot_log_probs = torch.full((batch_size, slot_count), -torch.inf, device=device)
        token_log_probs = torch.full(
            (batch_size, slot_count, TOTAL_ACTIONS), -torch.inf, device=device
        )
        close_id = get_action_id("close")
        for row in range(batch_size):
            slot = int(torch.nonzero(batch.slot_mask[row], as_tuple=False)[0, 0])
            if not bool(batch.legal_token_mask[row, slot, close_id]):
                raise RuntimeError("close 在测试状态中应为合法动作")
            slot_log_probs[row, slot] = self.anchor * 0.0
            token_log_probs[row, slot, close_id] = self.anchor * 0.0
        return PolicyOutput(
            slot_logits=slot_log_probs,
            token_logits=token_log_probs,
            slot_log_probs=slot_log_probs,
            token_log_probs=token_log_probs,
            slot_mask=batch.slot_mask,
            legal_token_mask=batch.legal_token_mask,
        )


class PolicySamplerTests(unittest.TestCase):
    def small_components(self, *, dropout: float = 0.0):
        search_space = SearchSpaceConfig(max_depth=3, max_nodes=5)
        adapter = StateAdapter(search_space)
        model = ForwardPolicyNetwork(
            ModelConfig(
                d_model=32,
                num_heads=4,
                num_layers=1,
                dim_feedforward=64,
                dropout=dropout,
            ),
            search_space,
        )
        return search_space, adapter, model

    def test_sampler_records_real_multi_parent_backward_probability(self):
        search_space = SearchSpaceConfig(max_depth=2, max_nodes=3)
        adapter = StateAdapter(search_space)
        state = GrammarState(search_space=search_space)
        state = state.step(DAGAction((), get_action_id("add")))
        state = state.step(DAGAction(state.open_slots()[0].path, get_action_id("open")))
        trajectory = sample_trajectory(
            _CloseOnlyPolicy(),
            adapter,
            sampling_config=SamplingConfig(greedy=True),
            initial_state=state,
        )
        self.assertEqual(len(trajectory.steps), 1)
        self.assertEqual(trajectory.sampling_mode, "greedy")
        self.assertFalse(trajectory.training_eligible)
        self.assertEqual(trajectory.steps[0].n_parents, 2)
        self.assertTrue(
            math.isclose(trajectory.steps[0].log_pb, -math.log(2.0), abs_tol=1e-12)
        )
        trajectory.replay(state)

    def test_random_batch_is_terminal_valid_and_differentiable(self):
        torch.manual_seed(17)
        search_space, adapter, model = self.small_components()
        trajectories = sample_trajectories(
            model,
            adapter,
            num_trajectories=6,
            sampling_config=SamplingConfig(),
        )
        self.assertEqual(len(trajectories), 6)
        for trajectory in trajectories:
            trajectory.validate()
            self.assertEqual(trajectory.sampling_mode, "stochastic")
            self.assertTrue(trajectory.training_eligible)
            self.assertLessEqual(len(trajectory.steps), search_space.max_nodes)
            self.assertEqual(
                trajectory.terminal_expression.stats.node_count,
                len(trajectory.steps),
            )
            for step in trajectory.steps:
                if step.normalized_policy_entropy is not None:
                    self.assertGreaterEqual(step.normalized_policy_entropy, 0.0)
                    self.assertLessEqual(step.normalized_policy_entropy, 1.0)
        objective = torch.stack([item.sum_log_pf for item in trajectories]).sum()
        self.assertTrue(objective.requires_grad)
        objective.backward()
        self.assertTrue(any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ))

    def test_greedy_sampling_disables_dropout_and_restores_training_mode(self):
        _, adapter, model = self.small_components(dropout=0.4)
        model.train()
        first = sample_trajectory(
            model, adapter, sampling_config=SamplingConfig(greedy=True)
        )
        self.assertTrue(model.training)
        second = sample_trajectory(
            model, adapter, sampling_config=SamplingConfig(greedy=True)
        )
        self.assertTrue(model.training)
        self.assertEqual(
            first.terminal_expression.structural_hash(),
            second.terminal_expression.structural_hash(),
        )

    def test_random_sampling_repeats_under_same_global_seed(self):
        _, adapter, model = self.small_components()
        torch.manual_seed(991)
        first = sample_trajectory(model, adapter)
        torch.manual_seed(991)
        second = sample_trajectory(model, adapter)
        self.assertEqual(
            [(step.selected_slot_path, step.selected_token_id) for step in first.steps],
            [(step.selected_slot_path, step.selected_token_id) for step in second.steps],
        )

    def test_model_mode_is_restored_when_sampling_raises(self):
        _, adapter, model = self.small_components()
        model.train()
        incompatible = GrammarState(
            search_space=SearchSpaceConfig(max_depth=2, max_nodes=5)
        )
        with self.assertRaises(ValueError):
            sample_trajectory(model, adapter, initial_state=incompatible)
        self.assertTrue(model.training)

    def test_invalid_batch_contracts_are_rejected(self):
        search_space, adapter, model = self.small_components()
        terminal = GrammarState(search_space=search_space).step(
            DAGAction((), get_action_id("close"))
        )
        with self.assertRaises(ValueError):
            sample_trajectories(model, adapter, num_trajectories=0)
        with self.assertRaises(ValueError):
            sample_trajectories(
                model,
                adapter,
                num_trajectories=2,
                initial_states=[GrammarState(search_space=search_space)],
            )
        with self.assertRaises(ValueError):
            sample_trajectory(model, adapter, initial_state=terminal)

    def test_sampling_config_rejects_nonfinite_and_ambiguous_types(self):
        for temperature in (float("nan"), float("inf"), 0.0, -1.0, True):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    SamplingConfig(temperature=temperature)
        for greedy in (0, 1, "yes"):
            with self.subTest(greedy=greedy):
                with self.assertRaises(ValueError):
                    SamplingConfig(greedy=greedy)
        config = SamplingConfig(temperature=1, greedy=False)
        self.assertIsInstance(config.temperature, float)
        self.assertEqual(config.temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
