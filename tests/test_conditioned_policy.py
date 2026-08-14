import unittest
from dataclasses import replace

import torch

from factor_gfn.gfn import (
    CONDITION_SCHEMA,
    ForwardPolicyNetwork,
    ModelConfig,
    SamplingConfig,
    StateAdapter,
    STATE_ADAPTER_SCHEMA,
    sample_trajectory,
    state_hash,
    state_adapter_manifest,
    target_condition_fingerprint,
)
from factor_gfn.grammar import (
    DAGAction,
    ExactNodeGrammarState,
    GrammarState,
    SearchSpaceConfig,
    get_action_id,
)


class ConditionedStateAdapterTests(unittest.TestCase):
    def setUp(self):
        self.search_space = SearchSpaceConfig(max_depth=4, max_nodes=9)
        self.adapter = StateAdapter(self.search_space)

    def source(self, target: int) -> ExactNodeGrammarState:
        return ExactNodeGrammarState.source(
            target_node_count=target,
            search_space=self.search_space,
        )

    def test_condition_input_contract_is_fingerprinted(self):
        manifest = state_adapter_manifest()
        self.assertEqual(CONDITION_SCHEMA, "factor_gfn.exact_node_condition.v1")
        self.assertEqual(STATE_ADAPTER_SCHEMA, "factor_gfn.state_adapter.v2")
        self.assertEqual(
            manifest["condition_features"],
            [
                "target_node_count/max_nodes",
                "(target_node_count-current_node_count)/max_nodes",
            ],
        )
        self.assertEqual(
            manifest["condition_projection"],
            "bias_free_linear_2_to_d_model",
        )

    def test_same_ast_has_conditioned_masks_without_changing_structure_identity(self):
        n1 = self.source(1)
        n3 = self.source(3)
        self.assertEqual(n1.state_key, n3.state_key)
        self.assertNotEqual(n1.conditioned_key, n3.conditioned_key)
        self.assertNotEqual(state_hash(n1), state_hash(n3))
        self.assertFalse(
            torch.equal(
                self.adapter.batch([n1]).legal_token_mask,
                self.adapter.batch([n3]).legal_token_mask,
            )
        )

        n1_terminal = n1.step(DAGAction((), get_action_id("close")))
        structural_terminal = GrammarState(search_space=self.search_space).step(
            DAGAction((), get_action_id("close"))
        )
        self.assertEqual(
            n1_terminal.to_expression().structural_hash(),
            structural_terminal.to_expression().structural_hash(),
        )

    def test_condition_features_are_normalized_and_legacy_features_are_zero(self):
        state = self.source(5).step(DAGAction((), get_action_id("div")))
        legacy = state.state
        batch = self.adapter.batch([state, legacy])
        torch.testing.assert_close(
            batch.condition_features[0],
            torch.tensor((5.0 / 9.0, 4.0 / 9.0)),
        )
        torch.testing.assert_close(
            batch.condition_features[1],
            torch.zeros(2),
        )
        self.assertTrue(
            torch.all(batch.condition_features >= 0.0)
            and torch.all(batch.condition_features <= 1.0)
        )


class ConditionedPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260812)
        self.search_space = SearchSpaceConfig(max_depth=4, max_nodes=9)
        self.adapter = StateAdapter(self.search_space)
        self.model = ForwardPolicyNetwork(
            ModelConfig(
                d_model=32,
                num_heads=4,
                num_layers=1,
                dim_feedforward=64,
                dropout=0.0,
                token_policy_mode="grammar_hierarchical",
            ),
            self.search_space,
        )

    def source(self, target: int) -> ExactNodeGrammarState:
        return ExactNodeGrammarState.source(
            target_node_count=target,
            search_space=self.search_space,
        )

    def test_same_ast_and_mask_produce_condition_dependent_logits(self):
        n3 = self.source(3)
        n5 = self.source(5)
        batch = self.adapter.batch([n3, n5])
        self.assertTrue(torch.equal(batch.legal_token_mask[0], batch.legal_token_mask[1]))
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch)
        self.assertFalse(torch.allclose(output.slot_logits[0], output.slot_logits[1]))

    def test_condition_projection_and_all_grammar_heads_receive_gradients(self):
        state = self.source(5).step(DAGAction((), get_action_id("div")))
        batch = self.adapter.batch([state])
        output = self.model(batch)
        token_id = get_action_id("ts_mean", 20)
        self.assertTrue(batch.legal_token_mask[0, 0, token_id])
        loss = -(output.slot_log_probs[0, 0] + output.token_log_probs[0, 0, token_id])
        loss.backward()
        for head in (
            self.model.condition_projection,
            self.model.slot_head,
            self.model.group_head,
            self.model.grammar_category_head,
            self.model.operator_head,
            self.model.window_head,
        ):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)

    def test_conditioned_trajectory_replay_preserves_actions_terminal_and_n(self):
        trajectory = sample_trajectory(
            self.model,
            self.adapter,
            target_node_count=5,
            sampling_config=SamplingConfig(greedy=True),
        )
        expected_actions = [
            (step.selected_slot_path, step.selected_token_id)
            for step in trajectory.steps
        ]
        replayed = trajectory.replay(
            GrammarState(search_space=self.search_space)
        )
        self.assertEqual(
            expected_actions,
            [
                (step.selected_slot_path, step.selected_token_id)
                for step in trajectory.steps
            ],
        )
        self.assertIsInstance(replayed, ExactNodeGrammarState)
        self.assertEqual(replayed.target_node_count, 5)
        self.assertEqual(replayed.node_count, 5)
        self.assertEqual(trajectory.target_node_count, 5)
        self.assertEqual(trajectory.terminal_node_count, 5)
        self.assertEqual(
            trajectory.condition_fingerprint,
            target_condition_fingerprint(5, self.search_space.fingerprint()),
        )
        self.assertEqual(
            replayed.to_expression().structural_hash(),
            trajectory.terminal_expression.structural_hash(),
        )

        with self.assertRaisesRegex(ValueError, "terminal_node_count"):
            replace(trajectory, terminal_node_count=4).validate()


if __name__ == "__main__":
    unittest.main()
