import unittest

import torch

from factor_gfn.gfn import (
    ForwardPolicyNetwork,
    ModelConfig,
    ROLE_ARG0,
    ROLE_ARG1,
    SearchSpaceConfig,
    StateAdapter,
)
from factor_gfn.grammar import DAGAction, GrammarState, TOTAL_ACTIONS, get_action_id


def fill(state: GrammarState, path: tuple[int, ...], name: str, window: int = 0) -> GrammarState:
    return state.step(DAGAction(path, get_action_id(name, window)))


class ForwardPolicyNetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260805)
        self.search_space = SearchSpaceConfig(max_depth=4, max_nodes=9)
        self.adapter = StateAdapter(self.search_space)
        self.model = ForwardPolicyNetwork(
            ModelConfig(
                d_model=32,
                num_heads=4,
                num_layers=1,
                dim_feedforward=64,
                dropout=0.0,
            ),
            self.search_space,
        )

    def initial(self) -> GrammarState:
        return GrammarState(search_space=self.search_space)

    def test_output_contract_has_one_shared_142_token_head_per_slot(self):
        state = fill(self.initial(), (), "div")
        batch = self.adapter.batch([state])
        output = self.model(batch)
        self.assertEqual(output.slot_logits.shape, (1, 2))
        self.assertEqual(output.token_logits.shape, (1, 2, TOTAL_ACTIONS))
        self.assertEqual(self.model.token_head[-1].out_features, TOTAL_ACTIONS)

    def test_noncommutative_slots_receive_different_conditioned_logits(self):
        state = fill(self.initial(), (), "div")
        batch = self.adapter.batch([state])
        self.assertEqual(batch.slot_role_ids[0, :2].tolist(), [ROLE_ARG0, ROLE_ARG1])
        output = self.model(batch)
        self.assertFalse(torch.allclose(output.token_logits[0, 0], output.token_logits[0, 1]))

    def test_slot_and_token_masks_are_strict_and_normalized(self):
        shallow = self.initial()
        branched = fill(self.initial(), (), "div")
        batch = self.adapter.batch([shallow, branched])
        output = self.model(batch)
        self.assertTrue(torch.isneginf(output.slot_logits[~batch.slot_mask]).all())
        self.assertTrue(torch.isneginf(output.token_logits[~batch.legal_token_mask]).all())
        for row in range(len(batch.states)):
            valid_slots = batch.slot_mask[row]
            self.assertAlmostEqual(
                float(output.slot_log_probs[row, valid_slots].exp().sum()), 1.0, places=6
            )
            for slot in torch.nonzero(valid_slots, as_tuple=False).flatten().tolist():
                valid_tokens = batch.legal_token_mask[row, slot]
                self.assertAlmostEqual(
                    float(output.token_log_probs[row, slot, valid_tokens].exp().sum()),
                    1.0,
                    places=6,
                )

    def test_canonical_state_is_history_invariant(self):
        source = fill(self.initial(), (), "sub")
        left_first = fill(fill(source, (0,), "neg"), (1,), "abs")
        right_first = fill(fill(source, (1,), "abs"), (0,), "neg")
        self.assertEqual(left_first.state_key, right_first.state_key)
        self.model.eval()
        with torch.no_grad():
            first = self.model(self.adapter.batch([left_first]))
            second = self.model(self.adapter.batch([right_first]))
        torch.testing.assert_close(first.slot_logits, second.slot_logits)
        torch.testing.assert_close(first.token_logits, second.token_logits)

    def test_both_policy_heads_receive_gradients(self):
        state = fill(self.initial(), (), "div")
        batch = self.adapter.batch([state])
        output = self.model(batch)
        first_legal_token = int(
            torch.nonzero(batch.legal_token_mask[0, 0], as_tuple=False)[0, 0]
        )
        loss = -(
            output.slot_log_probs[0, 0]
            + output.token_log_probs[0, 0, first_legal_token]
        )
        loss.backward()
        self.assertIsNotNone(self.model.slot_head.weight.grad)
        self.assertGreater(float(self.model.slot_head.weight.grad.abs().sum()), 0.0)
        token_output = self.model.token_head[-1]
        self.assertIsNotNone(token_output.weight.grad)
        self.assertGreater(float(token_output.weight.grad.abs().sum()), 0.0)

    def test_forward_does_not_modify_state_batch_tensors(self):
        batch = self.adapter.batch([fill(self.initial(), (), "div")])
        snapshots = {
            name: getattr(batch, name).clone()
            for name in (
                "token_ids",
                "path_parent_token_ids",
                "slot_node_indices",
                "slot_mask",
                "legal_token_mask",
                "auxiliary_features",
            )
        }
        self.model(batch)
        for name, expected in snapshots.items():
            self.assertTrue(torch.equal(getattr(batch, name), expected), name)

    def test_temperature_and_model_config_validation(self):
        batch = self.adapter.batch([self.initial()])
        with self.assertRaises(ValueError):
            self.model(batch, temperature=0.0)
        with self.assertRaises(ValueError):
            ModelConfig(d_model=30, num_heads=4)

    def test_equal_but_distinct_search_configs_are_accepted(self):
        adapter = StateAdapter(SearchSpaceConfig(max_depth=4, max_nodes=9))
        state = GrammarState(
            search_space=SearchSpaceConfig(max_depth=4, max_nodes=9)
        )
        batch = adapter.batch([state])
        model = ForwardPolicyNetwork(
            ModelConfig(
                d_model=16,
                num_heads=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
            ),
            SearchSpaceConfig(max_depth=4, max_nodes=9),
        )
        output = model(batch)
        self.assertEqual(output.slot_logits.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
