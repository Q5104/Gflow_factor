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
from factor_gfn.grammar import (
    ACTIONS,
    DAGAction,
    GrammarState,
    TOTAL_ACTIONS,
    get_action_id,
)


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
        with self.assertRaisesRegex(ValueError, "token_policy_mode"):
            ModelConfig(token_policy_mode="unknown")

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


class HierarchicalForwardPolicyNetworkTests(unittest.TestCase):
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
                token_policy_mode="arity_hierarchical",
            ),
            self.search_space,
        )

    def test_arity_mapping_covers_each_token_once(self):
        groups = self.model.token_group_lookup.tolist()
        self.assertEqual(len(groups), TOTAL_ACTIONS)
        self.assertEqual(groups, [action.arity for action in ACTIONS])
        self.assertEqual(groups.count(0), 6)
        self.assertEqual(groups.count(1), 106)
        self.assertEqual(groups.count(2), 30)

    def test_zero_initialized_group_head_gives_equal_root_group_mass(self):
        batch = self.adapter.batch([GrammarState(search_space=self.search_space)])
        output = self.model(batch)
        self.assertIsNotNone(output.group_log_probs)
        self.assertIsNotNone(output.legal_group_mask)
        self.assertEqual(output.legal_group_mask[0, 0].tolist(), [True, True, True])
        torch.testing.assert_close(
            output.group_log_probs[0, 0].exp(),
            torch.full((3,), 1.0 / 3.0),
            rtol=1e-6,
            atol=1e-6,
        )
        token_probabilities = output.token_log_probs[0, 0].exp()
        for group_id in range(3):
            members = self.model.token_group_lookup == group_id
            self.assertAlmostEqual(
                float(token_probabilities[members].sum()),
                1.0 / 3.0,
                places=6,
            )

    def test_illegal_groups_are_masked_and_joint_distribution_normalizes(self):
        constrained = SearchSpaceConfig(max_depth=0, max_nodes=1)
        adapter = StateAdapter(constrained)
        model = ForwardPolicyNetwork(
            ModelConfig(
                d_model=16,
                num_heads=4,
                num_layers=1,
                dim_feedforward=32,
                token_policy_mode="arity_hierarchical",
            ),
            constrained,
        )
        batch = adapter.batch([GrammarState(search_space=constrained)])
        output = model(batch)
        self.assertEqual(output.legal_group_mask[0, 0].tolist(), [True, False, False])
        self.assertEqual(output.group_log_probs[0, 0].exp().tolist(), [1.0, 0.0, 0.0])
        self.assertAlmostEqual(
            float(output.token_log_probs[0, 0].exp().sum()), 1.0, places=6
        )
        self.assertTrue(
            torch.isneginf(output.token_log_probs[0, 0][~batch.legal_token_mask[0, 0]]).all()
        )

    def test_slot_group_and_token_heads_all_receive_gradients(self):
        state = fill(GrammarState(search_space=self.search_space), (), "div")
        batch = self.adapter.batch([state])
        output = self.model(batch)
        close_id = get_action_id("close")
        loss = -(
            output.slot_log_probs[0, 0]
            + output.token_log_probs[0, 0, close_id]
        )
        loss.backward()
        self.assertGreater(float(self.model.slot_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(self.model.group_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(self.model.token_head[-1].weight.grad.abs().sum()), 0.0)

    def test_flat_mode_keeps_historical_output_contract(self):
        flat = ForwardPolicyNetwork(
            ModelConfig(
                d_model=16,
                num_heads=4,
                num_layers=1,
                dim_feedforward=32,
                token_policy_mode="flat",
            ),
            self.search_space,
        )
        output = flat(self.adapter.batch([GrammarState(search_space=self.search_space)]))
        self.assertIsNone(flat.group_head)
        self.assertIsNone(output.group_logits)
        self.assertIsNone(output.group_log_probs)
        self.assertIsNone(output.legal_group_mask)


class GrammarHierarchicalForwardPolicyNetworkTests(unittest.TestCase):
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

    def test_initial_probability_mass_follows_grammar_factorization(self):
        batch = self.adapter.batch([GrammarState(search_space=self.search_space)])
        output = self.model(batch)
        torch.testing.assert_close(
            output.group_log_probs[0, 0].exp(),
            torch.full((3,), 1.0 / 3.0),
            rtol=1e-6,
            atol=1e-6,
        )
        expected_categories = torch.tensor(
            (1.0 / 3.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 9.0)
        )
        torch.testing.assert_close(
            output.grammar_category_log_probs[0, 0].exp(),
            expected_categories,
            rtol=1e-6,
            atol=1e-6,
        )
        probabilities = output.token_log_probs[0, 0].exp()
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(probabilities[get_action_id("close")]), 1.0 / 18.0, places=6)
        ts_mean_mass = sum(
            float(probabilities[get_action_id("ts_mean", window)])
            for window in (5, 10, 20, 40, 60)
        )
        self.assertAlmostEqual(ts_mean_mass, (1.0 / 9.0) / 17.0, places=6)
        window_probs = output.window_log_probs[
            0, 0, self.model.action_operator_lookup[get_action_id("ts_mean", 5)]
        ].exp()
        torch.testing.assert_close(
            window_probs,
            torch.full((5,), 0.2),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_joint_distribution_respects_strict_legal_mask(self):
        constrained = SearchSpaceConfig(max_depth=0, max_nodes=1)
        adapter = StateAdapter(constrained)
        model = ForwardPolicyNetwork(
            ModelConfig(
                d_model=16,
                num_heads=4,
                num_layers=1,
                dim_feedforward=32,
                token_policy_mode="grammar_hierarchical",
            ),
            constrained,
        )
        batch = adapter.batch([GrammarState(search_space=constrained)])
        output = model(batch)
        self.assertEqual(output.legal_group_mask[0, 0].tolist(), [True, False, False])
        self.assertAlmostEqual(float(output.token_log_probs[0, 0].exp().sum()), 1.0, places=6)
        self.assertTrue(
            torch.isneginf(output.token_log_probs[0, 0][~batch.legal_token_mask[0, 0]]).all()
        )

    def test_all_conditional_heads_receive_gradients(self):
        state = fill(GrammarState(search_space=self.search_space), (), "div")
        batch = self.adapter.batch([state])
        output = self.model(batch)
        token_id = get_action_id("ts_mean", 20)
        loss = -(output.slot_log_probs[0, 0] + output.token_log_probs[0, 0, token_id])
        loss.backward()
        for head in (
            self.model.slot_head,
            self.model.group_head,
            self.model.grammar_category_head,
            self.model.operator_head,
            self.model.window_head,
        ):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
