import math
import unittest

import torch

from factor_gfn.gfn import (
    ForwardPolicyNetwork,
    GFNConfig,
    ModelConfig,
    RewardConfig,
    SamplingConfig,
    SearchSpaceConfig,
    StateAdapter,
    TrainingStats,
    ROLE_ARG0,
    ROLE_ARG1,
    ROLE_COMMUTATIVE_CHILD,
    sample_trajectories,
)
from factor_gfn.grammar import DAGAction, GrammarState, TOTAL_ACTIONS, get_action_id


def fill_path(state: GrammarState, path: tuple[int, ...], name: str, window: int = 0) -> GrammarState:
    return state.step(DAGAction(path, get_action_id(name, window)))


class ConfigContractTests(unittest.TestCase):
    def test_reserved_fields_default_to_none(self):
        reward = RewardConfig()
        stats = TrainingStats()
        self.assertIsNone(reward.reward_clip_min)
        self.assertIsNone(reward.reward_clip_max)
        self.assertIsNone(stats.batch_corr_mean)
        self.assertIsNone(stats.batch_corr_median)

    def test_unimplemented_reward_clip_cannot_be_silently_enabled(self):
        with self.assertRaises(NotImplementedError):
            RewardConfig(reward_clip_max=10.0)

    def test_config_fingerprint_is_stable_and_sensitive(self):
        first = GFNConfig()
        second = GFNConfig()
        changed = GFNConfig(model=ModelConfig(d_model=64, num_heads=4))
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertNotEqual(first.fingerprint(), changed.fingerprint())
        self.assertEqual(len(first.fingerprint()), 64)


class StateAdapterTests(unittest.TestCase):
    def setUp(self):
        self.search_space = SearchSpaceConfig(max_depth=4, max_nodes=9)
        self.adapter = StateAdapter(self.search_space)

    def initial(self) -> GrammarState:
        return GrammarState(search_space=self.search_space)

    def test_different_histories_encode_same_canonical_state(self):
        source = fill_path(self.initial(), (), "sub")
        left_first = fill_path(fill_path(source, (0,), "neg"), (1,), "abs")
        right_first = fill_path(fill_path(source, (1,), "abs"), (0,), "neg")
        self.assertEqual(left_first.state_key, right_first.state_key)
        first = self.adapter.batch([left_first])
        second = self.adapter.batch([right_first])
        for name in (
            "token_ids",
            "depths",
            "role_ids",
            "path_parent_token_ids",
            "path_role_ids",
            "slot_node_indices",
            "slot_mask",
            "legal_token_mask",
        ):
            self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)), name)

    def test_joint_mask_exactly_matches_canonical_transitions(self):
        state = fill_path(self.initial(), (), "sub")
        batch = self.adapter.batch([state])
        encoded = set()
        for slot_index, slot in enumerate(batch.open_slots[0]):
            for token_id in torch.nonzero(
                batch.legal_token_mask[0, slot_index], as_tuple=False
            ).flatten().tolist():
                encoded.add((slot.path, token_id))
        expected = {(action.slot_path, action.token_id) for action in state.legal_transitions()}
        self.assertEqual(encoded, expected)

    def test_argument_roles_preserve_direction_and_commutative_symmetry(self):
        div_state = fill_path(self.initial(), (), "div")
        div_batch = self.adapter.batch([div_state])
        self.assertEqual(
            div_batch.slot_role_ids[0, :2].tolist(),
            [ROLE_ARG0, ROLE_ARG1],
        )
        add_state = fill_path(self.initial(), (), "add")
        add_batch = self.adapter.batch([add_state])
        self.assertEqual(len(add_batch.open_slots[0]), 1)
        self.assertEqual(
            int(add_batch.slot_role_ids[0, 0]),
            ROLE_COMMUTATIVE_CHILD,
        )

    def test_three_manual_features_come_from_grammar_state(self):
        state = fill_path(self.initial(), (), "ts_mean", 20)
        batch = self.adapter.batch([state])
        torch.testing.assert_close(
            batch.auxiliary_features[0],
            torch.from_numpy(state.auxiliary_features()),
        )


class PolicyAndSamplerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.search_space = SearchSpaceConfig(max_depth=3, max_nodes=5)
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

    def test_policy_masks_and_normalizes_both_heads(self):
        state = fill_path(
            GrammarState(max_depth=3, max_nodes=5), (), "div"
        )
        batch = self.adapter.batch([state])
        output = self.model(batch)
        self.assertEqual(output.token_logits.shape[-1], TOTAL_ACTIONS)
        self.assertTrue(torch.isneginf(output.slot_logits[~batch.slot_mask]).all())
        self.assertTrue(
            torch.isneginf(output.token_logits[~batch.legal_token_mask]).all()
        )
        self.assertAlmostEqual(
            float(output.slot_log_probs[0, batch.slot_mask[0]].exp().sum()),
            1.0,
            places=6,
        )
        for slot_index in torch.nonzero(
            batch.slot_mask[0], as_tuple=False
        ).flatten().tolist():
            valid = batch.legal_token_mask[0, slot_index]
            self.assertAlmostEqual(
                float(output.token_log_probs[0, slot_index, valid].exp().sum()),
                1.0,
                places=6,
            )

    def test_depth_zero_leaf_only_policy_runs(self):
        search_space = SearchSpaceConfig(max_depth=0, max_nodes=1)
        adapter = StateAdapter(search_space)
        model = ForwardPolicyNetwork(
            ModelConfig(
                d_model=16,
                num_heads=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
            ),
            search_space,
        )
        state = GrammarState(max_depth=0, max_nodes=1)
        batch = adapter.batch([state])
        output = model(batch)
        valid = batch.legal_token_mask[0, 0]
        self.assertEqual(int(valid.sum()), 6)
        self.assertTrue(torch.isfinite(output.token_log_probs[0, 0, valid]).all())
        self.assertAlmostEqual(
            float(output.token_log_probs[0, 0, valid].exp().sum()),
            1.0,
            places=6,
        )

    def test_policy_rejects_batch_from_different_search_limits(self):
        foreign_config = SearchSpaceConfig(max_depth=3, max_nodes=7)
        foreign_batch = StateAdapter(foreign_config).batch(
            [GrammarState(max_depth=3, max_nodes=7)]
        )
        with self.assertRaisesRegex(ValueError, "SearchSpaceConfig"):
            self.model(foreign_batch)

    def test_action_embedding_separates_window_component(self):
        mean_5 = get_action_id("ts_mean", 5)
        mean_20 = get_action_id("ts_mean", 20)
        self.assertEqual(
            int(self.model.token_operator_lookup[mean_5]),
            int(self.model.token_operator_lookup[mean_20]),
        )
        self.assertNotEqual(
            int(self.model.token_window_lookup[mean_5]),
            int(self.model.token_window_lookup[mean_20]),
        )

    def test_policy_is_history_invariant(self):
        source = fill_path(GrammarState(max_depth=3, max_nodes=5), (), "sub")
        first = fill_path(fill_path(source, (0,), "neg"), (1,), "abs")
        second = fill_path(fill_path(source, (1,), "abs"), (0,), "neg")
        self.model.eval()
        with torch.no_grad():
            first_output = self.model(self.adapter.batch([first]))
            second_output = self.model(self.adapter.batch([second]))
        torch.testing.assert_close(first_output.slot_logits, second_output.slot_logits)
        torch.testing.assert_close(first_output.token_logits, second_output.token_logits)

    def test_sampled_trajectories_are_valid_and_differentiable(self):
        trajectories = sample_trajectories(
            self.model,
            self.adapter,
            num_trajectories=4,
            sampling_config=SamplingConfig(temperature=1.0, greedy=False),
        )
        self.assertEqual(len(trajectories), 4)
        objective = torch.stack([trajectory.sum_log_pf for trajectory in trajectories]).sum()
        self.assertTrue(objective.requires_grad)
        objective.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))
        for trajectory in trajectories:
            trajectory.validate()
            replayed = trajectory.replay(
                GrammarState(max_depth=3, max_nodes=5)
            )
            self.assertTrue(replayed.done)
            self.assertLessEqual(len(trajectory.steps), self.search_space.max_nodes)
            self.assertEqual(
                trajectory.terminal_expression.stats.node_count,
                len(trajectory.steps),
            )
            for step in trajectory.steps:
                self.assertTrue(
                    math.isclose(
                        step.log_pb,
                        -math.log(step.n_parents),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )


if __name__ == "__main__":
    unittest.main()
