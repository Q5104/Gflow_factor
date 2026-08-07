import unittest

import numpy as np

from factor_gfn.grammar import (
    ACTION_SPACE_SCHEMA,
    STATE_SPACE_SCHEMA,
    TRANSITION_SPACE_SCHEMA,
    TOTAL_ACTIONS,
    DAGAction,
    Expression,
    GrammarState,
    action_space_fingerprint,
    action_space_manifest,
    get_action_id,
    state_space_fingerprint,
    transition_space_fingerprint,
    validate_postfix,
)


class SpaceFingerprintTests(unittest.TestCase):
    def test_token_manifest_and_fingerprint_are_unchanged(self):
        manifest = action_space_manifest()
        self.assertEqual(len(manifest), TOTAL_ACTIONS)
        self.assertEqual(tuple(row[0] for row in manifest), tuple(range(TOTAL_ACTIONS)))
        self.assertEqual(manifest[0], (0, "LEAF", "open", 0, 0))
        self.assertEqual(manifest[-1], (141, "TS_BINARY_OP", "ts_orth", 60, 2))
        self.assertEqual(ACTION_SPACE_SCHEMA, "factor_gfn.action_space.v1")
        self.assertEqual(
            action_space_fingerprint(),
            "5689dbceb1bb42716773bcaf4cb5845041e578a3bb11fe67445ede6cde7938cc",
        )

    def test_state_and_transition_protocols_have_independent_fingerprints(self):
        self.assertEqual(STATE_SPACE_SCHEMA, "factor_gfn.dag_state.v1")
        self.assertEqual(TRANSITION_SPACE_SCHEMA, "factor_gfn.dag_transition.v1")
        self.assertEqual(
            state_space_fingerprint(),
            "5301fb14e197376dfc2a5aaf9c398aa47642bcc889f6eef87278839142b824d9",
        )
        self.assertEqual(
            transition_space_fingerprint(),
            "bb9c95a05c0bccb375c47828fb6d95821db1f625f4a2513ba55e7151bac55a4b",
        )


class GrammarExpressionIntegrationTests(unittest.TestCase):
    def test_every_token_can_start_and_complete_a_dag_expression(self):
        for token_id in range(TOTAL_ACTIONS):
            state = GrammarState(max_depth=2, max_nodes=5)
            state = state.step(DAGAction((), token_id))
            while not state.done:
                slot = state.open_slots()[0]
                state = state.step(DAGAction(slot.path, get_action_id("close")))
            with self.subTest(token_id=token_id):
                self.assertTrue(state.done)
                self.assertEqual(state.to_expression().root.action_id, token_id)

    def test_known_dag_trajectory_converts_to_existing_expression_interface(self):
        # add(close, ts_mean(volume, 20))，按任意开放槽位填充。
        state = GrammarState(max_depth=4, max_nodes=10)
        state = state.step(DAGAction((), get_action_id("add")))
        # add 是交换节点，两个 Hole 属于同一槽位轨道。
        state = state.step(DAGAction(state.open_slots()[0].path, get_action_id("close")))
        state = state.step(
            DAGAction(state.open_slots()[0].path, get_action_id("ts_mean", 20))
        )
        state = state.step(DAGAction(state.open_slots()[0].path, get_action_id("volume")))

        expression = state.to_expression()
        postfix = expression.to_postfix()
        validate_postfix(postfix)
        rebuilt = Expression.from_postfix(postfix)

        self.assertTrue(state.done)
        self.assertEqual(expression, rebuilt)
        self.assertEqual(expression.stats.node_count, state.node_count)
        self.assertEqual(expression.stats.operator_count, state.operator_count)
        self.assertEqual(expression.stats.depth, state.max_depth_seen)

    def test_seeded_random_dag_trajectories_match_expression_statistics(self):
        rng = np.random.default_rng(20260805)
        limits = ((0, 1), (1, 3), (3, 10), (8, 24))

        for trajectory_number in range(300):
            max_depth, max_nodes = limits[trajectory_number % len(limits)]
            state = GrammarState(max_depth=max_depth, max_nodes=max_nodes)
            while not state.done:
                slots = state.open_slots()
                slot = slots[int(rng.integers(len(slots)))]
                token_id = int(rng.choice(state.legal_token_ids(slot)))
                state = state.step(DAGAction(slot.path, token_id))

            expression = state.to_expression()
            rebuilt = Expression.from_postfix(expression.to_postfix())
            canonical = expression.canonicalize()

            with self.subTest(trajectory=trajectory_number):
                self.assertEqual(rebuilt, expression)
                self.assertEqual(expression, canonical)
                self.assertEqual(expression.stats.node_count, state.node_count)
                self.assertEqual(expression.stats.operator_count, state.operator_count)
                self.assertEqual(expression.stats.depth, state.max_depth_seen)
                self.assertEqual(canonical.structural_hash(), expression.structural_hash())


if __name__ == "__main__":
    unittest.main()
