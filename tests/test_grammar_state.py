import math
import unittest

import numpy as np

from factor_gfn.grammar import (
    LEAVES,
    TOTAL_ACTIONS,
    DAGAction,
    GrammarState,
    get_action,
    get_action_id,
)


def fill(state: GrammarState, name: str, window: int = 0, slot_index: int = 0) -> GrammarState:
    slot = state.open_slots()[slot_index]
    return state.step(DAGAction(slot.path, get_action_id(name, window)))


class GrammarStateBoundaryTests(unittest.TestCase):
    def test_constructor_rejects_invalid_limits(self):
        for invalid in (-1, 1.5, True):
            with self.subTest(max_depth=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    GrammarState(max_depth=invalid)
        for invalid in (0, -1, 1.5, True):
            with self.subTest(max_nodes=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    GrammarState(max_nodes=invalid)

    def test_depth_zero_allows_only_six_leaves(self):
        state = GrammarState(max_depth=0, max_nodes=30)
        slot = state.open_slots()[0]
        legal_ids = state.legal_token_ids(slot).tolist()
        self.assertEqual(len(legal_ids), len(LEAVES))
        self.assertTrue(all(get_action(token_id).arity == 0 for token_id in legal_ids))

    def test_node_budget_preserves_a_possible_completion(self):
        expected = {1: {0}, 2: {0, 1}, 3: {0, 1, 2}}
        for max_nodes, arities in expected.items():
            state = GrammarState(max_nodes=max_nodes)
            legal = state.legal_token_ids(state.open_slots()[0])
            self.assertEqual({get_action(int(token)).arity for token in legal}, arities)

    def test_operator_at_depth_limit_is_rejected(self):
        state = fill(GrammarState(max_depth=1, max_nodes=5), "ts_mean", 5)
        slot = state.open_slots()[0]
        self.assertEqual(slot.depth, 1)
        self.assertTrue(
            all(get_action(int(token)).arity == 0 for token in state.legal_token_ids(slot))
        )
        with self.assertRaises(ValueError):
            state.step(DAGAction(slot.path, get_action_id("neg")))

    def test_complete_state_is_immutable_and_has_no_forward_actions(self):
        source = GrammarState()
        terminal = fill(source, "close")
        self.assertFalse(source.done)
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.pending_slots, 0)
        self.assertEqual(terminal.open_slots(), ())
        self.assertEqual(terminal.legal_transitions(), ())
        with self.assertRaises(RuntimeError):
            terminal.step(DAGAction((), get_action_id("open")))

    def test_mask_shape_and_dtype_match_token_space(self):
        state = GrammarState()
        mask = state.get_legal_token_mask(state.open_slots()[0])
        self.assertEqual(mask.shape, (TOTAL_ACTIONS,))
        self.assertEqual(mask.dtype, np.dtype(np.bool_))


class GrammarStateDAGTests(unittest.TestCase):
    def test_commutative_holes_share_one_slot_orbit(self):
        state = fill(GrammarState(max_depth=3, max_nodes=7), "add")
        self.assertEqual(state.pending_slots, 2)
        self.assertEqual(len(state.open_slots()), 1)

    def test_two_fill_orders_merge_for_add(self):
        root = fill(GrammarState(max_depth=3, max_nodes=7), "add")
        open_then_close = fill(fill(root, "open"), "close")
        close_then_open = fill(fill(root, "close"), "open")

        self.assertEqual(open_then_close, close_then_open)
        self.assertEqual(open_then_close.state_key, close_then_open.state_key)
        self.assertEqual(open_then_close.to_expression().to_formula(), "add(close, open)")
        self.assertEqual(open_then_close.count_parents(), 2)
        self.assertAlmostEqual(open_then_close.log_backward_probability(), -math.log(2))

    def test_noncommutative_argument_order_is_preserved(self):
        root = fill(GrammarState(max_depth=3, max_nodes=7), "sub")
        first = fill(fill(root, "open", slot_index=0), "close", slot_index=0)

        root = fill(GrammarState(max_depth=3, max_nodes=7), "sub")
        # 非交换节点有两个不同槽位轨道，可显式选择第二个参数先填。
        second_slot_first = root.step(
            DAGAction(root.open_slots()[1].path, get_action_id("open"))
        )
        second = fill(second_slot_first, "close")

        self.assertEqual(first.to_expression().to_formula(), "sub(open, close)")
        self.assertEqual(second.to_expression().to_formula(), "sub(close, open)")
        self.assertNotEqual(first, second)

    def test_identical_commutative_children_do_not_create_duplicate_parent(self):
        state = fill(GrammarState(max_depth=3, max_nodes=7), "add")
        state = fill(state, "open")
        state = fill(state, "open")
        self.assertEqual(state.count_parents(), 1)

    def test_snapshot_and_auxiliary_features(self):
        state = fill(GrammarState(max_depth=2, max_nodes=5), "add")
        snapshot = state.snapshot()
        self.assertEqual(snapshot.hole_count, 2)
        self.assertEqual(len(snapshot.open_slots), 1)
        self.assertEqual(snapshot.node_count, 1)
        self.assertFalse(snapshot.done)
        np.testing.assert_allclose(
            state.auxiliary_features(),
            np.asarray([0.5, 0.2, 0.2], dtype=np.float32),
        )

    def test_source_has_no_backward_probability(self):
        source = GrammarState()
        self.assertEqual(source.count_parents(), 0)
        with self.assertRaises(ValueError):
            source.log_backward_probability()

    def test_seeded_random_trajectories_are_always_completable(self):
        rng = np.random.default_rng(20260805)
        for trajectory_number in range(300):
            state = GrammarState(max_depth=6, max_nodes=18)
            steps = 0
            while not state.done:
                slots = state.open_slots()
                slot = slots[int(rng.integers(len(slots)))]
                token_id = int(rng.choice(state.legal_token_ids(slot)))
                state = state.step(DAGAction(slot.path, token_id))
                steps += 1
                self.assertLessEqual(steps, state.max_nodes)
            expression = state.to_expression()
            with self.subTest(trajectory=trajectory_number):
                self.assertEqual(expression.stats.node_count, state.node_count)
                self.assertEqual(expression.stats.operator_count, state.operator_count)
                self.assertEqual(expression.stats.depth, state.max_depth_seen)
                self.assertGreaterEqual(state.count_parents(), 1)

    def test_small_graph_parent_enumeration_matches_forward_incoming_edges(self):
        allowed = {
            get_action_id("open"),
            get_action_id("close"),
            get_action_id("neg"),
            get_action_id("add"),
            get_action_id("sub"),
        }
        source = GrammarState(max_depth=2, max_nodes=3)
        states = {source.state_key: source}
        incoming: dict[str, set[str]] = {source.state_key: set()}
        queue = [source]
        while queue:
            parent = queue.pop()
            if parent.done:
                continue
            for action in parent.legal_transitions():
                if action.token_id not in allowed:
                    continue
                child = parent.step(action)
                incoming.setdefault(child.state_key, set()).add(parent.state_key)
                if child.state_key not in states:
                    states[child.state_key] = child
                    queue.append(child)

        for key, child in states.items():
            actual = {item.parent.state_key for item in child.enumerate_parents()}
            # 本测试的所有已用 Token 均在 allowed 中，因此完整父集合也应位于子图内。
            self.assertEqual(actual, incoming.get(key, set()), msg=key)


if __name__ == "__main__":
    unittest.main()
