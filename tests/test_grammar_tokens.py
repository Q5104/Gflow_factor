import unittest

from factor_gfn.grammar.operators import (
    ALL_SYMBOLS,
    BINARY_OPERATORS,
    CROSS_SECTIONAL_OPERATORS,
    LEAVES,
    NON_LEAF_OPERATORS,
    TS_BINARY_OPERATORS,
    TS_UNARY_OPERATORS,
    UNARY_OPERATORS,
    OperatorCategory,
)
from factor_gfn.grammar.tokens import (
    ACTIONS,
    ACTION_TO_ID,
    CATEGORY_TO_INDEX,
    NO_WINDOW,
    OPERATOR_TO_INDEX,
    TOTAL_ACTIONS,
    WINDOWS,
    WINDOW_TO_INDEX,
    get_action,
    get_action_id,
    get_token_indices,
)


class OperatorRegistryTests(unittest.TestCase):
    def test_report_operator_counts_are_complete(self):
        self.assertEqual(len(LEAVES), 6)
        self.assertEqual(len(UNARY_OPERATORS), 12)
        self.assertEqual(len(TS_UNARY_OPERATORS), 17)
        self.assertEqual(len(BINARY_OPERATORS), 10)
        self.assertEqual(len(TS_BINARY_OPERATORS), 4)
        self.assertEqual(len(CROSS_SECTIONAL_OPERATORS), 9)
        self.assertEqual(len(NON_LEAF_OPERATORS), 52)
        self.assertEqual(len(ALL_SYMBOLS), 58)

    def test_names_are_globally_unique(self):
        names = [symbol.name for symbol in ALL_SYMBOLS]
        self.assertEqual(len(names), len(set(names)))

    def test_signatures_match_each_category(self):
        for leaf in LEAVES:
            self.assertEqual(leaf.category, OperatorCategory.LEAF)
            self.assertEqual(leaf.arity, 0)
            self.assertFalse(leaf.requires_window)
        for operator in (*UNARY_OPERATORS, *CROSS_SECTIONAL_OPERATORS):
            self.assertEqual(operator.arity, 1)
            self.assertFalse(operator.requires_window)
        for operator in TS_UNARY_OPERATORS:
            self.assertEqual(operator.arity, 1)
            self.assertTrue(operator.requires_window)
        for operator in BINARY_OPERATORS:
            self.assertEqual(operator.arity, 2)
            self.assertFalse(operator.requires_window)
        for operator in TS_BINARY_OPERATORS:
            self.assertEqual(operator.arity, 2)
            self.assertTrue(operator.requires_window)


class TokenActionSpaceTests(unittest.TestCase):
    def test_full_action_count_and_breakdown(self):
        self.assertEqual(WINDOWS, (5, 10, 20, 40, 60))
        self.assertEqual(TOTAL_ACTIONS, 142)
        self.assertEqual(
            sum(action.category is OperatorCategory.LEAF for action in ACTIONS), 6
        )
        self.assertEqual(
            sum(action.window == NO_WINDOW and action.arity > 0 for action in ACTIONS),
            31,
        )
        self.assertEqual(
            sum(action.window in WINDOWS for action in ACTIONS),
            105,
        )

    def test_every_windowed_operator_has_all_five_variants(self):
        for operator in (*TS_UNARY_OPERATORS, *TS_BINARY_OPERATORS):
            actual = {action.window for action in ACTIONS if action.name == operator.name}
            self.assertEqual(actual, set(WINDOWS), operator.name)

    def test_action_mapping_is_unique_and_reversible(self):
        self.assertEqual(len(ACTION_TO_ID), TOTAL_ACTIONS)
        for expected_id, action in enumerate(ACTIONS):
            self.assertEqual(get_action(expected_id), action)
            self.assertEqual(get_action_id(action.name, action.window), expected_id)

    def test_action_id_order_is_stable_at_group_boundaries(self):
        self.assertEqual(get_action(0).name, "open")
        self.assertEqual(get_action(5).name, "volume")
        self.assertEqual(get_action(6).name, "abs")
        self.assertEqual(get_action(18).name, "add")
        self.assertEqual(get_action(28).name, "cs_rank")
        self.assertEqual((get_action(37).name, get_action(37).window), ("ts_mean", 5))
        self.assertEqual((get_action(41).name, get_action(41).window), ("ts_mean", 60))
        self.assertEqual((get_action(122).name, get_action(122).window), ("ts_corr", 5))
        self.assertEqual((get_action(141).name, get_action(141).window), ("ts_orth", 60))

    def test_embedding_indices_are_valid_and_deterministic(self):
        action_id = get_action_id("ts_mean", 20)
        self.assertEqual(
            get_token_indices(action_id),
            (
                CATEGORY_TO_INDEX[OperatorCategory.TS_UNARY],
                OPERATOR_TO_INDEX["ts_mean"],
                WINDOW_TO_INDEX[20],
            ),
        )

    def test_invalid_action_requests_fail_closed(self):
        with self.assertRaises(IndexError):
            get_action(-1)
        with self.assertRaises(IndexError):
            get_action(TOTAL_ACTIONS)
        with self.assertRaises(TypeError):
            get_action(True)
        with self.assertRaises(ValueError):
            get_action_id("ts_mean", NO_WINDOW)
        with self.assertRaises(ValueError):
            get_action_id("close", 5)
        with self.assertRaises(KeyError):
            get_action_id("unknown_operator")

    def test_numpy_integer_action_id_is_supported(self):
        import numpy as np

        self.assertEqual(get_action(np.int64(0)), get_action(0))


if __name__ == "__main__":
    unittest.main()
