import unittest

import numpy as np

from factor_gfn.evaluator import (
    FEATURE_NAMES,
    INTERPRETER_OPERATOR_FUNCTIONS,
    FactorInterpreter,
    evaluate_expression,
)
from factor_gfn.grammar import (
    DAGAction,
    NON_LEAF_OPERATORS,
    Expression,
    GrammarState,
    get_action_id,
)


def _sample_tensor(days=70, stocks=5):
    rng = np.random.default_rng(20260804)
    data = rng.uniform(1.0, 20.0, size=(days, 6, stocks))
    data[:, 5, :] = rng.uniform(100.0, 10_000.0, size=(days, stocks))
    return data


class InterpreterInputContractTests(unittest.TestCase):
    def test_feature_contract_and_invalid_shapes(self):
        self.assertEqual(
            FEATURE_NAMES, ("open", "high", "low", "close", "vwap", "volume")
        )
        with self.assertRaises(ValueError):
            FactorInterpreter(np.ones((10, 5)))
        with self.assertRaises(ValueError):
            FactorInterpreter(np.ones((10, 5, 3)))
        with self.assertRaises(ValueError):
            FactorInterpreter(np.ones((0, 6, 3)))
        with self.assertRaises(ValueError):
            FactorInterpreter(np.ones((10, 6, 0)))

    def test_leaf_evaluation_is_float64_and_does_not_share_memory(self):
        data = _sample_tensor(days=4, stocks=3).astype(np.float32)
        original = data.copy()
        expression = Expression.from_prefix((get_action_id("close"),))

        result = FactorInterpreter(data).evaluate(expression)
        np.testing.assert_allclose(result, original[:, 3, :])
        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertFalse(np.shares_memory(result, data))

        result[0, 0] = -999.0
        np.testing.assert_array_equal(data, original)


class InterpreterEvaluationTests(unittest.TestCase):
    def test_manual_add_close_and_mean_volume_formula(self):
        days, stocks = 25, 3
        data = np.zeros((days, 6, stocks), dtype=np.float64)
        data[:, 3, :] = np.arange(days, dtype=np.float64)[:, None] + 1.0
        data[:, 5, :] = np.arange(days, dtype=np.float64)[:, None] + 10.0
        prefix = (
            get_action_id("add"),
            get_action_id("close"),
            get_action_id("ts_mean", 20),
            get_action_id("volume"),
        )
        expression = Expression.from_prefix(prefix)

        expected = np.full((days, stocks), np.nan)
        for day in range(19, days):
            expected[day] = data[day, 3] + data[day - 19 : day + 1, 5].mean(axis=0)

        result = evaluate_expression(data, expression)
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_prefix_to_postfix_round_trip_has_same_result(self):
        data = _sample_tensor()
        prefix = (
            get_action_id("sub"),
            get_action_id("cs_rank"),
            get_action_id("close"),
            get_action_id("ts_delta", 5),
            get_action_id("vwap"),
        )
        from_prefix = Expression.from_prefix(prefix)
        from_postfix = Expression.from_postfix(from_prefix.to_postfix())
        interpreter = FactorInterpreter(data)

        np.testing.assert_allclose(
            interpreter.evaluate(from_prefix),
            interpreter.evaluate(from_postfix),
            equal_nan=True,
        )

    def test_interpreter_does_not_modify_source_tensor(self):
        data = _sample_tensor()
        data[8, 2, 1] = np.nan
        original = data.copy()
        expression = Expression.from_prefix(
            (
                get_action_id("mul"),
                get_action_id("ts_std", 5),
                get_action_id("low"),
                get_action_id("cs_zscore"),
                get_action_id("volume"),
            )
        )

        FactorInterpreter(data).evaluate(expression)
        np.testing.assert_equal(data, original)


class InterpreterCoverageTests(unittest.TestCase):
    def test_all_52_non_leaf_operators_have_exactly_one_mapping(self):
        expected = {operator.name for operator in NON_LEAF_OPERATORS}
        self.assertEqual(len(expected), 52)
        self.assertEqual(set(INTERPRETER_OPERATOR_FUNCTIONS), expected)

    def test_every_operator_is_reachable_through_an_expression(self):
        data = _sample_tensor(days=70, stocks=6)
        interpreter = FactorInterpreter(data)
        for operator in NON_LEAF_OPERATORS:
            window = 5 if operator.requires_window else 0
            children = [get_action_id("close")]
            if operator.arity == 2:
                children.append(get_action_id("volume"))
            prefix = (get_action_id(operator.name, window), *children)
            expression = Expression.from_prefix(prefix)
            with self.subTest(operator=operator.name):
                result = interpreter.evaluate(expression)
                self.assertEqual(result.shape, (70, 6))
                self.assertEqual(result.dtype, np.dtype(np.float64))
                self.assertFalse(np.isinf(result).any())

    def test_seeded_random_legal_expressions_evaluate_or_return_nan(self):
        rng = np.random.default_rng(314159)
        data = _sample_tensor(days=70, stocks=6)
        data[12, :, 2] = np.nan
        original = data.copy()
        interpreter = FactorInterpreter(data)

        for trajectory in range(200):
            state = GrammarState(max_depth=4, max_nodes=12)
            while not state.done:
                slots = state.open_slots()
                slot = slots[int(rng.integers(len(slots)))]
                token_id = int(rng.choice(state.legal_token_ids(slot)))
                state = state.step(DAGAction(slot.path, token_id))
            expression = state.to_expression()
            with self.subTest(trajectory=trajectory, formula=expression.to_formula()):
                result = interpreter.evaluate(expression)
                self.assertEqual(result.shape, (70, 6))
                self.assertFalse(np.isinf(result).any())

        np.testing.assert_equal(data, original)


if __name__ == "__main__":
    unittest.main()
