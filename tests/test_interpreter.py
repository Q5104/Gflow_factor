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
    DAILY_DERIVED_ACTION_REGISTRY,
    DAILY_DERIVED_FEATURE_NAMES,
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

    def test_read_only_float64_tensor_is_borrowed_but_leaf_result_is_independent(self):
        data = _sample_tensor(days=4, stocks=3)
        data.setflags(write=False)
        interpreter = FactorInterpreter(data)

        self.assertTrue(interpreter.borrows_input_data)
        self.assertTrue(np.shares_memory(interpreter._data, data))
        result = interpreter.evaluate(
            Expression.from_prefix((get_action_id("close"),))
        )
        self.assertFalse(np.shares_memory(result, data))
        self.assertTrue(result.flags.writeable)

    def test_writable_or_infinite_input_is_copied_and_normalized(self):
        writable = _sample_tensor(days=4, stocks=3)
        interpreter = FactorInterpreter(writable)
        expected = writable[:, 3, :].copy()
        writable[:, 3, :] = -999.0

        self.assertFalse(interpreter.borrows_input_data)
        np.testing.assert_allclose(
            interpreter.evaluate(Expression.from_prefix((get_action_id("close"),))),
            expected,
        )

        with_infinity = _sample_tensor(days=4, stocks=3)
        with_infinity[0, 3, 0] = np.inf
        with_infinity.setflags(write=False)
        sanitized = FactorInterpreter(with_infinity)
        self.assertFalse(sanitized.borrows_input_data)
        result = sanitized.evaluate(
            Expression.from_prefix((get_action_id("close"),))
        )
        self.assertTrue(np.isnan(result[0, 0]))

    def test_explicit_schema_rejects_count_and_invalid_names(self):
        data = np.ones((10, 2, 3))
        with self.assertRaisesRegex(ValueError, "feature 轴"):
            FactorInterpreter(data, ordered_feature_names=("only_one",))
        with self.assertRaisesRegex(ValueError, "重复"):
            FactorInterpreter(data, ordered_feature_names=("same", "same"))


class DerivedInterpreterPlumbingTests(unittest.TestCase):
    def test_ret_cc1_reads_exact_derived_index_and_preserves_nan(self):
        names = DAILY_DERIVED_FEATURE_NAMES
        tensor = np.zeros((7, len(names), 2), dtype=np.float64)
        tensor[:, 1, :] = np.arange(14, dtype=np.float64).reshape(7, 2)
        tensor[3, 1, 0] = np.nan
        interpreter = FactorInterpreter(tensor, ordered_feature_names=names)
        expression = Expression.from_prefix(
            (DAILY_DERIVED_ACTION_REGISTRY.get_action_id("ret_cc1"),),
            action_registry=DAILY_DERIVED_ACTION_REGISTRY,
        )

        result = interpreter.evaluate(expression)

        np.testing.assert_allclose(result, tensor[:, 1, :], equal_nan=True)
        self.assertTrue(np.isnan(result[3, 0]))
        self.assertEqual(interpreter.ordered_feature_names, names)

    def test_ts_mean_ret_cc1_uses_same_numeric_operator_path(self):
        names = DAILY_DERIVED_FEATURE_NAMES
        tensor = np.zeros((8, len(names), 1), dtype=np.float64)
        tensor[:, 1, 0] = np.arange(1.0, 9.0)
        interpreter = FactorInterpreter(tensor, ordered_feature_names=names)
        expression = Expression.from_prefix(
            (
                DAILY_DERIVED_ACTION_REGISTRY.get_action_id("ts_mean", 5),
                DAILY_DERIVED_ACTION_REGISTRY.get_action_id("ret_cc1"),
            ),
            action_registry=DAILY_DERIVED_ACTION_REGISTRY,
        )

        result = interpreter.evaluate(expression)

        expected = np.array([np.nan] * 4 + [3.0, 4.0, 5.0, 6.0])[:, None]
        np.testing.assert_allclose(result, expected, equal_nan=True)


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


class InterpreterSubexpressionCacheTests(unittest.TestCase):
    @staticmethod
    def _expression(root: str, child: str, leaf: str) -> Expression:
        return Expression.from_prefix(
            (
                get_action_id(root),
                get_action_id(child, 5),
                get_action_id("close"),
                get_action_id(leaf),
            )
        )

    def test_shared_non_leaf_subexpression_is_reused_without_numeric_change(self):
        data = _sample_tensor(days=30, stocks=4)
        matrix_bytes = data.shape[0] * data.shape[2] * 8
        cached = FactorInterpreter(
            data,
            subexpression_cache_max_bytes=2 * matrix_bytes,
        )
        first = self._expression("add", "ts_mean", "high")
        second = self._expression("sub", "ts_mean", "low")

        first_result = cached.evaluate(first)
        after_first = cached.subexpression_cache_info()
        second_result = cached.evaluate(second)
        after_second = cached.subexpression_cache_info()

        np.testing.assert_allclose(
            first_result,
            FactorInterpreter(data).evaluate(first),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            second_result,
            FactorInterpreter(data).evaluate(second),
            equal_nan=True,
        )
        self.assertEqual(after_first["entries"], 1)
        self.assertEqual(after_first["misses"], 1)
        self.assertEqual(after_second["hits"], 1)
        self.assertTrue(second_result.flags.writeable)

    def test_byte_cap_enforces_lru_eviction(self):
        data = _sample_tensor(days=30, stocks=4)
        matrix_bytes = data.shape[0] * data.shape[2] * 8
        interpreter = FactorInterpreter(
            data,
            subexpression_cache_max_bytes=matrix_bytes,
        )

        interpreter.evaluate(self._expression("add", "ts_mean", "high"))
        interpreter.evaluate(self._expression("add", "ts_std", "high"))
        before_revisit = interpreter.subexpression_cache_info()
        interpreter.evaluate(self._expression("sub", "ts_mean", "low"))
        after_revisit = interpreter.subexpression_cache_info()

        self.assertEqual(before_revisit["entries"], 1)
        self.assertLessEqual(before_revisit["current_bytes"], matrix_bytes)
        self.assertEqual(before_revisit["evictions"], 1)
        self.assertEqual(after_revisit["hits"], 0)
        self.assertEqual(after_revisit["misses"], 3)
        self.assertEqual(after_revisit["evictions"], 2)

    def test_oversized_subexpression_is_not_cached_and_invalid_limit_is_rejected(self):
        data = _sample_tensor(days=30, stocks=4)
        interpreter = FactorInterpreter(
            data,
            subexpression_cache_max_bytes=data.shape[0] * data.shape[2] * 8 - 1,
        )
        expression = self._expression("add", "ts_mean", "high")

        interpreter.evaluate(expression)
        interpreter.evaluate(expression)
        info = interpreter.subexpression_cache_info()

        self.assertEqual(info["entries"], 0)
        self.assertEqual(info["current_bytes"], 0)
        self.assertEqual(info["misses"], 2)
        with self.assertRaisesRegex(ValueError, "非负整数"):
            FactorInterpreter(data, subexpression_cache_max_bytes=-1)
        with self.assertRaisesRegex(ValueError, "非负整数"):
            FactorInterpreter(data, subexpression_cache_max_bytes=True)


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
