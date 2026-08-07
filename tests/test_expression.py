import unittest

from factor_gfn.grammar import (
    Expression,
    ExpressionNode,
    ExpressionParseError,
    get_action_id,
    validate_postfix,
)


def aid(name: str, window: int = 0) -> int:
    return get_action_id(name, window)


class ExpressionConversionTests(unittest.TestCase):
    def setUp(self):
        # add(close, ts_mean(volume, 20))
        self.prefix = (
            aid("add"),
            aid("close"),
            aid("ts_mean", 20),
            aid("volume"),
        )
        self.postfix = (
            aid("close"),
            aid("volume"),
            aid("ts_mean", 20),
            aid("add"),
        )

    def test_prefix_tree_prefix_round_trip_is_exact(self):
        expression = Expression.from_prefix(self.prefix)
        self.assertEqual(expression.to_prefix(), self.prefix)

    def test_prefix_and_postfix_build_the_same_tree(self):
        from_prefix = Expression.from_prefix(self.prefix)
        from_postfix = Expression.from_postfix(self.postfix)
        self.assertEqual(from_prefix, from_postfix)
        self.assertEqual(from_prefix.to_postfix(), self.postfix)
        validate_postfix(from_prefix.to_postfix())

    def test_formula_and_structure_statistics(self):
        expression = Expression.from_prefix(self.prefix)
        self.assertEqual(
            expression.to_formula(),
            "add(close, ts_mean(volume, 20))",
        )
        self.assertEqual(str(expression), expression.to_formula())
        self.assertEqual(expression.stats.node_count, 4)
        self.assertEqual(expression.stats.operator_count, 2)
        self.assertEqual(expression.stats.leaf_count, 2)
        self.assertEqual(expression.stats.depth, 2)
        self.assertEqual(expression.stats.operator_ratio, 0.5)

    def test_direct_tree_requires_exact_arity(self):
        close = ExpressionNode(aid("close"))
        with self.assertRaises(ValueError):
            ExpressionNode(aid("add"), (close,))
        with self.assertRaises(ValueError):
            ExpressionNode(aid("close"), (close,))


class ExpressionValidationTests(unittest.TestCase):
    def test_empty_prefix_is_rejected(self):
        with self.assertRaises(ExpressionParseError):
            Expression.from_prefix([])

    def test_missing_prefix_children_are_rejected(self):
        with self.assertRaises(ExpressionParseError):
            Expression.from_prefix([aid("add"), aid("close")])
        with self.assertRaises(ExpressionParseError):
            Expression.from_prefix([aid("ts_mean", 5)])

    def test_extra_prefix_tokens_are_rejected(self):
        with self.assertRaises(ExpressionParseError):
            Expression.from_prefix([aid("close"), aid("open")])

    def test_invalid_ids_are_rejected(self):
        with self.assertRaises(ExpressionParseError):
            Expression.from_prefix([999])
        with self.assertRaises(ExpressionParseError):
            Expression.from_postfix([-1])
        with self.assertRaises(ExpressionParseError):
            Expression.from_prefix([True])

    def test_invalid_postfix_stack_shapes_are_rejected(self):
        with self.assertRaises(ExpressionParseError):
            Expression.from_postfix([aid("close"), aid("open")])
        with self.assertRaises(ExpressionParseError):
            Expression.from_postfix([aid("close"), aid("add")])


class ExpressionCanonicalizationTests(unittest.TestCase):
    def expression(self, operator: str, left: str, right: str) -> Expression:
        return Expression.from_prefix([aid(operator), aid(left), aid(right)])

    def test_same_structure_has_same_key_and_hash(self):
        first = self.expression("add", "close", "open")
        second = Expression.from_prefix(first.to_prefix())
        self.assertEqual(first.canonical_key(), second.canonical_key())
        self.assertEqual(first.structural_hash(), second.structural_hash())
        self.assertEqual(len(first.structural_hash()), 64)

    def test_known_expression_hash_is_stable(self):
        expression = Expression.from_prefix(
            [
                aid("add"),
                aid("close"),
                aid("ts_mean", 20),
                aid("volume"),
            ]
        )
        self.assertEqual(
            expression.canonical_key(),
            '["add",0,[["close",0,[]],["ts_mean",20,[["volume",0,[]]]]]]',
        )
        self.assertEqual(
            expression.structural_hash(),
            "c0091e25f929b7dca9882319b2e6eb8299e3e0e0ddfccbeade044f3f55fc4b1c",
        )

    def test_declared_commutative_operators_deduplicate_swapped_children(self):
        for operator in ("add", "mul", "max2", "min2"):
            with self.subTest(operator=operator):
                left_right = self.expression(operator, "close", "open")
                right_left = self.expression(operator, "open", "close")
                self.assertEqual(
                    left_right.canonical_key(), right_left.canonical_key()
                )
                self.assertEqual(
                    left_right.structural_hash(), right_left.structural_hash()
                )
                self.assertEqual(
                    left_right.canonicalize().to_prefix(),
                    right_left.canonicalize().to_prefix(),
                )

    def test_non_commutative_operators_keep_child_order(self):
        for operator in ("sub", "div", "greater", "less", "signed_ratio", "log_ratio"):
            with self.subTest(operator=operator):
                left_right = self.expression(operator, "close", "open")
                right_left = self.expression(operator, "open", "close")
                self.assertNotEqual(
                    left_right.canonical_key(), right_left.canonical_key()
                )
                self.assertNotEqual(
                    left_right.structural_hash(), right_left.structural_hash()
                )

    def test_all_ts_binary_operators_keep_argument_order(self):
        for operator in ("ts_corr", "ts_cov", "ts_beta", "ts_orth"):
            with self.subTest(operator=operator):
                left_right = Expression.from_prefix(
                    [aid(operator, 5), aid("close"), aid("open")]
                )
                right_left = Expression.from_prefix(
                    [aid(operator, 5), aid("open"), aid("close")]
                )
                self.assertNotEqual(
                    left_right.structural_hash(), right_left.structural_hash()
                )

    def test_canonicalization_is_recursive_without_algebraic_simplification(self):
        # add(mul(volume, close), open) 与 add(open, mul(close, volume))
        first = Expression.from_prefix(
            [
                aid("add"),
                aid("mul"),
                aid("volume"),
                aid("close"),
                aid("open"),
            ]
        )
        second = Expression.from_prefix(
            [
                aid("add"),
                aid("open"),
                aid("mul"),
                aid("close"),
                aid("volume"),
            ]
        )
        self.assertEqual(first.structural_hash(), second.structural_hash())
        self.assertEqual(first.canonicalize().stats.node_count, 5)
        self.assertEqual(first.canonicalize().root.action.name, "add")


if __name__ == "__main__":
    unittest.main()
