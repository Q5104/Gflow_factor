"""将合法表达式解释为日频截面因子矩阵。"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from factor_gfn.grammar.expression import Expression, ExpressionNode
from factor_gfn.grammar.operators import NON_LEAF_OPERATORS, get_operator

from .ops_impl import ALL_OPERATOR_FUNCTIONS


FloatMatrix = npt.NDArray[np.float64]
OperatorFunction = Callable[..., FloatMatrix]

FEATURE_NAMES = ("open", "high", "low", "close", "vwap", "volume")
FEATURE_TO_INDEX = MappingProxyType(
    {name: index for index, name in enumerate(FEATURE_NAMES)}
)

# 解释器直接复用数值层的唯一注册表，不维护第二份手写算子映射。
INTERPRETER_OPERATOR_FUNCTIONS = MappingProxyType(dict(ALL_OPERATOR_FUNCTIONS))
SUBEXPRESSION_CACHE_SCHEMA = "factor_gfn.subexpression_lru.v1"


class InterpreterError(RuntimeError):
    """表达式、动作注册表或算子输出不满足解释器合同时抛出。"""


def _prepare_data_tensor(
    data_tensor: npt.ArrayLike,
    ordered_feature_names: tuple[str, ...],
) -> tuple[npt.NDArray[np.float64], bool]:
    try:
        candidate = np.asarray(data_tensor, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("expression feature tensor 必须能够转换为 float64 数组") from exc

    if candidate.ndim != 3:
        raise ValueError(
            "expression feature tensor 形状必须为 (date, feature, stock)，"
            f"实际 ndim={candidate.ndim}"
        )
    if candidate.shape[1] != len(ordered_feature_names):
        raise ValueError(
            f"feature 轴必须恰好包含 {len(ordered_feature_names)} 个特征 "
            f"{ordered_feature_names}，实际为 {candidate.shape[1]}"
        )
    if candidate.shape[0] == 0 or candidate.shape[2] == 0:
        raise ValueError("date 轴和 stock 轴均不能为空")

    borrowed = not candidate.flags.writeable and not np.isinf(candidate).any()
    data = candidate.view() if borrowed else candidate.copy()
    if not borrowed:
        data[~np.isfinite(data)] = np.nan
    data.setflags(write=False)
    return data, borrowed


def _normalize_feature_names(
    ordered_feature_names: tuple[str, ...],
) -> tuple[str, ...]:
    try:
        names = tuple(ordered_feature_names)
    except TypeError as exc:
        raise TypeError("ordered_feature_names 必须是有序特征名序列") from exc
    if not names:
        raise ValueError("ordered_feature_names 不能为空")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("ordered_feature_names 必须全部为非空字符串")
    if len(set(names)) != len(names):
        raise ValueError("ordered_feature_names 不允许重复特征名")
    return names


def _validate_dispatch_registry() -> None:
    expected = {operator.name for operator in NON_LEAF_OPERATORS}
    actual = set(INTERPRETER_OPERATOR_FUNCTIONS)
    if actual != expected or len(actual) != 52:
        raise RuntimeError(
            "解释器算子映射与文法注册表不一致："
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


_validate_dispatch_registry()


class FactorInterpreter:
    """基于后序栈计算表达式，输出形状为 ``(date, stock)`` 的矩阵。"""

    def __init__(
        self,
        data_tensor: npt.ArrayLike,
        *,
        ordered_feature_names: tuple[str, ...] = FEATURE_NAMES,
        subexpression_cache_max_bytes: int = 0,
    ) -> None:
        if (
            isinstance(subexpression_cache_max_bytes, bool)
            or not isinstance(subexpression_cache_max_bytes, int)
            or subexpression_cache_max_bytes < 0
        ):
            raise ValueError("subexpression_cache_max_bytes 必须是非负整数")
        self._ordered_feature_names = _normalize_feature_names(ordered_feature_names)
        self._feature_to_index = MappingProxyType(
            {name: index for index, name in enumerate(self._ordered_feature_names)}
        )
        self._data, self._borrows_input_data = _prepare_data_tensor(
            data_tensor,
            self._ordered_feature_names,
        )
        self._subexpression_cache_max_bytes = subexpression_cache_max_bytes
        self._subexpression_cache_bytes = 0
        self._subexpression_cache: OrderedDict[str, FloatMatrix] = OrderedDict()
        self._subexpression_cache_hits = 0
        self._subexpression_cache_misses = 0
        self._subexpression_cache_evictions = 0

    @property
    def data_shape(self) -> tuple[int, int, int]:
        return self._data.shape

    @property
    def borrows_input_data(self) -> bool:
        """是否直接借用调用方提供的只读 ``float64`` 存储。"""

        return self._borrows_input_data

    @property
    def ordered_feature_names(self) -> tuple[str, ...]:
        return self._ordered_feature_names

    def subexpression_cache_info(self) -> dict[str, int | str]:
        return {
            "schema": SUBEXPRESSION_CACHE_SCHEMA,
            "max_bytes": self._subexpression_cache_max_bytes,
            "current_bytes": self._subexpression_cache_bytes,
            "entries": len(self._subexpression_cache),
            "hits": self._subexpression_cache_hits,
            "misses": self._subexpression_cache_misses,
            "evictions": self._subexpression_cache_evictions,
        }

    def clear_subexpression_cache(self) -> None:
        self._subexpression_cache.clear()
        self._subexpression_cache_bytes = 0

    def _subexpression_cache_get(self, key: str) -> FloatMatrix | None:
        cached = self._subexpression_cache.get(key)
        if cached is None:
            self._subexpression_cache_misses += 1
            return None
        self._subexpression_cache.move_to_end(key)
        self._subexpression_cache_hits += 1
        return cached

    def _subexpression_cache_put(self, key: str, value: FloatMatrix) -> None:
        if (
            self._subexpression_cache_max_bytes == 0
            or value.nbytes > self._subexpression_cache_max_bytes
        ):
            return
        previous = self._subexpression_cache.pop(key, None)
        if previous is not None:
            self._subexpression_cache_bytes -= previous.nbytes
        while (
            self._subexpression_cache
            and self._subexpression_cache_bytes + value.nbytes
            > self._subexpression_cache_max_bytes
        ):
            _, evicted = self._subexpression_cache.popitem(last=False)
            self._subexpression_cache_bytes -= evicted.nbytes
            self._subexpression_cache_evictions += 1
        value.setflags(write=False)
        self._subexpression_cache[key] = value
        self._subexpression_cache_bytes += value.nbytes

    def evaluate(self, expression: Expression) -> FloatMatrix:
        """计算一个完整表达式；不会修改或返回原始张量的可写视图。"""

        if not isinstance(expression, Expression):
            raise TypeError("expression 必须是 Expression 实例")
        if expression.action_registry.leaf_names != self._ordered_feature_names:
            raise ValueError("Expression ActionRegistry 与 Interpreter feature schema 不一致")

        expected_shape = (self._data.shape[0], self._data.shape[2])

        def evaluate_node(node: ExpressionNode, *, is_root: bool) -> FloatMatrix:
            action = expression.action_registry.get_action(node.action_id)
            if action.arity == 0:
                if action.window != 0:
                    raise InterpreterError(
                        f"叶子特征 {action.name!r} 不允许携带窗口 {action.window}"
                    )
                try:
                    feature_index = self._feature_to_index[action.name]
                except KeyError as exc:
                    raise InterpreterError(
                        f"包含 schema 未登记的叶子特征：{action.name!r}"
                    ) from exc
                return self._data[:, feature_index, :]

            operator = get_operator(action.name)
            if action.arity != operator.arity:
                raise InterpreterError(
                    f"动作 {action.name} 元数不一致："
                    f"action={action.arity}, operator={operator.arity}"
                )
            if operator.requires_window != (action.window != 0):
                raise InterpreterError(
                    f"动作 {action.name} 窗口配置与算子签名不一致"
                )

            cache_key: str | None = None
            if not is_root and self._subexpression_cache_max_bytes > 0:
                cache_key = Expression(node).structural_hash()
                cached = self._subexpression_cache_get(cache_key)
                if cached is not None:
                    return cached

            arguments = [
                evaluate_node(child, is_root=False) for child in node.children
            ]
            try:
                function = INTERPRETER_OPERATOR_FUNCTIONS[action.name]
            except KeyError as exc:
                raise InterpreterError(f"算子 {action.name!r} 没有解释器映射") from exc

            if action.window == 0:
                result = function(*arguments)
            else:
                result = function(*arguments, action.window)

            result = np.asarray(result, dtype=np.float64)
            if result.shape != expected_shape:
                raise InterpreterError(
                    f"算子 {action.name} 输出形状应为 {expected_shape}，"
                    f"实际为 {result.shape}"
                )
            if np.isinf(result).any():
                raise InterpreterError(f"算子 {action.name} 返回了 Inf，违反数值层合同")
            if cache_key is not None:
                self._subexpression_cache_put(cache_key, result)
            return result

        result = evaluate_node(expression.root, is_root=True)
        if np.shares_memory(result, self._data):
            return result.copy()
        return result


def evaluate_expression(
    data_tensor: npt.ArrayLike,
    expression: Expression,
    *,
    ordered_feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> FloatMatrix:
    """一次性构造解释器并计算表达式的便捷入口。"""

    return FactorInterpreter(
        data_tensor,
        ordered_feature_names=ordered_feature_names,
    ).evaluate(expression)


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_TO_INDEX",
    "INTERPRETER_OPERATOR_FUNCTIONS",
    "SUBEXPRESSION_CACHE_SCHEMA",
    "FactorInterpreter",
    "InterpreterError",
    "evaluate_expression",
]
