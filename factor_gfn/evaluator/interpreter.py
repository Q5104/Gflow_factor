"""将合法表达式解释为日频截面因子矩阵。"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from factor_gfn.grammar.expression import Expression
from factor_gfn.grammar.operators import NON_LEAF_OPERATORS, get_operator
from factor_gfn.grammar.tokens import get_action

from .ops_impl import ALL_OPERATOR_FUNCTIONS


FloatMatrix = npt.NDArray[np.float64]
OperatorFunction = Callable[..., FloatMatrix]

FEATURE_NAMES = ("open", "high", "low", "close", "vwap", "volume")
FEATURE_TO_INDEX = MappingProxyType(
    {name: index for index, name in enumerate(FEATURE_NAMES)}
)

# 解释器直接复用数值层的唯一注册表，不维护第二份手写算子映射。
INTERPRETER_OPERATOR_FUNCTIONS = MappingProxyType(dict(ALL_OPERATOR_FUNCTIONS))


class InterpreterError(RuntimeError):
    """表达式、动作注册表或算子输出不满足解释器合同时抛出。"""


def _prepare_data_tensor(data_tensor: npt.ArrayLike) -> npt.NDArray[np.float64]:
    try:
        data = np.array(data_tensor, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("六特征张量必须能够转换为 float64 数组") from exc

    if data.ndim != 3:
        raise ValueError(
            "六特征张量形状必须为 (date, feature, stock)，"
            f"实际 ndim={data.ndim}"
        )
    if data.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"feature 轴必须恰好包含 {len(FEATURE_NAMES)} 个特征 "
            f"{FEATURE_NAMES}，实际为 {data.shape[1]}"
        )
    if data.shape[0] == 0 or data.shape[2] == 0:
        raise ValueError("date 轴和 stock 轴均不能为空")

    data[~np.isfinite(data)] = np.nan
    data.setflags(write=False)
    return data


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

    def __init__(self, data_tensor: npt.ArrayLike) -> None:
        self._data = _prepare_data_tensor(data_tensor)

    @property
    def data_shape(self) -> tuple[int, int, int]:
        return self._data.shape

    def evaluate(self, expression: Expression) -> FloatMatrix:
        """计算一个完整表达式；不会修改或返回原始张量的可写视图。"""

        if not isinstance(expression, Expression):
            raise TypeError("expression 必须是 Expression 实例")

        stack: list[FloatMatrix] = []
        expected_shape = (self._data.shape[0], self._data.shape[2])

        for position, action_id in enumerate(expression.to_postfix()):
            action = get_action(action_id)
            operator = get_operator(action.name)
            if action.arity != operator.arity:
                raise InterpreterError(
                    f"位置 {position} 的动作 {action.name} 元数不一致："
                    f"action={action.arity}, operator={operator.arity}"
                )
            if operator.requires_window != (action.window != 0):
                raise InterpreterError(
                    f"位置 {position} 的动作 {action.name} 窗口配置与算子签名不一致"
                )

            if action.arity == 0:
                try:
                    feature_index = FEATURE_TO_INDEX[action.name]
                except KeyError as exc:
                    raise InterpreterError(
                        f"位置 {position} 包含未知叶子特征：{action.name!r}"
                    ) from exc
                stack.append(self._data[:, feature_index, :].copy())
                continue

            if len(stack) < action.arity:
                raise InterpreterError(
                    f"位置 {position} 的 {action.name} 需要 {action.arity} 个输入，"
                    f"当前栈中只有 {len(stack)} 个"
                )

            arguments = stack[-action.arity :]
            del stack[-action.arity :]
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
            stack.append(result)

        if len(stack) != 1:
            raise InterpreterError(
                f"后序计算结束后栈长度应为 1，实际为 {len(stack)}"
            )
        return stack[0].copy()


def evaluate_expression(
    data_tensor: npt.ArrayLike, expression: Expression
) -> FloatMatrix:
    """一次性构造解释器并计算表达式的便捷入口。"""

    return FactorInterpreter(data_tensor).evaluate(expression)


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_TO_INDEX",
    "INTERPRETER_OPERATOR_FUNCTIONS",
    "FactorInterpreter",
    "InterpreterError",
    "evaluate_expression",
]
