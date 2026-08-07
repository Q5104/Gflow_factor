"""将文法签名展开为稳定、可逆的表达式动作空间。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

from .operators import (
    ALL_SYMBOLS,
    BINARY_OPERATORS,
    CROSS_SECTIONAL_OPERATORS,
    LEAVES,
    TS_BINARY_OPERATORS,
    TS_UNARY_OPERATORS,
    UNARY_OPERATORS,
    OperatorCategory,
    OperatorSpec,
    get_operator,
)


WINDOWS = (5, 10, 20, 40, 60)
NO_WINDOW = 0
ACTION_SPACE_SCHEMA = "factor_gfn.action_space.v1"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """一个可由生成策略选择的表达式动作。"""

    category: OperatorCategory
    name: str
    window: int
    arity: int

    @classmethod
    def from_operator(cls, operator: OperatorSpec, window: int) -> ActionSpec:
        if operator.requires_window:
            if window not in WINDOWS:
                raise ValueError(
                    f"时序算子 {operator.name} 的窗口必须属于 {WINDOWS}"
                )
        elif window != NO_WINDOW:
            raise ValueError(f"无窗口节点 {operator.name} 的窗口必须为 0")
        return cls(
            category=operator.category,
            name=operator.name,
            window=window,
            arity=operator.arity,
        )

    @property
    def key(self) -> tuple[OperatorCategory, str, int, int]:
        return (self.category, self.name, self.window, self.arity)


def _build_actions() -> tuple[ActionSpec, ...]:
    actions: list[ActionSpec] = []

    # 稳定顺序：叶子、无窗口一元、无窗口二元、截面、带窗口一元时序、
    # 带窗口二元时序。未来不得无意改变，否则历史模型的 action_id 会失效。
    for operator in LEAVES:
        actions.append(ActionSpec.from_operator(operator, NO_WINDOW))
    for group in (UNARY_OPERATORS, BINARY_OPERATORS, CROSS_SECTIONAL_OPERATORS):
        for operator in group:
            actions.append(ActionSpec.from_operator(operator, NO_WINDOW))
    for group in (TS_UNARY_OPERATORS, TS_BINARY_OPERATORS):
        for operator in group:
            for window in WINDOWS:
                actions.append(ActionSpec.from_operator(operator, window))

    return tuple(actions)


ACTIONS = _build_actions()
TOTAL_ACTIONS = len(ACTIONS)
ID_TO_ACTION = MappingProxyType(dict(enumerate(ACTIONS)))
ACTION_TO_ID = MappingProxyType({action.key: action_id for action_id, action in enumerate(ACTIONS)})

CATEGORY_TO_INDEX = MappingProxyType(
    {category: index for index, category in enumerate(OperatorCategory)}
)
OPERATOR_TO_INDEX = MappingProxyType(
    {symbol.name: index for index, symbol in enumerate(ALL_SYMBOLS)}
)
WINDOW_TO_INDEX = MappingProxyType(
    {window: index for index, window in enumerate((NO_WINDOW, *WINDOWS))}
)


def get_action(action_id: int) -> ActionSpec:
    """按动作 ID 获取动作，并拒绝负索引等 Python 序列隐式行为。"""

    if not isinstance(action_id, Integral) or isinstance(action_id, bool):
        raise TypeError("action_id 必须是整数")
    action_id = int(action_id)
    try:
        return ID_TO_ACTION[action_id]
    except KeyError as exc:
        raise IndexError(f"action_id 超出范围：{action_id}") from exc


def get_action_id(name: str, window: int = NO_WINDOW) -> int:
    """按节点名称和窗口获取唯一动作 ID。"""

    operator = get_operator(name)
    action = ActionSpec.from_operator(operator, window)
    return ACTION_TO_ID[action.key]


def get_token_indices(action_id: int) -> tuple[int, int, int]:
    """返回后续 Embedding 使用的 ``(类别, 名称, 窗口)`` 三个索引。"""

    action = get_action(action_id)
    return (
        CATEGORY_TO_INDEX[action.category],
        OPERATOR_TO_INDEX[action.name],
        WINDOW_TO_INDEX[action.window],
    )


def action_space_manifest() -> tuple[tuple[int, str, str, int, int], ...]:
    """返回按 action ID 排列、可持久化检查的动作空间清单。"""

    return tuple(
        (
            action_id,
            action.category.value,
            action.name,
            action.window,
            action.arity,
        )
        for action_id, action in enumerate(ACTIONS)
    )


def action_space_fingerprint() -> str:
    """返回锁定动作数量、内容和顺序的稳定 SHA-256 指纹。"""

    serialized = json.dumps(
        action_space_manifest(),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    payload = f"{ACTION_SPACE_SCHEMA}\0{serialized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_action_space() -> None:
    if WINDOWS != (5, 10, 20, 40, 60):
        raise RuntimeError(f"日频窗口口径被意外改变：{WINDOWS}")
    if TOTAL_ACTIONS != 142:
        raise RuntimeError(f"表达式动作数量应为 142，实际为 {TOTAL_ACTIONS}")
    if len(ACTION_TO_ID) != TOTAL_ACTIONS:
        raise RuntimeError("表达式动作必须全局唯一")


_validate_action_space()


__all__ = [
    "ACTIONS",
    "ACTION_TO_ID",
    "ACTION_SPACE_SCHEMA",
    "CATEGORY_TO_INDEX",
    "ID_TO_ACTION",
    "NO_WINDOW",
    "OPERATOR_TO_INDEX",
    "TOTAL_ACTIONS",
    "WINDOWS",
    "WINDOW_TO_INDEX",
    "ActionSpec",
    "action_space_fingerprint",
    "action_space_manifest",
    "get_action",
    "get_action_id",
    "get_token_indices",
]
