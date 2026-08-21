"""Build stable, immutable action registries for one explicit Feature Space."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

from factor_gfn.feature_spaces import (
    DAILY_DERIVED_V1_FEATURE_SPACE,
    FeatureSpaceSpec,
    RAW_DAILY_FEATURE_SPACE,
)

from .operators import (
    BINARY_OPERATORS,
    CROSS_SECTIONAL_OPERATORS,
    NON_LEAF_OPERATORS,
    TS_BINARY_OPERATORS,
    TS_UNARY_OPERATORS,
    UNARY_OPERATORS,
    OperatorCategory,
    OperatorSpec,
    build_leaf_operators,
)


WINDOWS = (5, 10, 20, 40, 60)
NO_WINDOW = 0
ACTION_SPACE_SCHEMA = "factor_gfn.action_space.v1"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    category: OperatorCategory
    name: str
    window: int
    arity: int

    @classmethod
    def from_operator(cls, operator: OperatorSpec, window: int) -> ActionSpec:
        if operator.requires_window:
            if window not in WINDOWS:
                raise ValueError(f"时序算子 {operator.name} 的窗口必须属于 {WINDOWS}")
        elif window != NO_WINDOW:
            raise ValueError(f"无窗口节点 {operator.name} 的窗口必须为 0")
        return cls(operator.category, operator.name, window, operator.arity)

    @property
    def key(self) -> tuple[OperatorCategory, str, int, int]:
        return (self.category, self.name, self.window, self.arity)


@dataclass(frozen=True, slots=True)
class ActionRegistry:
    """Immutable action/token vocabulary for one Feature Space."""

    feature_space: FeatureSpaceSpec
    actions: tuple[ActionSpec, ...] = field(init=False)
    id_to_action: Mapping[int, ActionSpec] = field(init=False, repr=False, compare=False)
    action_to_id: Mapping[tuple[OperatorCategory, str, int, int], int] = field(
        init=False, repr=False, compare=False
    )
    category_to_index: Mapping[OperatorCategory, int] = field(
        init=False, repr=False, compare=False
    )
    operator_to_index: Mapping[str, int] = field(init=False, repr=False, compare=False)
    window_to_index: Mapping[int, int] = field(init=False, repr=False, compare=False)
    operator_by_name: Mapping[str, OperatorSpec] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.feature_space, FeatureSpaceSpec):
            raise TypeError("feature_space 必须是 FeatureSpaceSpec")
        leaves = build_leaf_operators(self.feature_space.ordered_leaf_names)
        operators = leaves + NON_LEAF_OPERATORS
        operator_by_name = {operator.name: operator for operator in operators}
        if len(operator_by_name) != len(operators):
            raise ValueError("Feature leaves 与共享 non-leaf operators 名称冲突")

        actions: list[ActionSpec] = []
        for operator in leaves:
            actions.append(ActionSpec.from_operator(operator, NO_WINDOW))
        for group in (UNARY_OPERATORS, BINARY_OPERATORS, CROSS_SECTIONAL_OPERATORS):
            for operator in group:
                actions.append(ActionSpec.from_operator(operator, NO_WINDOW))
        for group in (TS_UNARY_OPERATORS, TS_BINARY_OPERATORS):
            for operator in group:
                for window in WINDOWS:
                    actions.append(ActionSpec.from_operator(operator, window))

        action_tuple = tuple(actions)
        action_to_id = {action.key: action_id for action_id, action in enumerate(action_tuple)}
        if len(action_to_id) != len(action_tuple):
            raise RuntimeError("ActionRegistry 动作必须全局唯一")
        object.__setattr__(self, "actions", action_tuple)
        object.__setattr__(self, "id_to_action", MappingProxyType(dict(enumerate(action_tuple))))
        object.__setattr__(self, "action_to_id", MappingProxyType(action_to_id))
        object.__setattr__(
            self,
            "category_to_index",
            MappingProxyType(
                {category: index for index, category in enumerate(OperatorCategory)}
            ),
        )
        object.__setattr__(
            self,
            "operator_to_index",
            MappingProxyType(
                {operator.name: index for index, operator in enumerate(operators)}
            ),
        )
        object.__setattr__(
            self,
            "window_to_index",
            MappingProxyType(
                {window: index for index, window in enumerate((NO_WINDOW, *WINDOWS))}
            ),
        )
        object.__setattr__(self, "operator_by_name", MappingProxyType(operator_by_name))

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def leaf_names(self) -> tuple[str, ...]:
        return self.feature_space.ordered_leaf_names

    @property
    def feature_space_fingerprint(self) -> str:
        return self.feature_space.fingerprint()

    def get_action(self, action_id: int) -> ActionSpec:
        if not isinstance(action_id, Integral) or isinstance(action_id, bool):
            raise TypeError("action_id 必须是整数")
        try:
            return self.id_to_action[int(action_id)]
        except KeyError as exc:
            raise IndexError(f"action_id 超出范围：{action_id}") from exc

    def get_operator(self, name: str) -> OperatorSpec:
        try:
            return self.operator_by_name[name]
        except KeyError as exc:
            raise KeyError(
                f"Feature Space {self.feature_space.feature_space_id!r} 未登记名称：{name!r}"
            ) from exc

    def get_action_id(self, name: str, window: int = NO_WINDOW) -> int:
        action = ActionSpec.from_operator(self.get_operator(name), window)
        try:
            return self.action_to_id[action.key]
        except KeyError as exc:
            raise KeyError(f"ActionRegistry 未登记动作：{name!r}, window={window}") from exc

    def get_token_indices(self, action_id: int) -> tuple[int, int, int]:
        action = self.get_action(action_id)
        return (
            self.category_to_index[action.category],
            self.operator_to_index[action.name],
            self.window_to_index[action.window],
        )

    def manifest(self) -> tuple[tuple[int, str, str, int, int], ...]:
        return tuple(
            (action_id, action.category.value, action.name, action.window, action.arity)
            for action_id, action in enumerate(self.actions)
        )

    def fingerprint(self) -> str:
        serialized = json.dumps(
            self.manifest(), ensure_ascii=True, separators=(",", ":")
        )
        payload = f"{ACTION_SPACE_SCHEMA}\0{serialized}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def build_action_registry(feature_space: FeatureSpaceSpec) -> ActionRegistry:
    return ActionRegistry(feature_space)


RAW_ACTION_REGISTRY = build_action_registry(RAW_DAILY_FEATURE_SPACE)
DAILY_DERIVED_ACTION_REGISTRY = build_action_registry(DAILY_DERIVED_V1_FEATURE_SPACE)

# Frozen Raw compatibility surface.
ACTIONS = RAW_ACTION_REGISTRY.actions
TOTAL_ACTIONS = RAW_ACTION_REGISTRY.action_count
ID_TO_ACTION = RAW_ACTION_REGISTRY.id_to_action
ACTION_TO_ID = RAW_ACTION_REGISTRY.action_to_id
CATEGORY_TO_INDEX = RAW_ACTION_REGISTRY.category_to_index
OPERATOR_TO_INDEX = RAW_ACTION_REGISTRY.operator_to_index
WINDOW_TO_INDEX = RAW_ACTION_REGISTRY.window_to_index


def get_action(action_id: int, registry: ActionRegistry = RAW_ACTION_REGISTRY) -> ActionSpec:
    return registry.get_action(action_id)


def get_action_id(
    name: str,
    window: int = NO_WINDOW,
    registry: ActionRegistry = RAW_ACTION_REGISTRY,
) -> int:
    return registry.get_action_id(name, window)


def get_token_indices(
    action_id: int,
    registry: ActionRegistry = RAW_ACTION_REGISTRY,
) -> tuple[int, int, int]:
    return registry.get_token_indices(action_id)


def action_space_manifest(
    registry: ActionRegistry = RAW_ACTION_REGISTRY,
) -> tuple[tuple[int, str, str, int, int], ...]:
    return registry.manifest()


def action_space_fingerprint(
    registry: ActionRegistry = RAW_ACTION_REGISTRY,
) -> str:
    return registry.fingerprint()


def _validate_raw_compatibility() -> None:
    if WINDOWS != (5, 10, 20, 40, 60):
        raise RuntimeError(f"日频窗口口径被意外改变：{WINDOWS}")
    if TOTAL_ACTIONS != 142:
        raise RuntimeError(f"Raw 表达式动作数量应为 142，实际为 {TOTAL_ACTIONS}")


_validate_raw_compatibility()


__all__ = [
    "ACTIONS",
    "ACTION_TO_ID",
    "ACTION_SPACE_SCHEMA",
    "CATEGORY_TO_INDEX",
    "DAILY_DERIVED_ACTION_REGISTRY",
    "ID_TO_ACTION",
    "NO_WINDOW",
    "OPERATOR_TO_INDEX",
    "RAW_ACTION_REGISTRY",
    "TOTAL_ACTIONS",
    "WINDOWS",
    "WINDOW_TO_INDEX",
    "ActionRegistry",
    "ActionSpec",
    "action_space_fingerprint",
    "action_space_manifest",
    "build_action_registry",
    "get_action",
    "get_action_id",
    "get_token_indices",
]
