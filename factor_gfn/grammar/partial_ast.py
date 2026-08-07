"""规范化部分表达式树。

GFlowNet 的生成状态由带显式 Hole 的部分 AST 表示。二元节点始终保留语义参数
位置；只有注册表中明确标记为交换律的算子会对子树做规范排序。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral

from .expression import Expression, ExpressionNode
from .operators import get_operator
from .tokens import get_action


PARTIAL_AST_SCHEMA = "factor_gfn.partial_ast.v1"
SlotPath = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PartialNode:
    """部分 AST 节点；``action_id=None`` 表示一个待填表达式槽位。"""

    action_id: int | None = None
    children: tuple["PartialNode", ...] = ()

    def __post_init__(self) -> None:
        children = tuple(self.children)
        if any(not isinstance(child, PartialNode) for child in children):
            raise TypeError("部分 AST 的 children 必须全部是 PartialNode")
        if self.action_id is None:
            if children:
                raise ValueError("Hole 节点不能包含子节点")
            object.__setattr__(self, "children", ())
            return
        if not isinstance(self.action_id, Integral) or isinstance(self.action_id, bool):
            raise TypeError("部分 AST 的 action_id 必须是整数或 None")
        action_id = int(self.action_id)
        action = get_action(action_id)
        if len(children) != action.arity:
            raise ValueError(
                f"{action.name} 需要 {action.arity} 个子节点，实际为 {len(children)}"
            )
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "children", children)

    @property
    def is_hole(self) -> bool:
        return self.action_id is None


HOLE = PartialNode()


@dataclass(frozen=True, slots=True)
class PartialStats:
    node_count: int
    operator_count: int
    hole_count: int
    max_depth_seen: int


def _payload(node: PartialNode, target_path: SlotPath | None = None, path: SlotPath = ()) -> list[object]:
    if node.is_hole:
        return ["target" if target_path == path else "hole"]

    assert node.action_id is not None
    action = get_action(node.action_id)
    children = [
        _payload(child, target_path, path + (index,))
        for index, child in enumerate(node.children)
    ]
    if get_operator(action.name).commutative:
        children.sort(key=_json)
    return ["node", action.name, action.window, children]


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def canonical_json(node: PartialNode) -> str:
    """返回不依赖 action ID 的规范部分树 JSON。"""

    return _json(_payload(node))


def targeted_slot_key(node: PartialNode, path: SlotPath) -> str:
    """返回用于识别对称开放槽位轨道的稳定键。"""

    target = get_node(node, path)
    if not target.is_hole:
        raise ValueError(f"目标路径不是开放槽位：{path}")
    return _json(_payload(node, target_path=path))


def canonicalize_partial(node: PartialNode) -> PartialNode:
    if node.is_hole:
        return HOLE
    assert node.action_id is not None
    children = tuple(canonicalize_partial(child) for child in node.children)
    action = get_action(node.action_id)
    if get_operator(action.name).commutative:
        children = tuple(sorted(children, key=canonical_json))
    return PartialNode(node.action_id, children)


def get_node(root: PartialNode, path: SlotPath) -> PartialNode:
    node = root
    for depth, index in enumerate(path):
        if node.is_hole:
            raise ValueError(f"路径在第 {depth} 层穿过 Hole：{path}")
        if not isinstance(index, Integral) or isinstance(index, bool):
            raise TypeError("槽位路径索引必须是整数")
        index = int(index)
        if index < 0 or index >= len(node.children):
            raise ValueError(f"槽位路径越界：{path}")
        node = node.children[index]
    return node


def replace_node(root: PartialNode, path: SlotPath, replacement: PartialNode) -> PartialNode:
    if not path:
        return replacement
    if root.is_hole:
        raise ValueError(f"无法穿过 Hole 替换节点：{path}")
    index = path[0]
    if not isinstance(index, Integral) or isinstance(index, bool):
        raise TypeError("槽位路径索引必须是整数")
    index = int(index)
    if index < 0 or index >= len(root.children):
        raise ValueError(f"槽位路径越界：{path}")
    children = list(root.children)
    children[index] = replace_node(children[index], path[1:], replacement)
    return PartialNode(root.action_id, tuple(children))


def fill_hole(root: PartialNode, path: SlotPath, action_id: int) -> PartialNode:
    if not get_node(root, path).is_hole:
        raise ValueError(f"目标路径不是开放槽位：{path}")
    action = get_action(action_id)
    replacement = PartialNode(action_id, tuple(HOLE for _ in range(action.arity)))
    return canonicalize_partial(replace_node(root, path, replacement))


def remove_frontier(root: PartialNode, path: SlotPath) -> PartialNode:
    node = get_node(root, path)
    if node.is_hole:
        raise ValueError("不能撤销 Hole")
    if node.children and any(not child.is_hole for child in node.children):
        raise ValueError(f"节点不是可撤销前沿节点：{path}")
    return canonicalize_partial(replace_node(root, path, HOLE))


def open_hole_paths(root: PartialNode) -> tuple[SlotPath, ...]:
    output: list[SlotPath] = []
    stack: list[tuple[PartialNode, SlotPath]] = [(root, ())]
    while stack:
        node, path = stack.pop()
        if node.is_hole:
            output.append(path)
            continue
        for index in range(len(node.children) - 1, -1, -1):
            stack.append((node.children[index], path + (index,)))
    return tuple(output)


def removable_frontier_paths(root: PartialNode) -> tuple[SlotPath, ...]:
    output: list[SlotPath] = []
    stack: list[tuple[PartialNode, SlotPath]] = [(root, ())]
    while stack:
        node, path = stack.pop()
        if node.is_hole:
            continue
        if not node.children or all(child.is_hole for child in node.children):
            output.append(path)
        for index in range(len(node.children) - 1, -1, -1):
            stack.append((node.children[index], path + (index,)))
    return tuple(output)


def partial_stats(root: PartialNode) -> PartialStats:
    node_count = 0
    operator_count = 0
    hole_count = 0
    max_depth_seen = 0
    stack: list[tuple[PartialNode, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.is_hole:
            hole_count += 1
            continue
        node_count += 1
        max_depth_seen = max(max_depth_seen, depth)
        assert node.action_id is not None
        operator_count += int(get_action(node.action_id).arity > 0)
        stack.extend((child, depth + 1) for child in node.children)
    return PartialStats(node_count, operator_count, hole_count, max_depth_seen)


def to_expression(root: PartialNode) -> Expression:
    if open_hole_paths(root):
        raise ValueError("部分 AST 尚未完成，不能转换为 Expression")

    def convert(node: PartialNode) -> ExpressionNode:
        if node.is_hole:
            raise RuntimeError("完整性检查后仍发现 Hole")
        assert node.action_id is not None
        return ExpressionNode(node.action_id, tuple(convert(child) for child in node.children))

    return Expression(convert(root))


__all__ = [
    "HOLE",
    "PARTIAL_AST_SCHEMA",
    "PartialNode",
    "PartialStats",
    "SlotPath",
    "canonical_json",
    "canonicalize_partial",
    "fill_hole",
    "get_node",
    "open_hole_paths",
    "partial_stats",
    "removable_frontier_paths",
    "remove_frontier",
    "replace_node",
    "targeted_slot_key",
    "to_expression",
]
