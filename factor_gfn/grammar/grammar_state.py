"""规范化部分 AST 的多路径 DAG 状态机。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .operators import NON_LEAF_OPERATORS
from .config import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NODES,
    DEFAULT_SEARCH_SPACE,
    SearchSpaceConfig,
)
from .partial_ast import (
    PARTIAL_AST_SCHEMA,
    PartialNode,
    SlotPath,
    canonical_json,
    canonicalize_partial,
    fill_hole,
    open_hole_paths,
    partial_stats,
    removable_frontier_paths,
    remove_frontier,
    targeted_slot_key,
    to_expression,
)
from .tokens import ActionRegistry, RAW_ACTION_REGISTRY, action_space_fingerprint


STATE_SPACE_SCHEMA = "factor_gfn.dag_state.v1"
TRANSITION_SPACE_SCHEMA = "factor_gfn.dag_transition.v1"


@dataclass(frozen=True, slots=True, order=True)
class OpenSlot:
    """规范部分 AST 中一个开放槽位轨道的代表。"""

    path: SlotPath
    depth: int
    orbit_key: str


@dataclass(frozen=True, slots=True)
class DAGAction:
    """完整前向转移：在一个开放槽位轨道中填入一个 Token。"""

    slot_path: SlotPath
    token_id: int
    action_registry: ActionRegistry = RAW_ACTION_REGISTRY

    def __post_init__(self) -> None:
        path = tuple(self.slot_path)
        if any(not isinstance(index, Integral) or isinstance(index, bool) or int(index) < 0 for index in path):
            raise TypeError("slot_path 必须由非负整数组成")
        if not isinstance(self.token_id, Integral) or isinstance(self.token_id, bool):
            raise TypeError("token_id 必须是整数")
        token_id = int(self.token_id)
        if not isinstance(self.action_registry, ActionRegistry):
            raise TypeError("action_registry 必须是 ActionRegistry")
        self.action_registry.get_action(token_id)
        object.__setattr__(self, "slot_path", tuple(int(index) for index in path))
        object.__setattr__(self, "token_id", token_id)


@dataclass(frozen=True, slots=True)
class GrammarSnapshot:
    state_key: str
    open_slots: tuple[OpenSlot, ...]
    hole_count: int
    node_count: int
    operator_count: int
    max_depth_seen: int
    max_depth: int
    max_nodes: int
    done: bool

    @property
    def pending_slots(self) -> int:
        return self.hole_count


@dataclass(frozen=True, slots=True)
class ParentTransition:
    parent: "GrammarState"
    forward_action: DAGAction


class GrammarState:
    """不可变的规范化部分 AST 状态。

    状态不保存生成历史。同一个规范部分 AST 无论通过何种槽位填充顺序到达，均
    使用相同的 ``state_key``。完整动作由 ``(slot_path, token_id)`` 共同定义。
    """

    __slots__ = (
        "_root",
        "search_space",
        "action_registry",
        "_stats",
        "_state_key",
    )

    def __init__(
        self,
        *,
        search_space: SearchSpaceConfig | None = None,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        action_registry: ActionRegistry = RAW_ACTION_REGISTRY,
        _root: PartialNode | None = None,
    ) -> None:
        if not isinstance(action_registry, ActionRegistry):
            raise TypeError("action_registry 必须是 ActionRegistry")
        self.action_registry = action_registry
        if search_space is not None:
            if not isinstance(search_space, SearchSpaceConfig):
                raise TypeError("search_space 必须是 SearchSpaceConfig")
            if max_depth is not None or max_nodes is not None:
                raise ValueError("search_space 不能与 max_depth/max_nodes 同时传入")
            self.search_space = search_space
        else:
            self.search_space = SearchSpaceConfig(
                max_depth=DEFAULT_MAX_DEPTH if max_depth is None else max_depth,
                max_nodes=DEFAULT_MAX_NODES if max_nodes is None else max_nodes,
            )
        if _root is None:
            _root = PartialNode(action_registry=action_registry)
        if not isinstance(_root, PartialNode):
            raise TypeError("_root 必须是 PartialNode")
        if _root.action_registry != action_registry:
            raise ValueError("_root 与 GrammarState 不得跨 ActionRegistry 混用")
        self._root = canonicalize_partial(_root)
        self._stats = partial_stats(self._root)
        if self._stats.node_count > self.max_nodes:
            raise ValueError("部分 AST 已超过 max_nodes")
        if self._stats.max_depth_seen > self.max_depth:
            raise ValueError("部分 AST 已超过 max_depth")
        self._state_key = canonical_json(self._root)

    @property
    def root(self) -> PartialNode:
        return self._root

    @property
    def max_depth(self) -> int:
        return self.search_space.max_depth

    @property
    def max_nodes(self) -> int:
        return self.search_space.max_nodes

    @property
    def state_key(self) -> str:
        return self._state_key

    @property
    def done(self) -> bool:
        return self._stats.hole_count == 0

    @property
    def node_count(self) -> int:
        return self._stats.node_count

    @property
    def operator_count(self) -> int:
        return self._stats.operator_count

    @property
    def pending_slots(self) -> int:
        return self._stats.hole_count

    @property
    def max_depth_seen(self) -> int:
        return self._stats.max_depth_seen

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GrammarState):
            return NotImplemented
        return (
            self.search_space == other.search_space
            and self.action_registry == other.action_registry
            and self.state_key == other.state_key
        )

    def __hash__(self) -> int:
        return hash((self.search_space, self.action_registry, self.state_key))

    def snapshot(self) -> GrammarSnapshot:
        return GrammarSnapshot(
            state_key=self.state_key,
            open_slots=self.open_slots(),
            hole_count=self.pending_slots,
            node_count=self.node_count,
            operator_count=self.operator_count,
            max_depth_seen=self.max_depth_seen,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            done=self.done,
        )

    def open_slots(self) -> tuple[OpenSlot, ...]:
        """返回对称槽位去重后的规范代表。"""

        if self.done:
            return ()
        representatives: dict[str, OpenSlot] = {}
        for path in open_hole_paths(self._root):
            orbit_key = targeted_slot_key(self._root, path)
            candidate = OpenSlot(path=path, depth=len(path), orbit_key=orbit_key)
            current = representatives.get(orbit_key)
            if current is None or candidate.path < current.path:
                representatives[orbit_key] = candidate
        return tuple(sorted(representatives.values(), key=lambda slot: slot.path))

    def _resolve_slot(self, path: SlotPath) -> OpenSlot:
        normalized = tuple(path)
        for slot in self.open_slots():
            if slot.path == normalized:
                return slot
        raise ValueError(f"路径不是当前状态的规范开放槽位代表：{normalized}")

    def get_legal_token_mask(self, slot: OpenSlot | SlotPath) -> NDArray[np.bool_]:
        """返回给定开放槽位上与所属 registry 等宽的 Token 合法掩码。"""

        mask = np.zeros(self.action_registry.action_count, dtype=np.bool_)
        if self.done:
            return mask
        resolved = self._resolve_slot(slot.path if isinstance(slot, OpenSlot) else slot)
        for token_id in range(self.action_registry.action_count):
            action = self.action_registry.get_action(token_id)
            if action.arity > 0 and resolved.depth >= self.max_depth:
                continue
            remaining_holes = self.pending_slots - 1 + action.arity
            minimum_final_nodes = self.node_count + 1 + remaining_holes
            if minimum_final_nodes > self.max_nodes:
                continue
            mask[token_id] = True
        if not mask.any():
            raise RuntimeError("开放槽位不存在任何可行 Token，状态无法合法收尾")
        return mask

    def legal_token_ids(self, slot: OpenSlot | SlotPath) -> NDArray[np.int64]:
        return np.flatnonzero(self.get_legal_token_mask(slot))

    def legal_transitions(self) -> tuple[DAGAction, ...]:
        if self.done:
            return ()
        transitions: list[DAGAction] = []
        successor_keys: set[str] = set()
        for slot in self.open_slots():
            for token_id in self.legal_token_ids(slot):
                action = DAGAction(slot.path, int(token_id), self.action_registry)
                successor = self._step_unchecked(action)
                if successor.state_key in successor_keys:
                    continue
                successor_keys.add(successor.state_key)
                transitions.append(action)
        if not transitions:
            raise RuntimeError("非终止状态不存在合法转移")
        return tuple(transitions)

    def _step_unchecked(self, action: DAGAction) -> "GrammarState":
        if action.action_registry != self.action_registry:
            raise ValueError("DAGAction 与 GrammarState 不得跨 ActionRegistry 混用")
        root = fill_hole(self._root, action.slot_path, action.token_id)
        return GrammarState(
            search_space=self.search_space,
            action_registry=self.action_registry,
            _root=root,
        )

    def step(self, action: DAGAction) -> "GrammarState":
        """执行一个联合动作并返回新的不可变状态。"""

        if self.done:
            raise RuntimeError("终止状态不能继续执行动作")
        if not isinstance(action, DAGAction):
            raise TypeError("step 需要 DAGAction(slot_path, token_id)")
        if action.action_registry != self.action_registry:
            raise ValueError("DAGAction 与 GrammarState 不得跨 ActionRegistry 混用")
        slot = self._resolve_slot(action.slot_path)
        if not bool(self.get_legal_token_mask(slot)[action.token_id]):
            token = self.action_registry.get_action(action.token_id)
            raise ValueError(
                f"Token 在槽位上不合法：path={slot.path}, name={token.name}, "
                f"window={token.window}, nodes={self.node_count}, holes={self.pending_slots}"
            )
        return self._step_unchecked(action)

    def to_expression(self):
        return to_expression(self._root)

    def enumerate_parents(self) -> tuple[ParentTransition, ...]:
        """枚举所有不同父状态及其唯一规范前向边。"""

        parents: dict[str, ParentTransition] = {}
        for path in removable_frontier_paths(self._root):
            parent_root = remove_frontier(self._root, path)
            parent = GrammarState(
                search_space=self.search_space,
                action_registry=self.action_registry,
                _root=parent_root,
            )
            if parent.state_key in parents:
                continue
            matching: list[DAGAction] = []
            for action in parent.legal_transitions():
                if parent._step_unchecked(action).state_key == self.state_key:
                    matching.append(action)
            if len(matching) != 1:
                raise RuntimeError(
                    "父状态必须恰有一条规范前向边到达子状态："
                    f"parent={parent.state_key}, matches={len(matching)}"
                )
            parents[parent.state_key] = ParentTransition(parent, matching[0])
        return tuple(parents[key] for key in sorted(parents))

    def count_parents(self) -> int:
        return len(self.enumerate_parents())

    def log_backward_probability(self) -> float:
        n_parents = self.count_parents()
        if n_parents < 1:
            raise ValueError("源状态没有后向概率")
        return -math.log(n_parents)

    def auxiliary_features(self) -> NDArray[np.float32]:
        """返回深度、算子预算和节点预算三个研报手工状态特征。"""

        depth_denominator = max(self.max_depth, 1)
        open_depths = tuple(len(path) for path in open_hole_paths(self._root))
        frontier_depth = max(open_depths, default=0)
        current_depth = max(self.max_depth_seen, frontier_depth)
        return np.asarray(
            (
                current_depth / depth_denominator,
                self.operator_count / self.max_nodes,
                self.node_count / self.max_nodes,
            ),
            dtype=np.float32,
        )


def _schema_digest(schema: str, manifest: object) -> str:
    serialized = json.dumps(manifest, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(f"{schema}\0{serialized}".encode("utf-8")).hexdigest()


def state_space_manifest() -> dict[str, object]:
    commutative = sorted(
        symbol.name for symbol in NON_LEAF_OPERATORS if symbol.commutative
    )
    return {
        "partial_ast_schema": PARTIAL_AST_SCHEMA,
        "hole_semantics": "one Expr slot",
        "ordered_children": True,
        "commutative_operators": commutative,
        "state_history_invariant": True,
        "parallel_edges": False,
    }


def state_space_fingerprint() -> str:
    return _schema_digest(STATE_SPACE_SCHEMA, state_space_manifest())


def transition_space_manifest(
    action_registry: ActionRegistry = RAW_ACTION_REGISTRY,
) -> dict[str, object]:
    return {
        "token_space_fingerprint": action_space_fingerprint(action_registry),
        "forward_action": ["open_slot_orbit", "token_id"],
        "slot_symmetry": "targeted partial AST canonical orbit",
        "backward_policy": "uniform over distinct parent states",
        "parent_enumeration": "remove frontier, canonicalize, deduplicate",
    }


def transition_space_fingerprint(
    action_registry: ActionRegistry = RAW_ACTION_REGISTRY,
) -> str:
    return _schema_digest(
        TRANSITION_SPACE_SCHEMA,
        transition_space_manifest(action_registry),
    )


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NODES",
    "DEFAULT_SEARCH_SPACE",
    "STATE_SPACE_SCHEMA",
    "TRANSITION_SPACE_SCHEMA",
    "DAGAction",
    "GrammarSnapshot",
    "GrammarState",
    "OpenSlot",
    "ParentTransition",
    "SearchSpaceConfig",
    "state_space_fingerprint",
    "state_space_manifest",
    "transition_space_fingerprint",
    "transition_space_manifest",
]
