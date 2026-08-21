"""把规范部分 AST 转换为路径条件化策略网络的批量张量。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from factor_gfn.grammar import (
    RAW_ACTION_REGISTRY,
    TOTAL_ACTIONS,
    ActionRegistry,
    ExactNodeGrammarState,
    GrammarState,
    OpenSlot,
    PartialNode,
    SearchSpaceConfig,
)

PolicyGrammarState = GrammarState | ExactNodeGrammarState

HOLE_TOKEN_ID = TOTAL_ACTIONS
PAD_TOKEN_ID = TOTAL_ACTIONS + 1
MODEL_TOKEN_COUNT = TOTAL_ACTIONS + 2

ROLE_PAD = 0
ROLE_ROOT = 1
ROLE_ARG0 = 2
ROLE_ARG1 = 3
ROLE_COMMUTATIVE_CHILD = 4
ROLE_COUNT = 5


@dataclass(frozen=True, slots=True)
class _NodeRecord:
    path: tuple[int, ...]
    token_id: int
    depth: int
    role_id: int
    parent_token_id: int
    path_parent_tokens: tuple[int, ...]
    path_roles: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StateBatch:
    states: tuple[PolicyGrammarState, ...]
    open_slots: tuple[tuple[OpenSlot, ...], ...]
    token_ids: Tensor
    depths: Tensor
    role_ids: Tensor
    node_mask: Tensor
    path_parent_token_ids: Tensor
    path_role_ids: Tensor
    path_mask: Tensor
    slot_node_indices: Tensor
    slot_depths: Tensor
    slot_role_ids: Tensor
    slot_parent_token_ids: Tensor
    slot_budget_features: Tensor
    slot_mask: Tensor
    legal_token_mask: Tensor
    auxiliary_features: Tensor
    condition_features: Tensor
    action_space_fingerprint: str

    def to(self, device: torch.device | str) -> "StateBatch":
        updates = {
            field: getattr(self, field).to(device)
            for field in (
                "token_ids",
                "depths",
                "role_ids",
                "node_mask",
                "path_parent_token_ids",
                "path_role_ids",
                "path_mask",
                "slot_node_indices",
                "slot_depths",
                "slot_role_ids",
                "slot_parent_token_ids",
                "slot_budget_features",
                "slot_mask",
                "legal_token_mask",
                "auxiliary_features",
                "condition_features",
            )
        }
        return replace(self, **updates)


def _child_role(
    parent_token_id: int,
    child_index: int,
    action_registry: ActionRegistry,
) -> int:
    action = action_registry.get_action(parent_token_id)
    if action_registry.get_operator(action.name).commutative:
        return ROLE_COMMUTATIVE_CHILD
    return ROLE_ARG0 if child_index == 0 else ROLE_ARG1


def _node_records(
    root: PartialNode,
    action_registry: ActionRegistry,
    hole_token_id: int,
    pad_token_id: int,
) -> tuple[_NodeRecord, ...]:
    records: list[_NodeRecord] = []

    def visit(
        node: PartialNode,
        path: tuple[int, ...],
        role_id: int,
        parent_token_id: int,
        path_parent_tokens: tuple[int, ...],
        path_roles: tuple[int, ...],
    ) -> None:
        token_id = hole_token_id if node.is_hole else int(node.action_id)
        records.append(
            _NodeRecord(
                path=path,
                token_id=token_id,
                depth=len(path),
                role_id=role_id,
                parent_token_id=parent_token_id,
                path_parent_tokens=path_parent_tokens,
                path_roles=path_roles,
            )
        )
        if node.is_hole:
            return
        assert node.action_id is not None
        for index, child in enumerate(node.children):
            child_role = _child_role(node.action_id, index, action_registry)
            visit(
                child,
                path + (index,),
                child_role,
                node.action_id,
                path_parent_tokens + (node.action_id,),
                path_roles + (child_role,),
            )

    visit(root, (), ROLE_ROOT, pad_token_id, (), ())
    return tuple(records)


class StateAdapter:
    """以阶段二规范状态为唯一真值来源构造模型输入。"""

    def __init__(
        self,
        search_space: SearchSpaceConfig = SearchSpaceConfig(),
        action_registry: ActionRegistry = RAW_ACTION_REGISTRY,
    ) -> None:
        self.search_space = search_space
        if not isinstance(action_registry, ActionRegistry):
            raise TypeError("action_registry 必须是 ActionRegistry")
        self.action_registry = action_registry

    @property
    def action_count(self) -> int:
        return self.action_registry.action_count

    @property
    def hole_token_id(self) -> int:
        return self.action_count

    @property
    def pad_token_id(self) -> int:
        return self.action_count + 1

    @property
    def model_token_count(self) -> int:
        return self.action_count + 2

    def _validate_state(self, state: PolicyGrammarState) -> None:
        if state.done:
            raise ValueError("终止状态没有前向动作，不能送入策略网络")
        if state.search_space != self.search_space:
            raise ValueError("GrammarState 与 StateAdapter 的 SearchSpaceConfig 必须等价")
        structural = state.state if isinstance(state, ExactNodeGrammarState) else state
        if structural.action_registry != self.action_registry:
            raise ValueError("GrammarState 与 StateAdapter 不得跨 ActionRegistry 混用")

    def batch(
        self,
        states: list[PolicyGrammarState] | tuple[PolicyGrammarState, ...],
    ) -> StateBatch:
        states = tuple(states)
        if not states:
            raise ValueError("states 不能为空")
        for state in states:
            self._validate_state(state)

        structural_states = tuple(
            state.state if isinstance(state, ExactNodeGrammarState) else state
            for state in states
        )
        all_records = tuple(
            _node_records(
                state.root,
                self.action_registry,
                self.hole_token_id,
                self.pad_token_id,
            )
            for state in structural_states
        )
        all_slots = tuple(state.open_slots() for state in states)
        batch_size = len(states)
        max_nodes_in_batch = max(len(records) for records in all_records)
        max_slots = max(len(slots) for slots in all_slots)
        max_depth = self.search_space.max_depth

        token_ids = torch.full(
            (batch_size, max_nodes_in_batch), self.pad_token_id, dtype=torch.long
        )
        depths = torch.zeros((batch_size, max_nodes_in_batch), dtype=torch.long)
        role_ids = torch.full((batch_size, max_nodes_in_batch), ROLE_PAD, dtype=torch.long)
        node_mask = torch.zeros((batch_size, max_nodes_in_batch), dtype=torch.bool)
        path_parent_token_ids = torch.full(
            (batch_size, max_nodes_in_batch, max_depth),
            self.pad_token_id,
            dtype=torch.long,
        )
        path_role_ids = torch.full(
            (batch_size, max_nodes_in_batch, max_depth), ROLE_PAD, dtype=torch.long
        )
        path_mask = torch.zeros((batch_size, max_nodes_in_batch, max_depth), dtype=torch.bool)

        slot_node_indices = torch.zeros((batch_size, max_slots), dtype=torch.long)
        slot_depths = torch.zeros((batch_size, max_slots), dtype=torch.long)
        slot_role_ids = torch.full((batch_size, max_slots), ROLE_PAD, dtype=torch.long)
        slot_parent_token_ids = torch.full(
            (batch_size, max_slots), self.pad_token_id, dtype=torch.long
        )
        slot_budget_features = torch.zeros((batch_size, max_slots, 3), dtype=torch.float32)
        slot_mask = torch.zeros((batch_size, max_slots), dtype=torch.bool)
        legal_token_mask = torch.zeros(
            (batch_size, max_slots, self.action_count), dtype=torch.bool
        )
        auxiliary_features = torch.zeros((batch_size, 3), dtype=torch.float32)
        condition_features = torch.zeros((batch_size, 2), dtype=torch.float32)

        for batch_index, (state, structural_state, records, slots) in enumerate(
            zip(states, structural_states, all_records, all_slots, strict=True)
        ):
            path_to_node = {record.path: index for index, record in enumerate(records)}
            for node_index, record in enumerate(records):
                token_ids[batch_index, node_index] = record.token_id
                depths[batch_index, node_index] = record.depth
                role_ids[batch_index, node_index] = record.role_id
                node_mask[batch_index, node_index] = True
                path_length = len(record.path_roles)
                if path_length:
                    path_parent_token_ids[batch_index, node_index, :path_length] = torch.tensor(
                        record.path_parent_tokens, dtype=torch.long
                    )
                    path_role_ids[batch_index, node_index, :path_length] = torch.tensor(
                        record.path_roles, dtype=torch.long
                    )
                    path_mask[batch_index, node_index, :path_length] = True

            canonical_edges = {
                (transition.slot_path, transition.token_id)
                for transition in state.legal_transitions()
            }
            for slot_index, slot in enumerate(slots):
                node_index = path_to_node[slot.path]
                record = records[node_index]
                token_mask = torch.tensor(
                    [
                        (slot.path, token_id) in canonical_edges
                        for token_id in range(self.action_count)
                    ],
                    dtype=torch.bool,
                )
                if not bool(token_mask.any()):
                    continue
                slot_node_indices[batch_index, slot_index] = node_index
                slot_depths[batch_index, slot_index] = slot.depth
                slot_role_ids[batch_index, slot_index] = record.role_id
                slot_parent_token_ids[batch_index, slot_index] = record.parent_token_id
                slot_mask[batch_index, slot_index] = True
                legal_token_mask[batch_index, slot_index] = token_mask
                remaining_nodes = (
                    self.search_space.max_nodes - state.node_count
                ) / self.search_space.max_nodes
                depth_denominator = max(self.search_space.max_depth, 1)
                remaining_depth = (
                    self.search_space.max_depth - slot.depth
                ) / depth_denominator
                holes = state.pending_slots / (self.search_space.max_nodes + 1)
                slot_budget_features[batch_index, slot_index] = torch.tensor(
                    (remaining_nodes, remaining_depth, holes), dtype=torch.float32
                )

            auxiliary_features[batch_index] = torch.from_numpy(
                structural_state.auxiliary_features()
            )
            if isinstance(state, ExactNodeGrammarState):
                condition_features[batch_index] = torch.tensor(
                    (
                        state.target_node_count / self.search_space.max_nodes,
                        (state.target_node_count - state.node_count)
                        / self.search_space.max_nodes,
                    ),
                    dtype=torch.float32,
                )

        if not bool(slot_mask.any(dim=1).all()):
            raise RuntimeError("非终止状态必须至少存在一条规范前向边")

        return StateBatch(
            states=states,
            open_slots=all_slots,
            token_ids=token_ids,
            depths=depths,
            role_ids=role_ids,
            node_mask=node_mask,
            path_parent_token_ids=path_parent_token_ids,
            path_role_ids=path_role_ids,
            path_mask=path_mask,
            slot_node_indices=slot_node_indices,
            slot_depths=slot_depths,
            slot_role_ids=slot_role_ids,
            slot_parent_token_ids=slot_parent_token_ids,
            slot_budget_features=slot_budget_features,
            slot_mask=slot_mask,
            legal_token_mask=legal_token_mask,
            auxiliary_features=auxiliary_features,
            condition_features=condition_features,
            action_space_fingerprint=self.action_registry.fingerprint(),
        )


__all__ = [
    "HOLE_TOKEN_ID",
    "MODEL_TOKEN_COUNT",
    "PAD_TOKEN_ID",
    "ROLE_ARG0",
    "ROLE_ARG1",
    "ROLE_COMMUTATIVE_CHILD",
    "ROLE_COUNT",
    "ROLE_PAD",
    "ROLE_ROOT",
    "StateAdapter",
    "StateBatch",
    "PolicyGrammarState",
]
