"""Exact-node reachability and conditioned Grammar DAG state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .config import SearchSpaceConfig
from .grammar_state import DAGAction, GrammarSnapshot, GrammarState, OpenSlot
from .partial_ast import SlotPath, open_hole_paths
from .tokens import ACTIONS, TOTAL_ACTIONS, get_action


ConditionedStateKey = tuple[str, int, str]


def _validate_target_node_count(
    target_node_count: int,
    search_space: SearchSpaceConfig,
) -> int:
    if not isinstance(target_node_count, Integral) or isinstance(target_node_count, bool):
        raise TypeError("target_node_count must be an integer")
    target_node_count = int(target_node_count)
    if target_node_count < 1 or target_node_count > search_space.max_nodes:
        raise ValueError(
            "target_node_count must be between 1 and search_space.max_nodes"
        )
    return target_node_count


@lru_cache(maxsize=None)
def _reachable_subtree_sizes_by_depth(
    max_depth: int,
    max_nodes: int,
    arities: tuple[int, ...],
) -> tuple[frozenset[int], ...]:
    """Return exact subtree sizes reachable from a Hole at each absolute depth."""

    sizes_by_depth: list[frozenset[int]] = [frozenset()] * (max_depth + 1)
    for depth in range(max_depth, -1, -1):
        sizes: set[int] = set()
        for arity in arities:
            if arity == 0:
                sizes.add(1)
                continue
            if depth >= max_depth:
                continue
            child_sizes = sizes_by_depth[depth + 1]
            child_sums = {0}
            for _ in range(arity):
                child_sums = {
                    left + right
                    for left in child_sums
                    for right in child_sizes
                    if left + right < max_nodes
                }
            sizes.update(
                1 + child_sum
                for child_sum in child_sums
                if 1 + child_sum <= max_nodes
            )
        sizes_by_depth[depth] = frozenset(sizes)
    return tuple(sizes_by_depth)


@dataclass(frozen=True, slots=True)
class ExactNodeStrata:
    """Resolved exact-node strata for one runtime search space."""

    resolved_feasible_node_counts: tuple[int, ...]
    resolved_infeasible_node_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExactNodeReachability:
    """Exact completion engine for a fixed runtime search space."""

    search_space: SearchSpaceConfig

    def __post_init__(self) -> None:
        if not isinstance(self.search_space, SearchSpaceConfig):
            raise TypeError("search_space must be SearchSpaceConfig")

    @property
    def _arities(self) -> tuple[int, ...]:
        return tuple(sorted({action.arity for action in ACTIONS}))

    def _validate_state(self, state: GrammarState) -> None:
        if not isinstance(state, GrammarState):
            raise TypeError("state must be GrammarState")
        if state.search_space != self.search_space:
            raise ValueError("state and reachability engine must share search_space")

    def reachable_terminal_node_counts(self, state: GrammarState) -> tuple[int, ...]:
        """Return all exact terminal sizes reachable from ``state``."""

        self._validate_state(state)
        remaining_budget = self.search_space.max_nodes - state.node_count
        completion_sizes = {0}
        subtree_sizes = _reachable_subtree_sizes_by_depth(
            self.search_space.max_depth,
            self.search_space.max_nodes,
            self._arities,
        )
        for path in open_hole_paths(state.root):
            depth = len(path)
            if depth > self.search_space.max_depth:
                return ()
            completion_sizes = {
                current + subtree_size
                for current in completion_sizes
                for subtree_size in subtree_sizes[depth]
                if current + subtree_size <= remaining_budget
            }
            if not completion_sizes:
                return ()
        return tuple(sorted(state.node_count + size for size in completion_sizes))

    def can_complete_exactly(
        self,
        state: GrammarState,
        target_node_count: int,
    ) -> bool:
        target_node_count = _validate_target_node_count(
            target_node_count, self.search_space
        )
        return target_node_count in self.reachable_terminal_node_counts(state)

    def conditioned_key(
        self,
        state: GrammarState,
        target_node_count: int,
    ) -> ConditionedStateKey:
        self._validate_state(state)
        target_node_count = _validate_target_node_count(
            target_node_count, self.search_space
        )
        return (
            state.state_key,
            target_node_count,
            self.search_space.fingerprint(),
        )

    def legal_token_mask(
        self,
        state: GrammarState,
        slot: OpenSlot | SlotPath,
        target_node_count: int,
    ) -> NDArray[np.bool_]:
        """Filter the legacy mask to actions with an exact-N completion."""

        self._validate_state(state)
        target_node_count = _validate_target_node_count(
            target_node_count, self.search_space
        )
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.bool_)
        if state.done:
            return mask
        legacy_mask = state.get_legal_token_mask(slot)
        slot_path = slot.path if isinstance(slot, OpenSlot) else tuple(slot)
        arity_is_feasible: dict[int, bool] = {}
        for token_id in np.flatnonzero(legacy_mask):
            token_id = int(token_id)
            arity = get_action(token_id).arity
            feasible = arity_is_feasible.get(arity)
            if feasible is None:
                successor = state._step_unchecked(DAGAction(slot_path, token_id))
                feasible = self.can_complete_exactly(successor, target_node_count)
                arity_is_feasible[arity] = feasible
            mask[token_id] = feasible
        return mask

    def resolve_strata(self) -> ExactNodeStrata:
        source = GrammarState(search_space=self.search_space)
        reachable = set(self.reachable_terminal_node_counts(source))
        candidates = range(1, self.search_space.max_nodes + 1)
        return ExactNodeStrata(
            resolved_feasible_node_counts=tuple(
                node_count for node_count in candidates if node_count in reachable
            ),
            resolved_infeasible_node_counts=tuple(
                node_count for node_count in candidates if node_count not in reachable
            ),
        )


@dataclass(frozen=True, slots=True)
class ExactNodeParentTransition:
    parent: "ExactNodeGrammarState"
    forward_action: DAGAction


@dataclass(frozen=True, slots=True)
class ExactNodeGrammarState:
    """A structural GrammarState paired with one external exact-node condition."""

    state: GrammarState
    target_node_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, GrammarState):
            raise TypeError("state must be GrammarState")
        target_node_count = _validate_target_node_count(
            self.target_node_count, self.state.search_space
        )
        object.__setattr__(self, "target_node_count", target_node_count)
        if not self.reachability.can_complete_exactly(self.state, target_node_count):
            raise ValueError(
                "state has no legal completion for target_node_count="
                f"{target_node_count}"
            )

    @classmethod
    def source(
        cls,
        *,
        target_node_count: int,
        search_space: SearchSpaceConfig,
    ) -> "ExactNodeGrammarState":
        return cls(
            GrammarState(search_space=search_space),
            target_node_count=target_node_count,
        )

    @property
    def reachability(self) -> ExactNodeReachability:
        return ExactNodeReachability(self.state.search_space)

    @property
    def search_space(self) -> SearchSpaceConfig:
        return self.state.search_space

    @property
    def state_key(self) -> str:
        return self.state.state_key

    @property
    def conditioned_key(self) -> ConditionedStateKey:
        return self.reachability.conditioned_key(self.state, self.target_node_count)

    @property
    def conditioned_cache_key(self) -> ConditionedStateKey:
        """Stable identity for any cache whose value depends on the condition."""

        return self.conditioned_key

    @property
    def done(self) -> bool:
        return self.state.done

    @property
    def node_count(self) -> int:
        return self.state.node_count

    @property
    def pending_slots(self) -> int:
        return self.state.pending_slots

    @property
    def max_depth_seen(self) -> int:
        return self.state.max_depth_seen

    def snapshot(self) -> GrammarSnapshot:
        return self.state.snapshot()

    def open_slots(self) -> tuple[OpenSlot, ...]:
        return self.state.open_slots()

    def get_legal_token_mask(self, slot: OpenSlot | SlotPath) -> NDArray[np.bool_]:
        return self.reachability.legal_token_mask(
            self.state, slot, self.target_node_count
        )

    def legal_token_ids(self, slot: OpenSlot | SlotPath) -> NDArray[np.int64]:
        return np.flatnonzero(self.get_legal_token_mask(slot))

    def legal_transitions(self) -> tuple[DAGAction, ...]:
        if self.done:
            return ()
        transitions: list[DAGAction] = []
        successor_keys: set[str] = set()
        for slot in self.open_slots():
            for token_id in self.legal_token_ids(slot):
                action = DAGAction(slot.path, int(token_id))
                successor = self.state._step_unchecked(action)
                if successor.state_key in successor_keys:
                    continue
                successor_keys.add(successor.state_key)
                transitions.append(action)
        if not transitions:
            raise RuntimeError("non-terminal exact-node state has no legal transition")
        return tuple(transitions)

    def step(self, action: DAGAction) -> "ExactNodeGrammarState":
        if self.done:
            raise RuntimeError("terminal exact-node state cannot take another action")
        if not isinstance(action, DAGAction):
            raise TypeError("step requires DAGAction(slot_path, token_id)")
        slot = self.state._resolve_slot(action.slot_path)
        if not bool(self.get_legal_token_mask(slot)[action.token_id]):
            raise ValueError(
                "action has no exact completion for target_node_count="
                f"{self.target_node_count}"
            )
        return ExactNodeGrammarState(
            self.state._step_unchecked(action),
            target_node_count=self.target_node_count,
        )

    def to_expression(self):
        return self.state.to_expression()

    def enumerate_parents(self) -> tuple[ExactNodeParentTransition, ...]:
        parents: dict[ConditionedStateKey, ExactNodeParentTransition] = {}
        for structural_transition in self.state.enumerate_parents():
            try:
                parent = ExactNodeGrammarState(
                    structural_transition.parent,
                    target_node_count=self.target_node_count,
                )
            except ValueError:
                continue
            matching = [
                action
                for action in parent.legal_transitions()
                if parent.state._step_unchecked(action).state_key == self.state_key
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    "conditioned parent must have exactly one canonical edge to child: "
                    f"parent={parent.conditioned_key}, matches={len(matching)}"
                )
            parents[parent.conditioned_key] = ExactNodeParentTransition(
                parent, matching[0]
            )
        return tuple(parents[key] for key in sorted(parents))

    def count_parents(self) -> int:
        return len(self.enumerate_parents())

    def log_backward_probability(self) -> float:
        n_parents = self.count_parents()
        if n_parents < 1:
            raise ValueError("source state has no backward probability")
        return -math.log(n_parents)


def resolve_exact_node_strata(search_space: SearchSpaceConfig) -> ExactNodeStrata:
    return ExactNodeReachability(search_space).resolve_strata()


__all__ = [
    "ConditionedStateKey",
    "ExactNodeGrammarState",
    "ExactNodeParentTransition",
    "ExactNodeReachability",
    "ExactNodeStrata",
    "resolve_exact_node_strata",
]
