"""Deterministic balanced scheduling over exact-node discovery strata."""

from __future__ import annotations

import random
from dataclasses import dataclass
from numbers import Integral
from typing import Any


SCHEDULER_SCHEMA = "factor_gfn.balanced_node_count_scheduler.v1"


def _normalize_strata(values: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("resolved_discovery_strata must be a non-empty tuple")
    normalized: list[int] = []
    for value in values:
        if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 1:
            raise ValueError("discovery node counts must be positive integers")
        normalized.append(int(value))
    if len(set(normalized)) != len(normalized):
        raise ValueError("resolved_discovery_strata must not contain duplicates")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ConditionAssignment:
    """Immutable token identifying one pending condition update."""

    cycle_index: int
    condition_position_in_cycle: int
    condition_N: int


@dataclass
class BalancedNodeCountScheduler:
    """Consume a shuffled permutation of every stratum before reshuffling."""

    resolved_discovery_strata: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        self.resolved_discovery_strata = _normalize_strata(
            self.resolved_discovery_strata
        )
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool):
            raise ValueError("scheduler seed must be an integer")
        self.seed = int(self.seed)
        self._rng = random.Random(self.seed)
        self.current_permutation = list(self.resolved_discovery_strata)
        self._rng.shuffle(self.current_permutation)
        self.cycle_index = 0
        self.position = 0

    def _start_next_cycle(self) -> None:
        self.cycle_index += 1
        self.current_permutation = list(self.resolved_discovery_strata)
        self._rng.shuffle(self.current_permutation)
        self.position = 0

    def peek(self) -> ConditionAssignment:
        """Return the pending assignment without consuming it."""

        if self.position < len(self.current_permutation):
            return ConditionAssignment(
                cycle_index=self.cycle_index,
                condition_position_in_cycle=self.position,
                condition_N=self.current_permutation[self.position],
            )

        prospective_rng = random.Random()
        prospective_rng.setstate(self._rng.getstate())
        prospective_permutation = list(self.resolved_discovery_strata)
        prospective_rng.shuffle(prospective_permutation)
        return ConditionAssignment(
            cycle_index=self.cycle_index + 1,
            condition_position_in_cycle=0,
            condition_N=prospective_permutation[0],
        )

    def commit(self, assignment: ConditionAssignment) -> None:
        """Consume exactly the assignment returned by :meth:`peek`."""

        if not isinstance(assignment, ConditionAssignment):
            raise TypeError("assignment must be a ConditionAssignment")
        pending = self.peek()
        if assignment != pending:
            raise ValueError("condition assignment is stale or does not match pending state")
        if self.position == len(self.current_permutation):
            self._start_next_cycle()
        self.position += 1

    def next_node_count(self) -> int:
        assignment = self.peek()
        self.commit(assignment)
        return assignment.condition_N

    def next_batch(self, batch_size: int) -> tuple[int, ...]:
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool):
            raise ValueError("batch_size must be a positive integer")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        return tuple(self.next_node_count() for _ in range(int(batch_size)))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULER_SCHEMA,
            "resolved_discovery_strata": self.resolved_discovery_strata,
            "current_permutation": tuple(self.current_permutation),
            "cycle_index": self.cycle_index,
            "position": self.position,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("schema") != SCHEDULER_SCHEMA:
            raise ValueError("scheduler state schema is incompatible")
        strata = _normalize_strata(tuple(state["resolved_discovery_strata"]))
        if strata != self.resolved_discovery_strata:
            raise ValueError("scheduler discovery strata do not match")
        permutation = tuple(int(value) for value in state["current_permutation"])
        if sorted(permutation) != list(strata):
            raise ValueError("scheduler current permutation is invalid")
        cycle_index = state["cycle_index"]
        position = state["position"]
        if (
            not isinstance(cycle_index, Integral)
            or isinstance(cycle_index, bool)
            or int(cycle_index) < 0
        ):
            raise ValueError("scheduler cycle_index is invalid")
        if (
            not isinstance(position, Integral)
            or isinstance(position, bool)
            or not 0 <= int(position) <= len(permutation)
        ):
            raise ValueError("scheduler position is invalid")
        self.current_permutation = list(permutation)
        self.cycle_index = int(cycle_index)
        self.position = int(position)
        self._rng.setstate(state["rng_state"])


__all__ = [
    "BalancedNodeCountScheduler",
    "ConditionAssignment",
    "SCHEDULER_SCHEMA",
]
