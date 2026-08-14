"""Pure F/D/E/L resolution for the no-anchor conditional Stage 5 design.

This module is deliberately not imported by :mod:`factor_gfn.gfn`.  It can be
tested while the legacy 6/20 diagnostic keeps its already-imported runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


NO_ANCHOR_CONTRACT_SCHEMA = "factor_gfn.complexity_conditioned_no_anchor.v1"


def _node_counts(values: Iterable[int], name: str) -> tuple[int, ...]:
    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in normalized
    ):
        raise ValueError(f"{name} must contain positive integer node counts")
    return normalized


@dataclass(frozen=True, slots=True)
class NoAnchorStrataContract:
    """Resolved strata with exhaustive knowledge separated from discovery."""

    feasible_node_counts: tuple[int, ...]
    discovery_node_counts: tuple[int, ...]
    exact_normalizer_node_counts: tuple[int, ...]
    learned_normalizer_node_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        feasible = _node_counts(self.feasible_node_counts, "feasible_node_counts")
        discovery = _node_counts(
            self.discovery_node_counts, "discovery_node_counts"
        )
        exact = tuple(sorted(set(self.exact_normalizer_node_counts)))
        learned = tuple(sorted(set(self.learned_normalizer_node_counts)))
        for values, name in (
            (exact, "exact_normalizer_node_counts"),
            (learned, "learned_normalizer_node_counts"),
        ):
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in values
            ):
                raise ValueError(f"{name} must contain positive integer node counts")
        if discovery != feasible:
            raise ValueError("no-anchor contract requires D=F")
        if not set(exact).issubset(feasible):
            raise ValueError("exact-normalizer strata E must be a subset of F")
        if set(learned) != set(feasible) - set(exact):
            raise ValueError("learned-normalizer strata L must equal F-E")
        if set(exact) & set(learned):
            raise ValueError("E and L must be disjoint")
        object.__setattr__(self, "feasible_node_counts", feasible)
        object.__setattr__(self, "discovery_node_counts", discovery)
        object.__setattr__(self, "exact_normalizer_node_counts", exact)
        object.__setattr__(self, "learned_normalizer_node_counts", learned)

    @classmethod
    def resolve(
        cls,
        feasible_node_counts: Iterable[int],
        exact_normalizer_node_counts: Iterable[int],
    ) -> "NoAnchorStrataContract":
        feasible = _node_counts(feasible_node_counts, "feasible_node_counts")
        exact = tuple(sorted(set(exact_normalizer_node_counts)))
        return cls(
            feasible_node_counts=feasible,
            discovery_node_counts=feasible,
            exact_normalizer_node_counts=exact,
            learned_normalizer_node_counts=tuple(
                value for value in feasible if value not in set(exact)
            ),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": NO_ANCHOR_CONTRACT_SCHEMA,
            "resolved_feasible_node_counts": list(self.feasible_node_counts),
            "resolved_normal_discovery_node_counts": list(
                self.discovery_node_counts
            ),
            "resolved_exact_normalizer_node_counts": list(
                self.exact_normalizer_node_counts
            ),
            "resolved_learned_normalizer_node_counts": list(
                self.learned_normalizer_node_counts
            ),
            "normal_discovery_equals_feasible": True,
            "exact_registry_equivalence_verification": "run_initialization_once",
            "exact_registry_discovery_lookup_key": "structural_hash",
            "canonical_reenumeration_per_discovery_candidate": False,
        }


__all__ = ["NO_ANCHOR_CONTRACT_SCHEMA", "NoAnchorStrataContract"]
