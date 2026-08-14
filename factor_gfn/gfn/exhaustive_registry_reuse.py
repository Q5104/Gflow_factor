"""Fail-closed per-stratum exhaustive registry reuse and Reward lookup.

The proof intentionally ignores the global search-space fingerprint.  Reuse is
instead bound to an exact canonical hash set and explicit computation semantics.
This module is not exported by the package while the legacy 6/20 run is active.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable

from factor_gfn.grammar import Expression

from .exhaustive import ExhaustiveCandidate, ExhaustiveRegistry


EXHAUSTIVE_REUSE_SCHEMA = "factor_gfn.exhaustive_registry_reuse.v1"


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ExhaustiveReuseSemantics:
    grammar_semantics_fingerprint: str
    operator_semantics_fingerprint: str
    interpreter_semantics_fingerprint: str
    provider_fingerprint: str
    data_context_fingerprint: str
    reward_config_fingerprint: str
    reward_floor: float

    def __post_init__(self) -> None:
        for name in (
            "grammar_semantics_fingerprint",
            "operator_semantics_fingerprint",
            "interpreter_semantics_fingerprint",
            "provider_fingerprint",
            "data_context_fingerprint",
            "reward_config_fingerprint",
        ):
            _non_empty(getattr(self, name), name)
        if not math.isfinite(self.reward_floor) or self.reward_floor <= 0.0:
            raise ValueError("reward_floor must be finite and positive")


@dataclass(frozen=True, slots=True)
class ExhaustiveStratumReuseProof:
    schema: str
    node_count: int
    canonical_structural_hashes: tuple[str, ...]
    semantics: ExhaustiveReuseSemantics
    exact_aggregation_fingerprint: str
    proof_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegistryRewardLookup:
    node_count: int
    structural_hash: str
    valid: bool
    reward: float | None
    log_reward: float | None
    rejection_reason: str | None
    metadata: dict[str, Any]
    reward_source: str = "exhaustive_registry_cache"


def _canonical_hashes(expressions: Iterable[Expression], node_count: int) -> tuple[str, ...]:
    hashes: list[str] = []
    for expression in expressions:
        if not isinstance(expression, Expression):
            raise TypeError("target expressions must be Expression instances")
        canonical = expression.canonicalize()
        if canonical.stats.node_count != node_count:
            raise ValueError("target expression belongs to a different node-count stratum")
        hashes.append(canonical.structural_hash())
    if len(hashes) != len(set(hashes)):
        raise ValueError("target canonical enumeration contains duplicate hashes")
    if not hashes:
        raise ValueError("target canonical enumeration must not be empty")
    return tuple(sorted(hashes))


def prove_exhaustive_stratum_reuse(
    registry: ExhaustiveRegistry,
    *,
    node_count: int,
    target_expressions: Iterable[Expression],
    source_semantics: ExhaustiveReuseSemantics,
    target_semantics: ExhaustiveReuseSemantics,
) -> ExhaustiveStratumReuseProof:
    """Prove exact per-N reuse without consulting a global search fingerprint."""

    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 1:
        raise ValueError("node_count must be a positive integer")
    if source_semantics != target_semantics:
        differing = [
            name
            for name in asdict(source_semantics)
            if getattr(source_semantics, name) != getattr(target_semantics, name)
        ]
        raise ValueError(f"exhaustive reuse semantics mismatch: {differing}")
    coverage = registry.coverage(node_count)
    if not coverage["coverage_complete"]:
        raise RuntimeError(f"N={node_count} exhaustive registry coverage is incomplete")
    registry_candidates = registry.evaluated_candidates(node_count)
    registry_hashes = tuple(sorted(item.structural_hash for item in registry_candidates))
    target_hashes = _canonical_hashes(target_expressions, node_count)
    if registry_hashes != target_hashes:
        missing = sorted(set(target_hashes) - set(registry_hashes))
        extra = sorted(set(registry_hashes) - set(target_hashes))
        raise ValueError(
            "canonical structural-hash set mismatch: "
            f"missing_in_registry={missing[:5]}, extra_in_registry={extra[:5]}"
        )
    for candidate in registry_candidates:
        if candidate.provider_fingerprint != source_semantics.provider_fingerprint:
            raise ValueError("registry candidate provider fingerprint mismatch")
        if candidate.context_fingerprint != source_semantics.data_context_fingerprint:
            raise ValueError("registry candidate data-context fingerprint mismatch")
    exact = registry.exact_mass_result(node_count)
    if exact.provider_fingerprint != source_semantics.provider_fingerprint:
        raise ValueError("exact mass provider fingerprint mismatch")
    if exact.context_fingerprint != source_semantics.data_context_fingerprint:
        raise ValueError("exact mass data-context fingerprint mismatch")
    if exact.reward_floor != source_semantics.reward_floor:
        raise ValueError("exact mass reward_floor mismatch")
    payload = {
        "schema": EXHAUSTIVE_REUSE_SCHEMA,
        "node_count": node_count,
        "canonical_structural_hashes": target_hashes,
        "semantics": asdict(target_semantics),
        "exact_aggregation_fingerprint": exact.aggregation_fingerprint,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return ExhaustiveStratumReuseProof(
        schema=EXHAUSTIVE_REUSE_SCHEMA,
        node_count=node_count,
        canonical_structural_hashes=target_hashes,
        semantics=target_semantics,
        exact_aggregation_fingerprint=exact.aggregation_fingerprint,
        proof_fingerprint=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )


class ProvenExhaustiveRewardLookup:
    """Read Reward only from a stratum covered by an explicit reuse proof."""

    def __init__(
        self,
        registry: ExhaustiveRegistry,
        proof: ExhaustiveStratumReuseProof,
    ) -> None:
        if proof.schema != EXHAUSTIVE_REUSE_SCHEMA:
            raise ValueError("reuse proof schema is incompatible")
        self._registry = registry
        self._proof = proof
        self._allowed_hashes = frozenset(proof.canonical_structural_hashes)

    def lookup(self, expression: Expression) -> RegistryRewardLookup:
        if not isinstance(expression, Expression):
            raise TypeError("expression must be an Expression")
        canonical = expression.canonicalize()
        if canonical.stats.node_count != self._proof.node_count:
            raise KeyError("expression belongs to a different exhaustive stratum")
        structural_hash = canonical.structural_hash()
        if structural_hash not in self._allowed_hashes:
            raise KeyError("expression is not covered by the approved reuse proof")
        candidate = self._registry.candidate(structural_hash)
        self._validate_identity(candidate, canonical)
        metadata = deepcopy(candidate.reward_details or {})
        metadata["provider_cache_hit"] = True
        metadata["reward_source"] = "exhaustive_registry_cache"
        metadata["exhaustive_reuse_proof_fingerprint"] = self._proof.proof_fingerprint
        if not candidate.valid:
            if not candidate.rejection_reason or candidate.target_mass != 0.0:
                raise ValueError("invalid registry candidate has inconsistent audit fields")
            return RegistryRewardLookup(
                node_count=candidate.node_count,
                structural_hash=structural_hash,
                valid=False,
                reward=None,
                log_reward=None,
                rejection_reason=candidate.rejection_reason,
                metadata=metadata,
            )
        details = metadata.get("reward_result", metadata)
        if not isinstance(details, dict):
            raise ValueError("registry Reward details are malformed")
        reward = float(details["reward"])
        log_reward = float(details["log_reward"])
        raw_reward = float(details["raw_reward"])
        expected = max(raw_reward, self._proof.semantics.reward_floor)
        if (
            not math.isfinite(reward)
            or reward <= 0.0
            or reward != expected
            or candidate.target_mass != reward
            or not math.isclose(log_reward, math.log(reward), rel_tol=1e-13, abs_tol=1e-15)
        ):
            raise ValueError("registry Reward is inconsistent with the approved contract")
        return RegistryRewardLookup(
            node_count=candidate.node_count,
            structural_hash=structural_hash,
            valid=True,
            reward=reward,
            log_reward=log_reward,
            rejection_reason=None,
            metadata=metadata,
        )

    def _validate_identity(
        self, candidate: ExhaustiveCandidate, expression: Expression
    ) -> None:
        semantics = self._proof.semantics
        if candidate.status != "evaluated":
            raise RuntimeError("registry candidate has not been evaluated")
        if candidate.node_count != self._proof.node_count:
            raise ValueError("registry candidate node count mismatch")
        if candidate.formula != expression.to_formula():
            raise ValueError("registry candidate formula mismatch")
        if candidate.prefix_token_ids != expression.to_prefix():
            raise ValueError("registry candidate prefix-token mismatch")
        if candidate.provider_fingerprint != semantics.provider_fingerprint:
            raise ValueError("registry candidate provider fingerprint mismatch")
        if candidate.context_fingerprint != semantics.data_context_fingerprint:
            raise ValueError("registry candidate data-context fingerprint mismatch")


__all__ = [
    "EXHAUSTIVE_REUSE_SCHEMA",
    "ExhaustiveReuseSemantics",
    "ExhaustiveStratumReuseProof",
    "ProvenExhaustiveRewardLookup",
    "RegistryRewardLookup",
    "prove_exhaustive_stratum_reuse",
]
