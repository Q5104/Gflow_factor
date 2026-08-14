"""Bounded canonical counting and resumable exhaustive candidate registry."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable

from factor_gfn.grammar import (
    ExactNodeGrammarState,
    Expression,
    SearchSpaceConfig,
    resolve_exact_node_strata,
)


EXHAUSTIVE_PLANNING_SCHEMA = "factor_gfn.exhaustive_planning.v1"
EXHAUSTIVE_REGISTRY_SCHEMA = "factor_gfn.exhaustive_registry.v2"
EXHAUSTIVE_SOURCE = "exhaustive_full_evaluation"


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: float, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _normalized_node_counts(values: Iterable[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of positive integers")
    normalized = tuple(_positive_int(value, name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class ExhaustivePlanningConfig:
    """Explicit preflight policy; resolving it is intentionally not implicit."""

    canonical_count_cap: int = 10_000
    estimated_real_reward_seconds_per_candidate: float = 0.75
    planned_real_reward_budget_seconds: float = 3_600.0
    max_budget_fraction: float = 0.20
    explicit_include_node_counts: tuple[int, ...] = ()
    explicit_exclude_node_counts: tuple[int, ...] = ()
    approve_explicit_include_over_budget: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "canonical_count_cap",
            _positive_int(self.canonical_count_cap, "canonical_count_cap"),
        )
        for name in (
            "estimated_real_reward_seconds_per_candidate",
            "planned_real_reward_budget_seconds",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        fraction = _positive_float(self.max_budget_fraction, "max_budget_fraction")
        if fraction > 1.0:
            raise ValueError("max_budget_fraction must be in (0, 1]")
        object.__setattr__(self, "max_budget_fraction", fraction)
        include = _normalized_node_counts(
            self.explicit_include_node_counts,
            "explicit_include_node_counts",
        )
        exclude = _normalized_node_counts(
            self.explicit_exclude_node_counts,
            "explicit_exclude_node_counts",
        )
        overlap = sorted(set(include) & set(exclude))
        if overlap:
            raise ValueError(
                "explicit include/exclude node counts overlap: " f"{overlap}"
            )
        object.__setattr__(self, "explicit_include_node_counts", include)
        object.__setattr__(self, "explicit_exclude_node_counts", exclude)
        if not isinstance(self.approve_explicit_include_over_budget, bool):
            raise ValueError("approve_explicit_include_over_budget must be bool")

    @property
    def exhaustive_budget_seconds(self) -> float:
        return self.planned_real_reward_budget_seconds * self.max_budget_fraction

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": EXHAUSTIVE_PLANNING_SCHEMA,
            **asdict(self),
            "exhaustive_budget_seconds": self.exhaustive_budget_seconds,
            "budget_semantics": "cumulative_estimated_cost_of_all_resolved_exhaustive_strata",
        }


@dataclass(frozen=True, slots=True)
class CanonicalCountResult:
    node_count: int
    canonical_terminal_count: int
    canonical_count_exact: bool
    count_cap_reached: bool
    count_relation: str
    depth_distribution: dict[int, int]
    depth_distribution_exact: bool
    estimated_evaluation_seconds: float
    estimated_evaluation_seconds_is_lower_bound: bool
    expressions: tuple[Expression, ...] = field(repr=False, compare=False)

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("expressions")
        return payload


def count_canonical_terminals(
    *,
    search_space: SearchSpaceConfig,
    target_node_count: int,
    canonical_count_cap: int | None = 10_000,
    estimated_seconds_per_candidate: float = 0.75,
) -> CanonicalCountResult:
    """Enumerate unique canonical terminals, stopping at cap + 1 when bounded."""

    if not isinstance(search_space, SearchSpaceConfig):
        raise TypeError("search_space must be SearchSpaceConfig")
    target_node_count = _positive_int(target_node_count, "target_node_count")
    feasible = resolve_exact_node_strata(search_space).resolved_feasible_node_counts
    if target_node_count not in feasible:
        raise ValueError(f"target_node_count={target_node_count} is infeasible")
    if canonical_count_cap is not None:
        canonical_count_cap = _positive_int(canonical_count_cap, "canonical_count_cap")
    seconds_per_candidate = _positive_float(
        estimated_seconds_per_candidate,
        "estimated_seconds_per_candidate",
    )

    source = ExactNodeGrammarState.source(
        target_node_count=target_node_count,
        search_space=search_space,
    )
    pending = [source]
    visited_states = {source.conditioned_key}
    terminals: dict[str, Expression] = {}
    depth_distribution: Counter[int] = Counter()
    cap_reached = False

    while pending:
        state = pending.pop()
        if state.done:
            expression = state.to_expression().canonicalize()
            structural_hash = expression.structural_hash()
            previous = terminals.get(structural_hash)
            if previous is not None:
                if previous.canonical_key() != expression.canonical_key():
                    raise RuntimeError("structural hash collision between expressions")
                continue
            terminals[structural_hash] = expression
            depth_distribution[expression.stats.depth] += 1
            if canonical_count_cap is not None and len(terminals) > canonical_count_cap:
                cap_reached = True
                break
            continue
        for action in reversed(state.legal_transitions()):
            successor = state.step(action)
            if successor.conditioned_key in visited_states:
                continue
            visited_states.add(successor.conditioned_key)
            pending.append(successor)

    count = len(terminals)
    exact = not cap_reached
    return CanonicalCountResult(
        node_count=target_node_count,
        canonical_terminal_count=count,
        canonical_count_exact=exact,
        count_cap_reached=cap_reached,
        count_relation="=" if exact else ">",
        depth_distribution=dict(sorted(depth_distribution.items())),
        depth_distribution_exact=exact,
        estimated_evaluation_seconds=count * seconds_per_candidate,
        estimated_evaluation_seconds_is_lower_bound=not exact,
        expressions=tuple(terminals[key] for key in sorted(terminals)),
    )


@dataclass(frozen=True, slots=True)
class ExhaustivePlan:
    search_space_fingerprint: str
    resolved_feasible_node_counts: tuple[int, ...]
    resolved_exhaustive_node_counts: tuple[int, ...]
    resolved_discovery_node_counts: tuple[int, ...]
    automatic_exhaustive_node_counts: tuple[int, ...]
    explicit_exhaustive_node_counts: tuple[int, ...]
    excluded_node_counts: tuple[int, ...]
    count_results: tuple[CanonicalCountResult, ...] = field(repr=False)
    exhaustive_budget_seconds: float
    resolved_estimated_evaluation_seconds: float
    explicit_over_budget_approval_used: bool
    planning_config: ExhaustivePlanningConfig

    def count_result(self, node_count: int) -> CanonicalCountResult:
        for result in self.count_results:
            if result.node_count == node_count:
                return result
        raise KeyError(node_count)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": EXHAUSTIVE_PLANNING_SCHEMA,
            "search_space_fingerprint": self.search_space_fingerprint,
            "resolved_feasible_node_counts": self.resolved_feasible_node_counts,
            "resolved_exhaustive_node_counts": self.resolved_exhaustive_node_counts,
            "resolved_discovery_node_counts": self.resolved_discovery_node_counts,
            "automatic_exhaustive_node_counts": self.automatic_exhaustive_node_counts,
            "explicit_exhaustive_node_counts": self.explicit_exhaustive_node_counts,
            "excluded_node_counts": self.excluded_node_counts,
            "canonical_counts": [result.manifest() for result in self.count_results],
            "exhaustive_budget_seconds": self.exhaustive_budget_seconds,
            "resolved_estimated_evaluation_seconds": self.resolved_estimated_evaluation_seconds,
            "explicit_over_budget_approval_used": self.explicit_over_budget_approval_used,
            "planning_config": self.planning_config.manifest(),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.manifest(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_exhaustive_plan(
    search_space: SearchSpaceConfig,
    config: ExhaustivePlanningConfig = ExhaustivePlanningConfig(),
) -> ExhaustivePlan:
    """Count feasible strata and resolve E/S under one cumulative cost budget."""

    if not isinstance(search_space, SearchSpaceConfig):
        raise TypeError("search_space must be SearchSpaceConfig")
    if not isinstance(config, ExhaustivePlanningConfig):
        raise TypeError("config must be ExhaustivePlanningConfig")
    feasible = resolve_exact_node_strata(search_space).resolved_feasible_node_counts
    feasible_set = set(feasible)
    configured = set(config.explicit_include_node_counts) | set(
        config.explicit_exclude_node_counts
    )
    unknown = sorted(configured - feasible_set)
    if unknown:
        raise ValueError(f"explicit include/exclude contains infeasible strata: {unknown}")

    include = set(config.explicit_include_node_counts)
    exclude = set(config.explicit_exclude_node_counts)
    budget = config.exhaustive_budget_seconds
    seconds_per_candidate = config.estimated_real_reward_seconds_per_candidate
    explicit_running_cost = 0.0
    results: list[CanonicalCountResult] = []
    for node_count in feasible:
        # Explicit exclusion is authoritative: no count or cost estimate is
        # needed to prove that this stratum remains discovery-only.
        if node_count in exclude:
            continue
        if node_count in include and config.approve_explicit_include_over_budget:
            cap = None
        elif node_count in include:
            remaining_budget = budget - explicit_running_cost
            affordable_candidates = math.floor(remaining_budget / seconds_per_candidate)
            if affordable_candidates < 1:
                raise ValueError(
                    "explicit include exceeds the cumulative exhaustive budget; set "
                    "approve_explicit_include_over_budget=True only after secondary approval"
                )
            # The next unique terminal is sufficient proof that the budget is exceeded.
            cap = affordable_candidates
        else:
            cap = config.canonical_count_cap
        result = count_canonical_terminals(
            search_space=search_space,
            target_node_count=node_count,
            canonical_count_cap=cap,
            estimated_seconds_per_candidate=seconds_per_candidate,
        )
        if node_count in include:
            if not result.canonical_count_exact:
                raise ValueError(
                    "explicit include exceeds the cumulative exhaustive budget; set "
                    "approve_explicit_include_over_budget=True only after secondary approval"
                )
            explicit_running_cost += result.estimated_evaluation_seconds
        results.append(result)
    by_node_count = {result.node_count: result for result in results}

    explicit_cost = sum(
        by_node_count[node_count].estimated_evaluation_seconds
        for node_count in include
    )
    approval_used = explicit_cost > budget

    resolved = set(include)
    automatic: list[int] = []
    running_cost = explicit_cost
    for result in results:
        node_count = result.node_count
        if node_count in include or node_count in exclude:
            continue
        if not result.canonical_count_exact:
            continue
        proposed_cost = running_cost + result.estimated_evaluation_seconds
        if proposed_cost <= budget:
            resolved.add(node_count)
            automatic.append(node_count)
            running_cost = proposed_cost

    exhaustive = tuple(node_count for node_count in feasible if node_count in resolved)
    discovery = tuple(node_count for node_count in feasible if node_count not in resolved)
    return ExhaustivePlan(
        search_space_fingerprint=search_space.fingerprint(),
        resolved_feasible_node_counts=feasible,
        resolved_exhaustive_node_counts=exhaustive,
        resolved_discovery_node_counts=discovery,
        automatic_exhaustive_node_counts=tuple(automatic),
        explicit_exhaustive_node_counts=tuple(sorted(include)),
        excluded_node_counts=tuple(sorted(exclude)),
        count_results=tuple(results),
        exhaustive_budget_seconds=budget,
        resolved_estimated_evaluation_seconds=running_cost,
        explicit_over_budget_approval_used=approval_used,
        planning_config=config,
    )


@dataclass(frozen=True, slots=True)
class ExhaustiveCandidate:
    structural_hash: str
    source: str
    node_count: int
    depth: int
    formula: str
    prefix_token_ids: tuple[int, ...]
    provider_fingerprint: str
    context_fingerprint: str
    status: str
    valid: bool | None
    rejection_reason: str | None
    reward_details: dict[str, Any] | None
    target_mass: float | None


@dataclass(frozen=True, slots=True)
class ExactMassResult:
    node_count: int
    valid_candidate_count: int
    invalid_candidate_count: int
    exact_raw_reward_log_mass: float | None
    raw_reward_mass_status: str
    exact_tb_log_z: float
    reward_floor: float
    provider_fingerprint: str
    context_fingerprint: str
    aggregation_fingerprint: str


class ExhaustiveRegistry:
    """SQLite-backed authoritative pool, separate from discovery evaluations."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self.read_only:
            self._connection.execute("PRAGMA query_only = ON")
            row = self._connection.execute(
                "SELECT value_json FROM metadata WHERE key = 'schema'"
            ).fetchone()
            if row is None or json.loads(row["value_json"]) != EXHAUSTIVE_REGISTRY_SCHEMA:
                self._connection.close()
                raise ValueError("read-only exhaustive registry schema is incompatible")
        else:
            self._create_schema()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ExhaustiveRegistry":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS strata (
                    node_count INTEGER PRIMARY KEY,
                    expected_canonical_count INTEGER NOT NULL,
                    enumeration_complete INTEGER NOT NULL CHECK (enumeration_complete IN (0, 1)),
                    exact_status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (exact_status IN ('pending', 'complete', 'failed_no_valid')),
                    exact_valid_candidate_count INTEGER,
                    exact_invalid_candidate_count INTEGER,
                    exact_raw_reward_log_mass REAL,
                    raw_reward_mass_status TEXT
                        CHECK (raw_reward_mass_status IN ('positive_mass', 'zero_mass') OR raw_reward_mass_status IS NULL),
                    exact_tb_log_z REAL,
                    exact_reward_floor REAL,
                    exact_aggregation_fingerprint TEXT
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    structural_hash TEXT PRIMARY KEY,
                    source TEXT NOT NULL CHECK (source = 'exhaustive_full_evaluation'),
                    node_count INTEGER NOT NULL REFERENCES strata(node_count),
                    depth INTEGER NOT NULL,
                    formula TEXT NOT NULL,
                    prefix_token_ids_json TEXT NOT NULL,
                    provider_fingerprint TEXT NOT NULL,
                    context_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'evaluated')),
                    valid INTEGER CHECK (valid IN (0, 1) OR valid IS NULL),
                    rejection_reason TEXT,
                    reward_details_json TEXT,
                    target_mass REAL,
                    CHECK (
                        (status = 'pending' AND valid IS NULL AND rejection_reason IS NULL
                         AND reward_details_json IS NULL AND target_mass IS NULL)
                        OR status = 'evaluated'
                    ),
                    CHECK (valid IS NULL OR valid = 1 OR target_mass = 0.0)
                );
                CREATE INDEX IF NOT EXISTS candidates_by_stratum_status
                    ON candidates(node_count, status, structural_hash);
                """
            )
            self._set_metadata("schema", EXHAUSTIVE_REGISTRY_SCHEMA)

    def _set_metadata(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        row = self._connection.execute(
            "SELECT value_json FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is not None and row["value_json"] != encoded:
            raise ValueError(f"registry metadata mismatch for {key}")
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value_json) VALUES (?, ?)",
            (key, encoded),
        )

    def _require_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("exhaustive registry is read-only")

    def register_plan(
        self,
        plan: ExhaustivePlan,
        *,
        provider_fingerprint: str,
        context_fingerprint: str,
    ) -> None:
        self._require_writable()
        if not provider_fingerprint or not context_fingerprint:
            raise ValueError("provider/context fingerprints must be non-empty")
        with self._connection:
            self._set_metadata("plan_fingerprint", plan.fingerprint())
            self._set_metadata("plan_manifest", plan.manifest())
            for node_count in plan.resolved_exhaustive_node_counts:
                result = plan.count_result(node_count)
                if not result.canonical_count_exact:
                    raise ValueError("exhaustive strata require complete canonical counting")
                self._register_stratum(
                    result,
                    provider_fingerprint=provider_fingerprint,
                    context_fingerprint=context_fingerprint,
                )

    def _register_stratum(
        self,
        result: CanonicalCountResult,
        *,
        provider_fingerprint: str,
        context_fingerprint: str,
    ) -> None:
        existing = self._connection.execute(
            "SELECT expected_canonical_count FROM strata WHERE node_count = ?",
            (result.node_count,),
        ).fetchone()
        if existing is not None and existing["expected_canonical_count"] != result.canonical_terminal_count:
            raise ValueError(f"stratum N={result.node_count} canonical count changed")
        self._connection.execute(
            "INSERT OR IGNORE INTO strata(node_count, expected_canonical_count, enumeration_complete) "
            "VALUES (?, ?, 0)",
            (result.node_count, result.canonical_terminal_count),
        )
        for expression in result.expressions:
            canonical = expression.canonicalize()
            row = (
                canonical.structural_hash(),
                EXHAUSTIVE_SOURCE,
                result.node_count,
                canonical.stats.depth,
                canonical.to_formula(),
                json.dumps(canonical.to_prefix(), separators=(",", ":")),
                provider_fingerprint,
                context_fingerprint,
            )
            current = self._connection.execute(
                "SELECT source, node_count, depth, formula, prefix_token_ids_json, "
                "provider_fingerprint, context_fingerprint FROM candidates "
                "WHERE structural_hash = ?",
                (row[0],),
            ).fetchone()
            if current is not None and tuple(current) != row[1:]:
                raise ValueError(f"candidate identity mismatch for {row[0]}")
            self._connection.execute(
                "INSERT OR IGNORE INTO candidates("
                "structural_hash, source, node_count, depth, formula, prefix_token_ids_json, "
                "provider_fingerprint, context_fingerprint, status"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                row,
            )
        actual = self._connection.execute(
            "SELECT COUNT(*) AS count FROM candidates WHERE node_count = ?",
            (result.node_count,),
        ).fetchone()["count"]
        if actual != result.canonical_terminal_count:
            raise RuntimeError(
                f"stratum N={result.node_count} registry count {actual} does not match "
                f"canonical count {result.canonical_terminal_count}"
            )
        self._connection.execute(
            "UPDATE strata SET enumeration_complete = 1 WHERE node_count = ?",
            (result.node_count,),
        )

    def pending_candidates(self, node_count: int | None = None) -> tuple[ExhaustiveCandidate, ...]:
        query = "SELECT * FROM candidates WHERE status = 'pending'"
        parameters: tuple[Any, ...] = ()
        if node_count is not None:
            query += " AND node_count = ?"
            parameters = (_positive_int(node_count, "node_count"),)
        query += " ORDER BY node_count, structural_hash"
        return tuple(self._candidate_from_row(row) for row in self._connection.execute(query, parameters))

    def evaluated_candidates(self, node_count: int) -> tuple[ExhaustiveCandidate, ...]:
        node_count = _positive_int(node_count, "node_count")
        return tuple(
            self._candidate_from_row(row)
            for row in self._connection.execute(
                "SELECT * FROM candidates WHERE node_count = ? AND status = 'evaluated' "
                "ORDER BY structural_hash",
                (node_count,),
            )
        )

    def record_evaluation(
        self,
        structural_hash: str,
        *,
        valid: bool,
        reward_details: dict[str, Any],
        rejection_reason: str | None = None,
        target_mass: float | None = None,
    ) -> None:
        self._require_writable()
        if not isinstance(valid, bool):
            raise TypeError("valid must be bool")
        if not isinstance(reward_details, dict):
            raise TypeError("reward_details must be a dict")
        if valid:
            if rejection_reason is not None:
                raise ValueError("valid candidates cannot have a rejection_reason")
            if (
                not isinstance(target_mass, Real)
                or isinstance(target_mass, bool)
                or not math.isfinite(float(target_mass))
                or float(target_mass) < 0.0
            ):
                raise ValueError("valid candidates require finite non-negative target_mass")
            normalized_mass = float(target_mass)
        else:
            if not rejection_reason:
                raise ValueError("invalid candidates require a rejection_reason")
            if target_mass not in (None, 0, 0.0):
                raise ValueError("invalid candidates must have zero target_mass")
            normalized_mass = 0.0
        details_json = json.dumps(
            reward_details, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        desired = (int(valid), rejection_reason, details_json, normalized_mass)
        with self._connection:
            current = self._connection.execute(
                "SELECT status, valid, rejection_reason, reward_details_json, target_mass "
                "FROM candidates WHERE structural_hash = ?",
                (structural_hash,),
            ).fetchone()
            if current is None:
                raise KeyError(structural_hash)
            if current["status"] == "evaluated":
                actual = (
                    current["valid"],
                    current["rejection_reason"],
                    current["reward_details_json"],
                    current["target_mass"],
                )
                if actual != desired:
                    raise ValueError("completed exhaustive evaluation cannot be overwritten")
                return
            self._connection.execute(
                "UPDATE candidates SET status = 'evaluated', valid = ?, rejection_reason = ?, "
                "reward_details_json = ?, target_mass = ? WHERE structural_hash = ?",
                (*desired, structural_hash),
            )

    @staticmethod
    def _reward_result_details(details: dict[str, Any]) -> dict[str, Any]:
        nested = details.get("reward_result")
        if nested is None:
            return details
        if not isinstance(nested, dict):
            raise ValueError("reward_details.reward_result must be a dict")
        return nested

    def compute_exact_masses(
        self,
        node_count: int,
        *,
        reward_floor: float,
    ) -> ExactMassResult:
        self._require_writable()
        """Compute authoritative raw and TB masses from full evaluated coverage."""

        node_count = _positive_int(node_count, "node_count")
        reward_floor = _positive_float(reward_floor, "reward_floor")
        coverage = self.coverage(node_count)
        if not coverage["evaluation_complete"]:
            raise RuntimeError(
                f"stratum N={node_count} lacks complete exhaustive evaluation coverage"
            )
        rows = tuple(
            self._connection.execute(
                "SELECT structural_hash, valid, reward_details_json, target_mass, "
                "provider_fingerprint, context_fingerprint FROM candidates "
                "WHERE node_count = ? ORDER BY structural_hash",
                (node_count,),
            )
        )
        provider_fingerprints = {row["provider_fingerprint"] for row in rows}
        context_fingerprints = {row["context_fingerprint"] for row in rows}
        if len(provider_fingerprints) != 1 or len(context_fingerprints) != 1:
            raise RuntimeError("exhaustive stratum mixes provider/context fingerprints")
        raw_rewards: list[float] = []
        tb_log_rewards: list[float] = []
        audit_payload: list[tuple[str, float, float, float]] = []
        invalid_count = 0
        for row in rows:
            if not bool(row["valid"]):
                invalid_count += 1
                if row["target_mass"] != 0.0:
                    raise ValueError("invalid exhaustive candidate must have zero target mass")
                continue
            details = json.loads(row["reward_details_json"])
            result = self._reward_result_details(details)
            missing = sorted(
                {"raw_reward", "reward", "log_reward"} - set(result)
            )
            if missing:
                raise ValueError(
                    f"valid exhaustive candidate lacks Reward fields: {missing}"
                )
            raw_reward = float(result["raw_reward"])
            tb_reward = float(result["reward"])
            log_reward = float(result["log_reward"])
            if (
                not math.isfinite(raw_reward)
                or raw_reward < 0.0
                or not math.isfinite(tb_reward)
                or tb_reward <= 0.0
                or not math.isfinite(log_reward)
            ):
                raise ValueError("valid exhaustive Reward fields must be finite")
            expected_tb_reward = max(raw_reward, reward_floor)
            if tb_reward != expected_tb_reward:
                raise ValueError("stored TB reward does not equal max(raw_reward, reward_floor)")
            if not math.isclose(
                log_reward, math.log(tb_reward), rel_tol=1e-13, abs_tol=1e-15
            ):
                raise ValueError("stored TB log_reward is inconsistent with TB reward")
            if float(row["target_mass"]) != tb_reward:
                raise ValueError("registry target mass does not equal stored TB reward")
            raw_rewards.append(raw_reward)
            tb_log_rewards.append(log_reward)
            audit_payload.append(
                (row["structural_hash"], raw_reward, tb_reward, log_reward)
            )

        valid_count = len(raw_rewards)
        if valid_count == 0:
            with self._connection:
                self._connection.execute(
                    "UPDATE strata SET exact_status = 'failed_no_valid', "
                    "exact_valid_candidate_count = 0, exact_invalid_candidate_count = ? "
                    "WHERE node_count = ?",
                    (invalid_count, node_count),
                )
            raise RuntimeError(
                f"stratum N={node_count} has no valid exhaustive candidate; "
                "exact Z and anchors are forbidden pending manual confirmation"
            )

        raw_total = math.fsum(raw_rewards)
        if raw_total == 0.0:
            raw_log_mass = None
            raw_status = "zero_mass"
        else:
            raw_log_mass = math.log(raw_total)
            raw_status = "positive_mass"
        maximum_log_reward = max(tb_log_rewards)
        exact_tb_log_z = maximum_log_reward + math.log(
            math.fsum(
                math.exp(value - maximum_log_reward) for value in tb_log_rewards
            )
        )
        if not math.isfinite(exact_tb_log_z):
            raise RuntimeError("exact TB logZ is not finite")
        aggregation_payload = json.dumps(
            {
                "schema": "factor_gfn.exact_reward_mass.v1",
                "node_count": node_count,
                "reward_floor": reward_floor,
                "candidates": audit_payload,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        aggregation_fingerprint = hashlib.sha256(
            aggregation_payload.encode("utf-8")
        ).hexdigest()
        result = ExactMassResult(
            node_count=node_count,
            valid_candidate_count=valid_count,
            invalid_candidate_count=invalid_count,
            exact_raw_reward_log_mass=raw_log_mass,
            raw_reward_mass_status=raw_status,
            exact_tb_log_z=exact_tb_log_z,
            reward_floor=reward_floor,
            provider_fingerprint=next(iter(provider_fingerprints)),
            context_fingerprint=next(iter(context_fingerprints)),
            aggregation_fingerprint=aggregation_fingerprint,
        )
        with self._connection:
            current = self._connection.execute(
                "SELECT exact_status, exact_aggregation_fingerprint FROM strata "
                "WHERE node_count = ?",
                (node_count,),
            ).fetchone()
            if current["exact_status"] == "complete":
                if current["exact_aggregation_fingerprint"] != aggregation_fingerprint:
                    raise ValueError("completed exact mass cannot be overwritten")
                return self.exact_mass_result(node_count)
            self._connection.execute(
                "UPDATE strata SET exact_status = 'complete', "
                "exact_valid_candidate_count = ?, exact_invalid_candidate_count = ?, "
                "exact_raw_reward_log_mass = ?, raw_reward_mass_status = ?, "
                "exact_tb_log_z = ?, exact_reward_floor = ?, "
                "exact_aggregation_fingerprint = ? WHERE node_count = ?",
                (
                    valid_count,
                    invalid_count,
                    raw_log_mass,
                    raw_status,
                    exact_tb_log_z,
                    reward_floor,
                    aggregation_fingerprint,
                    node_count,
                ),
            )
        return result

    def exact_mass_result(self, node_count: int) -> ExactMassResult:
        node_count = _positive_int(node_count, "node_count")
        row = self._connection.execute(
            "SELECT s.*, c.provider_fingerprint, c.context_fingerprint "
            "FROM strata s JOIN candidates c ON c.node_count = s.node_count "
            "WHERE s.node_count = ? LIMIT 1",
            (node_count,),
        ).fetchone()
        if row is None or row["exact_status"] != "complete":
            raise RuntimeError(f"stratum N={node_count} has no complete exact mass")
        return ExactMassResult(
            node_count=node_count,
            valid_candidate_count=row["exact_valid_candidate_count"],
            invalid_candidate_count=row["exact_invalid_candidate_count"],
            exact_raw_reward_log_mass=row["exact_raw_reward_log_mass"],
            raw_reward_mass_status=row["raw_reward_mass_status"],
            exact_tb_log_z=row["exact_tb_log_z"],
            reward_floor=row["exact_reward_floor"],
            provider_fingerprint=row["provider_fingerprint"],
            context_fingerprint=row["context_fingerprint"],
            aggregation_fingerprint=row["exact_aggregation_fingerprint"],
        )

    def coverage(self, node_count: int) -> dict[str, Any]:
        node_count = _positive_int(node_count, "node_count")
        row = self._connection.execute(
            "SELECT expected_canonical_count, enumeration_complete FROM strata WHERE node_count = ?",
            (node_count,),
        ).fetchone()
        if row is None:
            raise KeyError(node_count)
        counts = self._connection.execute(
            "SELECT COUNT(*) AS registered, "
            "SUM(CASE WHEN status = 'evaluated' THEN 1 ELSE 0 END) AS evaluated, "
            "SUM(CASE WHEN valid = 1 THEN 1 ELSE 0 END) AS valid, "
            "SUM(CASE WHEN valid = 0 THEN 1 ELSE 0 END) AS invalid "
            "FROM candidates WHERE node_count = ?",
            (node_count,),
        ).fetchone()
        expected = row["expected_canonical_count"]
        registered = counts["registered"]
        evaluated = counts["evaluated"] or 0
        enumeration_complete = bool(row["enumeration_complete"] and registered == expected)
        evaluation_complete = bool(enumeration_complete and evaluated == expected)
        return {
            "node_count": node_count,
            "expected_canonical_count": expected,
            "registered_count": registered,
            "evaluated_count": evaluated,
            "valid_count": counts["valid"] or 0,
            "invalid_count": counts["invalid"] or 0,
            "enumeration_complete": enumeration_complete,
            "evaluation_complete": evaluation_complete,
            "coverage_complete": evaluation_complete,
        }

    def candidate(self, structural_hash: str) -> ExhaustiveCandidate:
        row = self._connection.execute(
            "SELECT * FROM candidates WHERE structural_hash = ?", (structural_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(structural_hash)
        return self._candidate_from_row(row)

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> ExhaustiveCandidate:
        return ExhaustiveCandidate(
            structural_hash=row["structural_hash"],
            source=row["source"],
            node_count=row["node_count"],
            depth=row["depth"],
            formula=row["formula"],
            prefix_token_ids=tuple(json.loads(row["prefix_token_ids_json"])),
            provider_fingerprint=row["provider_fingerprint"],
            context_fingerprint=row["context_fingerprint"],
            status=row["status"],
            valid=None if row["valid"] is None else bool(row["valid"]),
            rejection_reason=row["rejection_reason"],
            reward_details=(
                None
                if row["reward_details_json"] is None
                else json.loads(row["reward_details_json"])
            ),
            target_mass=row["target_mass"],
        )


__all__ = [
    "EXHAUSTIVE_PLANNING_SCHEMA",
    "EXHAUSTIVE_REGISTRY_SCHEMA",
    "EXHAUSTIVE_SOURCE",
    "CanonicalCountResult",
    "ExhaustiveCandidate",
    "ExhaustivePlan",
    "ExhaustivePlanningConfig",
    "ExhaustiveRegistry",
    "ExactMassResult",
    "count_canonical_terminals",
    "resolve_exhaustive_plan",
]
