"""Training-only diagnostics for terminal depth boundary pressure."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


DEPTH_BOUNDARY_DIAGNOSTIC_SCHEMA = "factor_gfn.depth_boundary_diagnostic.v1"
DEPTH_BOUNDARY_RECOMMENDATIONS = (
    "consider_expansion",
    "no_expansion_evidence",
    "insufficient_evidence",
)
DEPTH_CANDIDATE_AUDIT_FIELDS = (
    "source",
    "structural_hash",
    "node_count",
    "depth",
    "at_max_depth",
    "valid",
    "rejection_reason",
    "reward",
    "abs_train_ic",
    "factor_seconds",
    "reward_seconds",
    "evaluation_seconds",
    "provider_cache_hit",
)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unit_interval(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


@dataclass(frozen=True, slots=True)
class DepthBoundaryDiagnosticConfig:
    """Conservative advisory thresholds; never mutates the search boundary."""

    minimum_total_unique_candidates: int = 500
    minimum_unique_candidates_per_focus_depth: int = 100
    minimum_valid_quality_samples_per_focus_depth: int = 100
    consider_expansion_min_boundary_share: float = 0.10
    no_expansion_max_boundary_share: float = 0.03
    valid_rate_non_degradation_tolerance: float = 0.05
    valid_rate_clear_decline: float = 0.10
    quality_relative_non_degradation_tolerance: float = 0.10
    quality_relative_clear_decline: float = 0.15

    def __post_init__(self) -> None:
        for name in (
            "minimum_total_unique_candidates",
            "minimum_unique_candidates_per_focus_depth",
            "minimum_valid_quality_samples_per_focus_depth",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "consider_expansion_min_boundary_share",
            "no_expansion_max_boundary_share",
            "valid_rate_non_degradation_tolerance",
            "valid_rate_clear_decline",
            "quality_relative_non_degradation_tolerance",
            "quality_relative_clear_decline",
        ):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        if self.no_expansion_max_boundary_share > self.consider_expansion_min_boundary_share:
            raise ValueError("no-expansion boundary share must not exceed consider-expansion share")
        if self.valid_rate_non_degradation_tolerance > self.valid_rate_clear_decline:
            raise ValueError("valid-rate tolerance must not exceed clear-decline threshold")
        if self.quality_relative_non_degradation_tolerance > self.quality_relative_clear_decline:
            raise ValueError("quality tolerance must not exceed clear-decline threshold")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DepthBoundaryDiagnosticResult:
    summary: dict[str, Any]
    depth_metrics: tuple[dict[str, Any], ...]
    candidate_audit: tuple[dict[str, Any], ...]


def _finite_optional(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _record_value(record: Mapping[str, Any], name: str) -> Any:
    if name in record:
        return record[name]
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        if name in metadata:
            return metadata[name]
        reward_result = metadata.get("reward_result")
        if isinstance(reward_result, Mapping) and name in reward_result:
            return reward_result[name]
    return None


def _normalize_candidate(
    record: Mapping[str, Any],
    *,
    max_depth: int,
    max_nodes: int,
    discovery_node_counts: set[int],
) -> dict[str, Any] | None:
    if "source" not in record:
        raise ValueError("candidate record must declare an explicit source")
    if record.get("source") != "discovery":
        return None
    structural_hash = record.get("structural_hash")
    if not isinstance(structural_hash, str) or not structural_hash:
        raise ValueError("discovery candidate lacks structural_hash")
    node_count = record.get("node_count")
    depth = record.get("depth")
    if isinstance(node_count, bool) or not isinstance(node_count, int):
        raise ValueError("discovery candidate node_count must be an integer")
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise ValueError("discovery candidate depth must be an integer")
    if node_count not in discovery_node_counts or not 1 <= node_count <= max_nodes:
        raise ValueError("discovery candidate node_count is outside resolved discovery strata")
    if not 0 <= depth <= max_depth:
        raise ValueError("discovery candidate depth is outside configured boundary")
    valid = record.get("valid")
    if not isinstance(valid, bool):
        raise ValueError("discovery candidate valid flag must be bool")
    factor_seconds = _finite_optional(_record_value(record, "factor_seconds"))
    reward_seconds = _finite_optional(_record_value(record, "reward_seconds"))
    if factor_seconds is None or reward_seconds is None or factor_seconds < 0 or reward_seconds < 0:
        raise ValueError("every unique discovery candidate requires finite non-negative timings")
    reward = _finite_optional(_record_value(record, "reward")) if valid else None
    train_ic = _finite_optional(_record_value(record, "train_ic")) if valid else None
    return {
        "source": "discovery",
        "structural_hash": structural_hash,
        "node_count": node_count,
        "depth": depth,
        "at_max_depth": depth == max_depth,
        "valid": valid,
        "rejection_reason": record.get("rejection_reason"),
        "reward": reward,
        "abs_train_ic": None if train_ic is None else abs(train_ic),
        "factor_seconds": factor_seconds,
        "reward_seconds": reward_seconds,
        "evaluation_seconds": factor_seconds + reward_seconds,
        "provider_cache_hit": bool(_record_value(record, "provider_cache_hit")),
    }


def _deduplicate_candidates(
    records: Iterable[Mapping[str, Any]],
    *,
    max_depth: int,
    max_nodes: int,
    discovery_node_counts: set[int],
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    identity_fields = ("node_count", "depth", "valid", "reward", "abs_train_ic")
    for record in records:
        normalized = _normalize_candidate(
            record,
            max_depth=max_depth,
            max_nodes=max_nodes,
            discovery_node_counts=discovery_node_counts,
        )
        if normalized is None:
            continue
        key = normalized["structural_hash"]
        existing = unique.get(key)
        if existing is None:
            unique[key] = normalized
            continue
        if any(existing[name] != normalized[name] for name in identity_fields):
            raise ValueError(f"conflicting duplicate discovery candidate: {key}")
        if existing["provider_cache_hit"] and not normalized["provider_cache_hit"]:
            unique[key] = normalized
    return [unique[key] for key in sorted(unique)]


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def _metric_row(
    candidates: list[dict[str, Any]],
    *,
    scope: str,
    node_count: int | None,
    depth: int,
    scope_total: int,
) -> dict[str, Any]:
    selected = [candidate for candidate in candidates if candidate["depth"] == depth]
    rewards = [candidate["reward"] for candidate in selected if candidate["reward"] is not None]
    abs_ics = [candidate["abs_train_ic"] for candidate in selected if candidate["abs_train_ic"] is not None]
    timings = [candidate["evaluation_seconds"] for candidate in selected]
    valid_count = sum(int(candidate["valid"]) for candidate in selected)
    return {
        "scope": scope,
        "node_count": node_count,
        "depth": depth,
        "unique_candidate_count": len(selected),
        "depth_share_within_scope": len(selected) / scope_total if scope_total else None,
        "valid_count": valid_count,
        "valid_rate": valid_count / len(selected) if selected else None,
        "finite_reward_count": len(rewards),
        "reward_p90": _quantile(rewards, 0.90),
        "reward_p99": _quantile(rewards, 0.99),
        "finite_abs_ic_count": len(abs_ics),
        "abs_ic_p90": _quantile(abs_ics, 0.90),
        "abs_ic_p99": _quantile(abs_ics, 0.99),
        "evaluation_seconds_mean": float(np.mean(timings)) if timings else None,
        "evaluation_seconds_median": float(np.median(timings)) if timings else None,
    }


def _relative_change(current: float, reference: float) -> float:
    return (current - reference) / max(abs(reference), 1e-12)


def _recommendation(
    *,
    config: DepthBoundaryDiagnosticConfig,
    max_depth: int,
    total: int,
    overall_by_depth: Mapping[int, Mapping[str, Any]],
) -> tuple[str, list[str], dict[str, Any], dict[str, Any]]:
    focus_depths = (max_depth - 2, max_depth - 1, max_depth)
    focus = {depth: overall_by_depth[depth] for depth in focus_depths}
    boundary_share = focus[max_depth]["depth_share_within_scope"] or 0.0
    sufficiency: dict[str, Any] = {
        "total_unique_candidates": total,
        "minimum_total_unique_candidates": config.minimum_total_unique_candidates,
        "focus_depths": {},
    }
    insufficient_reasons: list[str] = []
    if total < config.minimum_total_unique_candidates:
        insufficient_reasons.append(
            f"total_unique_candidates={total} < required={config.minimum_total_unique_candidates}"
        )
    for depth, row in focus.items():
        depth_status = {
            "unique_candidate_count": row["unique_candidate_count"],
            "finite_reward_count": row["finite_reward_count"],
            "finite_abs_ic_count": row["finite_abs_ic_count"],
        }
        sufficiency["focus_depths"][str(depth)] = depth_status
        if row["unique_candidate_count"] < config.minimum_unique_candidates_per_focus_depth:
            insufficient_reasons.append(
                f"depth={depth} unique_candidate_count={row['unique_candidate_count']} "
                f"< required={config.minimum_unique_candidates_per_focus_depth}"
            )
        for field in ("finite_reward_count", "finite_abs_ic_count"):
            if row[field] < config.minimum_valid_quality_samples_per_focus_depth:
                insufficient_reasons.append(
                    f"depth={depth} {field}={row[field]} "
                    f"< required={config.minimum_valid_quality_samples_per_focus_depth}"
                )
    sufficiency["sufficient"] = not insufficient_reasons

    if total >= config.minimum_total_unique_candidates and boundary_share <= config.no_expansion_max_boundary_share:
        return (
            "no_expansion_evidence",
            [
                f"max_depth_share={boundary_share:.6f} <= "
                f"{config.no_expansion_max_boundary_share:.6f}"
            ],
            sufficiency,
            {},
        )
    if insufficient_reasons:
        return "insufficient_evidence", insufficient_reasons, sufficiency, {}

    previous = focus[max_depth - 1]
    boundary = focus[max_depth]
    valid_rate_change = boundary["valid_rate"] - previous["valid_rate"]
    quality_changes = {
        name: _relative_change(boundary[name], previous[name])
        for name in ("reward_p90", "reward_p99", "abs_ic_p90", "abs_ic_p99")
    }
    comparisons = {
        "reference_depth": max_depth - 1,
        "boundary_depth": max_depth,
        "valid_rate_change": valid_rate_change,
        "relative_quality_changes": quality_changes,
    }
    clear_decline = config.quality_relative_clear_decline
    if valid_rate_change <= -config.valid_rate_clear_decline:
        return (
            "no_expansion_evidence",
            [f"valid_rate_change={valid_rate_change:.6f} shows clear decline"],
            sufficiency,
            comparisons,
        )
    if quality_changes["reward_p90"] <= -clear_decline and quality_changes["abs_ic_p90"] <= -clear_decline:
        return (
            "no_expansion_evidence",
            ["Reward P90 and abs(IC) P90 both show clear decline at max_depth"],
            sufficiency,
            comparisons,
        )

    tolerance = config.quality_relative_non_degradation_tolerance
    p90_not_degraded = (
        quality_changes["reward_p90"] >= -tolerance
        and quality_changes["abs_ic_p90"] >= -tolerance
    )
    p99_not_both_degraded = not (
        quality_changes["reward_p99"] < -tolerance
        and quality_changes["abs_ic_p99"] < -tolerance
    )
    if (
        boundary_share >= config.consider_expansion_min_boundary_share
        and valid_rate_change >= -config.valid_rate_non_degradation_tolerance
        and p90_not_degraded
        and p99_not_both_degraded
    ):
        return (
            "consider_expansion",
            ["max_depth has substantial, sufficiently sampled candidates without clear quality degradation"],
            sufficiency,
            comparisons,
        )
    return (
        "insufficient_evidence",
        ["signals are mixed or lie between conservative decision thresholds"],
        sufficiency,
        comparisons,
    )


def build_depth_boundary_diagnostic(
    records: Iterable[Mapping[str, Any]],
    *,
    max_depth: int,
    max_nodes: int,
    discovery_node_counts: Iterable[int],
    config_fingerprint: str,
    provider_fingerprint: str,
    context_fingerprint: str,
    provider_manifest: Mapping[str, Any],
    config: DepthBoundaryDiagnosticConfig = DepthBoundaryDiagnosticConfig(),
) -> DepthBoundaryDiagnosticResult:
    """Aggregate unique training-only discovery candidates without changing training."""

    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 2:
        raise ValueError("max_depth must be an integer >= 2")
    _positive_int(max_nodes, "max_nodes")
    for value, name in (
        (config_fingerprint, "config_fingerprint"),
        (provider_fingerprint, "provider_fingerprint"),
        (context_fingerprint, "context_fingerprint"),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if provider_manifest.get("data_scope") != "training_only":
        raise ValueError("depth diagnostic requires data_scope=training_only")
    if provider_manifest.get("validation_oos_loaded") is not False:
        raise ValueError("depth diagnostic requires validation_oos_loaded=False")
    if provider_manifest.get("context_fingerprint") != context_fingerprint:
        raise ValueError("provider/context fingerprint mismatch")
    node_counts = tuple(sorted(discovery_node_counts))
    if not node_counts or len(set(node_counts)) != len(node_counts):
        raise ValueError("discovery_node_counts must be non-empty and unique")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= max_nodes for value in node_counts):
        raise ValueError("invalid discovery_node_counts")
    candidates = _deduplicate_candidates(
        records,
        max_depth=max_depth,
        max_nodes=max_nodes,
        discovery_node_counts=set(node_counts),
    )
    metrics: list[dict[str, Any]] = []
    for node_count in node_counts:
        subset = [candidate for candidate in candidates if candidate["node_count"] == node_count]
        for depth in range(max_depth + 1):
            metrics.append(
                _metric_row(
                    subset,
                    scope="by_node_count",
                    node_count=node_count,
                    depth=depth,
                    scope_total=len(subset),
                )
            )
    overall_rows: dict[int, dict[str, Any]] = {}
    for depth in range(max_depth + 1):
        row = _metric_row(
            candidates,
            scope="overall",
            node_count=None,
            depth=depth,
            scope_total=len(candidates),
        )
        metrics.append(row)
        overall_rows[depth] = row
    recommendation, reasons, sufficiency, comparisons = _recommendation(
        config=config,
        max_depth=max_depth,
        total=len(candidates),
        overall_by_depth=overall_rows,
    )
    per_node_boundary_share = {
        str(node_count): next(
            row["depth_share_within_scope"]
            for row in metrics
            if row["scope"] == "by_node_count"
            and row["node_count"] == node_count
            and row["depth"] == max_depth
        )
        for node_count in node_counts
    }
    summary = {
        "schema": DEPTH_BOUNDARY_DIAGNOSTIC_SCHEMA,
        "recommendation": recommendation,
        "depth_boundary_status": recommendation,
        "reasons": reasons,
        "advisory_only": True,
        "automatic_boundary_change": False,
        "automatic_run_creation": False,
        "data_scope": "training_only",
        "validation_oos_loaded": False,
        "config_fingerprint": config_fingerprint,
        "provider_fingerprint": provider_fingerprint,
        "context_fingerprint": context_fingerprint,
        "candidate_source": "discovery",
        "deduplication_key": "structural_hash",
        "quality_population": "valid_and_finite_only",
        "timing_population": "all_unique_discovery_candidates",
        "evaluation_seconds_definition": "factor_seconds + reward_seconds",
        "quantile_method": "numpy.linear",
        "max_depth": max_depth,
        "max_nodes": max_nodes,
        "max_nodes_boundary_diagnostic": "disabled_by_design",
        "discovery_node_counts": node_counts,
        "focus_depths": (max_depth - 2, max_depth - 1, max_depth),
        "total_unique_discovery_candidates": len(candidates),
        "unique_discovery_candidate_count": len(candidates),
        "overall_max_depth_share": overall_rows[max_depth]["depth_share_within_scope"],
        "max_depth_share_by_node_count": per_node_boundary_share,
        "sample_sufficiency": sufficiency,
        "comparisons": comparisons,
        "configured_thresholds": config.manifest(),
    }
    if recommendation not in DEPTH_BOUNDARY_RECOMMENDATIONS:
        raise AssertionError("unexpected recommendation")
    return DepthBoundaryDiagnosticResult(
        summary=summary,
        depth_metrics=tuple(metrics),
        candidate_audit=tuple(candidates),
    )


def write_depth_boundary_outputs(
    result: DepthBoundaryDiagnosticResult,
    directory: str | Path,
) -> tuple[Path, Path, Path]:
    """Write the frozen three-file diagnostic contract."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "depth_boundary_summary.json"
    metrics_path = root / "depth_metrics.csv"
    audit_path = root / "depth_candidate_audit.csv"
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for path, rows, fieldnames in (
        (metrics_path, result.depth_metrics, list(result.depth_metrics[0])),
        (audit_path, result.candidate_audit, list(DEPTH_CANDIDATE_AUDIT_FIELDS)),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary_path, metrics_path, audit_path


__all__ = [
    "DEPTH_BOUNDARY_DIAGNOSTIC_SCHEMA",
    "DEPTH_BOUNDARY_RECOMMENDATIONS",
    "DEPTH_CANDIDATE_AUDIT_FIELDS",
    "DepthBoundaryDiagnosticConfig",
    "DepthBoundaryDiagnosticResult",
    "build_depth_boundary_diagnostic",
    "write_depth_boundary_outputs",
]
