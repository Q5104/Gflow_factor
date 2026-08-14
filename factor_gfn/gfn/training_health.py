"""Pure per-N TB/logZ initialization health diagnostics for Step 12."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Literal, Mapping

import numpy as np


TRAINING_HEALTH_SCHEMA = "factor_gfn.no_anchor_training_health.v1"
InitializationHealthStatus = Literal[
    "usable",
    "review_targeted_recalibration",
    "insufficient_evidence",
    "fixed_exact_diagnostic",
]


@dataclass(frozen=True, slots=True)
class LogZInitializationHealthConfig:
    """Conservative engineering defaults; these are not fitted parameters."""

    minimum_valid_trajectories_per_N: int = 7
    minimum_successful_gradient_exposures_per_N: int = 7
    early_exposure_count: int = 3
    late_exposure_count: int = 3
    initial_abs_delta_mean_review_threshold: float = 5.0
    late_abs_delta_mean_review_threshold: float = 2.5
    log_z_net_change_review_threshold: float = 0.75
    directional_drift_fraction_threshold: float = 0.80
    minimum_log_z_step_for_direction: float = 1e-4

    def __post_init__(self) -> None:
        integer_fields = (
            "minimum_valid_trajectories_per_N",
            "minimum_successful_gradient_exposures_per_N",
            "early_exposure_count",
            "late_exposure_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "initial_abs_delta_mean_review_threshold",
            "late_abs_delta_mean_review_threshold",
            "log_z_net_change_review_threshold",
            "minimum_log_z_step_for_direction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.directional_drift_fraction_threshold)
        if not math.isfinite(fraction) or not 0.5 < fraction <= 1.0:
            raise ValueError(
                "directional_drift_fraction_threshold must be in (0.5, 1]"
            )
        required = 1 + self.early_exposure_count + self.late_exposure_count
        if self.minimum_successful_gradient_exposures_per_N < required:
            raise ValueError(
                "minimum successful exposures must keep initialization, early, "
                "and late windows disjoint"
            )


@dataclass(frozen=True, slots=True)
class LogZInitializationHealthResult:
    schema: str
    config: dict[str, Any]
    per_N: dict[int, dict[str, Any]]
    targeted_recalibration_node_counts: tuple[int, ...]
    insufficient_evidence_node_counts: tuple[int, ...]
    all_learned_initializations_usable: bool
    enriched_trajectory_rows: tuple[dict[str, Any], ...]


def _finite_float(value: Any, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _phase_metrics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "rms": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
    }


def _normalized_mapping(value: Mapping[int | str, float], name: str) -> dict[int, float]:
    output: dict[int, float] = {}
    for key, item in value.items():
        node_count = int(key)
        if node_count in output:
            raise ValueError(f"{name} has ambiguous node-count keys")
        output[node_count] = _finite_float(item, f"{name}[{node_count}]")
    return output


def build_log_z_initialization_health(
    trajectory_rows: Iterable[Mapping[str, Any]],
    training_rows: Iterable[Mapping[str, Any]],
    *,
    node_counts: tuple[int, ...],
    learned_node_counts: tuple[int, ...],
    initial_log_z_by_N: Mapping[int | str, float],
    current_log_z_by_N: Mapping[int | str, float],
    config: LogZInitializationHealthConfig = LogZInitializationHealthConfig(),
) -> LogZInitializationHealthResult:
    """Separate first/pre-update, early and late evidence without overlap."""

    nodes = tuple(sorted(node_counts))
    learned = tuple(sorted(learned_node_counts))
    if len(nodes) != len(set(nodes)) or len(learned) != len(set(learned)):
        raise ValueError("node-count strata must be unique")
    if not set(learned).issubset(nodes):
        raise ValueError("learned node counts must be a subset of all node counts")
    initial = _normalized_mapping(initial_log_z_by_N, "initial_log_z_by_N")
    current = _normalized_mapping(current_log_z_by_N, "current_log_z_by_N")
    if set(initial) != set(nodes) or set(current) != set(nodes):
        raise ValueError("initial/current logZ coverage must exactly match node_counts")
    rows = [dict(row) for row in trajectory_rows]
    for index, row in enumerate(rows):
        node_count = int(row["target_node_count"])
        if node_count not in nodes:
            raise ValueError("trajectory row belongs to an unresolved node count")
        row["target_node_count"] = node_count
        row["tb_delta"] = _finite_float(row["tb_delta"], "tb_delta")
        row["selected_log_z"] = _finite_float(
            row["selected_log_z"], "selected_log_z"
        )
        row["logical_batch"] = int(row["logical_batch"])
        row["successful_gradient_exposure"] = bool(
            row.get("successful_gradient_exposure", False)
        )
        row["_input_order"] = index
    rows.sort(key=lambda row: (row["logical_batch"], row["_input_order"]))
    training = [dict(row) for row in training_rows]
    training.sort(key=lambda row: int(row["logical_batch"]))
    per_N: dict[int, dict[str, Any]] = {}
    targeted: list[int] = []
    insufficient: list[int] = []
    enriched: list[dict[str, Any]] = []
    for node_count in nodes:
        node_rows = [row for row in rows if row["target_node_count"] == node_count]
        successful = [row for row in node_rows if row["successful_gradient_exposure"]]
        required = 1 + config.early_exposure_count + config.late_exposure_count
        phase_by_order: dict[int, str] = {}
        if successful:
            phase_by_order[successful[0]["_input_order"]] = "initialization_pre_update"
        for row in successful[1 : 1 + config.early_exposure_count]:
            phase_by_order[row["_input_order"]] = "early"
        if len(successful) >= required:
            for row in successful[-config.late_exposure_count :]:
                phase_by_order[row["_input_order"]] = "late"
        for row in node_rows:
            clean = {key: value for key, value in row.items() if key != "_input_order"}
            clean["initialization_health_phase"] = phase_by_order.get(
                row["_input_order"], "unassigned"
            )
            enriched.append(clean)
        phase_values = {
            phase: [
                row["tb_delta"]
                for row in successful
                if phase_by_order.get(row["_input_order"]) == phase
            ]
            for phase in ("initialization_pre_update", "early", "late")
        }
        valid_count = len(node_rows)
        exposure_count = len(successful)
        enough = (
            valid_count >= config.minimum_valid_trajectories_per_N
            and exposure_count
            >= config.minimum_successful_gradient_exposures_per_N
            and all(phase_values.values())
        )
        changes: list[float] = []
        previous = initial[node_count]
        for row in training:
            post = row.get("learned_log_z_by_N", {})
            if node_count not in learned:
                continue
            if node_count in post or str(node_count) in post:
                value = _finite_float(
                    post.get(node_count, post.get(str(node_count))),
                    f"training logZ N={node_count}",
                )
                change = value - previous
                if abs(change) >= config.minimum_log_z_step_for_direction:
                    changes.append(change)
                previous = value
        directional_fraction = None
        if changes:
            positives = sum(change > 0.0 for change in changes)
            negatives = sum(change < 0.0 for change in changes)
            directional_fraction = max(positives, negatives) / len(changes)
        phase_metrics = {
            phase: _phase_metrics(values) for phase, values in phase_values.items()
        }
        initial_delta_mean = phase_metrics["initialization_pre_update"]["mean"]
        late_delta_mean = phase_metrics["late"]["mean"]
        net_change = current[node_count] - initial[node_count]
        if not enough:
            status: InitializationHealthStatus = "insufficient_evidence"
            reasons = [
                "per-N valid trajectory or successful gradient exposure is insufficient"
            ]
            insufficient.append(node_count)
        elif node_count not in learned:
            status = "fixed_exact_diagnostic"
            reasons = ["exact fixed logZ is diagnostic-only and cannot be recalibrated"]
        else:
            large_initial = abs(float(initial_delta_mean)) >= (
                config.initial_abs_delta_mean_review_threshold
            )
            persistent_late = abs(float(late_delta_mean)) >= (
                config.late_abs_delta_mean_review_threshold
            )
            large_drift = abs(net_change) >= config.log_z_net_change_review_threshold
            directional_drift = (
                directional_fraction is not None
                and directional_fraction
                >= config.directional_drift_fraction_threshold
            )
            if large_initial and (persistent_late or (large_drift and directional_drift)):
                status = "review_targeted_recalibration"
                reasons = [
                    "large initial TB offset remains late or drives persistent logZ drift"
                ]
                targeted.append(node_count)
            else:
                status = "usable"
                reasons = ["short-run initialization health shows no configured anomaly"]
        per_N[node_count] = {
            "node_count": node_count,
            "normalizer_kind": "learned" if node_count in learned else "exact_fixed",
            "status": status,
            "reasons": reasons,
            "valid_trajectory_count": valid_count,
            "successful_gradient_exposure_count": exposure_count,
            "evidence_minimums": {
                "valid_trajectories": config.minimum_valid_trajectories_per_N,
                "successful_gradient_exposures": (
                    config.minimum_successful_gradient_exposures_per_N
                ),
            },
            "tb_delta_by_phase": phase_metrics,
            "initial_log_z": initial[node_count],
            "current_log_z": current[node_count],
            "net_change_log_z": net_change,
            "directional_log_z_update_fraction": directional_fraction,
        }
    enriched.sort(
        key=lambda row: (
            int(row["logical_batch"]),
            int(row["target_node_count"]),
            str(row.get("structural_hash", "")),
        )
    )
    all_learned_usable = all(
        per_N[node_count]["status"] == "usable" for node_count in learned
    )
    return LogZInitializationHealthResult(
        schema=TRAINING_HEALTH_SCHEMA,
        config=asdict(config),
        per_N=per_N,
        targeted_recalibration_node_counts=tuple(targeted),
        insufficient_evidence_node_counts=tuple(insufficient),
        all_learned_initializations_usable=all_learned_usable,
        enriched_trajectory_rows=tuple(enriched),
    )


__all__ = [
    "TRAINING_HEALTH_SCHEMA",
    "InitializationHealthStatus",
    "LogZInitializationHealthConfig",
    "LogZInitializationHealthResult",
    "build_log_z_initialization_health",
]
