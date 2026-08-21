"""Fail-closed adapter for Stage 5 Hybrid reporting.

This module reads one stable persisted snapshot and reduces it to compact
tables.  It never evaluates expressions, recomputes Train metrics, or writes
training artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from factor_gfn.gfn.hybrid_checkpoint import HYBRID_CHECKPOINT_SCHEMA
from factor_gfn.gfn.hybrid_config import HybridVarianceGFNConfig
from factor_gfn.gfn.hybrid_search_runner import HYBRID_VARIANCE_RUNNER_SCHEMA
from factor_gfn.gfn.train_candidate_artifact import (
    TRAIN_CANDIDATE_ARTIFACT_SCHEMA,
    TRAIN_CANDIDATE_RECORD_SCHEMA,
    TRAIN_EVALUATION_CONTRACT_SCHEMA,
)
from factor_gfn.grammar import LEAVES, WINDOWS, get_action


STAGE5_REPORT_DATA_SCHEMA = "factor_gfn.reporting.stage5_data.v1"
_EXACT_NS = (1, 2)
_LPV_NS = tuple(range(3, 16))
_REQUIRED_FILES = {
    "run_config": "hybrid_run_config.json",
    "runner_state": "runner_state.json",
    "checkpoint": "checkpoint_latest.pt",
    "diagnostics": "hybrid_diagnostics.jsonl",
    "candidate_artifact": "train_candidate_artifact.json",
}
_COMMON_DIAGNOSTIC_FIELDS = {
    "cycle_index",
    "condition_position_in_cycle",
    "condition_N",
    "objective_kind",
    "global_optimizer_step",
    "trajectories_in_batch",
    "total_trajectories_seen",
    "requested_count",
    "accepted_count",
    "invalid_count",
    "retry_count",
    "retry_exhausted_count",
    "reward_mean",
    "policy_grad_norm",
}


@dataclass(frozen=True)
class Stage5ReportDataBundle:
    """Compact, renderer-ready representation of one stable Stage 5 snapshot."""

    snapshot_manifest: dict[str, Any]
    run_summary: pd.DataFrame
    training_updates: pd.DataFrame
    training_summary_by_n: pd.DataFrame
    exploration_by_cycle: pd.DataFrame
    exploration_summary: pd.DataFrame
    exploration_by_n: pd.DataFrame
    candidate_summary: pd.DataFrame
    candidate_quality_summary: pd.DataFrame
    quality_reference_counts: pd.DataFrame
    complexity_summary: pd.DataFrame
    operator_usage: pd.DataFrame
    field_usage: pd.DataFrame
    window_usage: pd.DataFrame
    selected_long_excess_series: pd.DataFrame
    long_excess_correlation_matrix: pd.DataFrame
    long_excess_correlation_summary: pd.DataFrame
    top_candidate_examples: pd.DataFrame
    availability_and_warnings: pd.DataFrame


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Stage 5 {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Stage 5 {label} must be a JSON object")
    return payload


def _read_diagnostics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                records.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"cannot read Stage 5 diagnostics JSONL at line {line_number}: {path}"
        ) from error
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    # Hybrid checkpoints contain Python/NumPy RNG state, so weights_only=True
    # cannot read the frozen v1 contract. Reporting reads trusted local runs only.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot read Stage 5 checkpoint: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Stage 5 checkpoint must contain a dictionary")
    keys = (
        "schema",
        "saved_at_utc",
        "objective_mode",
        "config_fingerprint",
        "reward_provider_fingerprint",
        "global_optimizer_step",
        "total_trajectories_seen",
    )
    return {key: payload.get(key) for key in keys}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_diagnostics(
    records: list[dict[str, Any]],
    *,
    trajectories_per_batch: int,
) -> int:
    for expected_step, record in enumerate(records, start=1):
        missing = _COMMON_DIAGNOSTIC_FIELDS.difference(record)
        _require(not missing, f"diagnostics record lacks fields: {sorted(missing)}")
        _require(
            record["global_optimizer_step"] == expected_step,
            "diagnostics optimizer steps are not contiguous",
        )
        _require(
            record["total_trajectories_seen"]
            == expected_step * trajectories_per_batch,
            "diagnostics trajectory totals are inconsistent",
        )
        node_count = record["condition_N"]
        objective = record["objective_kind"]
        if node_count in _EXACT_NS:
            _require(objective == "exact_tb", "N=1/2 diagnostics must use exact-TB")
            _require("tb_loss" in record, "exact-TB diagnostics lack tb_loss")
            _require(
                "variance_loss" not in record,
                "exact-TB diagnostics must not contain variance_loss",
            )
        elif node_count in _LPV_NS:
            _require(
                objective == "log_partition_variance",
                "N=3..15 diagnostics must use LPV",
            )
            _require("variance_loss" in record, "LPV diagnostics lack variance_loss")
            _require("tb_loss" not in record, "LPV diagnostics must not contain tb_loss")
        else:
            raise ValueError(f"diagnostics condition_N is outside 1..15: {node_count}")
    return int(records[-1]["global_optimizer_step"]) if records else 0


def _snapshot_gate(
    *,
    run_config_path: Path,
    run_config: Mapping[str, Any],
    state: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    diagnostics_step: int,
    artifact: Mapping[str, Any],
    expected_config: HybridVarianceGFNConfig,
    allow_incomplete: bool,
) -> dict[str, Any]:
    _require(
        run_config.get("schema") == HYBRID_VARIANCE_RUNNER_SCHEMA,
        "hybrid run config schema mismatch",
    )
    _require(
        state.get("schema") == HYBRID_VARIANCE_RUNNER_SCHEMA,
        "hybrid runner state schema mismatch",
    )
    _require(
        run_config.get("checkpoint_schema") == HYBRID_CHECKPOINT_SCHEMA,
        "hybrid run config checkpoint schema mismatch",
    )
    _require(
        checkpoint.get("schema") == HYBRID_CHECKPOINT_SCHEMA,
        "hybrid checkpoint schema mismatch",
    )
    _require(
        artifact.get("schema") == TRAIN_CANDIDATE_ARTIFACT_SCHEMA,
        "Train candidate artifact schema mismatch",
    )
    _require(
        run_config.get("objective_mode") == "hybrid_variance"
        and checkpoint.get("objective_mode") == "hybrid_variance",
        "Stage 5 objective mode mismatch",
    )
    artifact_manifest = run_config.get("train_candidate_artifact") or {}
    _require(
        artifact_manifest.get("enabled") is True
        and artifact_manifest.get("schema") == TRAIN_CANDIDATE_ARTIFACT_SCHEMA
        and artifact_manifest.get("filename") == _REQUIRED_FILES["candidate_artifact"],
        "hybrid run config Train artifact contract mismatch",
    )
    expected_fingerprint = expected_config.fingerprint()
    fingerprints = {
        "expected_config": expected_fingerprint,
        "run_config": run_config.get("config_fingerprint"),
        "checkpoint": checkpoint.get("config_fingerprint"),
    }
    _require(
        len(set(fingerprints.values())) == 1,
        f"config fingerprint mismatch: {fingerprints}",
    )
    reward_fingerprints = {
        "run_config": run_config.get("reward_provider_fingerprint"),
        "checkpoint": checkpoint.get("reward_provider_fingerprint"),
        "artifact_contract": (
            artifact.get("train_evaluation_contract") or {}
        ).get("provider_fingerprint"),
    }
    _require(
        None not in reward_fingerprints.values()
        and len(set(reward_fingerprints.values())) == 1,
        f"Reward provider fingerprint mismatch: {reward_fingerprints}",
    )
    train_contract = artifact.get("train_evaluation_contract") or {}
    _require(
        train_contract.get("schema") == TRAIN_EVALUATION_CONTRACT_SCHEMA,
        "Train evaluation contract schema mismatch",
    )
    contract_fingerprint = artifact.get("train_evaluation_contract_fingerprint")
    _require(
        isinstance(contract_fingerprint, str) and bool(contract_fingerprint),
        "Train evaluation contract fingerprint is missing",
    )
    steps = {
        "runner": state.get("global_optimizer_step"),
        "diagnostics": diagnostics_step,
        "artifact": artifact.get("committed_optimizer_step"),
        "checkpoint": checkpoint.get("global_optimizer_step"),
    }
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in steps.values()),
        f"snapshot steps must be integers: {steps}",
    )
    _require(len(set(steps.values())) == 1, f"snapshot step mismatch: {steps}")
    expected_trajectories = state.get("total_trajectories_seen")
    checkpoint_trajectories = checkpoint.get("total_trajectories_seen")
    diagnostics_trajectories = (
        steps["runner"] * expected_config.training.trajectories_per_batch
    )
    _require(
        expected_trajectories == checkpoint_trajectories == diagnostics_trajectories,
        "snapshot trajectory counters mismatch",
    )
    complete = state.get("complete")
    _require(isinstance(complete, bool), "runner complete flag must be boolean")
    if not allow_incomplete:
        _require(complete, "formal Stage 5 reporting requires complete=true")
        _require(
            state.get("pending_assignment") is None,
            "formal Stage 5 reporting requires pending_assignment=null",
        )
        _require(
            steps["runner"] == expected_config.training.total_optimizer_steps,
            "complete snapshot step differs from the expected training budget",
        )
    else:
        _require(
            steps["runner"] <= expected_config.training.total_optimizer_steps,
            "incomplete snapshot exceeds the expected training budget",
        )
    source_run = artifact.get("source_run") or {}
    _require(
        source_run.get("hybrid_run_config_sha256") == _sha256_file(run_config_path),
        "Train candidate artifact provenance does not match hybrid_run_config.json",
    )
    _require(
        source_run.get("run_directory_name") == run_config_path.parent.name,
        "Train candidate artifact run identity mismatch",
    )
    records = artifact.get("records")
    _require(isinstance(records, list), "Train candidate artifact records must be a list")
    _require(
        artifact.get("candidate_count") == len(records),
        "Train candidate artifact candidate_count mismatch",
    )
    hashes = [record.get("structural_hash") for record in records if isinstance(record, dict)]
    _require(
        len(hashes) == len(records)
        and hashes == sorted(hashes)
        and len(set(hashes)) == len(hashes),
        "candidate records must have unique sorted structural_hash values",
    )
    _require(
        all(
            record.get("schema") == TRAIN_CANDIDATE_RECORD_SCHEMA
            and record.get("train_evaluation_contract_fingerprint") == contract_fingerprint
            for record in records
        ),
        "candidate record schema or Train contract fingerprint mismatch",
    )
    return {
        "schema": STAGE5_REPORT_DATA_SCHEMA,
        "source_run_id": source_run.get("run_directory_name"),
        "snapshot_steps": steps,
        "fingerprints": {
            "config": expected_fingerprint,
            "reward_provider": reward_fingerprints["run_config"],
            "train_evaluation_contract": artifact.get(
                "train_evaluation_contract_fingerprint"
            ),
        },
        "schema_versions": {
            "runner": run_config.get("schema"),
            "checkpoint": checkpoint.get("schema"),
            "candidate_artifact": artifact.get("schema"),
        },
        "complete": complete,
        "allow_incomplete": allow_incomplete,
        "snapshot_consistency_status": "consistent",
    }


def _numeric_summary(values: Iterable[Any]) -> dict[str, float]:
    series = pd.to_numeric(pd.Series(list(values), dtype="float64"), errors="coerce")
    series = series[np.isfinite(series)]
    if series.empty:
        return {key: math.nan for key in ("first", "median", "last", "min", "max", "mean", "p95")}
    return {
        "first": float(series.iloc[0]),
        "median": float(series.median()),
        "last": float(series.iloc[-1]),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "p95": float(series.quantile(0.95)),
    }


def _training_tables(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    updates = pd.DataFrame(records).sort_values("global_optimizer_step").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for node_count, group in updates.groupby("condition_N", sort=True):
        objective = str(group["objective_kind"].iloc[0])
        metrics = ["tb_loss" if node_count in _EXACT_NS else "variance_loss", "policy_grad_norm", "reward_mean"]
        counters = {
            "successful_updates": int(len(group)),
            "requested_sum": int(group["requested_count"].sum()),
            "accepted_sum": int(group["accepted_count"].sum()),
            "invalid_sum": int(group["invalid_count"].sum()),
            "retry_sum": int(group["retry_count"].sum()),
            "retry_exhausted_sum": int(group["retry_exhausted_count"].sum()),
        }
        for metric in metrics:
            rows.append(
                {
                    "N": int(node_count),
                    "objective_kind": objective,
                    "metric": metric,
                    **_numeric_summary(group[metric]),
                    **counters,
                }
            )
    return updates, pd.DataFrame(rows)


def _validate_candidate(record: Mapping[str, Any]) -> None:
    _require(record.get("schema") == TRAIN_CANDIDATE_RECORD_SCHEMA, "candidate record schema mismatch")
    for key in (
        "structural_hash", "formula", "prefix_token_ids", "node_count", "depth",
        "train_ic", "train_direction", "train_long_ir", "train_barra_ts_corr",
        "train_ic_valid_periods", "train_long_valid_periods", "first_seen", "visit_count",
    ):
        _require(key in record, f"candidate record lacks {key}")
    _require(record["train_direction"] in (-1, 1), "candidate Train direction must be -1 or 1")
    _require(
        isinstance(record["prefix_token_ids"], list)
        and len(record["prefix_token_ids"]) == record["node_count"],
        "candidate prefix length must equal node_count",
    )


def _usage_tables(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(records)
    operator_counts: dict[tuple[str, str], list[int]] = {}
    family_counts: dict[str, list[int]] = {}
    field_counts = {leaf.name: [0, 0] for leaf in LEAVES}
    window_counts = {int(window): [0, 0] for window in WINDOWS}
    complexity_rows: list[dict[str, int]] = []
    for record in records:
        actions = [get_action(value) for value in record["prefix_token_ids"]]
        operator_count = sum(action.arity > 0 for action in actions)
        leaf_count = sum(action.arity == 0 for action in actions)
        complexity_rows.append(
            {
                "node_count": int(record["node_count"]),
                "depth": int(record["depth"]),
                "operator_count": operator_count,
                "leaf_count": leaf_count,
            }
        )
        seen_operators: set[tuple[str, str]] = set()
        seen_families: set[str] = set()
        seen_fields: set[str] = set()
        seen_windows: set[int] = set()
        for action in actions:
            if action.arity == 0:
                field_counts[action.name][0] += 1
                seen_fields.add(action.name)
            else:
                key = (action.category.value, action.name)
                operator_counts.setdefault(key, [0, 0])[0] += 1
                seen_operators.add(key)
                family_counts.setdefault(action.category.value, [0, 0])[0] += 1
                seen_families.add(action.category.value)
            if action.window:
                window_counts[int(action.window)][0] += 1
                seen_windows.add(int(action.window))
        for key in seen_operators:
            operator_counts[key][1] += 1
        for key in seen_families:
            family_counts[key][1] += 1
        for key in seen_fields:
            field_counts[key][1] += 1
        for key in seen_windows:
            window_counts[key][1] += 1

    def rows_from_counts(counts: Mapping[Any, list[int]], label: str) -> pd.DataFrame:
        rows = []
        for key, (occurrences, prevalence) in counts.items():
            row = {
                label: key,
                "occurrence_count": occurrences,
                "candidate_prevalence": prevalence,
                "prevalence_ratio": prevalence / total if total else math.nan,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    operator_rows = []
    for (family, name), (occurrences, prevalence) in operator_counts.items():
        operator_rows.append(
            {
                "operator": name,
                "operator_family": family,
                "usage_level": "operator",
                "occurrence_count": occurrences,
                "candidate_prevalence": prevalence,
                "prevalence_ratio": prevalence / total if total else math.nan,
            }
        )
    for family, (occurrences, prevalence) in family_counts.items():
        operator_rows.append(
            {
                "operator": family,
                "operator_family": family,
                "usage_level": "family",
                "occurrence_count": occurrences,
                "candidate_prevalence": prevalence,
                "prevalence_ratio": prevalence / total if total else math.nan,
            }
        )
    operator_usage = pd.DataFrame(operator_rows).sort_values(
        ["prevalence_ratio", "operator"], ascending=[False, True]
    ).reset_index(drop=True) if operator_rows else pd.DataFrame(columns=["operator", "operator_family", "usage_level", "occurrence_count", "candidate_prevalence", "prevalence_ratio"])
    field_usage = rows_from_counts(field_counts, "field").sort_values("field").reset_index(drop=True)
    window_usage = rows_from_counts(window_counts, "window").sort_values("window").reset_index(drop=True)
    complexity = pd.DataFrame(complexity_rows)
    complexity_summary = pd.DataFrame(
        [
            {"metric": metric, **{key: value for key, value in _numeric_summary(complexity[metric]).items() if key in {"min", "median", "max", "mean", "p95"}}}
            for metric in ("node_count", "depth", "operator_count", "leaf_count")
        ]
    )
    return complexity_summary, operator_usage, field_usage, window_usage


def _candidate_tables(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for record in records:
        _validate_candidate(record)
        first_seen = record["first_seen"]
        rows.append(
            {
                "structural_hash": record["structural_hash"],
                "formula": record["formula"],
                "node_count": int(record["node_count"]),
                "depth": int(record["depth"]),
                "train_ic": float(record["train_ic"]),
                "abs_train_ic": abs(float(record["train_ic"])),
                "train_direction": int(record["train_direction"]),
                "train_long_ir": float(record["train_long_ir"]),
                "train_barra_ts_corr": float(record["train_barra_ts_corr"]),
                "train_ic_valid_periods": int(record["train_ic_valid_periods"]),
                "train_long_valid_periods": int(record["train_long_valid_periods"]),
                "visit_count": int(record["visit_count"]),
                "first_seen_optimizer_step": int(first_seen["optimizer_step"]),
                "first_seen_cycle_index": int(first_seen["cycle_index"]),
            }
        )
    candidates = pd.DataFrame(rows)
    quality_rows = []
    for metric in ("abs_train_ic", "train_long_ir", "train_barra_ts_corr"):
        values = candidates[metric]
        quality_rows.append(
            {
                "metric": metric,
                "min": float(values.min()),
                "25%": float(values.quantile(0.25)),
                "50%": float(values.quantile(0.50)),
                "75%": float(values.quantile(0.75)),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        )
    ic_mask = candidates["abs_train_ic"] > 0.01
    ir_mask = candidates["train_long_ir"] > 0.25
    barra_mask = candidates["train_barra_ts_corr"] < 0.7
    reference = pd.DataFrame(
        [
            {"reference": "abs(train_ic) > 0.01", "candidate_count": int(ic_mask.sum())},
            {"reference": "train_long_ir > 0.25", "candidate_count": int(ir_mask.sum())},
            {"reference": "train_barra_ts_corr < 0.7", "candidate_count": int(barra_mask.sum())},
            {
                "reference": "Train-side descriptive reference count (all three)",
                "candidate_count": int((ic_mask & ir_mask & barra_mask).sum()),
            },
        ]
    )
    by_ic = candidates.sort_values(["abs_train_ic", "structural_hash"], ascending=[False, True]).head(5)
    by_ir = candidates.sort_values(["train_long_ir", "structural_hash"], ascending=[False, True]).head(5)
    examples = pd.concat([by_ic, by_ir], ignore_index=True).drop_duplicates("structural_hash")
    examples = examples.rename(columns={"train_direction": "direction"})[
        [
            "formula", "structural_hash", "node_count", "depth", "train_ic",
            "direction", "train_long_ir", "train_barra_ts_corr",
            "train_ic_valid_periods", "train_long_valid_periods", "visit_count",
        ]
    ]
    return candidates, pd.DataFrame(quality_rows), reference, examples


def _exploration_tables(
    updates: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted = updates.groupby("cycle_index", sort=True)["accepted_count"].sum()
    discovered = candidates.groupby("first_seen_cycle_index", sort=True).size()
    cycles = sorted(set(accepted.index).union(discovered.index))
    by_cycle = pd.DataFrame({"cycle_index": cycles})
    by_cycle["accepted_trajectories"] = by_cycle["cycle_index"].map(accepted).fillna(0).astype(int)
    by_cycle["new_unique_candidates"] = by_cycle["cycle_index"].map(discovered).fillna(0).astype(int)
    by_cycle["cumulative_unique_candidates"] = by_cycle["new_unique_candidates"].cumsum()
    by_cycle["cycle_block_start"] = (by_cycle["cycle_index"] // 10) * 10
    summary_rows = [
        {
            "scope": "overall",
            "accepted_trajectories": int(updates["accepted_count"].sum()),
            "new_unique_candidates": int(len(candidates)),
            "cumulative_unique_candidates_end": int(len(candidates)),
            "uniqueness_ratio": len(candidates) / updates["accepted_count"].sum() if updates["accepted_count"].sum() else math.nan,
        }
    ]
    for block, group in by_cycle.groupby("cycle_block_start", sort=True):
        accepted_count = int(group["accepted_trajectories"].sum())
        new_count = int(group["new_unique_candidates"].sum())
        summary_rows.append(
            {
                "scope": f"cycles_{int(block):03d}_{int(block) + 9:03d}",
                "accepted_trajectories": accepted_count,
                "new_unique_candidates": new_count,
                "cumulative_unique_candidates_end": int(group["cumulative_unique_candidates"].iloc[-1]),
                "uniqueness_ratio": new_count / accepted_count if accepted_count else math.nan,
            }
        )
    accepted_by_n = updates.groupby("condition_N")["accepted_count"].sum()
    unique_by_n = candidates.groupby("node_count").size()
    rows = []
    for node_count in range(1, 16):
        group = updates[updates["condition_N"] == node_count]
        accepted_count = int(accepted_by_n.get(node_count, 0))
        unique_count = int(unique_by_n.get(node_count, 0))
        lpv_values = group.get("unique_terminal_fraction", pd.Series(dtype=float)).dropna()
        rows.append(
            {
                "N": node_count,
                "objective_kind": "exact_tb" if node_count in _EXACT_NS else "log_partition_variance",
                "accepted_trajectories": accepted_count,
                "unique_candidate_count": unique_count,
                "candidate_share": unique_count / len(candidates) if len(candidates) else math.nan,
                "discovery_efficiency": unique_count / accepted_count if accepted_count else math.nan,
                "lpv_batch_unique_terminal_fraction_mean": float(lpv_values.mean()) if not lpv_values.empty else math.nan,
                "lpv_batch_unique_terminal_fraction_median": float(lpv_values.median()) if not lpv_values.empty else math.nan,
                "lpv_batch_unique_terminal_fraction_p95": float(lpv_values.quantile(0.95)) if not lpv_values.empty else math.nan,
            }
        )
    return by_cycle, pd.DataFrame(summary_rows), pd.DataFrame(rows)


def _long_excess_tables(
    records: list[dict[str, Any]],
    *,
    top_k: int,
    min_common_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = []
    for record in records:
        dates = record.get("train_long_excess_dates") or []
        values = record.get("train_long_excess_values") or []
        if dates and int(record["train_long_valid_periods"]) >= min_common_periods:
            _require(len(dates) == len(values), "Train long-excess dates/values length mismatch")
            _require(len(set(dates)) == len(dates), "Train long-excess dates must be unique")
            eligible.append(record)
    eligible.sort(key=lambda row: (-abs(float(row["train_ic"])), str(row["structural_hash"])))
    selected = eligible[:top_k]
    series_by_hash: dict[str, pd.Series] = {}
    selected_rows = []
    hashes: list[str] = []
    for rank, record in enumerate(selected, start=1):
        structural_hash = str(record["structural_hash"])
        hashes.append(structural_hash)
        values = pd.to_numeric(pd.Series(record["train_long_excess_values"]), errors="coerce")
        series = pd.Series(values.to_numpy(), index=pd.Index(record["train_long_excess_dates"], name="date"), name=structural_hash)
        series_by_hash[structural_hash] = series
        for date, value in zip(record["train_long_excess_dates"], values, strict=True):
            selected_rows.append(
                {
                    "rank": rank,
                    "structural_hash": structural_hash,
                    "formula": record["formula"],
                    "abs_train_ic": abs(float(record["train_ic"])),
                    "train_long_valid_periods": int(record["train_long_valid_periods"]),
                    "date": date,
                    "long_excess": float(value) if pd.notna(value) else math.nan,
                }
            )
    matrix = pd.DataFrame(np.nan, index=hashes, columns=hashes, dtype=float)
    for structural_hash, series in series_by_hash.items():
        finite_count = int(np.isfinite(series.to_numpy(dtype=float)).sum())
        if finite_count >= min_common_periods and series.dropna().nunique() > 1:
            matrix.loc[structural_hash, structural_hash] = 1.0
    signed: list[float] = []
    invalid_pairs = 0
    for left_index, left_hash in enumerate(hashes):
        for right_hash in hashes[left_index + 1 :]:
            pair = pd.concat([series_by_hash[left_hash], series_by_hash[right_hash]], axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < min_common_periods or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                invalid_pairs += 1
                continue
            correlation = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))
            if not math.isfinite(correlation):
                invalid_pairs += 1
                continue
            matrix.loc[left_hash, right_hash] = correlation
            matrix.loc[right_hash, left_hash] = correlation
            signed.append(correlation)
    signed_series = pd.Series(signed, dtype=float)
    absolute = signed_series.abs()
    def stat(series: pd.Series, name: str) -> dict[str, float]:
        if series.empty:
            return {f"{name}_{key}": math.nan for key in ("min", "mean", "median", "max")}
        return {
            f"{name}_min": float(series.min()),
            f"{name}_mean": float(series.mean()),
            f"{name}_median": float(series.median()),
            f"{name}_max": float(series.max()),
        }
    summary = pd.DataFrame(
        [
            {
                "selected_candidate_count": len(selected),
                "min_common_periods": min_common_periods,
                **stat(signed_series, "signed_corr"),
                **stat(absolute, "absolute_corr"),
                "valid_pair_count": len(signed),
                "invalid_pair_count": invalid_pairs,
            }
        ]
    )
    return pd.DataFrame(selected_rows), matrix, summary


def _run_summary(
    run_dir: Path,
    run_config: Mapping[str, Any],
    state: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    artifact: Mapping[str, Any],
    expected_config: HybridVarianceGFNConfig,
    snapshot: Mapping[str, Any],
) -> pd.DataFrame:
    completed_steps = int(state["global_optimizer_step"])
    return pd.DataFrame(
        [
            {
                "run_id": run_dir.name,
                "runner_schema": run_config["schema"],
                "checkpoint_schema": checkpoint["schema"],
                "candidate_artifact_schema": artifact["schema"],
                "config_fingerprint": run_config["config_fingerprint"],
                "reward_provider_fingerprint": run_config["reward_provider_fingerprint"],
                "train_evaluation_contract_fingerprint": artifact.get("train_evaluation_contract_fingerprint"),
                "max_depth": expected_config.search_space.max_depth,
                "max_nodes": expected_config.search_space.max_nodes,
                "K": expected_config.training.trajectories_per_batch,
                "N_range": "1..15",
                "objective_partition": "N=1/2 exact-TB; N=3..15 LPV",
                "planned_cycles": expected_config.training.max_cycles,
                "completed_cycles": completed_steps // expected_config.training.optimizer_steps_per_cycle,
                "planned_optimizer_steps": expected_config.training.total_optimizer_steps,
                "completed_optimizer_steps": completed_steps,
                "successful_trajectories": int(state.get("total_trajectories_seen", 0)),
                "unique_candidates": int(artifact["candidate_count"]),
                "created_time": run_config.get("created_at_utc"),
                "last_committed_time": state.get("updated_at_utc") or checkpoint.get("saved_at_utc"),
                "complete": bool(state["complete"]),
                "runner_step": snapshot["snapshot_steps"]["runner"],
                "diagnostics_step": snapshot["snapshot_steps"]["diagnostics"],
                "artifact_step": snapshot["snapshot_steps"]["artifact"],
                "checkpoint_step": snapshot["snapshot_steps"]["checkpoint"],
                "snapshot_consistency_status": snapshot["snapshot_consistency_status"],
            }
        ]
    )


def load_stage5_report_data(
    run_dir: str | Path,
    *,
    expected_config: HybridVarianceGFNConfig,
    allow_incomplete: bool = False,
    correlation_top_k: int = 20,
    correlation_min_common_periods: int = 60,
) -> Stage5ReportDataBundle:
    """Load and standardize one consistent Hybrid Stage 5 snapshot."""

    if not isinstance(expected_config, HybridVarianceGFNConfig):
        raise TypeError("expected_config must be HybridVarianceGFNConfig")
    directory = Path(run_dir).resolve()
    paths = {key: directory / filename for key, filename in _REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    _require(not missing, f"Stage 5 reporting inputs are missing: {missing}")
    run_config = _read_json(paths["run_config"], "run config")
    state_before = _read_json(paths["runner_state"], "runner state")
    diagnostics = _read_diagnostics(paths["diagnostics"])
    diagnostics_step = _validate_diagnostics(
        diagnostics,
        trajectories_per_batch=expected_config.training.trajectories_per_batch,
    )
    artifact = _read_json(paths["candidate_artifact"], "candidate artifact")
    checkpoint = _load_checkpoint_metadata(paths["checkpoint"])
    state_after = _read_json(paths["runner_state"], "runner state")
    _require(
        state_before == state_after,
        "runner state changed while reporting inputs were read; retry after a stable commit",
    )
    snapshot = _snapshot_gate(
        run_config_path=paths["run_config"],
        run_config=run_config,
        state=state_after,
        checkpoint=checkpoint,
        diagnostics_step=diagnostics_step,
        artifact=artifact,
        expected_config=expected_config,
        allow_incomplete=allow_incomplete,
    )
    records = artifact["records"]
    training_updates, training_summary = _training_tables(diagnostics)
    candidates, quality, references, examples = _candidate_tables(records)
    exploration_by_cycle, exploration_summary, exploration_by_n = _exploration_tables(training_updates, candidates)
    complexity, operators, fields, windows = _usage_tables(records)
    selected, correlation, correlation_summary = _long_excess_tables(
        records,
        top_k=correlation_top_k,
        min_common_periods=correlation_min_common_periods,
    )
    warnings = [
        ("factor_value_cross_sectional_correlation", "unavailable", "factor-value matrices are not persisted"),
        ("train_icir", "unavailable", "Train IC time series is not persisted"),
        ("trajectory_reward_quantiles", "unavailable", "only batch reward_mean/std are persisted"),
        ("post_clip_gradient_norm", "unavailable", "policy_grad_norm is pre-clip"),
        ("artifact_loading", "available_with_memory_tradeoff", "v1 artifact is one JSON object; it is fully parsed once and then reduced"),
    ]
    return Stage5ReportDataBundle(
        snapshot_manifest=snapshot,
        run_summary=_run_summary(directory, run_config, state_after, checkpoint, artifact, expected_config, snapshot),
        training_updates=training_updates,
        training_summary_by_n=training_summary,
        exploration_by_cycle=exploration_by_cycle,
        exploration_summary=exploration_summary,
        exploration_by_n=exploration_by_n,
        candidate_summary=candidates,
        candidate_quality_summary=quality,
        quality_reference_counts=references,
        complexity_summary=complexity,
        operator_usage=operators,
        field_usage=fields,
        window_usage=windows,
        selected_long_excess_series=selected,
        long_excess_correlation_matrix=correlation,
        long_excess_correlation_summary=correlation_summary,
        top_candidate_examples=examples,
        availability_and_warnings=pd.DataFrame(warnings, columns=["item", "status", "note"]),
    )


__all__ = [
    "STAGE5_REPORT_DATA_SCHEMA",
    "Stage5ReportDataBundle",
    "load_stage5_report_data",
]
