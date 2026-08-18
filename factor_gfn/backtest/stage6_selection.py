"""Deterministic Stage 6 hard screening and greedy decorrelation.

This module consumes only verified Stage 6 Train/Validation evaluation-cache
records.  It never reconstructs expressions, invokes the interpreter, or reads
OOS data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from .stage6_evaluation import _stable_hash
from .stage6_evaluation_store import EvaluationStore, VerifiedEvaluationRun


STAGE6_SELECTION_CONTRACT_SCHEMA = "factor_gfn.stage6_selection_contract.v1"
STAGE6_SELECTION_RESULT_SCHEMA = "factor_gfn.stage6_selection_result.v1"
STAGE6_SELECTION_MANIFEST_SCHEMA = "factor_gfn.stage6_selection_manifest.v1"
STAGE6_SELECTION_VERSION = "stage6-joint-screen-greedy-decorrelation-v1"

HARD_CONDITION_CODES = (
    "train_abs_ic_gt_0_01",
    "validation_abs_ic_gt_0_01",
    "train_validation_ic_same_sign",
    "train_long_ir_gt_0_25",
    "validation_long_ir_gt_0_25",
    "train_barra_ts_corr_lt_0_7",
)


class Stage6SelectionIntegrityError(RuntimeError):
    """Selection input or an existing immutable artifact failed validation."""


@dataclass(frozen=True, slots=True)
class Stage6SelectionConfig:
    mode: str = "provisional"
    train_abs_ic_min: float = 0.01
    validation_abs_ic_min: float = 0.01
    train_long_ir_min: float = 0.25
    validation_long_ir_min: float = 0.25
    train_barra_ts_corr_max: float = 0.7
    decorrelation_abs_corr_max: float = 0.7
    decorrelation_min_common_periods: int = 60

    def __post_init__(self) -> None:
        if self.mode != "provisional":
            raise ValueError("this implementation batch permits provisional mode only")
        finite_values = (
            self.train_abs_ic_min,
            self.validation_abs_ic_min,
            self.train_long_ir_min,
            self.validation_long_ir_min,
            self.train_barra_ts_corr_max,
            self.decorrelation_abs_corr_max,
        )
        if any(not math.isfinite(value) for value in finite_values):
            raise ValueError("selection thresholds must be finite")
        if self.decorrelation_min_common_periods < 2:
            raise ValueError("minimum common periods must be at least two")
        frozen = (
            self.train_abs_ic_min == 0.01
            and self.validation_abs_ic_min == 0.01
            and self.train_long_ir_min == 0.25
            and self.validation_long_ir_min == 0.25
            and self.train_barra_ts_corr_max == 0.7
            and self.decorrelation_abs_corr_max == 0.7
            and self.decorrelation_min_common_periods == 60
        )
        if not frozen:
            raise ValueError("the provisional Stage 6 selection thresholds are frozen")

    def deterministic_payload(self) -> dict[str, Any]:
        return {
            "schema": STAGE6_SELECTION_CONTRACT_SCHEMA,
            "version": STAGE6_SELECTION_VERSION,
            "mode": self.mode,
            "hard_filter": [
                {
                    "code": HARD_CONDITION_CODES[0],
                    "metric": "abs(train.ic.mean)",
                    "operator": ">",
                    "threshold": self.train_abs_ic_min,
                },
                {
                    "code": HARD_CONDITION_CODES[1],
                    "metric": "abs(validation.ic.mean)",
                    "operator": ">",
                    "threshold": self.validation_abs_ic_min,
                },
                {
                    "code": HARD_CONDITION_CODES[2],
                    "metric": "train.ic.mean * validation.ic.mean",
                    "operator": ">",
                    "threshold": 0.0,
                },
                {
                    "code": HARD_CONDITION_CODES[3],
                    "metric": "train.long.annualized_ir",
                    "operator": ">",
                    "threshold": self.train_long_ir_min,
                },
                {
                    "code": HARD_CONDITION_CODES[4],
                    "metric": "validation.long.annualized_ir",
                    "operator": ">",
                    "threshold": self.validation_long_ir_min,
                },
                {
                    "code": HARD_CONDITION_CODES[5],
                    "metric": "train.barra.max_abs_correlation",
                    "operator": "<",
                    "threshold": self.train_barra_ts_corr_max,
                },
            ],
            "sorting": [
                {"metric": "abs(train.ic.mean)", "order": "descending"},
                {"metric": "structural_hash", "order": "ascending"},
            ],
            "decorrelation": {
                "series": "train.long.excess_series",
                "alignment": "date_intersection_pairwise_finite",
                "correlation": "pearson",
                "minimum_common_periods": self.decorrelation_min_common_periods,
                "reject_operator": "abs(corr) >= threshold",
                "threshold": self.decorrelation_abs_corr_max,
                "retained_scan_order": "greedy_retention_order",
                "early_stop": "first_invalid_or_blocker",
            },
            "validation_pair_correlation": "diagnostic_only",
            "completed_invalid_policy": "evaluation_ineligible_not_hard_condition",
            "oos": "forbidden",
        }

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.deterministic_payload())


@dataclass(frozen=True, slots=True)
class Stage6SelectionResult:
    selection_fingerprint: str
    selection_manifest_fingerprint: str
    evaluation_run_id: str
    output_directory: Path
    hard_filter_pass_count: int
    hard_filter_fail_count: int
    evaluation_ineligible_count: int
    retained_count: int
    rejected_by_correlation_count: int
    decorrelation_invalid_count: int
    manifest: Mapping[str, Any]


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise Stage6SelectionIntegrityError(
                f"immutable selection artifact changed: {path.name}"
            )
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, (_json_text(value) + "\n").encode("utf-8"))


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(_json_text(dict(record)) + "\n" for record in records)
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _nested(record: Mapping[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _hard_filter_record(item: Mapping[str, Any]) -> dict[str, Any]:
    result = item["result"]
    train_ic = _finite_float(_nested(result, "train", "ic", "mean"))
    validation_ic = _finite_float(_nested(result, "validation", "ic", "mean"))
    train_long_ir = _finite_float(
        _nested(result, "train", "long", "annualized_ir")
    )
    validation_long_ir = _finite_float(
        _nested(result, "validation", "long", "annualized_ir")
    )
    train_barra = _finite_float(
        _nested(result, "train", "barra", "max_abs_correlation")
    )
    metrics = {
        "train_ic": train_ic,
        "validation_ic": validation_ic,
        "train_long_ir": train_long_ir,
        "validation_long_ir": validation_long_ir,
        "train_barra_ts_corr": train_barra,
    }
    config = item["_selection_config"]
    checks: dict[str, bool | None] = {
        HARD_CONDITION_CODES[0]: train_ic is not None
        and abs(train_ic) > config.train_abs_ic_min,
        HARD_CONDITION_CODES[1]: validation_ic is not None
        and abs(validation_ic) > config.validation_abs_ic_min,
        HARD_CONDITION_CODES[2]: train_ic is not None
        and validation_ic is not None
        and train_ic * validation_ic > 0.0,
        HARD_CONDITION_CODES[3]: train_long_ir is not None
        and train_long_ir > config.train_long_ir_min,
        HARD_CONDITION_CODES[4]: validation_long_ir is not None
        and validation_long_ir > config.validation_long_ir_min,
        HARD_CONDITION_CODES[5]: train_barra is not None
        and train_barra < config.train_barra_ts_corr_max,
    }
    prefilter = _nested(result, "train", "train_prefilter")
    prefilter_failed = (
        isinstance(prefilter, Mapping)
        and prefilter.get("status") == "train_prefilter_failed"
    )
    if prefilter_failed:
        # Validation was deliberately not evaluated because at least one frozen
        # Train-only necessary condition already proves final-screen failure.
        # Do not misreport unavailable Validation conditions as observed fails.
        checks[HARD_CONDITION_CODES[1]] = None
        checks[HARD_CONDITION_CODES[2]] = None
        checks[HARD_CONDITION_CODES[4]] = None
        declared = prefilter.get("failed_conditions", [])
        failed = [
            code
            for code in HARD_CONDITION_CODES
            if code in declared and checks.get(code) is False
        ]
        if not failed:
            raise Stage6SelectionIntegrityError(
                "train_prefilter_failed result lacks a failed Train condition"
            )
    else:
        failed = [code for code in HARD_CONDITION_CODES if checks[code] is False]
    eligible = result.get("status") == "completed" and not prefilter_failed
    expression = result.get("expression")
    if not isinstance(expression, Mapping):
        raise Stage6SelectionIntegrityError("verified result lacks expression identity")
    return {
        "schema": STAGE6_SELECTION_RESULT_SCHEMA,
        "evaluation_ordinal": int(item["ordinal"]),
        "structural_hash": str(item["structural_hash"]),
        "result_fingerprint": str(item["result_fingerprint"]),
        "expression": dict(expression),
        "source_identity": dict(result.get("source_identity", {})),
        "evaluation_status": result.get("status"),
        "evaluation_ineligible": not eligible,
        "evaluation_invalid_reasons": list(result.get("invalid_reasons", [])),
        "train_prefilter_status": (
            prefilter.get("status") if isinstance(prefilter, Mapping) else None
        ),
        "validation_evaluated": not prefilter_failed,
        "metrics": metrics,
        "condition_results": checks,
        "failed_conditions": failed,
        "hard_filter_pass": eligible and not failed,
        "train_direction": result.get("train_direction"),
    }


def _series_mapping(result: Mapping[str, Any], split: str) -> tuple[dict[str, float], str | None]:
    series = _nested(result, split, "long", "excess_series")
    if not isinstance(series, Mapping):
        return {}, "missing_excess_series"
    dates = series.get("dates")
    values = series.get("values")
    if not isinstance(dates, list) or not isinstance(values, list):
        return {}, "missing_excess_series_values"
    if len(dates) != len(values):
        return {}, "excess_series_length_mismatch"
    mapping: dict[str, float] = {}
    seen_dates: set[str] = set()
    for date, value in zip(dates, values, strict=True):
        if not isinstance(date, str) or not date:
            return {}, "invalid_excess_series_date"
        if date in seen_dates:
            return {}, "duplicate_excess_series_date"
        seen_dates.add(date)
        finite = _finite_float(value)
        if finite is not None:
            mapping[date] = finite
    return mapping, None


def _pair_correlation(
    left_result: Mapping[str, Any],
    right_result: Mapping[str, Any],
    split: str,
    minimum_common_periods: int,
) -> dict[str, Any]:
    left, left_error = _series_mapping(left_result, split)
    right, right_error = _series_mapping(right_result, split)
    error = left_error or right_error
    if error is not None:
        return {
            "status": "correlation_unavailable",
            "correlation": None,
            "common_valid_periods": 0,
            "failure_reason": error,
        }
    common_dates = sorted(left.keys() & right.keys())
    common = len(common_dates)
    correlation: float | None = None
    failure_reason: str | None = None
    if common >= 2:
        left_values = np.asarray([left[date] for date in common_dates], dtype=np.float64)
        right_values = np.asarray([right[date] for date in common_dates], dtype=np.float64)
        left_centered = left_values - left_values.mean()
        right_centered = right_values - right_values.mean()
        denominator = math.sqrt(
            float(np.dot(left_centered, left_centered))
            * float(np.dot(right_centered, right_centered))
        )
        if denominator > 0.0 and math.isfinite(denominator):
            observed = float(np.dot(left_centered, right_centered) / denominator)
            if math.isfinite(observed):
                correlation = min(1.0, max(-1.0, observed))
            else:
                failure_reason = "nonfinite_correlation"
        else:
            failure_reason = "zero_variance_or_nonfinite_denominator"
    else:
        failure_reason = "fewer_than_two_common_periods"
    if common < minimum_common_periods:
        status = "insufficient_common_periods"
        failure_reason = "common_valid_periods_below_minimum"
    elif correlation is None:
        status = "correlation_unavailable"
    else:
        status = "valid"
    return {
        "status": status,
        "correlation": correlation,
        "common_valid_periods": common,
        "failure_reason": failure_reason,
    }


def _greedy_decorrelate(
    sorted_rows: Sequence[Mapping[str, Any]],
    result_by_hash: Mapping[str, Mapping[str, Any]],
    config: Stage6SelectionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for rank, hard_row in enumerate(sorted_rows, start=1):
        structural_hash = str(hard_row["structural_hash"])
        candidate_result = result_by_hash[structural_hash]
        trace: list[dict[str, Any]] = []
        status = "retained"
        blocker: str | None = None
        blocking_corr: float | None = None
        invalid_reason: str | None = None
        invalid_peer: str | None = None
        invalid_common: int | None = None
        for retained_row in retained:
            retained_hash = str(retained_row["structural_hash"])
            retained_result = result_by_hash[retained_hash]
            train = _pair_correlation(
                candidate_result,
                retained_result,
                "train",
                config.decorrelation_min_common_periods,
            )
            validation = _pair_correlation(
                candidate_result,
                retained_result,
                "validation",
                config.decorrelation_min_common_periods,
            )
            if train["status"] != "valid":
                decision = "decorrelation_invalid"
            elif abs(float(train["correlation"])) >= config.decorrelation_abs_corr_max:
                decision = "rejected_by_correlation"
            else:
                decision = "continue"
            trace.append(
                {
                    "retained_structural_hash": retained_hash,
                    "train": train,
                    "validation": validation,
                    "decision": decision,
                }
            )
            if decision == "decorrelation_invalid":
                status = decision
                invalid_reason = str(train["failure_reason"])
                invalid_peer = retained_hash
                invalid_common = int(train["common_valid_periods"])
                break
            if decision == "rejected_by_correlation":
                status = decision
                blocker = retained_hash
                blocking_corr = float(train["correlation"])
                break
        outcome = {
            "schema": STAGE6_SELECTION_RESULT_SCHEMA,
            "sorted_rank": rank,
            "structural_hash": structural_hash,
            "result_fingerprint": hard_row["result_fingerprint"],
            "abs_train_ic": abs(float(hard_row["metrics"]["train_ic"])),
            "greedy_retained": status == "retained",
            "decorrelation_status": status,
            "decorrelation_invalid": status == "decorrelation_invalid",
            "blocked_by_structural_hash": blocker,
            "blocking_corr": blocking_corr,
            "decorrelation_failure_reason": invalid_reason,
            "compared_with_structural_hash": invalid_peer,
            "common_valid_periods": invalid_common,
            "comparison_trace": trace,
        }
        outcomes.append(outcome)
        if status == "retained":
            retained.append(
                {
                    **outcome,
                    "expression": dict(hard_row["expression"]),
                    "source_identity": dict(hard_row["source_identity"]),
                    "train_direction": hard_row["train_direction"],
                    "metrics": dict(hard_row["metrics"]),
                }
            )
    return outcomes, retained


def _select_verified_run(
    verified: VerifiedEvaluationRun,
    config: Stage6SelectionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not verified.records:
        raise Stage6SelectionIntegrityError("selection input is empty")
    result_by_hash: dict[str, Mapping[str, Any]] = {}
    hard_rows: list[dict[str, Any]] = []
    for stored in verified.records:
        structural_hash = str(stored["structural_hash"])
        result = stored["result"]
        if not isinstance(result, Mapping):
            raise Stage6SelectionIntegrityError("verified cached result is not a mapping")
        if "oos" in result:
            raise Stage6SelectionIntegrityError("OOS payload is forbidden in selection")
        if structural_hash in result_by_hash:
            raise Stage6SelectionIntegrityError("duplicate structural hash in selection input")
        result_by_hash[structural_hash] = result
        hard_rows.append(
            _hard_filter_record({**dict(stored), "_selection_config": config})
        )
    passed = [row for row in hard_rows if row["hard_filter_pass"]]
    passed.sort(
        key=lambda row: (
            -abs(float(row["metrics"]["train_ic"])),
            str(row["structural_hash"]),
        )
    )
    greedy, retained = _greedy_decorrelate(passed, result_by_hash, config)
    return hard_rows, greedy, retained


def run_stage6_selection(
    store: EvaluationStore,
    evaluation_run_id: str,
    output_root: str | Path,
    *,
    config: Stage6SelectionConfig | None = None,
) -> Stage6SelectionResult:
    """Run and materialize deterministic provisional Stage 6 selection."""

    selection_config = config or Stage6SelectionConfig()
    verified = store.load_verified_run_results(evaluation_run_id)
    hard_rows, greedy_rows, alpha_pool = _select_verified_run(
        verified, selection_config
    )
    input_identity = [
        {
            "ordinal": int(record["ordinal"]),
            "structural_hash": str(record["structural_hash"]),
            "cache_key": str(record["cache_key"]),
            "result_fingerprint": str(record["result_fingerprint"]),
        }
        for record in verified.records
    ]
    deterministic = {
        "schema": STAGE6_SELECTION_RESULT_SCHEMA,
        "selection_contract": selection_config.deterministic_payload(),
        "evaluation_run": {
            "run_id": evaluation_run_id,
            "context_fingerprint": verified.manifest["context_fingerprint"],
            "evaluation_contract_fingerprint": verified.manifest[
                "evaluation_contract_fingerprint"
            ],
            "ordered_result_set_fingerprint": verified.ordered_result_set_fingerprint,
            "ordered_results": input_identity,
        },
        "hard_filter_results": hard_rows,
        "greedy_decorrelation_results": greedy_rows,
        "alpha_pool": alpha_pool,
    }
    selection_fingerprint = _stable_hash(deterministic)
    counts = {
        "input_candidates": len(hard_rows),
        "evaluation_ineligible": sum(
            bool(row["evaluation_ineligible"]) for row in hard_rows
        ),
        "hard_filter_pass": sum(bool(row["hard_filter_pass"]) for row in hard_rows),
        "hard_filter_fail": sum(not bool(row["hard_filter_pass"]) for row in hard_rows),
        "retained": len(alpha_pool),
        "rejected_by_correlation": sum(
            row["decorrelation_status"] == "rejected_by_correlation"
            for row in greedy_rows
        ),
        "decorrelation_invalid": sum(
            row["decorrelation_status"] == "decorrelation_invalid"
            for row in greedy_rows
        ),
    }
    summary = {
        "schema": STAGE6_SELECTION_RESULT_SCHEMA,
        "selection_fingerprint": selection_fingerprint,
        "mode": selection_config.mode,
        "evaluation_run_id": evaluation_run_id,
        "counts": counts,
        "oos": "not_loaded_not_evaluated",
    }
    artifact_records = {
        "hard_filter_results.jsonl": _stable_hash(hard_rows),
        "greedy_decorrelation_results.jsonl": _stable_hash(greedy_rows),
        "alpha_pool.jsonl": _stable_hash(alpha_pool),
        "selection_summary.json": _stable_hash(summary),
    }
    manifest_core = {
        "schema": STAGE6_SELECTION_MANIFEST_SCHEMA,
        "version": STAGE6_SELECTION_VERSION,
        "selection_fingerprint": selection_fingerprint,
        "selection_contract_fingerprint": selection_config.fingerprint,
        "mode": selection_config.mode,
        "evaluation_run_id": evaluation_run_id,
        "context_fingerprint": verified.manifest["context_fingerprint"],
        "evaluation_contract_fingerprint": verified.manifest[
            "evaluation_contract_fingerprint"
        ],
        "ordered_result_set_fingerprint": verified.ordered_result_set_fingerprint,
        "artifact_record_fingerprints": artifact_records,
        "counts": counts,
        "oos": "not_loaded_not_evaluated",
    }
    manifest_fingerprint = _stable_hash(manifest_core)
    manifest = {
        **manifest_core,
        "selection_manifest_fingerprint": manifest_fingerprint,
    }
    output_directory = Path(output_root).resolve() / selection_fingerprint
    _write_jsonl(output_directory / "hard_filter_results.jsonl", hard_rows)
    _write_jsonl(
        output_directory / "greedy_decorrelation_results.jsonl", greedy_rows
    )
    _write_jsonl(output_directory / "alpha_pool.jsonl", alpha_pool)
    _write_json(output_directory / "selection_summary.json", summary)
    _write_json(output_directory / "selection_manifest.json", manifest)
    return Stage6SelectionResult(
        selection_fingerprint=selection_fingerprint,
        selection_manifest_fingerprint=manifest_fingerprint,
        evaluation_run_id=evaluation_run_id,
        output_directory=output_directory,
        hard_filter_pass_count=counts["hard_filter_pass"],
        hard_filter_fail_count=counts["hard_filter_fail"],
        evaluation_ineligible_count=counts["evaluation_ineligible"],
        retained_count=counts["retained"],
        rejected_by_correlation_count=counts["rejected_by_correlation"],
        decorrelation_invalid_count=counts["decorrelation_invalid"],
        manifest=MappingProxyType(manifest),
    )


__all__ = [
    "HARD_CONDITION_CODES",
    "STAGE6_SELECTION_CONTRACT_SCHEMA",
    "STAGE6_SELECTION_MANIFEST_SCHEMA",
    "STAGE6_SELECTION_RESULT_SCHEMA",
    "STAGE6_SELECTION_VERSION",
    "Stage6SelectionConfig",
    "Stage6SelectionIntegrityError",
    "Stage6SelectionResult",
    "run_stage6_selection",
]
