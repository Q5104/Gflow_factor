"""Fail-closed Stage 6 screening-report adapter.

The adapter reads frozen Stage 6 artifacts and emits renderer-ready tables. It
never evaluates an expression, runs selection, or changes the provisional pool.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from factor_gfn.backtest.candidate_import import CANDIDATE_IMPORT_MANIFEST_SCHEMA
from factor_gfn.backtest.expression_compatibility import (
    EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA,
)
from factor_gfn.backtest.sources import SOURCE_SET_SCHEMA, SOURCE_SNAPSHOT_SCHEMA
from factor_gfn.backtest.stage6_evaluation import _stable_hash
from factor_gfn.backtest.stage6_selection import (
    HARD_CONDITION_CODES,
    Stage6SelectionConfig,
)
from factor_gfn.backtest.stage6_survivor_enrichment import (
    STAGE6_ENRICHED_SELECTION_MANIFEST_SCHEMA,
)
from factor_gfn.backtest.stage6_two_phase_pipeline import (
    STAGE6_TRAIN_ENTRY_SCHEMA,
    STAGE6_TRAIN_PASS_MANIFEST_SCHEMA,
    STAGE6_TRAIN_PREPARATION_SCOPE,
    STAGE6_VALIDATION_ENTRY_SCHEMA,
    STAGE6_VALIDATION_SCOPE,
)
from factor_gfn.grammar import LEAVES, WINDOWS, get_action


STAGE6_REPORT_DATA_SCHEMA = "factor_gfn.reporting.stage6_data.v2"
PAIR_AUDIT_VERSION = "stage6-reporting-pair-audit-v1"
UNIVERSES = (
    "stage5_source",
    "train_prefilter_pass",
    "validation_evaluated",
    "hard_filter_pass",
    "decorrelation_input",
    "provisional_pool",
    "frozen_order_top100",
)
_VALIDATION_CODES = {
    "validation_abs_ic_gt_0_01",
    "train_validation_ic_same_sign",
    "validation_long_ir_gt_0_25",
}


@dataclass(frozen=True)
class Stage6ReportDataBundle:
    snapshot_manifest: dict[str, Any]
    source_candidates: pd.DataFrame
    train_prefilter_results: pd.DataFrame
    validation_candidate_metrics: pd.DataFrame
    hard_filter_results: pd.DataFrame
    funnel_summary: pd.DataFrame
    hard_filter_condition_summary: pd.DataFrame
    failure_combinations: pd.DataFrame
    stability_summary: pd.DataFrame
    before_after_quality_summary: pd.DataFrame
    decorrelation_input: pd.DataFrame
    decorrelation_outcomes: pd.DataFrame
    effective_train_long_excess: pd.DataFrame
    before_top30_correlation: pd.DataFrame
    after_top20_correlation: pd.DataFrame
    decorrelation_pair_summary: pd.DataFrame
    greedy_pair_audit: pd.DataFrame
    provisional_factor_pool: pd.DataFrame
    top100_candidate_metrics: pd.DataFrame
    top100_quality_summary: pd.DataFrame
    complexity_summary: pd.DataFrame
    top_candidate_examples: pd.DataFrame
    structure_shift_summary: pd.DataFrame
    operator_prevalence_shift: pd.DataFrame
    field_prevalence_shift: pd.DataFrame
    window_prevalence_shift: pd.DataFrame
    availability_and_warnings: pd.DataFrame


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Stage 6 JSON: {path}") from error
    _require(isinstance(value, dict), f"Stage 6 JSON must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                _require(bool(line.strip()), f"blank Stage 6 JSONL row: {path}:{number}")
                row = json.loads(line)
                _require(isinstance(row, dict), f"invalid Stage 6 JSONL row: {path}:{number}")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Stage 6 JSONL: {path}") from error
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact(directory: Path, name: str, metadata: Mapping[str, Any]) -> Path:
    path = directory / name
    _require(path.is_file(), f"Stage 6 artifact is missing: {path}")
    _require(path.stat().st_size == int(metadata.get("size_bytes", -1)), f"Stage 6 artifact size mismatch: {name}")
    _require(_sha256_file(path) == metadata.get("sha256"), f"Stage 6 artifact SHA mismatch: {name}")
    return path


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _hashes(rows: Iterable[Mapping[str, Any]], key: str = "structural_hash") -> list[str]:
    values = [str(row[key]) for row in rows]
    _require(len(values) == len(set(values)), f"duplicate {key} in Stage 6 universe")
    return values


def _metric_row(stored: Mapping[str, Any]) -> dict[str, Any]:
    result = stored["result"]
    expression = result["expression"]
    train_ic = _finite(result.get("train", {}).get("ic", {}).get("mean"))
    validation_ic = _finite(result.get("validation", {}).get("ic", {}).get("mean"))
    return {
        "structural_hash": str(stored["structural_hash"]),
        "formula": expression.get("formula"),
        "prefix_token_ids": expression.get("prefix_token_ids"),
        "node_count": expression.get("node_count"),
        "depth": expression.get("depth"),
        "train_ic": train_ic,
        "validation_ic": validation_ic,
        "abs_train_ic": abs(train_ic) if train_ic is not None else math.nan,
        "abs_validation_ic": abs(validation_ic) if validation_ic is not None else math.nan,
        "train_direction": result.get("train_direction"),
        "train_long_ir": _finite(result.get("train", {}).get("long", {}).get("annualized_ir")),
        "validation_long_ir": _finite(result.get("validation", {}).get("long", {}).get("annualized_ir")),
        "train_barra_ts_corr": _finite(result.get("train", {}).get("barra", {}).get("max_abs_correlation")),
        "result_fingerprint": stored.get("result_fingerprint"),
        "source_identity": result.get("source_identity", {}),
    }


def _series_from_result(result: Mapping[str, Any]) -> tuple[dict[str, float], str | None]:
    series = result.get("train", {}).get("long", {}).get("excess_series")
    if not isinstance(series, Mapping):
        return {}, "missing_excess_series"
    return _series_from_payload(series)


def _series_from_payload(series: Mapping[str, Any]) -> tuple[dict[str, float], str | None]:
    dates, values = series.get("dates"), series.get("values")
    if not isinstance(dates, list) or not isinstance(values, list) or len(dates) != len(values):
        return {}, "missing_or_misaligned_excess_series"
    output: dict[str, float] = {}
    for date, value in zip(dates, values, strict=True):
        if not isinstance(date, str) or not date or date in output:
            return {}, "invalid_or_duplicate_excess_series_date"
        number = _finite(value)
        if number is not None:
            output[date] = number
    return output, None


def pair_correlation(
    left: Mapping[str, float], right: Mapping[str, float], *, minimum_common_periods: int = 60,
) -> dict[str, Any]:
    common_dates = sorted(set(left) & set(right))
    common = len(common_dates)
    if common < minimum_common_periods:
        return {"status": "insufficient_common_periods", "correlation": None, "common_valid_periods": common, "failure_reason": "common_valid_periods_below_minimum"}
    left_values = np.asarray([left[date] for date in common_dates], dtype=np.float64)
    right_values = np.asarray([right[date] for date in common_dates], dtype=np.float64)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    left_values, right_values = left_values[finite], right_values[finite]
    common = int(finite.sum())
    if common < minimum_common_periods:
        return {"status": "insufficient_common_periods", "correlation": None, "common_valid_periods": common, "failure_reason": "common_valid_periods_below_minimum"}
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = math.sqrt(float(np.dot(left_centered, left_centered)) * float(np.dot(right_centered, right_centered)))
    if not math.isfinite(denominator) or denominator <= 0:
        return {"status": "correlation_unavailable", "correlation": None, "common_valid_periods": common, "failure_reason": "zero_variance_or_nonfinite_denominator"}
    corr = float(np.dot(left_centered, right_centered) / denominator)
    if not math.isfinite(corr):
        return {"status": "correlation_unavailable", "correlation": None, "common_valid_periods": common, "failure_reason": "nonfinite_correlation"}
    return {"status": "valid", "correlation": min(1.0, max(-1.0, corr)), "common_valid_periods": common, "failure_reason": None}


def _correlation_matrix(hashes: Sequence[str], series: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=list(hashes), columns=list(hashes), dtype=float)
    for index, left in enumerate(hashes):
        if left in series:
            matrix.loc[left, left] = 1.0
        for right in hashes[index + 1:]:
            if left not in series or right not in series:
                continue
            pair = pair_correlation(series[left], series[right])
            if pair["status"] == "valid":
                matrix.loc[left, right] = matrix.loc[right, left] = pair["correlation"]
    matrix.index.name = "structural_hash"
    return matrix


def _pair_summary(label: str, hashes: Sequence[str], series: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    values: list[float] = []
    invalid = 0
    for index, left in enumerate(hashes):
        for right in hashes[index + 1:]:
            pair = pair_correlation(series.get(left, {}), series.get(right, {}))
            if pair["status"] == "valid":
                values.append(abs(float(pair["correlation"])))
            else:
                invalid += 1
    array = np.asarray(values, dtype=float)
    return {
        "universe": label,
        "candidate_count": len(hashes),
        "valid_pairs": len(values),
        "invalid_pairs": invalid,
        "mean_abs_corr": float(array.mean()) if len(array) else math.nan,
        "median_abs_corr": float(np.median(array)) if len(array) else math.nan,
        "max_abs_corr": float(array.max()) if len(array) else math.nan,
    }


def _rank_hashes(hashes: Iterable[str], hard_by_hash: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(
        (str(value) for value in hashes),
        key=lambda value: (-abs(float(hard_by_hash[value]["metrics"]["train_ic"])), value),
    )


def _greedy_audit(greedy: Sequence[Mapping[str, Any]], series: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previously_retained: list[str] = []
    for row in sorted(greedy, key=lambda value: int(value["sorted_rank"])):
        structural_hash = str(row["structural_hash"])
        valid: list[float] = []
        invalid = 0
        for peer in previously_retained:
            pair = pair_correlation(series.get(structural_hash, {}), series.get(peer, {}))
            if pair["status"] == "valid":
                valid.append(abs(float(pair["correlation"])))
            else:
                invalid += 1
        status = str(row["decorrelation_status"])
        rows.append({
            "sorted_rank": int(row["sorted_rank"]),
            "structural_hash": structural_hash,
            "persisted_decorrelation_status": status,
            "blocked_by_structural_hash": row.get("blocked_by_structural_hash"),
            "persisted_blocking_corr": row.get("blocking_corr"),
            "previous_retained_count": len(previously_retained),
            "valid_pair_count": len(valid),
            "invalid_pair_count": invalid,
            "max_abs_valid_corr_to_previous_retained": max(valid) if valid else math.nan,
        })
        if status == "retained":
            previously_retained.append(structural_hash)
    return pd.DataFrame(rows)


def _summary(values: pd.Series) -> dict[str, float]:
    data = pd.to_numeric(values, errors="coerce").dropna()
    if data.empty:
        return {key: math.nan for key in ("mean", "median", "q25", "q75")}
    return {"mean": float(data.mean()), "median": float(data.median()), "q25": float(data.quantile(.25)), "q75": float(data.quantile(.75))}


def _extended_summary(values: pd.Series) -> dict[str, float | int]:
    data = pd.to_numeric(values, errors="coerce").dropna()
    if data.empty:
        return {
            "n": 0,
            **{key: math.nan for key in ("min", "mean", "median", "q25", "q75", "max")},
        }
    return {
        "n": int(len(data)),
        "min": float(data.min()),
        "mean": float(data.mean()),
        "median": float(data.median()),
        "q25": float(data.quantile(.25)),
        "q75": float(data.quantile(.75)),
        "max": float(data.max()),
    }


def _token_prevalence(frame: pd.DataFrame, universe: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(frame)
    op_occ: Counter[str] = Counter(); op_seen: Counter[str] = Counter()
    field_occ: Counter[str] = Counter(); field_seen: Counter[str] = Counter()
    window_occ: Counter[int] = Counter(); window_seen: Counter[int] = Counter()
    complexity: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        actions = [get_action(int(token)) for token in row["prefix_token_ids"]]
        seen_op: set[str] = set(); seen_field: set[str] = set(); seen_window: set[int] = set()
        for action in actions:
            if action.arity:
                op_occ[action.name] += 1; seen_op.add(action.name)
            else:
                field_occ[action.name] += 1; seen_field.add(action.name)
            if action.window:
                window_occ[int(action.window)] += 1; seen_window.add(int(action.window))
        op_seen.update(seen_op); field_seen.update(seen_field); window_seen.update(seen_window)
        complexity.append({"universe": universe, "structural_hash": row["structural_hash"], "node_count": int(row["node_count"]), "depth": int(row["depth"])})
    def table(keys: Iterable[Any], occurrence: Counter, prevalence: Counter, name: str) -> pd.DataFrame:
        return pd.DataFrame(
            [{name: key, "universe": universe, "candidate_count": total, "occurrence_count": occurrence[key], "candidate_prevalence": prevalence[key], "prevalence_ratio": prevalence[key] / total if total else math.nan} for key in keys],
            columns=[name, "universe", "candidate_count", "occurrence_count", "candidate_prevalence", "prevalence_ratio"],
        )
    return (pd.DataFrame(complexity), table(sorted(op_occ), op_occ, op_seen, "operator"), table([leaf.name for leaf in LEAVES], field_occ, field_seen, "field"), table([int(value) for value in WINDOWS], window_occ, window_seen, "window"))


def build_stage6_report_data(
    *, snapshot_manifest: Mapping[str, Any], source_candidates: Sequence[Mapping[str, Any]],
    train_prefilter_results: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]],
    hard_filter_results: Sequence[Mapping[str, Any]], greedy_results: Sequence[Mapping[str, Any]],
    alpha_pool: Sequence[Mapping[str, Any]], enrichment_results: Sequence[Mapping[str, Any]],
) -> Stage6ReportDataBundle:
    """Build reporting tables from already verified in-memory Stage 6 payloads."""

    source_hashes = _hashes(source_candidates, "current_structural_hash")
    train_rows = list(train_prefilter_results); validation_rows = list(validation_records)
    hard_rows = list(hard_filter_results); greedy_rows = list(greedy_results); pool_rows = list(alpha_pool)
    train_pass = [str(row["structural_hash"]) for row in train_rows if row.get("status") == "train_prefilter_passed"]
    validation_hashes = _hashes(validation_rows)
    hard_pass = [str(row["structural_hash"]) for row in hard_rows if row.get("hard_filter_pass") is True]
    greedy_hashes = _hashes(greedy_rows)
    retained = [str(row["structural_hash"]) for row in greedy_rows if row.get("decorrelation_status") == "retained"]
    pool_hashes = _hashes(pool_rows)
    _require(set(train_pass) == set(validation_hashes), "Train-pass set differs from Validation evaluated set")
    _require(set(hard_pass) == set(greedy_hashes), "hard-filter pass set differs from decorrelation input set")
    _require(set(retained) == set(pool_hashes), "persisted retained set differs from provisional pool")

    metrics = pd.DataFrame([_metric_row(row) for row in validation_rows])
    hard_frame = pd.DataFrame([{**row, **row.get("metrics", {})} for row in hard_rows])
    greedy_frame = pd.DataFrame(greedy_rows)
    source_frame = pd.DataFrame(source_candidates)
    train_frame = pd.DataFrame(train_rows)

    counts = [
        len(source_hashes), len(train_pass), len(validation_hashes), len(hard_pass),
        len(greedy_hashes), len(pool_hashes), min(100, len(pool_hashes)),
    ]
    funnel_rows = []
    for index, (stage, count) in enumerate(zip(UNIVERSES, counts, strict=True)):
        previous = counts[index - 1] if index else count
        funnel_rows.append({"stage": stage, "input_count": previous, "remaining_count": count, "rejected_count": previous - count, "retention_from_previous": count / previous if previous else math.nan, "retention_from_source": count / counts[0] if counts[0] else math.nan})
    funnel = pd.DataFrame(funnel_rows)

    prefilter_by_hash = {str(row["structural_hash"]): row for row in train_rows}
    hard_by_hash = {str(row["structural_hash"]): row for row in hard_rows}
    condition_rows = []
    for code in HARD_CONDITION_CODES:
        observed: list[bool] = []
        for structural_hash in source_hashes:
            if code in _VALIDATION_CODES:
                result = hard_by_hash.get(structural_hash, {}).get("condition_results", {}).get(code)
            else:
                result = prefilter_by_hash.get(structural_hash, {}).get("condition_results", {}).get(code)
                if result is None:
                    result = hard_by_hash.get(structural_hash, {}).get("condition_results", {}).get(code)
            if isinstance(result, bool): observed.append(result)
        condition_rows.append({"condition": code, "universe": "stage5_source", "observed_count": len(observed), "pass_count": sum(observed), "fail_count": len(observed) - sum(observed), "not_evaluated_count": len(source_hashes) - len(observed), "pass_rate_among_observed": sum(observed) / len(observed) if observed else math.nan})
    condition_summary = pd.DataFrame(condition_rows)

    combinations: Counter[tuple[str, str]] = Counter()
    for row in train_rows:
        if row.get("status") != "train_prefilter_passed":
            codes = [code for code in HARD_CONDITION_CODES if code in row.get("failed_conditions", [])]
            combinations[("train_prefilter_exit", " + ".join(codes) or "NO_RECORDED_FAILURE")] += 1
    for row in hard_rows:
        if not row.get("hard_filter_pass"):
            codes = [code for code in HARD_CONDITION_CODES if code in row.get("failed_conditions", [])]
            combinations[("six_condition_observed", " + ".join(codes) or "NO_RECORDED_FAILURE")] += 1
    failure_output = []
    for scope in ("train_prefilter_exit", "six_condition_observed"):
        scoped = sorted(((combo, count) for (item_scope, combo), count in combinations.items() if item_scope == scope), key=lambda item: (-item[1], item[0]))
        total = sum(count for _, count in scoped)
        top, rest = scoped[:15], scoped[15:]
        for combo, count in top:
            failure_output.append({"observation_scope": scope, "combination": combo, "count": count, "share": count / total if total else math.nan})
        if rest:
            count = sum(value for _, value in rest); failure_output.append({"observation_scope": scope, "combination": "OTHER", "count": count, "share": count / total})
    failures = pd.DataFrame(failure_output, columns=["observation_scope", "combination", "count", "share"])

    stability_rows = []
    for metric in ("train_ic", "validation_ic", "abs_train_ic", "abs_validation_ic", "train_long_ir", "validation_long_ir"):
        stats = _summary(metrics[metric]); stability_rows.append({"section": "level", "metric": metric, **stats})
    same_sign = (metrics["train_ic"] * metrics["validation_ic"] > 0)
    stability_rows.extend([
        {"section": "ic_sign", "metric": "same_sign", "count": int(same_sign.sum()), "ratio": float(same_sign.mean()) if len(same_sign) else math.nan},
        {"section": "change", "metric": "ic_strength_change", **_summary(metrics["abs_validation_ic"] - metrics["abs_train_ic"])},
        {"section": "change", "metric": "long_ir_change", **_summary(metrics["validation_long_ir"] - metrics["train_long_ir"])},
    ])
    stability = pd.DataFrame(stability_rows)

    after_metrics = metrics[metrics["structural_hash"].isin(hard_pass)]
    quality_rows = []
    for metric in ("abs_train_ic", "abs_validation_ic", "train_long_ir", "validation_long_ir", "train_barra_ts_corr"):
        before_stats, after_stats = _summary(metrics[metric]), _summary(after_metrics[metric])
        quality_rows.append({"metric": metric, "before_universe": "validation_evaluated", "after_universe": "hard_filter_pass", "before_n": int(metrics[metric].notna().sum()), "after_n": int(after_metrics[metric].notna().sum()), **{f"before_{key}": value for key, value in before_stats.items()}, **{f"after_{key}": value for key, value in after_stats.items()}})
    quality = pd.DataFrame(quality_rows)

    validation_by_hash = {str(row["structural_hash"]): row for row in validation_rows}
    enrichment_by_hash = {str(row["structural_hash"]): row for row in enrichment_results}
    effective: dict[str, dict[str, float]] = {}; series_rows = []; warnings = []
    for structural_hash in hard_pass:
        mapping, reason = _series_from_result(validation_by_hash[structural_hash]["result"])
        origin = "validation_result_train_long_excess"
        if reason is not None:
            enrichment = enrichment_by_hash.get(structural_hash, {})
            payload = enrichment.get("long_excess")
            if enrichment.get("status") == "completed" and isinstance(payload, Mapping):
                mapping, reason = _series_from_payload(payload); origin = "survivor_enrichment"
        if reason is None:
            effective[structural_hash] = mapping
            series_rows.extend({"structural_hash": structural_hash, "date": date, "value": value, "origin": origin} for date, value in sorted(mapping.items()))
        else:
            warnings.append({"category": "train_long_excess_unavailable", "structural_hash": structural_hash, "message": reason})

    ranked = _rank_hashes(hard_pass, hard_by_hash)
    pool_ranked = _rank_hashes(pool_hashes, hard_by_hash)
    before_matrix = _correlation_matrix(ranked[:30], effective)
    after_matrix = _correlation_matrix(pool_ranked[:20], effective)
    audit = _greedy_audit(greedy_rows, effective)
    pair_summary = pd.DataFrame([_pair_summary("decorrelation_input", ranked, effective), _pair_summary("provisional_pool", pool_ranked, effective)])
    deco_summary = {"input_count": len(greedy_rows), "retained_count": len(pool_hashes), "correlation_rejected_count": sum(row.get("decorrelation_status") == "rejected_by_correlation" for row in greedy_rows), "invalid_count": sum(row.get("decorrelation_status") == "decorrelation_invalid" for row in greedy_rows), "retention_rate": len(pool_hashes) / len(greedy_rows) if greedy_rows else math.nan}
    pair_summary = pair_summary.assign(**deco_summary)

    metrics_by_hash = metrics.set_index("structural_hash", drop=False)
    pool_output = []
    for provisional_rank, row in enumerate(pool_rows, start=1):
        structural_hash = str(row["structural_hash"]); metric = metrics_by_hash.loc[structural_hash]
        pool_output.append({"provisional_rank": provisional_rank, "sorted_rank": row.get("sorted_rank"), "structural_hash": structural_hash, "formula": row.get("expression", {}).get("formula", metric["formula"]), "prefix_token_ids": row.get("expression", {}).get("prefix_token_ids", metric["prefix_token_ids"]), "node_count": row.get("expression", {}).get("node_count", metric["node_count"]), "depth": row.get("expression", {}).get("depth", metric["depth"]), "train_ic": metric["train_ic"], "validation_ic": metric["validation_ic"], "train_direction": metric["train_direction"], "train_long_ir": metric["train_long_ir"], "validation_long_ir": metric["validation_long_ir"], "train_barra_ts_corr": metric["train_barra_ts_corr"], "decorrelation_status": "retained", "source_provenance": row.get("source_identity", {}), "validation_result_fingerprint": row.get("result_fingerprint"), "source_snapshot_fingerprint": snapshot_manifest.get("source_snapshot_fingerprint"), "source_set_fingerprint": snapshot_manifest.get("source_set_fingerprint"), "accepted_registry_fingerprint": snapshot_manifest.get("accepted_registry_fingerprint"), "train_entry_fingerprint": snapshot_manifest.get("train_entry_fingerprint"), "train_pass_fingerprint": snapshot_manifest.get("train_pass_fingerprint"), "validation_context_fingerprint": snapshot_manifest.get("validation_context_fingerprint"), "validation_evaluation_fingerprint": snapshot_manifest.get("validation_evaluation_fingerprint"), "selection_fingerprint": snapshot_manifest.get("selection_fingerprint"), "selection_contract_fingerprint": snapshot_manifest.get("selection_contract_fingerprint"), "enrichment_fingerprint": snapshot_manifest.get("enrichment_fingerprint")})
    pool = pd.DataFrame(pool_output)

    ranked_pool = pool.sort_values("provisional_rank", kind="stable").reset_index(drop=True)
    top100 = ranked_pool.head(100).copy()
    top100["abs_train_ic"] = pd.to_numeric(top100["train_ic"], errors="coerce").abs()
    top100["abs_validation_ic"] = pd.to_numeric(top100["validation_ic"], errors="coerce").abs()
    top100["operator_count"] = top100["prefix_token_ids"].map(
        lambda tokens: sum(1 for token in tokens if get_action(int(token)).arity)
    )
    top100["leaf_count"] = top100["prefix_token_ids"].map(
        lambda tokens: sum(1 for token in tokens if not get_action(int(token)).arity)
    )
    top100_columns = [
        "provisional_rank", "sorted_rank", "structural_hash", "formula",
        "prefix_token_ids", "node_count", "depth", "operator_count", "leaf_count",
        "train_direction", "train_ic", "validation_ic",
        "abs_train_ic", "abs_validation_ic", "train_long_ir",
        "validation_long_ir", "train_barra_ts_corr",
    ]
    top100_metrics = top100.reindex(columns=top100_columns)
    top100_quality = pd.DataFrame(
        [
            {"metric": metric, **_extended_summary(top100_metrics[metric])}
            for metric in (
                "abs_train_ic", "abs_validation_ic", "train_long_ir",
                "validation_long_ir", "train_barra_ts_corr",
            )
        ]
    )
    top_examples = top100_metrics.head(10).copy()

    complexity_rows = []
    for metric in ("node_count", "depth", "operator_count", "leaf_count"):
        values = pd.to_numeric(top100_metrics[metric], errors="coerce").dropna()
        complexity_rows.append(
            {
                "universe": "frozen_order_top100",
                "candidate_count": int(len(top100_metrics)),
                "metric": metric,
                "min": float(values.min()),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "p95": float(values.quantile(.95)),
                "max": float(values.max()),
            }
        )
    complexity_summary = pd.DataFrame(complexity_rows)

    before_structure = metrics[["structural_hash", "prefix_token_ids", "node_count", "depth"]].copy()
    after_structure = pool[["structural_hash", "prefix_token_ids", "node_count", "depth"]].copy()
    before_c, before_o, before_f, before_w = _token_prevalence(before_structure, "validation_evaluated")
    after_c, after_o, after_f, after_w = _token_prevalence(after_structure, "provisional_pool")
    complexity = pd.concat([before_c, after_c], ignore_index=True)
    structure_rows = []
    for metric_name in ("node_count", "depth"):
        for universe, frame in (("validation_evaluated", before_c), ("provisional_pool", after_c)):
            structure_rows.append({"category": "complexity", "item": metric_name, "universe": universe, "candidate_count": len(frame), **_summary(frame[metric_name])})
    def shift(before: pd.DataFrame, after: pd.DataFrame, key: str) -> pd.DataFrame:
        result = before.merge(after, on=key, how="outer", suffixes=("_before", "_after")).fillna({"candidate_prevalence_before": 0, "candidate_prevalence_after": 0, "occurrence_count_before": 0, "occurrence_count_after": 0, "prevalence_ratio_before": 0.0, "prevalence_ratio_after": 0.0})
        result["before_universe"] = "validation_evaluated"; result["after_universe"] = "provisional_pool"
        result["prevalence_change"] = result["prevalence_ratio_after"] - result["prevalence_ratio_before"]
        return result
    operator_shift, field_shift, window_shift = shift(before_o, after_o, "operator"), shift(before_f, after_f, "field"), shift(before_w, after_w, "window")
    structure = pd.concat([pd.DataFrame(structure_rows), operator_shift.assign(category="operator", item=operator_shift["operator"]), field_shift.assign(category="field", item=field_shift["field"]), window_shift.assign(category="window", item=window_shift["window"])], ignore_index=True, sort=False)

    return Stage6ReportDataBundle(
        snapshot_manifest=dict(snapshot_manifest), source_candidates=source_frame,
        train_prefilter_results=train_frame, validation_candidate_metrics=metrics,
        hard_filter_results=hard_frame, funnel_summary=funnel,
        hard_filter_condition_summary=condition_summary, failure_combinations=failures,
        stability_summary=stability, before_after_quality_summary=quality,
        decorrelation_input=hard_frame[hard_frame["hard_filter_pass"] == True].copy(),
        decorrelation_outcomes=greedy_frame, effective_train_long_excess=pd.DataFrame(series_rows, columns=["structural_hash", "date", "value", "origin"]),
        before_top30_correlation=before_matrix, after_top20_correlation=after_matrix,
        decorrelation_pair_summary=pair_summary, greedy_pair_audit=audit,
        provisional_factor_pool=pool, top100_candidate_metrics=top100_metrics,
        top100_quality_summary=top100_quality, complexity_summary=complexity_summary,
        top_candidate_examples=top_examples, structure_shift_summary=structure,
        operator_prevalence_shift=operator_shift, field_prevalence_shift=field_shift,
        window_prevalence_shift=window_shift,
        availability_and_warnings=pd.DataFrame(warnings, columns=["category", "structural_hash", "message"]),
    )


def _read_validation_run(entry: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    database = Path(str(entry["database_path"])).resolve()
    _require(database.is_file(), f"Validation EvaluationStore is missing: {database}")
    uri = database.as_uri().replace("file:///", "file:/") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        run_id = str(entry["evaluation_run_id"])
        run = connection.execute("SELECT manifest_json,status FROM runs WHERE run_id=?", (run_id,)).fetchone()
        _require(run is not None and run["status"] == "complete", "Validation evaluation run is not complete")
        manifest = json.loads(run["manifest_json"])
        _require(manifest.get("oos") == "not_loaded", "Validation run OOS lock changed")
        _require(manifest.get("context_fingerprint") == entry.get("context_fingerprint"), "Validation context fingerprint mismatch")
        _require(manifest.get("evaluation_contract_fingerprint") == entry.get("evaluation_contract_fingerprint"), "Validation evaluation contract fingerprint mismatch")
        conflicts = connection.execute("SELECT COUNT(*) FROM determinism_conflicts AS dc JOIN run_candidates AS rc ON rc.cache_key=dc.cache_key WHERE rc.run_id=?", (run_id,)).fetchone()[0]
        _require(int(conflicts) == 0, "Validation run has determinism conflicts")
        records = []
        identity = []
        rows = connection.execute("SELECT rc.ordinal,rc.structural_hash,rc.cache_key,rc.state,rc.result_fingerprint,e.result_json,e.result_fingerprint AS cache_fp FROM run_candidates rc JOIN evaluations e ON e.cache_key=rc.cache_key WHERE rc.run_id=? ORDER BY rc.ordinal", (run_id,)).fetchall()
        _require(len(rows) == int(manifest["candidate_count"]), "Validation candidate count mismatch")
        for row in rows:
            _require(row["state"] in {"completed", "completed_invalid"}, "Validation contains unfinished candidate")
            result = json.loads(row["result_json"])
            _require(result.get("result_fingerprint") == row["result_fingerprint"] == row["cache_fp"], "Validation result fingerprint mismatch")
            deterministic_keys = ("schema", "status", "invalid_reasons", "expression", "context_fingerprint", "evaluation_contract_fingerprint", "train_direction", "train", "validation", "factor_finite_coverage")
            _require(all(key in result for key in deterministic_keys), "Validation result deterministic payload is incomplete")
            _require(_stable_hash({key: result[key] for key in deterministic_keys}) == result.get("result_fingerprint"), "Validation result deterministic fingerprint mismatch")
            record = {"ordinal": int(row["ordinal"]), "structural_hash": str(row["structural_hash"]), "cache_key": str(row["cache_key"]), "result_fingerprint": str(row["result_fingerprint"]), "result": result}
            records.append(record); identity.append({key: record[key] for key in ("ordinal", "structural_hash", "cache_key", "result_fingerprint")})
        ordered = _stable_hash(identity)
        return manifest, records, ordered
    finally:
        connection.close()


def load_stage6_report_data(
    *, source_set_manifest_path: str | Path, candidate_import_manifest_path: str | Path,
    compatibility_manifest_path: str | Path, train_entry_manifest_path: str | Path,
    train_pass_manifest_path: str | Path, validation_entry_manifest_path: str | Path,
    selection_manifest_path: str | Path,
) -> Stage6ReportDataBundle:
    """Load one completed formal Hybrid Stage 6 snapshot and fail closed."""

    paths = {name: Path(value).resolve() for name, value in {
        "source_set": source_set_manifest_path, "candidate_import": candidate_import_manifest_path,
        "compatibility": compatibility_manifest_path, "train_entry": train_entry_manifest_path,
        "train_pass": train_pass_manifest_path, "validation_entry": validation_entry_manifest_path,
        "selection": selection_manifest_path,
    }.items()}
    payload = {name: _read_json(path) for name, path in paths.items()}
    source_set = payload["source_set"]
    _require(source_set.get("schema") == SOURCE_SET_SCHEMA, "source-set schema mismatch")
    source_set_core = {
        "schema": SOURCE_SET_SCHEMA,
        "mode": source_set.get("mode"),
        "sources": [
            {key: item.get(key) for key in ("source_id", "source_type", "source_role", "snapshot_fingerprint")}
            for item in source_set.get("sources", [])
        ],
    }
    _require(_stable_hash(source_set_core) == source_set.get("source_set_fingerprint"), "source-set fingerprint mismatch")
    sources = source_set.get("sources", [])
    _require(len(sources) == 1 and sources[0].get("source_type") == "hybrid_train_artifact", "formal reporting requires one Hybrid source only")
    snapshot_path = Path(source_set["source_manifests"][0]["snapshot_manifest"]).resolve()
    snapshot = _read_json(snapshot_path)
    _require(snapshot.get("schema") == SOURCE_SNAPSHOT_SCHEMA and snapshot.get("snapshot_kind") == "completed_hybrid_train_artifact", "Hybrid source is incomplete or legacy")
    _require(snapshot.get("cutoff", {}).get("complete") is True and snapshot.get("cutoff", {}).get("pending_assignment") is None, "Hybrid source cutoff is incomplete")
    _require(snapshot.get("snapshot_fingerprint") == sources[0].get("snapshot_fingerprint"), "source snapshot fingerprint mismatch")
    snapshot_core = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_id": snapshot.get("source_id"), "source_type": snapshot.get("source_type"),
        "source_role": snapshot.get("source_role"), "inclusion_status": snapshot.get("inclusion_status"),
        "approval_note": snapshot.get("approval_note"), "candidate_record_policy": snapshot.get("candidate_record_policy"),
        "source_semantics": snapshot.get("source_semantics"), "snapshot_kind": snapshot.get("snapshot_kind"),
        "cutoff": snapshot.get("cutoff"), "record_counts": snapshot.get("record_counts"),
        "artifacts": [{key: item.get(key) for key in ("name", "size_bytes", "sha256")} for item in snapshot.get("artifacts", [])],
        "logical_content_fingerprint": snapshot.get("logical_content_fingerprint"),
    }
    if "external_artifacts" in snapshot:
        snapshot_core["external_artifacts"] = snapshot.get("external_artifacts")
    _require(_stable_hash(snapshot_core) == snapshot.get("snapshot_fingerprint"), "source snapshot fingerprint payload mismatch")
    for artifact in snapshot.get("artifacts", []): _verify_artifact(snapshot_path.parent, artifact["name"], artifact)
    for artifact in snapshot.get("external_artifacts", []):
        external = Path(str(artifact.get("source_path", ""))).resolve()
        _require(external.is_file(), f"external Hybrid artifact is missing: {external}")
        _require(external.stat().st_size == int(artifact.get("size_bytes", -1)) and _sha256_file(external) == artifact.get("sha256"), f"external Hybrid artifact SHA mismatch: {artifact.get('name')}")

    candidate_manifest = payload["candidate_import"]
    _require(candidate_manifest.get("schema") == CANDIDATE_IMPORT_MANIFEST_SCHEMA and candidate_manifest.get("downstream_eligible") is True, "candidate import is not downstream eligible")
    _require(_stable_hash(candidate_manifest.get("fingerprint_payload")) == candidate_manifest.get("registry_fingerprint"), "candidate registry fingerprint mismatch")
    _require(candidate_manifest.get("source_set_fingerprint") == source_set.get("source_set_fingerprint"), "candidate import source-set mismatch")
    candidate_registry = _read_jsonl(_verify_artifact(paths["candidate_import"].parent, "candidate_registry.jsonl", candidate_manifest["artifacts"]["candidate_registry.jsonl"]))
    source_candidates = []
    for group in candidate_registry:
        _require(group.get("downstream_eligible") is True and len(group.get("representations", [])) == 1, "source registry contains incompatible group")
        representation = group["representations"][0]
        source_candidates.append({"current_structural_hash": group["source_claimed_structural_hash"], **{key: representation.get(key) for key in ("formula", "prefix_token_ids", "node_count", "depth")}})

    compatibility = payload["compatibility"]
    _require(compatibility.get("schema") == EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA and compatibility.get("downstream_eligible") is True, "compatibility audit is not downstream eligible")
    _require(_stable_hash(compatibility.get("fingerprint_payload")) == compatibility.get("audit_fingerprint"), "compatibility audit fingerprint mismatch")
    accepted = _read_jsonl(_verify_artifact(paths["compatibility"].parent, "auto_accepted_candidate_registry.jsonl", compatibility["artifacts"]["auto_accepted_candidate_registry.jsonl"]))
    _require(_stable_hash(accepted) == compatibility.get("accepted_registry_fingerprint"), "accepted registry fingerprint mismatch")
    _require(len(source_candidates) == len(accepted), "compatibility universe unexpectedly lost Hybrid candidates")

    train_entry, train_pass, validation_entry = payload["train_entry"], payload["train_pass"], payload["validation_entry"]
    _require(train_entry.get("schema") == STAGE6_TRAIN_ENTRY_SCHEMA and train_entry.get("oos") == "not_loaded_not_evaluated", "Train entry contract mismatch")
    _require(train_pass.get("schema") == STAGE6_TRAIN_PASS_MANIFEST_SCHEMA and train_pass.get("oos") == "not_loaded_not_evaluated", "Train-pass contract mismatch")
    _require(validation_entry.get("schema") == STAGE6_VALIDATION_ENTRY_SCHEMA and validation_entry.get("oos") == "not_loaded_not_evaluated", "Validation entry contract mismatch")
    _require(train_entry.get("evaluation_run_scope") == STAGE6_TRAIN_PREPARATION_SCOPE, "formal report requires full accepted-registry Train preparation")
    _require(validation_entry.get("evaluation_run_scope") == STAGE6_VALIDATION_SCOPE, "Validation entry scope mismatch")
    for entry, fingerprint_key in ((train_entry, "entry_manifest_fingerprint"), (validation_entry, "entry_manifest_fingerprint")):
        _require(_stable_hash({key: value for key, value in entry.items() if key != fingerprint_key}) == entry.get(fingerprint_key), f"{entry.get('schema')} fingerprint mismatch")
    _require(_stable_hash({key: value for key, value in train_pass.items() if key != "train_pass_manifest_fingerprint"}) == train_pass.get("train_pass_manifest_fingerprint"), "Train-pass manifest fingerprint mismatch")
    _require(int(train_entry.get("candidate_count", -1)) == len(accepted), "Train preparation universe differs from compatible universe")
    _require(train_pass.get("train_entry_manifest_fingerprint") == train_entry.get("entry_manifest_fingerprint"), "Train-pass entry fingerprint mismatch")
    _require(validation_entry.get("train_pass_manifest_fingerprint") == train_pass.get("train_pass_manifest_fingerprint"), "Validation Train-pass fingerprint mismatch")
    _require(validation_entry.get("accepted_registry_fingerprint") == compatibility.get("accepted_registry_fingerprint"), "Validation accepted registry mismatch")
    prefilter = _read_jsonl(_verify_artifact(paths["train_pass"].parent, "train_prefilter_results.jsonl", train_pass["artifacts"]["train_prefilter_results.jsonl"]))
    _, validation_records, ordered = _read_validation_run(validation_entry)

    selection = payload["selection"]
    _require(selection.get("schema") == STAGE6_ENRICHED_SELECTION_MANIFEST_SCHEMA, "selection manifest schema mismatch")
    _require(selection.get("engineering_smoke") is False, "formal reporting forbids engineering smoke selection")
    _require(selection.get("oos") == "not_loaded_not_evaluated", "selection OOS lock changed")
    _require("provisional_evaluation_universe" not in selection, "formal standard report requires provisional_selection scope")
    _require(selection.get("evaluation_run_id") == validation_entry.get("evaluation_run_id") and selection.get("evaluation_ordered_result_set_fingerprint") == ordered, "selection and Validation run differ")
    selection_core = {
        key: value
        for key, value in selection.items()
        if key
        not in {
            "enriched_selection_fingerprint",
            "counts",
            "artifacts",
            "created_at_utc",
            "created_at_excluded_from_fingerprint",
            "scope",
        }
    }
    _require(_stable_hash(selection_core) == selection.get("enriched_selection_fingerprint"), "enriched selection fingerprint mismatch")
    _require(selection.get("selection_contract_fingerprint") == Stage6SelectionConfig().fingerprint == train_pass.get("selection_config_fingerprint"), "selection contract differs from frozen contract")
    artifacts = {name: _read_jsonl(_verify_artifact(paths["selection"].parent, name, metadata)) for name, metadata in selection.get("artifacts", {}).items()}
    hard = artifacts["hard_filter_results.jsonl"]; greedy = artifacts["greedy_decorrelation_results.jsonl"]; pool = artifacts["alpha_pool.jsonl"]; enrichment = artifacts["survivor_long_excess_enrichment.jsonl"]
    _require(_stable_hash(hard) == selection.get("hard_filter_digest"), "hard-filter digest mismatch")
    _require(_stable_hash(greedy) == selection.get("greedy_digest"), "greedy digest mismatch")
    _require(_stable_hash(pool) == selection.get("alpha_pool_digest"), "alpha-pool digest mismatch")
    enrichment_digest_rows = [{key: value for key, value in row.items() if key not in {"factor_seconds", "train_long_excess_seconds", "total_seconds"}} for row in enrichment]
    _require(_stable_hash(enrichment_digest_rows) == selection.get("enrichment_digest"), "enrichment digest mismatch")

    report_snapshot = {
        "schema": STAGE6_REPORT_DATA_SCHEMA, "source_type": "hybrid_train_artifact",
        "source_snapshot_fingerprint": snapshot["snapshot_fingerprint"], "source_set_fingerprint": source_set["source_set_fingerprint"],
        "accepted_registry_fingerprint": compatibility["accepted_registry_fingerprint"],
        "train_entry_fingerprint": train_entry["entry_manifest_fingerprint"], "train_pass_fingerprint": train_pass["train_pass_manifest_fingerprint"],
        "validation_run_id": validation_entry["evaluation_run_id"], "validation_context_fingerprint": validation_entry["context_fingerprint"],
        "validation_evaluation_fingerprint": validation_entry["evaluation_contract_fingerprint"], "validation_ordered_result_set_fingerprint": ordered,
        "selection_fingerprint": selection["enriched_selection_fingerprint"], "selection_contract_fingerprint": selection["selection_contract_fingerprint"],
        "enrichment_fingerprint": selection["enrichment_contract_fingerprint"], "selection_scope": "provisional_selection", "engineering_smoke": False,
        "oos": "not_loaded_not_evaluated", "before_correlation_top_k": 30,
        "after_correlation_top_k": 20, "top100_count": 100,
        "top100_policy": "authoritative_provisional_pool_order_prefix",
        "minimum_common_periods": 60, "pair_audit_version": PAIR_AUDIT_VERSION,
    }
    return build_stage6_report_data(snapshot_manifest=report_snapshot, source_candidates=source_candidates, train_prefilter_results=prefilter, validation_records=validation_records, hard_filter_results=hard, greedy_results=greedy, alpha_pool=pool, enrichment_results=enrichment)


__all__ = ["PAIR_AUDIT_VERSION", "STAGE6_REPORT_DATA_SCHEMA", "Stage6ReportDataBundle", "build_stage6_report_data", "load_stage6_report_data", "pair_correlation"]
