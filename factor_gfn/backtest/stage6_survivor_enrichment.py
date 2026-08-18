"""Stage 6 step 9C survivor-only Train long-excess enrichment.

The hard filter always consumes the complete mixed evaluation run. Only its
survivors may receive a fresh Train long-excess series. Reused Train IC, Long
IR, Barra metrics, and direction are never replaced by the enrichment result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from factor_gfn.evaluator import (
    NeutralizationDiagnostics,
    clean_candidate_factor_cross_sections,
    encode_industry_panel,
    infer_long_direction,
    long_portfolio_series_from_cleaned,
    summarize_excess_returns,
)

from .stage6_evaluation import (
    Stage6CandidateEvaluator,
    Stage6EvaluationConfig,
    _diagnostics_payload,
    _finite_or_none,
    _float_series,
    _sha256_file,
    _stable_hash,
    build_stage6_evaluation_context,
)
from .stage6_evaluation_store import (
    EvaluationStore,
    Stage6EvaluationRunner,
    VerifiedEvaluationRun,
)
from .stage6_selection import (
    STAGE6_SELECTION_RESULT_SCHEMA,
    Stage6SelectionConfig,
    _hard_filter_record,
    _pair_correlation,
    _series_mapping,
)
from .stage6_mixed_evaluation import (
    Stage6MixedCandidateEvaluator,
    Stage6TrainReuseOverlay,
)


STAGE6_LONG_EXCESS_ENRICHMENT_SCHEMA = (
    "factor_gfn.stage6_train_long_excess_enrichment.v1"
)
STAGE6_LONG_EXCESS_ENRICHMENT_CONTRACT_SCHEMA = (
    "factor_gfn.stage6_train_long_excess_enrichment_contract.v1"
)
STAGE6_LONG_EXCESS_ENRICHER_VERSION = "stage6-survivor-long-excess-only-v1"
STAGE6_ENRICHED_SELECTION_MANIFEST_SCHEMA = (
    "factor_gfn.stage6_enriched_selection_manifest.v1"
)
STAGE6_ENRICHED_SELECTION_VERSION = "stage6-hard-filter-enrich-greedy-v1"


class Stage6EnrichmentIntegrityError(RuntimeError):
    """An enrichment input, direction, or immutable artifact is invalid."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Stage6LongExcessEnrichmentResult:
    structural_hash: str
    status: str
    failure_reason: str | None
    expected_direction: int | None
    derived_direction: int | None
    direction_match: bool
    context_fingerprint: str
    enrichment_contract_fingerprint: str
    long_excess: Mapping[str, Any] | None
    diagnostics: Mapping[str, Any] | None
    factor_seconds: float
    train_long_excess_seconds: float
    total_seconds: float
    result_fingerprint: str

    def deterministic_payload(self) -> dict[str, Any]:
        return {
            "schema": STAGE6_LONG_EXCESS_ENRICHMENT_SCHEMA,
            "structural_hash": self.structural_hash,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "expected_direction": self.expected_direction,
            "derived_direction": self.derived_direction,
            "direction_match": self.direction_match,
            "context_fingerprint": self.context_fingerprint,
            "enrichment_contract_fingerprint": self.enrichment_contract_fingerprint,
            "long_excess": dict(self.long_excess) if self.long_excess is not None else None,
            "diagnostics": dict(self.diagnostics) if self.diagnostics is not None else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.deterministic_payload(),
            "factor_seconds": self.factor_seconds,
            "train_long_excess_seconds": self.train_long_excess_seconds,
            "total_seconds": self.total_seconds,
            "result_fingerprint": self.result_fingerprint,
        }


class Stage6TrainLongExcessEnricher:
    """Compute only the missing Train directional long-excess series."""

    def __init__(self, fresh_evaluator: Stage6CandidateEvaluator) -> None:
        self.fresh_evaluator = fresh_evaluator
        self.context = fresh_evaluator.context
        self.contract: Mapping[str, Any] = MappingProxyType(
            {
                "schema": STAGE6_LONG_EXCESS_ENRICHMENT_CONTRACT_SCHEMA,
                "version": STAGE6_LONG_EXCESS_ENRICHER_VERSION,
                "base_fresh_evaluation_contract_fingerprint": (
                    fresh_evaluator.evaluation_contract_fingerprint
                ),
                "context_fingerprint": self.context.fingerprint,
                "split": "train_only",
                "interpretation": "one_full_history_pass_through_validation_for_warmup",
                "cleaning": "same_frozen_stage6_candidate_cleaning",
                "direction": "derive_from_preserved_train_ic_and_require_exact_match",
                "output": "directional_train_long_excess_dates_values_only",
                "forbidden_overwrite": [
                    "train_ic",
                    "train_long_ir",
                    "train_barra_ts_corr",
                    "train_barra_correlations",
                    "train_direction",
                ],
                "validation": "not_evaluated",
                "oos": "not_loaded_and_interface_rejected",
            }
        )
        self.contract_fingerprint = _stable_hash(dict(self.contract))

    def _failure(
        self,
        structural_hash: str,
        reason: str,
        expected_direction: int | None,
        derived_direction: int | None,
        total_started: float,
    ) -> Stage6LongExcessEnrichmentResult:
        deterministic = {
            "schema": STAGE6_LONG_EXCESS_ENRICHMENT_SCHEMA,
            "structural_hash": structural_hash,
            "status": "enrichment_invalid",
            "failure_reason": reason,
            "expected_direction": expected_direction,
            "derived_direction": derived_direction,
            "direction_match": False,
            "context_fingerprint": self.context.fingerprint,
            "enrichment_contract_fingerprint": self.contract_fingerprint,
            "long_excess": None,
            "diagnostics": None,
        }
        return Stage6LongExcessEnrichmentResult(
            structural_hash=structural_hash,
            status="enrichment_invalid",
            failure_reason=reason,
            expected_direction=expected_direction,
            derived_direction=derived_direction,
            direction_match=False,
            context_fingerprint=self.context.fingerprint,
            enrichment_contract_fingerprint=self.contract_fingerprint,
            long_excess=None,
            diagnostics=None,
            factor_seconds=0.0,
            train_long_excess_seconds=0.0,
            total_seconds=time.perf_counter() - total_started,
            result_fingerprint=_stable_hash(deterministic),
        )

    def evaluate(
        self,
        candidate: Mapping[str, Any],
        *,
        preserved_train_ic: float,
        preserved_train_direction: int,
        preserved_train_long_valid_periods: int,
    ) -> Stage6LongExcessEnrichmentResult:
        total_started = time.perf_counter()
        expression, identity = self.fresh_evaluator._expression(candidate)
        structural_hash = str(identity["structural_hash"])
        if not math.isfinite(preserved_train_ic) or preserved_train_ic == 0.0:
            return self._failure(
                structural_hash,
                "preserved_train_ic_cannot_define_direction",
                preserved_train_direction,
                None,
                total_started,
            )
        derived_direction = infer_long_direction(float(preserved_train_ic))
        if preserved_train_direction not in (-1, 1) or (
            derived_direction != preserved_train_direction
        ):
            return self._failure(
                structural_hash,
                "preserved_train_direction_mismatch",
                preserved_train_direction,
                derived_direction,
                total_started,
            )

        factor_started = time.perf_counter()
        factor = np.asarray(
            self.fresh_evaluator._interpreter.evaluate(expression), dtype=np.float64
        )
        factor_seconds = time.perf_counter() - factor_started
        expected_shape = (self.context.dates.size, self.context.stocks.size)
        if factor.shape != expected_shape:
            raise Stage6EnrichmentIntegrityError(
                f"FactorInterpreter returned {factor.shape}; expected {expected_shape}"
            )

        long_started = time.perf_counter()
        split = self.context.get_split_data("train")
        compact_factor = np.asarray(factor[split.global_rebalance_rows], dtype=np.float64)
        diagnostics = NeutralizationDiagnostics()
        encoded = encode_industry_panel(split.industry_labels, compact_factor.shape)
        cleaned = clean_candidate_factor_cross_sections(
            compact_factor,
            split.industry_labels,
            split.universe_mask,
            diagnostics=diagnostics,
            encoded_industries=encoded,
        )
        series = long_portfolio_series_from_cleaned(
            cleaned,
            split.forward_returns,
            preserved_train_direction,
            self.context.config.evaluation,
        )
        summary = summarize_excess_returns(
            series.excess_return, self.context.config.evaluation
        )
        dates = [str(value) for value in split.rebalance_dates]
        values = _float_series(series.excess_return)
        valid_periods = int(summary.valid_periods)
        long_seconds = time.perf_counter() - long_started
        valid_match = valid_periods == int(preserved_train_long_valid_periods)
        status = "completed" if valid_match else "enrichment_invalid"
        failure_reason = None if valid_match else "train_long_valid_periods_mismatch"
        long_excess = {
            "dates": dates,
            "values": values,
            "direction": preserved_train_direction,
            "valid_periods": valid_periods,
            "total_periods": int(summary.total_periods),
            "origin": "stage6_fresh_long_excess",
        }
        diagnostics_payload = {
            "neutralization": _diagnostics_payload(diagnostics, split),
            "factor_finite_coverage": {
                "finite_universe_values": int(
                    np.sum(split.universe_mask & np.isfinite(compact_factor))
                ),
                "eligible_universe_values": int(np.sum(split.universe_mask)),
            },
            "preserved_train_long_valid_periods": int(
                preserved_train_long_valid_periods
            ),
            "computed_train_long_valid_periods": valid_periods,
            "valid_periods_match": valid_match,
        }
        deterministic = {
            "schema": STAGE6_LONG_EXCESS_ENRICHMENT_SCHEMA,
            "structural_hash": structural_hash,
            "status": status,
            "failure_reason": failure_reason,
            "expected_direction": preserved_train_direction,
            "derived_direction": derived_direction,
            "direction_match": True,
            "context_fingerprint": self.context.fingerprint,
            "enrichment_contract_fingerprint": self.contract_fingerprint,
            "long_excess": long_excess,
            "diagnostics": diagnostics_payload,
        }
        return Stage6LongExcessEnrichmentResult(
            structural_hash=structural_hash,
            status=status,
            failure_reason=failure_reason,
            expected_direction=preserved_train_direction,
            derived_direction=derived_direction,
            direction_match=True,
            context_fingerprint=self.context.fingerprint,
            enrichment_contract_fingerprint=self.contract_fingerprint,
            long_excess=MappingProxyType(long_excess),
            diagnostics=MappingProxyType(diagnostics_payload),
            factor_seconds=float(factor_seconds),
            train_long_excess_seconds=float(long_seconds),
            total_seconds=float(time.perf_counter() - total_started),
            result_fingerprint=_stable_hash(deterministic),
        )


def select_stage6_9c_engineering_smoke_candidates(
    candidates: Sequence[Mapping[str, Any]],
    overlay: Stage6TrainReuseOverlay,
    *,
    count: int = 24,
) -> list[dict[str, Any]]:
    """Select only by the approved Train-side conditions and stable hash order."""

    if count < 1:
        raise ValueError("9C engineering smoke count must be positive")
    by_hash = {
        str(candidate["current_structural_hash"]): dict(candidate)
        for candidate in candidates
    }
    eligible: list[dict[str, Any]] = []
    for structural_hash, record in overlay.records.items():
        metrics = record["train_metrics"]
        train_ic = metrics.get("train_ic")
        train_ir = metrics.get("train_long_ir")
        barra = metrics.get("train_barra_ts_corr")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (train_ic, train_ir, barra)
        ):
            continue
        if (
            abs(float(train_ic)) > 0.01
            and float(train_ir) > 0.25
            and float(barra) < 0.7
            and structural_hash in by_hash
        ):
            eligible.append(by_hash[structural_hash])
    eligible.sort(key=lambda row: str(row["current_structural_hash"]))
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} overlay candidates satisfy the Train smoke gate; need {count}"
        )
    return eligible[:count]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _effective_result(
    base_result: Mapping[str, Any],
    base_result_fingerprint: str,
    enrichment: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    result = {**dict(base_result)}
    train = {**dict(result["train"])}
    long_result = {**dict(train["long"])}
    long_result["excess_series"] = {
        "dates": list(enrichment["long_excess"]["dates"]),
        "values": list(enrichment["long_excess"]["values"]),
    }
    long_result["excess_series_origin"] = "stage6_fresh_long_excess"
    long_result["excess_series_contract_fingerprint"] = enrichment[
        "enrichment_contract_fingerprint"
    ]
    long_result["excess_series_result_fingerprint"] = enrichment[
        "result_fingerprint"
    ]
    train["long"] = long_result
    result["train"] = train
    identity = {
        "base_result_fingerprint": base_result_fingerprint,
        "enrichment_result_fingerprint": enrichment["result_fingerprint"],
        "effective_train_long_excess": long_result["excess_series"],
    }
    return result, _stable_hash(identity)


def _already_available_record(
    stored: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    mapping, error = _series_mapping(result, "train")
    if error is not None:
        return {
            "schema": STAGE6_LONG_EXCESS_ENRICHMENT_SCHEMA,
            "structural_hash": stored["structural_hash"],
            "status": "enrichment_invalid",
            "failure_reason": error,
            "origin": "existing_stage6_evaluation",
            "base_result_fingerprint": stored["result_fingerprint"],
            "result_fingerprint": _stable_hash(
                {
                    "structural_hash": stored["structural_hash"],
                    "failure_reason": error,
                    "base_result_fingerprint": stored["result_fingerprint"],
                }
            ),
        }
    train = result.get("train", {})
    long_result = train.get("long", {}) if isinstance(train, Mapping) else {}
    series = (
        long_result.get("excess_series")
        if isinstance(long_result, Mapping)
        else None
    )
    series_origin = series.get("origin") if isinstance(series, Mapping) else None
    if not isinstance(series_origin, str) or not series_origin:
        legacy_origin = (
            long_result.get("excess_series_origin")
            if isinstance(long_result, Mapping)
            else None
        )
        metric_origin = train.get("metric_origin") if isinstance(train, Mapping) else None
        series_origin = (
            legacy_origin
            if isinstance(legacy_origin, str) and legacy_origin
            else metric_origin
            if isinstance(metric_origin, str) and metric_origin
            else "existing_stage6_evaluation"
        )
    provenance = {
        "origin": series_origin,
        "train_evaluation_contract_fingerprint": (
            series.get("train_evaluation_contract_fingerprint")
            if isinstance(series, Mapping)
            else None
        ),
        "overlay_record_fingerprint": (
            series.get("overlay_record_fingerprint")
            if isinstance(series, Mapping)
            else None
        ),
    }
    payload = {
        "schema": STAGE6_LONG_EXCESS_ENRICHMENT_SCHEMA,
        "structural_hash": stored["structural_hash"],
        "status": "already_available",
        "failure_reason": None,
        "origin": series_origin,
        "source_provenance": provenance,
        "base_result_fingerprint": stored["result_fingerprint"],
        "finite_periods": len(mapping),
    }
    return {**payload, "result_fingerprint": _stable_hash(payload)}


def _greedy_with_enrichment(
    sorted_hard_rows: Sequence[Mapping[str, Any]],
    effective_results: Mapping[str, Mapping[str, Any]],
    enrichment_by_hash: Mapping[str, Mapping[str, Any]],
    config: Stage6SelectionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for rank, hard_row in enumerate(sorted_hard_rows, start=1):
        structural_hash = str(hard_row["structural_hash"])
        enrichment = enrichment_by_hash[structural_hash]
        trace: list[dict[str, Any]] = []
        status = "retained"
        blocker: str | None = None
        blocking_corr: float | None = None
        invalid_reason: str | None = None
        invalid_peer: str | None = None
        invalid_common: int | None = None
        if enrichment["status"] == "enrichment_invalid":
            status = "decorrelation_invalid"
            invalid_reason = f"enrichment:{enrichment['failure_reason']}"
        else:
            candidate_result = effective_results[structural_hash]
            for retained_row in retained:
                retained_hash = str(retained_row["structural_hash"])
                train = _pair_correlation(
                    candidate_result,
                    effective_results[retained_hash],
                    "train",
                    config.decorrelation_min_common_periods,
                )
                validation = _pair_correlation(
                    candidate_result,
                    effective_results[retained_hash],
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
            "base_result_fingerprint": hard_row["base_result_fingerprint"],
            "effective_result_fingerprint": hard_row["result_fingerprint"],
            "enrichment_result_fingerprint": enrichment["result_fingerprint"],
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


def run_stage6_survivor_enrichment_selection(
    *,
    store: EvaluationStore,
    evaluation_run_id: str,
    accepted_candidates: Sequence[Mapping[str, Any]],
    enricher: Stage6TrainLongExcessEnricher,
    output_root: str | Path,
    config: Stage6SelectionConfig | None = None,
    engineering_smoke: bool = False,
    provisional_universe: Mapping[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Path:
    """Hard-filter all run records, enrich only survivors, then decorrelate."""

    selection_config = config or Stage6SelectionConfig()
    started = time.perf_counter()

    def notify(event_type: str, **details: Any) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "event_type": event_type,
                    "evaluation_run_id": evaluation_run_id,
                    "elapsed_seconds": time.perf_counter() - started,
                    **details,
                }
            )

    notify("selection_started")
    verified = store.load_verified_run_results(evaluation_run_id)
    if verified.manifest["context_fingerprint"] != enricher.context.fingerprint:
        raise Stage6EnrichmentIntegrityError("enricher context differs from evaluation run")
    candidates_by_hash = {
        str(candidate["current_structural_hash"]): dict(candidate)
        for candidate in accepted_candidates
    }
    hard_rows: list[dict[str, Any]] = []
    stored_by_hash: dict[str, Mapping[str, Any]] = {}
    for stored in verified.records:
        structural_hash = str(stored["structural_hash"])
        if structural_hash not in candidates_by_hash:
            raise Stage6EnrichmentIntegrityError("evaluation candidate is absent from registry")
        stored_by_hash[structural_hash] = stored
        hard_rows.append(
            _hard_filter_record(
                {**dict(stored), "_selection_config": selection_config}
            )
        )
    passed = [row for row in hard_rows if row["hard_filter_pass"]]
    failed_condition_counts: dict[str, int] = {}
    for row in hard_rows:
        for code in row["failed_conditions"]:
            failed_condition_counts[code] = failed_condition_counts.get(code, 0) + 1
    notify(
        "hard_filter_complete",
        input_candidates=len(hard_rows),
        evaluation_ineligible=sum(
            bool(row["evaluation_ineligible"]) for row in hard_rows
        ),
        hard_filter_pass=len(passed),
        hard_filter_fail=len(hard_rows) - len(passed),
        failed_condition_counts=dict(sorted(failed_condition_counts.items())),
    )
    passed.sort(
        key=lambda row: (
            -abs(float(row["metrics"]["train_ic"])),
            str(row["structural_hash"]),
        )
    )
    enrichment_rows: list[dict[str, Any]] = []
    effective_results: dict[str, Mapping[str, Any]] = {}
    effective_fingerprints: dict[str, str] = {}
    enrichment_completed = 0
    enrichment_invalid = 0
    enrichment_already_available = 0
    for survivor_ordinal, hard_row in enumerate(passed, start=1):
        structural_hash = str(hard_row["structural_hash"])
        notify(
            "survivor_started",
            survivor_ordinal=survivor_ordinal,
            survivor_count=len(passed),
            structural_hash=structural_hash,
            enrichment_completed=enrichment_completed,
            enrichment_invalid=enrichment_invalid,
            enrichment_already_available=enrichment_already_available,
        )
        stored = stored_by_hash[structural_hash]
        base_result = stored["result"]
        _, series_error = _series_mapping(base_result, "train")
        if series_error is None:
            record = _already_available_record(stored, base_result)
            enrichment_already_available += 1
            effective_results[structural_hash] = base_result
            effective_fingerprints[structural_hash] = str(stored["result_fingerprint"])
        else:
            train_ic = hard_row["metrics"]["train_ic"]
            train_direction = hard_row["train_direction"]
            valid_periods = base_result["train"]["long"].get("valid_periods")
            if not isinstance(valid_periods, int):
                raise Stage6EnrichmentIntegrityError(
                    "survivor lacks preserved Train long valid-period count"
                )
            observed = enricher.evaluate(
                candidates_by_hash[structural_hash],
                preserved_train_ic=float(train_ic),
                preserved_train_direction=int(train_direction),
                preserved_train_long_valid_periods=valid_periods,
            )
            record = observed.to_dict()
            record["base_result_fingerprint"] = stored["result_fingerprint"]
            if observed.status == "completed":
                enrichment_completed += 1
                effective, effective_fp = _effective_result(
                    base_result, str(stored["result_fingerprint"]), record
                )
                effective_results[structural_hash] = effective
                effective_fingerprints[structural_hash] = effective_fp
            else:
                enrichment_invalid += 1
                effective_results[structural_hash] = base_result
                effective_fingerprints[structural_hash] = str(
                    stored["result_fingerprint"]
                )
        enrichment_rows.append(record)
        notify(
            "survivor_resolved",
            survivor_ordinal=survivor_ordinal,
            survivor_count=len(passed),
            structural_hash=structural_hash,
            enrichment_status=record["status"],
            enrichment_completed=enrichment_completed,
            enrichment_invalid=enrichment_invalid,
            enrichment_already_available=enrichment_already_available,
        )

    enrichment_by_hash = {
        str(row["structural_hash"]): row for row in enrichment_rows
    }
    enriched_hard_rows: list[dict[str, Any]] = []
    for row in passed:
        structural_hash = str(row["structural_hash"])
        enriched_hard_rows.append(
            {
                **row,
                "base_result_fingerprint": row["result_fingerprint"],
                "result_fingerprint": effective_fingerprints[structural_hash],
            }
        )
    greedy_rows, alpha_pool = _greedy_with_enrichment(
        enriched_hard_rows,
        effective_results,
        enrichment_by_hash,
        selection_config,
    )
    notify(
        "greedy_complete",
        hard_filter_survivors=len(passed),
        retained=len(alpha_pool),
        rejected_by_correlation=sum(
            row["decorrelation_status"] == "rejected_by_correlation"
            for row in greedy_rows
        ),
        decorrelation_invalid=sum(
            row["decorrelation_status"] == "decorrelation_invalid"
            for row in greedy_rows
        ),
    )
    deterministic = {
        "schema": STAGE6_ENRICHED_SELECTION_MANIFEST_SCHEMA,
        "version": STAGE6_ENRICHED_SELECTION_VERSION,
        "engineering_smoke": engineering_smoke,
        "evaluation_run_id": evaluation_run_id,
        "evaluation_ordered_result_set_fingerprint": (
            verified.ordered_result_set_fingerprint
        ),
        "evaluation_contract_fingerprint": verified.manifest[
            "evaluation_contract_fingerprint"
        ],
        "context_fingerprint": verified.manifest["context_fingerprint"],
        "selection_contract_fingerprint": selection_config.fingerprint,
        "enrichment_contract_fingerprint": enricher.contract_fingerprint,
        "hard_filter_digest": _stable_hash(hard_rows),
        "enrichment_digest": _stable_hash(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "factor_seconds",
                        "train_long_excess_seconds",
                        "total_seconds",
                    }
                }
                for row in enrichment_rows
            ]
        ),
        "greedy_digest": _stable_hash(greedy_rows),
        "alpha_pool_digest": _stable_hash(alpha_pool),
        "oos": "not_loaded_not_evaluated",
    }
    if provisional_universe is not None:
        universe = _plain(provisional_universe)
        if int(universe["evaluation_eligible_count"]) < len(hard_rows):
            raise Stage6EnrichmentIntegrityError(
                "provisional universe is smaller than the selection input"
            )
        deterministic["provisional_evaluation_universe"] = universe
    fingerprint = _stable_hash(deterministic)
    root = Path(output_root).resolve()
    target = root / fingerprint
    manifest_path = target / "enriched_selection_manifest.json"
    if target.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("enriched_selection_fingerprint") != fingerprint
            or any(existing.get(key) != value for key, value in deterministic.items())
        ):
            raise Stage6EnrichmentIntegrityError(
                "existing enriched selection fingerprint conflict"
            )
        for name, metadata in existing.get("artifacts", {}).items():
            artifact_path = target / name
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != int(metadata["size_bytes"])
                or _sha256_file(artifact_path) != str(metadata["sha256"])
            ):
                raise Stage6EnrichmentIntegrityError(
                    f"existing enriched selection artifact changed: {name}"
                )
        notify(
            "selection_completed",
            manifest_path=str(manifest_path),
            reused_existing_artifact=True,
            counts=existing.get("counts", {}),
        )
        return manifest_path
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{fingerprint}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        _write_jsonl(temporary / "hard_filter_results.jsonl", hard_rows)
        _write_jsonl(
            temporary / "survivor_long_excess_enrichment.jsonl", enrichment_rows
        )
        _write_jsonl(temporary / "greedy_decorrelation_results.jsonl", greedy_rows)
        _write_jsonl(temporary / "alpha_pool.jsonl", alpha_pool)
        counts = {
            "input_candidates": len(hard_rows),
            "evaluation_ineligible": sum(
                bool(row["evaluation_ineligible"]) for row in hard_rows
            ),
            "hard_filter_pass": len(passed),
            "hard_filter_fail": len(hard_rows) - len(passed),
            "survivors_already_have_train_long_excess": sum(
                row["status"] == "already_available" for row in enrichment_rows
            ),
            "survivors_enriched": sum(
                row["status"] == "completed" for row in enrichment_rows
            ),
            "survivor_enrichment_invalid": sum(
                row["status"] == "enrichment_invalid" for row in enrichment_rows
            ),
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
        if provisional_universe is not None:
            counts.update(
                {
                    "original_accepted_candidate_count": int(
                        provisional_universe["original_accepted_candidate_count"]
                    ),
                    "evaluation_eligible_count": int(
                        provisional_universe["evaluation_eligible_count"]
                    ),
                    "deferred_candidate_count": int(
                        provisional_universe["deferred_candidate_count"]
                    ),
                    "deferred_reason_counts": _plain(
                        provisional_universe["deferred_reason_counts"]
                    ),
                }
            )
        artifacts = {}
        for name in (
            "hard_filter_results.jsonl",
            "survivor_long_excess_enrichment.jsonl",
            "greedy_decorrelation_results.jsonl",
            "alpha_pool.jsonl",
        ):
            path = temporary / name
            artifacts[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        manifest = {
            **deterministic,
            "enriched_selection_fingerprint": fingerprint,
            "counts": counts,
            "artifacts": artifacts,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "created_at_excluded_from_fingerprint": True,
            "scope": (
                "engineering_branch_coverage_not_provisional_selection"
                if engineering_smoke
                else (
                    "resource_limited_provisional_selection"
                    if provisional_universe is not None
                    else "provisional_selection"
                )
            ),
        }
        _write_json(temporary / "enriched_selection_manifest.json", manifest)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    notify(
        "selection_completed",
        manifest_path=str(manifest_path),
        reused_existing_artifact=False,
        counts=manifest["counts"],
    )
    return manifest_path


def run_current_stage6_9c_engineering_smoke(
    *,
    accepted_registry_path: str | Path,
    overlay_manifest_path: str | Path,
    output_root: str | Path,
    candidate_count: int = 24,
) -> Path:
    """Run the approved Train-gated 24-candidate 9C engineering smoke."""

    registry_path = Path(accepted_registry_path).resolve()
    candidates: list[dict[str, Any]] = []
    with registry_path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise Stage6EnrichmentIntegrityError(
                    f"blank accepted-registry row at line {line_number}"
                )
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise Stage6EnrichmentIntegrityError(
                    f"accepted-registry row {line_number} is not an object"
                )
            candidates.append(value)
    overlay = Stage6TrainReuseOverlay.load(overlay_manifest_path)
    if _stable_hash(candidates) != overlay.manifest["accepted_registry_fingerprint"]:
        raise Stage6EnrichmentIntegrityError("accepted registry logical digest mismatch")
    selected = select_stage6_9c_engineering_smoke_candidates(
        candidates, overlay, count=candidate_count
    )

    context = build_stage6_evaluation_context(Stage6EvaluationConfig())
    evaluator_kwargs = {
        "compatibility_audit_fingerprint": str(
            overlay.manifest["compatibility_audit_fingerprint"]
        ),
        "accepted_registry_fingerprint": str(
            overlay.manifest["accepted_registry_fingerprint"]
        ),
    }
    mixed_fresh = Stage6CandidateEvaluator(context, **evaluator_kwargs)
    mixed = Stage6MixedCandidateEvaluator(mixed_fresh, overlay)
    enrichment_fresh = Stage6CandidateEvaluator(context, **evaluator_kwargs)
    enricher = Stage6TrainLongExcessEnricher(enrichment_fresh)
    stable = {
        "schema": "factor_gfn.stage6_9c_engineering_smoke.v1",
        "candidate_selection": {
            "source": "verified_train_reuse_overlay_only",
            "conditions": [
                "abs(train_ic) > 0.01",
                "train_long_ir > 0.25",
                "train_barra_ts_corr < 0.7",
            ],
            "validation_information_used": False,
            "stable_order": "structural_hash_ascending",
            "count": candidate_count,
            "ordered_structural_hashes": [
                row["current_structural_hash"] for row in selected
            ],
        },
        "accepted_registry_fingerprint": overlay.manifest[
            "accepted_registry_fingerprint"
        ],
        "train_reuse_overlay_fingerprint": overlay.fingerprint,
        "context_fingerprint": context.fingerprint,
        "mixed_evaluation_contract_fingerprint": mixed.evaluation_contract_fingerprint,
        "enrichment_contract_fingerprint": enricher.contract_fingerprint,
        "selection_contract_fingerprint": Stage6SelectionConfig().fingerprint,
        "scope": "engineering_branch_coverage_not_provisional_selection",
        "oos": "not_loaded_not_evaluated",
    }
    smoke_fingerprint = _stable_hash(stable)
    target = Path(output_root).resolve() / smoke_fingerprint
    report_path = target / "stage6_9c_engineering_smoke_report.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("smoke_fingerprint") != smoke_fingerprint
            or any(existing.get(key) != value for key, value in stable.items())
        ):
            raise Stage6EnrichmentIntegrityError("9C smoke report fingerprint conflict")
        selection_path = Path(str(existing["enriched_selection_manifest"]))
        if not selection_path.is_file():
            raise Stage6EnrichmentIntegrityError("9C smoke selection manifest is missing")
        selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            selection_manifest.get("enriched_selection_fingerprint")
            != existing.get("enriched_selection_fingerprint")
        ):
            raise Stage6EnrichmentIntegrityError("9C smoke selection identity changed")
        for name, metadata in selection_manifest.get("artifacts", {}).items():
            artifact_path = selection_path.parent / name
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != int(metadata["size_bytes"])
                or _sha256_file(artifact_path) != str(metadata["sha256"])
            ):
                raise Stage6EnrichmentIntegrityError(
                    f"9C smoke selection artifact changed: {name}"
                )
        return report_path
    target.mkdir(parents=True, exist_ok=True)
    with EvaluationStore(
        target / "evaluation_store" / "stage6_evaluations.sqlite",
        target / "evaluation_runs",
    ) as store:
        frozen = store.create_run(selected, mixed)
        runner = Stage6EvaluationRunner(store, mixed)
        invocation = runner.run(frozen.run_id)
        if invocation.run_status != "complete":
            raise Stage6EnrichmentIntegrityError(
                f"9C smoke mixed evaluation did not complete: {invocation.run_status}"
            )
        selection_manifest_path = run_stage6_survivor_enrichment_selection(
            store=store,
            evaluation_run_id=frozen.run_id,
            accepted_candidates=selected,
            enricher=enricher,
            output_root=target / "enriched_selections",
            engineering_smoke=True,
        )
        resume = runner.run(frozen.run_id)
    selection_manifest = json.loads(
        selection_manifest_path.read_text(encoding="utf-8")
    )
    report = {
        **stable,
        "smoke_fingerprint": smoke_fingerprint,
        "mixed_evaluation_run_id": frozen.run_id,
        "mixed_evaluation_invocation": {
            key: getattr(invocation, key)
            for key in invocation.__dataclass_fields__
        },
        "mixed_evaluation_resume_verification": {
            key: getattr(resume, key) for key in resume.__dataclass_fields__
        },
        "enriched_selection_fingerprint": selection_manifest[
            "enriched_selection_fingerprint"
        ],
        "enriched_selection_manifest": str(selection_manifest_path),
        "branch_counts": selection_manifest["counts"],
        "branch_coverage_complete": (
            selection_manifest["counts"]["hard_filter_pass"] > 0
            and selection_manifest["counts"]["survivors_enriched"] > 0
        ),
        "provisional_selection_result": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "created_at_excluded_from_fingerprint": True,
    }
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    try:
        _write_json(temporary, report)
        os.replace(temporary, report_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report_path


__all__ = [
    "STAGE6_ENRICHED_SELECTION_MANIFEST_SCHEMA",
    "STAGE6_ENRICHED_SELECTION_VERSION",
    "STAGE6_LONG_EXCESS_ENRICHER_VERSION",
    "STAGE6_LONG_EXCESS_ENRICHMENT_CONTRACT_SCHEMA",
    "STAGE6_LONG_EXCESS_ENRICHMENT_SCHEMA",
    "Stage6EnrichmentIntegrityError",
    "Stage6LongExcessEnrichmentResult",
    "Stage6TrainLongExcessEnricher",
    "run_stage6_survivor_enrichment_selection",
    "run_current_stage6_9c_engineering_smoke",
    "select_stage6_9c_engineering_smoke_candidates",
]
