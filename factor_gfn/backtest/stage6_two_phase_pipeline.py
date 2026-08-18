"""Two globally separated, resumable Stage 6 provisional evaluation phases.

Phase 1 prepares Train metrics for the complete accepted registry, persists the
three-condition Train prefilter decision with every candidate, and only after
the run is complete freezes the final Train-pass manifest.
Phase 2 consumes that immutable manifest and evaluates Validation only.  The
two phases intentionally use separate evaluator contracts and stores.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from factor_gfn.evaluator import infer_long_direction
from factor_gfn.gfn.real_data import RealRewardDataPaths

from .stage6_evaluation import (
    STAGE6_EVALUATION_RESULT_SCHEMA,
    Stage6CandidateEvaluationResult,
    Stage6CandidateEvaluator,
    Stage6EvaluationContext,
    _stable_hash,
    build_stage6_evaluation_context,
)
from .stage6_evaluation_store import (
    EvaluationStore,
    EvaluationStoreIntegrityError,
    Stage6EvaluationRunner,
)
from .stage6_full_pipeline import (
    LEGACY_FULL_EVALUATION_SCOPE,
    load_stage6_accepted_registry,
    load_stage6_full_entry_manifest,
)
from .stage6_mixed_evaluation import (
    TRAIN_METRIC_ORIGIN_FRESH,
    VALIDATION_METRIC_ORIGIN_FRESH,
    Stage6MixedCandidateEvaluator,
    Stage6TrainReuseOverlay,
)
from .stage6_prefilter_evaluation import (
    TRAIN_PREFILTER_PASSED,
    train_prefilter_decision,
)
from .stage6_selection import Stage6SelectionConfig
from .stage6_survivor_enrichment import (
    Stage6TrainLongExcessEnricher,
    run_stage6_survivor_enrichment_selection,
)


STAGE6_TRAIN_PREPARATION_SCOPE = "train_preparation_full_accepted_registry"
STAGE6_RESOURCE_LIMITED_TRAIN_SCOPE = (
    "train_preparation_resource_limited_eligible_universe"
)
STAGE6_VALIDATION_SCOPE = "validation_from_frozen_train_pass_manifest"
STAGE6_TRAIN_ENTRY_SCHEMA = "factor_gfn.stage6_train_preparation_entry.v1"
STAGE6_TRAIN_PASS_MANIFEST_SCHEMA = "factor_gfn.stage6_train_pass_manifest.v1"
STAGE6_VALIDATION_ENTRY_SCHEMA = "factor_gfn.stage6_validation_entry.v1"
STAGE6_TWO_PHASE_VERSION = "factor_gfn.stage6_two_phase_pipeline.v1"
STAGE6_PROVISIONAL_UNIVERSE_SCHEMA = (
    "factor_gfn.stage6_resource_limited_evaluation_universe.v1"
)
DEFERRED_TRAIN_RECOMPUTE = "deferred_train_recompute"
HISTORICAL_TRAIN_CONTRACT_NOT_EQUIVALENT = (
    "historical_train_contract_not_equivalent"
)
NO_TRUSTED_TRAIN_RESULT_RESOURCE_LIMITED = (
    "no_trusted_train_result_resource_limited"
)
TRAIN_NOT_EVALUATED_VALIDATION = "not_evaluated_train_preparation_phase"


class Stage6TwoPhaseIntegrityError(RuntimeError):
    """A two-phase input, cache, or frozen handoff failed closed validation."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage6TwoPhaseIntegrityError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise Stage6TwoPhaseIntegrityError(f"JSON artifact is not an object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise Stage6TwoPhaseIntegrityError(
                        f"blank JSONL row: {path}:{line_number}"
                    )
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise Stage6TwoPhaseIntegrityError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                yield value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage6TwoPhaseIntegrityError(f"cannot read JSONL: {path}") from error


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_identity(
    evaluator: Stage6CandidateEvaluator,
    candidate: Mapping[str, Any],
    evaluation_path: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "compatibility_audit_fingerprint": evaluator.compatibility_audit_fingerprint,
        "accepted_registry_fingerprint": evaluator.accepted_registry_fingerprint,
        "compatibility_record_fingerprint": candidate.get(
            "compatibility_record_fingerprint"
        ),
        "source_claimed_structural_hash": candidate.get(
            "source_claimed_structural_hash"
        ),
        "origin_ids": list(candidate.get("origin_ids", [])),
        "source_ids": list(candidate.get("source_ids", [])),
        "evaluation_path": evaluation_path,
        **extra,
    }


def _build_result(
    *,
    evaluator: Any,
    expression: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    status: str,
    invalid_reasons: tuple[str, ...],
    direction: int | None,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    coverage: Mapping[str, Any],
    factor_seconds: float,
    train_seconds: float,
    validation_seconds: float,
    total_seconds: float,
) -> Stage6CandidateEvaluationResult:
    deterministic = {
        "schema": STAGE6_EVALUATION_RESULT_SCHEMA,
        "status": status,
        "invalid_reasons": list(invalid_reasons),
        "expression": _plain(expression),
        "context_fingerprint": evaluator.context.fingerprint,
        "evaluation_contract_fingerprint": evaluator.evaluation_contract_fingerprint,
        "train_direction": direction,
        "train": _plain(train),
        "validation": _plain(validation),
        "factor_finite_coverage": _plain(coverage),
    }
    return Stage6CandidateEvaluationResult(
        schema=STAGE6_EVALUATION_RESULT_SCHEMA,
        status=status,
        invalid_reasons=invalid_reasons,
        expression=MappingProxyType(_plain(expression)),
        source_identity=MappingProxyType(_plain(source_identity)),
        context_fingerprint=evaluator.context.fingerprint,
        evaluation_contract_fingerprint=evaluator.evaluation_contract_fingerprint,
        train_direction=direction,
        train=MappingProxyType(_plain(train)),
        validation=MappingProxyType(_plain(validation)),
        factor_finite_coverage=MappingProxyType(_plain(coverage)),
        factor_seconds=float(factor_seconds),
        train_evaluation_seconds=float(train_seconds),
        validation_evaluation_seconds=float(validation_seconds),
        total_seconds=float(total_seconds),
        result_fingerprint=_stable_hash(deterministic),
    )


class Stage6TrainPreparationEvaluator:
    """Prepare Train only; never inspect, calculate, or reuse Validation."""

    def __init__(
        self,
        fresh_evaluator: Stage6CandidateEvaluator,
        overlay: Stage6TrainReuseOverlay,
        *,
        selection_config: Stage6SelectionConfig = Stage6SelectionConfig(),
    ) -> None:
        self.fresh_evaluator = fresh_evaluator
        self.mixed_evaluator = Stage6MixedCandidateEvaluator(fresh_evaluator, overlay)
        self.overlay = overlay
        self.selection_config = selection_config
        self.context = fresh_evaluator.context
        self.compatibility_audit_fingerprint = (
            fresh_evaluator.compatibility_audit_fingerprint
        )
        self.accepted_registry_fingerprint = fresh_evaluator.accepted_registry_fingerprint
        self.evaluation_contract = MappingProxyType(
            {
                "schema": "factor_gfn.stage6_train_preparation_contract.v1",
                "version": STAGE6_TWO_PHASE_VERSION,
                "implementation_sha256": _sha256_file(Path(__file__).resolve()),
                "base_fresh_evaluation_contract_fingerprint": (
                    fresh_evaluator.evaluation_contract_fingerprint
                ),
                "train_reuse_overlay_fingerprint": overlay.fingerprint,
                "split": "train_only",
                "validation": "not_loaded_from_cache_and_not_evaluated",
                "old_stage6_train_reuse": "forbidden",
                "train_prefilter_timing": "immediate_per_candidate",
                "train_prefilter_config_fingerprint": selection_config.fingerprint,
                "oos": "not_loaded_and_interface_rejected",
            }
        )
        self.evaluation_contract_fingerprint = _stable_hash(dict(self.evaluation_contract))

    def resolve_candidate_identity(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.fresh_evaluator.resolve_candidate_identity(candidate)

    def has_reusable_train(self, candidate: Mapping[str, Any]) -> bool:
        return self.overlay.get(str(candidate["current_structural_hash"])) is not None

    def planned_path(self, candidate: Mapping[str, Any]) -> str:
        return (
            "verified_stage5_train_preparation_reuse"
            if self.has_reusable_train(candidate)
            else "stage6_fresh_train_preparation"
        )

    def _validation_placeholder(self) -> dict[str, Any]:
        split = self.context.get_split_data("validation")
        return {
            "availability": TRAIN_NOT_EVALUATED_VALIDATION,
            "metric_origin": TRAIN_NOT_EVALUATED_VALIDATION,
            "requested_date_range": [
                split.boundary.requested_start,
                split.boundary.requested_end,
            ],
            "actual_date_range": [split.boundary.actual_start, split.boundary.actual_end],
            "rebalance_dates": None,
            "rebalance_periods": int(split.rebalance_dates.size),
            "ic": {"mean": None, "valid_periods": 0},
            "long": {"direction": None, "annualized_ir": None, "excess_series": {"dates": None, "values": None}},
            "barra": {"max_abs_correlation": None, "correlations": {}, "common_valid_periods": {}},
            "factor_finite_coverage": {"finite_universe_values": None, "eligible_universe_values": None, "rate": None},
        }

    def evaluate(self, candidate: Mapping[str, Any]) -> Stage6CandidateEvaluationResult:
        total_started = time.perf_counter()
        expression, identity = self.fresh_evaluator._expression(candidate)
        overlay_record = self.overlay.get(str(identity["structural_hash"]))
        factor_seconds = 0.0
        if overlay_record is not None:
            train_started = time.perf_counter()
            train = self.mixed_evaluator._reused_train_payload(overlay_record)
            direction = int(train["long"]["direction"])
            train_seconds = time.perf_counter() - train_started
            evaluation_path = "verified_stage5_train_preparation_reuse"
        else:
            factor_started = time.perf_counter()
            factor = np.asarray(
                self.fresh_evaluator._interpreter.evaluate(expression), dtype=np.float64
            )
            factor_seconds = time.perf_counter() - factor_started
            expected_shape = (self.context.dates.size, self.context.stocks.size)
            if factor.shape != expected_shape:
                raise RuntimeError(
                    f"FactorInterpreter returned {factor.shape}; expected {expected_shape}"
                )
            train_started = time.perf_counter()
            prepared = self.fresh_evaluator._prepare_split(factor, "train")
            train_ic = prepared.result["ic"]["mean"]
            direction = (
                infer_long_direction(float(train_ic))
                if train_ic is not None and train_ic != 0.0
                else None
            )
            self.fresh_evaluator._apply_direction(prepared, direction)
            train = {**prepared.result, "metric_origin": TRAIN_METRIC_ORIGIN_FRESH}
            train_seconds = time.perf_counter() - train_started
            evaluation_path = "stage6_fresh_train_preparation"
        train = {
            **train,
            "train_prefilter": train_prefilter_decision(train, self.selection_config),
        }
        validation = self._validation_placeholder()
        invalid_reasons = () if direction is not None else ("train_direction_unavailable",)
        return _build_result(
            evaluator=self,
            expression=identity,
            source_identity=_source_identity(
                self.fresh_evaluator,
                candidate,
                evaluation_path,
                old_stage6_train_reuse=False,
                validation_evaluation_count=0,
                **(
                    {
                        "train_reuse_fallback_reason": (
                            self.overlay.fresh_train_fallback_reason
                        )
                    }
                    if overlay_record is None
                    else {}
                ),
            ),
            status="completed" if direction is not None else "completed_invalid",
            invalid_reasons=invalid_reasons,
            direction=direction,
            train=train,
            validation=validation,
            coverage={
                "train": _plain(train.get("factor_finite_coverage", {})),
                "validation": _plain(validation["factor_finite_coverage"]),
            },
            factor_seconds=factor_seconds,
            train_seconds=train_seconds,
            validation_seconds=0.0,
            total_seconds=time.perf_counter() - total_started,
        )


class _CacheOnlyTrainPreparationEvaluator:
    """Resolve an immutable eligible snapshot; evaluation is forbidden."""

    def __init__(
        self,
        *,
        context_fingerprint: str,
        evaluation_contract_fingerprint: str,
        compatibility_audit_fingerprint: str,
        accepted_registry_fingerprint: str,
        reusable_hashes: set[str],
    ) -> None:
        self.context = SimpleNamespace(fingerprint=context_fingerprint)
        self.evaluation_contract_fingerprint = evaluation_contract_fingerprint
        self.compatibility_audit_fingerprint = compatibility_audit_fingerprint
        self.accepted_registry_fingerprint = accepted_registry_fingerprint
        self._reusable_hashes = frozenset(reusable_hashes)

    def resolve_candidate_identity(
        self, candidate: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return {
            "structural_hash": candidate["current_structural_hash"],
            "formula": candidate["formula"],
            "prefix_token_ids": list(candidate["prefix_token_ids"]),
            "node_count": candidate["node_count"],
            "depth": candidate["depth"],
        }

    def has_reusable_train(self, candidate: Mapping[str, Any]) -> bool:
        return str(candidate["current_structural_hash"]) in self._reusable_hashes

    def planned_path(self, candidate: Mapping[str, Any]) -> str:
        return (
            "verified_stage5_train_preparation_reuse"
            if self.has_reusable_train(candidate)
            else "stage6_fresh_train_preparation"
        )

    def evaluate(self, candidate: Mapping[str, Any]) -> Stage6CandidateEvaluationResult:
        raise Stage6TwoPhaseIntegrityError(
            "resource-limited eligible snapshot forbids fresh Train evaluation"
        )


def _entry_payload(
    *, schema: str, scope: str, run: Any, database_path: Path,
    run_artifact_root: Path, extra: Mapping[str, Any]
) -> dict[str, Any]:
    stable = {
        "schema": schema,
        "pipeline_version": STAGE6_TWO_PHASE_VERSION,
        "evaluation_run_id": run.run_id,
        "evaluation_run_scope": scope,
        "candidate_count": run.candidate_count,
        "context_fingerprint": run.manifest["context_fingerprint"],
        "evaluation_contract_fingerprint": run.manifest[
            "evaluation_contract_fingerprint"
        ],
        "database_path": str(database_path),
        "run_artifact_root": str(run_artifact_root),
        "oos": "not_loaded_not_evaluated",
        **_plain(extra),
    }
    return {**stable, "entry_manifest_fingerprint": _stable_hash(stable)}


def _load_entry(
    path: str | Path, schema: str, scope: str | Iterable[str]
) -> dict[str, Any]:
    manifest = _read_json(Path(path).resolve())
    stable = {key: value for key, value in manifest.items() if key != "entry_manifest_fingerprint"}
    allowed_scopes = {scope} if isinstance(scope, str) else set(scope)
    if (
        manifest.get("schema") != schema
        or manifest.get("evaluation_run_scope") not in allowed_scopes
    ):
        raise Stage6TwoPhaseIntegrityError("two-phase entry schema or scope mismatch")
    if _stable_hash(stable) != manifest.get("entry_manifest_fingerprint"):
        raise Stage6TwoPhaseIntegrityError("two-phase entry fingerprint mismatch")
    if manifest.get("oos") != "not_loaded_not_evaluated":
        raise Stage6TwoPhaseIntegrityError("two-phase entry OOS lock changed")
    return manifest


def _freeze_train_pass_manifest(
    *, store: EvaluationStore, run_id: str, candidates: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any], output_root: Path,
    selection_config: Stage6SelectionConfig = Stage6SelectionConfig(),
    provisional_universe: Mapping[str, Any] | None = None,
) -> Path:
    verified = store.load_verified_run_results(run_id)
    if verified.manifest.get("scope") not in {
        STAGE6_TRAIN_PREPARATION_SCOPE,
        STAGE6_RESOURCE_LIMITED_TRAIN_SCOPE,
    }:
        raise Stage6TwoPhaseIntegrityError("Train-pass freeze requires the Train-only scope")
    if len(verified.records) != len(candidates):
        raise Stage6TwoPhaseIntegrityError("Train result count differs from accepted registry")
    candidates_by_hash = {str(row["current_structural_hash"]): dict(row) for row in candidates}
    decisions: list[dict[str, Any]] = []
    pass_candidates: list[dict[str, Any]] = []
    failed_condition_counts: dict[str, int] = {}
    reused = fresh = 0
    for ordinal, stored in enumerate(verified.records):
        result = stored["result"]
        if result.get("validation_evaluation_seconds") != 0.0:
            raise Stage6TwoPhaseIntegrityError("Train phase evaluated Validation")
        if result.get("validation", {}).get("availability") != TRAIN_NOT_EVALUATED_VALIDATION:
            raise Stage6TwoPhaseIntegrityError("Train phase contains Validation data")
        structural_hash = str(stored["structural_hash"])
        candidate = candidates_by_hash.get(structural_hash)
        if candidate is None:
            raise Stage6TwoPhaseIntegrityError("Train result candidate left accepted registry")
        stored_decision = result.get("train", {}).get("train_prefilter")
        if not isinstance(stored_decision, Mapping):
            raise Stage6TwoPhaseIntegrityError(
                "Train result lacks its persisted prefilter decision"
            )
        recomputed_decision = train_prefilter_decision(result["train"], selection_config)
        if _plain(stored_decision) != recomputed_decision:
            raise Stage6TwoPhaseIntegrityError(
                "persisted Train prefilter decision differs from frozen conditions"
            )
        decision = _plain(stored_decision)
        for code in decision["failed_conditions"]:
            failed_condition_counts[str(code)] = failed_condition_counts.get(str(code), 0) + 1
        path = str(result.get("source_identity", {}).get("evaluation_path", ""))
        if path == "verified_stage5_train_preparation_reuse":
            reused += 1
        else:
            fresh += 1
        passed = decision["status"] == TRAIN_PREFILTER_PASSED
        decisions.append(
            {
                "ordinal": ordinal,
                "structural_hash": structural_hash,
                "train_result_fingerprint": result["result_fingerprint"],
                "train_direction": result.get("train_direction"),
                "train_metric_origin": result.get("train", {}).get("metric_origin"),
                "status": decision["status"],
                "condition_results": decision["condition_results"],
                "failed_conditions": decision["failed_conditions"],
            }
        )
        if passed:
            pass_candidates.append(candidate)
    artifact_root = output_root / "train_pass_manifest"
    decisions_path = artifact_root / "train_prefilter_results.jsonl"
    candidates_path = artifact_root / "train_pass_candidates.jsonl"
    _write_jsonl_atomic(decisions_path, decisions)
    _write_jsonl_atomic(candidates_path, pass_candidates)
    stable = {
        "schema": STAGE6_TRAIN_PASS_MANIFEST_SCHEMA,
        "pipeline_version": STAGE6_TWO_PHASE_VERSION,
        "train_entry_manifest_fingerprint": entry["entry_manifest_fingerprint"],
        "train_evaluation_run_id": run_id,
        "train_ordered_result_set_fingerprint": verified.ordered_result_set_fingerprint,
        "accepted_registry_fingerprint": verified.manifest["accepted_registry_fingerprint"],
        "selection_config_fingerprint": selection_config.fingerprint,
        "candidate_count": len(candidates),
        "train_pass_count": len(pass_candidates),
        "train_prefilter_failed_count": len(candidates) - len(pass_candidates),
        "verified_stage5_train_reuse_count": reused,
        "stage6_fresh_train_count": fresh,
        "validation_evaluation_count": 0,
        "failed_condition_counts": dict(sorted(failed_condition_counts.items())),
        "artifacts": {
            decisions_path.name: {"sha256": _sha256_file(decisions_path), "size_bytes": decisions_path.stat().st_size, "count": len(decisions)},
            candidates_path.name: {"sha256": _sha256_file(candidates_path), "size_bytes": candidates_path.stat().st_size, "count": len(pass_candidates)},
        },
        "ordered_train_pass_hashes_fingerprint": _stable_hash(
            [row["current_structural_hash"] for row in pass_candidates]
        ),
        "oos": "not_loaded_not_evaluated",
    }
    if provisional_universe is not None:
        universe = _plain(provisional_universe)
        if int(universe["evaluation_eligible_count"]) != len(candidates):
            raise Stage6TwoPhaseIntegrityError(
                "provisional universe eligible count differs from Train candidates"
            )
        stable["provisional_evaluation_universe"] = {
            "fingerprint": universe["provisional_evaluation_universe_fingerprint"],
            "original_accepted_candidate_count": universe[
                "original_accepted_candidate_count"
            ],
            "evaluation_eligible_count": universe["evaluation_eligible_count"],
            "deferred_candidate_count": universe["deferred_candidate_count"],
            "deferred_reason_counts": universe["deferred_reason_counts"],
            "pool_basis": "resource_limited_evaluation_eligible_universe_only",
            "final_stage_may_reevaluate_deferred": True,
        }
    manifest = {**stable, "train_pass_manifest_fingerprint": _stable_hash(stable)}
    manifest_path = artifact_root / "train_pass_manifest.json"
    if manifest_path.is_file() and _read_json(manifest_path) != manifest:
        raise Stage6TwoPhaseIntegrityError("existing Train-pass manifest changed")
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def load_stage6_train_pass_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path)
    stable = {key: value for key, value in manifest.items() if key != "train_pass_manifest_fingerprint"}
    if manifest.get("schema") != STAGE6_TRAIN_PASS_MANIFEST_SCHEMA:
        raise Stage6TwoPhaseIntegrityError("Train-pass manifest schema mismatch")
    if _stable_hash(stable) != manifest.get("train_pass_manifest_fingerprint"):
        raise Stage6TwoPhaseIntegrityError("Train-pass manifest fingerprint mismatch")
    if manifest.get("validation_evaluation_count") != 0:
        raise Stage6TwoPhaseIntegrityError("Train phase Validation count is not zero")
    artifact_name = "train_pass_candidates.jsonl"
    metadata = manifest.get("artifacts", {}).get(artifact_name)
    artifact_path = manifest_path.parent / artifact_name
    if not isinstance(metadata, Mapping) or not artifact_path.is_file():
        raise Stage6TwoPhaseIntegrityError("Train-pass candidate artifact missing")
    if artifact_path.stat().st_size != int(metadata.get("size_bytes", -1)) or _sha256_file(artifact_path) != metadata.get("sha256"):
        raise Stage6TwoPhaseIntegrityError("Train-pass candidate artifact changed")
    candidates = list(_iter_jsonl(artifact_path))
    if len(candidates) != int(manifest.get("train_pass_count", -1)):
        raise Stage6TwoPhaseIntegrityError("Train-pass candidate count mismatch")
    if _stable_hash([row["current_structural_hash"] for row in candidates]) != manifest.get("ordered_train_pass_hashes_fingerprint"):
        raise Stage6TwoPhaseIntegrityError("Train-pass candidate order changed")
    return manifest, candidates


def run_current_stage6_train_preparation(
    *, compatibility_manifest_path: str | Path, overlay_manifest_path: str | Path,
    output_root: str | Path, max_new_evaluations: int | None = None,
    retry_failed: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    data_paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    compatibility, candidates = load_stage6_accepted_registry(compatibility_manifest_path)
    context = build_stage6_evaluation_context(paths=data_paths)
    fresh = Stage6CandidateEvaluator(
        context,
        compatibility_audit_fingerprint=str(compatibility["audit_fingerprint"]),
        accepted_registry_fingerprint=str(compatibility["accepted_registry_fingerprint"]),
    )
    overlay = Stage6TrainReuseOverlay.load(overlay_manifest_path)
    evaluator = Stage6TrainPreparationEvaluator(fresh, overlay)
    reused_candidates = [row for row in candidates if evaluator.has_reusable_train(row)]
    fresh_candidates = [row for row in candidates if not evaluator.has_reusable_train(row)]
    ordered_candidates = [*reused_candidates, *fresh_candidates]
    database_path = root / "evaluation_store" / "stage6_train_preparation.sqlite"
    run_artifact_root = root / "evaluation_runs"
    entry_path = root / "train_preparation_entry_manifest.json"
    with EvaluationStore(database_path, run_artifact_root) as store:
        frozen = store.create_run(
            ordered_candidates, evaluator, scope=STAGE6_TRAIN_PREPARATION_SCOPE
        )
        entry = _entry_payload(
            schema=STAGE6_TRAIN_ENTRY_SCHEMA, scope=STAGE6_TRAIN_PREPARATION_SCOPE,
            run=frozen, database_path=database_path, run_artifact_root=run_artifact_root,
            extra={
                "compatibility_audit_fingerprint": compatibility["audit_fingerprint"],
                "accepted_registry_fingerprint": compatibility["accepted_registry_fingerprint"],
                "train_reuse_overlay_fingerprint": overlay.fingerprint,
                "base_fresh_evaluation_contract_fingerprint": (
                    fresh.evaluation_contract_fingerprint
                ),
                "old_stage6_train_reuse": "forbidden",
                "validation_evaluation_count": 0,
                "verified_stage5_train_total": len(reused_candidates),
                "fresh_train_total": len(fresh_candidates),
                "candidate_order": "verified_stage5_train_then_fresh_train",
            },
        )
        if entry_path.is_file() and _read_json(entry_path) != entry:
            raise Stage6TwoPhaseIntegrityError("existing Train entry changed")
        _write_json_atomic(entry_path, entry)
        def train_progress(event: Mapping[str, Any]) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    **event,
                    "verified_stage5_train_total": len(reused_candidates),
                    "fresh_train_total": len(fresh_candidates),
                    "validation_completed": 0,
                }
            )

        summary = Stage6EvaluationRunner(store, evaluator).run(
            frozen.run_id, max_new_evaluations=max_new_evaluations,
            retry_failed=retry_failed, progress_callback=train_progress,
        )
        pass_manifest_path = None
        if summary.run_status == "complete":
            pass_manifest_path = _freeze_train_pass_manifest(
                store=store, run_id=frozen.run_id, candidates=ordered_candidates,
                entry=entry, output_root=root,
            )
    return {
        "entry_manifest_path": str(entry_path),
        "train_pass_manifest_path": str(pass_manifest_path) if pass_manifest_path else None,
        "evaluation_run_id": frozen.run_id,
        "invocation": asdict(summary),
        "verified_stage5_train_total": len(reused_candidates),
        "fresh_train_total": len(fresh_candidates),
    }


def load_stage6_provisional_evaluation_universe(
    path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path)
    stable = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "provisional_evaluation_universe_fingerprint",
            "artifacts",
            "created_at_utc",
            "created_at_excluded_from_fingerprint",
            "cache_only_resolution",
            "train_entry_manifest_path",
            "train_pass_manifest_path",
        }
    }
    if manifest.get("schema") != STAGE6_PROVISIONAL_UNIVERSE_SCHEMA:
        raise Stage6TwoPhaseIntegrityError("provisional universe schema mismatch")
    if _stable_hash(stable) != manifest.get(
        "provisional_evaluation_universe_fingerprint"
    ):
        raise Stage6TwoPhaseIntegrityError("provisional universe fingerprint mismatch")
    for name, metadata in manifest.get("artifacts", {}).items():
        artifact_path = manifest_path.parent / name
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != int(metadata.get("size_bytes", -1))
            or _sha256_file(artifact_path) != metadata.get("sha256")
        ):
            raise Stage6TwoPhaseIntegrityError(
                f"provisional universe artifact changed: {name}"
            )
    return manifest


def freeze_current_stage6_resource_limited_universe(
    *,
    compatibility_manifest_path: str | Path,
    overlay_manifest_path: str | Path,
    source_train_entry_manifest_path: str | Path,
    source_set_manifest_path: str | Path,
    v6_equivalence_manifest_path: str | Path,
    output_root: str | Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Freeze only already-trusted Train results; never evaluate a candidate."""

    from . import stage6_train_reuse as reuse

    compatibility, accepted = load_stage6_accepted_registry(
        compatibility_manifest_path
    )
    accepted_by_hash = {
        str(candidate["current_structural_hash"]): dict(candidate)
        for candidate in accepted
    }
    source_entry = _load_entry(
        source_train_entry_manifest_path,
        STAGE6_TRAIN_ENTRY_SCHEMA,
        STAGE6_TRAIN_PREPARATION_SCOPE,
    )
    if source_entry.get("accepted_registry_fingerprint") != compatibility.get(
        "accepted_registry_fingerprint"
    ):
        raise Stage6TwoPhaseIntegrityError(
            "source Train entry and accepted registry differ"
        )
    overlay = Stage6TrainReuseOverlay.load(overlay_manifest_path)
    if source_entry.get("train_reuse_overlay_fingerprint") != overlay.fingerprint:
        raise Stage6TwoPhaseIntegrityError("source Train entry and overlay differ")

    v6_manifest_path = Path(v6_equivalence_manifest_path).resolve()
    v6_manifest = _read_json(v6_manifest_path)
    if (
        v6_manifest.get("schema") != reuse.V6_EQUIVALENCE_MANIFEST_SCHEMA
        or v6_manifest.get("conclusion")
        != reuse.V6_EQUIVALENCE_EVIDENCE_FAILED
        or v6_manifest.get("reuse_authorized") is not False
        or v6_manifest.get("overlay_written") is not False
        or v6_manifest.get("current_overlay_fingerprint") != overlay.fingerprint
    ):
        raise Stage6TwoPhaseIntegrityError(
            "v6 non-equivalence evidence is missing or not fail-closed"
        )
    verification_metadata = v6_manifest.get("artifacts", {}).get(
        "v6_equivalence_verification.jsonl"
    )
    verification_path = v6_manifest_path.parent / "v6_equivalence_verification.jsonl"
    if (
        not isinstance(verification_metadata, Mapping)
        or not verification_path.is_file()
        or verification_path.stat().st_size
        != int(verification_metadata.get("size_bytes", -1))
        or _sha256_file(verification_path) != verification_metadata.get("sha256")
    ):
        raise Stage6TwoPhaseIntegrityError("v6 equivalence evidence artifact changed")

    source_set, snapshots = reuse._verify_source_set(
        Path(source_set_manifest_path).resolve()
    )
    if source_set.get("source_set_fingerprint") != v6_manifest.get(
        "source_set_fingerprint"
    ):
        raise Stage6TwoPhaseIntegrityError("v6 evidence and source set differ")
    batch = next(
        (
            item
            for item in reuse._batch_sources(snapshots)
            if item["batch_id"] == reuse.V6_EQUIVALENCE_TARGET_BATCH_ID
        ),
        None,
    )
    if batch is None:
        raise Stage6TwoPhaseIntegrityError("frozen v6 source batch is absent")
    v6_records: dict[str, list[dict[str, Any]]] = {}
    accepted_hashes = set(accepted_by_hash)
    for snapshot in snapshots:
        if snapshot["source_id"] not in batch["source_ids"]:
            continue
        for record in reuse._read_source_metric_records(snapshot, accepted_hashes):
            v6_records.setdefault(str(record["structural_hash"]), []).append(record)
    v6_complete_hashes = {
        structural_hash
        for structural_hash, records in v6_records.items()
        if reuse._canonical_metric_record(records)[0] is not None
    }

    source_database_path = Path(source_entry["database_path"]).resolve()
    source_run_artifact_root = Path(source_entry["run_artifact_root"]).resolve()
    with EvaluationStore(source_database_path, source_run_artifact_root) as source_store:
        source_status_row = source_store.connection.execute(
            "SELECT status FROM runs WHERE run_id=?",
            (source_entry["evaluation_run_id"],),
        ).fetchone()
        if source_status_row is None:
            raise Stage6TwoPhaseIntegrityError("source Train run is absent")
        partial = source_store.load_verified_completed_results(
            source_entry["evaluation_run_id"]
        )
    if partial.manifest.get("scope") != STAGE6_TRAIN_PREPARATION_SCOPE:
        raise Stage6TwoPhaseIntegrityError("partial source is not the Train-only run")

    eligible_candidates: list[dict[str, Any]] = []
    eligible_refs: list[dict[str, Any]] = []
    eligible_hashes: set[str] = set()
    reused_hashes: set[str] = set()
    fresh_hashes: set[str] = set()
    path_counts: dict[str, int] = {}
    for stored in partial.records:
        structural_hash = str(stored["structural_hash"])
        candidate = accepted_by_hash.get(structural_hash)
        if candidate is None or structural_hash in eligible_hashes:
            raise Stage6TwoPhaseIntegrityError(
                "trusted partial result is absent or duplicated in accepted registry"
            )
        result = stored["result"]
        if (
            result.get("validation_evaluation_seconds") != 0.0
            or result.get("validation", {}).get("availability")
            != TRAIN_NOT_EVALUATED_VALIDATION
        ):
            raise Stage6TwoPhaseIntegrityError(
                "trusted Train snapshot contains Validation data"
            )
        decision = result.get("train", {}).get("train_prefilter")
        if not isinstance(decision, Mapping) or _plain(decision) != train_prefilter_decision(
            result["train"], Stage6SelectionConfig()
        ):
            raise Stage6TwoPhaseIntegrityError(
                "trusted Train snapshot has no valid persisted prefilter decision"
            )
        evaluation_path = str(
            result.get("source_identity", {}).get("evaluation_path", "")
        )
        if evaluation_path not in {
            "verified_stage5_train_preparation_reuse",
            "stage6_fresh_train_preparation",
        }:
            raise Stage6TwoPhaseIntegrityError(
                "trusted Train snapshot has an unsupported evaluation path"
            )
        path_counts[evaluation_path] = path_counts.get(evaluation_path, 0) + 1
        if evaluation_path == "verified_stage5_train_preparation_reuse":
            reused_hashes.add(structural_hash)
        else:
            fresh_hashes.add(structural_hash)
        eligible_hashes.add(structural_hash)
        eligible_candidates.append(candidate)
        eligible_refs.append(
            {
                "ordinal": len(eligible_refs),
                "source_run_ordinal": stored["ordinal"],
                "structural_hash": structural_hash,
                "cache_key": stored["cache_key"],
                "result_fingerprint": stored["result_fingerprint"],
                "status": result["status"],
                "evaluation_path": evaluation_path,
                "validation_evaluation_count": 0,
            }
        )

    overlay_hashes = set(overlay.records)
    if reused_hashes != overlay_hashes:
        raise Stage6TwoPhaseIntegrityError(
            "verified overlay hashes differ from completed reuse-result hashes"
        )
    if reused_hashes & fresh_hashes:
        raise Stage6TwoPhaseIntegrityError(
            "verified overlay and current-contract fresh hashes overlap"
        )
    if eligible_hashes != reused_hashes | fresh_hashes:
        raise Stage6TwoPhaseIntegrityError(
            "eligible hashes differ from reuse/fresh set union"
        )
    if (len(reused_hashes), len(fresh_hashes), len(eligible_hashes)) != (
        9328,
        1645,
        10973,
    ):
        raise Stage6TwoPhaseIntegrityError(
            "frozen resource-limited set is not 9,328 + 1,645 = 10,973"
        )

    deferred_rows: list[dict[str, Any]] = []
    deferred_reason_counts: dict[str, int] = {}
    for candidate in accepted:
        structural_hash = str(candidate["current_structural_hash"])
        if structural_hash in eligible_hashes:
            continue
        reason = (
            HISTORICAL_TRAIN_CONTRACT_NOT_EQUIVALENT
            if structural_hash in v6_complete_hashes
            else NO_TRUSTED_TRAIN_RESULT_RESOURCE_LIMITED
        )
        deferred_reason_counts[reason] = deferred_reason_counts.get(reason, 0) + 1
        deferred_rows.append(
            {
                "schema": "factor_gfn.stage6_deferred_candidate.v1",
                "status": DEFERRED_TRAIN_RECOMPUTE,
                "reason": reason,
                "structural_hash": structural_hash,
                "provenance": {
                    "origin_ids": list(candidate.get("origin_ids", [])),
                    "source_ids": list(candidate.get("source_ids", [])),
                    "source_claimed_structural_hash": candidate.get(
                        "source_claimed_structural_hash"
                    ),
                },
                "candidate": dict(candidate),
            }
        )
    if len(eligible_candidates) + len(deferred_rows) != len(accepted):
        raise Stage6TwoPhaseIntegrityError("eligible/deferred partition is incomplete")

    source_database_sha256 = _sha256_file(source_database_path)
    stable_universe = {
        "schema": STAGE6_PROVISIONAL_UNIVERSE_SCHEMA,
        "pipeline_version": STAGE6_TWO_PHASE_VERSION,
        "resource_limited_provisional": True,
        "source_train_entry_manifest_fingerprint": source_entry[
            "entry_manifest_fingerprint"
        ],
        "source_train_evaluation_run_id": source_entry["evaluation_run_id"],
        "source_train_run_status_at_freeze": str(source_status_row["status"]),
        "source_completed_result_set_fingerprint": (
            partial.ordered_result_set_fingerprint
        ),
        "source_database_sha256_at_freeze": source_database_sha256,
        "accepted_registry_fingerprint": compatibility[
            "accepted_registry_fingerprint"
        ],
        "compatibility_audit_fingerprint": compatibility["audit_fingerprint"],
        "train_reuse_overlay_fingerprint": overlay.fingerprint,
        "v6_non_equivalence_fingerprint": v6_manifest[
            "v6_equivalence_fingerprint"
        ],
        "source_set_fingerprint": source_set["source_set_fingerprint"],
        "original_accepted_candidate_count": len(accepted),
        "evaluation_eligible_count": len(eligible_candidates),
        "deferred_candidate_count": len(deferred_rows),
        "deferred_status": DEFERRED_TRAIN_RECOMPUTE,
        "deferred_reason_counts": dict(sorted(deferred_reason_counts.items())),
        "trusted_train_path_counts": dict(sorted(path_counts.items())),
        "eligible_set_relationship": {
            "deduplication_key": "structural_hash",
            "verified_overlay_count": len(reused_hashes),
            "current_contract_fresh_count": len(fresh_hashes),
            "intersection_count": len(reused_hashes & fresh_hashes),
            "union_count": len(reused_hashes | fresh_hashes),
            "verified_overlay_hashes_fingerprint": _stable_hash(
                sorted(reused_hashes)
            ),
            "current_contract_fresh_hashes_fingerprint": _stable_hash(
                sorted(fresh_hashes)
            ),
            "union_hashes_fingerprint": _stable_hash(sorted(eligible_hashes)),
        },
        "ordered_eligible_hashes_fingerprint": _stable_hash(
            [candidate["current_structural_hash"] for candidate in eligible_candidates]
        ),
        "eligible_result_refs_fingerprint": _stable_hash(eligible_refs),
        "ordered_deferred_hashes_fingerprint": _stable_hash(
            [row["structural_hash"] for row in deferred_rows]
        ),
        "validation_evaluation_count": 0,
        "provisional_pool_basis": "resource_limited_evaluation_eligible_universe_only",
        "final_stage_may_reevaluate_deferred": True,
        "oos": "not_loaded_not_evaluated",
    }
    universe_fingerprint = _stable_hash(stable_universe)
    root = Path(output_root).resolve()
    target = root / universe_fingerprint
    manifest_path = target / "provisional_evaluation_universe_manifest.json"
    if target.exists():
        existing = load_stage6_provisional_evaluation_universe(manifest_path)
        return {
            "provisional_evaluation_universe_manifest_path": str(manifest_path),
            "train_entry_manifest_path": existing["train_entry_manifest_path"],
            "train_pass_manifest_path": existing["train_pass_manifest_path"],
            "counts": {
                "original_accepted_candidate_count": existing[
                    "original_accepted_candidate_count"
                ],
                "evaluation_eligible_count": existing["evaluation_eligible_count"],
                "deferred_candidate_count": existing["deferred_candidate_count"],
                "deferred_reason_counts": existing["deferred_reason_counts"],
            },
            "reused_existing_artifact": True,
        }

    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{universe_fingerprint}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        eligible_path = temporary / "evaluation_eligible_candidates.jsonl"
        deferred_path = temporary / "deferred_candidates.jsonl"
        refs_path = temporary / "eligible_train_result_refs.jsonl"
        _write_jsonl_atomic(eligible_path, eligible_candidates)
        _write_jsonl_atomic(deferred_path, deferred_rows)
        _write_jsonl_atomic(refs_path, eligible_refs)

        cloned_database = temporary / "evaluation_store" / "stage6_train_preparation.sqlite"
        cloned_database.parent.mkdir(parents=True)
        source_connection = sqlite3.connect(
            f"file:{source_database_path.as_posix()}?mode=ro", uri=True
        )
        destination_connection = sqlite3.connect(cloned_database)
        try:
            source_connection.backup(destination_connection)
        finally:
            destination_connection.close()
            source_connection.close()
        if _sha256_file(source_database_path) != source_database_sha256:
            raise Stage6TwoPhaseIntegrityError(
                "source Train database changed during resource-limited freeze"
            )

        final_database = target / "evaluation_store" / "stage6_train_preparation.sqlite"
        final_run_root = target / "evaluation_runs"
        evaluator = _CacheOnlyTrainPreparationEvaluator(
            context_fingerprint=source_entry["context_fingerprint"],
            evaluation_contract_fingerprint=source_entry[
                "evaluation_contract_fingerprint"
            ],
            compatibility_audit_fingerprint=compatibility["audit_fingerprint"],
            accepted_registry_fingerprint=compatibility[
                "accepted_registry_fingerprint"
            ],
            reusable_hashes=set(overlay.records),
        )
        with EvaluationStore(cloned_database, temporary / "evaluation_runs") as store:
            frozen = store.create_run(
                eligible_candidates,
                evaluator,
                scope=STAGE6_RESOURCE_LIMITED_TRAIN_SCOPE,
            )
            summary = Stage6EvaluationRunner(store, evaluator).run(
                frozen.run_id,
                progress_callback=progress_callback,
            )
            if (
                summary.run_status != "complete"
                or summary.newly_evaluated != 0
                or summary.cache_hits != len(eligible_candidates)
            ):
                raise Stage6TwoPhaseIntegrityError(
                    "resource-limited Train run was not resolved entirely by cache hits"
                )
            entry = _entry_payload(
                schema=STAGE6_TRAIN_ENTRY_SCHEMA,
                scope=STAGE6_RESOURCE_LIMITED_TRAIN_SCOPE,
                run=frozen,
                database_path=final_database,
                run_artifact_root=final_run_root,
                extra={
                    "compatibility_audit_fingerprint": compatibility[
                        "audit_fingerprint"
                    ],
                    "accepted_registry_fingerprint": compatibility[
                        "accepted_registry_fingerprint"
                    ],
                    "train_reuse_overlay_fingerprint": overlay.fingerprint,
                    "base_fresh_evaluation_contract_fingerprint": source_entry[
                        "base_fresh_evaluation_contract_fingerprint"
                    ],
                    "source_train_entry_manifest_fingerprint": source_entry[
                        "entry_manifest_fingerprint"
                    ],
                    "source_train_evaluation_run_id": source_entry[
                        "evaluation_run_id"
                    ],
                    "provisional_evaluation_universe_fingerprint": universe_fingerprint,
                    "original_accepted_candidate_count": len(accepted),
                    "evaluation_eligible_count": len(eligible_candidates),
                    "deferred_candidate_count": len(deferred_rows),
                    "deferred_reason_counts": dict(
                        sorted(deferred_reason_counts.items())
                    ),
                    "resource_limited_provisional": True,
                    "fresh_train_recompute": "forbidden",
                    "validation_evaluation_count": 0,
                },
            )
            entry_path = temporary / "train_preparation_entry_manifest.json"
            _write_json_atomic(entry_path, entry)
            universe_for_pass = {
                **stable_universe,
                "provisional_evaluation_universe_fingerprint": universe_fingerprint,
            }
            pass_path = _freeze_train_pass_manifest(
                store=store,
                run_id=frozen.run_id,
                candidates=eligible_candidates,
                entry=entry,
                output_root=temporary,
                provisional_universe=universe_for_pass,
            )

        artifact_paths = {
            "evaluation_eligible_candidates.jsonl": eligible_path,
            "deferred_candidates.jsonl": deferred_path,
            "eligible_train_result_refs.jsonl": refs_path,
            "train_preparation_entry_manifest.json": entry_path,
            "train_pass_manifest/train_pass_manifest.json": pass_path,
        }
        manifest = {
            **stable_universe,
            "provisional_evaluation_universe_fingerprint": universe_fingerprint,
            "cache_only_resolution": {
                "evaluation_run_id": frozen.run_id,
                "run_status": summary.run_status,
                "cache_hits": summary.cache_hits,
                "newly_evaluated": summary.newly_evaluated,
                "validation_evaluation_count": 0,
            },
            "train_entry_manifest_path": str(
                target / "train_preparation_entry_manifest.json"
            ),
            "train_pass_manifest_path": str(
                target / "train_pass_manifest" / "train_pass_manifest.json"
            ),
            "artifacts": {
                name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for name, path in artifact_paths.items()
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
            "created_at_excluded_from_fingerprint": True,
        }
        _write_json_atomic(
            temporary / "provisional_evaluation_universe_manifest.json", manifest
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    load_stage6_provisional_evaluation_universe(manifest_path)
    return {
        "provisional_evaluation_universe_manifest_path": str(manifest_path),
        "train_entry_manifest_path": str(
            target / "train_preparation_entry_manifest.json"
        ),
        "train_pass_manifest_path": str(
            target / "train_pass_manifest" / "train_pass_manifest.json"
        ),
        "counts": {
            "original_accepted_candidate_count": len(accepted),
            "evaluation_eligible_count": len(eligible_candidates),
            "deferred_candidate_count": len(deferred_rows),
            "deferred_reason_counts": dict(sorted(deferred_reason_counts.items())),
        },
        "cache_only_resolution": {
            "evaluation_run_id": frozen.run_id,
            "run_status": summary.run_status,
            "cache_hits": summary.cache_hits,
            "newly_evaluated": summary.newly_evaluated,
        },
        "reused_existing_artifact": False,
    }


def _load_train_records(
    entry_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    entry = _load_entry(
        entry_manifest_path,
        STAGE6_TRAIN_ENTRY_SCHEMA,
        {STAGE6_TRAIN_PREPARATION_SCOPE, STAGE6_RESOURCE_LIMITED_TRAIN_SCOPE},
    )
    with EvaluationStore(entry["database_path"], entry["run_artifact_root"]) as store:
        verified = store.load_verified_run_results(entry["evaluation_run_id"])
    return entry, {str(row["structural_hash"]): row["result"] for row in verified.records}


def _load_prior_validation_results(
    path: str | Path | None,
    *, current_context_fingerprint: str,
    allowed_contract_fingerprints: set[str],
) -> tuple[dict[str, Mapping[str, Any]], str]:
    if path is None:
        return {}, _stable_hash([])
    entry = load_stage6_full_entry_manifest(path)
    if entry.get("evaluation_run_scope") != LEGACY_FULL_EVALUATION_SCOPE:
        raise Stage6TwoPhaseIntegrityError("prior Validation seeds require the legacy full run")
    with EvaluationStore(entry["database_path"], entry["run_artifact_root"]) as store:
        verified = store.load_verified_completed_results(entry["evaluation_run_id"])
    accepted: dict[str, Mapping[str, Any]] = {}
    identities: list[dict[str, Any]] = []
    for row in verified.records:
        result = row["result"]
        contract = str(result.get("evaluation_contract_fingerprint"))
        if result.get("context_fingerprint") != current_context_fingerprint or contract not in allowed_contract_fingerprints:
            continue
        validation = result.get("validation")
        if not isinstance(validation, Mapping) or validation.get("availability") in {TRAIN_NOT_EVALUATED_VALIDATION, "not_evaluated_train_prefilter_failed"}:
            continue
        structural_hash = str(row["structural_hash"])
        accepted[structural_hash] = result
        identities.append({"structural_hash": structural_hash, "result_fingerprint": result["result_fingerprint"], "contract": contract})
    return accepted, _stable_hash(identities)


class Stage6ValidationFromFrozenTrainEvaluator:
    """Evaluate Validation only for the immutable Train-pass candidates."""

    def __init__(
        self, fresh_evaluator: Stage6CandidateEvaluator,
        *, train_records: Mapping[str, Mapping[str, Any]],
        train_pass_manifest: Mapping[str, Any],
        prior_validation_results: Mapping[str, Mapping[str, Any]],
        prior_validation_seed_set_fingerprint: str,
    ) -> None:
        self.fresh_evaluator = fresh_evaluator
        self.context = fresh_evaluator.context
        self.compatibility_audit_fingerprint = fresh_evaluator.compatibility_audit_fingerprint
        self.accepted_registry_fingerprint = fresh_evaluator.accepted_registry_fingerprint
        self.train_records = dict(train_records)
        self.train_pass_manifest = dict(train_pass_manifest)
        self.prior_validation_results = dict(prior_validation_results)
        self.evaluation_contract = MappingProxyType(
            {
                "schema": "factor_gfn.stage6_validation_from_frozen_train_contract.v1",
                "version": STAGE6_TWO_PHASE_VERSION,
                "implementation_sha256": _sha256_file(Path(__file__).resolve()),
                "base_fresh_evaluation_contract_fingerprint": fresh_evaluator.evaluation_contract_fingerprint,
                "train_pass_manifest_fingerprint": train_pass_manifest["train_pass_manifest_fingerprint"],
                "train_ordered_result_set_fingerprint": train_pass_manifest["train_ordered_result_set_fingerprint"],
                "prior_validation_seed_set_fingerprint": prior_validation_seed_set_fingerprint,
                "train_prefilter": "frozen_input_never_recomputed",
                "split": "validation_only",
                "validation_direction": "frozen_train_direction",
                "old_stage6_train_reuse": "forbidden",
                "oos": "not_loaded_and_interface_rejected",
            }
        )
        self.evaluation_contract_fingerprint = _stable_hash(dict(self.evaluation_contract))

    def resolve_candidate_identity(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.fresh_evaluator.resolve_candidate_identity(candidate)

    def has_reusable_train(self, candidate: Mapping[str, Any]) -> bool:
        return True

    def planned_path(self, candidate: Mapping[str, Any]) -> str:
        structural_hash = str(candidate["current_structural_hash"])
        prior = self.prior_validation_results.get(structural_hash)
        frozen = self.train_records.get(structural_hash)
        if (
            prior is not None
            and frozen is not None
            and prior.get("train_direction") == frozen.get("train_direction")
            and _plain(prior.get("expression", {}))
            == _plain(self.fresh_evaluator.resolve_candidate_identity(candidate))
        ):
            return "verified_prior_stage6_validation_reuse"
        return "stage6_fresh_validation_from_frozen_train"

    def evaluate(self, candidate: Mapping[str, Any]) -> Stage6CandidateEvaluationResult:
        total_started = time.perf_counter()
        expression, identity = self.fresh_evaluator._expression(candidate)
        structural_hash = str(identity["structural_hash"])
        frozen = self.train_records.get(structural_hash)
        if frozen is None:
            raise Stage6TwoPhaseIntegrityError("Validation candidate has no frozen Train result")
        train = _plain(frozen["train"])
        direction = frozen.get("train_direction")
        if direction not in (-1, 1):
            raise Stage6TwoPhaseIntegrityError("Train-pass candidate has invalid frozen direction")
        prior = self.prior_validation_results.get(structural_hash)
        factor_seconds = validation_seconds = 0.0
        if (
            prior is not None
            and prior.get("train_direction") == direction
            and _plain(prior.get("expression", {})) == _plain(identity)
        ):
            validation = {**_plain(prior["validation"]), "evaluation_path": "verified_prior_stage6_validation_reuse"}
            evaluation_path = "verified_prior_stage6_validation_reuse"
            prior_fingerprint = prior["result_fingerprint"]
        else:
            factor_started = time.perf_counter()
            factor = np.asarray(self.fresh_evaluator._interpreter.evaluate(expression), dtype=np.float64)
            factor_seconds = time.perf_counter() - factor_started
            expected_shape = (self.context.dates.size, self.context.stocks.size)
            if factor.shape != expected_shape:
                raise RuntimeError(f"FactorInterpreter returned {factor.shape}; expected {expected_shape}")
            validation_started = time.perf_counter()
            prepared = self.fresh_evaluator._prepare_split(factor, "validation")
            self.fresh_evaluator._apply_direction(prepared, int(direction))
            validation = {**prepared.result, "metric_origin": VALIDATION_METRIC_ORIGIN_FRESH, "evaluation_path": "stage6_fresh_validation_from_frozen_train"}
            validation_seconds = time.perf_counter() - validation_started
            evaluation_path = "stage6_fresh_validation_from_frozen_train"
            prior_fingerprint = None
        coverage = {
            "train": _plain(train.get("factor_finite_coverage", {})),
            "validation": _plain(validation.get("factor_finite_coverage", {})),
        }
        return _build_result(
            evaluator=self, expression=identity,
            source_identity=_source_identity(
                self.fresh_evaluator, candidate, evaluation_path,
                frozen_train_result_fingerprint=frozen["result_fingerprint"],
                prior_validation_result_fingerprint=prior_fingerprint,
                old_stage6_train_reuse=False,
            ),
            status="completed", invalid_reasons=(), direction=int(direction),
            train=train, validation=validation, coverage=coverage,
            factor_seconds=factor_seconds, train_seconds=0.0,
            validation_seconds=validation_seconds,
            total_seconds=time.perf_counter() - total_started,
        )


def run_current_stage6_validation_evaluation(
    *, compatibility_manifest_path: str | Path, overlay_manifest_path: str | Path,
    train_entry_manifest_path: str | Path, train_pass_manifest_path: str | Path,
    output_root: str | Path,
    prior_full_evaluation_entry_manifest_path: str | Path | None = None,
    max_new_evaluations: int | None = None, retry_failed: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    data_paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    compatibility, _ = load_stage6_accepted_registry(compatibility_manifest_path)
    pass_manifest, pass_candidates = load_stage6_train_pass_manifest(train_pass_manifest_path)
    train_entry, train_records = _load_train_records(train_entry_manifest_path)
    if pass_manifest["train_entry_manifest_fingerprint"] != train_entry["entry_manifest_fingerprint"]:
        raise Stage6TwoPhaseIntegrityError("Train-pass manifest and Train entry differ")
    context = build_stage6_evaluation_context(paths=data_paths)
    if pass_manifest.get("accepted_registry_fingerprint") != compatibility.get(
        "accepted_registry_fingerprint"
    ):
        raise Stage6TwoPhaseIntegrityError(
            "Train-pass manifest and accepted registry differ"
        )
    if train_entry.get("context_fingerprint") != context.fingerprint:
        raise Stage6TwoPhaseIntegrityError(
            "Train entry and current Stage 6 context differ"
        )
    fresh = Stage6CandidateEvaluator(
        context,
        compatibility_audit_fingerprint=str(compatibility["audit_fingerprint"]),
        accepted_registry_fingerprint=str(compatibility["accepted_registry_fingerprint"]),
    )
    if train_entry.get("base_fresh_evaluation_contract_fingerprint") != (
        fresh.evaluation_contract_fingerprint
    ):
        raise Stage6TwoPhaseIntegrityError(
            "Train entry and current Stage 6 evaluation contract differ"
        )
    overlay = Stage6TrainReuseOverlay.load(overlay_manifest_path)
    current_mixed = Stage6MixedCandidateEvaluator(fresh, overlay)
    prior, prior_fingerprint = _load_prior_validation_results(
        prior_full_evaluation_entry_manifest_path,
        current_context_fingerprint=context.fingerprint,
        allowed_contract_fingerprints={fresh.evaluation_contract_fingerprint, current_mixed.evaluation_contract_fingerprint},
    )
    evaluator = Stage6ValidationFromFrozenTrainEvaluator(
        fresh, train_records=train_records, train_pass_manifest=pass_manifest,
        prior_validation_results=prior,
        prior_validation_seed_set_fingerprint=prior_fingerprint,
    )
    database_path = root / "evaluation_store" / "stage6_validation.sqlite"
    run_artifact_root = root / "evaluation_runs"
    entry_path = root / "validation_evaluation_entry_manifest.json"
    with EvaluationStore(database_path, run_artifact_root) as store:
        frozen = store.create_run(pass_candidates, evaluator, scope=STAGE6_VALIDATION_SCOPE)
        validation_extra = {
            "compatibility_audit_fingerprint": compatibility["audit_fingerprint"],
            "accepted_registry_fingerprint": compatibility["accepted_registry_fingerprint"],
            "train_entry_manifest_path": str(Path(train_entry_manifest_path).resolve()),
            "train_entry_manifest_fingerprint": train_entry["entry_manifest_fingerprint"],
            "train_pass_manifest_path": str(Path(train_pass_manifest_path).resolve()),
            "train_pass_manifest_fingerprint": pass_manifest["train_pass_manifest_fingerprint"],
            "base_fresh_evaluation_contract_fingerprint": (
                fresh.evaluation_contract_fingerprint
            ),
            "prior_validation_seed_count": len(prior),
            "prior_validation_seed_set_fingerprint": prior_fingerprint,
        }
        if "provisional_evaluation_universe" in pass_manifest:
            validation_extra["provisional_evaluation_universe"] = _plain(
                pass_manifest["provisional_evaluation_universe"]
            )
        entry = _entry_payload(
            schema=STAGE6_VALIDATION_ENTRY_SCHEMA, scope=STAGE6_VALIDATION_SCOPE,
            run=frozen, database_path=database_path, run_artifact_root=run_artifact_root,
            extra=validation_extra,
        )
        if entry_path.is_file() and _read_json(entry_path) != entry:
            raise Stage6TwoPhaseIntegrityError("existing Validation entry changed")
        _write_json_atomic(entry_path, entry)
        summary = Stage6EvaluationRunner(store, evaluator).run(
            frozen.run_id, max_new_evaluations=max_new_evaluations,
            retry_failed=retry_failed, progress_callback=progress_callback,
        )
    return {"entry_manifest_path": str(entry_path), "evaluation_run_id": frozen.run_id, "invocation": asdict(summary), "candidate_count": len(pass_candidates), "prior_validation_seed_count": len(prior)}


def load_stage6_two_phase_ic_distribution(
    *, validation_entry_manifest_path: str | Path,
    include_validation: bool = False,
) -> dict[str, Any]:
    entry = _load_entry(validation_entry_manifest_path, STAGE6_VALIDATION_ENTRY_SCHEMA, STAGE6_VALIDATION_SCOPE)
    with EvaluationStore(entry["database_path"], entry["run_artifact_root"]) as store:
        verified = store.load_verified_run_results(entry["evaluation_run_id"])

    def collect(split: str) -> dict[str, Any]:
        values = [float(row["result"][split]["ic"]["mean"]) for row in verified.records if row["result"][split]["ic"].get("mean") is not None and math.isfinite(float(row["result"][split]["ic"]["mean"]))]
        if not values:
            raise Stage6TwoPhaseIntegrityError(f"no finite {split} IC values")
        array = np.asarray(values, dtype=np.float64)
        return {"values": values, "statistics": {"count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)), "std_ddof0": float(array.std(ddof=0)), "min": float(array.min()), "max": float(array.max())}}
    result = {"evaluation_run_id": verified.run_id, "ordered_result_set_fingerprint": verified.ordered_result_set_fingerprint, "candidate_scope": "frozen_train_pass_with_complete_validation", "scope_candidate_count": len(verified.records), "train_ic": collect("train"), "oos": "not_loaded_not_evaluated"}
    if include_validation:
        result["validation_ic"] = collect("validation")
    return result


def run_current_stage6_two_phase_provisional_selection(
    *, validation_entry_manifest_path: str | Path,
    compatibility_manifest_path: str | Path, overlay_manifest_path: str | Path,
    output_root: str | Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    data_paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> Path:
    entry = _load_entry(validation_entry_manifest_path, STAGE6_VALIDATION_ENTRY_SCHEMA, STAGE6_VALIDATION_SCOPE)
    compatibility, _ = load_stage6_accepted_registry(compatibility_manifest_path)
    pass_manifest, pass_candidates = load_stage6_train_pass_manifest(entry["train_pass_manifest_path"])
    if pass_manifest["train_pass_manifest_fingerprint"] != entry["train_pass_manifest_fingerprint"]:
        raise Stage6TwoPhaseIntegrityError("Validation entry and Train-pass manifest differ")
    context = build_stage6_evaluation_context(paths=data_paths)
    if entry.get("context_fingerprint") != context.fingerprint:
        raise Stage6TwoPhaseIntegrityError(
            "Validation entry and current Stage 6 context differ"
        )
    if entry.get("accepted_registry_fingerprint") != compatibility.get(
        "accepted_registry_fingerprint"
    ):
        raise Stage6TwoPhaseIntegrityError(
            "Validation entry and accepted registry differ"
        )
    fresh = Stage6CandidateEvaluator(context, compatibility_audit_fingerprint=str(compatibility["audit_fingerprint"]), accepted_registry_fingerprint=str(compatibility["accepted_registry_fingerprint"]))
    if entry.get("base_fresh_evaluation_contract_fingerprint") != (
        fresh.evaluation_contract_fingerprint
    ):
        raise Stage6TwoPhaseIntegrityError(
            "Validation entry and current Stage 6 evaluation contract differ"
        )
    Stage6TrainReuseOverlay.load(overlay_manifest_path)  # fail closed on the approved overlay artifact
    enricher = Stage6TrainLongExcessEnricher(fresh)
    with EvaluationStore(entry["database_path"], entry["run_artifact_root"]) as store:
        return run_stage6_survivor_enrichment_selection(
            store=store, evaluation_run_id=entry["evaluation_run_id"],
            accepted_candidates=pass_candidates, enricher=enricher,
            output_root=output_root,
            provisional_universe=entry.get("provisional_evaluation_universe"),
            progress_callback=progress_callback,
        )


__all__ = [
    "DEFERRED_TRAIN_RECOMPUTE",
    "HISTORICAL_TRAIN_CONTRACT_NOT_EQUIVALENT",
    "NO_TRUSTED_TRAIN_RESULT_RESOURCE_LIMITED",
    "STAGE6_PROVISIONAL_UNIVERSE_SCHEMA",
    "STAGE6_RESOURCE_LIMITED_TRAIN_SCOPE",
    "STAGE6_TRAIN_PREPARATION_SCOPE", "STAGE6_VALIDATION_SCOPE",
    "STAGE6_TRAIN_ENTRY_SCHEMA", "STAGE6_TRAIN_PASS_MANIFEST_SCHEMA",
    "STAGE6_VALIDATION_ENTRY_SCHEMA", "STAGE6_TWO_PHASE_VERSION",
    "Stage6TwoPhaseIntegrityError", "Stage6TrainPreparationEvaluator",
    "Stage6ValidationFromFrozenTrainEvaluator",
    "freeze_current_stage6_resource_limited_universe",
    "load_stage6_provisional_evaluation_universe",
    "load_stage6_train_pass_manifest",
    "load_stage6_two_phase_ic_distribution", "run_current_stage6_train_preparation",
    "run_current_stage6_validation_evaluation",
    "run_current_stage6_two_phase_provisional_selection",
]
