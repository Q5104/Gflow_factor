"""Train-necessary prefilter before expensive Stage 6 Validation evaluation.

The prefilter is mathematically lossless for the frozen six-condition screen:
any candidate failing one of the three Train-only necessary conditions cannot
pass the final joint screen regardless of Validation.  OOS is never loaded.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from factor_gfn.evaluator import infer_long_direction

from .stage6_evaluation import (
    STAGE6_EVALUATION_RESULT_SCHEMA,
    Stage6CandidateEvaluationResult,
    Stage6CandidateEvaluator,
    _stable_hash,
)
from .stage6_mixed_evaluation import (
    TRAIN_METRIC_ORIGIN_FRESH,
    TRAIN_METRIC_ORIGIN_REUSE,
    VALIDATION_METRIC_ORIGIN_FRESH,
    Stage6MixedCandidateEvaluator,
    Stage6TrainReuseOverlay,
)
from .stage6_selection import HARD_CONDITION_CODES, Stage6SelectionConfig


STAGE6_PREFILTER_CONTRACT_SCHEMA = "factor_gfn.stage6_train_prefilter_contract.v1"
STAGE6_PREFILTER_VERSION = "factor_gfn.stage6_train_prefilter_evaluator.v1"
TRAIN_PREFILTER_SCHEMA = "factor_gfn.stage6_train_prefilter.v1"
TRAIN_PREFILTER_FAILED = "train_prefilter_failed"
TRAIN_PREFILTER_PASSED = "train_prefilter_passed"
PRIOR_FULL_RESULT_REUSE = "stage6_prior_full_result_reuse"

_TRAIN_CODES = (
    HARD_CONDITION_CODES[0],
    HARD_CONDITION_CODES[3],
    HARD_CONDITION_CODES[5],
)


class Stage6TrainPrefilterIntegrityError(RuntimeError):
    """The prefilter inputs or prior full-result seeds failed validation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def train_prefilter_decision(
    train: Mapping[str, Any],
    config: Stage6SelectionConfig = Stage6SelectionConfig(),
) -> dict[str, Any]:
    """Evaluate exactly the three frozen Train-only necessary conditions."""

    train_ic = _finite(train.get("ic", {}).get("mean"))
    train_long_ir = _finite(train.get("long", {}).get("annualized_ir"))
    train_barra = _finite(train.get("barra", {}).get("max_abs_correlation"))
    checks = {
        _TRAIN_CODES[0]: train_ic is not None
        and abs(train_ic) > config.train_abs_ic_min,
        _TRAIN_CODES[1]: train_long_ir is not None
        and train_long_ir > config.train_long_ir_min,
        _TRAIN_CODES[2]: train_barra is not None
        and train_barra < config.train_barra_ts_corr_max,
    }
    failed = [code for code in _TRAIN_CODES if not checks[code]]
    return {
        "schema": TRAIN_PREFILTER_SCHEMA,
        "status": TRAIN_PREFILTER_FAILED if failed else TRAIN_PREFILTER_PASSED,
        "condition_results": checks,
        "failed_conditions": failed,
        "validation_required": not failed,
        "metrics": {
            "train_ic": train_ic,
            "train_long_ir": train_long_ir,
            "train_barra_ts_corr": train_barra,
        },
    }


class Stage6TrainPrefilterEvaluator:
    """Reuse trusted Train, stop proven failures, and fresh-evaluate survivors."""

    def __init__(
        self,
        fresh_evaluator: Stage6CandidateEvaluator,
        overlay: Stage6TrainReuseOverlay,
        *,
        prior_full_results: Mapping[str, Mapping[str, Any]] | None = None,
        selection_config: Stage6SelectionConfig = Stage6SelectionConfig(),
    ) -> None:
        self.fresh_evaluator = fresh_evaluator
        self.mixed_evaluator = Stage6MixedCandidateEvaluator(fresh_evaluator, overlay)
        self.overlay = overlay
        self.context = fresh_evaluator.context
        self.compatibility_audit_fingerprint = (
            fresh_evaluator.compatibility_audit_fingerprint
        )
        self.accepted_registry_fingerprint = fresh_evaluator.accepted_registry_fingerprint
        self.selection_config = selection_config
        self.prior_full_results = {
            str(key): _plain(value) for key, value in (prior_full_results or {}).items()
        }
        prior_identity: list[dict[str, Any]] = []
        for structural_hash, result in sorted(self.prior_full_results.items()):
            if result.get("status") not in {"completed", "completed_invalid"}:
                raise Stage6TrainPrefilterIntegrityError(
                    "prior seed is not a complete full-evaluation result"
                )
            if result.get("expression", {}).get("structural_hash") != structural_hash:
                raise Stage6TrainPrefilterIntegrityError(
                    "prior seed expression identity mismatch"
                )
            if result.get("context_fingerprint") != self.context.fingerprint:
                raise Stage6TrainPrefilterIntegrityError(
                    "prior seed context differs from current Stage 6 context"
                )
            if not isinstance(result.get("validation"), Mapping):
                raise Stage6TrainPrefilterIntegrityError(
                    "prior seed lacks its full Validation result"
                )
            prior_identity.append(
                {
                    "structural_hash": structural_hash,
                    "result_fingerprint": result.get("result_fingerprint"),
                    "evaluation_contract_fingerprint": result.get(
                        "evaluation_contract_fingerprint"
                    ),
                }
            )
        implementation_sha256 = _sha256_file(Path(__file__).resolve())
        self.prior_seed_set_fingerprint = _stable_hash(prior_identity)
        self.evaluation_contract = MappingProxyType(
            {
                "schema": STAGE6_PREFILTER_CONTRACT_SCHEMA,
                "version": STAGE6_PREFILTER_VERSION,
                "implementation_sha256": implementation_sha256,
                "context_fingerprint": self.context.fingerprint,
                "base_fresh_evaluation_contract_fingerprint": (
                    fresh_evaluator.evaluation_contract_fingerprint
                ),
                "train_reuse_overlay_fingerprint": overlay.fingerprint,
                "prior_full_result_seed_set_fingerprint": (
                    self.prior_seed_set_fingerprint
                ),
                "prior_full_result_seed_count": len(self.prior_full_results),
                "train_necessary_conditions": {
                    "abs_train_ic_gt": selection_config.train_abs_ic_min,
                    "train_long_ir_gt": selection_config.train_long_ir_min,
                    "train_barra_ts_corr_lt": (
                        selection_config.train_barra_ts_corr_max
                    ),
                    "strict_inequalities": True,
                },
                "execution_order": [
                    "reuse_prior_full_result_if_available",
                    "reuse_verified_stage5_train_if_available",
                    "otherwise_fresh_train",
                    "apply_train_necessary_prefilter",
                    "fresh_validation_only_if_train_prefilter_passed",
                ],
                "final_six_condition_contract_fingerprint": (
                    selection_config.fingerprint
                ),
                "oos": "not_loaded_and_interface_rejected",
            }
        )
        self.evaluation_contract_fingerprint = _stable_hash(
            dict(self.evaluation_contract)
        )

    def resolve_candidate_identity(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.fresh_evaluator.resolve_candidate_identity(candidate)

    def has_reusable_train(self, candidate: Mapping[str, Any]) -> bool:
        structural_hash = str(candidate["current_structural_hash"])
        return (
            structural_hash in self.prior_full_results
            or self.overlay.get(structural_hash) is not None
        )

    def planned_path(self, candidate: Mapping[str, Any]) -> str:
        structural_hash = str(candidate["current_structural_hash"])
        if structural_hash in self.prior_full_results:
            return PRIOR_FULL_RESULT_REUSE
        if self.overlay.get(structural_hash) is not None:
            return "verified_stage5_train_prefilter"
        return "stage6_fresh_train_prefilter"

    def prefilter_plan(
        self, candidates: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]
    ) -> dict[str, int]:
        """Return exact known strata without reading Validation information."""

        counts = {
            "candidate_count": len(candidates),
            "prior_full_result_reuse": 0,
            "verified_stage5_train_prefilter_failed": 0,
            "verified_stage5_train_prefilter_passed": 0,
            "stage6_fresh_train_required": 0,
        }
        for candidate in candidates:
            structural_hash = str(candidate["current_structural_hash"])
            if structural_hash in self.prior_full_results:
                counts["prior_full_result_reuse"] += 1
                continue
            record = self.overlay.get(structural_hash)
            if record is None:
                counts["stage6_fresh_train_required"] += 1
                continue
            train = self.mixed_evaluator._reused_train_payload(record)
            decision = train_prefilter_decision(train, self.selection_config)
            key = (
                "verified_stage5_train_prefilter_failed"
                if decision["status"] == TRAIN_PREFILTER_FAILED
                else "verified_stage5_train_prefilter_passed"
            )
            counts[key] += 1
        counts["known_fresh_validation_required"] = counts[
            "verified_stage5_train_prefilter_passed"
        ]
        counts["fresh_validation_upper_bound"] = (
            counts["verified_stage5_train_prefilter_passed"]
            + counts["stage6_fresh_train_required"]
        )
        return counts

    def _source_identity(
        self,
        candidate: Mapping[str, Any],
        *,
        evaluation_path: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "compatibility_audit_fingerprint": self.compatibility_audit_fingerprint,
            "accepted_registry_fingerprint": self.accepted_registry_fingerprint,
            "compatibility_record_fingerprint": candidate.get(
                "compatibility_record_fingerprint"
            ),
            "source_claimed_structural_hash": candidate.get(
                "source_claimed_structural_hash"
            ),
            "origin_ids": list(candidate.get("origin_ids", [])),
            "source_ids": list(candidate.get("source_ids", [])),
            "evaluation_path": evaluation_path,
            **dict(extra or {}),
        }

    def _validation_not_evaluated(self) -> dict[str, Any]:
        split = self.context.get_split_data("validation")
        total = int(split.rebalance_dates.size)
        return {
            "availability": "not_evaluated_train_prefilter_failed",
            "metric_origin": "not_evaluated_train_prefilter_failed",
            "requested_date_range": [
                split.boundary.requested_start,
                split.boundary.requested_end,
            ],
            "actual_date_range": [split.boundary.actual_start, split.boundary.actual_end],
            "rebalance_dates": None,
            "rebalance_periods": total,
            "ic": {
                "mean": None,
                "std": None,
                "icir": None,
                "valid_periods": 0,
                "total_periods": total,
            },
            "long": {
                "direction": None,
                "mean_period_return": None,
                "annualized_return": None,
                "annualized_ir": None,
                "std": None,
                "valid_periods": 0,
                "total_periods": total,
                "excess_series": {
                    "dates": None,
                    "values": None,
                    "availability": "not_evaluated_train_prefilter_failed",
                },
            },
            "barra": {
                "max_abs_correlation": None,
                "correlations": {},
                "common_valid_periods": {},
                "availability": "not_evaluated_train_prefilter_failed",
            },
            "raw_long_short_valid_periods": None,
            "factor_finite_coverage": {
                "finite_universe_values": None,
                "eligible_universe_values": None,
                "rate": None,
                "availability": "not_evaluated_train_prefilter_failed",
            },
            "neutralization": {
                "availability": "not_evaluated_train_prefilter_failed"
            },
        }

    def _build_result(
        self,
        *,
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
            "expression": dict(expression),
            "context_fingerprint": self.context.fingerprint,
            "evaluation_contract_fingerprint": self.evaluation_contract_fingerprint,
            "train_direction": direction,
            "train": _plain(train),
            "validation": _plain(validation),
            "factor_finite_coverage": _plain(coverage),
        }
        return Stage6CandidateEvaluationResult(
            schema=STAGE6_EVALUATION_RESULT_SCHEMA,
            status=status,
            invalid_reasons=invalid_reasons,
            expression=MappingProxyType(dict(expression)),
            source_identity=MappingProxyType(dict(source_identity)),
            context_fingerprint=self.context.fingerprint,
            evaluation_contract_fingerprint=self.evaluation_contract_fingerprint,
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

    def _prefilter_failed_result(
        self,
        *,
        candidate: Mapping[str, Any],
        expression: Mapping[str, Any],
        train: Mapping[str, Any],
        direction: int | None,
        decision: Mapping[str, Any],
        evaluation_path: str,
        factor_seconds: float,
        train_seconds: float,
        total_seconds: float,
    ) -> Stage6CandidateEvaluationResult:
        marked_train = {**_plain(train), "train_prefilter": _plain(decision)}
        validation = self._validation_not_evaluated()
        coverage = {
            "train": _plain(marked_train.get("factor_finite_coverage", {})),
            "validation": _plain(validation["factor_finite_coverage"]),
        }
        invalid_reasons = (
            TRAIN_PREFILTER_FAILED,
            *tuple(str(code) for code in decision["failed_conditions"]),
        )
        return self._build_result(
            expression=expression,
            source_identity=self._source_identity(
                candidate,
                evaluation_path=evaluation_path,
                extra={"validation_metric_origin": "not_evaluated_train_prefilter_failed"},
            ),
            status="completed_invalid",
            invalid_reasons=invalid_reasons,
            direction=direction,
            train=marked_train,
            validation=validation,
            coverage=coverage,
            factor_seconds=factor_seconds,
            train_seconds=train_seconds,
            validation_seconds=0.0,
            total_seconds=total_seconds,
        )

    def _repackage_full_result(
        self,
        candidate: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        evaluation_path: str,
        prior_result_fingerprint: str | None = None,
    ) -> Stage6CandidateEvaluationResult:
        train = {
            **_plain(result["train"]),
            "train_prefilter": {
                **train_prefilter_decision(result["train"], self.selection_config),
                "full_validation_already_available": True,
            },
        }
        validation = {
            **_plain(result["validation"]),
            "evaluation_path": evaluation_path,
        }
        return self._build_result(
            expression=result["expression"],
            source_identity=self._source_identity(
                candidate,
                evaluation_path=evaluation_path,
                extra={
                    "prior_full_result_fingerprint": prior_result_fingerprint,
                    "validation_metric_origin": result["validation"].get(
                        "metric_origin", VALIDATION_METRIC_ORIGIN_FRESH
                    ),
                },
            ),
            status=str(result["status"]),
            invalid_reasons=tuple(str(value) for value in result["invalid_reasons"]),
            direction=result.get("train_direction"),
            train=train,
            validation=validation,
            coverage=result["factor_finite_coverage"],
            factor_seconds=0.0 if prior_result_fingerprint else float(
                result.get("factor_seconds", 0.0)
            ),
            train_seconds=0.0 if prior_result_fingerprint else float(
                result.get("train_evaluation_seconds", 0.0)
            ),
            validation_seconds=0.0 if prior_result_fingerprint else float(
                result.get("validation_evaluation_seconds", 0.0)
            ),
            total_seconds=0.0 if prior_result_fingerprint else float(
                result.get("total_seconds", 0.0)
            ),
        )

    def evaluate(self, candidate: Mapping[str, Any]) -> Stage6CandidateEvaluationResult:
        total_started = time.perf_counter()
        expression, identity = self.fresh_evaluator._expression(candidate)
        structural_hash = str(identity["structural_hash"])
        prior = self.prior_full_results.get(structural_hash)
        if prior is not None:
            return self._repackage_full_result(
                candidate,
                prior,
                evaluation_path=PRIOR_FULL_RESULT_REUSE,
                prior_result_fingerprint=str(prior["result_fingerprint"]),
            )

        overlay_record = self.overlay.get(structural_hash)
        if overlay_record is not None:
            train_started = time.perf_counter()
            train = self.mixed_evaluator._reused_train_payload(overlay_record)
            train_seconds = time.perf_counter() - train_started
            direction = int(train["long"]["direction"])
            decision = train_prefilter_decision(train, self.selection_config)
            if decision["status"] == TRAIN_PREFILTER_FAILED:
                return self._prefilter_failed_result(
                    candidate=candidate,
                    expression=identity,
                    train=train,
                    direction=direction,
                    decision=decision,
                    evaluation_path="verified_stage5_train_prefilter_failed",
                    factor_seconds=0.0,
                    train_seconds=train_seconds,
                    total_seconds=time.perf_counter() - total_started,
                )
            full = self.mixed_evaluator.evaluate(candidate)
            return self._repackage_full_result(
                candidate,
                full.to_dict(),
                evaluation_path="verified_stage5_train_then_fresh_validation",
            )

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
        prepared_train = self.fresh_evaluator._prepare_split(factor, "train")
        train_ic = prepared_train.result["ic"]["mean"]
        direction = (
            infer_long_direction(float(train_ic))
            if train_ic is not None and train_ic != 0.0
            else None
        )
        self.fresh_evaluator._apply_direction(prepared_train, direction)
        train = {
            **prepared_train.result,
            "metric_origin": TRAIN_METRIC_ORIGIN_FRESH,
        }
        train_seconds = time.perf_counter() - train_started
        decision = train_prefilter_decision(train, self.selection_config)
        if decision["status"] == TRAIN_PREFILTER_FAILED:
            return self._prefilter_failed_result(
                candidate=candidate,
                expression=identity,
                train=train,
                direction=direction,
                decision=decision,
                evaluation_path="stage6_fresh_train_prefilter_failed",
                factor_seconds=factor_seconds,
                train_seconds=train_seconds,
                total_seconds=time.perf_counter() - total_started,
            )
        validation_started = time.perf_counter()
        prepared_validation = self.fresh_evaluator._prepare_split(factor, "validation")
        self.fresh_evaluator._apply_direction(prepared_validation, direction)
        validation = {
            **prepared_validation.result,
            "metric_origin": VALIDATION_METRIC_ORIGIN_FRESH,
            "evaluation_path": "stage6_fresh_train_then_fresh_validation",
        }
        validation_seconds = time.perf_counter() - validation_started
        marked_train = {**train, "train_prefilter": decision}
        coverage = {
            "train": dict(marked_train["factor_finite_coverage"]),
            "validation": dict(validation["factor_finite_coverage"]),
        }
        return self._build_result(
            expression=identity,
            source_identity=self._source_identity(
                candidate,
                evaluation_path="stage6_fresh_train_then_fresh_validation",
                extra={
                    "train_metric_origin": TRAIN_METRIC_ORIGIN_FRESH,
                    "validation_metric_origin": VALIDATION_METRIC_ORIGIN_FRESH,
                },
            ),
            status="completed",
            invalid_reasons=(),
            direction=direction,
            train=marked_train,
            validation=validation,
            coverage=coverage,
            factor_seconds=factor_seconds,
            train_seconds=train_seconds,
            validation_seconds=validation_seconds,
            total_seconds=time.perf_counter() - total_started,
        )


__all__ = [
    "PRIOR_FULL_RESULT_REUSE",
    "STAGE6_PREFILTER_CONTRACT_SCHEMA",
    "STAGE6_PREFILTER_VERSION",
    "TRAIN_PREFILTER_FAILED",
    "TRAIN_PREFILTER_PASSED",
    "TRAIN_PREFILTER_SCHEMA",
    "Stage6TrainPrefilterEvaluator",
    "Stage6TrainPrefilterIntegrityError",
    "train_prefilter_decision",
]
