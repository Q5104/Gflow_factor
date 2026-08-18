"""Stage 6 step 9B: verified Train reuse plus fresh Validation.

The mixed evaluator is deliberately separate from the full-fresh evaluator.
It consumes an immutable 9A overlay, reuses only its verified Train summary,
and always interprets the expression through Validation so time-series warmup
is unchanged. Candidates absent from the overlay fall back to the existing
full-fresh evaluator under an explicit mixed-contract result identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from factor_gfn.barra import STYLE_NAMES

from .stage6_evaluation import (
    STAGE6_EVALUATION_RESULT_SCHEMA,
    Stage6CandidateEvaluationResult,
    Stage6CandidateEvaluator,
    Stage6EvaluationConfig,
    build_stage6_evaluation_context,
    _finite_or_none,
    _sha256_file,
    _stable_hash,
)
from .stage6_evaluation_store import EvaluationStore, Stage6EvaluationRunner
from .stage6_train_reuse import (
    HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA,
    HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA,
    TRAIN_REUSE_MANIFEST_SCHEMA,
    TRAIN_REUSE_OVERLAY_SCHEMA,
)


STAGE6_MIXED_EVALUATION_CONTRACT_SCHEMA = (
    "factor_gfn.stage6_mixed_evaluation_contract.v1"
)
STAGE6_MIXED_EVALUATOR_VERSION = "stage6-verified-train-reuse-fresh-validation-v1"

TRAIN_METRIC_ORIGIN_REUSE = "stage5_verified_reuse"
TRAIN_METRIC_ORIGIN_FRESH = "stage6_fresh_evaluation"
VALIDATION_METRIC_ORIGIN_FRESH = "stage6_fresh_evaluation"

_MANIFEST_FINGERPRINT_KEYS_V1 = (
    "schema",
    "auditor_version",
    "source_set_fingerprint",
    "candidate_registry_fingerprint",
    "compatibility_audit_fingerprint",
    "accepted_registry_fingerprint",
    "stage6_context_fingerprint",
    "stage6_evaluation_contract_fingerprint",
    "target_provider_fingerprint",
    "target_train_contract_projection_fingerprint",
    "source_audit_digest",
    "numeric_verification_digest",
    "overlay_digest",
)
_MANIFEST_FINGERPRINT_KEYS_V2 = (
    "schema",
    "adapter_version",
    "verification_mode",
    "fresh_train_fallback_reason",
    "source_set_fingerprint",
    "source_snapshot_fingerprint",
    "candidate_registry_fingerprint",
    "compatibility_audit_fingerprint",
    "accepted_registry_fingerprint",
    "stage6_context_fingerprint",
    "stage6_evaluation_contract_fingerprint",
    "target_provider_fingerprint",
    "artifact_train_contract_fingerprint",
    "current_train_contract_fingerprint",
    "train_contract_verification_digest",
    "overlay_digest",
)


class TrainReuseOverlayIntegrityError(RuntimeError):
    """The immutable 9A overlay cannot be trusted by the mixed evaluator."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainReuseOverlayIntegrityError(f"cannot read overlay JSON: {path}") from error
    if not isinstance(value, dict):
        raise TrainReuseOverlayIntegrityError(f"overlay JSON must be an object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.strip():
                    raise TrainReuseOverlayIntegrityError(
                        f"blank overlay row: {path}:{line_number}"
                    )
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise TrainReuseOverlayIntegrityError(
                        f"overlay row is not an object: {path}:{line_number}"
                    )
                yield value
    except TrainReuseOverlayIntegrityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainReuseOverlayIntegrityError(f"cannot read overlay JSONL: {path}") from error


@dataclass(frozen=True, slots=True)
class Stage6TrainReuseOverlay:
    manifest_path: Path
    fingerprint: str
    verification_fingerprint: str
    manifest: Mapping[str, Any]
    records: Mapping[str, Mapping[str, Any]]

    @classmethod
    def load(cls, manifest_path: str | Path) -> "Stage6TrainReuseOverlay":
        path = Path(manifest_path).resolve()
        manifest = _read_json(path)
        schema = manifest.get("schema")
        if schema not in {
            TRAIN_REUSE_MANIFEST_SCHEMA,
            HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA,
        }:
            raise TrainReuseOverlayIntegrityError("Train reuse manifest schema mismatch")
        is_hybrid_v2 = schema == HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA
        try:
            fingerprint_keys = (
                _MANIFEST_FINGERPRINT_KEYS_V2
                if is_hybrid_v2
                else _MANIFEST_FINGERPRINT_KEYS_V1
            )
            payload = {key: manifest[key] for key in fingerprint_keys}
            fingerprint = str(manifest["train_reuse_overlay_fingerprint"])
            overlay_meta = manifest["artifacts"]["train_reuse_overlay.jsonl"]
        except (KeyError, TypeError) as error:
            raise TrainReuseOverlayIntegrityError("Train reuse manifest is incomplete") from error
        if _stable_hash(payload) != fingerprint or path.parent.name != fingerprint:
            raise TrainReuseOverlayIntegrityError("Train reuse manifest fingerprint mismatch")
        overlay_path = path.parent / "train_reuse_overlay.jsonl"
        if (
            not overlay_path.is_file()
            or overlay_path.stat().st_size != int(overlay_meta["size_bytes"])
            or _sha256_file(overlay_path) != str(overlay_meta["sha256"])
        ):
            raise TrainReuseOverlayIntegrityError("Train reuse overlay artifact mismatch")

        rows = list(_iter_jsonl(overlay_path))
        if _stable_hash(rows) != manifest.get("overlay_digest"):
            raise TrainReuseOverlayIntegrityError("Train reuse overlay logical digest mismatch")
        if is_hybrid_v2:
            verification_meta = manifest["artifacts"].get(
                "train_reuse_contract_verification.json"
            )
            verification_path = (
                path.parent / "train_reuse_contract_verification.json"
            )
            if (
                not isinstance(verification_meta, Mapping)
                or not verification_path.is_file()
                or verification_path.stat().st_size
                != int(verification_meta.get("size_bytes", -1))
                or _sha256_file(verification_path)
                != str(verification_meta.get("sha256"))
            ):
                raise TrainReuseOverlayIntegrityError(
                    "Hybrid Train contract verification artifact mismatch"
                )
            verification = _read_json(verification_path)
            verification_mode = verification.get("verification_mode")
            exact_contract = verification_mode == "hybrid_exact_contract"
            full_fresh_fallback = (
                verification_mode == "hybrid_full_fresh_train_fallback"
            )
            if (
                _stable_hash(verification)
                != manifest.get("train_contract_verification_digest")
                or verification_mode != manifest.get("verification_mode")
                or not (exact_contract or full_fresh_fallback)
            ):
                raise TrainReuseOverlayIntegrityError(
                    "Hybrid Train contract verification is invalid"
                )
            if exact_contract and (
                verification.get("numeric_verification")
                != "not_required_by_approved_hybrid_contract"
                or verification.get("result") != "TRAIN_METRICS_REUSABLE"
                or verification.get("old_train_metrics_allowed") is not True
                or manifest.get("fresh_train_fallback_reason") is not None
                or manifest.get("artifact_train_contract_fingerprint")
                != manifest.get("current_train_contract_fingerprint")
            ):
                raise TrainReuseOverlayIntegrityError(
                    "Hybrid exact-contract verification is invalid"
                )
            if full_fresh_fallback and (
                verification.get("numeric_verification")
                != "forbidden_because_train_contract_mismatch"
                or verification.get("result") != "FULL_FRESH_TRAIN_FALLBACK"
                or verification.get("old_train_metrics_allowed") is not False
                or verification.get("fresh_train_fallback_reason")
                != "current_train_contract_mismatch"
                or manifest.get("fresh_train_fallback_reason")
                != "current_train_contract_mismatch"
                or manifest.get("artifact_train_contract_fingerprint")
                == manifest.get("current_train_contract_fingerprint")
                or not isinstance(
                    verification.get("contract_difference_paths"), list
                )
                or not verification.get("contract_difference_paths")
                or rows
            ):
                raise TrainReuseOverlayIntegrityError(
                    "Hybrid full-fresh fallback verification is invalid"
                )
        records: dict[str, Mapping[str, Any]] = {}
        expected_row_schema = (
            HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA
            if is_hybrid_v2
            else TRAIN_REUSE_OVERLAY_SCHEMA
        )
        for row in rows:
            if row.get("schema") != expected_row_schema:
                raise TrainReuseOverlayIntegrityError("Train reuse overlay record schema mismatch")
            stable = {key: value for key, value in row.items() if key != "record_fingerprint"}
            if _stable_hash(stable) != row.get("record_fingerprint"):
                raise TrainReuseOverlayIntegrityError("Train reuse overlay record fingerprint mismatch")
            structural_hash = str(row.get("structural_hash"))
            if structural_hash in records:
                raise TrainReuseOverlayIntegrityError("duplicate structural hash in Train reuse overlay")
            if row.get("train_metric_origin") != TRAIN_METRIC_ORIGIN_REUSE:
                raise TrainReuseOverlayIntegrityError("overlay contains an unverified Train origin")
            if is_hybrid_v2 and (
                row.get("verification_mode") != "hybrid_exact_contract"
                or row.get("train_evaluation_contract_fingerprint")
                != manifest.get("artifact_train_contract_fingerprint")
            ):
                raise TrainReuseOverlayIntegrityError(
                    "Hybrid overlay record contract provenance mismatch"
                )
            if is_hybrid_v2:
                node_count = row.get("node_count")
                long_excess = row.get("train_long_excess")
                if (
                    isinstance(node_count, bool)
                    or not isinstance(node_count, int)
                    or node_count < 1
                ):
                    raise TrainReuseOverlayIntegrityError(
                        "Hybrid overlay record node_count is invalid"
                    )
                if long_excess is None and node_count > 2:
                    raise TrainReuseOverlayIntegrityError(
                        "N>2 Hybrid overlay record lacks Train long-excess"
                    )
                if long_excess is not None and not isinstance(long_excess, Mapping):
                    raise TrainReuseOverlayIntegrityError(
                        "Hybrid overlay Train long-excess is invalid"
                    )
            records[structural_hash] = MappingProxyType(dict(row))
        expected_count = int(manifest.get("counts", {}).get("overlay_candidates", -1))
        if len(records) != expected_count:
            raise TrainReuseOverlayIntegrityError("Train reuse overlay count mismatch")
        if is_hybrid_v2:
            counts = manifest.get("counts", {})
            accepted_count = int(counts.get("accepted_candidates", -1))
            fresh_count = int(counts.get("fresh_train_fallback_candidates", -1))
            if manifest.get("verification_mode") == "hybrid_exact_contract":
                counts_valid = (
                    accepted_count >= 0
                    and expected_count == accepted_count
                    and fresh_count == 0
                )
            else:
                counts_valid = (
                    accepted_count >= 0
                    and expected_count == 0
                    and fresh_count == accepted_count
                    and manifest.get("coverage_ratio") == 0.0
                )
            if not counts_valid:
                raise TrainReuseOverlayIntegrityError(
                    "Hybrid Train reuse decision counts are inconsistent"
                )
        return cls(
            manifest_path=path,
            fingerprint=fingerprint,
            verification_fingerprint=str(
                manifest[
                    "train_contract_verification_digest"
                    if is_hybrid_v2
                    else "numeric_verification_digest"
                ]
            ),
            manifest=MappingProxyType(manifest),
            records=MappingProxyType(records),
        )

    def get(self, structural_hash: str) -> Mapping[str, Any] | None:
        return self.records.get(structural_hash)

    @property
    def fresh_train_fallback_reason(self) -> str:
        reason = self.manifest.get("fresh_train_fallback_reason")
        return (
            str(reason)
            if isinstance(reason, str) and reason
            else "candidate_absent_from_verified_overlay"
        )


def _fresh_metric_origin() -> dict[str, Any]:
    return {
        "metric_origin": TRAIN_METRIC_ORIGIN_FRESH,
        "train_source_id": None,
        "train_source_snapshot_fingerprint": None,
        "train_reuse_verification_fingerprint": None,
        "train_evaluation_contract_fingerprint": None,
    }


class Stage6MixedCandidateEvaluator:
    """Evaluate one candidate with verified Train reuse when available."""

    def __init__(
        self,
        fresh_evaluator: Stage6CandidateEvaluator,
        overlay: Stage6TrainReuseOverlay,
    ) -> None:
        if fresh_evaluator.compatibility_audit_fingerprint is None:
            raise ValueError("mixed evaluation requires compatibility audit identity")
        if fresh_evaluator.accepted_registry_fingerprint is None:
            raise ValueError("mixed evaluation requires accepted registry identity")
        expected = {
            "stage6_context_fingerprint": fresh_evaluator.context.fingerprint,
            "stage6_evaluation_contract_fingerprint": (
                fresh_evaluator.evaluation_contract_fingerprint
            ),
            "compatibility_audit_fingerprint": (
                fresh_evaluator.compatibility_audit_fingerprint
            ),
            "accepted_registry_fingerprint": fresh_evaluator.accepted_registry_fingerprint,
        }
        mismatches = {
            key: (overlay.manifest.get(key), value)
            for key, value in expected.items()
            if overlay.manifest.get(key) != value
        }
        if mismatches:
            raise TrainReuseOverlayIntegrityError(
                f"overlay cannot seed this Stage 6 evaluator: {mismatches}"
            )
        self.fresh_evaluator = fresh_evaluator
        self.overlay = overlay
        self.context = fresh_evaluator.context
        self.compatibility_audit_fingerprint = (
            fresh_evaluator.compatibility_audit_fingerprint
        )
        self.accepted_registry_fingerprint = fresh_evaluator.accepted_registry_fingerprint
        self.evaluation_contract: Mapping[str, Any] = MappingProxyType(
            {
                "schema": STAGE6_MIXED_EVALUATION_CONTRACT_SCHEMA,
                "evaluator_version": STAGE6_MIXED_EVALUATOR_VERSION,
                "base_fresh_evaluation_contract_fingerprint": (
                    fresh_evaluator.evaluation_contract_fingerprint
                ),
                "stage6_context_fingerprint": self.context.fingerprint,
                "train_reuse_overlay_fingerprint": overlay.fingerprint,
                "train_reuse_verification_fingerprint": (
                    overlay.verification_fingerprint
                ),
                "train_policy": "verified_overlay_else_full_fresh",
                "validation_policy": "always_stage6_fresh",
                "interpretation": "one_full_history_pass_through_validation",
                "reused_train_long_excess": (
                    "when_persisted_by_hybrid_artifact"
                    if overlay.manifest.get("schema")
                    == HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA
                    else False
                ),
                "oos": "not_loaded_and_interface_rejected",
            }
        )
        self.evaluation_contract_fingerprint = _stable_hash(dict(self.evaluation_contract))

    def resolve_candidate_identity(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.fresh_evaluator.resolve_candidate_identity(candidate)

    def has_reusable_train(self, candidate: Mapping[str, Any]) -> bool:
        identity = self.resolve_candidate_identity(candidate)
        return self.overlay.get(str(identity["structural_hash"])) is not None

    def _result(
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
            "train": dict(train),
            "validation": dict(validation),
            "factor_finite_coverage": dict(coverage),
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
            train=MappingProxyType(dict(train)),
            validation=MappingProxyType(dict(validation)),
            factor_finite_coverage=MappingProxyType(dict(coverage)),
            factor_seconds=float(factor_seconds),
            train_evaluation_seconds=float(train_seconds),
            validation_evaluation_seconds=float(validation_seconds),
            total_seconds=float(total_seconds),
            result_fingerprint=_stable_hash(deterministic),
        )

    def _repackage_fresh(
        self,
        candidate: Mapping[str, Any],
        fresh: Stage6CandidateEvaluationResult,
        total_seconds: float,
    ) -> Stage6CandidateEvaluationResult:
        train = {**dict(fresh.train), **_fresh_metric_origin()}
        validation = {
            **dict(fresh.validation),
            "metric_origin": VALIDATION_METRIC_ORIGIN_FRESH,
        }
        source_identity = {
            **dict(fresh.source_identity),
            "train_metric_origin": TRAIN_METRIC_ORIGIN_FRESH,
            "validation_metric_origin": VALIDATION_METRIC_ORIGIN_FRESH,
            "train_reuse_fallback_reason": self.overlay.fresh_train_fallback_reason,
            "train_reuse_overlay_fingerprint": self.overlay.fingerprint,
        }
        return self._result(
            expression=fresh.expression,
            source_identity=source_identity,
            status=fresh.status,
            invalid_reasons=fresh.invalid_reasons,
            direction=fresh.train_direction,
            train=train,
            validation=validation,
            coverage=fresh.factor_finite_coverage,
            factor_seconds=fresh.factor_seconds,
            train_seconds=fresh.train_evaluation_seconds,
            validation_seconds=fresh.validation_evaluation_seconds,
            total_seconds=total_seconds,
        )

    def _reused_train_payload(self, record: Mapping[str, Any]) -> dict[str, Any]:
        metrics = record["train_metrics"]
        split = self.context.get_split_data("train")
        origins = [dict(value) for value in record["origins"]]
        source_ids = sorted({str(value["source_id"]) for value in origins})
        snapshot_fingerprints = sorted(
            {str(value["snapshot_fingerprint"]) for value in origins}
        )
        correlations = {
            name: _finite_or_none(metrics["train_barra_correlations"][name])
            for name in STYLE_NAMES
        }
        valid_periods = {
            name: int(metrics["train_barra_valid_periods_by_style"][name])
            for name in STYLE_NAMES
        }
        excess_series: dict[str, Any]
        if record.get("schema") == HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA:
            persisted = record.get("train_long_excess")
            if persisted is None:
                if int(record["node_count"]) > 2:
                    raise TrainReuseOverlayIntegrityError(
                        "N>2 Hybrid record lacks persisted Train long-excess"
                    )
                excess_series = {
                    "dates": None,
                    "values": None,
                    "availability": "missing_allowed_exact_n_1_2",
                    "origin": None,
                }
            else:
                if not isinstance(persisted, Mapping):
                    raise TrainReuseOverlayIntegrityError(
                        "Hybrid Train long-excess payload is invalid"
                    )
                dates = persisted.get("dates")
                values = persisted.get("values")
                if not isinstance(dates, list) or not isinstance(values, list):
                    raise TrainReuseOverlayIntegrityError(
                        "Hybrid Train long-excess dates/values are invalid"
                    )
                expected_dates = [str(value) for value in split.rebalance_dates]
                if dates != expected_dates or len(values) != len(dates):
                    raise TrainReuseOverlayIntegrityError(
                        "Hybrid Train long-excess calendar differs from Stage 6 Train"
                    )
                normalized_values: list[float | None] = []
                for value in values:
                    if value is None:
                        normalized_values.append(None)
                    elif (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                    ):
                        normalized_values.append(float(value))
                    else:
                        raise TrainReuseOverlayIntegrityError(
                            "Hybrid Train long-excess contains an invalid value"
                        )
                finite_periods = sum(
                    value is not None for value in normalized_values
                )
                if (
                    persisted.get("direction") != metrics["train_direction"]
                    or persisted.get("valid_periods")
                    != metrics["train_long_valid_periods"]
                    or persisted.get("finite_periods") != finite_periods
                    or finite_periods != metrics["train_long_valid_periods"]
                    or persisted.get("total_periods") != len(expected_dates)
                    or persisted.get("origin")
                    != "stage5_hybrid_train_artifact_reuse"
                ):
                    raise TrainReuseOverlayIntegrityError(
                        "Hybrid Train long-excess summary provenance is inconsistent"
                    )
                excess_series = {
                    "dates": list(dates),
                    "values": normalized_values,
                    "availability": "available_hybrid_artifact_reuse",
                    "origin": "stage5_hybrid_train_artifact_reuse",
                    "train_evaluation_contract_fingerprint": record[
                        "train_evaluation_contract_fingerprint"
                    ],
                    "overlay_record_fingerprint": record["record_fingerprint"],
                }
        else:
            excess_series = {
                "dates": None,
                "values": None,
                "availability": "missing_requires_9c_survivor_enrichment",
            }
        return {
            "requested_date_range": [
                split.boundary.requested_start,
                split.boundary.requested_end,
            ],
            "actual_date_range": [split.boundary.actual_start, split.boundary.actual_end],
            "rebalance_dates": [str(value) for value in split.rebalance_dates],
            "rebalance_periods": int(split.rebalance_dates.size),
            "ic": {
                "mean": _finite_or_none(metrics["train_ic"]),
                "std": None,
                "icir": None,
                "valid_periods": int(metrics["train_ic_valid_periods"]),
                "total_periods": int(split.rebalance_dates.size),
            },
            "long": {
                "direction": int(metrics["train_direction"]),
                "mean_period_return": None,
                "annualized_return": None,
                "annualized_ir": _finite_or_none(metrics["train_long_ir"]),
                "std": None,
                "valid_periods": int(metrics["train_long_valid_periods"]),
                "total_periods": int(split.rebalance_dates.size),
                "excess_series": excess_series,
            },
            "barra": {
                "max_abs_correlation": _finite_or_none(metrics["train_barra_ts_corr"]),
                "correlations": correlations,
                "common_valid_periods": valid_periods,
            },
            "raw_long_short_valid_periods": None,
            "factor_finite_coverage": {
                "finite_universe_values": None,
                "eligible_universe_values": None,
                "rate": None,
                "availability": "not_persisted_by_stage5",
            },
            "neutralization": dict(metrics["neutralization"]),
            "metric_origin": TRAIN_METRIC_ORIGIN_REUSE,
            "train_source_id": source_ids,
            "train_source_snapshot_fingerprint": snapshot_fingerprints,
            "train_reuse_verification_fingerprint": self.overlay.verification_fingerprint,
            "train_evaluation_contract_fingerprint": self.overlay.manifest[
                "artifact_train_contract_fingerprint"
                if self.overlay.manifest.get("schema")
                == HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA
                else "stage6_evaluation_contract_fingerprint"
            ],
            "train_reuse_overlay_record_fingerprint": record["record_fingerprint"],
        }

    def evaluate(self, candidate: Mapping[str, Any]) -> Stage6CandidateEvaluationResult:
        total_started = time.perf_counter()
        expression, expression_identity = self.fresh_evaluator._expression(candidate)
        record = self.overlay.get(str(expression_identity["structural_hash"]))
        if record is None:
            fresh = self.fresh_evaluator.evaluate(candidate)
            return self._repackage_fresh(
                candidate, fresh, time.perf_counter() - total_started
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
        train = self._reused_train_payload(record)
        direction = int(train["long"]["direction"])
        if direction not in (-1, 1):
            raise TrainReuseOverlayIntegrityError("reused Train direction is invalid")
        train_seconds = time.perf_counter() - train_started

        validation_started = time.perf_counter()
        prepared_validation = self.fresh_evaluator._prepare_split(factor, "validation")
        self.fresh_evaluator._apply_direction(prepared_validation, direction)
        validation = {
            **prepared_validation.result,
            "metric_origin": VALIDATION_METRIC_ORIGIN_FRESH,
        }
        validation_seconds = time.perf_counter() - validation_started
        coverage = {
            "train": dict(train["factor_finite_coverage"]),
            "validation": dict(validation["factor_finite_coverage"]),
        }
        origins = [dict(value) for value in record["origins"]]
        source_identity = {
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
            "train_metric_origin": TRAIN_METRIC_ORIGIN_REUSE,
            "validation_metric_origin": VALIDATION_METRIC_ORIGIN_FRESH,
            "train_reuse_overlay_fingerprint": self.overlay.fingerprint,
            "train_reuse_verification_fingerprint": self.overlay.verification_fingerprint,
            "train_reuse_origins": origins,
        }
        return self._result(
            expression=expression_identity,
            source_identity=source_identity,
            status="completed",
            invalid_reasons=(),
            direction=direction,
            train=train,
            validation=validation,
            coverage=coverage,
            factor_seconds=factor_seconds,
            train_seconds=train_seconds,
            validation_seconds=validation_seconds,
            total_seconds=time.perf_counter() - total_started,
        )


def select_stage6_mixed_smoke_candidates(
    candidates: Sequence[Mapping[str, Any]],
    overlay: Stage6TrainReuseOverlay,
    *,
    reused_count: int = 6,
    fresh_count: int = 6,
) -> list[dict[str, Any]]:
    """Freeze a deterministic stratified sample without using metric values."""

    if reused_count < 1 or fresh_count < 1:
        raise ValueError("mixed smoke requires positive reused and fresh counts")
    ordered = sorted(candidates, key=lambda row: str(row["current_structural_hash"]))
    reused = [
        dict(row)
        for row in ordered
        if str(row["current_structural_hash"]) in overlay.records
    ]
    fresh = [
        dict(row)
        for row in ordered
        if str(row["current_structural_hash"]) not in overlay.records
    ]
    if len(reused) < reused_count or len(fresh) < fresh_count:
        raise ValueError("registry cannot satisfy the requested mixed smoke strata")

    def spread(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if count == 1:
            return [dict(rows[0])]
        positions = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
        return [dict(rows[int(position)]) for position in positions]

    reused_sample = spread(reused, reused_count)
    fresh_sample = spread(fresh, fresh_count)
    result: list[dict[str, Any]] = []
    for index in range(max(len(reused_sample), len(fresh_sample))):
        if index < len(reused_sample):
            result.append(reused_sample[index])
        if index < len(fresh_sample):
            result.append(fresh_sample[index])
    return result


def _without_metric_origin(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {
        "metric_origin",
        "train_source_id",
        "train_source_snapshot_fingerprint",
        "train_reuse_verification_fingerprint",
        "train_evaluation_contract_fingerprint",
        "train_reuse_overlay_record_fingerprint",
    }}


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / len(rows))


def run_stage6_mixed_evaluation_smoke(
    *,
    accepted_registry_path: str | Path,
    overlay_manifest_path: str | Path,
    output_root: str | Path,
    context: Any | None = None,
    reused_count: int = 6,
    fresh_count: int = 6,
) -> Path:
    """Run the bounded 9B full-fresh versus mixed real-data comparison."""

    registry_path = Path(accepted_registry_path).resolve()
    candidates = list(_iter_jsonl(registry_path))
    overlay = Stage6TrainReuseOverlay.load(overlay_manifest_path)
    if _stable_hash(candidates) != overlay.manifest["accepted_registry_fingerprint"]:
        raise TrainReuseOverlayIntegrityError("accepted registry logical digest mismatch")
    if len(candidates) != int(overlay.manifest["counts"]["accepted_candidates"]):
        raise TrainReuseOverlayIntegrityError("accepted registry count mismatch")
    evaluation_context = context or build_stage6_evaluation_context(
        Stage6EvaluationConfig()
    )
    fresh = Stage6CandidateEvaluator(
        evaluation_context,
        compatibility_audit_fingerprint=str(
            overlay.manifest["compatibility_audit_fingerprint"]
        ),
        accepted_registry_fingerprint=str(
            overlay.manifest["accepted_registry_fingerprint"]
        ),
    )
    mixed_fresh = Stage6CandidateEvaluator(
        evaluation_context,
        compatibility_audit_fingerprint=fresh.compatibility_audit_fingerprint,
        accepted_registry_fingerprint=fresh.accepted_registry_fingerprint,
    )
    mixed = Stage6MixedCandidateEvaluator(mixed_fresh, overlay)
    sample = select_stage6_mixed_smoke_candidates(
        candidates,
        overlay,
        reused_count=reused_count,
        fresh_count=fresh_count,
    )
    sample_hashes = [str(row["current_structural_hash"]) for row in sample]
    stable = {
        "schema": "factor_gfn.stage6_mixed_evaluation_smoke.v1",
        "accepted_registry_fingerprint": overlay.manifest[
            "accepted_registry_fingerprint"
        ],
        "stage6_context_fingerprint": evaluation_context.fingerprint,
        "full_fresh_evaluation_contract_fingerprint": (
            fresh.evaluation_contract_fingerprint
        ),
        "mixed_evaluation_contract_fingerprint": mixed.evaluation_contract_fingerprint,
        "train_reuse_overlay_fingerprint": overlay.fingerprint,
        "train_reuse_verification_fingerprint": overlay.verification_fingerprint,
        "selection_method": "structural_hash_sorted_even_spread_stratified_v1",
        "benchmark_execution_order": "full_fresh_then_mixed_same_process",
        "go_no_go_estimator": (
            "reuse_only_train_evaluation_seconds_saved_excludes_cross_run_"
            "factor_validation_difference_v1"
        ),
        "ordered_candidate_hashes": sample_hashes,
        "reused_count": reused_count,
        "fresh_count": fresh_count,
        "oos": "not_loaded",
    }
    smoke_fingerprint = _stable_hash(stable)
    root = Path(output_root).resolve()
    target = root / smoke_fingerprint
    report_path = target / "mixed_evaluation_smoke_report.json"
    if target.exists():
        report = _read_json(report_path)
        if report.get("smoke_fingerprint") != smoke_fingerprint:
            raise RuntimeError("existing mixed smoke artifact fingerprint conflict")
        return report_path
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{smoke_fingerprint}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        with EvaluationStore(
            temporary / "evaluation_store" / "stage6_evaluations.sqlite",
            temporary / "evaluation_runs",
        ) as store:
            full_run = store.create_run(sample, fresh)
            mixed_run = store.create_run(sample, mixed)
            full_runner = Stage6EvaluationRunner(store, fresh)
            mixed_runner = Stage6EvaluationRunner(store, mixed)
            full_summary = full_runner.run(full_run.run_id)
            mixed_summary = mixed_runner.run(mixed_run.run_id)
            full_verified = store.load_verified_run_results(full_run.run_id)
            mixed_verified = store.load_verified_run_results(mixed_run.run_id)
            full_resume = full_runner.run(full_run.run_id)
            mixed_resume = mixed_runner.run(mixed_run.run_id)
            mixed_determinism = mixed_runner.verify_determinism(
                mixed_run.run_id, [0, 1]
            )

        full_by_hash = {
            str(row["structural_hash"]): dict(row["result"])
            for row in full_verified.records
        }
        mixed_by_hash = {
            str(row["structural_hash"]): dict(row["result"])
            for row in mixed_verified.records
        }
        comparisons: list[dict[str, Any]] = []
        for structural_hash in sample_hashes:
            full_result = full_by_hash[structural_hash]
            mixed_result = mixed_by_hash[structural_hash]
            reused = structural_hash in overlay.records
            if _without_metric_origin(mixed_result["validation"]) != full_result["validation"]:
                raise RuntimeError("mixed Validation differs from full-fresh Validation")
            if reused:
                train_checks = {
                    "train_ic": (
                        full_result["train"]["ic"]["mean"],
                        mixed_result["train"]["ic"]["mean"],
                    ),
                    "train_long_ir": (
                        full_result["train"]["long"]["annualized_ir"],
                        mixed_result["train"]["long"]["annualized_ir"],
                    ),
                    "train_barra_ts_corr": (
                        full_result["train"]["barra"]["max_abs_correlation"],
                        mixed_result["train"]["barra"]["max_abs_correlation"],
                    ),
                }
                for name, (expected, observed) in train_checks.items():
                    if expected is None or observed is None or not math.isclose(
                        float(expected), float(observed), rel_tol=1.0e-10, abs_tol=1.0e-12
                    ):
                        raise RuntimeError(f"reused {name} differs from full-fresh value")
            elif _without_metric_origin(mixed_result["train"]) != full_result["train"]:
                raise RuntimeError("mixed fresh fallback differs from full-fresh Train")
            comparisons.append(
                {
                    "structural_hash": structural_hash,
                    "train_metric_origin": mixed_result["train"]["metric_origin"],
                    "full_fresh": {
                        key: full_result[key]
                        for key in (
                            "factor_seconds",
                            "train_evaluation_seconds",
                            "validation_evaluation_seconds",
                            "total_seconds",
                        )
                    },
                    "mixed": {
                        key: mixed_result[key]
                        for key in (
                            "factor_seconds",
                            "train_evaluation_seconds",
                            "validation_evaluation_seconds",
                            "total_seconds",
                        )
                    },
                    "validation_exact_match": True,
                    "screening_train_metrics_match": True,
                }
            )

        reused_rows = [
            row for row in comparisons
            if row["train_metric_origin"] == TRAIN_METRIC_ORIGIN_REUSE
        ]
        fresh_rows = [
            row for row in comparisons
            if row["train_metric_origin"] == TRAIN_METRIC_ORIGIN_FRESH
        ]
        coverage = float(overlay.manifest["coverage_ratio"])

        def summarize(rows: Sequence[Mapping[str, Any]], path: str) -> dict[str, float]:
            selected = [row[path] for row in rows]
            return {
                key: _mean(selected, key)
                for key in (
                    "factor_seconds",
                    "train_evaluation_seconds",
                    "validation_evaluation_seconds",
                    "total_seconds",
                )
            }

        full_reused = summarize(reused_rows, "full_fresh")
        mixed_reused = summarize(reused_rows, "mixed")
        full_fresh_stratum = summarize(fresh_rows, "full_fresh")
        mixed_fresh_stratum = summarize(fresh_rows, "mixed")
        projected_full = (
            coverage * full_reused["total_seconds"]
            + (1.0 - coverage) * full_fresh_stratum["total_seconds"]
        )
        projected_mixed = (
            coverage * mixed_reused["total_seconds"]
            + (1.0 - coverage) * mixed_fresh_stratum["total_seconds"]
        )
        accepted_count = int(overlay.manifest["counts"]["accepted_candidates"])
        projected_saved = max(0.0, projected_full - projected_mixed) * accepted_count
        train_saved_per_reused = (
            full_reused["train_evaluation_seconds"]
            - mixed_reused["train_evaluation_seconds"]
        )
        conservative_saved_per_candidate = coverage * max(0.0, train_saved_per_reused)
        conservative_projected_mixed = projected_full - conservative_saved_per_candidate
        conservative_saved_total = conservative_saved_per_candidate * accepted_count
        report = {
            **stable,
            "smoke_fingerprint": smoke_fingerprint,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "created_at_excluded_from_fingerprint": True,
            "full_fresh_run_id": full_run.run_id,
            "mixed_run_id": mixed_run.run_id,
            "full_fresh_ordered_result_set_fingerprint": (
                full_verified.ordered_result_set_fingerprint
            ),
            "mixed_ordered_result_set_fingerprint": (
                mixed_verified.ordered_result_set_fingerprint
            ),
            "comparisons": comparisons,
            "summary": {
                "registry_reuse_coverage_ratio": coverage,
                "registry_reusable_candidates": len(overlay.records),
                "registry_accepted_candidates": accepted_count,
                "sample_reused_candidates": len(reused_rows),
                "sample_fresh_candidates": len(fresh_rows),
                "full_fresh_reused_stratum_mean_seconds": full_reused,
                "mixed_reused_stratum_mean_seconds": mixed_reused,
                "full_fresh_nonreused_stratum_mean_seconds": full_fresh_stratum,
                "mixed_nonreused_stratum_mean_seconds": mixed_fresh_stratum,
                "reused_stratum_speedup_ratio": (
                    full_reused["total_seconds"] / mixed_reused["total_seconds"]
                ),
                "reused_stratum_fraction_saved": (
                    1.0 - mixed_reused["total_seconds"] / full_reused["total_seconds"]
                ),
                "raw_ordered_projected_full_fresh_seconds_per_candidate": projected_full,
                "raw_ordered_projected_mixed_seconds_per_candidate": projected_mixed,
                "raw_ordered_projected_overall_speedup_ratio": projected_full / projected_mixed,
                "raw_ordered_projected_overall_fraction_saved": 1.0 - projected_mixed / projected_full,
                "raw_ordered_projected_full_registry_seconds_saved": projected_saved,
                "raw_ordered_projected_full_registry_hours_saved": projected_saved / 3600.0,
                "train_evaluation_seconds_saved_per_reused_candidate": train_saved_per_reused,
                "go_no_go_projected_full_fresh_seconds_per_candidate": projected_full,
                "go_no_go_projected_mixed_seconds_per_candidate": conservative_projected_mixed,
                "go_no_go_projected_overall_speedup_ratio": (
                    projected_full / conservative_projected_mixed
                ),
                "go_no_go_projected_overall_fraction_saved": (
                    conservative_saved_per_candidate / projected_full
                ),
                "go_no_go_projected_full_registry_seconds_saved": (
                    conservative_saved_total
                ),
                "go_no_go_projected_full_registry_hours_saved": (
                    conservative_saved_total / 3600.0
                ),
                "go_no_go_excluded_cross_run_factor_validation_speed_difference": True,
            },
            "runner": {
                "full_fresh": asdict(full_summary),
                "mixed": asdict(mixed_summary),
                "full_fresh_resume": asdict(full_resume),
                "mixed_resume": asdict(mixed_resume),
                "mixed_determinism": mixed_determinism,
            },
            "go_no_go": "HUMAN_REVIEW_REQUIRED",
            "scope": "9B_only_no_9C_no_selection_no_oos",
        }
        (temporary / "mixed_evaluation_smoke_report.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return report_path


__all__ = [
    "STAGE6_MIXED_EVALUATION_CONTRACT_SCHEMA",
    "STAGE6_MIXED_EVALUATOR_VERSION",
    "TRAIN_METRIC_ORIGIN_FRESH",
    "TRAIN_METRIC_ORIGIN_REUSE",
    "VALIDATION_METRIC_ORIGIN_FRESH",
    "Stage6MixedCandidateEvaluator",
    "Stage6TrainReuseOverlay",
    "TrainReuseOverlayIntegrityError",
    "select_stage6_mixed_smoke_candidates",
    "run_stage6_mixed_evaluation_smoke",
]
