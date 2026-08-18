"""Source-batch audit for temporary Stage 5 -> Stage 6 Train reuse.

This module implements Stage 6 step 9A only.  It never mutates the accepted
candidate registry and does not seed :mod:`stage6_evaluation_store`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Callable, Iterable, Mapping, Sequence

from factor_gfn.barra import STYLE_NAMES
from factor_gfn.evaluator.numba_kernels import NUMBA_KERNEL_SCHEMA
from factor_gfn.gfn.real_data import (
    RealRewardDataConfig,
    RealRewardDataPaths,
    build_real_reward_data_context,
)
from factor_gfn.gfn.real_reward import RealRewardProvider
from factor_gfn.grammar.tokens import get_action

from .candidate_import import CANDIDATE_IMPORT_MANIFEST_SCHEMA
from .expression_compatibility import (
    ACCEPTED_REGISTRY_SCHEMA,
    EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA,
)
from .sources import SOURCE_SET_SCHEMA, SOURCE_SNAPSHOT_SCHEMA
from .stage6_evaluation import (
    Stage6CandidateEvaluationResult,
    Stage6CandidateEvaluator,
    Stage6EvaluationConfig,
    build_stage6_evaluation_context,
)


TRAIN_REUSE_AUDITOR_VERSION = "factor_gfn.stage6_train_reuse_auditor.v1"
TRAIN_REUSE_SOURCE_AUDIT_SCHEMA = "factor_gfn.stage6_train_reuse_source_audit.v1"
TRAIN_REUSE_VERIFICATION_SCHEMA = "factor_gfn.stage6_train_reuse_verification.v1"
TRAIN_REUSE_OVERLAY_SCHEMA = "factor_gfn.stage6_train_reuse_overlay_record.v1"
TRAIN_REUSE_MANIFEST_SCHEMA = "factor_gfn.stage6_train_reuse_manifest.v1"
HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA = (
    "factor_gfn.stage6_train_reuse_overlay_record.v2"
)
HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA = "factor_gfn.stage6_train_reuse_manifest.v2"
HYBRID_TRAIN_REUSE_ADAPTER_VERSION = (
    "factor_gfn.stage6.hybrid_train_artifact_reuse_adapter.v1"
)
HYBRID_EXACT_CONTRACT_VERIFICATION = "hybrid_exact_contract"
HYBRID_FULL_FRESH_TRAIN_FALLBACK = "hybrid_full_fresh_train_fallback"
HYBRID_TRAIN_CONTRACT_MISMATCH_REASON = "current_train_contract_mismatch"
V6_EQUIVALENCE_AUDITOR_VERSION = "factor_gfn.stage6_v6_equivalence_auditor.v2"
V6_EQUIVALENCE_RECORD_SCHEMA = "factor_gfn.stage6_v6_equivalence_record.v2"
V6_EQUIVALENCE_MANIFEST_SCHEMA = "factor_gfn.stage6_v6_equivalence_manifest.v2"

V6_EQUIVALENCE_TARGET_BATCH_ID = (
    "bd9b609cd73000ff007622ec20a15eb18ad1c26f674795f43b47d6218151da15"
)
V6_EQUIVALENCE_EVIDENCE_PASSED = "EQUIVALENCE_EVIDENCE_PASSED"
V6_EQUIVALENCE_EVIDENCE_FAILED = "EQUIVALENCE_EVIDENCE_FAILED"

TRAIN_REUSE_NOT_ALLOWED = "TRAIN_REUSE_NOT_ALLOWED"
TRAIN_REUSE_NUMERIC_VERIFICATION_REQUIRED = "TRAIN_REUSE_NUMERIC_VERIFICATION_REQUIRED"
TRAIN_METRICS_REUSABLE = "TRAIN_METRICS_REUSABLE"

_NUMERIC_ATOL = 1.0e-12
_NUMERIC_RTOL = 1.0e-10
_SAMPLE_SIZE = 24
_CURRENT_COMPATIBLE_PROVIDER_SCHEMAS = frozenset(
    {
        "factor_gfn.real_reward_provider.v7",
        "factor_gfn.real_reward_provider.v8",
    }
)
_V6_EXPECTED_PROJECTION_DIFFERENCES = frozenset(
    {
        "industry_policy.encoding_schema",
        "industry_policy.projection",
        "numeric_kernel_schema",
        "numba_kernel_schema",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value)).hexdigest()


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
        raise RuntimeError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.strip():
                    raise RuntimeError(f"blank JSONL row: {path}:{line_number}")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise RuntimeError(f"JSONL row is not an object: {path}:{line_number}")
                yield line_number, value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSONL artifact: {path}") from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise RuntimeError(f"{label} fingerprint mismatch: {path}")


def _snapshot_fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": manifest["schema"],
        "source_id": manifest["source_id"],
        "source_type": manifest["source_type"],
        "source_role": manifest["source_role"],
        "inclusion_status": manifest["inclusion_status"],
        "approval_note": manifest["approval_note"],
        "candidate_record_policy": manifest["candidate_record_policy"],
        "source_semantics": manifest["source_semantics"],
        "snapshot_kind": manifest["snapshot_kind"],
        "cutoff": manifest["cutoff"],
        "record_counts": manifest["record_counts"],
        "artifacts": [
            {
                "name": item["name"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in manifest["artifacts"]
        ],
        "logical_content_fingerprint": manifest["logical_content_fingerprint"],
    }
    if "external_artifacts" in manifest:
        payload["external_artifacts"] = manifest["external_artifacts"]
    return payload


def _source_set_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest["schema"],
        "mode": manifest["mode"],
        "sources": [
            {
                "source_id": source["source_id"],
                "source_type": source["source_type"],
                "source_role": source["source_role"],
                "snapshot_fingerprint": source["snapshot_fingerprint"],
            }
            for source in manifest["sources"]
        ],
    }


def _verify_source_set(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(path)
    if manifest.get("schema") != SOURCE_SET_SCHEMA:
        raise RuntimeError("source-set schema mismatch")
    claimed = str(manifest.get("source_set_fingerprint"))
    if _stable_hash(_source_set_payload(manifest)) != claimed or path.parent.name != claimed:
        raise RuntimeError("source-set fingerprint mismatch")
    snapshots: list[dict[str, Any]] = []
    for source in manifest.get("source_manifests", []):
        snapshot_path = Path(str(source["snapshot_manifest"])).resolve()
        snapshot = _read_json(snapshot_path)
        if snapshot.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
            raise RuntimeError(f"snapshot schema mismatch: {source['source_id']}")
        fingerprint = _stable_hash(_snapshot_fingerprint_payload(snapshot))
        if (
            fingerprint != source.get("snapshot_fingerprint")
            or fingerprint != snapshot.get("snapshot_fingerprint")
            or snapshot_path.parent.name != fingerprint
        ):
            raise RuntimeError(f"snapshot fingerprint mismatch: {source['source_id']}")
        if snapshot.get("source_id") != source.get("source_id"):
            raise RuntimeError("source-set/snapshot source identity mismatch")
        for artifact in snapshot.get("artifacts", []):
            artifact_path = snapshot_path.parent / str(artifact["name"])
            _require_sha(artifact_path, str(artifact["sha256"]), "snapshot artifact")
            if artifact_path.stat().st_size != int(artifact["size_bytes"]):
                raise RuntimeError(f"snapshot artifact size mismatch: {artifact_path}")
        for artifact in snapshot.get("external_artifacts", []):
            if not isinstance(artifact, Mapping):
                raise RuntimeError("external snapshot artifact entry is invalid")
            artifact_path = Path(str(artifact.get("source_path"))).resolve()
            _require_sha(
                artifact_path,
                str(artifact.get("sha256")),
                "external snapshot artifact",
            )
            if artifact_path.stat().st_size != int(artifact.get("size_bytes", -1)):
                raise RuntimeError(
                    f"external snapshot artifact size mismatch: {artifact_path}"
                )
        snapshots.append({**snapshot, "_directory": snapshot_path.parent})
    if len(snapshots) != len(manifest.get("sources", [])):
        raise RuntimeError("source-set source manifest count mismatch")
    return manifest, snapshots


def _verify_candidate_inputs(
    candidate_import_manifest_path: Path,
    compatibility_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    candidate_manifest = _read_json(candidate_import_manifest_path)
    if candidate_manifest.get("schema") != CANDIDATE_IMPORT_MANIFEST_SCHEMA:
        raise RuntimeError("candidate import manifest schema mismatch")
    compatibility_manifest = _read_json(compatibility_manifest_path)
    if compatibility_manifest.get("schema") != EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA:
        raise RuntimeError("compatibility manifest schema mismatch")
    if (
        compatibility_manifest.get("candidate_registry_fingerprint")
        != candidate_manifest.get("registry_fingerprint")
    ):
        raise RuntimeError("candidate/compatibility registry fingerprint mismatch")
    artifact = compatibility_manifest.get("artifacts", {}).get(
        "auto_accepted_candidate_registry.jsonl"
    )
    if not isinstance(artifact, Mapping):
        raise RuntimeError("accepted registry artifact identity missing")
    accepted_path = compatibility_manifest_path.parent / "auto_accepted_candidate_registry.jsonl"
    _require_sha(accepted_path, str(artifact["sha256"]), "accepted registry")
    rows = [row for _, row in _iter_jsonl(accepted_path)]
    if len(rows) != int(compatibility_manifest.get("counts", {}).get("AUTO_ACCEPT", -1)):
        raise RuntimeError("accepted registry row count mismatch")
    for row in rows:
        if row.get("schema") != ACCEPTED_REGISTRY_SCHEMA:
            raise RuntimeError("accepted registry row schema mismatch")
    if _stable_hash(rows) != compatibility_manifest.get("digests", {}).get(
        "accepted_registry"
    ):
        raise RuntimeError("accepted registry logical digest mismatch")
    return candidate_manifest, compatibility_manifest, rows


def _verify_existing_overlay(path: Path) -> tuple[dict[str, Any], set[str]]:
    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != TRAIN_REUSE_MANIFEST_SCHEMA:
        raise RuntimeError("existing Train reuse manifest schema mismatch")
    artifact_name = "train_reuse_overlay.jsonl"
    metadata = manifest.get("artifacts", {}).get(artifact_name)
    artifact_path = manifest_path.parent / artifact_name
    if not isinstance(metadata, Mapping) or not artifact_path.is_file():
        raise RuntimeError("existing Train reuse overlay artifact missing")
    if artifact_path.stat().st_size != int(metadata.get("size_bytes", -1)):
        raise RuntimeError("existing Train reuse overlay size mismatch")
    if _sha256_file(artifact_path) != metadata.get("sha256"):
        raise RuntimeError("existing Train reuse overlay digest mismatch")
    rows = [row for _, row in _iter_jsonl(artifact_path)]
    if any(row.get("schema") != TRAIN_REUSE_OVERLAY_SCHEMA for row in rows):
        raise RuntimeError("existing Train reuse overlay row schema mismatch")
    hashes = {str(row["structural_hash"]) for row in rows}
    if len(hashes) != len(rows):
        raise RuntimeError("existing Train reuse overlay contains duplicate candidates")
    if len(rows) != int(manifest.get("counts", {}).get("overlay_candidates", -1)):
        raise RuntimeError("existing Train reuse overlay count mismatch")
    return manifest, hashes


def _provider_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    reward_config = manifest.get("reward_config", {})
    interpreter = manifest.get("interpreter", {})
    industry = manifest.get("industry_neutralization", {})
    if not isinstance(reward_config, Mapping):
        reward_config = {}
    if not isinstance(interpreter, Mapping):
        interpreter = {}
    if not isinstance(industry, Mapping):
        industry = {}
    return {
        "context_fingerprint": manifest.get("context_fingerprint"),
        "reward_evaluator_context_fingerprint": manifest.get(
            "reward_evaluator_context_fingerprint"
        ),
        "evaluation_config": manifest.get("evaluation_config"),
        "candidate_industry_neutralization": reward_config.get(
            "candidate_industry_neutralization"
        ),
        "barra_min_common_periods": reward_config.get("barra_min_common_periods"),
        "calendar": manifest.get("calendar"),
        "reward_panel": manifest.get("reward_panel"),
        "numeric_kernel_schema": interpreter.get("numeric_kernel_schema"),
        "numba_kernel_schema": interpreter.get("numba_kernel_schema"),
        "industry_policy": {
            "enabled": industry.get("enabled"),
            "policy_schema": industry.get("policy_schema"),
            "encoding_schema": industry.get("encoding_schema"),
            "projection": industry.get("projection"),
            "failed_date_action": industry.get("failed_date_action"),
            "unknown_industry_stock_action": industry.get(
                "unknown_industry_stock_action"
            ),
            "calendar_action": industry.get("calendar_action"),
        },
        "cleaning_contract": {
            "winsor_quantiles": [0.01, 0.99],
            "zscore_ddof": 0,
            "rank_ic": "shared_reward_metrics_implementation",
            "long_ir": "shared_reward_metrics_implementation",
            "barra": "shared_reward_barra_implementation",
        },
    }


def _provider_manifest_catalog(
    snapshots: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    catalog: dict[str, dict[str, Any]] = {}
    conflicts: dict[str, list[str]] = defaultdict(list)
    for snapshot in snapshots:
        directory = Path(snapshot["_directory"])
        metadata_path = directory / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = _read_json(metadata_path)
        provider = metadata.get("reward_provider")
        claimed = metadata.get("reward_provider_fingerprint")
        if not isinstance(provider, dict) or not isinstance(claimed, str):
            continue
        if _stable_hash(provider) != claimed:
            conflicts[claimed].append(str(snapshot["source_id"]))
            continue
        existing = catalog.get(claimed)
        if existing is not None and existing != provider:
            conflicts[claimed].append(str(snapshot["source_id"]))
            continue
        catalog[claimed] = provider
    return catalog, dict(conflicts)


def _batch_sources(
    snapshots: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        semantics = snapshot.get("source_semantics", {})
        provider = semantics.get("provider_fingerprint")
        context = semantics.get("context_fingerprint")
        key = (str(provider) if provider else "<missing>", str(context) if context else "<missing>")
        grouped[key].append(snapshot)
    batches: list[dict[str, Any]] = []
    for (provider, context), members in grouped.items():
        source_ids = sorted(str(member["source_id"]) for member in members)
        payload = {
            "provider_fingerprint": provider,
            "context_fingerprint": context,
            "source_ids": source_ids,
        }
        batches.append(
            {
                "batch_id": _stable_hash(payload),
                **payload,
                "snapshots": sorted(members, key=lambda value: str(value["source_id"])),
            }
        )
    return sorted(batches, key=lambda value: value["batch_id"])


def _static_audit_batches(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    target_provider_manifest: Mapping[str, Any],
    target_provider_fingerprint: str,
) -> list[dict[str, Any]]:
    catalog, catalog_conflicts = _provider_manifest_catalog(snapshots)
    target_projection = _provider_projection(target_provider_manifest)
    target_projection_fingerprint = _stable_hash(target_projection)
    audits: list[dict[str, Any]] = []
    for batch in _batch_sources(snapshots):
        provider_fp = str(batch["provider_fingerprint"])
        provider = catalog.get(provider_fp)
        reasons: list[str] = []
        evidence: dict[str, Any] = {
            "provider_manifest_resolved": provider is not None,
            "provider_fingerprint_matches_current": provider_fp
            == target_provider_fingerprint,
            "target_numeric_kernel_schema": NUMBA_KERNEL_SCHEMA,
        }
        if provider_fp in catalog_conflicts:
            reasons.append("provider_manifest_conflict")
        if provider is None:
            reasons.append("implementation_identity_missing")
            source_projection = None
        else:
            source_projection = _provider_projection(provider)
            schema = provider.get("schema")
            evidence["provider_schema"] = schema
            evidence["source_projection_fingerprint"] = _stable_hash(source_projection)
            if schema not in _CURRENT_COMPATIBLE_PROVIDER_SCHEMAS:
                reasons.append("provider_schema_not_proven_compatible")
            for field in (
                "reward_evaluator_context_fingerprint",
                "numeric_kernel_schema",
                "numba_kernel_schema",
            ):
                if not source_projection.get(field):
                    reasons.append(f"implementation_identity_missing:{field}")
            for field in sorted(target_projection):
                if source_projection.get(field) != target_projection.get(field):
                    reasons.append(f"train_contract_mismatch:{field}")
        status = (
            TRAIN_REUSE_NOT_ALLOWED
            if reasons
            else TRAIN_REUSE_NUMERIC_VERIFICATION_REQUIRED
        )
        audits.append(
            {
                "schema": TRAIN_REUSE_SOURCE_AUDIT_SCHEMA,
                "batch_id": batch["batch_id"],
                "source_ids": batch["source_ids"],
                "provider_fingerprint": provider_fp,
                "context_fingerprint": batch["context_fingerprint"],
                "status": status,
                "reason_codes": sorted(set(reasons)),
                "evidence": evidence,
                "target_provider_fingerprint": target_provider_fingerprint,
                "target_train_contract_projection_fingerprint": target_projection_fingerprint,
            }
        )
    return audits


def _extract_metrics(reward_result: Any) -> dict[str, Any] | None:
    if not isinstance(reward_result, Mapping):
        return None
    correlations = reward_result.get("barra_correlations")
    periods = reward_result.get("barra_valid_periods")
    if not isinstance(correlations, Mapping) or not isinstance(periods, Mapping):
        return None
    payload = {
        "train_ic": reward_result.get("train_ic"),
        "train_ic_valid_periods": reward_result.get("ic_valid_periods"),
        "train_direction": reward_result.get("long_direction"),
        "train_long_ir": reward_result.get("train_long_ir"),
        "train_long_valid_periods": reward_result.get("long_ir_valid_periods"),
        "train_barra_ts_corr": reward_result.get("barra_ts_corr"),
        "train_barra_correlations": {
            name: correlations.get(name) for name in STYLE_NAMES
        },
        "train_barra_valid_periods_by_style": {
            name: periods.get(name) for name in STYLE_NAMES
        },
        "neutralization": {
            "industry_neutralized": reward_result.get("industry_neutralized"),
            "skipped_dates": reward_result.get("neutralization_skipped_dates"),
            "skipped_rate": reward_result.get("neutralization_skipped_rate"),
            "details": reward_result.get("neutralization_skipped_details"),
        },
    }
    finite_fields = (
        payload["train_ic"],
        payload["train_long_ir"],
        payload["train_barra_ts_corr"],
        *payload["train_barra_correlations"].values(),
    )
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in finite_fields):
        return None
    if payload["train_direction"] not in (-1, 1):
        return None
    integer_fields = (
        payload["train_ic_valid_periods"],
        payload["train_long_valid_periods"],
        *payload["train_barra_valid_periods_by_style"].values(),
    )
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in integer_fields):
        return None
    if payload["neutralization"]["industry_neutralized"] is not True:
        return None
    if not isinstance(payload["neutralization"]["skipped_dates"], list):
        return None
    if not isinstance(payload["neutralization"]["details"], list):
        return None
    if not isinstance(payload["neutralization"]["skipped_rate"], (int, float)):
        return None
    return payload


def _record_reward_result(record: Mapping[str, Any]) -> Any:
    metadata = record.get("metadata")
    return metadata.get("reward_result") if isinstance(metadata, Mapping) else None


def _read_source_metric_records(
    snapshot: Mapping[str, Any],
    accepted_hashes: set[str],
) -> list[dict[str, Any]]:
    directory = Path(snapshot["_directory"])
    source_id = str(snapshot["source_id"])
    source_type = str(snapshot["source_type"])
    rows: list[dict[str, Any]] = []
    if source_type in {"discovery_run", "diagnostic_audit"}:
        artifact_name = (
            "evaluations.jsonl" if source_type == "discovery_run" else "candidate_audit.jsonl"
        )
        for line_number, record in _iter_jsonl(directory / artifact_name):
            structural_hash = record.get("structural_hash")
            if structural_hash not in accepted_hashes:
                continue
            rows.append(
                {
                    "structural_hash": structural_hash,
                    "source_id": source_id,
                    "snapshot_fingerprint": snapshot["snapshot_fingerprint"],
                    "locator": {"artifact": artifact_name, "line_number": line_number},
                    "metrics": _extract_metrics(_record_reward_result(record)),
                }
            )
    elif source_type == "exhaustive_registry":
        connection = sqlite3.connect(f"file:{(directory / 'exhaustive_registry.sqlite3').as_posix()}?mode=ro", uri=True)
        try:
            query = "SELECT rowid, structural_hash, reward_details_json FROM candidates ORDER BY structural_hash"
            for rowid, structural_hash, details_json in connection.execute(query):
                if structural_hash not in accepted_hashes:
                    continue
                try:
                    details = json.loads(details_json) if details_json else {}
                except json.JSONDecodeError:
                    details = {}
                rows.append(
                    {
                        "structural_hash": structural_hash,
                        "source_id": source_id,
                        "snapshot_fingerprint": snapshot["snapshot_fingerprint"],
                        "locator": {"artifact": "exhaustive_registry.sqlite3", "rowid": rowid},
                        "metrics": _extract_metrics(details.get("reward_result")),
                    }
                )
        finally:
            connection.close()
    return rows


def _numeric_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return left == right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return False
    return abs(float(left) - float(right)) <= _NUMERIC_ATOL + _NUMERIC_RTOL * max(
        abs(float(left)), abs(float(right))
    )


def _flatten_metrics(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, Mapping):
            flattened.update(_flatten_metrics(item, path))
        elif path != "neutralization.details":
            flattened[path] = item
    return flattened


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, path))
        else:
            flattened[path] = item
    return flattened


def _projection_difference_evidence(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    source_flat = _flatten_mapping(source)
    target_flat = _flatten_mapping(target)
    differences = {
        field: {"stage5_v6": source_flat.get(field), "current_stage6": target_flat.get(field)}
        for field in sorted(set(source_flat) | set(target_flat))
        if source_flat.get(field) != target_flat.get(field)
    }
    unexpected = sorted(set(differences) - _V6_EXPECTED_PROJECTION_DIFFERENCES)
    return {
        "differences": differences,
        "expected_difference_fields": sorted(_V6_EXPECTED_PROJECTION_DIFFERENCES),
        "unexpected_difference_fields": unexpected,
        "all_differences_are_declared_implementation_fields": not unexpected,
    }


def _compare_v6_equivalence_metrics(
    stage5_metrics: Mapping[str, Any], stage6_metrics: Mapping[str, Any]
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    old_flat = _flatten_metrics(stage5_metrics)
    fresh_flat = _flatten_metrics(stage6_metrics)
    comparisons: dict[str, Any] = {}
    metrics_pass = set(old_flat) == set(fresh_flat)
    for field in sorted(set(old_flat) | set(fresh_flat)):
        old_value = old_flat.get(field)
        fresh_value = fresh_flat.get(field)
        field_pass = field in old_flat and field in fresh_flat and _numeric_equal(
            old_value, fresh_value
        )
        comparisons[field] = {
            "stage5_v6": old_value,
            "current_stage6": fresh_value,
            "absolute_error": (
                abs(float(old_value) - float(fresh_value))
                if isinstance(old_value, (int, float))
                and isinstance(fresh_value, (int, float))
                and not isinstance(old_value, bool)
                and not isinstance(fresh_value, bool)
                else None
            ),
            "pass": field_pass,
        }
        metrics_pass &= field_pass

    old_details = stage5_metrics.get("neutralization", {}).get("details")
    fresh_details = stage6_metrics.get("neutralization", {}).get("details")

    def normalize_details(value: Any) -> Any:
        if not isinstance(value, list):
            return None
        normalized = []
        for item in value:
            if not isinstance(item, Mapping):
                return None
            normalized.append(
                {
                    "date": item.get("date"),
                    "global_row": item.get("global_row", item.get("row_index")),
                    "factor_valid_count": item.get("factor_valid_count"),
                    "known_industry_count": item.get("known_industry_count"),
                    "industry_count": item.get("industry_count"),
                    "required_regression_count": item.get(
                        "required_regression_count"
                    ),
                    "reason": item.get("reason"),
                }
            )
        return normalized

    old_normalized = normalize_details(old_details)
    fresh_normalized = normalize_details(fresh_details)
    details_pass = old_normalized is not None and old_normalized == fresh_normalized
    details_comparison = {
        "comparison": "ordered_semantic_records_with_row_index_mapped_to_global_row",
        "stage5_v6": old_details,
        "current_stage6": fresh_details,
        "stage5_v6_normalized": old_normalized,
        "current_stage6_normalized": fresh_normalized,
        "raw_exact_pass": old_details == fresh_details,
        "stage5_v6_digest": _stable_hash(old_details) if isinstance(old_details, list) else None,
        "current_stage6_digest": (
            _stable_hash(fresh_details) if isinstance(fresh_details, list) else None
        ),
        "pass": details_pass,
    }
    return metrics_pass and details_pass, comparisons, details_comparison


def _metrics_from_stage6(result: Stage6CandidateEvaluationResult) -> dict[str, Any]:
    train = result.train
    return {
        "train_ic": train["ic"]["mean"],
        "train_ic_valid_periods": train["ic"]["valid_periods"],
        "train_direction": result.train_direction,
        "train_long_ir": train["long"]["annualized_ir"],
        "train_long_valid_periods": train["long"]["valid_periods"],
        "train_barra_ts_corr": train["barra"]["max_abs_correlation"],
        "train_barra_correlations": dict(train["barra"]["correlations"]),
        "train_barra_valid_periods_by_style": dict(
            train["barra"]["common_valid_periods"]
        ),
        "neutralization": {
            "industry_neutralized": True,
            "skipped_dates": list(train["neutralization"]["skipped_dates"]),
            "skipped_rate": train["neutralization"]["skipped_rate"],
            "details": list(train["neutralization"]["details"]),
        },
    }


def _candidate_tags(candidate: Mapping[str, Any]) -> set[str]:
    node_count = int(candidate["node_count"])
    depth = int(candidate["depth"])
    actions = [get_action(int(token_id)) for token_id in candidate["prefix_token_ids"]]
    tags = {
        f"node:{node_count}",
        f"depth:{depth}",
        "length:short" if node_count <= 5 else "length:medium" if node_count <= 12 else "length:long",
    }
    if any(action.arity == 1 for action in actions):
        tags.add("arity:unary")
    if any(action.arity == 2 for action in actions):
        tags.add("arity:binary")
    if any(action.name.startswith("ts_") for action in actions):
        tags.add("family:ts")
    if any(action.name.startswith("cs_") for action in actions):
        tags.add("family:cs")
    return tags


def select_representative_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    limit: int = _SAMPLE_SIZE,
) -> list[Mapping[str, Any]]:
    """Select a metric-blind deterministic structural coverage sample."""

    remaining = sorted(candidates, key=lambda value: str(value["current_structural_hash"]))
    selected: list[Mapping[str, Any]] = []
    covered: set[str] = set()
    while remaining and len(selected) < limit:
        def score(candidate: Mapping[str, Any]) -> tuple[int, str]:
            new_tags = _candidate_tags(candidate) - covered
            weighted = sum(
                10 if tag.startswith(("length:", "arity:", "family:")) else 2
                for tag in new_tags
            )
            return (-weighted, str(candidate["current_structural_hash"]))

        chosen = min(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        covered.update(_candidate_tags(chosen))
    return selected


def _canonical_metric_record(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any] | None, list[str]]:
    complete = [record for record in records if record.get("metrics") is not None]
    if not complete:
        return None, ["reusable_train_fields_incomplete"]
    first = complete[0]
    first_flat = _flatten_metrics(first["metrics"])
    for record in complete[1:]:
        current_flat = _flatten_metrics(record["metrics"])
        if set(first_flat) != set(current_flat) or any(
            not _numeric_equal(first_flat[key], current_flat[key])
            for key in first_flat
        ):
            return None, ["source_metric_representation_conflict"]
    return dict(first), []


def _hybrid_train_scope_projection(
    provider_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    reward_config = provider_manifest.get("reward_config") or {}
    if not isinstance(reward_config, Mapping):
        reward_config = {}
    return {
        "provider_schema": provider_manifest.get("schema"),
        "data_scope": provider_manifest.get("data_scope"),
        "context_fingerprint": provider_manifest.get("context_fingerprint"),
        "reward_evaluator_context_fingerprint": provider_manifest.get(
            "reward_evaluator_context_fingerprint"
        ),
        "evaluation_config": provider_manifest.get("evaluation_config"),
        "barra_metric_config": {
            "min_common_periods": reward_config.get("barra_min_common_periods"),
            "candidate_industry_neutralization": reward_config.get(
                "candidate_industry_neutralization"
            ),
        },
        "calendar": provider_manifest.get("calendar"),
        "reward_panel": provider_manifest.get("reward_panel"),
        "interpreter": provider_manifest.get("interpreter"),
        "industry_neutralization": provider_manifest.get(
            "industry_neutralization"
        ),
    }


def _current_hybrid_train_contract(
    provider_manifest: Mapping[str, Any],
    provider_fingerprint: str,
) -> tuple[dict[str, Any], str]:
    gfn_root = Path(__file__).resolve().parents[1] / "gfn"
    contract = {
        "schema": "factor_gfn.train_evaluation_contract.v1",
        "provider_fingerprint": provider_fingerprint,
        "train_scope_projection": _hybrid_train_scope_projection(
            provider_manifest
        ),
        "implementation": {
            "artifact_module_sha256": _sha256_file(
                gfn_root / "train_candidate_artifact.py"
            ),
            "reward_module_sha256": _sha256_file(gfn_root / "reward.py"),
            "real_reward_module_sha256": _sha256_file(
                gfn_root / "real_reward.py"
            ),
        },
    }
    return contract, _stable_hash(contract)


def _hybrid_artifact_train_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    correlations = record.get("train_barra_correlations")
    periods = record.get("train_barra_valid_periods_by_style")
    neutralization = record.get("neutralization_diagnostics")
    if not isinstance(correlations, Mapping) or not isinstance(periods, Mapping):
        raise RuntimeError("Hybrid artifact Train Barra summary is incomplete")
    if not isinstance(neutralization, Mapping):
        raise RuntimeError("Hybrid artifact neutralization diagnostics are incomplete")
    payload = {
        "train_ic": record.get("train_ic"),
        "train_ic_valid_periods": record.get("train_ic_valid_periods"),
        "train_direction": record.get("train_direction"),
        "train_long_ir": record.get("train_long_ir"),
        "train_long_valid_periods": record.get("train_long_valid_periods"),
        "train_barra_ts_corr": record.get("train_barra_ts_corr"),
        "train_barra_correlations": {
            name: correlations.get(name) for name in STYLE_NAMES
        },
        "train_barra_valid_periods_by_style": {
            name: periods.get(name) for name in STYLE_NAMES
        },
        "neutralization": {
            "industry_neutralized": neutralization.get("industry_neutralized"),
            "skipped_dates": neutralization.get("skipped_dates"),
            "skipped_rate": neutralization.get("skipped_rate"),
            "details": neutralization.get("details"),
        },
    }
    finite_fields = (
        payload["train_ic"],
        payload["train_long_ir"],
        payload["train_barra_ts_corr"],
        *payload["train_barra_correlations"].values(),
        payload["neutralization"]["skipped_rate"],
    )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in finite_fields
    ):
        raise RuntimeError("Hybrid artifact Train summary contains non-finite values")
    integer_fields = (
        payload["train_ic_valid_periods"],
        payload["train_long_valid_periods"],
        *payload["train_barra_valid_periods_by_style"].values(),
    )
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in integer_fields
    ):
        raise RuntimeError("Hybrid artifact Train valid-period counts are invalid")
    if payload["train_direction"] not in (-1, 1):
        raise RuntimeError("Hybrid artifact Train direction is invalid")
    if payload["neutralization"]["industry_neutralized"] is not True:
        raise RuntimeError("Hybrid artifact Train neutralization flag is invalid")
    if not isinstance(payload["neutralization"]["skipped_dates"], list) or not isinstance(
        payload["neutralization"]["details"], list
    ):
        raise RuntimeError("Hybrid artifact neutralization ledger is invalid")
    return payload


def _hybrid_artifact_long_excess(
    record: Mapping[str, Any],
    *,
    train_calendar: Mapping[str, Any],
) -> dict[str, Any] | None:
    dates = record.get("train_long_excess_dates")
    values = record.get("train_long_excess_values")
    node_count = record.get("node_count")
    if not isinstance(dates, list) or not isinstance(values, list):
        raise RuntimeError("Hybrid artifact Train long-excess must use lists")
    if len(dates) != len(values):
        raise RuntimeError("Hybrid artifact Train long-excess length mismatch")
    if not dates:
        if node_count in (1, 2):
            return None
        raise RuntimeError("N>2 Hybrid candidate lacks Train long-excess")
    if len(set(dates)) != len(dates) or any(
        not isinstance(date, str) or not date for date in dates
    ):
        raise RuntimeError("Hybrid artifact Train long-excess dates are invalid")
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
            raise RuntimeError("Hybrid artifact Train long-excess value is invalid")
    finite_periods = sum(value is not None for value in normalized_values)
    valid_periods = record.get("train_long_valid_periods")
    direction = record.get("train_direction")
    if finite_periods != valid_periods:
        raise RuntimeError(
            "Hybrid artifact Train long-excess finite/valid periods differ"
        )
    if direction not in (-1, 1):
        raise RuntimeError("Hybrid artifact Train long-excess direction is invalid")
    calendar_periods = train_calendar.get("periods")
    if calendar_periods != len(dates):
        raise RuntimeError("Hybrid artifact Train long-excess calendar size differs")
    if train_calendar.get("first_date") not in (None, dates[0]) or (
        train_calendar.get("last_date") not in (None, dates[-1])
    ):
        raise RuntimeError("Hybrid artifact Train long-excess calendar boundary differs")
    return {
        "dates": list(dates),
        "values": normalized_values,
        "direction": direction,
        "valid_periods": valid_periods,
        "finite_periods": finite_periods,
        "total_periods": len(dates),
        "origin": "stage5_hybrid_train_artifact_reuse",
    }


def _mapping_difference_paths(
    left: Any,
    right: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return deterministic field paths only; never copy contract values."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(
                    _mapping_difference_paths(left[key], right[key], prefix=path)
                )
        return paths
    return [] if left == right else [prefix or "$"]


def _validate_hybrid_artifact_records(
    artifact: Mapping[str, Any],
    *,
    artifact_contract_fingerprint: str,
    train_calendar: Mapping[str, Any],
) -> dict[str, tuple[int, Mapping[str, Any], dict[str, Any], dict[str, Any] | None]]:
    """Validate the complete frozen artifact before deciding whether to reuse it."""

    records = artifact.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Hybrid Train artifact records are missing")
    if artifact.get("candidate_count") != len(records):
        raise RuntimeError("Hybrid Train artifact candidate_count mismatch")
    records_by_hash: dict[
        str,
        tuple[int, Mapping[str, Any], dict[str, Any], dict[str, Any] | None],
    ] = {}
    ordered_hashes: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RuntimeError("Hybrid Train artifact record is invalid")
        if record.get("schema") != "factor_gfn.stage5_train_candidate_record.v1":
            raise RuntimeError("Hybrid Train artifact record schema mismatch")
        structural_hash = record.get("structural_hash")
        if not isinstance(structural_hash, str) or not structural_hash:
            raise RuntimeError("Hybrid Train artifact record structural_hash is invalid")
        if structural_hash in records_by_hash:
            raise RuntimeError("Hybrid Train artifact contains duplicate candidates")
        if record.get("train_evaluation_contract_fingerprint") != (
            artifact_contract_fingerprint
        ):
            raise RuntimeError("Hybrid record/Train contract fingerprint mismatch")
        metrics = _hybrid_artifact_train_metrics(record)
        long_excess = _hybrid_artifact_long_excess(
            record,
            train_calendar=train_calendar,
        )
        records_by_hash[structural_hash] = (index, record, metrics, long_excess)
        ordered_hashes.append(structural_hash)
    if ordered_hashes != sorted(ordered_hashes):
        raise RuntimeError("Hybrid Train artifact structural_hash order is invalid")
    return records_by_hash


def run_stage6_hybrid_train_reuse_overlay(
    *,
    source_set_manifest_path: Path,
    candidate_import_manifest_path: Path,
    compatibility_manifest_path: Path,
    evaluator: Stage6CandidateEvaluator,
    target_provider_manifest: Mapping[str, Any],
    target_provider_fingerprint: str,
    output_root: Path,
) -> Path:
    """Build an exact-contract v2 overlay from one completed Hybrid source."""

    source_set_path = Path(source_set_manifest_path).resolve()
    candidate_path = Path(candidate_import_manifest_path).resolve()
    compatibility_path = Path(compatibility_manifest_path).resolve()
    source_set, snapshots = _verify_source_set(source_set_path)
    if len(snapshots) != 1 or snapshots[0].get("source_type") != (
        "hybrid_train_artifact"
    ):
        raise RuntimeError("Hybrid Train reuse requires exactly one Hybrid source")
    snapshot = snapshots[0]
    if snapshot.get("snapshot_kind") != "completed_hybrid_train_artifact":
        raise RuntimeError("Hybrid Train reuse requires a completed-run snapshot")

    candidate_manifest, compatibility_manifest, accepted = _verify_candidate_inputs(
        candidate_path, compatibility_path
    )
    if candidate_manifest.get("downstream_eligible") is not True:
        raise RuntimeError(
            "Hybrid candidate import is not downstream eligible: "
            f"{candidate_manifest.get('downstream_block_reasons', [])}"
        )
    if compatibility_manifest.get("downstream_eligible") is not True:
        raise RuntimeError(
            "Hybrid compatibility audit is not downstream eligible: "
            f"{compatibility_manifest.get('downstream_block_reasons', [])}"
        )
    compatibility_counts = compatibility_manifest.get("counts", {})
    if not isinstance(compatibility_counts, Mapping) or int(
        compatibility_counts.get("AUTO_REJECT", -1)
    ) != 0:
        raise RuntimeError(
            "Hybrid source contains structurally incompatible candidates"
        )
    source_set_fingerprint = source_set.get("source_set_fingerprint")
    if source_set_fingerprint != candidate_manifest.get("source_set_fingerprint"):
        raise RuntimeError("source-set/candidate registry fingerprint mismatch")
    if source_set_fingerprint != compatibility_manifest.get(
        "source_set_fingerprint"
    ):
        raise RuntimeError("source-set/compatibility fingerprint mismatch")
    accepted_fingerprint = compatibility_manifest.get("digests", {}).get(
        "accepted_registry"
    )
    if evaluator.accepted_registry_fingerprint != accepted_fingerprint:
        raise RuntimeError("evaluator accepted registry fingerprint mismatch")
    audit_fingerprint = compatibility_manifest.get("audit_fingerprint")
    if not isinstance(audit_fingerprint, str):
        audit_fingerprint = compatibility_path.parent.name
    if (
        evaluator.compatibility_audit_fingerprint != audit_fingerprint
        or compatibility_path.parent.name != audit_fingerprint
    ):
        raise RuntimeError("evaluator compatibility audit fingerprint mismatch")
    if _stable_hash(target_provider_manifest) != target_provider_fingerprint:
        raise RuntimeError("target Provider manifest fingerprint mismatch")

    directory = Path(snapshot["_directory"])
    artifact_path = directory / "train_candidate_artifact.json"
    artifact = _read_json(artifact_path)
    if artifact.get("schema") != "factor_gfn.stage5_train_candidate_artifact.v1":
        raise RuntimeError("Hybrid Train artifact schema mismatch")
    artifact_contract = artifact.get("train_evaluation_contract")
    artifact_contract_fingerprint = artifact.get(
        "train_evaluation_contract_fingerprint"
    )
    if not isinstance(artifact_contract, Mapping) or _stable_hash(
        artifact_contract
    ) != artifact_contract_fingerprint:
        raise RuntimeError("Hybrid Train artifact contract fingerprint mismatch")
    semantics = snapshot.get("source_semantics", {})
    if not isinstance(semantics, Mapping) or (
        semantics.get("train_evaluation_contract_fingerprint")
        != artifact_contract_fingerprint
        or semantics.get("provider_fingerprint")
        != artifact_contract.get("provider_fingerprint")
    ):
        raise RuntimeError("Hybrid snapshot/Train contract provenance mismatch")

    artifact_train_scope = artifact_contract.get("train_scope_projection")
    artifact_train_calendar = (
        artifact_train_scope.get("calendar")
        if isinstance(artifact_train_scope, Mapping)
        else None
    )
    if not isinstance(artifact_train_calendar, Mapping):
        raise RuntimeError("Hybrid Train artifact contract lacks calendar")
    records_by_hash = _validate_hybrid_artifact_records(
        artifact,
        artifact_contract_fingerprint=artifact_contract_fingerprint,
        train_calendar=artifact_train_calendar,
    )

    current_contract, current_contract_fingerprint = _current_hybrid_train_contract(
        target_provider_manifest, target_provider_fingerprint
    )
    exact_contract = artifact_contract == current_contract and (
        artifact_contract_fingerprint == current_contract_fingerprint
    )
    verification_mode = (
        HYBRID_EXACT_CONTRACT_VERIFICATION
        if exact_contract
        else HYBRID_FULL_FRESH_TRAIN_FALLBACK
    )

    source_id = str(snapshot["source_id"])
    overlay_rows: list[dict[str, Any]] = []
    for candidate in sorted(
        accepted, key=lambda value: str(value["current_structural_hash"])
    ):
        structural_hash = str(candidate["current_structural_hash"])
        indexed = records_by_hash.get(structural_hash)
        if indexed is None:
            raise RuntimeError("accepted Hybrid candidate is absent from Train artifact")
        index, record, metrics, long_excess = indexed
        if set(candidate.get("source_ids", [])) != {source_id}:
            raise RuntimeError("accepted Hybrid candidate left the isolated source")
        for field, accepted_field in (
            ("structural_hash", "current_structural_hash"),
            ("formula", "formula"),
            ("prefix_token_ids", "prefix_token_ids"),
            ("node_count", "node_count"),
            ("depth", "depth"),
        ):
            if record.get(field) != candidate.get(accepted_field):
                raise RuntimeError(
                    f"Hybrid artifact/accepted candidate identity mismatch: {field}"
                )
        if not exact_contract:
            continue
        source_record_fingerprint = _stable_hash(record)
        origin = {
            "verification_mode": HYBRID_EXACT_CONTRACT_VERIFICATION,
            "source_id": source_id,
            "snapshot_fingerprint": snapshot["snapshot_fingerprint"],
            "locator": {
                "artifact": "train_candidate_artifact.json",
                "record_index": index,
            },
            "source_record_fingerprint": source_record_fingerprint,
        }
        payload = {
            "schema": HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA,
            "structural_hash": structural_hash,
            "node_count": candidate["node_count"],
            "train_metric_origin": "stage5_verified_reuse",
            "verification_mode": HYBRID_EXACT_CONTRACT_VERIFICATION,
            "train_evaluation_contract_fingerprint": (
                artifact_contract_fingerprint
            ),
            "train_metrics": metrics,
            "train_long_excess": long_excess,
            "origins": [origin],
        }
        overlay_rows.append({**payload, "record_fingerprint": _stable_hash(payload)})

    artifact_meta = next(
        (
            item
            for item in snapshot.get("artifacts", [])
            if item.get("name") == "train_candidate_artifact.json"
        ),
        None,
    )
    if not isinstance(artifact_meta, Mapping):
        raise RuntimeError("Hybrid snapshot lacks Train artifact identity")
    verification = {
        "verification_mode": verification_mode,
        "numeric_verification": (
            "not_required_by_approved_hybrid_contract"
            if exact_contract
            else "forbidden_because_train_contract_mismatch"
        ),
        "source_id": source_id,
        "source_snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        "source_artifact_sha256": artifact_meta["sha256"],
        "artifact_train_contract_fingerprint": artifact_contract_fingerprint,
        "current_train_contract_fingerprint": current_contract_fingerprint,
        "provider_fingerprint": target_provider_fingerprint,
        "train_scope_projection_fingerprint": _stable_hash(
            current_contract["train_scope_projection"]
        ),
        "implementation_fingerprint": _stable_hash(
            current_contract["implementation"]
        ),
        "contract_difference_paths": _mapping_difference_paths(
            artifact_contract, current_contract
        ),
        "fresh_train_fallback_reason": (
            None if exact_contract else HYBRID_TRAIN_CONTRACT_MISMATCH_REASON
        ),
        "old_train_metrics_allowed": exact_contract,
        "result": (
            "TRAIN_METRICS_REUSABLE"
            if exact_contract
            else "FULL_FRESH_TRAIN_FALLBACK"
        ),
    }
    fingerprint_payload = {
        "schema": HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA,
        "adapter_version": HYBRID_TRAIN_REUSE_ADAPTER_VERSION,
        "verification_mode": verification_mode,
        "fresh_train_fallback_reason": (
            None if exact_contract else HYBRID_TRAIN_CONTRACT_MISMATCH_REASON
        ),
        "source_set_fingerprint": source_set_fingerprint,
        "source_snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        "candidate_registry_fingerprint": candidate_manifest[
            "registry_fingerprint"
        ],
        "compatibility_audit_fingerprint": audit_fingerprint,
        "accepted_registry_fingerprint": accepted_fingerprint,
        "stage6_context_fingerprint": evaluator.context.fingerprint,
        "stage6_evaluation_contract_fingerprint": (
            evaluator.evaluation_contract_fingerprint
        ),
        "target_provider_fingerprint": target_provider_fingerprint,
        "artifact_train_contract_fingerprint": artifact_contract_fingerprint,
        "current_train_contract_fingerprint": current_contract_fingerprint,
        "train_contract_verification_digest": _stable_hash(verification),
        "overlay_digest": _stable_hash(overlay_rows),
    }
    fingerprint = _stable_hash(fingerprint_payload)
    root = Path(output_root).resolve()
    target = root / fingerprint
    manifest_path = target / "train_reuse_manifest.json"
    if target.exists():
        existing = _read_json(manifest_path)
        if existing.get("train_reuse_overlay_fingerprint") != fingerprint:
            raise RuntimeError("existing Hybrid Train reuse overlay conflict")
        return manifest_path

    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{fingerprint}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        _write_jsonl(temporary / "train_reuse_overlay.jsonl", overlay_rows)
        _write_json(
            temporary / "train_reuse_contract_verification.json", verification
        )
        manifest = {
            **fingerprint_payload,
            "train_reuse_overlay_fingerprint": fingerprint,
            "counts": {
                "accepted_candidates": len(accepted),
                "overlay_candidates": len(overlay_rows),
                "fresh_train_fallback_candidates": (
                    0 if exact_contract else len(accepted)
                ),
                "numeric_verification_candidates": 0,
            },
            "coverage_ratio": (
                (1.0 if accepted else 0.0) if exact_contract else 0.0
            ),
            "created_at_utc": _utc_now(),
            "created_at_excluded_from_fingerprint": True,
            "artifacts": {
                name: {
                    "size_bytes": (temporary / name).stat().st_size,
                    "sha256": _sha256_file(temporary / name),
                }
                for name in (
                    "train_reuse_overlay.jsonl",
                    "train_reuse_contract_verification.json",
                )
            },
        }
        _write_json(temporary / "train_reuse_manifest.json", manifest)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest_path


def run_stage6_train_reuse_audit(
    *,
    source_set_manifest_path: Path,
    candidate_import_manifest_path: Path,
    compatibility_manifest_path: Path,
    evaluator: Stage6CandidateEvaluator,
    target_provider_manifest: Mapping[str, Any],
    target_provider_fingerprint: str,
    output_root: Path,
) -> Path:
    """Execute 9A and materialize an immutable audit/overlay directory."""

    source_set_path = Path(source_set_manifest_path).resolve()
    candidate_path = Path(candidate_import_manifest_path).resolve()
    compatibility_path = Path(compatibility_manifest_path).resolve()
    source_set, snapshots = _verify_source_set(source_set_path)
    candidate_manifest, compatibility_manifest, accepted = _verify_candidate_inputs(
        candidate_path, compatibility_path
    )
    if source_set.get("source_set_fingerprint") != candidate_manifest.get(
        "source_set_fingerprint"
    ):
        raise RuntimeError("source-set/candidate registry fingerprint mismatch")
    if source_set.get("source_set_fingerprint") != compatibility_manifest.get(
        "source_set_fingerprint"
    ):
        raise RuntimeError("source-set/compatibility fingerprint mismatch")
    if evaluator.accepted_registry_fingerprint != compatibility_manifest.get(
        "digests", {}
    ).get("accepted_registry"):
        raise RuntimeError("evaluator accepted registry fingerprint mismatch")
    if evaluator.compatibility_audit_fingerprint != compatibility_path.parent.name:
        raise RuntimeError("evaluator compatibility audit fingerprint mismatch")

    accepted_by_hash = {
        str(candidate["current_structural_hash"]): candidate for candidate in accepted
    }
    accepted_hashes = set(accepted_by_hash)
    records_by_source_hash: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_field_counts: dict[str, dict[str, int]] = {}
    for snapshot in snapshots:
        source_id = str(snapshot["source_id"])
        records = _read_source_metric_records(snapshot, accepted_hashes)
        complete = sum(record["metrics"] is not None for record in records)
        source_field_counts[source_id] = {
            "accepted_records_found": len(records),
            "complete_reusable_train_records": complete,
            "incomplete_reusable_train_records": len(records) - complete,
        }
        for record in records:
            records_by_source_hash[(source_id, str(record["structural_hash"]))].append(record)

    static_audits = _static_audit_batches(
        snapshots,
        target_provider_manifest=target_provider_manifest,
        target_provider_fingerprint=target_provider_fingerprint,
    )
    batch_members = {
        batch["batch_id"]: batch for batch in _batch_sources(snapshots)
    }
    verification_rows: list[dict[str, Any]] = []
    passed_batches: set[str] = set()
    reusable_by_batch_hash: dict[tuple[str, str], dict[str, Any]] = {}

    for audit in static_audits:
        batch_id = str(audit["batch_id"])
        batch = batch_members[batch_id]
        candidate_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source_id in batch["source_ids"]:
            for structural_hash, candidate in accepted_by_hash.items():
                if source_id not in candidate.get("source_ids", []):
                    continue
                candidate_records[structural_hash].extend(
                    records_by_source_hash.get((source_id, structural_hash), [])
                )
        eligible: list[Mapping[str, Any]] = []
        conflicts: dict[str, list[str]] = {}
        for structural_hash, records in candidate_records.items():
            canonical, reasons = _canonical_metric_record(records)
            if canonical is None:
                conflicts[structural_hash] = reasons
                continue
            reusable_by_batch_hash[(batch_id, structural_hash)] = canonical
            eligible.append(accepted_by_hash[structural_hash])
        audit["field_coverage"] = {
            "source_counts": {
                source_id: source_field_counts[source_id]
                for source_id in batch["source_ids"]
            },
            "accepted_candidates_with_any_batch_origin": len(candidate_records),
            "complete_reusable_candidates": len(eligible),
            "incomplete_or_conflicting_candidates": len(conflicts),
            "incomplete_or_conflicting_reason_counts": {
                reason: sum(reason in reasons for reasons in conflicts.values())
                for reason in sorted({item for reasons in conflicts.values() for item in reasons})
            },
        }
        if audit["status"] != TRAIN_REUSE_NUMERIC_VERIFICATION_REQUIRED:
            continue
        sample = select_representative_candidates(eligible)
        audit["sample_selection"] = {
            "method": "metric_blind_weighted_structural_tag_coverage_then_structural_hash_v1",
            "requested": _SAMPLE_SIZE,
            "selected": len(sample),
            "structural_hashes": [candidate["current_structural_hash"] for candidate in sample],
            "digest": _stable_hash(
                [candidate["current_structural_hash"] for candidate in sample]
            ),
        }
        if not sample:
            audit["status"] = TRAIN_REUSE_NOT_ALLOWED
            audit["reason_codes"] = sorted(
                set(audit["reason_codes"] + ["no_complete_candidates_for_numeric_verification"])
            )
            continue
        batch_pass = len(sample) == min(_SAMPLE_SIZE, len(eligible))
        for candidate in sample:
            structural_hash = str(candidate["current_structural_hash"])
            old_record = reusable_by_batch_hash[(batch_id, structural_hash)]
            started = datetime.now(UTC)
            try:
                result = evaluator.evaluate(candidate)
                fresh = _metrics_from_stage6(result)
                old_flat = _flatten_metrics(old_record["metrics"])
                fresh_flat = _flatten_metrics(fresh)
                comparisons: dict[str, Any] = {}
                passed = set(old_flat) == set(fresh_flat)
                for field in sorted(set(old_flat) | set(fresh_flat)):
                    old_value = old_flat.get(field)
                    fresh_value = fresh_flat.get(field)
                    field_pass = field in old_flat and field in fresh_flat and _numeric_equal(
                        old_value, fresh_value
                    )
                    comparisons[field] = {
                        "stage5": old_value,
                        "stage6": fresh_value,
                        "absolute_error": (
                            abs(float(old_value) - float(fresh_value))
                            if isinstance(old_value, (int, float))
                            and isinstance(fresh_value, (int, float))
                            and not isinstance(old_value, bool)
                            and not isinstance(fresh_value, bool)
                            else None
                        ),
                        "pass": field_pass,
                    }
                    passed &= field_pass
                error = None
                result_fingerprint = result.result_fingerprint
                elapsed = result.total_seconds
            except Exception as exc:  # fail-closed ledger, caller still gets all batches
                passed = False
                comparisons = {}
                error = f"{type(exc).__name__}: {exc}"
                result_fingerprint = None
                elapsed = (datetime.now(UTC) - started).total_seconds()
            batch_pass &= passed
            verification_rows.append(
                {
                    "schema": TRAIN_REUSE_VERIFICATION_SCHEMA,
                    "batch_id": batch_id,
                    "structural_hash": structural_hash,
                    "node_count": candidate["node_count"],
                    "depth": candidate["depth"],
                    "structural_tags": sorted(_candidate_tags(candidate)),
                    "source_id": old_record["source_id"],
                    "snapshot_fingerprint": old_record["snapshot_fingerprint"],
                    "source_locator": old_record["locator"],
                    "comparison": comparisons,
                    "pass": passed,
                    "error": error,
                    "stage6_result_fingerprint": result_fingerprint,
                    "elapsed_seconds_observed": elapsed,
                    "elapsed_seconds_excluded_from_fingerprint": True,
                }
            )
        if batch_pass:
            audit["status"] = TRAIN_METRICS_REUSABLE
            passed_batches.add(batch_id)
        else:
            audit["status"] = TRAIN_REUSE_NOT_ALLOWED
            audit["reason_codes"] = sorted(
                set(audit["reason_codes"] + ["numeric_verification_failed"])
            )

    overlay_rows: list[dict[str, Any]] = []
    for structural_hash in sorted(accepted_by_hash):
        candidates: list[tuple[str, dict[str, Any]]] = []
        for batch_id in sorted(passed_batches):
            record = reusable_by_batch_hash.get((batch_id, structural_hash))
            if record is not None:
                candidates.append((batch_id, record))
        if not candidates:
            continue
        first_batch, first = candidates[0]
        first_flat = _flatten_metrics(first["metrics"])
        if any(
            set(_flatten_metrics(record["metrics"])) != set(first_flat)
            or any(
                not _numeric_equal(first_flat[key], _flatten_metrics(record["metrics"])[key])
                for key in first_flat
            )
            for _, record in candidates[1:]
        ):
            continue
        origins = [
            {
                "batch_id": batch_id,
                "source_id": record["source_id"],
                "snapshot_fingerprint": record["snapshot_fingerprint"],
                "locator": record["locator"],
            }
            for batch_id, record in candidates
        ]
        payload = {
            "schema": TRAIN_REUSE_OVERLAY_SCHEMA,
            "structural_hash": structural_hash,
            "train_metric_origin": "stage5_verified_reuse",
            "train_metrics": first["metrics"],
            "origins": origins,
        }
        overlay_rows.append({**payload, "record_fingerprint": _stable_hash(payload)})

    static_audits.sort(key=lambda value: str(value["batch_id"]))
    verification_rows.sort(
        key=lambda value: (str(value["batch_id"]), str(value["structural_hash"]))
    )
    overlay_rows.sort(key=lambda value: str(value["structural_hash"]))
    fingerprint_payload = {
        "schema": TRAIN_REUSE_MANIFEST_SCHEMA,
        "auditor_version": TRAIN_REUSE_AUDITOR_VERSION,
        "source_set_fingerprint": source_set["source_set_fingerprint"],
        "candidate_registry_fingerprint": candidate_manifest["registry_fingerprint"],
        "compatibility_audit_fingerprint": compatibility_path.parent.name,
        "accepted_registry_fingerprint": compatibility_manifest["digests"][
            "accepted_registry"
        ],
        "stage6_context_fingerprint": evaluator.context.fingerprint,
        "stage6_evaluation_contract_fingerprint": evaluator.evaluation_contract_fingerprint,
        "target_provider_fingerprint": target_provider_fingerprint,
        "target_train_contract_projection_fingerprint": _stable_hash(
            _provider_projection(target_provider_manifest)
        ),
        "source_audit_digest": _stable_hash(static_audits),
        "numeric_verification_digest": _stable_hash(
            [
                {key: value for key, value in row.items() if not key.startswith("elapsed_seconds")}
                for row in verification_rows
            ]
        ),
        "overlay_digest": _stable_hash(overlay_rows),
    }
    fingerprint = _stable_hash(fingerprint_payload)
    root = Path(output_root).resolve()
    target = root / fingerprint
    manifest_path = target / "train_reuse_manifest.json"
    if target.exists():
        existing = _read_json(manifest_path)
        if existing.get("train_reuse_overlay_fingerprint") != fingerprint:
            raise RuntimeError("existing Train reuse overlay fingerprint conflict")
        return manifest_path
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{fingerprint}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        _write_jsonl(temporary / "train_reuse_source_audit.jsonl", static_audits)
        _write_jsonl(
            temporary / "train_reuse_numeric_verification.jsonl", verification_rows
        )
        _write_jsonl(temporary / "train_reuse_overlay.jsonl", overlay_rows)
        counts = {
            "source_batches": len(static_audits),
            "static_rejected_batches": sum(
                row["status"] == TRAIN_REUSE_NOT_ALLOWED
                and "numeric_verification_failed" not in row["reason_codes"]
                for row in static_audits
            ),
            "numeric_verified_batches": len(passed_batches),
            "numeric_rejected_batches": sum(
                "numeric_verification_failed" in row["reason_codes"]
                for row in static_audits
            ),
            "numeric_verification_candidates": len(verification_rows),
            "overlay_candidates": len(overlay_rows),
            "accepted_candidates": len(accepted),
        }
        manifest = {
            **fingerprint_payload,
            "train_reuse_overlay_fingerprint": fingerprint,
            "counts": counts,
            "coverage_ratio": len(overlay_rows) / len(accepted) if accepted else 0.0,
            "batch_results": [
                {
                    "batch_id": row["batch_id"],
                    "source_ids": row["source_ids"],
                    "status": row["status"],
                    "reason_codes": row["reason_codes"],
                    "field_coverage": row.get("field_coverage"),
                    "sample_selection": row.get("sample_selection"),
                }
                for row in static_audits
            ],
            "numeric_tolerance": {"absolute": _NUMERIC_ATOL, "relative": _NUMERIC_RTOL},
            "created_at_utc": _utc_now(),
            "created_at_excluded_from_fingerprint": True,
            "artifacts": {
                name: {
                    "size_bytes": (temporary / name).stat().st_size,
                    "sha256": _sha256_file(temporary / name),
                }
                for name in (
                    "train_reuse_source_audit.jsonl",
                    "train_reuse_numeric_verification.jsonl",
                    "train_reuse_overlay.jsonl",
                )
            },
        }
        _write_json(temporary / "train_reuse_manifest.json", manifest)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest_path


def run_stage6_v6_equivalence_verification(
    *,
    source_set_manifest_path: Path,
    candidate_import_manifest_path: Path,
    compatibility_manifest_path: Path,
    current_overlay_manifest_path: Path,
    evaluator: Stage6CandidateEvaluator,
    target_provider_manifest: Mapping[str, Any],
    target_provider_fingerprint: str,
    output_root: Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Path:
    """Collect bounded evidence for the one frozen v6 source batch.

    This is deliberately not an approval or overlay-building entry point.  A
    passing result means only that the frozen static comparison and the 24-row
    representative numeric check passed under this auditor version.
    """

    source_set_path = Path(source_set_manifest_path).resolve()
    candidate_path = Path(candidate_import_manifest_path).resolve()
    compatibility_path = Path(compatibility_manifest_path).resolve()
    overlay_manifest, current_overlay_hashes = _verify_existing_overlay(
        current_overlay_manifest_path
    )
    source_set, snapshots = _verify_source_set(source_set_path)
    candidate_manifest, compatibility_manifest, accepted = _verify_candidate_inputs(
        candidate_path, compatibility_path
    )
    if source_set.get("source_set_fingerprint") != candidate_manifest.get(
        "source_set_fingerprint"
    ):
        raise RuntimeError("source-set/candidate registry fingerprint mismatch")
    if source_set.get("source_set_fingerprint") != compatibility_manifest.get(
        "source_set_fingerprint"
    ):
        raise RuntimeError("source-set/compatibility fingerprint mismatch")
    if evaluator.accepted_registry_fingerprint != compatibility_manifest.get(
        "digests", {}
    ).get("accepted_registry"):
        raise RuntimeError("evaluator accepted registry fingerprint mismatch")
    if evaluator.compatibility_audit_fingerprint != compatibility_path.parent.name:
        raise RuntimeError("evaluator compatibility audit fingerprint mismatch")
    if overlay_manifest.get("compatibility_audit_fingerprint") != compatibility_path.parent.name:
        raise RuntimeError("existing overlay compatibility fingerprint mismatch")
    if overlay_manifest.get("accepted_registry_fingerprint") != compatibility_manifest.get(
        "digests", {}
    ).get("accepted_registry"):
        raise RuntimeError("existing overlay accepted registry fingerprint mismatch")
    if overlay_manifest.get("target_provider_fingerprint") != target_provider_fingerprint:
        raise RuntimeError("existing overlay target provider fingerprint mismatch")

    batches = {
        str(batch["batch_id"]): batch for batch in _batch_sources(snapshots)
    }
    batch = batches.get(V6_EQUIVALENCE_TARGET_BATCH_ID)
    if batch is None:
        raise RuntimeError("frozen v6 equivalence batch is absent from source set")
    catalog, catalog_conflicts = _provider_manifest_catalog(snapshots)
    provider_fingerprint = str(batch["provider_fingerprint"])
    if provider_fingerprint in catalog_conflicts:
        raise RuntimeError("frozen v6 batch has conflicting provider manifests")
    source_provider = catalog.get(provider_fingerprint)
    if source_provider is None:
        raise RuntimeError("frozen v6 batch provider manifest is unavailable")
    if source_provider.get("schema") != "factor_gfn.real_reward_provider.v6":
        raise RuntimeError("frozen equivalence batch is not provider v6")

    source_projection = _provider_projection(source_provider)
    target_projection = _provider_projection(target_provider_manifest)
    projection_evidence = _projection_difference_evidence(
        source_projection, target_projection
    )
    static_semantics_pass = bool(
        projection_evidence["all_differences_are_declared_implementation_fields"]
    )

    accepted_by_hash = {
        str(candidate["current_structural_hash"]): candidate for candidate in accepted
    }
    accepted_hashes = set(accepted_by_hash)
    batch_snapshots = [
        snapshot
        for snapshot in snapshots
        if str(snapshot["source_id"]) in set(batch["source_ids"])
    ]
    source_field_counts: dict[str, dict[str, int]] = {}
    candidate_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for snapshot in batch_snapshots:
        source_id = str(snapshot["source_id"])
        records = _read_source_metric_records(snapshot, accepted_hashes)
        complete = sum(record["metrics"] is not None for record in records)
        source_field_counts[source_id] = {
            "accepted_records_found": len(records),
            "complete_reusable_train_records": complete,
            "incomplete_reusable_train_records": len(records) - complete,
        }
        for record in records:
            candidate_records[str(record["structural_hash"])].append(record)

    eligible: list[Mapping[str, Any]] = []
    canonical_by_hash: dict[str, dict[str, Any]] = {}
    conflicts: dict[str, list[str]] = {}
    for structural_hash, records in candidate_records.items():
        canonical, reasons = _canonical_metric_record(records)
        if canonical is None:
            conflicts[structural_hash] = reasons
            continue
        canonical_by_hash[structural_hash] = canonical
        eligible.append(accepted_by_hash[structural_hash])
    verification_population = [
        candidate
        for candidate in eligible
        if str(candidate["current_structural_hash"]) not in current_overlay_hashes
    ]
    sample = select_representative_candidates(verification_population)
    if len(sample) != min(_SAMPLE_SIZE, len(verification_population)):
        raise RuntimeError("v6 representative sample is incomplete")
    if not sample:
        raise RuntimeError("v6 batch has no complete candidates for verification")

    sample_hashes = [str(candidate["current_structural_hash"]) for candidate in sample]
    if progress_callback is not None:
        progress_callback(
            {
                "event_type": "v6_equivalence_started",
                "batch_id": V6_EQUIVALENCE_TARGET_BATCH_ID,
                "eligible_candidates": len(eligible),
                "verification_population": len(verification_population),
                "sample_size": len(sample),
                "completed": 0,
            }
        )

    verification_rows: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(sample, start=1):
        structural_hash = str(candidate["current_structural_hash"])
        old_record = canonical_by_hash[structural_hash]
        started = datetime.now(UTC)
        try:
            result = evaluator.evaluate(candidate)
            fresh_metrics = _metrics_from_stage6(result)
            passed, comparisons, details_comparison = _compare_v6_equivalence_metrics(
                old_record["metrics"], fresh_metrics
            )
            error = None
            result_fingerprint = result.result_fingerprint
            elapsed = result.total_seconds
        except Exception as exc:  # preserve a complete fail-closed evidence ledger
            passed = False
            comparisons = {}
            details_comparison = {
                "comparison": "ordered_semantic_records_with_row_index_mapped_to_global_row",
                "stage5_v6": old_record["metrics"].get("neutralization", {}).get(
                    "details"
                ),
                "current_stage6": None,
                "stage5_v6_normalized": None,
                "current_stage6_normalized": None,
                "raw_exact_pass": False,
                "stage5_v6_digest": None,
                "current_stage6_digest": None,
                "pass": False,
            }
            error = f"{type(exc).__name__}: {exc}"
            result_fingerprint = None
            elapsed = (datetime.now(UTC) - started).total_seconds()
        row = {
            "schema": V6_EQUIVALENCE_RECORD_SCHEMA,
            "batch_id": V6_EQUIVALENCE_TARGET_BATCH_ID,
            "sample_ordinal": ordinal,
            "structural_hash": structural_hash,
            "node_count": candidate["node_count"],
            "depth": candidate["depth"],
            "structural_tags": sorted(_candidate_tags(candidate)),
            "source_id": old_record["source_id"],
            "snapshot_fingerprint": old_record["snapshot_fingerprint"],
            "source_locator": old_record["locator"],
            "metric_comparison": comparisons,
            "neutralization_details_comparison": details_comparison,
            "pass": passed,
            "error": error,
            "stage6_result_fingerprint": result_fingerprint,
            "elapsed_seconds_observed": elapsed,
            "elapsed_seconds_excluded_from_fingerprint": True,
        }
        verification_rows.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    "event_type": "v6_equivalence_candidate_completed",
                    "batch_id": V6_EQUIVALENCE_TARGET_BATCH_ID,
                    "eligible_candidates": len(eligible),
                    "sample_size": len(sample),
                    "completed": ordinal,
                    "structural_hash": structural_hash,
                    "pass": passed,
                    "elapsed_seconds_observed": elapsed,
                }
            )

    numeric_pass = all(row["pass"] for row in verification_rows)
    conclusion = (
        V6_EQUIVALENCE_EVIDENCE_PASSED
        if static_semantics_pass and numeric_pass
        else V6_EQUIVALENCE_EVIDENCE_FAILED
    )
    stable_rows = [
        {
            key: value
            for key, value in row.items()
            if not key.startswith("elapsed_seconds")
        }
        for row in verification_rows
    ]
    fingerprint_payload = {
        "schema": V6_EQUIVALENCE_MANIFEST_SCHEMA,
        "auditor_version": V6_EQUIVALENCE_AUDITOR_VERSION,
        "batch_id": V6_EQUIVALENCE_TARGET_BATCH_ID,
        "source_ids": list(batch["source_ids"]),
        "source_set_fingerprint": source_set["source_set_fingerprint"],
        "candidate_registry_fingerprint": candidate_manifest["registry_fingerprint"],
        "compatibility_audit_fingerprint": compatibility_path.parent.name,
        "accepted_registry_fingerprint": compatibility_manifest["digests"][
            "accepted_registry"
        ],
        "current_overlay_fingerprint": overlay_manifest[
            "train_reuse_overlay_fingerprint"
        ],
        "stage6_context_fingerprint": evaluator.context.fingerprint,
        "stage6_evaluation_contract_fingerprint": (
            evaluator.evaluation_contract_fingerprint
        ),
        "stage6_evidence_origin": getattr(
            evaluator, "evidence_origin", {"mode": "direct_evaluator"}
        ),
        "source_provider_fingerprint": provider_fingerprint,
        "source_provider_schema": source_provider["schema"],
        "target_provider_fingerprint": target_provider_fingerprint,
        "source_train_contract_projection_fingerprint": _stable_hash(
            source_projection
        ),
        "target_train_contract_projection_fingerprint": _stable_hash(
            target_projection
        ),
        "projection_difference_evidence": projection_evidence,
        "sample_selection": {
            "method": "metric_blind_weighted_structural_tag_coverage_then_structural_hash_v1",
            "population": "complete_v6_candidates_not_already_in_current_overlay",
            "requested": _SAMPLE_SIZE,
            "selected": len(sample),
            "structural_hashes": sample_hashes,
            "digest": _stable_hash(sample_hashes),
        },
        "numeric_tolerance": {"absolute": _NUMERIC_ATOL, "relative": _NUMERIC_RTOL},
        "verification_digest": _stable_hash(stable_rows),
        "conclusion": conclusion,
        "reuse_authorized": False,
        "overlay_written": False,
    }
    fingerprint = _stable_hash(fingerprint_payload)
    root = Path(output_root).resolve()
    target = root / fingerprint
    manifest_path = target / "v6_equivalence_manifest.json"
    if target.exists():
        existing = _read_json(manifest_path)
        if existing.get("v6_equivalence_fingerprint") != fingerprint:
            raise RuntimeError("existing v6 equivalence evidence fingerprint conflict")
        return manifest_path
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{fingerprint}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        artifact_name = "v6_equivalence_verification.jsonl"
        _write_jsonl(temporary / artifact_name, verification_rows)
        manifest = {
            **fingerprint_payload,
            "v6_equivalence_fingerprint": fingerprint,
            "counts": {
                "accepted_candidates_with_any_batch_origin": len(candidate_records),
                "complete_reusable_candidates": len(eligible),
                "already_covered_by_current_overlay": len(eligible)
                - len(verification_population),
                "verification_population": len(verification_population),
                "incomplete_or_conflicting_candidates": len(conflicts),
                "sampled_candidates": len(sample),
                "passed_candidates": sum(row["pass"] for row in verification_rows),
                "failed_candidates": sum(not row["pass"] for row in verification_rows),
            },
            "source_field_counts": source_field_counts,
            "created_at_utc": _utc_now(),
            "created_at_excluded_from_fingerprint": True,
            "artifacts": {
                artifact_name: {
                    "size_bytes": (temporary / artifact_name).stat().st_size,
                    "sha256": _sha256_file(temporary / artifact_name),
                }
            },
        }
        _write_json(temporary / "v6_equivalence_manifest.json", manifest)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if progress_callback is not None:
        progress_callback(
            {
                "event_type": "v6_equivalence_completed",
                "batch_id": V6_EQUIVALENCE_TARGET_BATCH_ID,
                "eligible_candidates": len(eligible),
                "sample_size": len(sample),
                "completed": len(sample),
                "passed": sum(row["pass"] for row in verification_rows),
                "failed": sum(not row["pass"] for row in verification_rows),
                "conclusion": conclusion,
                "manifest_path": str(manifest_path),
            }
        )
    return manifest_path


def run_current_stage6_train_reuse_audit(
    *,
    source_set_manifest_path: Path,
    candidate_import_manifest_path: Path,
    compatibility_manifest_path: Path,
    output_root: Path,
    data_paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> Path:
    """Build current real contexts sequentially and execute the bounded 9A audit."""

    train_context = build_real_reward_data_context(
        RealRewardDataConfig(), data_paths
    )
    provider = RealRewardProvider(
        train_context,
        subexpression_cache_max_bytes=0,
    )
    provider_manifest = provider.manifest()
    provider_fingerprint = provider.fingerprint()
    del provider, train_context
    gc.collect()

    stage6_context = build_stage6_evaluation_context(
        Stage6EvaluationConfig(), data_paths
    )
    evaluator = Stage6CandidateEvaluator(
        stage6_context,
        compatibility_audit_fingerprint=Path(compatibility_manifest_path).resolve().parent.name,
        accepted_registry_fingerprint=_read_json(
            Path(compatibility_manifest_path).resolve()
        )["digests"]["accepted_registry"],
    )
    return run_stage6_train_reuse_audit(
        source_set_manifest_path=source_set_manifest_path,
        candidate_import_manifest_path=candidate_import_manifest_path,
        compatibility_manifest_path=compatibility_manifest_path,
        evaluator=evaluator,
        target_provider_manifest=provider_manifest,
        target_provider_fingerprint=provider_fingerprint,
        output_root=output_root,
    )


def run_current_stage6_v6_equivalence_verification(
    *,
    source_set_manifest_path: Path,
    candidate_import_manifest_path: Path,
    compatibility_manifest_path: Path,
    current_overlay_manifest_path: Path,
    output_root: Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    data_paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> Path:
    """Build current contexts and verify only the frozen provider-v6 batch."""

    train_context = build_real_reward_data_context(RealRewardDataConfig(), data_paths)
    provider = RealRewardProvider(
        train_context,
        subexpression_cache_max_bytes=0,
    )
    provider_manifest = provider.manifest()
    provider_fingerprint = provider.fingerprint()
    del provider, train_context
    gc.collect()

    stage6_context = build_stage6_evaluation_context(
        Stage6EvaluationConfig(), data_paths
    )
    evaluator = Stage6CandidateEvaluator(
        stage6_context,
        compatibility_audit_fingerprint=Path(
            compatibility_manifest_path
        ).resolve().parent.name,
        accepted_registry_fingerprint=_read_json(
            Path(compatibility_manifest_path).resolve()
        )["digests"]["accepted_registry"],
    )
    return run_stage6_v6_equivalence_verification(
        source_set_manifest_path=source_set_manifest_path,
        candidate_import_manifest_path=candidate_import_manifest_path,
        compatibility_manifest_path=compatibility_manifest_path,
        current_overlay_manifest_path=current_overlay_manifest_path,
        evaluator=evaluator,
        target_provider_manifest=provider_manifest,
        target_provider_fingerprint=provider_fingerprint,
        output_root=output_root,
        progress_callback=progress_callback,
    )


__all__ = [
    "HYBRID_TRAIN_REUSE_ADAPTER_VERSION",
    "HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA",
    "HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA",
    "TRAIN_METRICS_REUSABLE",
    "TRAIN_REUSE_AUDITOR_VERSION",
    "TRAIN_REUSE_MANIFEST_SCHEMA",
    "TRAIN_REUSE_NOT_ALLOWED",
    "TRAIN_REUSE_NUMERIC_VERIFICATION_REQUIRED",
    "TRAIN_REUSE_OVERLAY_SCHEMA",
    "TRAIN_REUSE_SOURCE_AUDIT_SCHEMA",
    "TRAIN_REUSE_VERIFICATION_SCHEMA",
    "V6_EQUIVALENCE_AUDITOR_VERSION",
    "V6_EQUIVALENCE_EVIDENCE_FAILED",
    "V6_EQUIVALENCE_EVIDENCE_PASSED",
    "V6_EQUIVALENCE_MANIFEST_SCHEMA",
    "V6_EQUIVALENCE_RECORD_SCHEMA",
    "V6_EQUIVALENCE_TARGET_BATCH_ID",
    "run_current_stage6_v6_equivalence_verification",
    "run_current_stage6_train_reuse_audit",
    "run_stage6_hybrid_train_reuse_overlay",
    "run_stage6_v6_equivalence_verification",
    "run_stage6_train_reuse_audit",
    "select_representative_candidates",
]
