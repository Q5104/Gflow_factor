"""Immutable Baseline Factor Pool freeze and verification contracts.

This module is deliberately a D-side boundary.  It verifies the complete
formal Hybrid Stage 6 authority chain, copies the exact provisional pool, and
publishes a content-addressed artifact.  It neither selects factors nor grants
OOS access.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .candidate_import import (
    CANDIDATE_IMPORT_MANIFEST_SCHEMA,
    _verify_source_set,
)
from .expression_compatibility import (
    EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA,
    _verify_candidate_registry,
)
from .stage6_evaluation import _stable_hash
from .stage6_evaluation_store import EvaluationStore
from .stage6_full_pipeline import load_stage6_accepted_registry
from .stage6_mixed_evaluation import Stage6TrainReuseOverlay
from .stage6_selection import Stage6SelectionConfig
from .stage6_survivor_enrichment import (
    STAGE6_ENRICHED_SELECTION_MANIFEST_SCHEMA,
    STAGE6_ENRICHED_SELECTION_VERSION,
)
from .stage6_train_reuse import HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA
from .stage6_two_phase_pipeline import (
    STAGE6_TRAIN_ENTRY_SCHEMA,
    STAGE6_TRAIN_PASS_MANIFEST_SCHEMA,
    STAGE6_TRAIN_PREPARATION_SCOPE,
    STAGE6_VALIDATION_ENTRY_SCHEMA,
    STAGE6_VALIDATION_SCOPE,
    TRAIN_NOT_EVALUATED_VALIDATION,
)


BASELINE_FACTOR_RECORD_SCHEMA = "factor_gfn.baseline_factor_record.v1"
BASELINE_FACTOR_POOL_MANIFEST_SCHEMA = (
    "factor_gfn.baseline_factor_pool_manifest.v1"
)
BASELINE_FACTOR_POOL_VERSION = "baseline-factor-pool-freeze-v1"
BASELINE_FACTOR_POOL_FILENAME = "baseline_factor_pool.jsonl"
BASELINE_FACTOR_POOL_MANIFEST_FILENAME = "baseline_factor_pool_manifest.json"
OOS_UNTOUCHED = "not_loaded_not_evaluated"


class BaselineFactorPoolIntegrityError(RuntimeError):
    """An upstream authority or immutable frozen-pool artifact is invalid."""


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class BaselineFactorPoolFreezeInputs:
    source_set_manifest_path: Path
    candidate_import_manifest_path: Path
    compatibility_manifest_path: Path
    train_reuse_manifest_path: Path
    train_entry_manifest_path: Path
    train_pass_manifest_path: Path
    validation_entry_manifest_path: Path
    enriched_selection_manifest_path: Path

    def resolved(self) -> "BaselineFactorPoolFreezeInputs":
        return BaselineFactorPoolFreezeInputs(
            **{
                field: Path(getattr(self, field)).resolve()
                for field in self.__dataclass_fields__
            }
        )

    def manifest_paths(self) -> dict[str, str]:
        resolved = self.resolved()
        return {
            field.removesuffix("_path"): str(getattr(resolved, field))
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_manifest_paths(
        cls, paths: Mapping[str, Any]
    ) -> "BaselineFactorPoolFreezeInputs":
        values: dict[str, Path] = {}
        for field in cls.__dataclass_fields__:
            key = field.removesuffix("_path")
            value = paths.get(key)
            if not isinstance(value, str) or not value:
                raise BaselineFactorPoolIntegrityError(
                    f"frozen pool lacks upstream manifest path: {key}"
                )
            values[field] = Path(value)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class FrozenBaselineFactorRecord:
    provisional_rank: int
    stage6_sorted_rank: int
    structural_hash: str
    formula: str
    prefix_token_ids: tuple[int, ...]
    node_count: int
    depth: int
    train_direction: int
    train_metrics: Mapping[str, Any]
    validation_metrics: Mapping[str, Any]
    selection_status: Mapping[str, Any]
    result_identity: Mapping[str, Any]
    source_identity: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        source_identity = dict(self.source_identity)
        for key in ("source_ids", "origin_ids"):
            if isinstance(source_identity.get(key), tuple):
                source_identity[key] = list(source_identity[key])
        return {
            "schema": BASELINE_FACTOR_RECORD_SCHEMA,
            "provisional_rank": self.provisional_rank,
            "stage6_sorted_rank": self.stage6_sorted_rank,
            "structural_hash": self.structural_hash,
            "formula": self.formula,
            "prefix_token_ids": list(self.prefix_token_ids),
            "node_count": self.node_count,
            "depth": self.depth,
            "train_direction": self.train_direction,
            "train_metrics": dict(self.train_metrics),
            "validation_metrics": dict(self.validation_metrics),
            "selection_status": dict(self.selection_status),
            "result_identity": dict(self.result_identity),
            "source_identity": source_identity,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "FrozenBaselineFactorRecord":
        if row.get("schema") != BASELINE_FACTOR_RECORD_SCHEMA:
            raise BaselineFactorPoolIntegrityError("frozen factor record schema mismatch")
        try:
            prefix = tuple(int(value) for value in row["prefix_token_ids"])
            record = cls(
                provisional_rank=int(row["provisional_rank"]),
                stage6_sorted_rank=int(row["stage6_sorted_rank"]),
                structural_hash=str(row["structural_hash"]),
                formula=str(row["formula"]),
                prefix_token_ids=prefix,
                node_count=int(row["node_count"]),
                depth=int(row["depth"]),
                train_direction=int(row["train_direction"]),
                train_metrics=MappingProxyType(dict(row["train_metrics"])),
                validation_metrics=MappingProxyType(
                    dict(row["validation_metrics"])
                ),
                selection_status=MappingProxyType(
                    dict(row["selection_status"])
                ),
                result_identity=MappingProxyType(dict(row["result_identity"])),
                source_identity=MappingProxyType(
                    {
                        **dict(row["source_identity"]),
                        "source_ids": tuple(row["source_identity"].get("source_ids", [])),
                        "origin_ids": tuple(row["source_identity"].get("origin_ids", [])),
                    }
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BaselineFactorPoolIntegrityError(
                "frozen factor record is incomplete"
            ) from error
        if record.provisional_rank < 1 or record.stage6_sorted_rank < 1:
            raise BaselineFactorPoolIntegrityError("frozen factor rank is invalid")
        if record.node_count < 1 or record.depth < 0 or not record.prefix_token_ids:
            raise BaselineFactorPoolIntegrityError("frozen expression identity is invalid")
        if record.train_direction not in (-1, 1):
            raise BaselineFactorPoolIntegrityError("frozen Train direction is invalid")
        if record.selection_status != {
            "hard_filter_pass": True,
            "decorrelation_status": "retained",
        }:
            raise BaselineFactorPoolIntegrityError("frozen selection status is invalid")
        return record


@dataclass(frozen=True, slots=True)
class BaselineFactorPoolArtifact:
    manifest_path: Path
    records_path: Path
    baseline_factor_pool_fingerprint: str
    factor_count: int
    reused_existing_artifact: bool


@dataclass(frozen=True, slots=True)
class VerifiedFrozenBaselineFactorPool:
    manifest_path: Path
    records_path: Path
    baseline_factor_pool_fingerprint: str
    manifest: Mapping[str, Any]
    records: tuple[FrozenBaselineFactorRecord, ...]
    ordered_structural_hashes: tuple[str, ...]
    frozen_train_directions: tuple[int, ...]
    upstream_provenance: Mapping[str, Any]
    oos_status: str


@dataclass(frozen=True, slots=True)
class _VerifiedStage6Authority:
    provenance: Mapping[str, Any]
    frozen_records: tuple[FrozenBaselineFactorRecord, ...]
    enriched_selection_fingerprint: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BaselineFactorPoolIntegrityError(
            f"cannot read immutable JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise BaselineFactorPoolIntegrityError(f"JSON must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("row is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise BaselineFactorPoolIntegrityError(
            f"cannot read immutable JSONL: {path}"
        ) from error
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _require_artifact(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> Path:
    metadata = manifest.get("artifacts", {}).get(name)
    path = manifest_path.parent / name
    if not isinstance(metadata, Mapping) or not path.is_file():
        raise BaselineFactorPoolIntegrityError(f"upstream artifact is missing: {path}")
    if (
        path.stat().st_size != int(metadata.get("size_bytes", -1))
        or _sha256_file(path) != metadata.get("sha256")
    ):
        raise BaselineFactorPoolIntegrityError(f"upstream artifact changed: {path}")
    return path


def _verify_entry(
    path: Path, *, schema: str, scope: str
) -> dict[str, Any]:
    manifest = _read_json(path)
    stable = {
        key: value
        for key, value in manifest.items()
        if key != "entry_manifest_fingerprint"
    }
    if manifest.get("schema") != schema or manifest.get(
        "evaluation_run_scope"
    ) != scope:
        raise BaselineFactorPoolIntegrityError("Stage 6 entry schema or scope mismatch")
    if _stable_hash(stable) != manifest.get("entry_manifest_fingerprint"):
        raise BaselineFactorPoolIntegrityError("Stage 6 entry fingerprint mismatch")
    if manifest.get("oos") != OOS_UNTOUCHED:
        raise BaselineFactorPoolIntegrityError("Stage 6 entry OOS lock changed")
    return manifest


def _load_verified_run(entry: Mapping[str, Any]):
    try:
        with EvaluationStore(
            entry["database_path"], entry["run_artifact_root"], read_only=True
        ) as store:
            return store.load_verified_run_results(str(entry["evaluation_run_id"]))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise BaselineFactorPoolIntegrityError(
            "Stage 6 EvaluationStore verification failed"
        ) from error


def _metric_summary(result: Mapping[str, Any], split: str) -> dict[str, Any]:
    try:
        split_result = result[split]
        ic = split_result["ic"]
        long_result = split_result["long"]
        summary = {
            "ic_mean": ic["mean"],
            "ic_std": ic["std"],
            "icir": ic["icir"],
            "ic_valid_periods": int(ic["valid_periods"]),
            "ic_total_periods": int(ic["total_periods"]),
            "long_ir": long_result["annualized_ir"],
            "long_valid_periods": int(long_result["valid_periods"]),
            "long_total_periods": int(long_result["total_periods"]),
        }
        if split == "train":
            summary["barra_max_abs_correlation"] = split_result["barra"][
                "max_abs_correlation"
            ]
        # Canonical serialization is also the finite/NaN fail-closed check.
        _stable_hash(summary)
        return summary
    except (KeyError, TypeError, ValueError) as error:
        raise BaselineFactorPoolIntegrityError(
            f"{split} result lacks the frozen metric contract"
        ) from error


def _verify_train_pass(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _read_json(path)
    stable = {
        key: value
        for key, value in manifest.items()
        if key != "train_pass_manifest_fingerprint"
    }
    if manifest.get("schema") != STAGE6_TRAIN_PASS_MANIFEST_SCHEMA:
        raise BaselineFactorPoolIntegrityError("Train-pass manifest schema mismatch")
    if _stable_hash(stable) != manifest.get("train_pass_manifest_fingerprint"):
        raise BaselineFactorPoolIntegrityError("Train-pass manifest fingerprint mismatch")
    if manifest.get("oos") != OOS_UNTOUCHED:
        raise BaselineFactorPoolIntegrityError("Train-pass OOS lock changed")
    decisions_path = _require_artifact(
        path, manifest, "train_prefilter_results.jsonl"
    )
    candidates_path = _require_artifact(
        path, manifest, "train_pass_candidates.jsonl"
    )
    decisions = _read_jsonl(decisions_path)
    candidates = _read_jsonl(candidates_path)
    if len(decisions) != int(manifest.get("candidate_count", -1)):
        raise BaselineFactorPoolIntegrityError("Train-prefilter decision count mismatch")
    if len(candidates) != int(manifest.get("train_pass_count", -1)):
        raise BaselineFactorPoolIntegrityError("Train-pass candidate count mismatch")
    hashes = [str(row.get("current_structural_hash")) for row in candidates]
    if _stable_hash(hashes) != manifest.get("ordered_train_pass_hashes_fingerprint"):
        raise BaselineFactorPoolIntegrityError("Train-pass candidate order changed")
    passed_decisions = {
        str(row.get("structural_hash"))
        for row in decisions
        if row.get("status") == "train_prefilter_passed"
    }
    if passed_decisions != set(hashes):
        raise BaselineFactorPoolIntegrityError(
            "Train-pass candidates differ from persisted prefilter decisions"
        )
    return manifest, decisions, candidates


def _verify_enriched_selection(
    path: Path,
    *,
    validation_run: Any,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest = _read_json(path)
    if (
        manifest.get("schema") != STAGE6_ENRICHED_SELECTION_MANIFEST_SCHEMA
        or manifest.get("version") != STAGE6_ENRICHED_SELECTION_VERSION
    ):
        raise BaselineFactorPoolIntegrityError(
            "enriched selection schema or version mismatch"
        )
    if manifest.get("scope") != "provisional_selection":
        raise BaselineFactorPoolIntegrityError(
            "only formal provisional_selection can be frozen"
        )
    if manifest.get("engineering_smoke") is not False:
        raise BaselineFactorPoolIntegrityError("engineering smoke cannot be frozen")
    if manifest.get("provisional_evaluation_universe") is not None:
        raise BaselineFactorPoolIntegrityError(
            "resource-limited provisional universe cannot be frozen"
        )
    if manifest.get("oos") != OOS_UNTOUCHED:
        raise BaselineFactorPoolIntegrityError("enriched selection OOS lock changed")
    if manifest.get("selection_contract_fingerprint") != Stage6SelectionConfig().fingerprint:
        raise BaselineFactorPoolIntegrityError("selection contract fingerprint changed")
    if (
        manifest.get("evaluation_run_id") != validation_run.run_id
        or manifest.get("evaluation_ordered_result_set_fingerprint")
        != validation_run.ordered_result_set_fingerprint
        or manifest.get("evaluation_contract_fingerprint")
        != validation_run.manifest["evaluation_contract_fingerprint"]
        or manifest.get("context_fingerprint")
        != validation_run.manifest["context_fingerprint"]
    ):
        raise BaselineFactorPoolIntegrityError(
            "enriched selection and Validation run identity differ"
        )
    deterministic = {
        key: value
        for key, value in manifest.items()
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
    fingerprint = str(manifest.get("enriched_selection_fingerprint"))
    if _stable_hash(deterministic) != fingerprint or path.parent.name != fingerprint:
        raise BaselineFactorPoolIntegrityError(
            "enriched selection fingerprint mismatch"
        )
    hard_path = _require_artifact(path, manifest, "hard_filter_results.jsonl")
    enrichment_path = _require_artifact(
        path, manifest, "survivor_long_excess_enrichment.jsonl"
    )
    greedy_path = _require_artifact(
        path, manifest, "greedy_decorrelation_results.jsonl"
    )
    alpha_path = _require_artifact(path, manifest, "alpha_pool.jsonl")
    hard_rows = _read_jsonl(hard_path)
    enrichment_rows = _read_jsonl(enrichment_path)
    greedy_rows = _read_jsonl(greedy_path)
    alpha_rows = _read_jsonl(alpha_path)
    stable_enrichment = [
        {
            key: value
            for key, value in row.items()
            if key not in {"factor_seconds", "train_long_excess_seconds", "total_seconds"}
        }
        for row in enrichment_rows
    ]
    if (
        _stable_hash(hard_rows) != manifest.get("hard_filter_digest")
        or _stable_hash(stable_enrichment) != manifest.get("enrichment_digest")
        or _stable_hash(greedy_rows) != manifest.get("greedy_digest")
        or _stable_hash(alpha_rows) != manifest.get("alpha_pool_digest")
    ):
        raise BaselineFactorPoolIntegrityError(
            "enriched selection logical artifact digest mismatch"
        )
    counts = manifest.get("counts", {})
    if (
        len(hard_rows) != int(counts.get("input_candidates", -1))
        or len(alpha_rows) != int(counts.get("retained", -1))
        or not alpha_rows
    ):
        raise BaselineFactorPoolIntegrityError(
            "enriched selection count or empty-pool contract mismatch"
        )
    validation_hashes = {
        str(row["structural_hash"]) for row in validation_run.records
    }
    hard_hashes = {str(row.get("structural_hash")) for row in hard_rows}
    hard_pass_hashes = {
        str(row.get("structural_hash"))
        for row in hard_rows
        if row.get("hard_filter_pass") is True
    }
    greedy_hashes = {str(row.get("structural_hash")) for row in greedy_rows}
    retained_hashes = {
        str(row.get("structural_hash"))
        for row in greedy_rows
        if row.get("greedy_retained") is True
        and row.get("decorrelation_status") == "retained"
    }
    alpha_hashes = [str(row.get("structural_hash")) for row in alpha_rows]
    if hard_hashes != validation_hashes:
        raise BaselineFactorPoolIntegrityError(
            "hard-filter universe differs from Validation evaluated set"
        )
    if hard_pass_hashes != greedy_hashes:
        raise BaselineFactorPoolIntegrityError(
            "hard-filter pass set differs from decorrelation input"
        )
    if retained_hashes != set(alpha_hashes) or len(alpha_hashes) != len(
        set(alpha_hashes)
    ):
        raise BaselineFactorPoolIntegrityError(
            "greedy retained set differs from alpha pool"
        )
    retained_in_order = [
        str(row["structural_hash"])
        for row in greedy_rows
        if row.get("greedy_retained") is True
        and row.get("decorrelation_status") == "retained"
    ]
    if retained_in_order != alpha_hashes:
        raise BaselineFactorPoolIntegrityError("alpha-pool line order changed")
    return manifest, hard_rows, greedy_rows, alpha_rows


def _verify_formal_stage6(
    inputs: BaselineFactorPoolFreezeInputs,
) -> _VerifiedStage6Authority:
    paths = inputs.resolved()
    try:
        source_set, source_details = _verify_source_set(
            paths.source_set_manifest_path
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise BaselineFactorPoolIntegrityError("Hybrid source-set verification failed") from error
    if (
        len(source_details) != 1
        or source_details[0]["source"].get("source_type")
        != "hybrid_train_artifact"
    ):
        raise BaselineFactorPoolIntegrityError(
            "formal freeze requires exactly one Hybrid source"
        )
    source = source_details[0]["source"]
    snapshot = source_details[0]["manifest"]
    if (
        snapshot.get("snapshot_kind") != "completed_hybrid_train_artifact"
        or snapshot.get("cutoff", {}).get("complete") is not True
        or snapshot.get("cutoff", {}).get("pending_assignment") is not None
    ):
        raise BaselineFactorPoolIntegrityError("Hybrid Stage 5 source is incomplete")
    snapshot_directory = source_details[0]["path"].parent
    runner_state = _read_json(snapshot_directory / "runner_state.json")
    run_config = _read_json(snapshot_directory / "hybrid_run_config.json")
    artifact = _read_json(snapshot_directory / "train_candidate_artifact.json")
    checkpoint_metadata = _read_json(snapshot_directory / "checkpoint_metadata.json")
    source_contract = artifact.get("train_evaluation_contract")
    source_contract_fingerprint = artifact.get(
        "train_evaluation_contract_fingerprint"
    )
    runner_step = runner_state.get("global_optimizer_step")
    if (
        runner_state.get("schema") != "factor_gfn.hybrid_variance_runner.v1"
        or runner_state.get("complete") is not True
        or runner_state.get("pending_assignment") is not None
        or run_config.get("schema") != "factor_gfn.hybrid_variance_runner.v1"
        or run_config.get("checkpoint_schema")
        != "factor_gfn.checkpoint.hybrid_variance.v1"
        or run_config.get("objective_mode") != "hybrid_variance"
        or artifact.get("schema")
        != "factor_gfn.stage5_train_candidate_artifact.v1"
        or checkpoint_metadata.get("schema")
        != "factor_gfn.checkpoint.hybrid_variance.v1"
        or artifact.get("candidate_count") != len(artifact.get("records", []))
        or not isinstance(source_contract, Mapping)
        or source_contract.get("schema")
        != "factor_gfn.train_evaluation_contract.v1"
        or _stable_hash(source_contract) != source_contract_fingerprint
        or runner_step != artifact.get("committed_optimizer_step")
        or runner_step != checkpoint_metadata.get("global_optimizer_step")
        or runner_step != snapshot.get("cutoff", {}).get("committed_optimizer_step")
        or artifact.get("candidate_count")
        != snapshot.get("cutoff", {}).get("candidate_count")
        or run_config.get("config_fingerprint")
        != checkpoint_metadata.get("config_fingerprint")
        or run_config.get("reward_provider_fingerprint")
        != checkpoint_metadata.get("reward_provider_fingerprint")
        or source_contract.get("provider_fingerprint")
        != run_config.get("reward_provider_fingerprint")
        or any(
            row.get("train_evaluation_contract_fingerprint")
            != source_contract_fingerprint
            for row in artifact.get("records", [])
        )
        or _stable_hash(artifact.get("records", []))
        != snapshot.get("logical_content_fingerprint")
    ):
        raise BaselineFactorPoolIntegrityError(
            "Hybrid runner, checkpoint, artifact, or Train contract identity mismatch"
        )

    try:
        candidate_manifest, _, _ = _verify_candidate_registry(
            paths.candidate_import_manifest_path
        )
        compatibility, accepted = load_stage6_accepted_registry(
            paths.compatibility_manifest_path
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise BaselineFactorPoolIntegrityError(
            "candidate import or compatibility verification failed"
        ) from error
    if (
        candidate_manifest.get("schema") != CANDIDATE_IMPORT_MANIFEST_SCHEMA
        or candidate_manifest.get("downstream_eligible") is not True
        or candidate_manifest.get("source_set_fingerprint")
        != source_set.get("source_set_fingerprint")
    ):
        raise BaselineFactorPoolIntegrityError("candidate import is not downstream eligible")
    if (
        compatibility.get("schema") != EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA
        or compatibility.get("candidate_registry_fingerprint")
        != candidate_manifest.get("registry_fingerprint")
        or compatibility.get("source_set_fingerprint")
        != source_set.get("source_set_fingerprint")
        or _stable_hash(compatibility.get("fingerprint_payload"))
        != compatibility.get("audit_fingerprint")
    ):
        raise BaselineFactorPoolIntegrityError(
            "compatibility manifest provenance mismatch"
        )
    source_hashes = {
        str(row.get("structural_hash")) for row in artifact.get("records", [])
    }
    accepted_hashes = {str(row.get("current_structural_hash")) for row in accepted}
    claimed_hashes = {
        str(row.get("source_claimed_structural_hash")) for row in accepted
    }
    if (
        len(source_hashes) != int(snapshot.get("record_counts", {}).get("records", -1))
        or source_hashes != accepted_hashes
        or source_hashes != claimed_hashes
    ):
        raise BaselineFactorPoolIntegrityError(
            "Stage 5 source universe differs from accepted compatible universe"
        )

    try:
        overlay = Stage6TrainReuseOverlay.load(paths.train_reuse_manifest_path)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise BaselineFactorPoolIntegrityError("Hybrid Train reuse verification failed") from error
    overlay_manifest = overlay.manifest
    if (
        overlay_manifest.get("schema") != HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA
        or overlay_manifest.get("verification_mode") != "hybrid_exact_contract"
        or overlay_manifest.get("source_set_fingerprint")
        != source_set.get("source_set_fingerprint")
        or overlay_manifest.get("source_snapshot_fingerprint")
        != snapshot.get("snapshot_fingerprint")
        or overlay_manifest.get("candidate_registry_fingerprint")
        != candidate_manifest.get("registry_fingerprint")
        or overlay_manifest.get("compatibility_audit_fingerprint")
        != compatibility.get("audit_fingerprint")
        or overlay_manifest.get("accepted_registry_fingerprint")
        != compatibility.get("accepted_registry_fingerprint")
        or set(overlay.records) != accepted_hashes
    ):
        raise BaselineFactorPoolIntegrityError("Hybrid Train reuse provenance mismatch")

    train_entry = _verify_entry(
        paths.train_entry_manifest_path,
        schema=STAGE6_TRAIN_ENTRY_SCHEMA,
        scope=STAGE6_TRAIN_PREPARATION_SCOPE,
    )
    train_run = _load_verified_run(train_entry)
    train_by_hash = {
        str(row["structural_hash"]): row["result"] for row in train_run.records
    }
    if (
        train_entry.get("candidate_count") != len(accepted_hashes)
        or set(train_by_hash) != accepted_hashes
        or train_entry.get("compatibility_audit_fingerprint")
        != compatibility.get("audit_fingerprint")
        or train_entry.get("accepted_registry_fingerprint")
        != compatibility.get("accepted_registry_fingerprint")
        or train_entry.get("train_reuse_overlay_fingerprint") != overlay.fingerprint
        or train_entry.get("evaluation_run_id") != train_run.run_id
        or train_entry.get("context_fingerprint")
        != train_run.manifest.get("context_fingerprint")
        or train_entry.get("evaluation_contract_fingerprint")
        != train_run.manifest.get("evaluation_contract_fingerprint")
        or train_run.manifest.get("scope") != STAGE6_TRAIN_PREPARATION_SCOPE
    ):
        raise BaselineFactorPoolIntegrityError("Train preparation provenance mismatch")
    for result in train_by_hash.values():
        if (
            result.get("validation_evaluation_seconds") != 0.0
            or result.get("validation", {}).get("availability")
            != TRAIN_NOT_EVALUATED_VALIDATION
        ):
            raise BaselineFactorPoolIntegrityError(
                "Train preparation contains Validation information"
            )

    train_pass, _, pass_candidates = _verify_train_pass(
        paths.train_pass_manifest_path
    )
    pass_hashes = [str(row["current_structural_hash"]) for row in pass_candidates]
    if (
        train_pass.get("train_entry_manifest_fingerprint")
        != train_entry.get("entry_manifest_fingerprint")
        or train_pass.get("train_evaluation_run_id") != train_run.run_id
        or train_pass.get("train_ordered_result_set_fingerprint")
        != train_run.ordered_result_set_fingerprint
        or train_pass.get("accepted_registry_fingerprint")
        != compatibility.get("accepted_registry_fingerprint")
        or train_pass.get("selection_config_fingerprint")
        != Stage6SelectionConfig().fingerprint
        or train_pass.get("validation_evaluation_count") != 0
    ):
        raise BaselineFactorPoolIntegrityError("Train-pass provenance mismatch")

    validation_entry = _verify_entry(
        paths.validation_entry_manifest_path,
        schema=STAGE6_VALIDATION_ENTRY_SCHEMA,
        scope=STAGE6_VALIDATION_SCOPE,
    )
    validation_run = _load_verified_run(validation_entry)
    validation_by_hash = {
        str(row["structural_hash"]): row["result"]
        for row in validation_run.records
    }
    if (
        set(pass_hashes) != set(validation_by_hash)
        or len(pass_hashes) != len(validation_by_hash)
        or validation_entry.get("candidate_count") != len(pass_hashes)
        or validation_entry.get("train_entry_manifest_fingerprint")
        != train_entry.get("entry_manifest_fingerprint")
        or validation_entry.get("train_pass_manifest_fingerprint")
        != train_pass.get("train_pass_manifest_fingerprint")
        or validation_entry.get("accepted_registry_fingerprint")
        != compatibility.get("accepted_registry_fingerprint")
        or validation_entry.get("compatibility_audit_fingerprint")
        != compatibility.get("audit_fingerprint")
        or validation_entry.get("evaluation_run_id") != validation_run.run_id
        or validation_entry.get("context_fingerprint")
        != validation_run.manifest.get("context_fingerprint")
        or validation_entry.get("evaluation_contract_fingerprint")
        != validation_run.manifest.get("evaluation_contract_fingerprint")
        or validation_run.manifest.get("scope") != STAGE6_VALIDATION_SCOPE
    ):
        raise BaselineFactorPoolIntegrityError("Validation provenance mismatch")
    for structural_hash, result in validation_by_hash.items():
        train_result = train_by_hash[structural_hash]
        if (
            result.get("train_direction") != train_result.get("train_direction")
            or result.get("train") != train_result.get("train")
            or result.get("source_identity", {}).get(
                "frozen_train_result_fingerprint"
            )
            != train_result.get("result_fingerprint")
        ):
            raise BaselineFactorPoolIntegrityError(
                "Validation result did not preserve its frozen Train result"
            )

    selection, hard_rows, _, alpha_rows = _verify_enriched_selection(
        paths.enriched_selection_manifest_path,
        validation_run=validation_run,
    )
    hard_by_hash = {str(row["structural_hash"]): row for row in hard_rows}
    frozen: list[FrozenBaselineFactorRecord] = []
    for provisional_rank, alpha in enumerate(alpha_rows, start=1):
        structural_hash = str(alpha.get("structural_hash"))
        result = validation_by_hash.get(structural_hash)
        hard = hard_by_hash.get(structural_hash)
        if result is None or hard is None:
            raise BaselineFactorPoolIntegrityError(
                "alpha pool candidate is absent from verified Validation"
            )
        expression = result.get("expression")
        source_identity = result.get("source_identity")
        if not isinstance(expression, Mapping) or not isinstance(
            source_identity, Mapping
        ):
            raise BaselineFactorPoolIntegrityError(
                "alpha pool result lacks factor identity"
            )
        if (
            alpha.get("base_result_fingerprint") != result.get("result_fingerprint")
            or alpha.get("expression") != expression
            or alpha.get("source_identity") != source_identity
            or alpha.get("train_direction") != result.get("train_direction")
            or hard.get("hard_filter_pass") is not True
            or alpha.get("decorrelation_status") != "retained"
            or alpha.get("greedy_retained") is not True
        ):
            raise BaselineFactorPoolIntegrityError(
                "alpha pool identity differs from authoritative selection inputs"
            )
        train_metrics = _metric_summary(result, "train")
        validation_metrics = _metric_summary(result, "validation")
        expected_compact = {
            "train_ic": train_metrics["ic_mean"],
            "validation_ic": validation_metrics["ic_mean"],
            "train_long_ir": train_metrics["long_ir"],
            "validation_long_ir": validation_metrics["long_ir"],
            "train_barra_ts_corr": train_metrics["barra_max_abs_correlation"],
        }
        if alpha.get("metrics") != expected_compact:
            raise BaselineFactorPoolIntegrityError(
                "alpha pool compact metrics differ from Validation EvaluationStore"
            )
        frozen.append(
            FrozenBaselineFactorRecord(
                provisional_rank=provisional_rank,
                stage6_sorted_rank=int(alpha["sorted_rank"]),
                structural_hash=structural_hash,
                formula=str(expression["formula"]),
                prefix_token_ids=tuple(int(value) for value in expression["prefix_token_ids"]),
                node_count=int(expression["node_count"]),
                depth=int(expression["depth"]),
                train_direction=int(result["train_direction"]),
                train_metrics=MappingProxyType(train_metrics),
                validation_metrics=MappingProxyType(validation_metrics),
                selection_status=MappingProxyType(
                    {
                        "hard_filter_pass": True,
                        "decorrelation_status": "retained",
                    }
                ),
                result_identity=MappingProxyType(
                    {
                        "base_result_fingerprint": alpha[
                            "base_result_fingerprint"
                        ],
                        "effective_result_fingerprint": alpha[
                            "effective_result_fingerprint"
                        ],
                        "enrichment_result_fingerprint": alpha[
                            "enrichment_result_fingerprint"
                        ],
                        "validation_result_fingerprint": result[
                            "result_fingerprint"
                        ],
                    }
                ),
                source_identity=MappingProxyType(
                    {
                        "source_claimed_structural_hash": source_identity.get(
                            "source_claimed_structural_hash"
                        ),
                        "source_ids": tuple(source_identity.get("source_ids", [])),
                        "origin_ids": tuple(source_identity.get("origin_ids", [])),
                        "compatibility_record_fingerprint": source_identity.get(
                            "compatibility_record_fingerprint"
                        ),
                    }
                ),
            )
        )

    alpha_metadata = selection["artifacts"]["alpha_pool.jsonl"]
    provenance = {
        "source_id": source["source_id"],
        "source_snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        "source_set_fingerprint": source_set["source_set_fingerprint"],
        "candidate_registry_fingerprint": candidate_manifest["registry_fingerprint"],
        "compatibility_audit_fingerprint": compatibility["audit_fingerprint"],
        "accepted_registry_fingerprint": compatibility[
            "accepted_registry_fingerprint"
        ],
        "train_reuse_overlay_fingerprint": overlay.fingerprint,
        "train_entry_manifest_fingerprint": train_entry[
            "entry_manifest_fingerprint"
        ],
        "train_pass_manifest_fingerprint": train_pass[
            "train_pass_manifest_fingerprint"
        ],
        "validation_run_id": validation_run.run_id,
        "validation_context_fingerprint": validation_run.manifest[
            "context_fingerprint"
        ],
        "validation_evaluation_contract_fingerprint": validation_run.manifest[
            "evaluation_contract_fingerprint"
        ],
        "validation_ordered_result_set_fingerprint": validation_run.ordered_result_set_fingerprint,
        "selection_contract_fingerprint": selection[
            "selection_contract_fingerprint"
        ],
        "enrichment_contract_fingerprint": selection[
            "enrichment_contract_fingerprint"
        ],
        "enriched_selection_fingerprint": selection[
            "enriched_selection_fingerprint"
        ],
        "alpha_pool_digest": selection["alpha_pool_digest"],
        "alpha_pool_sha256": alpha_metadata["sha256"],
    }
    return _VerifiedStage6Authority(
        provenance=MappingProxyType(provenance),
        frozen_records=tuple(frozen),
        enriched_selection_fingerprint=selection[
            "enriched_selection_fingerprint"
        ],
    )


def _freeze_contract() -> dict[str, Any]:
    return {
        "source": "exact_formal_stage6_provisional_pool",
        "composition_policy": "preserve_exactly",
        "ordering_source": "alpha_pool_jsonl_order",
        "direction_source": "frozen_train_direction",
        "empty_pool_allowed": False,
        "oos_use": "forbidden_during_freeze",
    }


def _fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest.get("schema"),
        "version": manifest.get("version"),
        "freeze_contract": manifest.get("freeze_contract"),
        "upstream_provenance": manifest.get("upstream_provenance"),
        "pool": manifest.get("pool"),
        "oos_status": manifest.get("oos_status"),
    }


def _build_manifest(
    *,
    authority: _VerifiedStage6Authority,
    records_bytes: bytes,
    records_digest: str,
    created_at_utc: str,
    inputs: BaselineFactorPoolFreezeInputs,
) -> dict[str, Any]:
    records = [record.to_dict() for record in authority.frozen_records]
    records_sha = hashlib.sha256(records_bytes).hexdigest()
    pool = {
        "factor_count": len(records),
        "ordered_structural_hashes": [row["structural_hash"] for row in records],
        "ordered_train_directions": [row["train_direction"] for row in records],
        "ordered_provisional_ranks": [row["provisional_rank"] for row in records],
        "ordered_stage6_sorted_ranks": [row["stage6_sorted_rank"] for row in records],
        "factor_records_digest": records_digest,
        "factor_records_sha256": records_sha,
    }
    manifest: dict[str, Any] = {
        "schema": BASELINE_FACTOR_POOL_MANIFEST_SCHEMA,
        "version": BASELINE_FACTOR_POOL_VERSION,
        "freeze_contract": _freeze_contract(),
        "authorization": {
            "confirmed_for_freeze": True,
            "method": "explicit_invocation",
            "reviewed_selection_fingerprint": authority.enriched_selection_fingerprint,
        },
        "upstream_provenance": dict(authority.provenance),
        "pool": pool,
        "oos_status": OOS_UNTOUCHED,
        "artifact_metadata": {
            BASELINE_FACTOR_POOL_FILENAME: {
                "size_bytes": len(records_bytes),
                "sha256": records_sha,
                "record_count": len(records),
            }
        },
        "upstream_manifest_paths": inputs.manifest_paths(),
        "created_at_utc": created_at_utc,
        "created_at_excluded_from_fingerprint": True,
    }
    manifest["baseline_factor_pool_fingerprint"] = _stable_hash(
        _fingerprint_payload(manifest)
    )
    return manifest


def _verify_pool_directory(
    manifest_path: Path,
    *,
    require_directory_identity: bool,
    verify_upstream: bool,
) -> VerifiedFrozenBaselineFactorPool:
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != BASELINE_FACTOR_POOL_MANIFEST_SCHEMA
        or manifest.get("version") != BASELINE_FACTOR_POOL_VERSION
    ):
        raise BaselineFactorPoolIntegrityError(
            "Baseline Factor Pool manifest schema or version mismatch"
        )
    fingerprint = str(manifest.get("baseline_factor_pool_fingerprint"))
    if _stable_hash(_fingerprint_payload(manifest)) != fingerprint:
        raise BaselineFactorPoolIntegrityError(
            "Baseline Factor Pool fingerprint mismatch"
        )
    if require_directory_identity and manifest_path.parent.name != fingerprint:
        raise BaselineFactorPoolIntegrityError(
            "Baseline Factor Pool directory identity mismatch"
        )
    if manifest.get("freeze_contract") != _freeze_contract():
        raise BaselineFactorPoolIntegrityError("freeze contract changed")
    if manifest.get("authorization") != {
        "confirmed_for_freeze": True,
        "method": "explicit_invocation",
        "reviewed_selection_fingerprint": manifest.get(
            "upstream_provenance", {}
        ).get("enriched_selection_fingerprint"),
    }:
        raise BaselineFactorPoolIntegrityError("freeze authorization is invalid")
    if manifest.get("oos_status") != OOS_UNTOUCHED:
        raise BaselineFactorPoolIntegrityError("frozen pool OOS state changed")
    records_path = manifest_path.parent / BASELINE_FACTOR_POOL_FILENAME
    metadata = manifest.get("artifact_metadata", {}).get(
        BASELINE_FACTOR_POOL_FILENAME
    )
    if not isinstance(metadata, Mapping) or not records_path.is_file():
        raise BaselineFactorPoolIntegrityError("frozen factor records are missing")
    if (
        records_path.stat().st_size != int(metadata.get("size_bytes", -1))
        or _sha256_file(records_path) != metadata.get("sha256")
    ):
        raise BaselineFactorPoolIntegrityError("frozen factor records changed")
    raw_records = _read_jsonl(records_path)
    records = tuple(FrozenBaselineFactorRecord.from_dict(row) for row in raw_records)
    pool = manifest.get("pool", {})
    ordered_hashes = tuple(record.structural_hash for record in records)
    directions = tuple(record.train_direction for record in records)
    provisional_ranks = tuple(record.provisional_rank for record in records)
    stage6_ranks = tuple(record.stage6_sorted_rank for record in records)
    if (
        not records
        or len(records) != int(metadata.get("record_count", -1))
        or len(records) != int(pool.get("factor_count", -1))
        or list(ordered_hashes) != pool.get("ordered_structural_hashes")
        or list(directions) != pool.get("ordered_train_directions")
        or list(provisional_ranks) != pool.get("ordered_provisional_ranks")
        or list(stage6_ranks) != pool.get("ordered_stage6_sorted_ranks")
        or provisional_ranks != tuple(range(1, len(records) + 1))
        or _stable_hash(raw_records) != pool.get("factor_records_digest")
        or metadata.get("sha256") != pool.get("factor_records_sha256")
    ):
        raise BaselineFactorPoolIntegrityError(
            "frozen factor record count, order, rank, digest, or SHA mismatch"
        )
    if verify_upstream:
        inputs = BaselineFactorPoolFreezeInputs.from_manifest_paths(
            manifest.get("upstream_manifest_paths", {})
        )
        authority = _verify_formal_stage6(inputs)
        current_records = [record.to_dict() for record in authority.frozen_records]
        if (
            dict(authority.provenance) != manifest.get("upstream_provenance")
            or current_records != raw_records
        ):
            raise BaselineFactorPoolIntegrityError(
                "frozen pool differs from its current verified Stage 6 authority"
            )
    return VerifiedFrozenBaselineFactorPool(
        manifest_path=manifest_path,
        records_path=records_path,
        baseline_factor_pool_fingerprint=fingerprint,
        manifest=_deep_freeze(manifest),
        records=records,
        ordered_structural_hashes=ordered_hashes,
        frozen_train_directions=directions,
        upstream_provenance=_deep_freeze(
            dict(manifest.get("upstream_provenance", {}))
        ),
        oos_status=OOS_UNTOUCHED,
    )


def freeze_baseline_factor_pool(
    inputs: BaselineFactorPoolFreezeInputs,
    runs_root: str | Path,
    *,
    confirmed_for_freeze: bool = False,
    reviewed_selection_fingerprint: str | None = None,
) -> BaselineFactorPoolArtifact:
    """Verify and freeze the exact formal Stage 6 provisional pool.

    ``runs_root`` is the parent of the content-addressed
    ``baseline_factor_pools`` directory.  The function never selects,
    evaluates, or repairs factors.
    """

    if confirmed_for_freeze is not True:
        raise BaselineFactorPoolIntegrityError(
            "formal freeze requires confirmed_for_freeze=True"
        )
    authority = _verify_formal_stage6(inputs)
    if reviewed_selection_fingerprint != authority.enriched_selection_fingerprint:
        raise BaselineFactorPoolIntegrityError(
            "reviewed selection fingerprint does not match authoritative Stage 6"
        )
    raw_records = [record.to_dict() for record in authority.frozen_records]
    if not raw_records:
        raise BaselineFactorPoolIntegrityError("empty formal pool cannot be frozen")
    records_bytes = _jsonl_bytes(raw_records)
    records_digest = _stable_hash(raw_records)
    manifest = _build_manifest(
        authority=authority,
        records_bytes=records_bytes,
        records_digest=records_digest,
        created_at_utc=datetime.now(UTC).isoformat(),
        inputs=inputs,
    )
    fingerprint = manifest["baseline_factor_pool_fingerprint"]
    pool_root = Path(runs_root).resolve() / "baseline_factor_pools"
    target = pool_root / fingerprint
    manifest_path = target / BASELINE_FACTOR_POOL_MANIFEST_FILENAME
    if target.exists():
        if not manifest_path.is_file():
            raise BaselineFactorPoolIntegrityError(
                "content-addressed freeze target exists without its manifest"
            )
        verified = _verify_pool_directory(
            manifest_path,
            require_directory_identity=True,
            verify_upstream=True,
        )
        if (
            verified.records_path.read_bytes() != records_bytes
            or verified.baseline_factor_pool_fingerprint != fingerprint
        ):
            raise BaselineFactorPoolIntegrityError(
                "content-addressed freeze target conflicts with requested freeze"
            )
        return BaselineFactorPoolArtifact(
            manifest_path=verified.manifest_path,
            records_path=verified.records_path,
            baseline_factor_pool_fingerprint=fingerprint,
            factor_count=len(raw_records),
            reused_existing_artifact=True,
        )

    pool_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.tmp-", dir=pool_root))
    try:
        records_path = temporary / BASELINE_FACTOR_POOL_FILENAME
        records_path.write_bytes(records_bytes)
        temporary_manifest_path = temporary / BASELINE_FACTOR_POOL_MANIFEST_FILENAME
        _write_json(temporary_manifest_path, manifest)
        _verify_pool_directory(
            temporary_manifest_path,
            require_directory_identity=False,
            verify_upstream=True,
        )
        try:
            os.replace(temporary, target)
        except OSError as error:
            if not target.exists():
                raise
            if not manifest_path.is_file():
                raise BaselineFactorPoolIntegrityError(
                    "concurrent freeze target is incomplete"
                ) from error
            verified = _verify_pool_directory(
                manifest_path,
                require_directory_identity=True,
                verify_upstream=True,
            )
            if verified.records_path.read_bytes() != records_bytes:
                raise BaselineFactorPoolIntegrityError(
                    "concurrent freeze target conflicts with requested freeze"
                ) from error
            return BaselineFactorPoolArtifact(
                manifest_path=verified.manifest_path,
                records_path=verified.records_path,
                baseline_factor_pool_fingerprint=fingerprint,
                factor_count=len(raw_records),
                reused_existing_artifact=True,
            )
        verified = _verify_pool_directory(
            manifest_path,
            require_directory_identity=True,
            verify_upstream=True,
        )
        return BaselineFactorPoolArtifact(
            manifest_path=verified.manifest_path,
            records_path=verified.records_path,
            baseline_factor_pool_fingerprint=fingerprint,
            factor_count=len(raw_records),
            reused_existing_artifact=False,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_verified_baseline_factor_pool(
    manifest_path: str | Path,
) -> VerifiedFrozenBaselineFactorPool:
    """Load a frozen pool only after artifact and upstream re-verification."""

    return _verify_pool_directory(
        Path(manifest_path).resolve(),
        require_directory_identity=True,
        verify_upstream=True,
    )


__all__ = [
    "BASELINE_FACTOR_POOL_FILENAME",
    "BASELINE_FACTOR_POOL_MANIFEST_FILENAME",
    "BASELINE_FACTOR_POOL_MANIFEST_SCHEMA",
    "BASELINE_FACTOR_POOL_VERSION",
    "BASELINE_FACTOR_RECORD_SCHEMA",
    "BaselineFactorPoolArtifact",
    "BaselineFactorPoolFreezeInputs",
    "BaselineFactorPoolIntegrityError",
    "FrozenBaselineFactorRecord",
    "VerifiedFrozenBaselineFactorPool",
    "freeze_baseline_factor_pool",
    "load_verified_baseline_factor_pool",
]
