"""Fail-closed import of historical implied-logZ medians as constants only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .exhaustive_registry_reuse import ExhaustiveReuseSemantics


HISTORICAL_LOG_Z_INITIALIZATION_SCHEMA = (
    "factor_gfn.historical_log_z_initialization.v1"
)
_SOURCE_DIAGNOSTIC_SCHEMA = "factor_gfn.conditional_diagnostic.v1"
_SOURCE_CHECKPOINT_SCHEMA = "factor_gfn.checkpoint.v5"


def _canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"historical artifact must contain a JSON object: {path}")
    return value


def _integer_keys(value: Any, name: str) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        normalized = {int(key): item for key, item in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} keys must be node counts") from error
    if len(normalized) != len(value):
        raise ValueError(f"{name} has ambiguous node-count keys")
    return normalized


@dataclass(frozen=True, slots=True)
class HistoricalLogZInitialization:
    """Auditable constants imported without any historical training state."""

    schema: str
    source_diagnostic_schema: str
    source_checkpoint_schema: str
    source_run_id: str
    source_config_fingerprint: str
    source_provider_fingerprint: str
    source_context_fingerprint: str
    source_summary_sha256: str
    source_context_sha256: str
    source_checkpoint_sha256: str
    learned_node_counts: tuple[int, ...]
    median_log_z_by_N: dict[int, float]
    calibration_statistics_by_N: dict[int, dict[str, Any]]
    semantics: ExhaustiveReuseSemantics
    provenance_fingerprint: str
    reuse_scope: str = "initialization_constants_only"
    restored_training_state: bool = False

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_LOG_Z_INITIALIZATION_SCHEMA:
            raise ValueError("historical logZ initialization schema is incompatible")
        if self.reuse_scope != "initialization_constants_only":
            raise ValueError("historical logZ reuse must be constants-only")
        if self.restored_training_state is not False:
            raise ValueError("historical initialization cannot restore training state")
        learned = tuple(sorted(self.learned_node_counts))
        if learned != self.learned_node_counts or len(learned) != len(set(learned)):
            raise ValueError("learned_node_counts must be sorted and unique")
        if set(self.median_log_z_by_N) != set(learned):
            raise ValueError("historical median coverage must exactly match learned strata")
        if set(self.calibration_statistics_by_N) != set(learned):
            raise ValueError("historical calibration audit must exactly match learned strata")
        if any(not math.isfinite(float(value)) for value in self.median_log_z_by_N.values()):
            raise ValueError("historical median logZ values must be finite")
        fingerprint_payload = asdict(self)
        fingerprint_payload.pop("provenance_fingerprint")
        if _canonical_fingerprint(fingerprint_payload) != self.provenance_fingerprint:
            raise ValueError("historical logZ provenance fingerprint mismatch")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def _source_semantics(checkpoint: dict[str, Any]) -> ExhaustiveReuseSemantics:
    run_metadata = checkpoint.get("run_metadata")
    if not isinstance(run_metadata, dict):
        raise ValueError("historical checkpoint lacks run_metadata")
    config_manifest = run_metadata.get("config_manifest")
    provider_manifest = run_metadata.get("reward_provider")
    if not isinstance(config_manifest, dict) or not isinstance(provider_manifest, dict):
        raise ValueError("historical checkpoint lacks config/provider manifests")
    config_payload = config_manifest.get("config")
    if not isinstance(config_payload, dict) or not isinstance(
        config_payload.get("reward"), dict
    ):
        raise ValueError("historical checkpoint lacks Reward configuration")
    interpreter_payload = provider_manifest.get("interpreter")
    if interpreter_payload is None:
        interpreter_payload = {
            "provider_schema": provider_manifest.get("schema"),
            "declared_interpreter": False,
        }
    return ExhaustiveReuseSemantics(
        grammar_semantics_fingerprint=_canonical_fingerprint(
            {
                "state_space": config_manifest.get("state_space_fingerprint"),
                "transition_space": config_manifest.get(
                    "transition_space_fingerprint"
                ),
            }
        ),
        operator_semantics_fingerprint=str(
            config_manifest.get("token_space_fingerprint", "")
        ),
        interpreter_semantics_fingerprint=_canonical_fingerprint(
            interpreter_payload
        ),
        provider_fingerprint=str(checkpoint.get("reward_provider_fingerprint", "")),
        data_context_fingerprint=str(provider_manifest.get("context_fingerprint", "")),
        reward_config_fingerprint=_canonical_fingerprint(config_payload["reward"]),
        reward_floor=float(config_payload["reward"]["reward_floor"]),
    )


def load_verified_historical_log_z_initialization(
    source_directory: str | Path,
    *,
    target_search_space: Mapping[str, int],
    target_feasible_node_counts: tuple[int, ...],
    target_exact_node_counts: tuple[int, ...],
    target_learned_node_counts: tuple[int, ...],
    target_semantics: ExhaustiveReuseSemantics,
    import_node_counts: tuple[int, ...] | None = None,
) -> HistoricalLogZInitialization:
    """Verify old diagnostic artifacts and return medians, never model state."""

    root = Path(source_directory).resolve()
    summary_path = root / "diagnostic_summary.json"
    context_path = root / "diagnostic_context.json"
    checkpoint_path = root / "diagnostic_checkpoint.pt"
    summary = _load_json(summary_path)
    context = _load_json(context_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("historical checkpoint payload must be a mapping")
    if summary.get("schema") != _SOURCE_DIAGNOSTIC_SCHEMA:
        raise ValueError("historical summary schema is incompatible")
    if context.get("schema") != _SOURCE_DIAGNOSTIC_SCHEMA:
        raise ValueError("historical context schema is incompatible")
    if checkpoint.get("schema") != _SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("historical initialization requires the approved v5 diagnostic")
    if summary.get("training_only") is not True:
        raise ValueError("historical initialization source is not training-only")
    if summary.get("validation_oos_not_loaded") is not True:
        raise ValueError("historical initialization source loaded validation/OOS")
    if summary.get("industry_neutralization_on") is not True:
        raise ValueError("historical initialization source lacks industry neutralization")
    expected_search_space = {
        "max_depth": int(target_search_space["max_depth"]),
        "max_nodes": int(target_search_space["max_nodes"]),
    }
    if context.get("search_space") != expected_search_space:
        raise ValueError("historical/current search-space boundary mismatch")
    expected_F = list(target_feasible_node_counts)
    expected_E = list(target_exact_node_counts)
    expected_L = list(target_learned_node_counts)
    selected = (
        target_learned_node_counts
        if import_node_counts is None
        else tuple(sorted(import_node_counts))
    )
    if len(selected) != len(set(selected)) or not set(selected).issubset(
        target_learned_node_counts
    ):
        raise ValueError("historical import node counts must be a unique subset of L")
    if summary.get("resolved_F") != expected_F:
        raise ValueError("historical/current feasible strata mismatch")
    if summary.get("resolved_E") != expected_E:
        raise ValueError("historical/current exact strata mismatch")
    if summary.get("resolved_S") != expected_L:
        raise ValueError("historical diagnostic discovery strata do not match current L")
    resolved = context.get("resolved_complexity")
    if not isinstance(resolved, dict):
        raise ValueError("historical context lacks resolved complexity")
    if resolved.get("resolved_feasible_node_counts") != expected_F:
        raise ValueError("historical context feasible strata mismatch")
    if resolved.get("resolved_exhaustive_node_counts") != expected_E:
        raise ValueError("historical context exact strata mismatch")
    if resolved.get("resolved_discovery_node_counts") != expected_L:
        raise ValueError("historical context discovery strata mismatch")
    source_config_fingerprint = str(context.get("config_fingerprint", ""))
    if summary.get("config_fingerprint") != source_config_fingerprint:
        raise ValueError("historical summary/context config fingerprints differ")
    if checkpoint.get("config_fingerprint") != source_config_fingerprint:
        raise ValueError("historical checkpoint/context config fingerprints differ")
    source_provider_fingerprint = str(context.get("provider_fingerprint", ""))
    source_context_fingerprint = str(context.get("context_fingerprint", ""))
    if checkpoint.get("reward_provider_fingerprint") != source_provider_fingerprint:
        raise ValueError("historical provider fingerprints differ")
    run_metadata = checkpoint["run_metadata"]
    provider_manifest = run_metadata.get("reward_provider", {})
    if provider_manifest.get("data_scope") != "training_only":
        raise ValueError("historical checkpoint provider is not training-only")
    if provider_manifest.get("validation_oos_loaded") is not False:
        raise ValueError("historical checkpoint provider loaded validation/OOS")
    if provider_manifest.get("context_fingerprint") != source_context_fingerprint:
        raise ValueError("historical data-context fingerprints differ")
    source_semantics = _source_semantics(checkpoint)
    if source_semantics != target_semantics:
        differing = [
            name
            for name in asdict(source_semantics)
            if getattr(source_semantics, name) != getattr(target_semantics, name)
        ]
        raise ValueError(f"historical/current semantics mismatch: {differing}")
    calibration = _integer_keys(summary.get("calibration_by_N"), "calibration_by_N")
    statistics: dict[int, dict[str, Any]] = {}
    medians: dict[int, float] = {}
    for node_count in target_learned_node_counts:
        item = calibration.get(node_count)
        if not isinstance(item, dict):
            raise ValueError(f"historical calibration lacks N={node_count}")
        if int(item.get("node_count", -1)) != node_count:
            raise ValueError(f"historical calibration node identity mismatch for N={node_count}")
        requested = int(item.get("calibration_requested", 0))
        valid = int(item.get("calibration_valid", 0))
        sampled = int(item.get("calibration_sampled_attempts", 0))
        median = float(item.get("median", math.nan))
        if requested < 1 or valid < 1 or sampled < requested or valid > requested:
            raise ValueError(f"historical calibration counts are invalid for N={node_count}")
        if not math.isfinite(median):
            raise ValueError(f"historical calibration median is not finite for N={node_count}")
        if node_count in selected:
            statistics[node_count] = dict(item)
            medians[node_count] = median
    exact = _integer_keys(summary.get("exact_by_N"), "exact_by_N")
    if set(exact) != set(target_exact_node_counts):
        raise ValueError("historical exact-mass coverage mismatch")
    for node_count, item in exact.items():
        if not isinstance(item, dict):
            raise ValueError(f"historical exact mass is malformed for N={node_count}")
        if item.get("provider_fingerprint") != source_provider_fingerprint:
            raise ValueError("historical exact mass provider fingerprint mismatch")
        if item.get("context_fingerprint") != source_context_fingerprint:
            raise ValueError("historical exact mass context fingerprint mismatch")
        if float(item.get("reward_floor", math.nan)) != target_semantics.reward_floor:
            raise ValueError("historical exact mass reward floor mismatch")
        if not math.isfinite(float(item.get("exact_tb_log_z", math.nan))):
            raise ValueError("historical exact TB logZ is not finite")
    provenance_payload = {
        "schema": HISTORICAL_LOG_Z_INITIALIZATION_SCHEMA,
        "source_diagnostic_schema": summary["schema"],
        "source_checkpoint_schema": checkpoint["schema"],
        "source_run_id": str(checkpoint.get("run_id", "")),
        "source_config_fingerprint": source_config_fingerprint,
        "source_provider_fingerprint": source_provider_fingerprint,
        "source_context_fingerprint": source_context_fingerprint,
        "source_summary_sha256": _file_sha256(summary_path),
        "source_context_sha256": _file_sha256(context_path),
        "source_checkpoint_sha256": _file_sha256(checkpoint_path),
        "learned_node_counts": selected,
        "median_log_z_by_N": medians,
        "calibration_statistics_by_N": statistics,
        "semantics": asdict(source_semantics),
        "reuse_scope": "initialization_constants_only",
        "restored_training_state": False,
    }
    record_payload = dict(provenance_payload)
    record_payload["semantics"] = source_semantics
    record_payload["provenance_fingerprint"] = _canonical_fingerprint(
        provenance_payload
    )
    return HistoricalLogZInitialization(**record_payload)


def historical_log_z_initialization_from_manifest(
    value: Mapping[str, Any],
) -> HistoricalLogZInitialization:
    payload = dict(value)
    payload["learned_node_counts"] = tuple(payload["learned_node_counts"])
    payload["median_log_z_by_N"] = _integer_keys(
        payload["median_log_z_by_N"], "median_log_z_by_N"
    )
    payload["calibration_statistics_by_N"] = _integer_keys(
        payload["calibration_statistics_by_N"], "calibration_statistics_by_N"
    )
    semantics = payload["semantics"]
    if not isinstance(semantics, ExhaustiveReuseSemantics):
        payload["semantics"] = ExhaustiveReuseSemantics(**semantics)
    return HistoricalLogZInitialization(**payload)


__all__ = [
    "HISTORICAL_LOG_Z_INITIALIZATION_SCHEMA",
    "HistoricalLogZInitialization",
    "historical_log_z_initialization_from_manifest",
    "load_verified_historical_log_z_initialization",
]
