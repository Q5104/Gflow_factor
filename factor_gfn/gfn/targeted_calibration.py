"""Auditable targeted implied-logZ calibration artifacts and resumable progress."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import torch

from .calibration import NormalizerCalibration, require_training_only_provider_manifest
from .calibration_stability import CalibrationStabilityConfig
from .no_anchor_config import NoAnchorGFNConfig

if TYPE_CHECKING:
    from .trainer import GFNTrainer


TARGETED_CALIBRATION_PROGRESS_SCHEMA = (
    "factor_gfn.targeted_log_z_calibration.progress.v1"
)
TARGETED_CALIBRATION_ARTIFACT_SCHEMA = (
    "factor_gfn.targeted_log_z_calibration.artifact.v1"
)
TARGETED_LOG_Z_INITIALIZATION_SCHEMA = (
    "factor_gfn.targeted_log_z_initialization.v1"
)


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


def policy_state_fingerprint(trainer: "GFNTrainer") -> str:
    """Hash the exact initialized policy state without including optimizer state."""

    digest = hashlib.sha256()
    for name, tensor in sorted(trainer.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _proof_fingerprint(trainer: "GFNTrainer") -> str:
    manifests = {
        str(node_count): asdict(proof)
        for node_count, proof in sorted(trainer.exhaustive_reuse_proofs_by_N.items())
    }
    return _canonical_fingerprint(manifests)


def targeted_calibration_context(trainer: "GFNTrainer") -> dict[str, Any]:
    """Return the semantics used by calibration, excluding unused training dynamics."""

    if not trainer.no_anchor_mode or not isinstance(trainer.config, NoAnchorGFNConfig):
        raise RuntimeError("targeted calibration requires a no-anchor Trainer")
    config = trainer.config
    if not config.calibration.enabled:
        raise RuntimeError("targeted calibration is disabled")
    manifest = trainer.reward_provider.manifest()
    require_training_only_provider_manifest(manifest)
    if trainer.historical_log_z_initialization is None:
        raise RuntimeError("targeted calibration requires verified historical initialization")
    expected_historical = set(trainer.resolved_learned_node_counts) - set(
        config.calibration.target_node_counts
    )
    if set(trainer.historical_log_z_initialization.learned_node_counts) != expected_historical:
        raise RuntimeError("historical initialization does not cover exactly L-targets")
    return {
        "schema": "factor_gfn.targeted_log_z_calibration.context.v1",
        "search_space": asdict(config.search_space),
        "model": asdict(config.model),
        "sampling": asdict(config.sampling),
        "reward": asdict(config.reward),
        "seed": int(config.training.seed),
        "retry_budget": int(config.complexity.exact_node_retry_budget),
        "calibration": asdict(config.calibration),
        "resolved_F": list(trainer.resolved_feasible_node_counts),
        "resolved_E": list(trainer.resolved_exhaustive_node_counts),
        "resolved_L": list(trainer.resolved_learned_node_counts),
        "provider_fingerprint": trainer.reward_provider.fingerprint(),
        "context_fingerprint": manifest["context_fingerprint"],
        "provider_data_scope": manifest["data_scope"],
        "validation_oos_loaded": manifest["validation_oos_loaded"],
        "semantics": asdict(trainer.target_exhaustive_reuse_semantics()),
        "registry_equivalence_fingerprint": _proof_fingerprint(trainer),
        "historical_provenance_fingerprint": (
            trainer.historical_log_z_initialization.provenance_fingerprint
        ),
        "initial_policy_state_fingerprint": policy_state_fingerprint(trainer),
    }


def _fresh_calibration_engine(trainer: "GFNTrainer") -> NormalizerCalibration:
    config = trainer.config.calibration
    return NormalizerCalibration(
        node_counts=config.target_node_counts,
        exhaustive_node_counts=(),
        minimum_valid_samples=config.minimum_valid_samples,
        maximum_requested_slots_per_N=config.maximum_requested_slots_per_N,
        seed=trainer.config.training.seed,
        stability_config=CalibrationStabilityConfig(
            minimum_valid_samples=config.minimum_valid_samples,
            maximum_requested_slots=config.maximum_requested_slots_per_N,
            comparison_window=config.comparison_window,
            median_absolute_tolerance=config.median_absolute_tolerance,
            iqr_absolute_tolerance=config.iqr_absolute_tolerance,
        ),
    )


def _require_fresh_calibration_trainer(trainer: "GFNTrainer") -> None:
    if trainer.step != 0 or trainer.optimizer_step != 0 or trainer.history:
        raise RuntimeError("targeted calibration requires a fresh training state")
    if trainer.optimizer.state:
        raise RuntimeError("targeted calibration forbids existing optimizer state")
    if trainer.calibration is None:
        raise RuntimeError("targeted calibration engine is unavailable")


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise ValueError("targeted calibration progress requires CUDA RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_targeted_calibration_progress(
    path: str | os.PathLike[str], trainer: "GFNTrainer"
) -> None:
    """Atomically save calibration-only state; no trained model/optimizer is stored."""

    _require_fresh_calibration_trainer(trainer)
    context = targeted_calibration_context(trainer)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "schema": TARGETED_CALIBRATION_PROGRESS_SCHEMA,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": trainer.run_id,
        "created_at_utc": trainer.created_at_utc,
        "context": context,
        "context_fingerprint": _canonical_fingerprint(context),
        "calibration_engine": trainer.calibration.state_dict(),
        "rng_state": _capture_rng_state(),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_targeted_calibration_progress(
    path: str | os.PathLike[str], trainer: "GFNTrainer"
) -> None:
    """Resume only calibration scheduler/observations/RNG after strict context proof."""

    _require_fresh_calibration_trainer(trainer)
    payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != TARGETED_CALIBRATION_PROGRESS_SCHEMA:
        raise ValueError("targeted calibration progress schema is incompatible")
    context = targeted_calibration_context(trainer)
    if payload.get("context") != context:
        raise ValueError("targeted calibration progress context mismatch")
    if payload.get("context_fingerprint") != _canonical_fingerprint(context):
        raise ValueError("targeted calibration progress fingerprint mismatch")
    engine = _fresh_calibration_engine(trainer)
    engine.load_state_dict(payload["calibration_engine"])
    trainer.calibration = engine
    if engine.status == "complete":
        assert trainer.tb_loss.learned_log_z_initialized_mask is not None
        targets = tuple(trainer.config.calibration.target_node_counts)
        if any(
            bool(trainer.tb_loss.learned_log_z_initialized_mask[node_count - 1])
            for node_count in targets
        ):
            raise RuntimeError("targeted progress target scalar is already initialized")
        for node_count in targets:
            trainer.tb_loss.initialize_learned_log_z(
                node_count,
                engine.statistics_by_N[node_count].median,
            )
    elif engine.status == "failed":
        raise RuntimeError(
            f"targeted calibration progress is fail-closed: {engine.failure_reason}"
        )
    trainer.run_id = str(payload["run_id"])
    trainer.created_at_utc = str(payload["created_at_utc"])
    _restore_rng_state(payload["rng_state"])


@dataclass(frozen=True, slots=True)
class TargetedLogZInitialization:
    schema: str
    source_artifact_schema: str
    source_artifact_sha256: str
    source_run_id: str
    target_node_counts: tuple[int, ...]
    median_log_z_by_N: dict[int, float]
    calibration_statistics_by_N: dict[int, dict[str, Any]]
    calibration_stability_by_N: dict[int, dict[str, Any]]
    calibration_context_fingerprint: str
    artifact_fingerprint: str
    provenance_fingerprint: str
    initialization_status: str = "strict_stability_calibration"
    strict_stability_check: str = "passed"
    approval_reason: str | None = None
    reuse_scope: str = "initialization_constants_only"
    restored_training_state: bool = False

    def __post_init__(self) -> None:
        if self.schema != TARGETED_LOG_Z_INITIALIZATION_SCHEMA:
            raise ValueError("targeted logZ initialization schema is incompatible")
        targets = tuple(sorted(self.target_node_counts))
        if targets != self.target_node_counts or len(targets) != len(set(targets)):
            raise ValueError("targeted node counts must be sorted and unique")
        if set(self.median_log_z_by_N) != set(targets):
            raise ValueError("targeted median coverage mismatch")
        if set(self.calibration_statistics_by_N) != set(targets):
            raise ValueError("targeted statistics coverage mismatch")
        if set(self.calibration_stability_by_N) != set(targets):
            raise ValueError("targeted stability coverage mismatch")
        if any(not math.isfinite(float(value)) for value in self.median_log_z_by_N.values()):
            raise ValueError("targeted median values must be finite")
        if self.reuse_scope != "initialization_constants_only":
            raise ValueError("targeted result reuse is constants-only")
        if self.restored_training_state is not False:
            raise ValueError("targeted result cannot restore training state")
        if self.initialization_status not in (
            "strict_stability_calibration",
            "high_variance_engineering_estimate",
        ):
            raise ValueError("unsupported targeted initialization status")
        expected_check = (
            "passed"
            if self.initialization_status == "strict_stability_calibration"
            else "failed"
        )
        if self.strict_stability_check != expected_check:
            raise ValueError("targeted initialization stability label is inconsistent")
        if (
            self.initialization_status == "high_variance_engineering_estimate"
            and not self.approval_reason
        ):
            raise ValueError("high-variance engineering initialization requires approval")
        fingerprint_payload = asdict(self)
        fingerprint_payload.pop("provenance_fingerprint")
        if _canonical_fingerprint(fingerprint_payload) != self.provenance_fingerprint:
            raise ValueError("targeted logZ provenance fingerprint mismatch")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def write_targeted_calibration_artifact(
    path: str | os.PathLike[str],
    trainer: "GFNTrainer",
) -> dict[str, Any]:
    """Write the completed result as JSON, separate from resumable progress state."""

    _require_fresh_calibration_trainer(trainer)
    assert trainer.calibration is not None
    if trainer.calibration.status != "complete":
        raise RuntimeError("targeted calibration artifact requires complete calibration")
    context = targeted_calibration_context(trainer)
    targets = tuple(trainer.config.calibration.target_node_counts)
    stability = trainer.calibration.stability_by_N
    if set(stability) != set(targets) or any(
        result.status != "stable" for result in stability.values()
    ):
        raise RuntimeError("targeted calibration artifact requires stable results for every N")
    statistics = trainer.calibration.statistics_by_N
    payload: dict[str, Any] = {
        "schema": TARGETED_CALIBRATION_ARTIFACT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": trainer.run_id,
        "target_node_counts": list(targets),
        "calibration_context": context,
        "calibration_context_fingerprint": _canonical_fingerprint(context),
        "calibration_engine": trainer.calibration.state_dict(),
        "median_log_z_by_N": {
            str(node_count): statistics[node_count].median for node_count in targets
        },
        "calibration_statistics_by_N": {
            str(node_count): asdict(statistics[node_count]) for node_count in targets
        },
        "calibration_stability_by_N": {
            str(node_count): asdict(stability[node_count]) for node_count in targets
        },
        "initialization_status": "strict_stability_calibration",
        "strict_stability_check": "passed",
        "approval_reason": None,
        "reuse_scope": "initialization_constants_only",
        "restored_training_state": False,
    }
    payload["artifact_fingerprint"] = _canonical_fingerprint(payload)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, target)
    return payload


def write_high_variance_engineering_artifact_from_progress(
    progress_path: str | os.PathLike[str],
    artifact_path: str | os.PathLike[str],
    *,
    approval_reason: str,
) -> dict[str, Any]:
    """Freeze all valid observations as an explicitly non-stable engineering estimate.

    This is deliberately separate from ``write_targeted_calibration_artifact``:
    the strict writer remains fail-closed, while this path requires an explicit
    human approval string and preserves the failed stability diagnostics.
    """

    if not isinstance(approval_reason, str) or not approval_reason.strip():
        raise ValueError("engineering initialization requires an approval reason")
    source = Path(progress_path).resolve()
    progress = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(progress, dict) or progress.get("schema") != TARGETED_CALIBRATION_PROGRESS_SCHEMA:
        raise ValueError("targeted calibration progress schema is incompatible")
    context = progress.get("context")
    if not isinstance(context, dict):
        raise ValueError("targeted calibration progress lacks context")
    if progress.get("context_fingerprint") != _canonical_fingerprint(context):
        raise ValueError("targeted calibration progress fingerprint mismatch")
    engine_state = dict(progress["calibration_engine"])
    targets = tuple(int(value) for value in context["calibration"]["target_node_counts"])
    observations = _integer_keys(
        engine_state["implied_log_z_by_N"], "implied_log_z_by_N"
    )
    requested = _integer_keys(engine_state["requested_by_N"], "requested_by_N")
    valid = _integer_keys(engine_state["valid_by_N"], "valid_by_N")
    attempts = _integer_keys(
        engine_state["sampled_attempts_by_N"], "sampled_attempts_by_N"
    )
    if set(observations) != set(targets):
        raise ValueError("targeted progress strata mismatch")
    minimum_valid = int(context["calibration"]["minimum_valid_samples"])
    statistics = {}
    for node_count in targets:
        values = [float(value) for value in observations[node_count]]
        if len(values) != valid[node_count] or len(values) < minimum_valid:
            raise ValueError("engineering estimate lacks minimum valid samples")
        statistics[node_count] = NormalizerCalibration._summarize(
            node_count,
            values,
            requested=requested[node_count],
            sampled_attempts=attempts[node_count],
            exact_tb_log_z=None,
        )
    stability = _integer_keys(
        engine_state.get("stability_by_N", {}), "stability_by_N"
    )
    if set(stability) != set(targets):
        raise ValueError("engineering estimate lacks per-N stability diagnostics")
    if all(str(stability[node_count].get("status")) == "stable" for node_count in targets):
        raise ValueError("stable observations must use the strict artifact writer")
    import_state = dict(engine_state)
    import_state["status"] = "complete"
    import_state["failure_reason"] = None
    import_state["statistics_by_N"] = {
        node_count: asdict(statistics[node_count]) for node_count in targets
    }
    payload: dict[str, Any] = {
        "schema": TARGETED_CALIBRATION_ARTIFACT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": str(progress["run_id"]),
        "source_progress_schema": TARGETED_CALIBRATION_PROGRESS_SCHEMA,
        "source_progress_sha256": _file_sha256(source),
        "target_node_counts": list(targets),
        "calibration_context": context,
        "calibration_context_fingerprint": _canonical_fingerprint(context),
        "calibration_engine": import_state,
        "median_log_z_by_N": {
            str(node_count): statistics[node_count].median for node_count in targets
        },
        "calibration_statistics_by_N": {
            str(node_count): asdict(statistics[node_count]) for node_count in targets
        },
        "calibration_stability_by_N": {
            str(node_count): dict(stability[node_count]) for node_count in targets
        },
        "initialization_status": "high_variance_engineering_estimate",
        "strict_stability_check": "failed",
        "approval_reason": approval_reason.strip(),
        "reuse_scope": "initialization_constants_only",
        "restored_training_state": False,
    }
    payload["artifact_fingerprint"] = _canonical_fingerprint(payload)
    target = Path(artifact_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, target)
    return payload


def _integer_keys(value: Any, name: str) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized = {int(key): item for key, item in value.items()}
    if len(normalized) != len(value):
        raise ValueError(f"{name} contains ambiguous keys")
    return normalized


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def load_verified_targeted_calibration_artifact(
    path: str | os.PathLike[str], trainer: "GFNTrainer"
) -> TargetedLogZInitialization:
    """Initialize only named logZ scalars in a fresh Trainer, fail-atomically."""

    _require_fresh_calibration_trainer(trainer)
    if getattr(trainer, "targeted_log_z_initialization", None) is not None:
        raise RuntimeError("targeted logZ initialization is already configured")
    target = Path(path).resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != TARGETED_CALIBRATION_ARTIFACT_SCHEMA:
        raise ValueError("targeted calibration artifact schema is incompatible")
    fingerprint = payload.get("artifact_fingerprint")
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("artifact_fingerprint", None)
    if fingerprint != _canonical_fingerprint(fingerprint_payload):
        raise ValueError("targeted calibration artifact fingerprint mismatch")
    context = targeted_calibration_context(trainer)
    source_context = payload.get("calibration_context")
    if _canonical_fingerprint(source_context) != _canonical_fingerprint(context):
        raise ValueError("targeted calibration/current context mismatch")
    if payload.get("calibration_context_fingerprint") != _canonical_fingerprint(context):
        raise ValueError("targeted calibration context fingerprint mismatch")
    targets = tuple(int(value) for value in payload["target_node_counts"])
    if targets != tuple(trainer.config.calibration.target_node_counts):
        raise ValueError("targeted calibration strata mismatch")
    engine_state = payload["calibration_engine"]
    engine_state["scheduler"]["rng_state"] = _nested_tuple(
        engine_state["scheduler"]["rng_state"]
    )
    engine = _fresh_calibration_engine(trainer)
    engine.load_state_dict(engine_state)
    if engine.status != "complete":
        raise ValueError("targeted calibration artifact is incomplete")
    initialization_status = str(
        payload.get("initialization_status", "strict_stability_calibration")
    )
    strict_stability_check = str(payload.get("strict_stability_check", "passed"))
    approval_reason = payload.get("approval_reason")
    if set(engine.stability_by_N) != set(targets):
        raise ValueError("targeted calibration artifact lacks stability diagnostics")
    if initialization_status == "strict_stability_calibration":
        if strict_stability_check != "passed" or any(
            result.status != "stable" for result in engine.stability_by_N.values()
        ):
            raise ValueError("targeted calibration artifact is not stable")
    elif initialization_status == "high_variance_engineering_estimate":
        if strict_stability_check != "failed" or not approval_reason:
            raise ValueError("high-variance engineering artifact lacks explicit approval")
        if any(
            engine.valid_by_N[node_count] < engine.minimum_valid_samples
            for node_count in targets
        ):
            raise ValueError("high-variance engineering artifact lacks minimum samples")
    else:
        raise ValueError("unsupported targeted initialization status")
    medians = _integer_keys(payload["median_log_z_by_N"], "median_log_z_by_N")
    statistics = _integer_keys(
        payload["calibration_statistics_by_N"], "calibration_statistics_by_N"
    )
    stability = _integer_keys(
        payload["calibration_stability_by_N"], "calibration_stability_by_N"
    )
    if set(medians) != set(targets) or any(
        float(medians[node_count]) != engine.statistics_by_N[node_count].median
        for node_count in targets
    ):
        raise ValueError("targeted calibration medians disagree with engine state")
    assert trainer.tb_loss.learned_log_z_initialized_mask is not None
    if any(
        bool(trainer.tb_loss.learned_log_z_initialized_mask[node_count - 1])
        for node_count in targets
    ):
        raise RuntimeError("targeted logZ scalar is already initialized")
    record_payload = {
        "schema": TARGETED_LOG_Z_INITIALIZATION_SCHEMA,
        "source_artifact_schema": payload["schema"],
        "source_artifact_sha256": _file_sha256(target),
        "source_run_id": str(payload["source_run_id"]),
        "target_node_counts": targets,
        "median_log_z_by_N": {key: float(value) for key, value in medians.items()},
        "calibration_statistics_by_N": {
            key: dict(value) for key, value in statistics.items()
        },
        "calibration_stability_by_N": {
            key: dict(value) for key, value in stability.items()
        },
        "calibration_context_fingerprint": str(
            payload["calibration_context_fingerprint"]
        ),
        "artifact_fingerprint": str(fingerprint),
        "initialization_status": initialization_status,
        "strict_stability_check": strict_stability_check,
        "approval_reason": approval_reason,
        "reuse_scope": "initialization_constants_only",
        "restored_training_state": False,
    }
    record_payload["provenance_fingerprint"] = _canonical_fingerprint(record_payload)
    record = TargetedLogZInitialization(**record_payload)
    for node_count in targets:
        trainer.tb_loss.initialize_learned_log_z(node_count, medians[node_count])
    trainer.calibration = engine
    trainer.targeted_log_z_initialization = record
    return record


def targeted_log_z_initialization_from_manifest(
    value: Mapping[str, Any],
) -> TargetedLogZInitialization:
    payload = dict(value)
    payload.setdefault("initialization_status", "strict_stability_calibration")
    payload.setdefault("strict_stability_check", "passed")
    payload.setdefault("approval_reason", None)
    payload["target_node_counts"] = tuple(payload["target_node_counts"])
    for name in (
        "median_log_z_by_N",
        "calibration_statistics_by_N",
        "calibration_stability_by_N",
    ):
        payload[name] = _integer_keys(payload[name], name)
    return TargetedLogZInitialization(**payload)


__all__ = [
    "TARGETED_CALIBRATION_ARTIFACT_SCHEMA",
    "TARGETED_CALIBRATION_PROGRESS_SCHEMA",
    "TARGETED_LOG_Z_INITIALIZATION_SCHEMA",
    "TargetedLogZInitialization",
    "load_targeted_calibration_progress",
    "load_verified_targeted_calibration_artifact",
    "policy_state_fingerprint",
    "save_targeted_calibration_progress",
    "targeted_calibration_context",
    "targeted_log_z_initialization_from_manifest",
    "write_targeted_calibration_artifact",
    "write_high_variance_engineering_artifact_from_progress",
]
