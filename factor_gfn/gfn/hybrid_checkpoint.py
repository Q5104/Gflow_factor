"""Distinct atomic checkpoints for Stage 5 hybrid-variance training."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import random
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import torch

from .complexity_scheduler import BalancedNodeCountScheduler
from .hybrid_config import HybridVarianceGFNConfig
from .hybrid_trainer import (
    HybridExactTBDiagnostics,
    HybridLPVDiagnostics,
    HybridUpdateDiagnostics,
)

if TYPE_CHECKING:
    from .hybrid_trainer import HybridVarianceTrainer


HYBRID_CHECKPOINT_SCHEMA = "factor_gfn.checkpoint.hybrid_variance.v1"
HYBRID_OBJECTIVE_MODE = "hybrid_variance"


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _cpu_byte_rng_state(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise TypeError(f"{label} RNG state must be a torch.ByteTensor")
    return value.detach().cpu().contiguous()


def _validate_rng_state(states: Any) -> dict[str, Any]:
    if not isinstance(states, dict):
        raise ValueError("hybrid checkpoint RNG state must be a dictionary")
    for key in ("python", "numpy", "torch_cpu", "torch_cuda"):
        if key not in states:
            raise ValueError(f"hybrid checkpoint lacks {key} RNG state")
    _cpu_byte_rng_state(states["torch_cpu"], label="torch CPU")
    cuda_states = states["torch_cuda"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "hybrid checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError("torch CUDA RNG state must be a sequence")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("hybrid checkpoint CUDA device count differs from runtime")
        for index, value in enumerate(cuda_states):
            _cpu_byte_rng_state(value, label=f"torch CUDA {index}")
    return states


def _restore_rng_state(states: dict[str, Any]) -> None:
    validated = _validate_rng_state(states)
    random.setstate(validated["python"])
    np.random.set_state(validated["numpy"])
    torch.set_rng_state(
        _cpu_byte_rng_state(validated["torch_cpu"], label="torch CPU")
    )
    cuda_states = validated["torch_cuda"]
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(
            [
                _cpu_byte_rng_state(value, label=f"torch CUDA {index}")
                for index, value in enumerate(cuda_states)
            ]
        )


def _proof_manifest(trainer: "HybridVarianceTrainer") -> dict[int, dict[str, Any]]:
    return {
        int(node_count): asdict(proof)
        for node_count, proof in trainer.exhaustive_reuse_proofs_by_N.items()
    }


def _exact_mass_manifest(
    trainer: "HybridVarianceTrainer",
) -> dict[int, dict[str, Any]]:
    return {
        int(node_count): asdict(result)
        for node_count, result in trainer.registered_exact_masses_by_N.items()
    }


def _normalize_int_key_manifest(value: Any, *, label: str) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"hybrid checkpoint {label} must be a mapping")
    try:
        return {int(key): nested for key, nested in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"hybrid checkpoint {label} has invalid keys") from error


def _diagnostic_from_manifest(value: Any) -> HybridUpdateDiagnostics:
    if not isinstance(value, dict):
        raise ValueError("hybrid checkpoint diagnostic history item must be a dict")
    objective_kind = value.get("objective_kind")
    try:
        if objective_kind == "exact_tb":
            return HybridExactTBDiagnostics(**value)
        if objective_kind == "log_partition_variance":
            return HybridLPVDiagnostics(**value)
    except TypeError as error:
        raise ValueError("hybrid checkpoint diagnostic fields are incompatible") from error
    raise ValueError("hybrid checkpoint diagnostic objective_kind is incompatible")


def _diagnostic_counters(
    history: list[HybridUpdateDiagnostics],
) -> dict[str, int]:
    return {
        "successful_updates": len(history),
        "requested_count": sum(item.requested_count for item in history),
        "accepted_count": sum(item.accepted_count for item in history),
        "invalid_count": sum(item.invalid_count for item in history),
        "retry_count": sum(item.retry_count for item in history),
        "retry_exhausted_count": sum(
            item.retry_exhausted_count for item in history
        ),
    }


def _require_ready(trainer: "HybridVarianceTrainer") -> None:
    if (
        not isinstance(trainer.config, HybridVarianceGFNConfig)
        or not trainer.hybrid_mode
    ):
        raise RuntimeError("hybrid checkpoint requires HybridVarianceTrainer")
    expected = set(trainer.config.objective.exact_tb_node_counts)
    if set(trainer.registered_exact_masses_by_N) != expected:
        raise RuntimeError("hybrid checkpoint requires exact masses for N=1/2")
    if set(trainer.exhaustive_reuse_proofs_by_N) != expected:
        raise RuntimeError("hybrid checkpoint requires reuse proofs for N=1/2")
    if set(trainer.exhaustive_reward_lookups_by_N) != expected:
        raise RuntimeError("hybrid checkpoint requires Reward lookups for N=1/2")
    if len(trainer.diagnostic_history) != trainer.optimizer_step:
        raise RuntimeError("hybrid diagnostic history must match optimizer steps")
    expected_trajectories = (
        trainer.optimizer_step * trainer.config.training.trajectories_per_batch
    )
    if trainer.total_trajectories_seen != expected_trajectories:
        raise RuntimeError("hybrid trajectory counter is inconsistent")
    for expected_step, item in enumerate(trainer.diagnostic_history, start=1):
        if item.global_optimizer_step != expected_step:
            raise RuntimeError("hybrid diagnostic optimizer steps are not contiguous")
        if item.total_trajectories_seen != (
            expected_step * trainer.config.training.trajectories_per_batch
        ):
            raise RuntimeError("hybrid diagnostic trajectory totals are inconsistent")


def save_hybrid_checkpoint(
    path: str | os.PathLike[str],
    trainer: "HybridVarianceTrainer",
) -> None:
    """Atomically persist one fully committed hybrid update boundary."""

    _require_ready(trainer)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "schema": HYBRID_CHECKPOINT_SCHEMA,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_mode": HYBRID_OBJECTIVE_MODE,
        "config_fingerprint": trainer.config.fingerprint(),
        "reward_provider_fingerprint": trainer.reward_provider.fingerprint(),
        "device_type": trainer.device.type,
        "model_state": trainer.model.state_dict(),
        "optimizer_contract": trainer.optimizer_contract(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "fixed_exact_state": trainer.exact_tb_loss.state_dict(),
        "fixed_exact_manifest": trainer.exact_tb_loss.fixed_state_manifest(),
        "exhaustive_registry_equivalence": _proof_manifest(trainer),
        "exact_mass_manifest": _exact_mass_manifest(trainer),
        "scheduler_state": trainer.complexity_scheduler.state_dict(),
        "global_optimizer_step": trainer.optimizer_step,
        "total_trajectories_seen": trainer.total_trajectories_seen,
        "diagnostic_history": [
            item.to_dict() for item in trainer.diagnostic_history
        ],
        "diagnostic_counters": _diagnostic_counters(
            trainer.diagnostic_history
        ),
        "rng_state": _capture_rng_state(),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_hybrid_checkpoint(
    path: str | os.PathLike[str],
    trainer: "HybridVarianceTrainer",
) -> dict[str, Any]:
    """Load only the matching hybrid schema into a registry-configured Trainer."""

    _require_ready(trainer)
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != HYBRID_CHECKPOINT_SCHEMA:
        raise ValueError("HybridVarianceTrainer rejects legacy checkpoint schemas")
    if payload.get("objective_mode") != HYBRID_OBJECTIVE_MODE:
        raise ValueError("hybrid checkpoint objective mode is incompatible")
    if payload.get("config_fingerprint") != trainer.config.fingerprint():
        raise ValueError("hybrid checkpoint config fingerprint mismatch")
    if (
        payload.get("reward_provider_fingerprint")
        != trainer.reward_provider.fingerprint()
    ):
        raise ValueError("hybrid checkpoint Reward provider fingerprint mismatch")
    if payload.get("device_type") != trainer.device.type:
        raise ValueError("deterministic hybrid resume requires the same device type")
    if payload.get("optimizer_contract") != trainer.optimizer_contract():
        raise ValueError("hybrid checkpoint policy optimizer contract mismatch")

    saved_proofs = _normalize_int_key_manifest(
        payload.get("exhaustive_registry_equivalence"),
        label="exhaustive registry proof manifest",
    )
    if saved_proofs != _proof_manifest(trainer):
        raise ValueError("hybrid checkpoint/current reuse proofs differ")
    saved_masses = _normalize_int_key_manifest(
        payload.get("exact_mass_manifest"),
        label="exact mass manifest",
    )
    if saved_masses != _exact_mass_manifest(trainer):
        raise ValueError("hybrid checkpoint/current exact masses differ")
    if payload.get("fixed_exact_manifest") != (
        trainer.exact_tb_loss.fixed_state_manifest()
    ):
        raise ValueError("hybrid checkpoint/current fixed exact buffers differ")

    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError("hybrid checkpoint optimizer state is invalid")
    parameter_groups = optimizer_state.get("param_groups")
    if (
        not isinstance(parameter_groups, list)
        or len(parameter_groups) != 1
        or parameter_groups[0].get("name") != "policy"
    ):
        raise ValueError("hybrid checkpoint must contain one policy parameter group")

    raw_history = payload.get("diagnostic_history")
    if not isinstance(raw_history, list):
        raise ValueError("hybrid checkpoint diagnostic history is invalid")
    history = [_diagnostic_from_manifest(item) for item in raw_history]
    optimizer_step = int(payload.get("global_optimizer_step", -1))
    total_trajectories_seen = int(payload.get("total_trajectories_seen", -1))
    if optimizer_step < 0 or len(history) != optimizer_step:
        raise ValueError("hybrid checkpoint optimizer/history counters differ")
    expected_trajectories = (
        optimizer_step * trainer.config.training.trajectories_per_batch
    )
    if total_trajectories_seen != expected_trajectories:
        raise ValueError("hybrid checkpoint trajectory counter is inconsistent")
    if payload.get("diagnostic_counters") != _diagnostic_counters(history):
        raise ValueError("hybrid checkpoint diagnostic counters are inconsistent")
    for expected_step, item in enumerate(history, start=1):
        if item.global_optimizer_step != expected_step:
            raise ValueError("hybrid checkpoint diagnostic steps are not contiguous")
        if item.total_trajectories_seen != (
            expected_step * trainer.config.training.trajectories_per_batch
        ):
            raise ValueError("hybrid checkpoint diagnostic totals are inconsistent")

    scheduler_state = payload.get("scheduler_state")
    candidate_scheduler = BalancedNodeCountScheduler(
        trainer.config.resolved_condition_node_counts,
        seed=trainer.config.training.seed,
    )
    candidate_scheduler.load_state_dict(scheduler_state)
    rng_state = _validate_rng_state(payload.get("rng_state"))

    trainer.model.load_state_dict(payload["model_state"], strict=True)
    trainer.exact_tb_loss.load_state_dict(payload["fixed_exact_state"], strict=True)
    trainer.optimizer.load_state_dict(optimizer_state)
    trainer.complexity_scheduler.load_state_dict(scheduler_state)
    trainer.optimizer_step = optimizer_step
    trainer.total_trajectories_seen = total_trajectories_seen
    trainer.diagnostic_history = history
    _restore_rng_state(rng_state)
    return {
        "schema": payload["schema"],
        "saved_at_utc": payload["saved_at_utc"],
        "objective_mode": payload["objective_mode"],
        "config_fingerprint": payload["config_fingerprint"],
        "global_optimizer_step": optimizer_step,
        "total_trajectories_seen": total_trajectories_seen,
    }


__all__ = [
    "HYBRID_CHECKPOINT_SCHEMA",
    "HYBRID_OBJECTIVE_MODE",
    "load_hybrid_checkpoint",
    "save_hybrid_checkpoint",
]
