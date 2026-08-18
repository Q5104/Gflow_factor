"""No-anchor checkpoints plus strict read-only Stage 4 legacy loading."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from .config import GFNConfig, TrainingStats
from .no_anchor_config import NoAnchorGFNConfig
from .historical_log_z import historical_log_z_initialization_from_manifest
from .targeted_calibration import targeted_log_z_initialization_from_manifest

if TYPE_CHECKING:
    from .trainer import GFNTrainer


CHECKPOINT_SCHEMA = "factor_gfn.checkpoint.no_anchor.v1"
LEGACY_CHECKPOINT_SCHEMAS = frozenset(
    f"factor_gfn.checkpoint.v{version}" for version in range(1, 6)
)
_LEGACY_ANCHOR_TOP_LEVEL_FIELDS = frozenset(
    {"anchor_state", "anchor_optimizer_step", "total_policy_optimizer_step"}
)
_LEGACY_ANCHOR_HISTORY_FIELDS = frozenset(
    {"anchor_optimizer_step", "total_policy_optimizer_step", "anchor_loss"}
)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _cpu_byte_rng_state(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise TypeError(f"{label} RNG state must be a torch.ByteTensor")
    return value.detach().cpu().contiguous()


def _restore_rng_state(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(_cpu_byte_rng_state(states["torch_cpu"], label="torch CPU"))
    cuda_states = states.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA device count differs from runtime")
        torch.cuda.set_rng_state_all(
            [
                _cpu_byte_rng_state(value, label=f"torch CUDA {index}")
                for index, value in enumerate(cuda_states)
            ]
        )


def _proof_manifest(trainer: "GFNTrainer") -> dict[int, dict[str, Any]]:
    return {
        node_count: asdict(proof)
        for node_count, proof in trainer.exhaustive_reuse_proofs_by_N.items()
    }


def _reject_anchor_payload(payload: dict[str, Any]) -> None:
    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_text = str(key)
                nested_path = f"{path}.{key_text}" if path else key_text
                if "anchor" in key_text.lower():
                    raise ValueError(
                        f"no-anchor checkpoint rejects anchor fields: {nested_path}"
                    )
                visit(nested, nested_path)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(payload, "")


def save_checkpoint(path: str | os.PathLike[str], trainer: "GFNTrainer") -> None:
    if not isinstance(trainer.config, NoAnchorGFNConfig) or not trainer.no_anchor_mode:
        raise RuntimeError("legacy Trainer checkpoint writes are disabled (read-only)")
    trainer._require_no_anchor_exact_ready()
    trainer._require_no_anchor_normalizer_initialized()
    if set(trainer.exhaustive_reuse_proofs_by_N) != set(
        trainer.resolved_exhaustive_node_counts
    ):
        raise RuntimeError("no-anchor checkpoint requires equivalence proof for every E")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": trainer.config.fingerprint(),
        "reward_provider_fingerprint": trainer.reward_provider.fingerprint(),
        "device_type": trainer.device.type,
        "model_state": trainer.model.state_dict(),
        "tb_loss_state": trainer.tb_loss.state_dict(),
        "normalizer_manifest": trainer.tb_loss.normalizer_manifest(),
        "optimizer_contract": trainer.optimizer_contract(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "step": trainer.step,
        "optimizer_step": trainer.optimizer_step,
        "history": [asdict(item) for item in trainer.history],
        "run_id": trainer.run_id,
        "created_at_utc": trainer.created_at_utc,
        "rng_state": _capture_rng_state(),
        "run_metadata": trainer.run_metadata(),
        "complexity_state": trainer.complexity_state_dict(),
        "calibration_state": trainer.calibration_state_dict(),
        "historical_log_z_initialization": (
            None
            if trainer.historical_log_z_initialization is None
            else trainer.historical_log_z_initialization.manifest()
        ),
        "targeted_log_z_initialization": (
            None
            if trainer.targeted_log_z_initialization is None
            else trainer.targeted_log_z_initialization.manifest()
        ),
        "exhaustive_registry_equivalence": _proof_manifest(trainer),
    }
    _reject_anchor_payload(payload)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _legacy_config_fingerprints(trainer: "GFNTrainer") -> set[str]:
    manifests: list[dict[str, Any]] = []
    current = trainer.config.manifest()
    manifests.append(current)

    v9 = trainer.config.manifest()
    v9["schema"] = "factor_gfn.gfn_config.v9"
    v9["config"]["exhaustive_anchors"] = {
        "enabled": False,
        "frequency": 0,
        "batch_size": 8,
        "seed_offset": 1,
    }
    manifests.append(v9)

    v8 = trainer.config.manifest()
    v8["schema"] = "factor_gfn.gfn_config.v8"
    manifests.append(v8)

    v7 = trainer.config.manifest()
    v7["schema"] = "factor_gfn.gfn_config.v7"
    v7["config"].pop("calibration", None)
    manifests.append(v7)

    stage4 = trainer.config.manifest()
    stage4["schema"] = "factor_gfn.gfn_config.v5"
    stage4["config"].pop("complexity_scheduler", None)
    stage4["config"].pop("calibration", None)
    stage4["state_adapter"] = {
        "schema": "factor_gfn.state_adapter.v1",
        "state_source": "canonical_partial_ast_not_action_history",
        "node_embeddings": ["category", "operator_or_feature", "window"],
        "manual_features": [
            "max(filled_node_depth, open_slot_depth)/max_depth",
            "operator_count/max_nodes",
            "node_count/max_nodes",
        ],
        "legal_mask_source": "GrammarState.legal_transitions",
    }
    manifests.append(stage4)
    stage4_v4 = json.loads(json.dumps(stage4))
    stage4_v4["schema"] = "factor_gfn.gfn_config.v4"
    manifests.append(stage4_v4)
    if trainer.config.model.token_policy_mode == "flat":
        stage4_v3 = json.loads(json.dumps(stage4))
        stage4_v3["schema"] = "factor_gfn.gfn_config.v3"
        stage4_v3["config"]["model"].pop("token_policy_mode", None)
        manifests.append(stage4_v3)

    values = set()
    for manifest in manifests:
        encoded = json.dumps(
            manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        values.add(hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    return values


def _legacy_reward_provider_fingerprints(trainer: "GFNTrainer") -> set[str]:
    values = {trainer.reward_provider.fingerprint()}
    manifest = trainer.reward_provider.manifest()
    schema = manifest.get("schema")
    if schema == "factor_gfn.real_reward_provider.v8":
        manifest["schema"] = "factor_gfn.real_reward_provider.v7"
    elif schema == "factor_gfn.synthetic_reward.v2":
        manifest["schema"] = "factor_gfn.synthetic_reward.v1"
        manifest.pop("context_fingerprint", None)
    else:
        return values
    manifest.pop("data_scope", None)
    manifest.pop("validation_oos_loaded", None)
    encoded = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    values.add(hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    return values


def _load_legacy_stage4(payload: dict[str, Any], trainer: "GFNTrainer") -> dict[str, Any]:
    if not isinstance(trainer.config, GFNConfig) or trainer.no_anchor_mode:
        raise ValueError("legacy checkpoint can only load into legacy GFNConfig")
    if trainer.complexity_scheduler is not None:
        raise ValueError("legacy checkpoint compatibility is limited to non-conditioned Stage 4")
    if trainer.tb_loss.normalizer_mode != "legacy_scalar":
        raise ValueError("legacy Stage 4 compatibility requires scalar logZ")
    anchor_state = payload.get("anchor_state")
    anchor_step = int(payload.get("anchor_optimizer_step", 0))
    if anchor_state is not None or anchor_step != 0:
        raise ValueError("factor_gfn.checkpoint.v5 anchor checkpoint is rejected")
    for item in payload.get("history", ()):
        if int(item.get("anchor_optimizer_step", 0)) != 0:
            raise ValueError("legacy history contains anchor optimizer updates")
        if item.get("anchor_loss") is not None:
            raise ValueError("legacy history contains anchor loss")
    if payload.get("config_fingerprint") not in _legacy_config_fingerprints(trainer):
        raise ValueError("legacy checkpoint config fingerprint mismatch")
    if (
        payload.get("reward_provider_fingerprint")
        not in _legacy_reward_provider_fingerprints(trainer)
    ):
        raise ValueError("legacy checkpoint Reward provider fingerprint mismatch")
    if payload.get("device_type") != trainer.device.type:
        raise ValueError("legacy deterministic load requires the same device type")
    model_state = dict(payload["model_state"])
    condition_key = "condition_projection.weight"
    if condition_key not in model_state:
        model_state[condition_key] = torch.zeros_like(
            trainer.model.state_dict()[condition_key]
        )
    trainer.model.load_state_dict(model_state, strict=True)
    trainer.tb_loss.load_state_dict(payload["tb_loss_state"], strict=True)
    optimizer_state = payload["optimizer_state"]
    saved_policy_ids = optimizer_state["param_groups"][0]["params"]
    current_template = trainer.optimizer.state_dict()
    current_policy_ids = current_template["param_groups"][0]["params"]
    if len(saved_policy_ids) == len(current_policy_ids) - 1:
        names = [name for name, _ in trainer.model.named_parameters()]
        condition_index = names.index(condition_key)
        transformed = {"state": {}, "param_groups": []}
        for saved_id, current_id in zip(
            saved_policy_ids,
            current_policy_ids[:condition_index] + current_policy_ids[condition_index + 1 :],
            strict=True,
        ):
            if saved_id in optimizer_state["state"]:
                transformed["state"][current_id] = optimizer_state["state"][saved_id]
        for group_index, (saved_group, current_group) in enumerate(
            zip(optimizer_state["param_groups"], current_template["param_groups"], strict=True)
        ):
            group = dict(saved_group)
            group["params"] = current_group["params"]
            transformed["param_groups"].append(group)
            if group_index == 1:
                for saved_id, current_id in zip(
                    saved_group["params"], current_group["params"], strict=True
                ):
                    if saved_id in optimizer_state["state"]:
                        transformed["state"][current_id] = optimizer_state["state"][saved_id]
        optimizer_state = transformed
    trainer.optimizer.load_state_dict(optimizer_state)
    trainer.step = int(payload["step"])
    trainer.optimizer_step = int(payload["optimizer_step"])
    cleaned_history = []
    for item in payload.get("history", ()):
        cleaned = dict(item)
        for name in _LEGACY_ANCHOR_HISTORY_FIELDS:
            cleaned.pop(name, None)
        cleaned_history.append(TrainingStats(**cleaned))
    trainer.history = cleaned_history
    trainer.run_id = str(payload["run_id"])
    trainer.created_at_utc = str(payload["created_at_utc"])
    _restore_rng_state(payload["rng_state"])
    return dict(payload.get("run_metadata", {}))


def _load_no_anchor(payload: dict[str, Any], trainer: "GFNTrainer") -> dict[str, Any]:
    if not isinstance(trainer.config, NoAnchorGFNConfig) or not trainer.no_anchor_mode:
        raise ValueError("no-anchor checkpoint requires NoAnchorGFNConfig")
    _reject_anchor_payload(payload)
    required_proofs = _proof_manifest(trainer)
    saved_proofs = {
        int(key): value
        for key, value in payload.get("exhaustive_registry_equivalence", {}).items()
    }
    if not required_proofs or saved_proofs != required_proofs:
        raise ValueError("checkpoint/current exhaustive equivalence proofs differ")
    if payload.get("config_fingerprint") != trainer.config.fingerprint():
        raise ValueError("checkpoint NoAnchorGFNConfig fingerprint mismatch")
    if payload.get("reward_provider_fingerprint") != trainer.reward_provider.fingerprint():
        raise ValueError("checkpoint Reward provider fingerprint mismatch")
    if payload.get("device_type") != trainer.device.type:
        raise ValueError("deterministic resume requires the same device type")
    if payload.get("normalizer_manifest") != trainer.tb_loss.normalizer_manifest():
        raise ValueError("checkpoint normalizer vector/exact strata mismatch")
    saved_optimizer_contract = payload.get("optimizer_contract")
    if saved_optimizer_contract is None:
        saved_optimizer_contract = {
            **trainer.optimizer_contract(),
            "normalizer_optimizer": "adam",
            "normalizer_learning_rate": trainer.config.training.log_z_learning_rate,
            "normalizer_momentum": None,
            "normalizer_active_indices_only": False,
        }
    if saved_optimizer_contract != trainer.optimizer_contract():
        raise ValueError("checkpoint optimizer contract mismatch")
    if payload["normalizer_manifest"].get("mode") != "conditional_vector":
        raise ValueError("no-anchor checkpoint rejects legacy scalar logZ")
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    trainer.tb_loss.load_state_dict(payload["tb_loss_state"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.step = int(payload["step"])
    trainer.optimizer_step = int(payload["optimizer_step"])
    trainer.history = [TrainingStats(**item) for item in payload["history"]]
    trainer.run_id = str(payload["run_id"])
    trainer.created_at_utc = str(payload["created_at_utc"])
    trainer.load_complexity_state_dict(payload["complexity_state"])
    trainer.load_calibration_state_dict(payload["calibration_state"])
    historical = payload.get("historical_log_z_initialization")
    trainer.historical_log_z_initialization = (
        None
        if historical is None
        else historical_log_z_initialization_from_manifest(historical)
    )
    if trainer.historical_log_z_initialization is not None:
        if (
            trainer.historical_log_z_initialization.semantics
            != trainer.target_exhaustive_reuse_semantics()
        ):
            raise ValueError("checkpoint historical logZ semantics mismatch")
    targeted = payload.get("targeted_log_z_initialization")
    trainer.targeted_log_z_initialization = (
        None
        if targeted is None
        else targeted_log_z_initialization_from_manifest(targeted)
    )
    if trainer.targeted_log_z_initialization is not None:
        expected_targets = tuple(trainer.config.calibration.target_node_counts)
        if trainer.targeted_log_z_initialization.target_node_counts != expected_targets:
            raise ValueError("checkpoint targeted logZ strata mismatch")
    trainer._require_no_anchor_normalizer_initialized()
    _restore_rng_state(payload["rng_state"])
    return dict(payload["run_metadata"])


def load_checkpoint(
    path: str | os.PathLike[str],
    trainer: "GFNTrainer",
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    schema = payload.get("schema")
    if trainer.no_anchor_mode:
        if schema != CHECKPOINT_SCHEMA:
            raise ValueError(
                "no-anchor Trainer rejects legacy v1-v5 and scalar checkpoints"
            )
        return _load_no_anchor(payload, trainer)
    if schema == CHECKPOINT_SCHEMA:
        raise ValueError("legacy Trainer cannot load no-anchor checkpoint")
    if schema not in LEGACY_CHECKPOINT_SCHEMAS:
        raise ValueError("checkpoint schema is incompatible")
    return _load_legacy_stage4(payload, trainer)


def write_run_metadata(
    path: str | os.PathLike[str],
    trainer: "GFNTrainer",
) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(trainer.run_metadata(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "LEGACY_CHECKPOINT_SCHEMAS",
    "load_checkpoint",
    "save_checkpoint",
    "write_run_metadata",
]
