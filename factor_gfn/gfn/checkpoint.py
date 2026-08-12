"""训练检查点、随机状态和运行元数据的原子持久化。"""

from __future__ import annotations

import json
import hashlib
import os
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from .config import TrainingStats

if TYPE_CHECKING:
    from .trainer import GFNTrainer


CHECKPOINT_SCHEMA = "factor_gfn.checkpoint.v1"


def _legacy_v3_config_fingerprint(trainer: "GFNTrainer") -> str | None:
    """Rebuild the pre-hierarchical flat-policy fingerprint when applicable."""

    if trainer.config.model.token_policy_mode != "flat":
        return None
    manifest = trainer.config.manifest()
    manifest["schema"] = "factor_gfn.gfn_config.v3"
    manifest["config"]["model"].pop("token_policy_mode", None)
    payload = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _legacy_v4_config_fingerprint(trainer: "GFNTrainer") -> str | None:
    """Rebuild the pre-grammar-hierarchy fingerprint for historical policies."""

    if trainer.config.model.token_policy_mode == "grammar_hierarchical":
        return None
    manifest = trainer.config.manifest()
    manifest["schema"] = "factor_gfn.gfn_config.v4"
    payload = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _cpu_byte_rng_state(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} RNG state 必须是 torch.Tensor")
    if value.dtype != torch.uint8:
        raise TypeError(f"{label} RNG state 必须是 torch.ByteTensor")
    return value.detach().cpu().contiguous()


def _restore_rng_state(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(
        _cpu_byte_rng_state(states["torch_cpu"], label="PyTorch CPU")
    )
    cuda_states = states.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("检查点包含 CUDA RNG 状态，但当前 CUDA 不可用")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("检查点 CUDA 设备数与当前环境不一致")
        torch.cuda.set_rng_state_all(
            [
                _cpu_byte_rng_state(state, label=f"PyTorch CUDA {index}")
                for index, state in enumerate(cuda_states)
            ]
        )


def save_checkpoint(path: str | os.PathLike[str], trainer: "GFNTrainer") -> None:
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
        "optimizer_state": trainer.optimizer.state_dict(),
        "step": trainer.step,
        "optimizer_step": trainer.optimizer_step,
        "history": [asdict(item) for item in trainer.history],
        "run_id": trainer.run_id,
        "created_at_utc": trainer.created_at_utc,
        "rng_state": _capture_rng_state(),
        "run_metadata": trainer.run_metadata(),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | os.PathLike[str],
    trainer: "GFNTrainer",
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    # RNG states must remain CPU ByteTensors. Model and optimizer load_state_dict
    # move their tensors to the trainer's target device after this CPU staging load.
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("检查点 schema 不兼容")
    accepted_config_fingerprints = {trainer.config.fingerprint()}
    legacy_v3_fingerprint = _legacy_v3_config_fingerprint(trainer)
    if legacy_v3_fingerprint is not None:
        accepted_config_fingerprints.add(legacy_v3_fingerprint)
    legacy_v4_fingerprint = _legacy_v4_config_fingerprint(trainer)
    if legacy_v4_fingerprint is not None:
        accepted_config_fingerprints.add(legacy_v4_fingerprint)
    if payload.get("config_fingerprint") not in accepted_config_fingerprints:
        raise ValueError("检查点与当前 GFNConfig 指纹不一致")
    if payload.get("reward_provider_fingerprint") != trainer.reward_provider.fingerprint():
        raise ValueError("检查点与当前 Reward Provider 指纹不一致")
    if payload.get("device_type") != trainer.device.type:
        raise ValueError("确定性恢复要求检查点与当前设备类型一致")
    saved_runtime = payload.get("run_metadata", {}).get("runtime", {})
    saved_cublas_config = saved_runtime.get("cublas_workspace_config")
    current_cublas_config = trainer.run_metadata()["runtime"][
        "cublas_workspace_config"
    ]
    if trainer.device.type == "cuda" and trainer.config.training.deterministic_algorithms:
        pristine_legacy_checkpoint = (
            saved_cublas_config is None
            and int(payload.get("step", -1)) == 0
            and int(payload.get("optimizer_step", -1)) == 0
            and not payload.get("history")
        )
        if not pristine_legacy_checkpoint and saved_cublas_config != current_cublas_config:
            raise ValueError(
                "确定性恢复要求检查点与当前 CUBLAS_WORKSPACE_CONFIG 一致"
            )

    trainer.model.load_state_dict(payload["model_state"], strict=True)
    trainer.tb_loss.load_state_dict(payload["tb_loss_state"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.step = int(payload["step"])
    trainer.optimizer_step = int(payload["optimizer_step"])
    trainer.history = [TrainingStats(**item) for item in payload["history"]]
    trainer.run_id = str(payload["run_id"])
    trainer.created_at_utc = str(payload["created_at_utc"])
    _restore_rng_state(payload["rng_state"])
    return dict(payload["run_metadata"])


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


__all__ = ["CHECKPOINT_SCHEMA", "load_checkpoint", "save_checkpoint", "write_run_metadata"]
