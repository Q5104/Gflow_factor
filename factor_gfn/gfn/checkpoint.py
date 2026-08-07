"""训练检查点、随机状态和运行元数据的原子持久化。"""

from __future__ import annotations

import json
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


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"].cpu())
    cuda_states = states.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("检查点包含 CUDA RNG 状态，但当前 CUDA 不可用")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("检查点 CUDA 设备数与当前环境不一致")
        torch.cuda.set_rng_state_all(cuda_states)


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
    payload = torch.load(source, map_location=trainer.device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("检查点 schema 不兼容")
    if payload.get("config_fingerprint") != trainer.config.fingerprint():
        raise ValueError("检查点与当前 GFNConfig 指纹不一致")
    if payload.get("reward_provider_fingerprint") != trainer.reward_provider.fingerprint():
        raise ValueError("检查点与当前 Reward Provider 指纹不一致")
    if payload.get("device_type") != trainer.device.type:
        raise ValueError("确定性恢复要求检查点与当前设备类型一致")

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
