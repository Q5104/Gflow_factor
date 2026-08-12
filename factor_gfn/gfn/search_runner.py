"""GPU-only stage-five real candidate search orchestration.

The runner keeps real-data evaluation on the existing NumPy path while the
policy, TB loss, and optimizer live on CUDA. It never starts work at import
time; long runs remain explicit user actions through the stage-five notebook.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch

from factor_gfn.evaluator import IndustryNeutralizationWarning
from factor_gfn.grammar import Expression

from .config import (
    GFNConfig,
    STAGE5_TOKEN_POLICY_MODE,
    TrainingStats,
    build_stage5_real_training_config,
)
from .real_data import (
    RealRewardDataConfig,
    RealRewardDataPaths,
    build_real_reward_data_context,
)
from .real_reward import DEFAULT_SUBEXPRESSION_CACHE_MAX_BYTES, RealRewardProvider
from .trainer import (
    GFNTrainer,
    RewardAssignment,
    RewardProvider,
    configure_cuda_determinism,
)


REAL_SEARCH_SCHEMA = "factor_gfn.real_search.v2"
REAL_SEARCH_STATS_SCHEMA = "factor_gfn.real_search_stats.v3"
REAL_SEARCH_STATE_SCHEMA = "factor_gfn.real_search_state.v2"
REAL_SEARCH_MANIFEST_SCHEMA = "factor_gfn.real_search_manifest.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload: Any) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                _json_safe(payload),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        lines = [
            json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False)
            for record in records
        ]
        temporary.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 文件无法解析：{path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON 文件顶层必须是对象：{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSONL 行无法解析：{path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 行必须是对象：{path}:{line_number}")
        result.append(value)
    return result


@dataclass(frozen=True, slots=True)
class RealSearchSettings:
    """Run identity and persistence settings; target step is supplied per call."""

    max_steps: int
    seed: int = 42
    checkpoint_interval: int = 10
    device: str = "cuda:0"
    cache_max_entries: int = 50_000
    subexpression_cache_max_bytes: int = DEFAULT_SUBEXPRESSION_CACHE_MAX_BYTES
    tensorboard_enabled: bool = True
    console_progress: bool = True
    run_root: Path = Path("runs/real_search")

    def __post_init__(self) -> None:
        for name in ("max_steps", "checkpoint_interval", "cache_max_entries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if (
            isinstance(self.subexpression_cache_max_bytes, bool)
            or not isinstance(self.subexpression_cache_max_bytes, int)
            or self.subexpression_cache_max_bytes < 0
        ):
            raise ValueError("subexpression_cache_max_bytes 必须是非负整数")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed 必须是整数")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("阶段 5 正式搜索只允许 cuda 或 cuda:<index>")
        object.__setattr__(self, "device", str(device))
        object.__setattr__(self, "run_root", Path(self.run_root).resolve())
        for name in ("tensorboard_enabled", "console_progress"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须是布尔值")

    def manifest(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "seed": self.seed,
            "checkpoint_interval": self.checkpoint_interval,
            "device": self.device,
            "cache_max_entries": self.cache_max_entries,
            "subexpression_cache_max_bytes": self.subexpression_cache_max_bytes,
            "tensorboard_enabled": self.tensorboard_enabled,
            "console_progress": self.console_progress,
            "run_root": str(self.run_root),
        }

    @classmethod
    def from_manifest(cls, value: dict[str, Any]) -> "RealSearchSettings":
        return cls(
            max_steps=int(value["max_steps"]),
            seed=int(value["seed"]),
            checkpoint_interval=int(value["checkpoint_interval"]),
            device=str(value["device"]),
            cache_max_entries=int(value["cache_max_entries"]),
            subexpression_cache_max_bytes=int(
                value.get(
                    "subexpression_cache_max_bytes",
                    DEFAULT_SUBEXPRESSION_CACHE_MAX_BYTES,
                )
            ),
            tensorboard_enabled=bool(value.get("tensorboard_enabled", True)),
            console_progress=bool(value.get("console_progress", True)),
            run_root=Path(value["run_root"]),
        )


def require_cuda_device(device: str | torch.device) -> tuple[torch.device, dict[str, Any]]:
    """Resolve one CUDA device and fail instead of silently falling back to CPU."""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("阶段 5 正式搜索必须显式使用 CUDA")
    cublas_workspace_config = configure_cuda_determinism(
        resolved, deterministic_algorithms=True
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；正式搜索禁止静默回退到 CPU")
    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
        resolved = torch.device("cuda", index)
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA 设备索引越界：{index}，当前设备数={torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(index)
    environment = {
        "device": str(resolved),
        "device_index": index,
        "device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": cublas_workspace_config,
    }
    return resolved, environment


def _validate_training_data_config(config: RealRewardDataConfig) -> None:
    if config.train_start != "2010-01-01" or config.train_end != "2018-12-31":
        raise ValueError("阶段 5 Reward 训练区间必须固定为 2010-01-01 至 2018-12-31")


class RecordingRewardProvider(RewardProvider):
    """Record every request in the current logical step, including cache hits."""

    def __init__(self, delegate: RewardProvider) -> None:
        self.delegate = delegate
        self._logical_step: int | None = None
        self._request_index = 0
        self._pending: list[dict[str, Any]] = []

    def manifest(self) -> dict[str, Any]:
        return self.delegate.manifest()

    def fingerprint(self) -> str:
        return self.delegate.fingerprint()

    def set_request_index(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("request index 必须是非负整数")
        self._request_index = value

    def begin_step(self, logical_step: int) -> None:
        if self._logical_step is not None or self._pending:
            raise RuntimeError("上一个逻辑步骤尚未结束")
        if (
            isinstance(logical_step, bool)
            or not isinstance(logical_step, int)
            or logical_step < 1
        ):
            raise ValueError("logical_step 必须是正整数")
        self._logical_step = logical_step

    def finish_step(self) -> list[dict[str, Any]]:
        if self._logical_step is None:
            raise RuntimeError("当前没有正在记录的逻辑步骤")
        records = self._pending
        self._pending = []
        self._logical_step = None
        return records

    def abort_step(self) -> None:
        self._pending = []
        self._logical_step = None

    def evaluate(self, expression: Expression) -> RewardAssignment:
        if self._logical_step is None:
            raise RuntimeError("Reward 请求必须位于 begin_step()/finish_step() 之间")
        assignment = self.delegate.evaluate(expression)
        metadata = deepcopy(assignment.metadata) if assignment.metadata else {}
        self._request_index += 1
        self._pending.append(
            {
                "request_index": self._request_index,
                "branch": "main",
                "phase": f"train_step_{self._logical_step}",
                "logical_step": self._logical_step,
                "formula": expression.to_formula(),
                "structural_hash": expression.structural_hash(),
                "prefix_token_ids": list(expression.to_prefix()),
                "node_count": expression.stats.node_count,
                "depth": expression.stats.depth,
                "valid": assignment.valid,
                "reward": assignment.reward,
                "log_reward": assignment.log_reward,
                "rejection_reason": assignment.rejection_reason,
                "provider_cache_hit": metadata.get("provider_cache_hit"),
                "metadata": metadata,
            }
        )
        return assignment


@dataclass(frozen=True, slots=True)
class SearchStepMetrics:
    step: int
    optimizer_step: int
    wall_seconds: float
    reward_requests: int
    valid_reward_requests: int
    unique_expressions: int
    factor_seconds: float
    reward_seconds: float
    provider_cache_hits: int
    subexpression_cache_hits: int
    subexpression_cache_misses: int
    subexpression_cache_evictions: int
    gpu_memory_allocated_bytes: int
    gpu_memory_reserved_bytes: int
    gpu_peak_memory_allocated_bytes: int
    gpu_peak_memory_reserved_bytes: int
    sampling_seconds: float = 0.0
    reward_provider_seconds: float = 0.0
    training_update_seconds: float = 0.0
    tb_loss_forward_cuda_seconds: float | None = None
    backward_cuda_seconds: float | None = None
    optimizer_cuda_seconds: float | None = None


class RealSearchRunner:
    """One new or resumed stage-five run with atomic, step-aligned artifacts."""

    def __init__(
        self,
        *,
        settings: RealSearchSettings,
        config: GFNConfig,
        trainer: GFNTrainer,
        recording_provider: RecordingRewardProvider,
        run_dir: Path,
        cuda_environment: dict[str, Any],
        evaluation_records: list[dict[str, Any]] | None = None,
        step_metrics: list[dict[str, Any]] | None = None,
    ) -> None:
        if trainer.device.type != "cuda":
            raise ValueError("RealSearchRunner 拒绝非 CUDA Trainer")
        self.settings = settings
        self.config = config
        self.trainer = trainer
        self.recording_provider = recording_provider
        self.run_dir = run_dir.resolve()
        self.cuda_environment = deepcopy(cuda_environment)
        self.evaluation_records = list(evaluation_records or [])
        self.step_metrics = list(step_metrics or [])
        self.recording_provider.set_request_index(len(self.evaluation_records))

    @property
    def evaluations_path(self) -> Path:
        return self.run_dir / "evaluations.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "step_metrics.jsonl"

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoint_latest.pt"

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"

    def _archive_checkpoint_path(self, step: int) -> Path:
        return self.run_dir / "checkpoints" / f"checkpoint_step_{step:08d}.pt"

    def _persist_tables(self) -> None:
        _atomic_write_jsonl(self.evaluations_path, self.evaluation_records)
        _atomic_write_jsonl(self.metrics_path, self.step_metrics)

    def _valid_records(self) -> list[dict[str, Any]]:
        return [
            record
            for record in self.evaluation_records
            if record.get("valid") and record.get("reward") is not None
        ]

    def _write_run_state(
        self,
        *,
        status: str,
        complete: bool,
        active_step: int | None = None,
        step_started_at_utc: str | None = None,
        last_completed_at_utc: str | None = None,
        last_error: str | None = None,
    ) -> None:
        previous: dict[str, Any] = {}
        state_path = self.run_dir / "run_state.json"
        if state_path.is_file():
            previous = _read_json(state_path)
        _atomic_write_json(
            state_path,
            {
                "schema": REAL_SEARCH_STATE_SCHEMA,
                "updated_at_utc": _utc_now(),
                "run_id": self.trainer.run_id,
                "status": status,
                "active_step": active_step,
                "step_started_at_utc": step_started_at_utc,
                "last_completed_at_utc": (
                    last_completed_at_utc
                    if last_completed_at_utc is not None
                    else previous.get("last_completed_at_utc")
                ),
                "last_error": last_error,
                "current_step": self.trainer.step,
                "optimizer_step": self.trainer.optimizer_step,
                "latest_checkpoint": str(self.latest_checkpoint_path),
                "evaluation_records": len(self.evaluation_records),
                "step_metric_records": len(self.step_metrics),
                "complete": complete,
            },
        )

    def _write_summary(
        self,
        *,
        complete: bool,
        last_completed_at_utc: str | None = None,
    ) -> None:
        valid = self._valid_records()
        best = max(valid, key=lambda item: float(item["reward"])) if valid else None
        _atomic_write_json(
            self.run_dir / "training_stats.json",
            {
                "schema": REAL_SEARCH_STATS_SCHEMA,
                "complete": complete,
                "current_step": self.trainer.step,
                "optimizer_step": self.trainer.optimizer_step,
                "history": [asdict(item) for item in self.trainer.history],
                "performance": self.step_metrics,
                "total_reward_requests": len(self.evaluation_records),
                "valid_reward_requests": len(valid),
                "unique_expressions": len(
                    {record["structural_hash"] for record in self.evaluation_records}
                ),
                "total_wall_seconds": float(
                    sum(float(item["wall_seconds"]) for item in self.step_metrics)
                ),
            },
        )
        _atomic_write_json(
            self.run_dir / "best_candidate.json",
            {
                "schema": "factor_gfn.real_search_best_candidate.v1",
                "candidate": best,
            },
        )
        self._write_run_state(
            status="completed" if complete else "ready",
            complete=complete,
            last_completed_at_utc=last_completed_at_utc,
        )
        artifact_paths = [
            self.run_dir / "search_run_config.json",
            self.run_dir / "run_metadata.json",
            self.run_dir / "run_state.json",
            self.run_dir / "training_stats.json",
            self.evaluations_path,
            self.metrics_path,
            self.run_dir / "best_candidate.json",
            self.latest_checkpoint_path,
            self.run_dir / "experiment_manifest.json",
        ]
        if self.tensorboard_dir.exists():
            artifact_paths.append(self.tensorboard_dir)
        artifact_paths.extend(sorted((self.run_dir / "checkpoints").glob("*.pt")))
        artifact_paths.extend(
            sorted((self.run_dir / "recovery_archives").glob("*.json"))
        )
        _atomic_write_json(
            self.run_dir / "experiment_manifest.json",
            {
                "schema": REAL_SEARCH_MANIFEST_SCHEMA,
                "updated_at_utc": _utc_now(),
                "run_id": self.trainer.run_id,
                "run_dir": str(self.run_dir),
                "complete": complete,
                "current_step": self.trainer.step,
                "optimizer_step": self.trainer.optimizer_step,
                "config_fingerprint": self.config.fingerprint(),
                "reward_provider_fingerprint": self.recording_provider.fingerprint(),
                "main_reward_requests": len(self.evaluation_records),
                "unique_main_expressions": len(
                    {record["structural_hash"] for record in self.evaluation_records}
                ),
                "valid_main_requests": len(valid),
                "artifacts": [str(path) for path in artifact_paths],
            },
        )

    def _create_tensorboard_writer(self) -> Any | None:
        if not self.settings.tensorboard_enabled:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard 已启用但依赖未安装；请运行 "
                ".\\.venv\\python.exe -m pip install \"tensorboard>=2.18,<3\""
            ) from error
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(
            log_dir=str(self.tensorboard_dir),
            purge_step=self.trainer.step + 1,
        )

    @staticmethod
    def _add_finite_scalar(writer: Any, tag: str, value: Any, step: int) -> None:
        if value is None:
            return
        numeric = float(value)
        if math.isfinite(numeric):
            writer.add_scalar(tag, numeric, step)

    def _write_tensorboard_step(
        self,
        writer: Any | None,
        *,
        stats: TrainingStats,
        metrics: SearchStepMetrics,
        records: list[dict[str, Any]],
    ) -> None:
        if writer is None:
            return
        step = stats.step
        scalar_values = {
            "train/loss": stats.loss,
            "train/reward_mean": stats.reward_mean,
            "train/reward_median": stats.reward_median,
            "train/log_z": stats.log_z,
            "train/gradient_norm": stats.gradient_norm,
            "diagnostics/tb_delta_mean": stats.tb_delta_mean,
            "diagnostics/tb_delta_std": stats.tb_delta_std,
            "diagnostics/tb_delta_rms": stats.tb_delta_rms,
            "diagnostics/tb_delta_mean_square_ratio": (
                stats.tb_delta_mean_square_ratio
            ),
            "diagnostics/tb_delta_std_square_ratio": (
                stats.tb_delta_std_square_ratio
            ),
            "diagnostics/mean_log_pf": stats.mean_log_pf,
            "diagnostics/mean_log_pb": stats.mean_log_pb,
            "diagnostics/mean_log_reward": stats.log_reward_mean,
            "diagnostics/model_gradient_norm_before_clip": (
                stats.model_gradient_norm_before_clip
            ),
            "diagnostics/log_z_gradient_before_clip": (
                stats.log_z_gradient_before_clip
            ),
            "diagnostics/model_gradient_clip_coefficient": (
                stats.model_gradient_clip_coefficient
            ),
            "diagnostics/log_z_gradient_clip_coefficient": (
                stats.log_z_gradient_clip_coefficient
            ),
            "diagnostics/model_parameter_update_norm": (
                stats.model_parameter_update_norm
            ),
            "diagnostics/model_relative_update_norm": (
                stats.model_relative_update_norm
            ),
            "diagnostics/log_z_update": stats.log_z_update,
            "train/effective_batch_size": stats.effective_batch_size,
            "train/rejection_rate": stats.batch_rejection_rate,
            "train/expression_unique_rate": stats.expression_unique_rate,
            "train/trajectory_length_mean": stats.trajectory_length_mean,
            "train/terminal_node_count_p50": stats.terminal_node_count_p50,
            "train/terminal_node_count_p90": stats.terminal_node_count_p90,
            "train/max_node_terminal_rate": stats.max_node_terminal_rate,
            "train/policy_entropy": stats.policy_entropy_mean,
            "train/policy_entropy_normalized": (
                stats.policy_entropy_normalized_mean
            ),
            "policy/group_entropy": stats.group_entropy_mean,
            "policy/group_entropy_normalized": (
                stats.group_entropy_normalized_mean
            ),
            "policy/leaf_group_probability": (
                stats.leaf_group_probability_mean
            ),
            "policy/unary_group_probability": (
                stats.unary_group_probability_mean
            ),
            "policy/binary_group_probability": (
                stats.binary_group_probability_mean
            ),
            "policy/leaf_action_rate": stats.leaf_action_rate,
            "policy/unary_action_rate": stats.unary_action_rate,
            "policy/binary_action_rate": stats.binary_action_rate,
            "policy/grammar_category_entropy": stats.grammar_category_entropy_mean,
            "policy/grammar_category_entropy_normalized": stats.grammar_category_entropy_normalized_mean,
            "policy/operator_entropy": stats.operator_entropy_mean,
            "policy/operator_entropy_normalized": stats.operator_entropy_normalized_mean,
            "policy/window_entropy": stats.window_entropy_mean,
            "policy/window_entropy_normalized": stats.window_entropy_normalized_mean,
            "policy/temporal_operator_action_rate": stats.temporal_operator_action_rate,
            "train/illegal_action_rate": stats.illegal_action_rate,
            "progress/current_step": stats.step,
            "progress/optimizer_step": stats.optimizer_step,
            "performance/wall_seconds": metrics.wall_seconds,
            "performance/factor_seconds": metrics.factor_seconds,
            "performance/reward_seconds": metrics.reward_seconds,
            "performance/sampling_seconds": metrics.sampling_seconds,
            "performance/reward_provider_seconds": metrics.reward_provider_seconds,
            "performance/training_update_seconds": metrics.training_update_seconds,
            "performance/tb_loss_forward_cuda_seconds": metrics.tb_loss_forward_cuda_seconds,
            "performance/backward_cuda_seconds": metrics.backward_cuda_seconds,
            "performance/optimizer_cuda_seconds": metrics.optimizer_cuda_seconds,
            "performance/reward_requests": metrics.reward_requests,
            "performance/valid_reward_requests": metrics.valid_reward_requests,
            "performance/subexpression_cache_hits": (
                metrics.subexpression_cache_hits
            ),
            "performance/subexpression_cache_misses": (
                metrics.subexpression_cache_misses
            ),
            "performance/subexpression_cache_evictions": (
                metrics.subexpression_cache_evictions
            ),
            "performance/subexpression_cache_hit_rate": (
                metrics.subexpression_cache_hits
                / (
                    metrics.subexpression_cache_hits
                    + metrics.subexpression_cache_misses
                )
                if metrics.subexpression_cache_hits
                + metrics.subexpression_cache_misses
                else 0.0
            ),
            "performance/gpu_memory_allocated_mib": (
                metrics.gpu_memory_allocated_bytes / 1024**2
            ),
            "performance/gpu_peak_memory_allocated_mib": (
                metrics.gpu_peak_memory_allocated_bytes / 1024**2
            ),
        }
        for name in (
            "feature", "unary", "ts_unary", "binary", "ts_binary", "cross_sectional"
        ):
            scalar_values[f"policy/grammar_probability/{name}"] = getattr(
                stats, f"{name}_category_probability_mean"
            )
            scalar_values[f"policy/grammar_action_rate/{name}"] = getattr(
                stats, f"{name}_category_action_rate"
            )
        for window in (5, 10, 20, 40, 60):
            scalar_values[f"policy/window_probability/{window}"] = getattr(
                stats, f"window_{window}_probability_mean"
            )
            scalar_values[f"policy/window_action_rate/{window}"] = getattr(
                stats, f"window_{window}_action_rate"
            )
        for tag, value in scalar_values.items():
            self._add_finite_scalar(writer, tag, value, step)
        rewards = np.asarray(
            [
                float(record["reward"])
                for record in records
                if record.get("valid") and record.get("reward") is not None
            ],
            dtype=np.float64,
        )
        if rewards.size:
            writer.add_histogram("reward/valid_candidates", rewards, step)
        writer.flush()

    def _print_step_progress(
        self,
        *,
        target_step: int,
        stats: TrainingStats,
        metrics: SearchStepMetrics,
    ) -> None:
        if not self.settings.console_progress:
            return
        def shown(value: Any, digits: int = 6) -> str:
            if value is None:
                return "NA"
            numeric = float(value)
            return f"{numeric:.{digits}f}" if math.isfinite(numeric) else str(numeric)

        print(
            "[real-search] "
            f"current_step={stats.step}/{self.config.training.max_steps} "
            f"optimizer_step={stats.optimizer_step} target_step={target_step} "
            f"wall={metrics.wall_seconds:.2f}s factor={metrics.factor_seconds:.2f}s "
            f"reward={metrics.reward_seconds:.2f}s requests={metrics.reward_requests} "
            f"sample={metrics.sampling_seconds:.2f}s "
            f"update={metrics.training_update_seconds:.2f}s "
            f"valid={metrics.valid_reward_requests} "
            f"subexpr_hits={metrics.subexpression_cache_hits}/"
            f"{metrics.subexpression_cache_hits + metrics.subexpression_cache_misses} "
            f"rejection={shown(stats.batch_rejection_rate, 4)} "
            f"loss={shown(stats.loss)} reward_mean={shown(stats.reward_mean)} "
            f"log_z={shown(stats.log_z)} "
            f"delta_mean={shown(stats.tb_delta_mean)} "
            f"delta_std={shown(stats.tb_delta_std)} "
            f"model_clip={shown(stats.model_gradient_clip_coefficient, 6)} "
            f"log_z_clip={shown(stats.log_z_gradient_clip_coefficient, 6)} "
            f"model_update={shown(stats.model_relative_update_norm, 8)} "
            f"log_z_update={shown(stats.log_z_update, 8)} "
            f"entropy={shown(stats.policy_entropy_normalized_mean, 4)} "
            f"group_p={shown(stats.leaf_group_probability_mean, 3)}/"
            f"{shown(stats.unary_group_probability_mean, 3)}/"
            f"{shown(stats.binary_group_probability_mean, 3)} "
            f"action_rate={shown(stats.leaf_action_rate, 3)}/"
            f"{shown(stats.unary_action_rate, 3)}/"
            f"{shown(stats.binary_action_rate, 3)} "
            f"grammar_rate={shown(stats.feature_category_action_rate, 3)}/"
            f"{shown(stats.unary_category_action_rate, 3)}/"
            f"{shown(stats.ts_unary_category_action_rate, 3)}/"
            f"{shown(stats.binary_category_action_rate, 3)}/"
            f"{shown(stats.ts_binary_category_action_rate, 3)}/"
            f"{shown(stats.cross_sectional_category_action_rate, 3)} "
            f"window_rate={shown(stats.window_5_action_rate, 3)}/"
            f"{shown(stats.window_10_action_rate, 3)}/"
            f"{shown(stats.window_20_action_rate, 3)}/"
            f"{shown(stats.window_40_action_rate, 3)}/"
            f"{shown(stats.window_60_action_rate, 3)} "
            f"nodes_p50/p90={shown(stats.terminal_node_count_p50, 1)}/"
            f"{shown(stats.terminal_node_count_p90, 1)} "
            f"max_node_rate={shown(stats.max_node_terminal_rate, 3)} "
            f"unique_rate={shown(stats.expression_unique_rate, 4)} "
            f"checkpoint={self.latest_checkpoint_path}",
            flush=True,
        )

    def run_until(self, target_step: int) -> list[TrainingStats]:
        if isinstance(target_step, bool) or not isinstance(target_step, int):
            raise ValueError("target_step 必须是整数")
        if not self.trainer.step < target_step <= self.config.training.max_steps:
            raise ValueError(
                f"target_step 必须位于 ({self.trainer.step}, "
                f"{self.config.training.max_steps}]"
            )
        produced: list[TrainingStats] = []
        device = self.trainer.device
        writer = self._create_tensorboard_writer()
        active_step: int | None = None
        if self.settings.console_progress:
            print(
                f"[real-search] tensorboard_logdir={self.tensorboard_dir} "
                f"current_step={self.trainer.step} optimizer_step="
                f"{self.trainer.optimizer_step}",
                flush=True,
            )
        try:
            while self.trainer.step < target_step:
                expected_step = self.trainer.step + 1
                active_step = expected_step
                step_started_at_utc = _utc_now()
                self._write_run_state(
                    status="running",
                    complete=False,
                    active_step=expected_step,
                    step_started_at_utc=step_started_at_utc,
                )
                self.recording_provider.begin_step(expected_step)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = perf_counter()
                # 严格中性化失败已逐候选持久化完整审计；正式长跑控制台只保留
                # 每步训练摘要，避免数百条预期的逐日期警告淹没进度信息。
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", IndustryNeutralizationWarning)
                    stats = self.trainer.train_step()
                torch.cuda.synchronize(device)
                wall_seconds = perf_counter() - started
                records = self.recording_provider.finish_step()
                if stats.step != expected_step:
                    raise RuntimeError("Trainer 步数与 Reward 请求记录未对齐")
                self.evaluation_records.extend(records)
                factor_seconds = float(
                    sum(
                        (
                            0.0
                            if record.get("provider_cache_hit")
                            else float(
                                record.get("metadata", {}).get("factor_seconds") or 0.0
                            )
                        )
                        for record in records
                    )
                )
                reward_seconds = float(
                    sum(
                        (
                            0.0
                            if record.get("provider_cache_hit")
                            else float(
                                record.get("metadata", {}).get("reward_seconds") or 0.0
                            )
                        )
                        for record in records
                    )
                )
                trainer_timings = getattr(self.trainer, "last_step_timings", {})
                metrics = SearchStepMetrics(
                    step=stats.step,
                    optimizer_step=stats.optimizer_step,
                    wall_seconds=wall_seconds,
                    reward_requests=len(records),
                    valid_reward_requests=sum(bool(record["valid"]) for record in records),
                    unique_expressions=len(
                        {record["structural_hash"] for record in records}
                    ),
                    factor_seconds=factor_seconds,
                    reward_seconds=reward_seconds,
                    provider_cache_hits=sum(
                        bool(record.get("provider_cache_hit")) for record in records
                    ),
                    subexpression_cache_hits=sum(
                        int(
                            record.get("metadata", {}).get(
                                "subexpression_cache_hits", 0
                            )
                            or 0
                        )
                        for record in records
                        if not record.get("provider_cache_hit")
                    ),
                    subexpression_cache_misses=sum(
                        int(
                            record.get("metadata", {}).get(
                                "subexpression_cache_misses", 0
                            )
                            or 0
                        )
                        for record in records
                        if not record.get("provider_cache_hit")
                    ),
                    subexpression_cache_evictions=sum(
                        int(
                            record.get("metadata", {}).get(
                                "subexpression_cache_evictions", 0
                            )
                            or 0
                        )
                        for record in records
                        if not record.get("provider_cache_hit")
                    ),
                    gpu_memory_allocated_bytes=int(torch.cuda.memory_allocated(device)),
                    gpu_memory_reserved_bytes=int(torch.cuda.memory_reserved(device)),
                    gpu_peak_memory_allocated_bytes=int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    gpu_peak_memory_reserved_bytes=int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                    sampling_seconds=float(
                        trainer_timings.get("sampling_seconds")
                        or 0.0
                    ),
                    reward_provider_seconds=float(
                        trainer_timings.get("reward_provider_seconds")
                        or 0.0
                    ),
                    training_update_seconds=float(
                        trainer_timings.get("training_update_seconds")
                        or 0.0
                    ),
                    tb_loss_forward_cuda_seconds=trainer_timings.get(
                        "tb_loss_forward_cuda_seconds"
                    ),
                    backward_cuda_seconds=trainer_timings.get(
                        "backward_cuda_seconds"
                    ),
                    optimizer_cuda_seconds=trainer_timings.get(
                        "optimizer_cuda_seconds"
                    ),
                )
                metric_record = asdict(metrics)
                metric_record["policy_diagnostics"] = {
                    key: value
                    for key, value in asdict(stats).items()
                    if key.startswith((
                        "grammar_",
                        "operator_entropy",
                        "window_",
                        "feature_category_",
                        "unary_category_",
                        "ts_unary_category_",
                        "binary_category_",
                        "ts_binary_category_",
                        "cross_sectional_category_",
                        "temporal_operator_",
                    ))
                }
                self.step_metrics.append(metric_record)
                self._persist_tables()
                self.trainer.save_checkpoint(self.latest_checkpoint_path)
                if (
                    stats.step % self.settings.checkpoint_interval == 0
                    or stats.step == target_step
                    or stats.step == self.config.training.max_steps
                ):
                    self.trainer.save_checkpoint(
                        self._archive_checkpoint_path(stats.step)
                    )
                complete = stats.step == self.config.training.max_steps
                self._write_summary(
                    complete=complete,
                    last_completed_at_utc=_utc_now(),
                )
                self._write_tensorboard_step(
                    writer, stats=stats, metrics=metrics, records=records
                )
                self._print_step_progress(
                    target_step=target_step, stats=stats, metrics=metrics
                )
                produced.append(stats)
                active_step = None
        except KeyboardInterrupt as error:
            self.recording_provider.abort_step()
            self._write_run_state(
                status="interrupted",
                complete=False,
                active_step=active_step,
                last_error=f"{type(error).__name__}: {error}",
            )
            raise
        except Exception as error:
            self.recording_provider.abort_step()
            self._write_run_state(
                status="failed",
                complete=False,
                active_step=active_step,
                last_error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            if writer is not None:
                writer.close()
        return produced


def _build_real_components(
    settings: RealSearchSettings,
    *,
    data_config: RealRewardDataConfig,
    paths: RealRewardDataPaths,
) -> tuple[
    GFNConfig,
    GFNTrainer,
    RecordingRewardProvider,
    dict[str, Any],
    Any,
]:
    _validate_training_data_config(data_config)
    device, cuda_environment = require_cuda_device(settings.device)
    config = build_stage5_real_training_config(
        max_steps=settings.max_steps, seed=settings.seed
    )
    context = build_real_reward_data_context(data_config, paths)
    if (
        context.config.train_start != "2010-01-01"
        or context.config.train_end != "2018-12-31"
    ):
        raise RuntimeError("真实 Reward 上下文暴露了非冻结训练区间")
    provider = RealRewardProvider(
        context,
        config.reward,
        cache_max_entries=settings.cache_max_entries,
        subexpression_cache_max_bytes=settings.subexpression_cache_max_bytes,
    )
    recording_provider = RecordingRewardProvider(provider)
    trainer = GFNTrainer(config, recording_provider, device=device)
    return config, trainer, recording_provider, cuda_environment, context


def create_real_search_runner(
    settings: RealSearchSettings,
    *,
    data_config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> RealSearchRunner:
    """Create a new, exclusive ``runs/real_search/<run_id>`` directory."""

    config, trainer, provider, cuda_environment, context = _build_real_components(
        settings, data_config=data_config, paths=paths
    )
    run_dir = settings.run_root / trainer.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    runner = RealSearchRunner(
        settings=settings,
        config=config,
        trainer=trainer,
        recording_provider=provider,
        run_dir=run_dir,
        cuda_environment=cuda_environment,
    )
    _atomic_write_json(
        run_dir / "search_run_config.json",
        {
            "schema": REAL_SEARCH_SCHEMA,
            "created_at_utc": _utc_now(),
            "run_id": trainer.run_id,
            "settings": settings.manifest(),
            "config_fingerprint": config.fingerprint(),
            "reward_provider_fingerprint": provider.fingerprint(),
            "context_fingerprint": context.fingerprint,
            "requested_train_start": data_config.train_start,
            "requested_train_end": data_config.train_end,
            "cuda_environment": cuda_environment,
        },
    )
    _atomic_write_json(run_dir / "run_metadata.json", trainer.run_metadata())
    runner._persist_tables()
    trainer.save_checkpoint(runner.latest_checkpoint_path)
    runner._write_summary(complete=False)
    return runner


def _archive_orphans(
    run_dir: Path,
    *,
    checkpoint_step: int,
    evaluations: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_evaluations = [
        record
        for record in evaluations
        if int(record.get("logical_step", -1)) <= checkpoint_step
    ]
    kept_metrics = [
        record for record in metrics if int(record.get("step", -1)) <= checkpoint_step
    ]
    kept_evaluation_ids = {id(record) for record in kept_evaluations}
    kept_metric_ids = {id(record) for record in kept_metrics}
    orphan_evaluations = [
        record for record in evaluations if id(record) not in kept_evaluation_ids
    ]
    orphan_metrics = [record for record in metrics if id(record) not in kept_metric_ids]
    if orphan_evaluations or orphan_metrics:
        recovery_dir = run_dir / "recovery_archives"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _atomic_write_json(
            recovery_dir / f"orphan_records_{stamp}.json",
            {
                "schema": "factor_gfn.real_search_recovery_orphans.v1",
                "checkpoint_step": checkpoint_step,
                "evaluations": orphan_evaluations,
                "metrics": orphan_metrics,
            },
        )
    return kept_evaluations, kept_metrics


def _validate_or_upgrade_cuda_environment(
    saved_environment: Any,
    current_environment: dict[str, Any],
    run_state: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    if not isinstance(saved_environment, dict):
        raise ValueError("search_run_config.json 缺少 CUDA 环境合同")
    upgraded = dict(saved_environment)
    legacy_cublas_upgrade = "cublas_workspace_config" not in upgraded
    if legacy_cublas_upgrade:
        if any(
            int(run_state.get(name, -1)) != 0
            for name in (
                "current_step",
                "optimizer_step",
                "evaluation_records",
                "step_metric_records",
            )
        ):
            raise ValueError(
                "仅允许尚未开始训练的旧 run 补录 CUBLAS_WORKSPACE_CONFIG"
            )
        upgraded["cublas_workspace_config"] = current_environment[
            "cublas_workspace_config"
        ]
    if upgraded != current_environment:
        raise ValueError("恢复要求 CUDA 型号、能力、设备数和运行时环境与原 run 完全一致")
    return upgraded, legacy_cublas_upgrade


def resume_real_search_runner(
    run_dir: str | Path,
    *,
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> RealSearchRunner:
    """Resume only when config, Provider, data, and CUDA environment all match."""

    directory = Path(run_dir).resolve()
    config_path = directory / "search_run_config.json"
    checkpoint_path = directory / "checkpoint_latest.pt"
    metadata_path = directory / "run_metadata.json"
    for path in (config_path, checkpoint_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    saved = _read_json(config_path)
    if saved.get("schema") != REAL_SEARCH_SCHEMA:
        raise ValueError("search_run_config.json schema 不兼容")
    metadata = _read_json(metadata_path)
    saved_policy_mode = (
        metadata.get("config_manifest", {})
        .get("config", {})
        .get("model", {})
        .get("token_policy_mode")
    )
    if saved_policy_mode != STAGE5_TOKEN_POLICY_MODE:
        raise ValueError(
            "该 run 使用已退役的历史策略，只允许只读审计和阶段6候选导入；"
            "正式搜索入口禁止跨策略恢复"
        )
    settings = RealSearchSettings.from_manifest(saved["settings"])
    data_config = RealRewardDataConfig(
        train_start=str(saved["requested_train_start"]),
        train_end=str(saved["requested_train_end"]),
    )
    config, trainer, provider, cuda_environment, context = _build_real_components(
        settings, data_config=data_config, paths=paths
    )
    _, legacy_cublas_upgrade = _validate_or_upgrade_cuda_environment(
        saved.get("cuda_environment"),
        cuda_environment,
        _read_json(directory / "run_state.json"),
    )
    if config.fingerprint() != saved.get("config_fingerprint"):
        raise ValueError("恢复时正式训练配置指纹不一致")
    if provider.fingerprint() != saved.get("reward_provider_fingerprint"):
        raise ValueError("恢复时 Reward Provider 指纹不一致")
    if context.fingerprint != saved.get("context_fingerprint"):
        raise ValueError("恢复时真实数据上下文指纹不一致")
    if metadata.get("config_fingerprint") != config.fingerprint():
        raise ValueError("run_metadata.json 配置指纹不一致")
    if metadata.get("reward_provider_fingerprint") != provider.fingerprint():
        raise ValueError("run_metadata.json Provider 指纹不一致")
    trainer.load_checkpoint(checkpoint_path)
    if trainer.run_id != saved.get("run_id") or directory.name != trainer.run_id:
        raise ValueError("run_id、目录名与检查点不一致")
    if legacy_cublas_upgrade:
        saved["cuda_environment"] = cuda_environment
        _atomic_write_json(config_path, saved)
    _atomic_write_json(metadata_path, trainer.run_metadata())
    evaluations, metrics = _archive_orphans(
        directory,
        checkpoint_step=trainer.step,
        evaluations=_read_jsonl(directory / "evaluations.jsonl"),
        metrics=_read_jsonl(directory / "step_metrics.jsonl"),
    )
    runner = RealSearchRunner(
        settings=settings,
        config=config,
        trainer=trainer,
        recording_provider=provider,
        run_dir=directory,
        cuda_environment=cuda_environment,
        evaluation_records=evaluations,
        step_metrics=metrics,
    )
    runner._persist_tables()
    runner._write_summary(complete=trainer.step == config.training.max_steps)
    return runner


__all__ = [
    "REAL_SEARCH_MANIFEST_SCHEMA",
    "REAL_SEARCH_SCHEMA",
    "REAL_SEARCH_STATE_SCHEMA",
    "REAL_SEARCH_STATS_SCHEMA",
    "RealSearchRunner",
    "RealSearchSettings",
    "RecordingRewardProvider",
    "SearchStepMetrics",
    "create_real_search_runner",
    "require_cuda_device",
    "resume_real_search_runner",
]
