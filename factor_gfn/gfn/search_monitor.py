"""Read-only status monitoring and explicit baseline freezing for real search runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SEARCH_BASELINE_SCHEMA = "factor_gfn.real_search_baseline.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 行必须是对象：{path}:{line_number}")
        last = value
    return last


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 行必须是对象：{path}:{line_number}")
        result.append(value)
    return result


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_search_status(run_dir: str | Path) -> dict[str, Any]:
    """Return one status snapshot without writing to or loading the checkpoint."""

    directory = Path(run_dir).resolve()
    state_path = directory / "run_state.json"
    stats_path = directory / "training_stats.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = _read_json(state_path)
    stats = _read_json(stats_path) if stats_path.is_file() else {}
    history = stats.get("history")
    last_history = history[-1] if isinstance(history, list) and history else None
    last_metric = _read_last_jsonl(directory / "step_metrics.jsonl")
    status = state.get("status")
    if not isinstance(status, str):
        status = "completed" if state.get("complete") else "ready"
    active_started = _parse_utc(state.get("step_started_at_utc"))
    active_elapsed: float | None = None
    if status == "running" and active_started is not None:
        active_elapsed = max(
            0.0, (datetime.now(timezone.utc) - active_started).total_seconds()
        )
    alerts: list[str] = []
    current_step = int(state.get("current_step", 0))
    optimizer_step = int(state.get("optimizer_step", 0))
    if status in {"failed", "interrupted"}:
        alerts.append(status)
    if optimizer_step > current_step:
        alerts.append("optimizer_step_exceeds_current_step")
    if last_history is not None:
        rejection = last_history.get("batch_rejection_rate")
        if rejection is not None and float(rejection) > 0.8:
            alerts.append("rejection_rate_above_80pct")
        loss = last_history.get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            alerts.append("non_finite_loss")
        illegal = last_history.get("illegal_action_rate")
        if illegal is not None and float(illegal) != 0.0:
            alerts.append("illegal_action_rate_nonzero")
    return {
        "run_dir": str(directory),
        "run_id": state.get("run_id", directory.name),
        "status": status,
        "current_step": current_step,
        "optimizer_step": optimizer_step,
        "active_step": state.get("active_step"),
        "active_elapsed_seconds": active_elapsed,
        "updated_at_utc": state.get("updated_at_utc"),
        "last_completed_at_utc": state.get("last_completed_at_utc"),
        "last_error": state.get("last_error"),
        "evaluation_records": state.get("evaluation_records"),
        "step_metric_records": state.get("step_metric_records"),
        "latest_checkpoint": state.get("latest_checkpoint"),
        "complete": bool(state.get("complete")),
        "last_training_stats": last_history,
        "last_step_metrics": last_metric,
        "alerts": alerts,
    }


def _shown(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else str(numeric)


def format_search_status(snapshot: dict[str, Any]) -> str:
    """Format the most useful progress and health fields for a terminal."""

    history = snapshot.get("last_training_stats") or {}
    metrics = snapshot.get("last_step_metrics") or {}
    alert_text = ",".join(snapshot.get("alerts") or []) or "none"
    return (
        f"[{snapshot['status']}] run_id={snapshot['run_id']} "
        f"current_step={snapshot['current_step']} "
        f"optimizer_step={snapshot['optimizer_step']} "
        f"active_step={snapshot.get('active_step')} "
        f"active_elapsed={_shown(snapshot.get('active_elapsed_seconds'), 1)}s\n"
        f"loss={_shown(history.get('loss'))} "
        f"reward_mean={_shown(history.get('reward_mean'))} "
        f"delta_mean={_shown(history.get('tb_delta_mean'))} "
        f"delta_std={_shown(history.get('tb_delta_std'))} "
        f"model_clip={_shown(history.get('model_gradient_clip_coefficient', history.get('gradient_clip_coefficient')))} "
        f"log_z_clip={_shown(history.get('log_z_gradient_clip_coefficient'))} "
        f"model_update={_shown(history.get('model_relative_update_norm'))} "
        f"log_z_update={_shown(history.get('log_z_update'))} "
        f"rejection={_shown(history.get('batch_rejection_rate'))} "
        f"entropy={_shown(history.get('policy_entropy_normalized_mean'))} "
        f"group_p={_shown(history.get('leaf_group_probability_mean'), 3)}/"
        f"{_shown(history.get('unary_group_probability_mean'), 3)}/"
        f"{_shown(history.get('binary_group_probability_mean'), 3)} "
        f"action_rate={_shown(history.get('leaf_action_rate'), 3)}/"
        f"{_shown(history.get('unary_action_rate'), 3)}/"
        f"{_shown(history.get('binary_action_rate'), 3)} "
        f"grammar_rate={_shown(history.get('feature_category_action_rate'), 3)}/"
        f"{_shown(history.get('unary_category_action_rate'), 3)}/"
        f"{_shown(history.get('ts_unary_category_action_rate'), 3)}/"
        f"{_shown(history.get('binary_category_action_rate'), 3)}/"
        f"{_shown(history.get('ts_binary_category_action_rate'), 3)}/"
        f"{_shown(history.get('cross_sectional_category_action_rate'), 3)} "
        f"window_rate={_shown(history.get('window_5_action_rate'), 3)}/"
        f"{_shown(history.get('window_10_action_rate'), 3)}/"
        f"{_shown(history.get('window_20_action_rate'), 3)}/"
        f"{_shown(history.get('window_40_action_rate'), 3)}/"
        f"{_shown(history.get('window_60_action_rate'), 3)} "
        f"nodes_p50/p90={_shown(history.get('terminal_node_count_p50'), 1)}/"
        f"{_shown(history.get('terminal_node_count_p90'), 1)} "
        f"max_node_rate={_shown(history.get('max_node_terminal_rate'), 3)} "
        f"wall={_shown(metrics.get('wall_seconds'), 1)}s "
        f"factor={_shown(metrics.get('factor_seconds'), 1)}s "
        f"reward={_shown(metrics.get('reward_seconds'), 1)}s "
        f"sample={_shown(metrics.get('sampling_seconds'), 1)}s "
        f"update={_shown(metrics.get('training_update_seconds'), 1)}s "
        f"alerts={alert_text}\n"
        f"checkpoint={snapshot.get('latest_checkpoint')}"
    )


def watch_search_status(
    run_dir: str | Path,
    *,
    interval_seconds: float = 10.0,
) -> None:
    """Continuously print read-only snapshots until interrupted by the user."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds 必须大于 0")
    try:
        while True:
            print(format_search_status(load_search_status(run_dir)), flush=True)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("监控已停止；训练进程未被修改。", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_search_baseline(
    run_dir: str | Path,
    *,
    expected_step: int | None = None,
) -> Path:
    """Hash the completed artifacts of one stopped run without altering them."""

    directory = Path(run_dir).resolve()
    state = _read_json(directory / "run_state.json")
    current_step = int(state.get("current_step", -1))
    optimizer_step = int(state.get("optimizer_step", -1))
    if expected_step is not None and current_step != expected_step:
        raise ValueError(
            f"基线 step 不一致：expected={expected_step}, actual={current_step}"
        )
    if state.get("status") == "running" or state.get("active_step") is not None:
        raise RuntimeError("训练仍在运行，不能冻结基线")
    relative_paths = [
        "search_run_config.json",
        "run_metadata.json",
        "run_state.json",
        "training_stats.json",
        "evaluations.jsonl",
        "step_metrics.jsonl",
        "best_candidate.json",
        "experiment_manifest.json",
        "checkpoint_latest.pt",
        f"checkpoints/checkpoint_step_{current_step:08d}.pt",
    ]
    artifacts: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    stats = _read_json(directory / "training_stats.json")
    payload = {
        "schema": SEARCH_BASELINE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": state.get("run_id", directory.name),
        "run_dir": str(directory),
        "current_step": current_step,
        "optimizer_step": optimizer_step,
        "evaluation_records": state.get("evaluation_records"),
        "step_metric_records": state.get("step_metric_records"),
        "total_wall_seconds": stats.get("total_wall_seconds"),
        "unique_expressions": stats.get("unique_expressions"),
        "valid_reward_requests": stats.get("valid_reward_requests"),
        "artifacts": artifacts,
    }
    target = (
        directory
        / "baselines"
        / f"baseline_step_{current_step:08d}.json"
    )
    if target.is_file():
        existing = _read_json(target)
        comparable_existing = dict(existing)
        comparable_payload = dict(payload)
        comparable_existing.pop("created_at_utc", None)
        comparable_payload.pop("created_at_utc", None)
        if comparable_existing != comparable_payload:
            raise RuntimeError(f"已有基线与当前产物不一致，拒绝覆盖：{target}")
        return target
    _atomic_write_json(target, payload)
    return target


def export_tensorboard_history(run_dir: str | Path) -> Path:
    """Export already-persisted steps once, without touching source artifacts."""

    directory = Path(run_dir).resolve()
    stats_path = directory / "training_stats.json"
    metrics_path = directory / "step_metrics.jsonl"
    evaluations_path = directory / "evaluations.jsonl"
    for path in (stats_path, metrics_path, evaluations_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    stats = _read_json(stats_path)
    history = stats.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("training_stats.json 没有可导出的训练历史")
    current_step = int(stats.get("current_step", history[-1].get("step", -1)))
    source_hashes = {
        path.name: _sha256(path)
        for path in (stats_path, metrics_path, evaluations_path)
    }
    tensorboard_dir = directory / "tensorboard"
    marker = tensorboard_dir / f"history_export_step_{current_step:08d}.json"
    if marker.is_file():
        existing = _read_json(marker)
        if existing.get("source_sha256") != source_hashes:
            raise RuntimeError(f"TensorBoard 历史导出来源已变化，拒绝覆盖：{marker}")
        return tensorboard_dir
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("TensorBoard 依赖未安装，无法导出历史") from error
    performance = {
        int(item["step"]): item for item in _read_jsonl(metrics_path)
    }
    rewards_by_step: dict[int, list[float]] = {}
    for record in _read_jsonl(evaluations_path):
        if not record.get("valid") or record.get("reward") is None:
            continue
        logical_step = int(record.get("logical_step", -1))
        if logical_step >= 0:
            rewards_by_step.setdefault(logical_step, []).append(
                float(record["reward"])
            )
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    try:
        for item in history:
            step = int(item["step"])
            metric = performance.get(step, {})
            scalar_values = {
                "train/loss": item.get("loss"),
                "train/reward_mean": item.get("reward_mean"),
                "train/reward_median": item.get("reward_median"),
                "train/log_z": item.get("log_z"),
                "train/gradient_norm": item.get("gradient_norm"),
                "diagnostics/tb_delta_mean": item.get("tb_delta_mean"),
                "diagnostics/tb_delta_std": item.get("tb_delta_std"),
                "diagnostics/tb_delta_rms": item.get("tb_delta_rms"),
                "diagnostics/tb_delta_mean_square_ratio": item.get(
                    "tb_delta_mean_square_ratio"
                ),
                "diagnostics/tb_delta_std_square_ratio": item.get(
                    "tb_delta_std_square_ratio"
                ),
                "diagnostics/mean_log_pf": item.get("mean_log_pf"),
                "diagnostics/mean_log_pb": item.get("mean_log_pb"),
                "diagnostics/mean_log_reward": item.get("log_reward_mean"),
                "diagnostics/model_gradient_norm_before_clip": item.get(
                    "model_gradient_norm_before_clip"
                ),
                "diagnostics/log_z_gradient_before_clip": item.get(
                    "log_z_gradient_before_clip"
                ),
                "diagnostics/model_gradient_clip_coefficient": item.get(
                    "model_gradient_clip_coefficient",
                    item.get("gradient_clip_coefficient"),
                ),
                "diagnostics/log_z_gradient_clip_coefficient": item.get(
                    "log_z_gradient_clip_coefficient"
                ),
                "diagnostics/model_parameter_update_norm": item.get(
                    "model_parameter_update_norm"
                ),
                "diagnostics/model_relative_update_norm": item.get(
                    "model_relative_update_norm"
                ),
                "diagnostics/log_z_update": item.get("log_z_update"),
                "train/effective_batch_size": item.get("effective_batch_size"),
                "train/rejection_rate": item.get("batch_rejection_rate"),
                "train/expression_unique_rate": item.get("expression_unique_rate"),
                "train/trajectory_length_mean": item.get("trajectory_length_mean"),
                "train/terminal_node_count_p50": item.get(
                    "terminal_node_count_p50"
                ),
                "train/terminal_node_count_p90": item.get(
                    "terminal_node_count_p90"
                ),
                "train/max_node_terminal_rate": item.get(
                    "max_node_terminal_rate"
                ),
                "train/policy_entropy": item.get("policy_entropy_mean"),
                "train/policy_entropy_normalized": item.get(
                    "policy_entropy_normalized_mean"
                ),
                "policy/group_entropy": item.get("group_entropy_mean"),
                "policy/group_entropy_normalized": item.get(
                    "group_entropy_normalized_mean"
                ),
                "policy/leaf_group_probability": item.get(
                    "leaf_group_probability_mean"
                ),
                "policy/unary_group_probability": item.get(
                    "unary_group_probability_mean"
                ),
                "policy/binary_group_probability": item.get(
                    "binary_group_probability_mean"
                ),
                "policy/leaf_action_rate": item.get("leaf_action_rate"),
                "policy/unary_action_rate": item.get("unary_action_rate"),
                "policy/binary_action_rate": item.get("binary_action_rate"),
                "policy/grammar_category_entropy": item.get("grammar_category_entropy_mean"),
                "policy/grammar_category_entropy_normalized": item.get("grammar_category_entropy_normalized_mean"),
                "policy/operator_entropy": item.get("operator_entropy_mean"),
                "policy/operator_entropy_normalized": item.get("operator_entropy_normalized_mean"),
                "policy/window_entropy": item.get("window_entropy_mean"),
                "policy/window_entropy_normalized": item.get("window_entropy_normalized_mean"),
                "policy/temporal_operator_action_rate": item.get("temporal_operator_action_rate"),
                "train/illegal_action_rate": item.get("illegal_action_rate"),
                "progress/current_step": step,
                "progress/optimizer_step": item.get("optimizer_step"),
                "performance/wall_seconds": metric.get("wall_seconds"),
                "performance/factor_seconds": metric.get("factor_seconds"),
                "performance/reward_seconds": metric.get("reward_seconds"),
                "performance/sampling_seconds": metric.get("sampling_seconds"),
                "performance/reward_provider_seconds": metric.get(
                    "reward_provider_seconds"
                ),
                "performance/training_update_seconds": metric.get(
                    "training_update_seconds"
                ),
                "performance/tb_loss_forward_cuda_seconds": metric.get(
                    "tb_loss_forward_cuda_seconds"
                ),
                "performance/backward_cuda_seconds": metric.get(
                    "backward_cuda_seconds"
                ),
                "performance/optimizer_cuda_seconds": metric.get(
                    "optimizer_cuda_seconds"
                ),
                "performance/reward_requests": metric.get("reward_requests"),
                "performance/valid_reward_requests": metric.get(
                    "valid_reward_requests"
                ),
                "performance/gpu_memory_allocated_mib": (
                    float(metric["gpu_memory_allocated_bytes"]) / 1024**2
                    if metric.get("gpu_memory_allocated_bytes") is not None
                    else None
                ),
                "performance/gpu_peak_memory_allocated_mib": (
                    float(metric["gpu_peak_memory_allocated_bytes"]) / 1024**2
                    if metric.get("gpu_peak_memory_allocated_bytes") is not None
                    else None
                ),
            }
            for name in (
                "feature", "unary", "ts_unary", "binary", "ts_binary", "cross_sectional"
            ):
                scalar_values[f"policy/grammar_probability/{name}"] = item.get(
                    f"{name}_category_probability_mean"
                )
                scalar_values[f"policy/grammar_action_rate/{name}"] = item.get(
                    f"{name}_category_action_rate"
                )
            for window in (5, 10, 20, 40, 60):
                scalar_values[f"policy/window_probability/{window}"] = item.get(
                    f"window_{window}_probability_mean"
                )
                scalar_values[f"policy/window_action_rate/{window}"] = item.get(
                    f"window_{window}_action_rate"
                )
            for tag, value in scalar_values.items():
                if value is not None and math.isfinite(float(value)):
                    writer.add_scalar(tag, float(value), step)
            rewards = rewards_by_step.get(step)
            if rewards:
                writer.add_histogram(
                    "reward/valid_candidates",
                    np.asarray(rewards, dtype=np.float64),
                    step,
                )
        writer.flush()
    finally:
        writer.close()
    _atomic_write_json(
        marker,
        {
            "schema": "factor_gfn.tensorboard_history_export.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": stats.get("run_id", directory.name),
            "through_step": current_step,
            "source_sha256": source_hashes,
        },
    )
    return tensorboard_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "watch", "export-tensorboard"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-dir", required=True, type=Path)
        if command == "watch":
            child.add_argument("--interval", type=float, default=10.0)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--run-dir", required=True, type=Path)
    freeze.add_argument("--expected-step", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "status":
        print(format_search_status(load_search_status(args.run_dir)))
    elif args.command == "watch":
        watch_search_status(args.run_dir, interval_seconds=args.interval)
    elif args.command == "export-tensorboard":
        print(export_tensorboard_history(args.run_dir))
    else:
        print(
            freeze_search_baseline(
                args.run_dir, expected_step=args.expected_step
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SEARCH_BASELINE_SCHEMA",
    "format_search_status",
    "export_tensorboard_history",
    "freeze_search_baseline",
    "load_search_status",
    "main",
    "watch_search_status",
]
