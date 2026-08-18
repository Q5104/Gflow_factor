"""Formal real-data Stage 5 runner for the frozen 6/20 no-anchor contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .exhaustive import ExhaustiveRegistry
from .no_anchor_config import (
    FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT,
    FORMAL_STAGE5_NO_ANCHOR_MAX_STEPS,
    FORMAL_STAGE5_NO_ANCHOR_SEED,
    STAGE5_LOGZ_ADAM_LR2E2_AB_CONFIG_FINGERPRINT,
    STAGE5_LOGZ_ADAM_LR2E2_AB_EXPERIMENT_ID,
    STAGE5_LOGZ_SGD_LR1E1_B1_CONFIG_FINGERPRINT,
    STAGE5_LOGZ_SGD_LR1E1_B1_EXPERIMENT_ID,
    NoAnchorGFNConfig,
    build_frozen_stage5_no_anchor_6_20_config,
    build_stage5_logz_adam_lr2e2_ab_config,
    build_stage5_logz_sgd_lr1e1_b1_config,
)
from .real_data import RealRewardDataConfig, RealRewardDataPaths, build_real_reward_data_context
from .real_reward import RealRewardProvider
from .search_runner import (
    REAL_SEARCH_MANIFEST_SCHEMA,
    REAL_SEARCH_STATE_SCHEMA,
    REAL_SEARCH_STATS_SCHEMA,
    RealSearchRunner,
    RealSearchSettings,
    RecordingRewardProvider,
    SearchStepMetrics,
    _archive_orphans,
    _atomic_write_json,
    _read_json,
    _read_jsonl,
    _utc_now,
    _validate_training_data_config,
    require_cuda_device,
)
from .trainer import GFNTrainer


NO_ANCHOR_REAL_SEARCH_SCHEMA = "factor_gfn.no_anchor_real_search.v1"
NO_ANCHOR_LOGZ_ADAM_LR2E2_AB_SEARCH_SCHEMA = (
    "factor_gfn.no_anchor_logz_adam_lr2e2_ab_search.v1"
)
NO_ANCHOR_LOGZ_SGD_LR1E1_B1_SEARCH_SCHEMA = (
    "factor_gfn.no_anchor_logz_sgd_lr1e1_b1_search.v1"
)


class NoAnchorRealSearchRunner(RealSearchRunner):
    """RealSearchRunner with a compact one-line formal Stage 5 progress record."""

    config: NoAnchorGFNConfig
    exhaustive_registry: ExhaustiveRegistry

    def _print_step_progress(
        self,
        *,
        target_step: int,
        stats: Any,
        metrics: SearchStepMetrics,
    ) -> None:
        if not self.settings.console_progress:
            return

        def shown(value: Any, digits: int = 4) -> str:
            if value is None:
                return "NA"
            numeric = float(value)
            return f"{numeric:.{digits}f}" if np.isfinite(numeric) else str(numeric)

        recent = [float(item["wall_seconds"]) for item in self.step_metrics[-10:]]
        seconds_per_step = float(np.mean(recent)) if recent else metrics.wall_seconds
        eta_seconds = max(0, target_step - stats.step) * seconds_per_step
        retry_total = sum((stats.retry_exhausted_count_by_N or {}).values())
        learned = self.trainer.tb_loss.log_z_by_node_count.detach().cpu().numpy()[2:]
        gpu_gib = metrics.gpu_peak_memory_allocated_bytes / 1024**3
        print(
            "[stage5] "
            f"step={stats.step}/{self.config.training.max_steps} "
            f"target={target_step} opt={stats.optimizer_step} "
            f"status={'skip' if stats.skipped_update else 'ok'} "
            f"valid={metrics.valid_reward_requests}/{metrics.reward_requests} "
            f"retry_exhausted_total={retry_total} "
            f"loss={shown(stats.loss)} reward={shown(stats.reward_mean, 6)} "
            f"delta_mean/rms={shown(stats.tb_delta_mean)}/{shown(stats.tb_delta_rms)} "
            f"clip={shown(stats.model_gradient_clip_coefficient, 5)} "
            f"update={shown(stats.model_relative_update_norm, 7)} "
            f"entropy={shown(stats.policy_entropy_normalized_mean, 4)} "
            f"logZ_L={shown(learned.min(), 2)}..{shown(learned.max(), 2)} "
            f"wall={metrics.wall_seconds:.1f}s eta={eta_seconds / 60:.1f}m "
            f"gpu={gpu_gib:.2f}GiB",
            flush=True,
        )

    def close(self) -> None:
        self.exhaustive_registry.close()


def _require_frozen_settings(settings: RealSearchSettings) -> None:
    if settings.max_steps != FORMAL_STAGE5_NO_ANCHOR_MAX_STEPS:
        raise ValueError("formal no-anchor Stage 5 max_steps must remain 1000")
    if settings.seed != FORMAL_STAGE5_NO_ANCHOR_SEED:
        raise ValueError("first formal no-anchor Stage 5 seed must remain 42")


def _configure_registry(trainer: GFNTrainer, registry: ExhaustiveRegistry) -> None:
    semantics = trainer.target_exhaustive_reuse_semantics()
    trainer.configure_no_anchor_exhaustive_registry(
        registry,
        source_semantics_by_N={node_count: semantics for node_count in (1, 2)},
    )


def _build_components(
    settings: RealSearchSettings,
    *,
    config: NoAnchorGFNConfig,
    registry_path: Path,
    data_config: RealRewardDataConfig,
    paths: RealRewardDataPaths,
    normalizer_optimizer: str = "adam",
) -> tuple[NoAnchorGFNConfig, GFNTrainer, RecordingRewardProvider, ExhaustiveRegistry, dict[str, Any], Any]:
    _require_frozen_settings(settings)
    _validate_training_data_config(data_config)
    device, cuda_environment = require_cuda_device(settings.device)
    context = build_real_reward_data_context(data_config, paths)
    provider = RealRewardProvider(
        context,
        config.reward,
        cache_max_entries=settings.cache_max_entries,
        subexpression_cache_max_bytes=settings.subexpression_cache_max_bytes,
    )
    recording_provider = RecordingRewardProvider(provider)
    registry = ExhaustiveRegistry(registry_path, read_only=True)
    try:
        trainer = GFNTrainer(
            config,
            recording_provider,
            device=device,
            normalizer_optimizer=normalizer_optimizer,
        )
        _configure_registry(trainer, registry)
    except Exception:
        registry.close()
        raise
    return config, trainer, recording_provider, registry, cuda_environment, context


def _create_initialized_no_anchor_runner(
    settings: RealSearchSettings,
    *,
    config: NoAnchorGFNConfig,
    search_schema: str,
    experiment_contract: dict[str, Any],
    registry_path: str | Path,
    historical_diagnostic_root: str | Path,
    targeted_artifact_path: str | Path,
    data_config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
    normalizer_optimizer: str = "adam",
) -> NoAnchorRealSearchRunner:
    """Create a fresh initialized no-anchor run from read-only calibration sources."""

    registry_path = Path(registry_path).resolve()
    historical_root = Path(historical_diagnostic_root).resolve()
    targeted_path = Path(targeted_artifact_path).resolve()
    config, trainer, provider, registry, cuda_environment, context = _build_components(
        settings,
        config=config,
        registry_path=registry_path,
        data_config=data_config,
        paths=paths,
        normalizer_optimizer=normalizer_optimizer,
    )
    try:
        historical = trainer.initialize_verified_historical_log_z(historical_root)
        targeted = trainer.initialize_verified_targeted_log_z(targeted_path)
        if targeted.initialization_status != "high_variance_engineering_estimate":
            raise RuntimeError("formal N=17/18 initialization status is unexpected")
        run_dir = settings.run_root / trainer.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        runner = NoAnchorRealSearchRunner(
            settings=settings,
            config=config,
            trainer=trainer,
            recording_provider=provider,
            run_dir=run_dir,
            cuda_environment=cuda_environment,
        )
        runner.exhaustive_registry = registry
        _atomic_write_json(
            run_dir / "search_run_config.json",
            {
                "schema": search_schema,
                "created_at_utc": _utc_now(),
                "run_id": trainer.run_id,
                "settings": settings.manifest(),
                "config_fingerprint": config.fingerprint(),
                "reward_provider_fingerprint": provider.fingerprint(),
                "context_fingerprint": context.fingerprint,
                "registry_path": str(registry_path),
                "historical_diagnostic_root": str(historical_root),
                "targeted_artifact_path": str(targeted_path),
                "historical_provenance_fingerprint": historical.provenance_fingerprint,
                "targeted_provenance_fingerprint": targeted.provenance_fingerprint,
                "diagnostic_sources_read_only": True,
                "experiment_contract": experiment_contract,
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
    except Exception:
        registry.close()
        raise


def create_no_anchor_real_search_runner(
    settings: RealSearchSettings,
    *,
    registry_path: str | Path,
    historical_diagnostic_root: str | Path,
    targeted_artifact_path: str | Path,
    data_config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> NoAnchorRealSearchRunner:
    """Create a new formal run; all diagnostic sources remain read-only."""

    return _create_initialized_no_anchor_runner(
        settings,
        config=build_frozen_stage5_no_anchor_6_20_config(),
        search_schema=NO_ANCHOR_REAL_SEARCH_SCHEMA,
        experiment_contract={
            "kind": "formal_seed42_baseline",
            "config_fingerprint": FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT,
        },
        registry_path=registry_path,
        historical_diagnostic_root=historical_diagnostic_root,
        targeted_artifact_path=targeted_artifact_path,
        data_config=data_config,
        paths=paths,
    )


def create_no_anchor_logz_adam_lr2e2_ab_runner(
    settings: RealSearchSettings,
    *,
    registry_path: str | Path,
    historical_diagnostic_root: str | Path,
    targeted_artifact_path: str | Path,
    data_config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> NoAnchorRealSearchRunner:
    """Create experiment A with only learned-logZ Adam LR changed to 2e-2."""

    return _create_initialized_no_anchor_runner(
        settings,
        config=build_stage5_logz_adam_lr2e2_ab_config(),
        search_schema=NO_ANCHOR_LOGZ_ADAM_LR2E2_AB_SEARCH_SCHEMA,
        experiment_contract={
            "kind": "training_dynamics_ab",
            "experiment_id": STAGE5_LOGZ_ADAM_LR2E2_AB_EXPERIMENT_ID,
            "baseline_config_fingerprint": FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT,
            "config_fingerprint": STAGE5_LOGZ_ADAM_LR2E2_AB_CONFIG_FINGERPRINT,
            "single_changed_training_field": "log_z_learning_rate",
            "baseline_value": 1e-2,
            "experiment_value": 2e-2,
            "selection_uses_reward_or_ic": False,
        },
        registry_path=registry_path,
        historical_diagnostic_root=historical_diagnostic_root,
        targeted_artifact_path=targeted_artifact_path,
        data_config=data_config,
        paths=paths,
    )


def create_no_anchor_logz_sgd_lr1e1_b1_runner(
    settings: RealSearchSettings,
    *,
    registry_path: str | Path,
    historical_diagnostic_root: str | Path,
    targeted_artifact_path: str | Path,
    data_config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> NoAnchorRealSearchRunner:
    """Create B1 with policy Adam unchanged and active-index learned-logZ SGD."""

    return _create_initialized_no_anchor_runner(
        settings,
        config=build_stage5_logz_sgd_lr1e1_b1_config(),
        search_schema=NO_ANCHOR_LOGZ_SGD_LR1E1_B1_SEARCH_SCHEMA,
        experiment_contract={
            "kind": "training_dynamics_ab",
            "experiment_id": STAGE5_LOGZ_SGD_LR1E1_B1_EXPERIMENT_ID,
            "baseline_config_fingerprint": FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT,
            "config_fingerprint": STAGE5_LOGZ_SGD_LR1E1_B1_CONFIG_FINGERPRINT,
            "policy_optimizer": "adam",
            "policy_learning_rate": 1e-4,
            "normalizer_optimizer": "sgd",
            "normalizer_learning_rate": 1e-1,
            "normalizer_momentum": 0.0,
            "normalizer_weight_decay": 0.0,
            "normalizer_gradient_clip_norm": 5.0,
            "normalizer_active_indices_only": True,
            "safety_gate_successful_optimizer_updates": [20, 30],
            "continuation_successful_optimizer_updates": [100, 150],
            "selection_uses_reward_or_ic": False,
        },
        registry_path=registry_path,
        historical_diagnostic_root=historical_diagnostic_root,
        targeted_artifact_path=targeted_artifact_path,
        data_config=data_config,
        paths=paths,
        normalizer_optimizer="sgd",
    )


def _resume_no_anchor_runner(
    run_dir: str | Path,
    *,
    expected_schema: str,
    config: NoAnchorGFNConfig,
    data_config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
    normalizer_optimizer: str = "adam",
) -> NoAnchorRealSearchRunner:
    """Resume a matching no-anchor run without changing its frozen config."""

    directory = Path(run_dir).resolve()
    saved = _read_json(directory / "search_run_config.json")
    if saved.get("schema") != expected_schema:
        raise ValueError("no-anchor search schema is incompatible")
    settings = RealSearchSettings.from_manifest(saved["settings"])
    config, trainer, provider, registry, cuda_environment, context = _build_components(
        settings,
        config=config,
        registry_path=Path(saved["registry_path"]),
        data_config=data_config,
        paths=paths,
        normalizer_optimizer=normalizer_optimizer,
    )
    try:
        if config.fingerprint() != saved.get("config_fingerprint"):
            raise ValueError("formal no-anchor config fingerprint mismatch")
        if provider.fingerprint() != saved.get("reward_provider_fingerprint"):
            raise ValueError("formal no-anchor provider fingerprint mismatch")
        if context.fingerprint != saved.get("context_fingerprint"):
            raise ValueError("formal no-anchor data context mismatch")
        trainer.load_checkpoint(directory / "checkpoint_latest.pt")
        if trainer.run_id != saved.get("run_id") or directory.name != trainer.run_id:
            raise ValueError("formal no-anchor run identity mismatch")
        if trainer.historical_log_z_initialization is None or trainer.targeted_log_z_initialization is None:
            raise ValueError("formal checkpoint lacks initialization provenance")
        if trainer.historical_log_z_initialization.provenance_fingerprint != saved.get("historical_provenance_fingerprint"):
            raise ValueError("historical initialization provenance mismatch")
        if trainer.targeted_log_z_initialization.provenance_fingerprint != saved.get("targeted_provenance_fingerprint"):
            raise ValueError("targeted initialization provenance mismatch")
        evaluations, metrics, trajectory_diagnostics = _archive_orphans(
            directory,
            checkpoint_step=trainer.step,
            evaluations=_read_jsonl(directory / "evaluations.jsonl"),
            metrics=_read_jsonl(directory / "step_metrics.jsonl"),
            trajectory_diagnostics=_read_jsonl(
                directory / "trajectory_diagnostics.jsonl"
            ),
        )
        runner = NoAnchorRealSearchRunner(
            settings=settings,
            config=config,
            trainer=trainer,
            recording_provider=provider,
            run_dir=directory,
            cuda_environment=cuda_environment,
            evaluation_records=evaluations,
            step_metrics=metrics,
            trajectory_diagnostics=trajectory_diagnostics,
        )
        runner.exhaustive_registry = registry
        return runner
    except Exception:
        registry.close()
        raise


def resume_no_anchor_real_search_runner(
    run_dir: str | Path,
    *,
    data_config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> NoAnchorRealSearchRunner:
    """Resume only the matching formal no-anchor schema and frozen fingerprint."""

    return _resume_no_anchor_runner(
        run_dir,
        expected_schema=NO_ANCHOR_REAL_SEARCH_SCHEMA,
        config=build_frozen_stage5_no_anchor_6_20_config(),
        data_config=data_config,
        paths=paths,
    )


def resume_no_anchor_logz_adam_lr2e2_ab_runner(
    run_dir: str | Path,
    *,
    data_config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> NoAnchorRealSearchRunner:
    """Resume only experiment A with its independent schema and fingerprint."""

    return _resume_no_anchor_runner(
        run_dir,
        expected_schema=NO_ANCHOR_LOGZ_ADAM_LR2E2_AB_SEARCH_SCHEMA,
        config=build_stage5_logz_adam_lr2e2_ab_config(),
        data_config=data_config,
        paths=paths,
    )


def resume_no_anchor_logz_sgd_lr1e1_b1_runner(
    run_dir: str | Path,
    *,
    data_config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> NoAnchorRealSearchRunner:
    """Resume only B1 with its frozen SGD optimizer contract."""

    return _resume_no_anchor_runner(
        run_dir,
        expected_schema=NO_ANCHOR_LOGZ_SGD_LR1E1_B1_SEARCH_SCHEMA,
        config=build_stage5_logz_sgd_lr1e1_b1_config(),
        data_config=data_config,
        paths=paths,
        normalizer_optimizer="sgd",
    )


__all__ = [
    "NO_ANCHOR_REAL_SEARCH_SCHEMA",
    "NO_ANCHOR_LOGZ_ADAM_LR2E2_AB_SEARCH_SCHEMA",
    "NO_ANCHOR_LOGZ_SGD_LR1E1_B1_SEARCH_SCHEMA",
    "NoAnchorRealSearchRunner",
    "create_no_anchor_real_search_runner",
    "create_no_anchor_logz_adam_lr2e2_ab_runner",
    "create_no_anchor_logz_sgd_lr1e1_b1_runner",
    "resume_no_anchor_real_search_runner",
    "resume_no_anchor_logz_adam_lr2e2_ab_runner",
    "resume_no_anchor_logz_sgd_lr1e1_b1_runner",
]
