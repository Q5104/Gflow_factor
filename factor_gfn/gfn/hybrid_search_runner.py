"""Small step-aligned runner for the isolated hybrid-variance Trainer."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from .hybrid_checkpoint import (
    HYBRID_CHECKPOINT_SCHEMA,
    HYBRID_OBJECTIVE_MODE,
    load_hybrid_checkpoint,
    save_hybrid_checkpoint,
)
from .hybrid_trainer import HybridUpdateOutput, HybridVarianceTrainer
from .train_candidate_artifact import (
    TRAIN_CANDIDATE_ARTIFACT_FILENAME,
    TRAIN_CANDIDATE_ARTIFACT_SCHEMA,
    TrainCandidateArtifactWriter,
    supports_train_candidate_artifact,
)


HYBRID_VARIANCE_RUNNER_SCHEMA = "factor_gfn.hybrid_variance_runner.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _atomic_write_json(path: Path, payload: Any) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        lines = [
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            for record in records
        ]
        temporary.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read hybrid runner manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("hybrid runner manifest must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"hybrid diagnostics JSONL line {line_number} is invalid"
            ) from error
        if not isinstance(value, dict):
            raise ValueError("hybrid diagnostics JSONL records must be objects")
        records.append(value)
    return records


class HybridVarianceRunner:
    """Persist every successful hybrid update at its committed boundary."""

    def __init__(
        self,
        *,
        trainer: HybridVarianceTrainer,
        run_dir: Path,
        train_artifact_writer: TrainCandidateArtifactWriter | None = None,
    ) -> None:
        if not isinstance(trainer, HybridVarianceTrainer):
            raise TypeError("trainer must be HybridVarianceTrainer")
        self.trainer = trainer
        self.run_dir = run_dir.resolve()
        self.train_artifact_writer = train_artifact_writer

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoint_latest.pt"

    @property
    def diagnostics_path(self) -> Path:
        return self.run_dir / "hybrid_diagnostics.jsonl"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "runner_state.json"

    @property
    def train_candidate_artifact_path(self) -> Path:
        return self.run_dir / TRAIN_CANDIDATE_ARTIFACT_FILENAME

    @property
    def complete(self) -> bool:
        return (
            self.trainer.optimizer_step
            == self.trainer.config.training.total_optimizer_steps
        )

    def _history_manifest(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.trainer.diagnostic_history]

    def _write_state(self) -> None:
        _atomic_write_json(
            self.state_path,
            {
                "schema": HYBRID_VARIANCE_RUNNER_SCHEMA,
                "updated_at_utc": _utc_now(),
                "global_optimizer_step": self.trainer.optimizer_step,
                "total_trajectories_seen": self.trainer.total_trajectories_seen,
                "pending_assignment": (
                    None
                    if self.complete
                    else asdict(self.trainer.complexity_scheduler.peek())
                ),
                "complete": self.complete,
                "latest_checkpoint": str(self.latest_checkpoint_path),
            },
        )

    def run_attempts(self, attempts: int) -> list[HybridUpdateOutput]:
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("attempts must be a positive integer")
        outputs: list[HybridUpdateOutput] = []
        for _ in range(attempts):
            output = self.trainer.train_step()
            outputs.append(output)
            if not output.updated:
                continue
            # train_step has already completed optimizer.step(), scheduler.commit(),
            # and all counters. The checkpoint is the first runner-side write.
            save_hybrid_checkpoint(self.latest_checkpoint_path, self.trainer)
            if self.train_artifact_writer is not None:
                self.train_artifact_writer.commit_update(output, self.trainer)
            _atomic_write_jsonl(self.diagnostics_path, self._history_manifest())
            self._write_state()
        return outputs


def create_hybrid_variance_runner(
    trainer: HybridVarianceTrainer,
    run_dir: str | os.PathLike[str],
) -> HybridVarianceRunner:
    """Create a new isolated run directory without overwriting prior artifacts."""

    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    train_artifact_enabled = supports_train_candidate_artifact(
        trainer.reward_provider
    )
    _atomic_write_json(
        directory / "hybrid_run_config.json",
        {
            "schema": HYBRID_VARIANCE_RUNNER_SCHEMA,
            "created_at_utc": _utc_now(),
            "checkpoint_schema": HYBRID_CHECKPOINT_SCHEMA,
            "objective_mode": HYBRID_OBJECTIVE_MODE,
            "config_fingerprint": trainer.config.fingerprint(),
            "reward_provider_fingerprint": trainer.reward_provider.fingerprint(),
            "optimizer_contract": _json_normalized(trainer.optimizer_contract()),
            "train_candidate_artifact": {
                "enabled": train_artifact_enabled,
                "schema": TRAIN_CANDIDATE_ARTIFACT_SCHEMA,
                "filename": TRAIN_CANDIDATE_ARTIFACT_FILENAME,
            },
        },
    )
    save_hybrid_checkpoint(directory / "checkpoint_latest.pt", trainer)
    train_artifact_writer = (
        TrainCandidateArtifactWriter(
            run_dir=directory,
            provider=trainer.reward_provider,
            expected_optimizer_step=trainer.optimizer_step,
            create=True,
        )
        if train_artifact_enabled
        else None
    )
    runner = HybridVarianceRunner(
        trainer=trainer,
        run_dir=directory,
        train_artifact_writer=train_artifact_writer,
    )
    _atomic_write_jsonl(runner.diagnostics_path, runner._history_manifest())
    runner._write_state()
    return runner


def resume_hybrid_variance_runner(
    run_dir: str | os.PathLike[str],
    trainer: HybridVarianceTrainer,
) -> HybridVarianceRunner:
    """Resume a matching hybrid run with its checkpoint as the authority."""

    directory = Path(run_dir).resolve()
    manifest = _read_json(directory / "hybrid_run_config.json")
    if manifest.get("schema") != HYBRID_VARIANCE_RUNNER_SCHEMA:
        raise ValueError("hybrid runner schema is incompatible")
    if manifest.get("checkpoint_schema") != HYBRID_CHECKPOINT_SCHEMA:
        raise ValueError("hybrid runner checkpoint schema is incompatible")
    if manifest.get("objective_mode") != HYBRID_OBJECTIVE_MODE:
        raise ValueError("hybrid runner objective mode is incompatible")
    if manifest.get("config_fingerprint") != trainer.config.fingerprint():
        raise ValueError("hybrid runner config fingerprint mismatch")
    if (
        manifest.get("reward_provider_fingerprint")
        != trainer.reward_provider.fingerprint()
    ):
        raise ValueError("hybrid runner Reward provider fingerprint mismatch")
    if manifest.get("optimizer_contract") != _json_normalized(
        trainer.optimizer_contract()
    ):
        raise ValueError("hybrid runner optimizer contract mismatch")

    artifact_manifest = manifest.get("train_candidate_artifact")
    if artifact_manifest is None:
        # Pre-B1 Hybrid runs remain resumable as legacy runs without an artifact.
        train_artifact_enabled = False
    else:
        if not isinstance(artifact_manifest, dict):
            raise ValueError("hybrid runner Train artifact declaration is malformed")
        train_artifact_enabled = supports_train_candidate_artifact(
            trainer.reward_provider
        )
        if artifact_manifest.get("enabled") is not train_artifact_enabled:
            raise ValueError("hybrid runner Train artifact capability mismatch")
        if artifact_manifest.get("schema") != TRAIN_CANDIDATE_ARTIFACT_SCHEMA:
            raise ValueError("hybrid runner Train artifact schema mismatch")
        if artifact_manifest.get("filename") != TRAIN_CANDIDATE_ARTIFACT_FILENAME:
            raise ValueError("hybrid runner Train artifact filename mismatch")

    runner = HybridVarianceRunner(trainer=trainer, run_dir=directory)
    load_hybrid_checkpoint(runner.latest_checkpoint_path, trainer)
    if train_artifact_enabled:
        runner.train_artifact_writer = TrainCandidateArtifactWriter(
            run_dir=directory,
            provider=trainer.reward_provider,
            expected_optimizer_step=trainer.optimizer_step,
            create=False,
        )
    checkpoint_history = runner._history_manifest()
    persisted_history = _read_jsonl(runner.diagnostics_path)
    if persisted_history != checkpoint_history:
        if persisted_history != checkpoint_history[: len(persisted_history)]:
            raise ValueError("hybrid diagnostics diverge from checkpoint history")
        _atomic_write_jsonl(runner.diagnostics_path, checkpoint_history)
    runner._write_state()
    return runner


__all__ = [
    "HYBRID_VARIANCE_RUNNER_SCHEMA",
    "TRAIN_CANDIDATE_ARTIFACT_FILENAME",
    "HybridVarianceRunner",
    "create_hybrid_variance_runner",
    "resume_hybrid_variance_runner",
]
