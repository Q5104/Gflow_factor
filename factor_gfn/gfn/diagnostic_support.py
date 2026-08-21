"""Small, progress-visible helpers for the no-anchor diagnostic notebooks."""

from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Callable

import torch

from factor_gfn.grammar import (
    RAW_ACTION_REGISTRY,
    ActionRegistry,
    Expression,
    SearchSpaceConfig,
)

from .exhaustive import ExhaustivePlanningConfig, ExhaustiveRegistry, resolve_exhaustive_plan
from .targeted_calibration import save_targeted_calibration_progress
from .trainer import GFNTrainer, RewardProvider


Progress = Callable[[str], None]


def _default_progress(message: str) -> None:
    print(message, flush=True)


@contextmanager
def progress_heartbeat(
    label: str,
    *,
    interval_seconds: float = 20.0,
    progress: Progress = _default_progress,
):
    """Print elapsed time during opaque calls that otherwise produce no output."""

    if interval_seconds <= 0.0:
        raise ValueError("heartbeat interval_seconds must be positive")
    stopped = Event()
    started = perf_counter()

    def emit() -> None:
        while not stopped.wait(interval_seconds):
            progress(f"[{label}] still running; elapsed={perf_counter()-started:.1f}s")

    worker = Thread(target=emit, name=f"factor-gfn-{label}-heartbeat", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=max(1.0, interval_seconds))


class PhaseTrackingRewardProvider:
    """Delegate Reward semantics while auditing every phase-tagged request."""

    def __init__(
        self,
        provider: RewardProvider,
        *,
        audit_path: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.current_phase: str | None = None
        self.audit_path = None if audit_path is None else Path(audit_path)
        self.audit_records: list[dict[str, Any]] = []
        if self.audit_path is not None and self.audit_path.is_file():
            self.audit_records = [
                json.loads(line)
                for line in self.audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def fingerprint(self) -> str:
        return self.provider.fingerprint()

    def manifest(self) -> dict[str, Any]:
        return self.provider.manifest()

    @contextmanager
    def phase(self, name: str):
        if self.current_phase is not None:
            raise RuntimeError("Reward diagnostic phases cannot be nested")
        self.current_phase = name
        try:
            yield self
        finally:
            self.current_phase = None

    def evaluate(self, expression: Expression):
        if self.current_phase is None:
            raise RuntimeError("Reward evaluation requires an explicit diagnostic phase")
        assignment = self.provider.evaluate(expression)
        metadata = deepcopy(assignment.metadata)
        reward_result = metadata.get("reward_result", {})
        record = {
                "source": self.current_phase,
                "structural_hash": expression.structural_hash(),
                "node_count": expression.stats.node_count,
                "depth": expression.stats.depth,
                "valid": assignment.valid,
                "rejection_reason": assignment.rejection_reason,
                "reward": assignment.reward if assignment.valid else None,
                "train_ic": reward_result.get("train_ic"),
                "factor_seconds": metadata.get("factor_seconds", 0.0),
                "reward_seconds": metadata.get("reward_seconds", 0.0),
                "provider_cache_hit": bool(metadata.get("provider_cache_hit", False)),
            }
        self.audit_records.append(record)
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        return assignment


def build_or_resume_n1_n2_registry(
    path: str | Path,
    provider: RewardProvider,
    *,
    reward_floor: float,
    progress_every: int = 25,
    progress: Progress = _default_progress,
    action_registry: ActionRegistry = RAW_ACTION_REGISTRY,
    approve_explicit_include_over_budget: bool = False,
) -> ExhaustiveRegistry:
    """Build only the previously approved N=1/2 registry, with resumable progress."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    if not isinstance(action_registry, ActionRegistry):
        raise TypeError("action_registry must be ActionRegistry")
    plan = resolve_exhaustive_plan(
        SearchSpaceConfig(max_depth=2, max_nodes=2),
        ExhaustivePlanningConfig(
            explicit_include_node_counts=(1, 2),
            approve_explicit_include_over_budget=(
                approve_explicit_include_over_budget
            ),
        ),
        action_registry=action_registry,
    )
    registry = ExhaustiveRegistry(path, action_registry=action_registry)
    try:
        manifest = provider.manifest()
        registry.register_plan(
            plan,
            provider_fingerprint=provider.fingerprint(),
            context_fingerprint=manifest["context_fingerprint"],
        )
        pending = registry.pending_candidates()
        total = len(pending)
        completed_before = sum(
            registry.coverage(node_count)["evaluated_count"]
            for node_count in (1, 2)
        )
        started = perf_counter()
        progress(
            f"[exhaustive] pending={total}, completed_before={completed_before}"
        )
        for index, candidate in enumerate(pending, start=1):
            expression = Expression.from_prefix(
                candidate.prefix_token_ids,
                action_registry=action_registry,
            )
            assignment = provider.evaluate(expression)
            if assignment.valid:
                reward_details = dict(assignment.metadata)
                reward_details.setdefault(
                    "reward_result",
                    {
                        "raw_reward": assignment.reward,
                        "reward": assignment.reward,
                        "log_reward": assignment.log_reward,
                    },
                )
                registry.record_evaluation(
                    candidate.structural_hash,
                    valid=True,
                    reward_details=reward_details,
                    target_mass=assignment.reward,
                )
            else:
                registry.record_evaluation(
                    candidate.structural_hash,
                    valid=False,
                    reward_details=dict(assignment.metadata),
                    rejection_reason=assignment.rejection_reason,
                    target_mass=0.0,
                )
            if index % progress_every == 0 or index == total:
                elapsed = perf_counter() - started
                rate = index / elapsed if elapsed else 0.0
                eta = (total - index) / rate if rate else None
                progress(
                    f"[exhaustive] {index}/{total}, elapsed={elapsed:.1f}s, "
                    f"eta={eta:.1f}s"
                    if eta is not None
                    else f"[exhaustive] {index}/{total}"
                )
        for node_count in (1, 2):
            registry.compute_exact_masses(node_count, reward_floor=reward_floor)
            coverage = registry.coverage(node_count)
            progress(f"[exhaustive] N={node_count} coverage={coverage}")
        return registry
    except BaseException:
        registry.close()
        raise


def configure_registry_once(trainer: GFNTrainer, registry: ExhaustiveRegistry) -> None:
    semantics = trainer.target_exhaustive_reuse_semantics()
    proofs = trainer.configure_no_anchor_exhaustive_registry(
        registry,
        source_semantics_by_N={node_count: semantics for node_count in (1, 2)},
    )
    counts = {
        node_count: len(proof.canonical_structural_hashes)
        for node_count, proof in proofs.items()
    }
    _default_progress(f"[registry-proof] verified_once={counts}")


def run_calibration_with_progress(
    trainer: GFNTrainer,
    *,
    progress_every: int = 20,
    checkpoint_path: str | Path | None = None,
    progress: Progress = _default_progress,
) -> dict[int, dict[str, Any]]:
    """Run real calibration, always exposing counts, elapsed time and upper-bound ETA."""

    if trainer.calibration is None:
        raise RuntimeError("trainer calibration is disabled")
    started = perf_counter()
    slots = 0
    result = None
    while result is None:
        next_slot = slots + 1
        if next_slot == 1 or next_slot % progress_every == 0:
            progress(
                f"[calibration] starting slot={next_slot}, "
                f"requested={trainer.calibration.requested_by_N}, "
                f"valid={trainer.calibration.valid_by_N}"
            )
        result = trainer.calibration_step()
        slots += 1
        if slots % progress_every == 0 or result is not None:
            elapsed = perf_counter() - started
            maximum = sum(
                trainer.calibration.maximum_requested_slots_per_N
                for _ in trainer.calibration.node_counts
            )
            rate = slots / elapsed if elapsed else 0.0
            eta = max(0, maximum - slots) / rate if rate else None
            progress(
                f"[calibration] slots={slots}/{maximum}, status={trainer.calibration.status}, "
                f"requested={trainer.calibration.requested_by_N}, "
                f"valid={trainer.calibration.valid_by_N}, elapsed={elapsed:.1f}s, "
                f"upper_bound_eta={eta:.1f}s" if eta is not None else
                f"[calibration] slots={slots}/{maximum}, status={trainer.calibration.status}"
            )
            if checkpoint_path is not None:
                trainer.save_checkpoint(checkpoint_path)
                progress(f"[calibration] checkpoint={Path(checkpoint_path)}")
    return trainer.calibration_report()


def run_targeted_calibration_with_progress(
    trainer: GFNTrainer,
    *,
    progress_checkpoint_path: str | Path,
    progress_every: int = 20,
    progress: Progress = _default_progress,
) -> dict[int, dict[str, Any]]:
    """Run targeted calibration with a heartbeat and calibration-only resume state."""

    if trainer.calibration is None:
        raise RuntimeError("trainer targeted calibration is disabled")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    if trainer.calibration.status == "complete":
        progress("[targeted calibration] restored state is already complete")
        return trainer.calibration_report()
    if trainer.calibration.status != "collecting":
        raise RuntimeError(
            f"targeted calibration cannot continue from status={trainer.calibration.status}"
        )
    started = perf_counter()
    invocation_slots = 0
    result = None
    maximum = (
        trainer.calibration.maximum_requested_slots_per_N
        * len(trainer.calibration.node_counts)
    )
    while result is None:
        requested_before = sum(trainer.calibration.requested_by_N.values())
        progress(
            f"[targeted calibration] starting invocation_slot={invocation_slots + 1}, "
            f"total_requested_before={requested_before}, "
            f"requested={trainer.calibration.requested_by_N}, "
            f"valid={trainer.calibration.valid_by_N}"
        )
        with progress_heartbeat(
            f"targeted calibration slot={requested_before + 1}",
            progress=progress,
            interval_seconds=20.0,
        ):
            result = trainer.calibration_step()
        invocation_slots += 1
        total_requested = sum(trainer.calibration.requested_by_N.values())
        if (
            invocation_slots == 1
            or total_requested % progress_every == 0
            or result is not None
        ):
            elapsed = perf_counter() - started
            rate = invocation_slots / elapsed if elapsed else 0.0
            remaining_upper_bound = max(0, maximum - total_requested)
            eta = remaining_upper_bound / rate if rate else None
            stability = {
                node_count: (
                    None
                    if node_count not in trainer.calibration.stability_by_N
                    else asdict(trainer.calibration.stability_by_N[node_count])
                )
                for node_count in trainer.calibration.node_counts
            }
            suffix = f", upper_bound_eta={eta:.1f}s" if eta is not None else ""
            progress(
                f"[targeted calibration] total_requested={total_requested}/{maximum}, "
                f"status={trainer.calibration.status}, "
                f"requested={trainer.calibration.requested_by_N}, "
                f"valid={trainer.calibration.valid_by_N}, "
                f"sampled_attempts={trainer.calibration.sampled_attempts_by_N}, "
                f"stability={stability}, elapsed={elapsed:.1f}s{suffix}"
            )
        save_targeted_calibration_progress(progress_checkpoint_path, trainer)
        progress(
            f"[targeted calibration] progress_checkpoint={Path(progress_checkpoint_path)}"
        )
    return trainer.calibration_report()


def run_training_with_progress(
    trainer: GFNTrainer,
    *,
    logical_batches: int,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 8,
    training_audit_path: str | Path | None = None,
    trajectory_audit_path: str | Path | None = None,
    progress: Progress = _default_progress,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run short health diagnostics and print after every logical batch."""

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    started = perf_counter()
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    for invocation_batch in range(1, logical_batches + 1):
        logical_batch = trainer.step + 1
        progress(
            f"[training] starting invocation_batch={invocation_batch}/{logical_batches}, "
            f"logical_batch={logical_batch}, "
            f"trainer_step={trainer.step}, successful_updates={trainer.optimizer_step}"
        )
        optimizer_step_before = trainer.optimizer_step
        learned_log_z_pre_update_by_N = {
            node_count: float(trainer.tb_loss.log_z_by_node_count[node_count - 1])
            for node_count in trainer.resolved_learned_node_counts
        }
        batch_started = perf_counter()
        with progress_heartbeat(
            f"training logical_batch={logical_batch}",
            progress=progress,
        ):
            stats = trainer.train_step()
        row = asdict(stats)
        row["logical_batch"] = logical_batch
        row["invocation_batch"] = invocation_batch
        row["optimizer_step_before"] = optimizer_step_before
        row["successful_optimizer_update"] = not stats.skipped_update
        row["batch_wall_seconds"] = perf_counter() - batch_started
        row["timings"] = dict(trainer.last_step_timings)
        row["learned_log_z_pre_update_by_N"] = learned_log_z_pre_update_by_N
        row["learned_log_z_by_N"] = {
            node_count: float(trainer.tb_loss.log_z_by_node_count[node_count - 1])
            for node_count in trainer.resolved_learned_node_counts
        }
        row["cuda_peak_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated(trainer.device))
            if trainer.device.type == "cuda"
            else 0
        )
        rows.append(row)
        batch_candidates = [
            dict(item) for item in trainer.last_discovery_trajectory_diagnostics
        ]
        for candidate_index, candidate in enumerate(batch_candidates):
            candidate["logical_batch"] = logical_batch
            candidate["invocation_batch"] = invocation_batch
            candidate["candidate_index_in_batch"] = candidate_index
            candidate["optimizer_step_before"] = optimizer_step_before
            candidate["optimizer_step_after"] = trainer.optimizer_step
            candidate["successful_gradient_exposure"] = not stats.skipped_update
        candidate_rows.extend(batch_candidates)
        if training_audit_path is not None:
            path = Path(training_audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        if trajectory_audit_path is not None:
            path = Path(trajectory_audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for candidate in batch_candidates:
                    handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                handle.flush()
        elapsed = perf_counter() - started
        rate = invocation_batch / elapsed if elapsed else 0.0
        eta = (logical_batches - invocation_batch) / rate if rate else None
        progress(
            f"[training] invocation_batch={invocation_batch}/{logical_batches}, "
            f"logical_batch={logical_batch}, "
            f"successful_updates={trainer.optimizer_step}, skipped={stats.skipped_update}, "
            f"loss={stats.loss}, tb_rms={stats.tb_delta_rms}, "
            f"requested={stats.requested_count_by_N}, retries={stats.retry_exhausted_count_by_N}, "
            f"elapsed={elapsed:.1f}s, eta={eta:.1f}s" if eta is not None else
            f"[training] batch={logical_batch}/{logical_batches}"
        )
        if checkpoint_path is not None and (
            invocation_batch % checkpoint_every == 0
            or invocation_batch == logical_batches
        ):
            trainer.save_checkpoint(checkpoint_path)
            progress(f"[training] checkpoint={Path(checkpoint_path)}")
    return rows, candidate_rows


def real_discovery_depth_records(
    trajectory_rows: list[dict[str, Any]],
    provider: Any,
) -> list[dict[str, Any]]:
    """Join discovery identities to real evaluation details for depth-only diagnostics."""

    records = {record.expression_hash: record for record in provider.evaluation_records}
    output: list[dict[str, Any]] = []
    for row in trajectory_rows:
        record = records.get(row["structural_hash"])
        if record is None:
            raise RuntimeError(
                "discovery structural hash lacks a RealReward evaluation record"
            )
        output.append(
            {
                "source": "discovery",
                "structural_hash": row["structural_hash"],
                "node_count": row["terminal_node_count"],
                "depth": row["terminal_depth"],
                "valid": bool(record.result.valid),
                "rejection_reason": record.result.invalid_reason,
                "reward": record.result.reward,
                "train_ic": record.result.train_ic,
                "factor_seconds": record.factor_seconds,
                "reward_seconds": record.reward_seconds,
                "provider_cache_hit": False,
            }
        )
    return output


__all__ = [
    "PhaseTrackingRewardProvider",
    "build_or_resume_n1_n2_registry",
    "configure_registry_once",
    "progress_heartbeat",
    "real_discovery_depth_records",
    "run_calibration_with_progress",
    "run_targeted_calibration_with_progress",
    "run_training_with_progress",
]
