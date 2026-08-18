"""Manually started observable Stage 6 full evaluation and selection entrypoints.

This module is the Stage 6 step 9D control layer.  It reuses the frozen mixed
evaluator, immutable EvaluationStore, survivor enrichment, and selection
contracts.  It never loads OOS and it does not start work at import time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from factor_gfn.gfn.real_data import RealRewardDataPaths

from .expression_compatibility import (
    ACCEPTED_REGISTRY_SCHEMA,
    EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA,
)
from .stage6_evaluation import (
    Stage6CandidateEvaluator,
    Stage6EvaluationConfig,
    Stage6EvaluationContext,
    _stable_hash,
    build_stage6_evaluation_context,
)
from .stage6_evaluation_store import (
    EvaluationStore,
    EvaluationStoreIntegrityError,
    FrozenEvaluationRun,
    Stage6EvaluationRunner,
)
from .stage6_mixed_evaluation import (
    Stage6TrainReuseOverlay,
)
from .stage6_prefilter_evaluation import Stage6TrainPrefilterEvaluator
from .stage6_survivor_enrichment import (
    Stage6TrainLongExcessEnricher,
    run_stage6_survivor_enrichment_selection,
)


STAGE6_FULL_ENTRY_SCHEMA = "factor_gfn.stage6_full_evaluation_entry.v1"
STAGE6_FULL_PIPELINE_VERSION = "factor_gfn.stage6_full_pipeline.v2"
FULL_EVALUATION_SCOPE = "train_prefilter_then_validation_full_registry"
LEGACY_FULL_EVALUATION_SCOPE = "full_accepted_registry_train_validation_evaluation"


class Stage6FullPipelineIntegrityError(RuntimeError):
    """A 9D registry, run, or entry artifact failed closed validation."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage6FullPipelineIntegrityError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise Stage6FullPipelineIntegrityError(f"JSON artifact is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.strip():
                    raise Stage6FullPipelineIntegrityError(
                        f"blank JSONL row: {path}:{line_number}"
                    )
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise Stage6FullPipelineIntegrityError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                yield value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage6FullPipelineIntegrityError(f"cannot read JSONL: {path}") from error


def load_stage6_accepted_registry(
    compatibility_manifest_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and revalidate the complete immutable AUTO_ACCEPT registry."""

    manifest_path = Path(compatibility_manifest_path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA:
        raise Stage6FullPipelineIntegrityError("compatibility manifest schema mismatch")
    if manifest.get("audit_status") != "complete" or not manifest.get(
        "downstream_eligible"
    ):
        raise Stage6FullPipelineIntegrityError(
            "compatibility registry is incomplete or downstream-ineligible"
        )
    audit_fingerprint = str(manifest.get("audit_fingerprint"))
    if manifest_path.parent.name != audit_fingerprint:
        raise Stage6FullPipelineIntegrityError("compatibility audit path identity mismatch")
    artifact_name = "auto_accepted_candidate_registry.jsonl"
    metadata = manifest.get("artifacts", {}).get(artifact_name)
    registry_path = manifest_path.parent / artifact_name
    if (
        not isinstance(metadata, Mapping)
        or not registry_path.is_file()
        or registry_path.stat().st_size != int(metadata.get("size_bytes", -1))
        or _sha256_file(registry_path) != metadata.get("sha256")
    ):
        raise Stage6FullPipelineIntegrityError("accepted registry artifact changed")
    candidates = list(_iter_jsonl(registry_path))
    if len(candidates) != int(
        manifest.get("counts", {}).get("accepted_registry_candidates", -1)
    ):
        raise Stage6FullPipelineIntegrityError("accepted registry count mismatch")
    if any(row.get("schema") != ACCEPTED_REGISTRY_SCHEMA for row in candidates):
        raise Stage6FullPipelineIntegrityError("accepted registry row schema mismatch")
    digest = _stable_hash(candidates)
    if digest != manifest.get("digests", {}).get("accepted_registry"):
        raise Stage6FullPipelineIntegrityError("accepted registry logical digest mismatch")
    if digest != manifest.get("accepted_registry_fingerprint"):
        raise Stage6FullPipelineIntegrityError("accepted registry fingerprint mismatch")
    return manifest, candidates


@dataclass(frozen=True, slots=True)
class CurrentStage6Pipeline:
    compatibility_manifest: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]
    context: Stage6EvaluationContext
    fresh_evaluator: Stage6CandidateEvaluator
    mixed_evaluator: Stage6TrainPrefilterEvaluator
    enricher: Stage6TrainLongExcessEnricher
    overlay: Stage6TrainReuseOverlay
    prior_full_result_seed_count: int
    prior_full_result_seed_set_fingerprint: str
    prior_full_evaluation_entry_manifest_path: str | None
    prefilter_plan: Mapping[str, int]


def _load_prior_full_result_seeds(
    entry_manifest_path: str | Path | None,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any] | None]:
    if entry_manifest_path is None:
        return {}, None
    entry = load_stage6_full_entry_manifest(entry_manifest_path)
    if entry.get("evaluation_run_scope") != LEGACY_FULL_EVALUATION_SCOPE:
        raise Stage6FullPipelineIntegrityError(
            "prior full-result seeds must come from the superseded full-evaluation run"
        )
    with EvaluationStore(entry["database_path"], entry["run_artifact_root"]) as store:
        verified = store.load_verified_completed_results(entry["evaluation_run_id"])
    records = {
        str(item["structural_hash"]): item["result"] for item in verified.records
    }
    return records, entry


def build_current_stage6_pipeline(
    *,
    compatibility_manifest_path: str | Path,
    overlay_manifest_path: str | Path,
    prior_full_evaluation_entry_manifest_path: str | Path | None = None,
    data_paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> CurrentStage6Pipeline:
    """Build the frozen Train/Validation-only 9D evaluator components."""

    compatibility, candidates = load_stage6_accepted_registry(
        compatibility_manifest_path
    )
    overlay = Stage6TrainReuseOverlay.load(overlay_manifest_path)
    prior_full_results, prior_entry = _load_prior_full_result_seeds(
        prior_full_evaluation_entry_manifest_path
    )
    accepted_fingerprint = str(compatibility["accepted_registry_fingerprint"])
    if (
        prior_entry is not None
        and prior_entry.get("accepted_registry_fingerprint") != accepted_fingerprint
    ):
        raise Stage6FullPipelineIntegrityError(
            "prior full-result registry differs from the current accepted registry"
        )
    if overlay.manifest["accepted_registry_fingerprint"] != accepted_fingerprint:
        raise Stage6FullPipelineIntegrityError("overlay/registry fingerprint mismatch")
    context = build_stage6_evaluation_context(Stage6EvaluationConfig(), paths=data_paths)
    evaluator_kwargs = {
        "compatibility_audit_fingerprint": str(compatibility["audit_fingerprint"]),
        "accepted_registry_fingerprint": accepted_fingerprint,
    }
    fresh = Stage6CandidateEvaluator(context, **evaluator_kwargs)
    mixed = Stage6TrainPrefilterEvaluator(
        fresh,
        overlay,
        prior_full_results=prior_full_results,
    )
    enrichment_fresh = Stage6CandidateEvaluator(context, **evaluator_kwargs)
    enricher = Stage6TrainLongExcessEnricher(enrichment_fresh)
    return CurrentStage6Pipeline(
        compatibility_manifest=MappingProxyType(compatibility),
        candidates=tuple(MappingProxyType(dict(row)) for row in candidates),
        context=context,
        fresh_evaluator=fresh,
        mixed_evaluator=mixed,
        enricher=enricher,
        overlay=overlay,
        prior_full_result_seed_count=len(prior_full_results),
        prior_full_result_seed_set_fingerprint=mixed.prior_seed_set_fingerprint,
        prior_full_evaluation_entry_manifest_path=(
            str(Path(prior_full_evaluation_entry_manifest_path).resolve())
            if prior_entry is not None
            else None
        ),
        prefilter_plan=MappingProxyType(mixed.prefilter_plan(candidates)),
    )


def prepare_stage6_full_evaluation_run(
    store: EvaluationStore,
    pipeline: CurrentStage6Pipeline,
) -> FrozenEvaluationRun:
    """Freeze the complete accepted registry and its order before execution."""

    frozen = store.create_run(
        pipeline.candidates,
        pipeline.mixed_evaluator,
        scope=FULL_EVALUATION_SCOPE,
    )
    if frozen.candidate_count != len(pipeline.candidates):
        raise Stage6FullPipelineIntegrityError("full run candidate count changed")
    if frozen.manifest.get("scope") != FULL_EVALUATION_SCOPE:
        raise Stage6FullPipelineIntegrityError("full run scope mismatch")
    return frozen


def stage6_evaluation_run_snapshot(
    store: EvaluationStore,
    run_id: str,
    *,
    overlay: Stage6TrainReuseOverlay | None = None,
    prior_full_result_hashes: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return a read-only observable snapshot of one frozen evaluation run."""

    manifest = store.get_run_manifest(run_id)
    items = store.run_candidates(run_id)
    verified_completed = store.load_verified_completed_results(run_id)
    completed_results = {
        str(record["structural_hash"]): record["result"]
        for record in verified_completed.records
    }
    states: dict[str, int] = {}
    resolutions: dict[str, int] = {}
    planned_reuse = 0
    completed_reuse = 0
    evaluation_paths: dict[str, int] = {}
    train_prefilter_failed = 0
    validation_completed = 0
    current: dict[str, Any] | None = None
    for item in items:
        state = str(item["state"])
        states[state] = states.get(state, 0) + 1
        resolution = str(item["resolution"] or "unresolved")
        resolutions[resolution] = resolutions.get(resolution, 0) + 1
        reusable = (
            str(item["structural_hash"]) in prior_full_result_hashes
            or (
                overlay is not None
                and str(item["structural_hash"]) in overlay.records
            )
        )
        planned_reuse += int(reusable)
        completed_reuse += int(reusable and state in {"completed", "completed_invalid"})
        if state in {"completed", "completed_invalid"}:
            result = completed_results.get(str(item["structural_hash"]))
            if result is None:
                raise Stage6FullPipelineIntegrityError(
                    "completed run row lacks its immutable cached result"
                )
            source = result.get("source_identity", {})
            path = str(source.get("evaluation_path", "unspecified"))
            evaluation_paths[path] = evaluation_paths.get(path, 0) + 1
            prefilter = result.get("train", {}).get("train_prefilter", {})
            failed_prefilter = prefilter.get("status") == "train_prefilter_failed"
            train_prefilter_failed += int(failed_prefilter)
            validation_completed += int(not failed_prefilter)
        if state == "running":
            current = {
                "ordinal": int(item["ordinal"]),
                "structural_hash": str(item["structural_hash"]),
                "reusable_train": reusable,
            }
    completed = states.get("completed", 0)
    completed_invalid = states.get("completed_invalid", 0)
    completed_total = completed + completed_invalid
    event_counts = {
        str(row["event_type"]): int(row["count"])
        for row in store.connection.execute(
            "SELECT event_type,COUNT(*) AS count FROM run_events "
            "WHERE run_id=? GROUP BY event_type",
            (run_id,),
        ).fetchall()
    }
    return {
        "run_id": run_id,
        "run_scope": manifest.get("scope"),
        "candidate_count": len(items),
        "states": dict(sorted(states.items())),
        "resolutions": dict(sorted(resolutions.items())),
        "validation_completed": validation_completed,
        "train_prefilter_failed": train_prefilter_failed,
        "evaluation_path_counts": dict(sorted(evaluation_paths.items())),
        "evaluation_ineligible": completed_invalid,
        "planned_reused_train": planned_reuse,
        "planned_fresh_train": len(items) - planned_reuse,
        "completed_reused_train": completed_reuse,
        "completed_fresh_train": completed_total - completed_reuse,
        "resume_skipped_events": event_counts.get("resume_skipped", 0),
        "cache_hit_events": event_counts.get("cache_hit", 0),
        "failed": states.get("failed", 0),
        "pending": states.get("pending", 0),
        "current_candidate": current,
        "oos": manifest.get("oos"),
    }


def _entry_manifest(
    *,
    pipeline: CurrentStage6Pipeline,
    frozen: FrozenEvaluationRun,
    database_path: Path,
    run_artifact_root: Path,
) -> dict[str, Any]:
    stable = {
        "schema": STAGE6_FULL_ENTRY_SCHEMA,
        "pipeline_version": STAGE6_FULL_PIPELINE_VERSION,
        "evaluation_run_id": frozen.run_id,
        "evaluation_run_scope": frozen.manifest["scope"],
        "candidate_count": frozen.candidate_count,
        "compatibility_audit_fingerprint": pipeline.compatibility_manifest[
            "audit_fingerprint"
        ],
        "accepted_registry_fingerprint": pipeline.compatibility_manifest[
            "accepted_registry_fingerprint"
        ],
        "train_reuse_overlay_fingerprint": pipeline.overlay.fingerprint,
        "prior_full_result_seed_count": pipeline.prior_full_result_seed_count,
        "prior_full_result_seed_set_fingerprint": (
            pipeline.prior_full_result_seed_set_fingerprint
        ),
        "prior_full_evaluation_entry_manifest_path": (
            pipeline.prior_full_evaluation_entry_manifest_path
        ),
        "train_prefilter_plan": dict(pipeline.prefilter_plan),
        "context_fingerprint": pipeline.context.fingerprint,
        "evaluation_contract_fingerprint": (
            pipeline.mixed_evaluator.evaluation_contract_fingerprint
        ),
        "database_path": str(database_path),
        "run_artifact_root": str(run_artifact_root),
        "oos": "not_loaded_not_evaluated",
    }
    return {**stable, "entry_manifest_fingerprint": _stable_hash(stable)}


def run_current_stage6_full_evaluation(
    *,
    compatibility_manifest_path: str | Path,
    overlay_manifest_path: str | Path,
    prior_full_evaluation_entry_manifest_path: str | Path | None = None,
    output_root: str | Path,
    max_new_evaluations: int | None = None,
    retry_failed: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    data_paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> dict[str, Any]:
    """Create or resume the manually invoked prefiltered full-registry run."""

    root = Path(output_root).resolve()
    database_path = root / "evaluation_store" / "stage6_evaluations.sqlite"
    run_artifact_root = root / "evaluation_runs"
    entry_path = root / "full_evaluation_entry_manifest.json"
    build_started = time.perf_counter()
    if progress_callback is not None:
        progress_callback(
            {
                "event_type": "pipeline_build_started",
                "message": "正在只读加载候选、复用清单、旧结果和统一数据上下文",
                "elapsed_seconds": 0.0,
            }
        )
    pipeline = build_current_stage6_pipeline(
        compatibility_manifest_path=compatibility_manifest_path,
        overlay_manifest_path=overlay_manifest_path,
        prior_full_evaluation_entry_manifest_path=(
            prior_full_evaluation_entry_manifest_path
        ),
        data_paths=data_paths,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event_type": "pipeline_build_completed",
                "message": "新评价合同与 Train 预淘汰分层已冻结",
                "elapsed_seconds": time.perf_counter() - build_started,
                "candidate_count": len(pipeline.candidates),
                "train_prefilter_plan": dict(pipeline.prefilter_plan),
                "prior_full_result_seed_count": (
                    pipeline.prior_full_result_seed_count
                ),
            }
        )
    with EvaluationStore(database_path, run_artifact_root) as store:
        frozen = prepare_stage6_full_evaluation_run(store, pipeline)
        entry = _entry_manifest(
            pipeline=pipeline,
            frozen=frozen,
            database_path=database_path,
            run_artifact_root=run_artifact_root,
        )
        if entry_path.is_file():
            if _read_json(entry_path) != entry:
                raise Stage6FullPipelineIntegrityError(
                    "existing full evaluation entry manifest changed"
                )
        else:
            _write_json_atomic(entry_path, entry)
        summary = Stage6EvaluationRunner(store, pipeline.mixed_evaluator).run(
            frozen.run_id,
            max_new_evaluations=max_new_evaluations,
            retry_failed=retry_failed,
            progress_callback=progress_callback,
        )
        snapshot = stage6_evaluation_run_snapshot(
            store,
            frozen.run_id,
            overlay=pipeline.overlay,
            prior_full_result_hashes=set(
                pipeline.mixed_evaluator.prior_full_results
            ),
        )
    return {
        "entry_manifest_path": str(entry_path),
        "evaluation_run_id": frozen.run_id,
        "invocation": asdict(summary),
        "snapshot": snapshot,
    }


def load_stage6_full_entry_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != STAGE6_FULL_ENTRY_SCHEMA:
        raise Stage6FullPipelineIntegrityError("full entry manifest schema mismatch")
    stable = {
        key: value
        for key, value in manifest.items()
        if key != "entry_manifest_fingerprint"
    }
    if _stable_hash(stable) != manifest.get("entry_manifest_fingerprint"):
        raise Stage6FullPipelineIntegrityError("full entry manifest fingerprint mismatch")
    if manifest.get("evaluation_run_scope") not in {
        FULL_EVALUATION_SCOPE,
        LEGACY_FULL_EVALUATION_SCOPE,
    }:
        raise Stage6FullPipelineIntegrityError("entry is not a full-registry run")
    if manifest.get("oos") != "not_loaded_not_evaluated":
        raise Stage6FullPipelineIntegrityError("entry OOS lock changed")
    return manifest


def load_stage6_mean_ic_distribution(
    *,
    entry_manifest_path: str | Path,
    include_validation: bool = False,
    train_prefilter_pass_only: bool = False,
) -> dict[str, Any]:
    """Read complete screening-eligible results and summarize mean IC values."""

    entry = load_stage6_full_entry_manifest(entry_manifest_path)
    with EvaluationStore(entry["database_path"], entry["run_artifact_root"]) as store:
        verified = store.load_verified_run_results(entry["evaluation_run_id"])
    if verified.manifest.get("scope") != FULL_EVALUATION_SCOPE:
        raise Stage6FullPipelineIntegrityError("IC distribution requires the full run")

    eligible_records = list(verified.records)
    if train_prefilter_pass_only:
        eligible_records = [
            item
            for item in verified.records
            if item["result"].get("train", {})
            .get("train_prefilter", {})
            .get("status")
            == "train_prefilter_passed"
            and item["result"].get("validation", {}).get("availability")
            != "not_evaluated_train_prefilter_failed"
        ]
        if not eligible_records:
            raise Stage6FullPipelineIntegrityError(
                "no Train-prefilter-pass candidates have complete Stage 6 evaluation"
            )

    def collect(split: str) -> tuple[list[float], dict[str, float | int]]:
        values: list[float] = []
        for item in eligible_records:
            value = item["result"][split]["ic"].get("mean")
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        if not values:
            raise Stage6FullPipelineIntegrityError(f"no finite {split} mean IC values")
        array = np.asarray(values, dtype=np.float64)
        stats: dict[str, float | int] = {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "std_ddof0": float(np.std(array, ddof=0)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
        return values, stats

    train_values, train_stats = collect("train")
    result: dict[str, Any] = {
        "evaluation_run_id": verified.run_id,
        "ordered_result_set_fingerprint": verified.ordered_result_set_fingerprint,
        "candidate_scope": (
            "train_prefilter_pass_with_complete_stage6_evaluation"
            if train_prefilter_pass_only
            else "all_completed_evaluation_results"
        ),
        "scope_candidate_count": len(eligible_records),
        "train_ic": {"values": train_values, "statistics": train_stats},
        "oos": "not_loaded_not_evaluated",
    }
    if include_validation:
        validation_values, validation_stats = collect("validation")
        result["validation_ic"] = {
            "values": validation_values,
            "statistics": validation_stats,
        }
    return result


def run_current_stage6_provisional_selection(
    *,
    entry_manifest_path: str | Path,
    compatibility_manifest_path: str | Path,
    overlay_manifest_path: str | Path,
    output_root: str | Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    data_paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> Path:
    """Run the frozen provisional screen/enrichment/decorrelation pipeline."""

    entry = load_stage6_full_entry_manifest(entry_manifest_path)
    pipeline = build_current_stage6_pipeline(
        compatibility_manifest_path=compatibility_manifest_path,
        overlay_manifest_path=overlay_manifest_path,
        prior_full_evaluation_entry_manifest_path=entry.get(
            "prior_full_evaluation_entry_manifest_path"
        ),
        data_paths=data_paths,
    )
    expected = {
        "compatibility_audit_fingerprint": pipeline.compatibility_manifest[
            "audit_fingerprint"
        ],
        "accepted_registry_fingerprint": pipeline.compatibility_manifest[
            "accepted_registry_fingerprint"
        ],
        "train_reuse_overlay_fingerprint": pipeline.overlay.fingerprint,
        "context_fingerprint": pipeline.context.fingerprint,
        "evaluation_contract_fingerprint": (
            pipeline.mixed_evaluator.evaluation_contract_fingerprint
        ),
    }
    mismatches = {
        key: (entry.get(key), value)
        for key, value in expected.items()
        if entry.get(key) != value
    }
    if mismatches:
        raise Stage6FullPipelineIntegrityError(
            f"selection components differ from full evaluation entry: {mismatches}"
        )
    with EvaluationStore(entry["database_path"], entry["run_artifact_root"]) as store:
        manifest = store.validate_run_evaluator(
            entry["evaluation_run_id"], pipeline.mixed_evaluator
        )
        if manifest.get("scope") != FULL_EVALUATION_SCOPE:
            raise EvaluationStoreIntegrityError("selection input is not the full run")
        return run_stage6_survivor_enrichment_selection(
            store=store,
            evaluation_run_id=entry["evaluation_run_id"],
            accepted_candidates=pipeline.candidates,
            enricher=pipeline.enricher,
            output_root=output_root,
            engineering_smoke=False,
            progress_callback=progress_callback,
        )


__all__ = [
    "FULL_EVALUATION_SCOPE",
    "STAGE6_FULL_ENTRY_SCHEMA",
    "STAGE6_FULL_PIPELINE_VERSION",
    "CurrentStage6Pipeline",
    "Stage6FullPipelineIntegrityError",
    "build_current_stage6_pipeline",
    "load_stage6_accepted_registry",
    "load_stage6_full_entry_manifest",
    "load_stage6_mean_ic_distribution",
    "prepare_stage6_full_evaluation_run",
    "run_current_stage6_full_evaluation",
    "run_current_stage6_provisional_selection",
    "stage6_evaluation_run_snapshot",
]
