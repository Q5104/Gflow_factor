"""Immutable candidate-source snapshots for the provisional Stage 6 pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping


CANDIDATE_SOURCE_SCHEMA = "factor_gfn.stage6_candidate_source.v1"
SOURCE_SNAPSHOT_SCHEMA = "factor_gfn.stage6_source_snapshot.v1"
SOURCE_SET_SCHEMA = "factor_gfn.stage6_source_set.v1"

SourceType = Literal[
    "exhaustive_registry",
    "discovery_run",
    "diagnostic_audit",
    "hybrid_train_artifact",
]
SourceRole = Literal["exhaustive", "formal_discovery", "historical_discovery"]
InclusionStatus = Literal["approved", "pending_review", "excluded"]
SourceSetMode = Literal["provisional", "final"]

_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint_safe(value: Any) -> Any:
    """Represent legacy NaN/Inf deterministically without changing source bytes."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {"__factor_gfn_nonfinite_float__": label}
    if isinstance(value, Mapping):
        return {str(key): _fingerprint_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_safe(item) for item in value]
    return value


def _stable_json(value: Any) -> bytes:
    return json.dumps(
        _fingerprint_safe(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 文件无法解析：{path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON 文件必须是对象：{path}")
    return value


def _read_json_bytes(data: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON 文件无法解析：{path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON 文件必须是对象：{path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nested_get(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


@dataclass(frozen=True, slots=True)
class CandidateSourceSpec:
    """Human-approved description of one physical candidate source."""

    source_id: str
    source_type: SourceType
    source_role: SourceRole
    source_path: str | Path
    inclusion_status: InclusionStatus = "approved"
    approval_note: str = ""
    included_branches: tuple[str, ...] = ("main",)
    included_record_sources: tuple[str, ...] = ()
    excluded_record_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SOURCE_ID_RE.fullmatch(self.source_id):
            raise ValueError(
                "source_id 只能包含 ASCII 字母、数字、点、下划线和连字符"
            )
        if not self.approval_note.strip():
            raise ValueError("每个来源必须记录非空 approval_note")
        for name, values in (
            ("included_branches", self.included_branches),
            ("included_record_sources", self.included_record_sources),
            ("excluded_record_sources", self.excluded_record_sources),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} 不能包含重复值")
        overlap = set(self.included_record_sources) & set(self.excluded_record_sources)
        if overlap:
            raise ValueError(f"record source 同时包含和排除：{sorted(overlap)}")

    @property
    def resolved_path(self) -> Path:
        return Path(self.source_path).resolve()

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = CANDIDATE_SOURCE_SCHEMA
        value["source_path"] = str(self.resolved_path)
        value["included_branches"] = list(self.included_branches)
        value["included_record_sources"] = list(self.included_record_sources)
        value["excluded_record_sources"] = list(self.excluded_record_sources)
        return value


def _parse_jsonl_bytes(
    data: bytes,
    *,
    path: Path,
) -> tuple[list[dict[str, Any]], str]:
    if data and not data.endswith(b"\n"):
        raise ValueError(f"JSONL 截止记录没有完整换行：{path}")
    records: list[dict[str, Any]] = []
    logical = hashlib.sha256()
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"JSONL 不允许空行：{path}:{line_number}")
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"JSONL 行无法解析：{path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 行必须是对象：{path}:{line_number}")
        records.append(value)
        logical.update(_stable_json(value))
        logical.update(b"\n")
    return records, logical.hexdigest()


def _read_jsonl_prefix(path: Path, record_count: int) -> bytes:
    if record_count < 0:
        raise ValueError("committed evaluation_records 不能为负数")
    chunks: list[bytes] = []
    with path.open("rb") as stream:
        for index in range(record_count):
            line = stream.readline()
            if not line:
                raise RuntimeError(
                    f"run_state 声明 {record_count} 条评价，但仅存在 {index} 条完整记录"
                )
            if not line.endswith(b"\n"):
                raise RuntimeError(
                    f"第 {index + 1} 条 committed 评价没有完整换行"
                )
            chunks.append(line)
    return b"".join(chunks)


def _record_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    branches: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for record in records:
        branch = record.get("branch")
        source = record.get("source")
        branches["<missing>" if branch is None else str(branch)] += 1
        sources["<missing>" if source is None else str(source)] += 1
    return {
        "records": sum(branches.values()),
        "branches": dict(sorted(branches.items())),
        "record_sources": dict(sorted(sources.items())),
    }


def _reward_fingerprint(metadata: Mapping[str, Any]) -> tuple[str | None, str]:
    reported = metadata.get("reward_fingerprint")
    if isinstance(reported, str) and len(reported) == 64:
        return reported, "reported"
    reward_config = _nested_get(metadata, "reward_provider", "reward_config")
    if isinstance(reward_config, Mapping):
        return _stable_hash(reward_config), "derived_from_reward_config"
    return None, "missing"


def _run_semantics(metadata: Mapping[str, Any]) -> dict[str, Any]:
    reward_fingerprint, reward_method = _reward_fingerprint(metadata)
    seed = _nested_get(metadata, "config_manifest", "config", "training", "seed")
    return {
        "source_artifact_schema": metadata.get("schema"),
        "run_id": metadata.get("run_id"),
        "seed": seed,
        "seed_method": "reported" if seed is not None else "missing",
        "generation_config_fingerprint": metadata.get("config_fingerprint"),
        "provider_fingerprint": metadata.get("reward_provider_fingerprint"),
        "context_fingerprint": _nested_get(
            metadata, "reward_provider", "context_fingerprint"
        ),
        "reward_fingerprint": reward_fingerprint,
        "reward_fingerprint_method": reward_method,
    }


def _snapshot_fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_id": manifest["source_id"],
        "source_type": manifest["source_type"],
        "source_role": manifest["source_role"],
        "inclusion_status": manifest["inclusion_status"],
        "approval_note": manifest["approval_note"],
        "candidate_record_policy": manifest["candidate_record_policy"],
        "source_semantics": manifest["source_semantics"],
        "snapshot_kind": manifest["snapshot_kind"],
        "cutoff": manifest["cutoff"],
        "record_counts": manifest["record_counts"],
        "artifacts": [
            {
                "name": item["name"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in manifest["artifacts"]
        ],
        "logical_content_fingerprint": manifest["logical_content_fingerprint"],
    }
    if "external_artifacts" in manifest:
        payload["external_artifacts"] = manifest["external_artifacts"]
    return payload


def _materialize_directory(
    *,
    spec: CandidateSourceSpec,
    output_root: Path,
    builder: Any,
) -> Path:
    if spec.inclusion_status != "approved":
        raise ValueError(
            f"只有 approved 来源可以物化：{spec.source_id}={spec.inclusion_status}"
        )
    source_parent = output_root.resolve() / "sources" / spec.source_id
    source_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=source_parent))
    try:
        manifest = builder(temporary)
        fingerprint = _stable_hash(_snapshot_fingerprint_payload(manifest))
        manifest["snapshot_fingerprint"] = fingerprint
        manifest["created_at_utc"] = _utc_now()
        manifest["source_path"] = str(spec.resolved_path)
        target = source_parent / fingerprint
        if target.exists():
            existing = _read_json(target / "source_snapshot.json")
            if existing.get("snapshot_fingerprint") != fingerprint:
                raise RuntimeError(f"既有来源快照指纹冲突：{target}")
            return target / "source_snapshot.json"
        _write_json(temporary / "source_snapshot.json", manifest)
        os.replace(temporary, target)
        return target / "source_snapshot.json"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _discovery_snapshot_builder(
    spec: CandidateSourceSpec,
    temporary: Path,
) -> dict[str, Any]:
    directory = spec.resolved_path
    evaluations_path = directory / "evaluations.jsonl"
    metadata_path = directory / "run_metadata.json"
    if not evaluations_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"discovery source 缺少 evaluations.jsonl/run_metadata.json：{directory}"
        )
    metadata_before = metadata_path.read_bytes()
    metadata = _read_json(metadata_path)
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"run_metadata 缺少 run_id：{metadata_path}")
    if directory.name != run_id:
        raise ValueError(f"run 目录名与 run_id 不一致：{directory.name} != {run_id}")

    state_path = directory / "run_state.json"
    state_before = _read_json(state_path) if state_path.is_file() else None
    active = bool(
        state_before is not None
        and (
            state_before.get("status") == "running"
            or state_before.get("active_step") is not None
        )
    )
    size_before = evaluations_path.stat().st_size
    if active:
        if state_before.get("run_id") != run_id:
            raise ValueError("run_state 与 run_metadata 的 run_id 不一致")
        committed = int(state_before.get("evaluation_records", -1))
        data = _read_jsonl_prefix(evaluations_path, committed)
        snapshot_kind = "committed_jsonl_prefix"
        cutoff = {
            "committed_evaluation_records": committed,
            "current_step": int(state_before.get("current_step", -1)),
            "optimizer_step": int(state_before.get("optimizer_step", -1)),
            "step_metric_records": int(state_before.get("step_metric_records", -1)),
            "active_step_at_start": state_before.get("active_step"),
            "state_updated_at_utc": state_before.get("updated_at_utc"),
            "prefix_end_byte_exclusive": len(data),
        }
    else:
        data = evaluations_path.read_bytes()
        snapshot_kind = "stable_full_jsonl"
        full_record_count = len(data.splitlines())
        if (
            state_before is not None
            and state_before.get("evaluation_records") is not None
            and int(state_before["evaluation_records"]) != full_record_count
        ):
            raise RuntimeError(
                "稳定历史 run 的 run_state.evaluation_records 与 JSONL 行数不一致"
            )
        cutoff = {
            "committed_evaluation_records": full_record_count,
            "current_step": (
                int(state_before.get("current_step", -1))
                if state_before is not None
                else None
            ),
            "optimizer_step": (
                int(state_before.get("optimizer_step", -1))
                if state_before is not None
                else None
            ),
            "step_metric_records": (
                int(state_before.get("step_metric_records", -1))
                if state_before is not None
                else None
            ),
            "active_step_at_start": (
                state_before.get("active_step") if state_before is not None else None
            ),
            "state_updated_at_utc": (
                state_before.get("updated_at_utc") if state_before is not None else None
            ),
            "prefix_end_byte_exclusive": len(data),
        }

    records, logical_fingerprint = _parse_jsonl_bytes(data, path=evaluations_path)
    if len(records) != cutoff["committed_evaluation_records"]:
        raise RuntimeError("物化记录数与 committed cutoff 不一致")
    size_after = evaluations_path.stat().st_size
    metadata_after = metadata_path.read_bytes()
    if metadata_after != metadata_before:
        raise RuntimeError("快照期间 run_metadata.json 发生变化")
    if size_after < len(data):
        raise RuntimeError("快照期间 evaluations.jsonl 被截断")
    with evaluations_path.open("rb") as stream:
        if stream.read(len(data)) != data:
            raise RuntimeError("快照期间 committed JSONL 前缀发生变化")

    state_after = _read_json(state_path) if state_path.is_file() else None
    if active:
        assert state_before is not None and state_after is not None
        if state_after.get("run_id") != run_id:
            raise RuntimeError("快照期间 run 身份发生变化")
        for key in ("current_step", "optimizer_step", "evaluation_records"):
            if int(state_after.get(key, -1)) < int(state_before.get(key, -1)):
                raise RuntimeError(f"快照期间 {key} 发生倒退")
    elif size_after != size_before:
        raise RuntimeError("稳定历史 run 的 evaluations.jsonl 在快照期间发生变化")

    (temporary / "evaluations.jsonl").write_bytes(data)
    (temporary / "run_metadata.json").write_bytes(metadata_before)
    if state_before is not None:
        _write_json(temporary / "run_state_at_cutoff.json", state_before)

    artifacts = []
    for name in (
        "evaluations.jsonl",
        "run_metadata.json",
        "run_state_at_cutoff.json",
    ):
        path = temporary / name
        if path.is_file():
            artifacts.append(
                {"name": name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            )
    return {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_id": spec.source_id,
        "source_type": spec.source_type,
        "source_role": spec.source_role,
        "inclusion_status": spec.inclusion_status,
        "approval_note": spec.approval_note,
        "candidate_record_policy": {
            "included_branches": list(spec.included_branches),
            "included_record_sources": list(spec.included_record_sources),
            "excluded_record_sources": list(spec.excluded_record_sources),
        },
        "source_semantics": _run_semantics(metadata),
        "snapshot_kind": snapshot_kind,
        "cutoff": cutoff,
        "source_observation": {
            "size_before": size_before,
            "size_after": size_after,
            "growth_during_snapshot_allowed": active,
        },
        "record_counts": _record_counts(records),
        "artifacts": artifacts,
        "logical_content_fingerprint": logical_fingerprint,
    }


def _diagnostic_snapshot_builder(
    spec: CandidateSourceSpec,
    temporary: Path,
) -> dict[str, Any]:
    directory = spec.resolved_path
    audit_path = directory / "candidate_audit.jsonl"
    context_path = directory / "diagnostic_context.json"
    if not audit_path.is_file() or not context_path.is_file():
        raise FileNotFoundError(
            f"diagnostic source 缺少 candidate_audit.jsonl/diagnostic_context.json：{directory}"
        )
    size_before = audit_path.stat().st_size
    raw_data = audit_path.read_bytes()
    raw_records, _ = _parse_jsonl_bytes(raw_data, path=audit_path)
    raw_lines = raw_data.splitlines(keepends=True)
    selected_lines: list[bytes] = []
    selected_records: list[dict[str, Any]] = []
    included = set(spec.included_record_sources)
    excluded = set(spec.excluded_record_sources)
    for raw_line, record in zip(raw_lines, raw_records, strict=True):
        source = str(record.get("source", ""))
        if included and source not in included:
            continue
        if source in excluded:
            continue
        selected_lines.append(raw_line)
        selected_records.append(record)
    if not selected_records:
        raise ValueError("diagnostic source 过滤后没有候选记录")
    materialized = b"".join(selected_lines)
    _, logical_fingerprint = _parse_jsonl_bytes(materialized, path=audit_path)
    if audit_path.stat().st_size != size_before or audit_path.read_bytes() != raw_data:
        raise RuntimeError("快照期间 candidate_audit.jsonl 发生变化")
    context_data = context_path.read_bytes()
    context = _read_json(context_path)
    (temporary / "candidate_audit.jsonl").write_bytes(materialized)
    (temporary / "diagnostic_context.json").write_bytes(context_data)
    artifacts = [
        {
            "name": name,
            "size_bytes": (temporary / name).stat().st_size,
            "sha256": _sha256_file(temporary / name),
        }
        for name in ("candidate_audit.jsonl", "diagnostic_context.json")
    ]
    semantics = {
        "source_artifact_schema": context.get("schema"),
        "run_id": directory.name,
        "seed": 42 if "seed42" in directory.name.lower() else None,
        "seed_method": (
            "derived_from_source_directory"
            if "seed42" in directory.name.lower()
            else "missing"
        ),
        "generation_config_fingerprint": context.get("config_fingerprint"),
        "provider_fingerprint": context.get("provider_fingerprint"),
        "context_fingerprint": context.get("context_fingerprint"),
        "reward_fingerprint": None,
        "reward_fingerprint_method": "missing",
    }
    return {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_id": spec.source_id,
        "source_type": spec.source_type,
        "source_role": spec.source_role,
        "inclusion_status": spec.inclusion_status,
        "approval_note": spec.approval_note,
        "candidate_record_policy": {
            "included_branches": list(spec.included_branches),
            "included_record_sources": list(spec.included_record_sources),
            "excluded_record_sources": list(spec.excluded_record_sources),
        },
        "source_semantics": semantics,
        "snapshot_kind": "stable_filtered_diagnostic_jsonl",
        "cutoff": {
            "raw_records": len(raw_records),
            "materialized_records": len(selected_records),
            "prefix_end_byte_exclusive": None,
        },
        "source_observation": {
            "size_before": size_before,
            "size_after": audit_path.stat().st_size,
            "growth_during_snapshot_allowed": False,
        },
        "record_counts": _record_counts(selected_records),
        "artifacts": artifacts,
        "logical_content_fingerprint": logical_fingerprint,
    }


def _hybrid_checkpoint_metadata(path: Path) -> dict[str, Any]:
    # Hybrid checkpoints are trusted local run artifacts.  Loading is lazy so
    # legacy Stage 6 source materialization does not acquire a torch dependency.
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Hybrid checkpoint 无法读取：{path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Hybrid checkpoint payload 必须是对象")
    return {
        "schema": payload.get("schema"),
        "objective_mode": payload.get("objective_mode"),
        "config_fingerprint": payload.get("config_fingerprint"),
        "reward_provider_fingerprint": payload.get(
            "reward_provider_fingerprint"
        ),
        "global_optimizer_step": payload.get("global_optimizer_step"),
        "total_trajectories_seen": payload.get("total_trajectories_seen"),
    }


def _hybrid_snapshot_builder(
    spec: CandidateSourceSpec,
    temporary: Path,
) -> dict[str, Any]:
    if spec.source_role != "formal_discovery":
        raise ValueError("Hybrid Train artifact source 必须使用 formal_discovery role")

    directory = spec.resolved_path
    paths = {
        "runner_state.json": directory / "runner_state.json",
        "hybrid_run_config.json": directory / "hybrid_run_config.json",
        "train_candidate_artifact.json": directory
        / "train_candidate_artifact.json",
        "checkpoint_latest.pt": directory / "checkpoint_latest.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Hybrid source 缺少必需 artifact：{missing}")

    json_bytes = {
        name: paths[name].read_bytes()
        for name in (
            "runner_state.json",
            "hybrid_run_config.json",
            "train_candidate_artifact.json",
        )
    }
    runner_state = _read_json_bytes(
        json_bytes["runner_state.json"], path=paths["runner_state.json"]
    )
    run_config = _read_json_bytes(
        json_bytes["hybrid_run_config.json"],
        path=paths["hybrid_run_config.json"],
    )
    artifact = _read_json_bytes(
        json_bytes["train_candidate_artifact.json"],
        path=paths["train_candidate_artifact.json"],
    )

    runner_schema = "factor_gfn.hybrid_variance_runner.v1"
    checkpoint_schema = "factor_gfn.checkpoint.hybrid_variance.v1"
    artifact_schema = "factor_gfn.stage5_train_candidate_artifact.v1"
    record_schema = "factor_gfn.stage5_train_candidate_record.v1"
    contract_schema = "factor_gfn.train_evaluation_contract.v1"
    if runner_state.get("schema") != runner_schema:
        raise ValueError("Hybrid runner_state schema 不兼容")
    if runner_state.get("complete") is not True:
        raise ValueError("Hybrid run 尚未 complete，不能进入 Stage 6")
    if runner_state.get("pending_assignment") is not None:
        raise ValueError("completed Hybrid run 仍有 pending_assignment")
    latest_checkpoint = runner_state.get("latest_checkpoint")
    if (
        not isinstance(latest_checkpoint, str)
        or Path(latest_checkpoint).resolve() != paths["checkpoint_latest.pt"]
    ):
        raise ValueError("Hybrid runner_state latest_checkpoint provenance 不一致")
    if run_config.get("schema") != runner_schema:
        raise ValueError("Hybrid run config schema 不兼容")
    if run_config.get("checkpoint_schema") != checkpoint_schema:
        raise ValueError("Hybrid run config checkpoint schema 不兼容")
    if run_config.get("objective_mode") != "hybrid_variance":
        raise ValueError("Hybrid run objective mode 不兼容")
    artifact_declaration = run_config.get("train_candidate_artifact")
    if not isinstance(artifact_declaration, Mapping) or (
        artifact_declaration.get("enabled") is not True
        or artifact_declaration.get("schema") != artifact_schema
        or artifact_declaration.get("filename") != "train_candidate_artifact.json"
    ):
        raise ValueError("Hybrid run 未声明兼容的 Train candidate artifact")
    if artifact.get("schema") != artifact_schema:
        raise ValueError("Hybrid Train candidate artifact schema 不兼容")

    contract = artifact.get("train_evaluation_contract")
    contract_fingerprint = artifact.get(
        "train_evaluation_contract_fingerprint"
    )
    if not isinstance(contract, Mapping) or contract.get("schema") != contract_schema:
        raise ValueError("Hybrid Train evaluation contract 不兼容")
    if contract_fingerprint != _stable_hash(contract):
        raise ValueError("Hybrid Train evaluation contract fingerprint 不符")
    if contract.get("provider_fingerprint") != run_config.get(
        "reward_provider_fingerprint"
    ):
        raise ValueError("Hybrid artifact/run config Provider fingerprint 不一致")

    source_run = artifact.get("source_run")
    if not isinstance(source_run, Mapping):
        raise ValueError("Hybrid artifact 缺少 source_run provenance")
    if source_run.get("run_directory_name") != directory.name:
        raise ValueError("Hybrid artifact run directory provenance 不一致")
    config_sha256 = _sha256_bytes(json_bytes["hybrid_run_config.json"])
    if source_run.get("hybrid_run_config_sha256") != config_sha256:
        raise ValueError("Hybrid artifact run config SHA256 不一致")

    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError("Hybrid artifact records 必须是列表")
    if artifact.get("candidate_count") != len(records):
        raise ValueError("Hybrid artifact candidate_count 不一致")
    hashes: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Hybrid artifact record 必须是对象")
        if record.get("schema") != record_schema:
            raise ValueError("Hybrid artifact record schema 不兼容")
        if record.get("train_evaluation_contract_fingerprint") != contract_fingerprint:
            raise ValueError("Hybrid artifact record contract fingerprint 不一致")
        structural_hash = record.get("structural_hash")
        if not isinstance(structural_hash, str):
            raise ValueError("Hybrid artifact record 缺少 structural_hash")
        hashes.append(structural_hash)
    if hashes != sorted(hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("Hybrid artifact structural_hash 必须唯一且有序")

    checkpoint_path = paths["checkpoint_latest.pt"]
    checkpoint_size = checkpoint_path.stat().st_size
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint_metadata = _hybrid_checkpoint_metadata(checkpoint_path)
    if checkpoint_metadata.get("schema") != checkpoint_schema:
        raise ValueError("Hybrid checkpoint schema 不兼容")
    if checkpoint_metadata.get("objective_mode") != "hybrid_variance":
        raise ValueError("Hybrid checkpoint objective mode 不兼容")

    runner_step = runner_state.get("global_optimizer_step")
    artifact_step = artifact.get("committed_optimizer_step")
    checkpoint_step = checkpoint_metadata.get("global_optimizer_step")
    if (
        isinstance(runner_step, bool)
        or not isinstance(runner_step, int)
        or runner_step < 0
        or runner_step != artifact_step
        or runner_step != checkpoint_step
    ):
        raise ValueError("Hybrid runner/checkpoint/artifact optimizer step 不一致")
    if checkpoint_metadata.get("config_fingerprint") != run_config.get(
        "config_fingerprint"
    ):
        raise ValueError("Hybrid checkpoint/run config fingerprint 不一致")
    if checkpoint_metadata.get("reward_provider_fingerprint") != run_config.get(
        "reward_provider_fingerprint"
    ):
        raise ValueError("Hybrid checkpoint/run config Provider fingerprint 不一致")
    if checkpoint_metadata.get("total_trajectories_seen") != runner_state.get(
        "total_trajectories_seen"
    ):
        raise ValueError("Hybrid runner/checkpoint trajectory count 不一致")

    for name, data in json_bytes.items():
        if paths[name].read_bytes() != data:
            raise RuntimeError(f"快照期间 Hybrid artifact 发生变化：{name}")
        (temporary / name).write_bytes(data)
    if (
        checkpoint_path.stat().st_size != checkpoint_size
        or _sha256_file(checkpoint_path) != checkpoint_sha256
    ):
        raise RuntimeError("快照期间 Hybrid checkpoint 发生变化")
    _write_json(temporary / "checkpoint_metadata.json", checkpoint_metadata)

    artifacts = [
        {
            "name": name,
            "size_bytes": (temporary / name).stat().st_size,
            "sha256": _sha256_file(temporary / name),
        }
        for name in (
            "runner_state.json",
            "hybrid_run_config.json",
            "train_candidate_artifact.json",
            "checkpoint_metadata.json",
        )
    ]
    train_scope = contract.get("train_scope_projection")
    context_fingerprint = (
        train_scope.get("context_fingerprint")
        if isinstance(train_scope, Mapping)
        else None
    )
    return {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_id": spec.source_id,
        "source_type": spec.source_type,
        "source_role": spec.source_role,
        "inclusion_status": spec.inclusion_status,
        "approval_note": spec.approval_note,
        "candidate_record_policy": {
            "selection": "all_canonical_train_candidate_artifact_records",
            "included_branches": [],
            "included_record_sources": [],
            "excluded_record_sources": [],
        },
        "source_semantics": {
            "source_artifact_schema": artifact_schema,
            "run_id": directory.name,
            "seed": None,
            "seed_method": "not_exposed_by_hybrid_run_manifest",
            "generation_config_fingerprint": run_config.get(
                "config_fingerprint"
            ),
            "provider_fingerprint": run_config.get(
                "reward_provider_fingerprint"
            ),
            "context_fingerprint": context_fingerprint,
            "reward_fingerprint": None,
            "reward_fingerprint_method": "not_applicable_train_artifact",
            "train_evaluation_contract_fingerprint": contract_fingerprint,
        },
        "snapshot_kind": "completed_hybrid_train_artifact",
        "cutoff": {
            "committed_optimizer_step": runner_step,
            "candidate_count": len(records),
            "complete": True,
            "pending_assignment": None,
        },
        "source_observation": {
            "growth_during_snapshot_allowed": False,
            "checkpoint_copied": False,
        },
        "record_counts": {
            "records": len(records),
            "node_counts": dict(
                sorted(
                    Counter(str(record.get("node_count")) for record in records).items()
                )
            ),
        },
        "artifacts": artifacts,
        "external_artifacts": [
            {
                "name": "checkpoint_latest.pt",
                "source_path": str(checkpoint_path),
                "size_bytes": checkpoint_size,
                "sha256": checkpoint_sha256,
                "metadata": checkpoint_metadata,
            }
        ],
        "logical_content_fingerprint": _stable_hash(records),
    }
def _sqlite_logical_fingerprint(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for table, order_by in (
        ("metadata", "key"),
        ("strata", "node_count"),
        ("candidates", "structural_hash"),
    ):
        columns = [
            row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
        ]
        if not columns:
            raise ValueError(f"exhaustive registry 缺少表：{table}")
        digest.update(_stable_json({"table": table, "columns": columns}))
        for row in connection.execute(
            f'SELECT * FROM "{table}" ORDER BY {order_by}'
        ):
            digest.update(_stable_json(list(row)))
            digest.update(b"\n")
    return digest.hexdigest()


def _exhaustive_snapshot_builder(
    spec: CandidateSourceSpec,
    temporary: Path,
) -> dict[str, Any]:
    source = spec.resolved_path
    if source.is_dir():
        source = source / "exhaustive_registry.sqlite3"
    if not source.is_file():
        raise FileNotFoundError(source)
    source_size = source.stat().st_size
    source_uri = f"file:{source.as_posix()}?mode=ro"
    source_connection = sqlite3.connect(source_uri, uri=True)
    target = temporary / "exhaustive_registry.sqlite3"
    target_connection = sqlite3.connect(target)
    try:
        quick_check = source_connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise ValueError(f"exhaustive registry quick_check 失败：{quick_check}")
        schema_row = source_connection.execute(
            "SELECT value_json FROM metadata WHERE key='schema'"
        ).fetchone()
        if schema_row is None or json.loads(schema_row[0]) != "factor_gfn.exhaustive_registry.v2":
            raise ValueError("exhaustive registry schema 不兼容")
        strata = source_connection.execute(
            "SELECT node_count, expected_canonical_count, enumeration_complete, exact_status "
            "FROM strata ORDER BY node_count"
        ).fetchall()
        if strata != [(1, 6, 1, "complete"), (2, 636, 1, "complete")]:
            raise ValueError(f"N=1/2 exhaustive strata 不完整：{strata}")
        counts = source_connection.execute(
            "SELECT node_count, COUNT(*), SUM(status='evaluated') "
            "FROM candidates GROUP BY node_count ORDER BY node_count"
        ).fetchall()
        if counts != [(1, 6, 6), (2, 636, 636)]:
            raise ValueError(f"N=1/2 exhaustive candidate count 不完整：{counts}")
        semantic_rows = source_connection.execute(
            "SELECT DISTINCT provider_fingerprint, context_fingerprint FROM candidates"
        ).fetchall()
        if len(semantic_rows) != 1:
            raise ValueError(
                f"exhaustive registry 包含多个 Provider/Context：{semantic_rows}"
            )
        plan_fingerprint_row = source_connection.execute(
            "SELECT value_json FROM metadata WHERE key='plan_fingerprint'"
        ).fetchone()
        logical_fingerprint = _sqlite_logical_fingerprint(source_connection)
        source_connection.backup(target_connection)
    finally:
        target_connection.close()
        source_connection.close()
    if source.stat().st_size != source_size:
        raise RuntimeError("快照期间 exhaustive registry 文件大小发生变化")
    copied = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True)
    try:
        if _sqlite_logical_fingerprint(copied) != logical_fingerprint:
            raise RuntimeError("exhaustive registry backup 逻辑内容不一致")
    finally:
        copied.close()
    artifact = {
        "name": target.name,
        "size_bytes": target.stat().st_size,
        "sha256": _sha256_file(target),
    }
    return {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_id": spec.source_id,
        "source_type": spec.source_type,
        "source_role": spec.source_role,
        "inclusion_status": spec.inclusion_status,
        "approval_note": spec.approval_note,
        "candidate_record_policy": {
            "included_branches": list(spec.included_branches),
            "included_record_sources": list(spec.included_record_sources),
            "excluded_record_sources": list(spec.excluded_record_sources),
        },
        "source_semantics": {
            "source_artifact_schema": "factor_gfn.exhaustive_registry.v2",
            "run_id": source.parent.name,
            "seed": 42 if "seed42" in source.parent.name.lower() else None,
            "seed_method": (
                "derived_from_source_directory"
                if "seed42" in source.parent.name.lower()
                else "missing"
            ),
            "generation_config_fingerprint": (
                json.loads(plan_fingerprint_row[0])
                if plan_fingerprint_row is not None
                else None
            ),
            "provider_fingerprint": semantic_rows[0][0],
            "context_fingerprint": semantic_rows[0][1],
            "reward_fingerprint": None,
            "reward_fingerprint_method": "missing",
        },
        "snapshot_kind": "sqlite_consistent_backup",
        "cutoff": {
            "node_counts": [1, 2],
            "expected_canonical_counts": {"1": 6, "2": 636},
            "materialized_records": 642,
        },
        "source_observation": {
            "size_before": source_size,
            "size_after": source.stat().st_size,
            "growth_during_snapshot_allowed": False,
            "source_file_sha256": _sha256_file(source),
        },
        "record_counts": {
            "records": 642,
            "node_counts": {"1": 6, "2": 636},
            "statuses": {"evaluated": 642},
        },
        "artifacts": [artifact],
        "logical_content_fingerprint": logical_fingerprint,
    }


def materialize_candidate_source(
    spec: CandidateSourceSpec,
    output_root: str | Path,
) -> Path:
    """Materialize one approved source without writing into its source directory."""

    root = Path(output_root).resolve()
    source = spec.resolved_path
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("snapshot output 与 source 目录不得互为父子目录")
    if spec.source_type == "discovery_run":
        builder = lambda temporary: _discovery_snapshot_builder(spec, temporary)
    elif spec.source_type == "diagnostic_audit":
        builder = lambda temporary: _diagnostic_snapshot_builder(spec, temporary)
    elif spec.source_type == "exhaustive_registry":
        builder = lambda temporary: _exhaustive_snapshot_builder(spec, temporary)
    elif spec.source_type == "hybrid_train_artifact":
        builder = lambda temporary: _hybrid_snapshot_builder(spec, temporary)
    else:  # pragma: no cover - Literal plus dataclass validation protects this path.
        raise ValueError(f"未知 source_type：{spec.source_type}")
    return _materialize_directory(
        spec=spec,
        output_root=root,
        builder=builder,
    )


def materialize_source_set(
    specs: Iterable[CandidateSourceSpec],
    output_root: str | Path,
    *,
    mode: SourceSetMode = "provisional",
) -> Path:
    """Materialize approved sources and freeze their order-independent source set."""

    sources = tuple(specs)
    if not sources:
        raise ValueError("source set 至少需要一个来源")
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("source set 的 source_id 必须唯一")
    invalid = [
        f"{source.source_id}:{source.inclusion_status}"
        for source in sources
        if source.inclusion_status != "approved"
    ]
    if invalid:
        raise ValueError(f"source set 包含未批准来源：{invalid}")
    hybrid_sources = [
        source for source in sources if source.source_type == "hybrid_train_artifact"
    ]
    if hybrid_sources and (len(sources) != 1 or len(hybrid_sources) != 1):
        raise ValueError("Hybrid Stage 6 source set 必须只包含一个 Hybrid source")
    root = Path(output_root).resolve()
    entries: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda item: item.source_id):
        manifest_path = materialize_candidate_source(source, root)
        manifest = _read_json(manifest_path)
        entries.append(
            {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_role": source.source_role,
                "snapshot_fingerprint": manifest["snapshot_fingerprint"],
                "snapshot_manifest": str(manifest_path),
            }
        )
    fingerprint_payload = {
        "schema": SOURCE_SET_SCHEMA,
        "mode": mode,
        "sources": [
            {key: value for key, value in entry.items() if key != "snapshot_manifest"}
            for entry in entries
        ],
    }
    fingerprint = _stable_hash(fingerprint_payload)
    target = root / "source_sets" / fingerprint / "source_set_manifest.json"
    payload = {
        **fingerprint_payload,
        "source_set_fingerprint": fingerprint,
        "created_at_utc": _utc_now(),
        "source_manifests": entries,
    }
    if target.is_file():
        existing = _read_json(target)
        if existing.get("source_set_fingerprint") != fingerprint:
            raise RuntimeError(f"既有 source-set manifest 指纹冲突：{target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=False)
    _write_json(target, payload)
    return target


__all__ = [
    "CANDIDATE_SOURCE_SCHEMA",
    "SOURCE_SET_SCHEMA",
    "SOURCE_SNAPSHOT_SCHEMA",
    "CandidateSourceSpec",
    "materialize_candidate_source",
    "materialize_source_set",
]
