"""Candidate registry and provenance audit for stage five."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from factor_gfn.grammar import Expression


CANDIDATE_REGISTRY_SCHEMA = "factor_gfn.stage5_candidate_registry.v1"


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateOrigin:
    run_id: str
    run_dir: str
    source_line: int
    request_index: int
    branch: str
    phase: str
    provider_fingerprint: str
    compatibility_group: str
    old_valid: bool
    old_reward: float | None
    old_rejection_reason: str | None
    old_reward_result: Mapping[str, Any] | None
    has_neutralization_skip_diagnostics: bool


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    formula: str
    prefix_token_ids: tuple[int, ...]
    structural_hash: str
    node_count: int
    depth: int
    train_long_direction: None
    origins: tuple[CandidateOrigin, ...]


@dataclass(frozen=True, slots=True)
class RunImportAudit:
    run_id: str
    run_dir: str
    compatibility_group: str
    total_evaluation_rows: int
    candidate_rows: int
    ignored_rows: int
    optional_files_present: tuple[str, ...]
    optional_files_missing: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateRegistry:
    candidates: tuple[CandidateRecord, ...]
    run_audits: tuple[RunImportAudit, ...]
    compatibility_groups: Mapping[str, tuple[str, ...]]
    fingerprint: str

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_REGISTRY_SCHEMA,
            "fingerprint": self.fingerprint,
            "candidate_count": len(self.candidates),
            "source_run_count": len(self.run_audits),
            "compatibility_groups": {
                key: list(run_ids) for key, run_ids in self.compatibility_groups.items()
            },
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "run_audits": [asdict(audit) for audit in self.run_audits],
        }


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON 对象")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _require_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 文件无法解析：{path}: {error}") from error


def _compatibility_manifest(metadata: Mapping[str, Any]) -> dict[str, Any]:
    provider = _require_mapping(metadata.get("reward_provider"), "reward_provider")
    context_fingerprint = provider.get("context_fingerprint")
    provider_fingerprint = metadata.get("reward_provider_fingerprint")
    if not isinstance(context_fingerprint, str) or len(context_fingerprint) != 64:
        raise ValueError("reward_provider 缺少有效 context_fingerprint")
    if not isinstance(provider_fingerprint, str) or len(provider_fingerprint) != 64:
        raise ValueError("run_metadata 缺少有效 reward_provider_fingerprint")
    evaluation_config = _require_mapping(
        provider.get("evaluation_config"), "reward_provider.evaluation_config"
    )
    reward_config = _require_mapping(
        provider.get("reward_config"), "reward_provider.reward_config"
    )
    return {
        "context_fingerprint": context_fingerprint,
        "provider_fingerprint": provider_fingerprint,
        "evaluation_config": evaluation_config,
        "reward_config": reward_config,
    }


def _audit_expression_row(
    row: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    expected_provider_fingerprint: str,
) -> tuple[Expression, dict[str, Any], dict[str, Any] | None]:
    label = f"{path}:{line_number}"
    required = (
        "formula",
        "prefix_token_ids",
        "structural_hash",
        "node_count",
        "depth",
        "request_index",
        "branch",
        "phase",
        "valid",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{label} 缺少字段：{missing}")
    try:
        prefix = tuple(int(token) for token in row["prefix_token_ids"])
        expression = Expression.from_prefix(prefix)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} prefix_token_ids 无法重建表达式") from error
    if expression.to_formula() != row["formula"]:
        raise ValueError(f"{label} 公式与 prefix Token 重建结果不一致")
    if expression.to_prefix() != prefix:
        raise ValueError(f"{label} prefix Token 未能无损往返")
    if expression.structural_hash() != row["structural_hash"]:
        raise ValueError(f"{label} 结构哈希与重建表达式不一致")
    if expression.stats.node_count != row["node_count"]:
        raise ValueError(f"{label} node_count 与重建表达式不一致")
    if expression.stats.depth != row["depth"]:
        raise ValueError(f"{label} depth 与重建表达式不一致")

    assignment = _require_mapping(row.get("metadata"), f"{label} metadata")
    comparisons = {
        "formula": expression.to_formula(),
        "prefix_token_ids": list(prefix),
        "expression_hash": expression.structural_hash(),
        "node_count": expression.stats.node_count,
        "depth": expression.stats.depth,
    }
    for key, expected in comparisons.items():
        if assignment.get(key) != expected:
            raise ValueError(f"{label} metadata.{key} 与顶层记录不一致")
    provider_fingerprint = assignment.get("provider_fingerprint")
    if provider_fingerprint != expected_provider_fingerprint:
        raise ValueError(f"{label} Provider 指纹与 run_metadata.json 不一致")
    reward_result_value = assignment.get("reward_result")
    reward_result = (
        _require_mapping(reward_result_value, f"{label} reward_result")
        if reward_result_value is not None
        else None
    )
    if reward_result is not None and reward_result.get("expression_hash") != expression.structural_hash():
        raise ValueError(f"{label} Reward 拆解中的表达式哈希不一致")
    return expression, assignment, reward_result


def import_candidate_runs(
    run_dirs: Iterable[str | Path],
    *,
    allow_mixed_contexts: bool = False,
    candidate_branches: tuple[str, ...] = ("main",),
) -> CandidateRegistry:
    """Import, audit, and structurally deduplicate stage-four candidates."""

    directories = tuple(Path(path).resolve() for path in run_dirs)
    if not directories:
        raise ValueError("至少需要一个候选来源运行目录")
    if not candidate_branches or len(set(candidate_branches)) != len(candidate_branches):
        raise ValueError("candidate_branches 必须是非空且不重复的分支名")

    candidates: dict[str, tuple[Expression, list[CandidateOrigin]]] = {}
    audits: list[RunImportAudit] = []
    groups: dict[str, list[str]] = {}
    optional_names = (
        "experiment_manifest.json",
        "best_candidate.json",
        "training_stats.json",
        "determinism_report.json",
    )
    seen_dirs: set[Path] = set()
    seen_run_ids: set[str] = set()
    for run_dir in directories:
        if run_dir in seen_dirs:
            raise ValueError(f"重复的候选来源运行目录：{run_dir}")
        seen_dirs.add(run_dir)
        evaluations_path = run_dir / "evaluations.jsonl"
        metadata_path = run_dir / "run_metadata.json"
        missing = [str(path) for path in (evaluations_path, metadata_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"候选来源缺少强制文件：{missing}")
        metadata = _read_json(metadata_path)
        run_id = str(metadata.get("run_id", ""))
        if not run_id:
            raise ValueError(f"{metadata_path} 缺少 run_id")
        if run_id in seen_run_ids:
            raise ValueError(f"不同来源目录使用了重复 run_id：{run_id}")
        seen_run_ids.add(run_id)
        expected_provider = str(metadata.get("reward_provider_fingerprint", ""))
        if len(expected_provider) != 64:
            raise ValueError(f"{metadata_path} 缺少有效 reward_provider_fingerprint")
        compatibility_group = _stable_hash(_compatibility_manifest(metadata))
        groups.setdefault(compatibility_group, []).append(run_id)

        warnings: list[str] = []
        total_rows = 0
        candidate_rows = 0
        seen_hashes_all: set[str] = set()
        for line_number, raw_line in enumerate(
            evaluations_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            total_rows += 1
            try:
                row = _require_mapping(json.loads(raw_line), f"{evaluations_path}:{line_number}")
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 行无法解析：{evaluations_path}:{line_number}: {error}"
                ) from error
            expression, assignment, reward_result = _audit_expression_row(
                row,
                path=evaluations_path,
                line_number=line_number,
                expected_provider_fingerprint=expected_provider,
            )
            expression_hash = expression.structural_hash()
            seen_hashes_all.add(expression_hash)
            if str(row["branch"]) not in candidate_branches:
                continue
            candidate_rows += 1
            old_reward_value = row.get("reward")
            origin = CandidateOrigin(
                run_id=run_id,
                run_dir=str(run_dir),
                source_line=line_number,
                request_index=int(row["request_index"]),
                branch=str(row["branch"]),
                phase=str(row["phase"]),
                provider_fingerprint=str(assignment["provider_fingerprint"]),
                compatibility_group=compatibility_group,
                old_valid=bool(row["valid"]),
                old_reward=(float(old_reward_value) if old_reward_value is not None else None),
                old_rejection_reason=row.get("rejection_reason"),
                old_reward_result=(
                    deepcopy(reward_result)
                    if reward_result is not None
                    else None
                ),
                has_neutralization_skip_diagnostics=bool(
                    reward_result is not None
                    and "neutralization_skipped_dates" in reward_result
                    and "neutralization_skipped_rate" in reward_result
                ),
            )
            existing = candidates.get(expression_hash)
            if existing is None:
                candidates[expression_hash] = (expression, [origin])
            else:
                old_expression, origins = existing
                if (
                    old_expression.to_formula() != expression.to_formula()
                    or old_expression.to_prefix() != expression.to_prefix()
                ):
                    raise ValueError(
                        f"结构哈希 {expression_hash} 对应了不同公式或 prefix Token"
                    )
                origins.append(origin)

        present = tuple(name for name in optional_names if (run_dir / name).is_file())
        absent = tuple(name for name in optional_names if not (run_dir / name).is_file())
        manifest_path = run_dir / "experiment_manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if str(manifest.get("run_id", "")) != run_id:
                raise ValueError(f"{manifest_path} 的 run_id 与 run_metadata.json 不一致")
            declared = manifest.get("main_reward_requests")
            main_rows = sum(
                1
                for raw in evaluations_path.read_text(encoding="utf-8").splitlines()
                if raw.strip() and json.loads(raw).get("branch") == "main"
            )
            if declared is not None and int(declared) != main_rows:
                raise ValueError(f"{manifest_path} 的 main_reward_requests 与 JSONL 不一致")
            artifacts = manifest.get("artifacts")
            if isinstance(artifacts, list) and not any(
                Path(str(path)).name == manifest_path.name for path in artifacts
            ):
                warnings.append("历史 experiment_manifest 文件清单未包含自身")
        best_path = run_dir / "best_candidate.json"
        if best_path.is_file():
            best = _read_json(best_path).get("candidate")
            if isinstance(best, dict) and best.get("structural_hash") not in seen_hashes_all:
                raise ValueError(f"{best_path} 的最佳候选未出现在 evaluations.jsonl")
        for optional_name in ("training_stats.json", "determinism_report.json"):
            optional_path = run_dir / optional_name
            if optional_path.is_file():
                _read_json(optional_path)
        if absent:
            warnings.append("可选审计文件缺失，不影响候选导入")
        audits.append(
            RunImportAudit(
                run_id=run_id,
                run_dir=str(run_dir),
                compatibility_group=compatibility_group,
                total_evaluation_rows=total_rows,
                candidate_rows=candidate_rows,
                ignored_rows=total_rows - candidate_rows,
                optional_files_present=present,
                optional_files_missing=absent,
                warnings=tuple(warnings),
            )
        )

    if len(groups) > 1 and not allow_mixed_contexts:
        summary = {key: tuple(value) for key, value in groups.items()}
        raise ValueError(f"候选来源包含不兼容的数据或 Reward 口径：{summary}")

    records: list[CandidateRecord] = []
    for structural_hash in sorted(candidates):
        expression, origins = candidates[structural_hash]
        records.append(
            CandidateRecord(
                formula=expression.to_formula(),
                prefix_token_ids=expression.to_prefix(),
                structural_hash=structural_hash,
                node_count=expression.stats.node_count,
                depth=expression.stats.depth,
                train_long_direction=None,
                origins=tuple(origins),
            )
        )
    fingerprint_payload = {
        "schema": CANDIDATE_REGISTRY_SCHEMA,
        "candidates": [asdict(record) for record in records],
        "groups": {key: sorted(value) for key, value in sorted(groups.items())},
    }
    fingerprint = _stable_hash(fingerprint_payload)
    return CandidateRegistry(
        candidates=tuple(records),
        run_audits=tuple(audits),
        compatibility_groups=MappingProxyType(
            {key: tuple(value) for key, value in sorted(groups.items())}
        ),
        fingerprint=fingerprint,
    )


__all__ = [
    "CANDIDATE_REGISTRY_SCHEMA",
    "CandidateOrigin",
    "CandidateRecord",
    "CandidateRegistry",
    "RunImportAudit",
    "import_candidate_runs",
]
