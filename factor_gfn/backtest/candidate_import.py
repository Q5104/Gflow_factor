"""Stage 6 candidate normalization and source-claimed hash grouping.

This module deliberately treats a source-provided structural hash as an opaque
claim.  It does not reconstruct expressions, compute canonical hashes, reuse
old metrics for selection, or inspect OOS data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .sources import SOURCE_SET_SCHEMA, SOURCE_SNAPSHOT_SCHEMA


NORMALIZED_ORIGIN_SCHEMA = "factor_gfn.stage6_normalized_candidate_origin.v1"
CANDIDATE_REGISTRY_SCHEMA = "factor_gfn.stage6_claimed_hash_registry.v1"
CANDIDATE_IMPORT_MANIFEST_SCHEMA = "factor_gfn.stage6_candidate_import_manifest.v1"
NORMALIZATION_SCHEMA = "factor_gfn.stage6_candidate_normalization.v1"

DISCOVERY_ADAPTER_VERSION = "factor_gfn.stage6.discovery_jsonl_adapter.v1"
DIAGNOSTIC_ADAPTER_VERSION = "factor_gfn.stage6.diagnostic_jsonl_adapter.v1"
EXHAUSTIVE_ADAPTER_VERSION = "factor_gfn.stage6.exhaustive_sqlite_adapter.v1"
HYBRID_TRAIN_ARTIFACT_ADAPTER_VERSION = (
    "factor_gfn.stage6.hybrid_train_artifact_adapter.v1"
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _stable_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON 文件无法读取：{path}: {error}") from error
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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    os.replace(temporary, path)


def _source_set_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source-set manifest 缺少 sources 列表")
    normalized = []
    for entry in sources:
        if not isinstance(entry, Mapping):
            raise ValueError("source-set sources 必须全部是对象")
        normalized.append(
            {
                "source_id": entry.get("source_id"),
                "source_type": entry.get("source_type"),
                "source_role": entry.get("source_role"),
                "snapshot_fingerprint": entry.get("snapshot_fingerprint"),
            }
        )
    return {
        "schema": SOURCE_SET_SCHEMA,
        "mode": manifest.get("mode"),
        "sources": normalized,
    }


def _snapshot_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("snapshot manifest 缺少 artifacts 列表")
    payload = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_id": manifest.get("source_id"),
        "source_type": manifest.get("source_type"),
        "source_role": manifest.get("source_role"),
        "inclusion_status": manifest.get("inclusion_status"),
        "approval_note": manifest.get("approval_note"),
        "candidate_record_policy": manifest.get("candidate_record_policy"),
        "source_semantics": manifest.get("source_semantics"),
        "snapshot_kind": manifest.get("snapshot_kind"),
        "cutoff": manifest.get("cutoff"),
        "record_counts": manifest.get("record_counts"),
        "artifacts": [
            {
                "name": item.get("name"),
                "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
            }
            for item in artifacts
            if isinstance(item, Mapping)
        ],
        "logical_content_fingerprint": manifest.get(
            "logical_content_fingerprint"
        ),
    }
    if "external_artifacts" in manifest:
        payload["external_artifacts"] = manifest.get("external_artifacts")
    return payload


def _resolve_snapshot_manifest(
    source_set_path: Path,
    entry: Mapping[str, Any],
) -> Path:
    source_manifests = source_set_path.parent.parent.parent / "sources"
    expected = (
        source_manifests
        / str(entry["source_id"])
        / str(entry["snapshot_fingerprint"])
        / "source_snapshot.json"
    )
    if expected.is_file():
        return expected
    reported = entry.get("snapshot_manifest")
    if isinstance(reported, str) and Path(reported).is_file():
        return Path(reported).resolve()
    raise FileNotFoundError(f"找不到冻结来源快照 manifest：{expected}")


def _verify_source_set(source_set_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(source_set_path)
    if manifest.get("schema") != SOURCE_SET_SCHEMA:
        raise ValueError("source-set schema 不受支持")
    claimed = manifest.get("source_set_fingerprint")
    computed = _stable_hash(_source_set_payload(manifest))
    if claimed != computed or source_set_path.parent.name != computed:
        raise RuntimeError("source-set fingerprint 不符")

    details_by_id: dict[str, Mapping[str, Any]] = {}
    for detail in manifest.get("source_manifests", []):
        if not isinstance(detail, Mapping) or not isinstance(detail.get("source_id"), str):
            raise ValueError("source_manifests 条目无效")
        details_by_id[str(detail["source_id"])] = detail

    verified: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        if not isinstance(source, Mapping):
            raise ValueError("sources 条目无效")
        source_id = str(source.get("source_id"))
        detail = details_by_id.get(source_id, source)
        path = _resolve_snapshot_manifest(source_set_path, detail)
        snapshot = _read_json(path)
        if snapshot.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
            raise ValueError(f"snapshot schema 不受支持：{source_id}")
        for key in ("source_id", "source_type", "source_role"):
            if snapshot.get(key) != source.get(key):
                raise RuntimeError(f"source-set/snapshot {key} 不一致：{source_id}")
        if snapshot.get("inclusion_status") != "approved":
            raise RuntimeError(f"snapshot 来源未获批准：{source_id}")
        fingerprint = _stable_hash(_snapshot_payload(snapshot))
        if (
            fingerprint != source.get("snapshot_fingerprint")
            or fingerprint != snapshot.get("snapshot_fingerprint")
            or path.parent.name != fingerprint
        ):
            raise RuntimeError(f"snapshot fingerprint 不符：{source_id}")
        for artifact in snapshot.get("artifacts", []):
            if not isinstance(artifact, Mapping):
                raise ValueError(f"snapshot artifact 条目无效：{source_id}")
            artifact_path = path.parent / str(artifact.get("name"))
            if not artifact_path.is_file():
                raise FileNotFoundError(f"snapshot artifact 缺失：{artifact_path}")
            if (
                artifact_path.stat().st_size != artifact.get("size_bytes")
                or _sha256_file(artifact_path) != artifact.get("sha256")
            ):
                raise RuntimeError(f"snapshot artifact 指纹不符：{artifact_path}")
        for artifact in snapshot.get("external_artifacts", []):
            if not isinstance(artifact, Mapping):
                raise ValueError(f"external snapshot artifact 条目无效：{source_id}")
            reported_path = artifact.get("source_path")
            if not isinstance(reported_path, str):
                raise ValueError(f"external snapshot artifact 缺少路径：{source_id}")
            artifact_path = Path(reported_path).resolve()
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"external snapshot artifact 缺失：{artifact_path}"
                )
            if (
                artifact_path.stat().st_size != artifact.get("size_bytes")
                or _sha256_file(artifact_path) != artifact.get("sha256")
            ):
                raise RuntimeError(
                    f"external snapshot artifact 指纹不符：{artifact_path}"
                )
        verified.append({"source": dict(source), "manifest": snapshot, "path": path})
    return manifest, verified


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} 必须是 >= {minimum} 的整数")
    return value


def _require_record_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    structural_hash = record.get("structural_hash")
    if not isinstance(structural_hash, str) or not _HASH_RE.fullmatch(structural_hash):
        raise ValueError("structural_hash 必须是 64 位小写十六进制字符串")
    formula = record.get("formula")
    if not isinstance(formula, str) or not formula:
        raise ValueError("formula 必须是非空字符串")
    prefix = record.get("prefix_token_ids")
    if (
        not isinstance(prefix, list)
        or not prefix
        or any(isinstance(item, bool) or not isinstance(item, int) for item in prefix)
    ):
        raise ValueError("prefix_token_ids 必须是非空整数列表")
    node_count = _require_int(record.get("node_count"), "node_count", minimum=1)
    depth = _require_int(record.get("depth"), "depth", minimum=0)
    valid = record.get("valid")
    if not isinstance(valid, bool):
        raise ValueError("valid 必须是布尔值")
    reward = record.get("reward")
    if reward is not None and (
        isinstance(reward, bool) or not isinstance(reward, (int, float))
    ):
        raise ValueError("reward 必须是数值或 null")
    rejection_reason = record.get("rejection_reason")
    if rejection_reason is not None and not isinstance(rejection_reason, str):
        raise ValueError("rejection_reason 必须是字符串或 null")
    if isinstance(reward, float) and not math.isfinite(reward):
        if math.isnan(reward):
            reward = {"nonfinite_float": "nan"}
        elif reward > 0:
            reward = {"nonfinite_float": "positive_infinity"}
        else:
            reward = {"nonfinite_float": "negative_infinity"}
    return {
        "source_claimed_structural_hash": structural_hash,
        "formula": formula,
        "prefix_token_ids": prefix,
        "node_count": node_count,
        "depth": depth,
        "old_valid": valid,
        "old_reward": reward,
        "old_rejection_reason": rejection_reason,
    }


def _formal_exact_n_contract(metadata: Mapping[str, Any]) -> bool:
    config_manifest = metadata.get("config_manifest")
    if not isinstance(config_manifest, Mapping):
        return False
    strata = config_manifest.get("strata")
    state_adapter = config_manifest.get("state_adapter")
    return bool(
        config_manifest.get("schema") == "factor_gfn.gfn_config.no_anchor.v1"
        and isinstance(strata, Mapping)
        and strata.get("schema") == "factor_gfn.complexity_conditioned_no_anchor.v1"
        and strata.get("normal_discovery_equals_feasible") is True
        and isinstance(state_adapter, Mapping)
        and state_adapter.get("schema") == "factor_gfn.state_adapter.v2"
        and state_adapter.get("condition_features")
    )


def _normalize_origin(
    common: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    adapter_version: str,
    locator: Mapping[str, Any],
    source_record: Mapping[str, Any],
    target_node_count: int | None,
    target_method: str,
) -> dict[str, Any]:
    representation = {
        "formula": common["formula"],
        "prefix_token_ids": common["prefix_token_ids"],
        "node_count": common["node_count"],
        "depth": common["depth"],
    }
    identity = {
        "source_id": source["source_id"],
        "snapshot_fingerprint": source["snapshot_fingerprint"],
        "locator": locator,
    }
    return {
        "schema": NORMALIZED_ORIGIN_SCHEMA,
        "origin_id": _stable_hash(identity),
        "source_claimed_structural_hash": common[
            "source_claimed_structural_hash"
        ],
        "representation_digest": _stable_hash(representation),
        **representation,
        "target_node_count": target_node_count,
        "target_node_count_method": target_method,
        "provenance": {
            "source_id": source["source_id"],
            "source_type": source["source_type"],
            "source_role": source["source_role"],
            "snapshot_fingerprint": source["snapshot_fingerprint"],
            "snapshot_kind": snapshot["snapshot_kind"],
            "adapter_version": adapter_version,
            "locator": locator,
            "source_record": dict(source_record),
        },
        "old_metric_audit": {
            "old_valid": common["old_valid"],
            "old_reward": common["old_reward"],
            "old_rejection_reason": common["old_rejection_reason"],
            "reuse_for_stage6_selection": False,
        },
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"冻结 JSONL 损坏：{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"冻结 JSONL 行不是对象：{path}:{line_number}")
            yield line_number, value


def _adapt_discovery(
    source: Mapping[str, Any], snapshot: Mapping[str, Any], directory: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = _read_json(directory / "run_metadata.json")
    formal = source["source_role"] == "formal_discovery"
    if formal and not _formal_exact_n_contract(metadata):
        raise RuntimeError("formal discovery 来源不满足已批准的 exact-N no-anchor contract")
    policy = snapshot["candidate_record_policy"]
    included_branches = set(policy.get("included_branches", []))
    origins: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    observed = 0
    for line_number, record in _iter_jsonl(directory / "evaluations.jsonl"):
        if included_branches and record.get("branch") not in included_branches:
            continue
        observed += 1
        locator = {"artifact": "evaluations.jsonl", "line_number": line_number}
        try:
            common = _require_record_fields(record)
            target = common["node_count"] if formal else None
            method = (
                "derived_from_exact_n_no_anchor_formal_contract"
                if formal
                else "unavailable_unconditioned_historical_source"
            )
            origins.append(
                _normalize_origin(
                    common,
                    source=source,
                    snapshot=snapshot,
                    adapter_version=DISCOVERY_ADAPTER_VERSION,
                    locator=locator,
                    source_record={
                        key: record.get(key)
                        for key in (
                            "branch",
                            "request_index",
                            "logical_step",
                            "phase",
                        )
                    },
                    target_node_count=target,
                    target_method=method,
                )
            )
        except ValueError as error:
            rejected.append(_rejection(source, locator, error))
    expected = snapshot["record_counts"]["records"]
    if observed != expected:
        raise RuntimeError(
            f"冻结计数不符：{source['source_id']} expected={expected} observed={observed}"
        )
    return origins, rejected


def _adapt_diagnostic(
    source: Mapping[str, Any], snapshot: Mapping[str, Any], directory: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = snapshot["candidate_record_policy"]
    included = set(policy.get("included_record_sources", []))
    excluded = set(policy.get("excluded_record_sources", []))
    origins: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    observed = 0
    for line_number, record in _iter_jsonl(directory / "candidate_audit.jsonl"):
        record_source = record.get("source")
        if (included and record_source not in included) or record_source in excluded:
            continue
        observed += 1
        locator = {"artifact": "candidate_audit.jsonl", "line_number": line_number}
        try:
            common = _require_record_fields(record)
            origins.append(
                _normalize_origin(
                    common,
                    source=source,
                    snapshot=snapshot,
                    adapter_version=DIAGNOSTIC_ADAPTER_VERSION,
                    locator=locator,
                    source_record={"record_source": record_source},
                    target_node_count=None,
                    target_method="unavailable_historical_diagnostic_source",
                )
            )
        except ValueError as error:
            rejected.append(_rejection(source, locator, error))
    expected = snapshot["record_counts"]["records"]
    if observed != expected:
        raise RuntimeError(
            f"冻结计数不符：{source['source_id']} expected={expected} observed={observed}"
        )
    return origins, rejected


def _adapt_exhaustive(
    source: Mapping[str, Any], snapshot: Mapping[str, Any], directory: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = directory / "exhaustive_registry.sqlite3"
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    origins: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        rows = connection.execute(
            """
            SELECT structural_hash, source, node_count, depth, formula,
                   prefix_token_ids_json, status, valid, rejection_reason,
                   reward_details_json, target_mass
            FROM candidates ORDER BY structural_hash
            """
        )
        observed = 0
        for row in rows:
            observed += 1
            locator = {
                "artifact": "exhaustive_registry.sqlite3",
                "table": "candidates",
                "primary_key": row["structural_hash"],
            }
            try:
                prefix = json.loads(row["prefix_token_ids_json"])
                reward_details = (
                    json.loads(row["reward_details_json"])
                    if row["reward_details_json"] is not None
                    else {}
                )
                reward_result = reward_details.get("reward_result", {})
                record = {
                    "structural_hash": row["structural_hash"],
                    "formula": row["formula"],
                    "prefix_token_ids": prefix,
                    "node_count": row["node_count"],
                    "depth": row["depth"],
                    "valid": (
                        bool(row["valid"])
                        if row["valid"] in (0, 1)
                        else row["valid"]
                    ),
                    "reward": reward_result.get("reward", row["target_mass"]),
                    "rejection_reason": row["rejection_reason"],
                }
                common = _require_record_fields(record)
                origins.append(
                    _normalize_origin(
                        common,
                        source=source,
                        snapshot=snapshot,
                        adapter_version=EXHAUSTIVE_ADAPTER_VERSION,
                        locator=locator,
                        source_record={
                            "record_source": row["source"],
                            "status": row["status"],
                        },
                        target_node_count=None,
                        target_method="not_applicable_exhaustive_enumeration",
                    )
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                rejected.append(_rejection(source, locator, error))
    except sqlite3.DatabaseError as error:
        raise RuntimeError(f"冻结 SQLite 来源损坏：{path}: {error}") from error
    finally:
        connection.close()
    expected = snapshot["record_counts"]["records"]
    if observed != expected:
        raise RuntimeError(
            f"冻结计数不符：{source['source_id']} expected={expected} observed={observed}"
        )
    return origins, rejected


def _adapt_hybrid_train_artifact(
    source: Mapping[str, Any], snapshot: Mapping[str, Any], directory: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifact = _read_json(directory / "train_candidate_artifact.json")
    records = artifact.get("records")
    if not isinstance(records, list):
        raise RuntimeError("冻结 Hybrid Train artifact 缺少 records 列表")
    origins: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        locator = {
            "artifact": "train_candidate_artifact.json",
            "record_index": index,
        }
        try:
            if not isinstance(record, Mapping):
                raise ValueError("Hybrid Train artifact record 必须是对象")
            # These fields only bridge the legacy normalization schema.  They
            # are never Train-pass or Stage 6 selection evidence.
            import_record = {
                "structural_hash": record.get("structural_hash"),
                "formula": record.get("formula"),
                "prefix_token_ids": record.get("prefix_token_ids"),
                "node_count": record.get("node_count"),
                "depth": record.get("depth"),
                "valid": True,
                "reward": None,
                "rejection_reason": None,
            }
            common = _require_record_fields(import_record)
            origin = _normalize_origin(
                common,
                source=source,
                snapshot=snapshot,
                adapter_version=HYBRID_TRAIN_ARTIFACT_ADAPTER_VERSION,
                locator=locator,
                source_record={
                    "record_schema": record.get("schema"),
                    "train_evaluation_contract_fingerprint": record.get(
                        "train_evaluation_contract_fingerprint"
                    ),
                    "first_seen": record.get("first_seen"),
                    "last_seen": record.get("last_seen"),
                    "visit_count": record.get("visit_count"),
                    "normalization_valid_semantics": (
                        "candidate_import_schema_compatibility_only"
                    ),
                },
                target_node_count=common["node_count"],
                target_method="reported_by_hybrid_train_candidate_artifact",
            )
            origin["old_metric_audit"]["valid_semantics"] = (
                "candidate_import_schema_compatibility_only"
            )
            origins.append(origin)
        except ValueError as error:
            rejected.append(_rejection(source, locator, error))
    expected = snapshot["record_counts"]["records"]
    if len(records) != expected:
        raise RuntimeError(
            f"冻结计数不符：{source['source_id']} "
            f"expected={expected} observed={len(records)}"
        )
    return origins, rejected


def _rejection(
    source: Mapping[str, Any], locator: Mapping[str, Any], error: Exception
) -> dict[str, Any]:
    return {
        "source_id": source["source_id"],
        "snapshot_fingerprint": source["snapshot_fingerprint"],
        "locator": dict(locator),
        "reason": "schema_rejected",
        "detail": str(error),
    }


def _build_groups(
    origins: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for origin in origins:
        grouped[str(origin["source_claimed_structural_hash"])].append(origin)
    groups: list[dict[str, Any]] = []
    conflict_ledger: list[dict[str, Any]] = []
    for structural_hash in sorted(grouped):
        members = sorted(grouped[structural_hash], key=lambda item: str(item["origin_id"]))
        variants: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for member in members:
            variants[str(member["representation_digest"])].append(member)
        variant_summaries = []
        for digest in sorted(variants):
            variant_members = variants[digest]
            first = variant_members[0]
            variant_summaries.append(
                {
                    "representation_digest": digest,
                    "formula": first["formula"],
                    "prefix_token_ids": first["prefix_token_ids"],
                    "node_count": first["node_count"],
                    "depth": first["depth"],
                    "origin_ids": sorted(item["origin_id"] for item in variant_members),
                }
            )
        conflict = len(variant_summaries) > 1
        group = {
            "schema": CANDIDATE_REGISTRY_SCHEMA,
            "source_claimed_structural_hash": structural_hash,
            "origin_count": len(members),
            "origin_ids": [item["origin_id"] for item in members],
            "source_ids": sorted(
                {item["provenance"]["source_id"] for item in members}
            ),
            "representations": variant_summaries,
            "representation_conflict": conflict,
            "downstream_eligible": not conflict,
            "downstream_block_reason": (
                "representation_conflict" if conflict else None
            ),
        }
        groups.append(group)
        if conflict:
            conflict_ledger.append(
                {
                    "source_claimed_structural_hash": structural_hash,
                    "representation_digests": sorted(variants),
                    "origin_ids": group["origin_ids"],
                }
            )
    return groups, conflict_ledger


def import_candidate_source_set(
    source_set_manifest: str | Path,
    output_root: str | Path,
) -> Path:
    """Normalize one frozen source set into an immutable provisional registry."""

    source_set_path = Path(source_set_manifest).resolve()
    output = Path(output_root).resolve()
    if output == source_set_path.parent or output in source_set_path.parents:
        raise ValueError("registry output 不得覆盖 source-set 快照目录")
    source_set, verified = _verify_source_set(source_set_path)

    all_origins: list[dict[str, Any]] = []
    rejection_ledger: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}
    adapters: set[str] = set()
    for item in sorted(verified, key=lambda value: value["source"]["source_id"]):
        source = item["source"]
        snapshot = item["manifest"]
        directory = item["path"].parent
        source_type = source["source_type"]
        if source_type == "discovery_run":
            adapter = DISCOVERY_ADAPTER_VERSION
            origins, rejected = _adapt_discovery(source, snapshot, directory)
        elif source_type == "diagnostic_audit":
            adapter = DIAGNOSTIC_ADAPTER_VERSION
            origins, rejected = _adapt_diagnostic(source, snapshot, directory)
        elif source_type == "exhaustive_registry":
            adapter = EXHAUSTIVE_ADAPTER_VERSION
            origins, rejected = _adapt_exhaustive(source, snapshot, directory)
        elif source_type == "hybrid_train_artifact":
            adapter = HYBRID_TRAIN_ARTIFACT_ADAPTER_VERSION
            origins, rejected = _adapt_hybrid_train_artifact(
                source, snapshot, directory
            )
        else:
            raise ValueError(f"不支持的 source_type：{source_type}")
        adapters.add(adapter)
        all_origins.extend(origins)
        rejection_ledger.extend(rejected)
        source_counts[source["source_id"]] = {
            "normalized": len(origins),
            "schema_rejected": len(rejected),
        }

    all_origins.sort(key=lambda item: item["origin_id"])
    rejection_ledger.sort(key=lambda item: _stable_json(item))
    groups, conflict_ledger = _build_groups(all_origins)
    origins_digest = _stable_hash(all_origins)
    groups_digest = _stable_hash(groups)
    rejection_digest = _stable_hash(rejection_ledger)
    conflict_digest = _stable_hash(conflict_ledger)
    fingerprint_payload = {
        "source_set_fingerprint": source_set["source_set_fingerprint"],
        "normalization_schema": NORMALIZATION_SCHEMA,
        "adapter_versions": sorted(adapters),
        "normalized_origins_digest": origins_digest,
        "claimed_hash_groups_digest": groups_digest,
        "schema_rejection_ledger_digest": rejection_digest,
        "representation_conflict_ledger_digest": conflict_digest,
    }
    fingerprint = _stable_hash(fingerprint_payload)
    target = output / fingerprint
    manifest_path = target / "candidate_import_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("registry_fingerprint") != fingerprint:
            raise RuntimeError(f"既有 registry 指纹冲突：{target}")
        return manifest_path

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=target.parent))
    try:
        _write_jsonl(temporary / "normalized_candidate_origins.jsonl", all_origins)
        _write_jsonl(temporary / "candidate_registry.jsonl", groups)
        incomplete = bool(rejection_ledger)
        conflicted = bool(conflict_ledger)
        counts = {
            "source_records": sum(
                values["normalized"] + values["schema_rejected"]
                for values in source_counts.values()
            ),
            "normalized_origins": len(all_origins),
            "schema_rejected": len(rejection_ledger),
            "claimed_hash_groups": len(groups),
            "duplicate_origins": len(all_origins) - len(groups),
            "representation_conflicts": len(conflict_ledger),
            "downstream_eligible_groups": sum(
                bool(group["downstream_eligible"]) for group in groups
            ),
        }
        manifest = {
            "schema": CANDIDATE_IMPORT_MANIFEST_SCHEMA,
            "mode": "provisional",
            "source_set_fingerprint": source_set["source_set_fingerprint"],
            "normalization_schema": NORMALIZATION_SCHEMA,
            "adapter_versions": sorted(adapters),
            "registry_fingerprint": fingerprint,
            "registry_status": "incomplete" if incomplete else "complete",
            "downstream_eligible": not incomplete and not conflicted,
            "downstream_block_reasons": [
                reason
                for condition, reason in (
                    (incomplete, "unresolved_schema_rejections"),
                    (conflicted, "unresolved_representation_conflicts"),
                )
                if condition
            ],
            "counts": counts,
            "source_counts": source_counts,
            "digests": {
                "normalized_origins": origins_digest,
                "claimed_hash_groups": groups_digest,
                "schema_rejection_ledger": rejection_digest,
                "representation_conflict_ledger": conflict_digest,
            },
            "schema_rejection_ledger": rejection_ledger,
            "representation_conflict_ledger": conflict_ledger,
            "fingerprint_payload": fingerprint_payload,
            "artifacts": {
                name: {
                    "size_bytes": (temporary / name).stat().st_size,
                    "sha256": _sha256_file(temporary / name),
                }
                for name in (
                    "normalized_candidate_origins.jsonl",
                    "candidate_registry.jsonl",
                )
            },
        }
        _write_json(temporary / "candidate_import_manifest.json", manifest)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return manifest_path


__all__ = [
    "CANDIDATE_IMPORT_MANIFEST_SCHEMA",
    "CANDIDATE_REGISTRY_SCHEMA",
    "DIAGNOSTIC_ADAPTER_VERSION",
    "DISCOVERY_ADAPTER_VERSION",
    "EXHAUSTIVE_ADAPTER_VERSION",
    "HYBRID_TRAIN_ARTIFACT_ADAPTER_VERSION",
    "NORMALIZATION_SCHEMA",
    "NORMALIZED_ORIGIN_SCHEMA",
    "import_candidate_source_set",
]
