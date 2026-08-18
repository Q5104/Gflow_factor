"""Current-semantics compatibility audit for Stage 6 candidate expressions."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from factor_gfn.evaluator.interpreter import (
    FEATURE_NAMES,
    INTERPRETER_OPERATOR_FUNCTIONS,
    FactorInterpreter,
)
from factor_gfn.grammar.config import SearchSpaceConfig
from factor_gfn.grammar.expression import HASH_SCHEMA, Expression, ExpressionNode
from factor_gfn.grammar.grammar_state import (
    DAGAction,
    GrammarState,
    state_space_fingerprint,
    transition_space_fingerprint,
)
from factor_gfn.grammar.operators import ALL_SYMBOLS, get_operator
from factor_gfn.grammar.partial_ast import (
    HOLE,
    PartialNode,
    get_node,
    remove_frontier,
    removable_frontier_paths,
)
from factor_gfn.grammar.tokens import (
    WINDOWS,
    action_space_fingerprint,
    action_space_manifest,
    get_action,
)

from .candidate_import import (
    CANDIDATE_IMPORT_MANIFEST_SCHEMA,
    CANDIDATE_REGISTRY_SCHEMA,
    _read_json,
    _sha256_file,
    _stable_hash,
    _verify_source_set,
)


EXPRESSION_COMPATIBILITY_SCHEMA = "factor_gfn.stage6_expression_compatibility.v1"
ACCEPTED_REGISTRY_SCHEMA = "factor_gfn.stage6_compatible_candidate_registry.v1"
EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA = (
    "factor_gfn.stage6_expression_compatibility_manifest.v1"
)
AUDITOR_VERSION = "factor_gfn.stage6_expression_compatibility_auditor.v1"

AUTO_ACCEPT = "AUTO_ACCEPT"
AUTO_REJECT = "AUTO_REJECT"
REVIEW_REQUIRED = "REVIEW_REQUIRED"

STAGE6_SEARCH_SPACE = SearchSpaceConfig(max_depth=6, max_nodes=20)
SYNTHETIC_FIXTURE_SPEC = {
    "schema": "factor_gfn.stage6_interpreter_smoke_fixture.v1",
    "shape": [72, 6, 8],
    "construction": "deterministic_bounded_trigonometric_with_fixed_nan_positions",
    "finite_requirement": "record_only_all_nan_allowed",
}


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
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            stream.write("\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"冻结 JSONL 损坏：{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"冻结 JSONL 行不是对象：{path}:{line_number}")
            rows.append(value)
    return rows


def _verify_candidate_registry(
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != CANDIDATE_IMPORT_MANIFEST_SCHEMA:
        raise ValueError("candidate import manifest schema 不受支持")
    fingerprint = manifest.get("registry_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or _stable_hash(manifest.get("fingerprint_payload")) != fingerprint
        or manifest_path.parent.name != fingerprint
    ):
        raise RuntimeError("candidate registry fingerprint 不符")
    if manifest.get("registry_status") != "complete":
        raise RuntimeError("candidate registry 存在 unresolved schema rejection")
    if int(manifest.get("counts", {}).get("schema_rejected", -1)) != 0:
        raise RuntimeError("candidate registry schema rejection ledger 非空")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("candidate registry manifest 缺少 artifacts")
    for name in ("normalized_candidate_origins.jsonl", "candidate_registry.jsonl"):
        expected = artifacts.get(name)
        path = manifest_path.parent / name
        if not isinstance(expected, Mapping) or not path.is_file():
            raise FileNotFoundError(f"candidate registry artifact 缺失：{path}")
        if (
            path.stat().st_size != expected.get("size_bytes")
            or _sha256_file(path) != expected.get("sha256")
        ):
            raise RuntimeError(f"candidate registry artifact 指纹不符：{path}")

    origins = _read_jsonl(manifest_path.parent / "normalized_candidate_origins.jsonl")
    groups = _read_jsonl(manifest_path.parent / "candidate_registry.jsonl")
    if _stable_hash(origins) != manifest.get("digests", {}).get("normalized_origins"):
        raise RuntimeError("normalized origins digest 不符")
    if _stable_hash(groups) != manifest.get("digests", {}).get("claimed_hash_groups"):
        raise RuntimeError("claimed hash groups digest 不符")
    if len(origins) != manifest["counts"]["normalized_origins"]:
        raise RuntimeError("normalized origins 冻结计数不符")
    if len(groups) != manifest["counts"]["claimed_hash_groups"]:
        raise RuntimeError("claimed hash groups 冻结计数不符")
    origins_by_id = {str(row.get("origin_id")): row for row in origins}
    if len(origins_by_id) != len(origins) or "None" in origins_by_id:
        raise RuntimeError("normalized origins 的 origin_id 缺失或重复")
    return manifest, groups, origins_by_id


def _source_action_fingerprint(snapshot_path: Path, source_type: str) -> str | None:
    directory = snapshot_path.parent
    if source_type == "discovery_run":
        metadata = _read_json(directory / "run_metadata.json")
        config = metadata.get("config_manifest")
        if isinstance(config, Mapping):
            value = config.get("token_space_fingerprint")
            return value if isinstance(value, str) else None
    elif source_type == "diagnostic_audit":
        context = _read_json(directory / "diagnostic_context.json")
        for key in ("token_space_fingerprint", "action_space_fingerprint"):
            value = context.get(key)
            if isinstance(value, str):
                return value
    elif source_type == "exhaustive_registry":
        database = directory / "exhaustive_registry.sqlite3"
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
        try:
            rows = dict(connection.execute("SELECT key, value_json FROM metadata"))
        finally:
            connection.close()
        for key in ("token_space_fingerprint", "action_space_fingerprint"):
            if key in rows:
                value = json.loads(rows[key])
                return value if isinstance(value, str) else None
    return None


def _source_semantics(
    verified_sources: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    current_action = action_space_fingerprint()
    for item in verified_sources:
        source = item["source"]
        snapshot = item["manifest"]
        source_id = str(source["source_id"])
        action_fingerprint = _source_action_fingerprint(
            item["path"], str(source["source_type"])
        )
        semantics = snapshot.get("source_semantics", {})
        output[source_id] = {
            "source_id": source_id,
            "source_type": source["source_type"],
            "source_role": source["source_role"],
            "snapshot_fingerprint": source["snapshot_fingerprint"],
            "source_action_space_fingerprint": action_fingerprint,
            "action_space_relation": (
                "missing"
                if action_fingerprint is None
                else "same"
                if action_fingerprint == current_action
                else "different"
            ),
            "generation_config_fingerprint": semantics.get(
                "generation_config_fingerprint"
            ),
            "provider_fingerprint": semantics.get("provider_fingerprint"),
            "context_fingerprint": semantics.get("context_fingerprint"),
            "reward_fingerprint": semantics.get("reward_fingerprint"),
            "historical_metric_reuse": "forbidden",
            "stage6_metric_recompute_required": True,
        }
    return output


def _operator_manifest() -> list[dict[str, Any]]:
    return [
        {
            "name": operator.name,
            "category": operator.category.value,
            "arity": operator.arity,
            "requires_window": operator.requires_window,
            "commutative": operator.commutative,
        }
        for operator in ALL_SYMBOLS
    ]


def _target_semantics_manifest() -> dict[str, Any]:
    dispatch = sorted(INTERPRETER_OPERATOR_FUNCTIONS)
    return {
        "expression_hash_schema": HASH_SCHEMA,
        "action_space_fingerprint": action_space_fingerprint(),
        "action_space_manifest_digest": _stable_hash(action_space_manifest()),
        "state_space_fingerprint": state_space_fingerprint(),
        "transition_space_fingerprint": transition_space_fingerprint(),
        "search_space": STAGE6_SEARCH_SPACE.manifest(),
        "search_space_fingerprint": STAGE6_SEARCH_SPACE.fingerprint(),
        "operator_registry_digest": _stable_hash(_operator_manifest()),
        "interpreter_dispatch_digest": _stable_hash(dispatch),
        "interpreter_dispatch_count": len(dispatch),
        "feature_names": list(FEATURE_NAMES),
        "synthetic_fixture_spec": SYNTHETIC_FIXTURE_SPEC,
        "synthetic_fixture_fingerprint": _stable_hash(SYNTHETIC_FIXTURE_SPEC),
        "auditor_version": AUDITOR_VERSION,
    }


def _synthetic_fixture() -> np.ndarray:
    date = np.arange(72, dtype=np.float64)[:, None, None]
    feature = np.arange(6, dtype=np.float64)[None, :, None]
    stock = np.arange(8, dtype=np.float64)[None, None, :]
    data = 0.55 + 0.04 * np.sin((date + 1.0) * (feature + 1.0) / 17.0)
    data = data + 0.03 * np.cos((date + stock + 1.0) / 11.0)
    data = np.broadcast_to(data, (72, 6, 8)).copy()
    data[3, 0, 0] = np.nan
    data[17, 4, 3] = np.nan
    data[41, 5, 7] = np.nan
    data.setflags(write=False)
    return data


def _to_partial(node: ExpressionNode) -> PartialNode:
    return PartialNode(
        node.action_id,
        tuple(_to_partial(child) for child in node.children),
    )


def _find_grammar_path(expression: Expression) -> tuple[DAGAction, ...] | None:
    """Find one legal root-to-terminal path, memoizing failed partial states."""

    try:
        terminal = GrammarState(
            search_space=STAGE6_SEARCH_SPACE,
            _root=_to_partial(expression.canonicalize().root),
        )
    except (TypeError, ValueError, RuntimeError):
        return None
    failed: set[str] = set()

    def find_to_root(state: GrammarState) -> tuple[DAGAction, ...] | None:
        if state.root == HOLE:
            return ()
        if state.state_key in failed:
            return None
        for path in removable_frontier_paths(state.root):
            removed = get_node(state.root, path)
            if removed.action_id is None:
                continue
            try:
                parent = GrammarState(
                    search_space=STAGE6_SEARCH_SPACE,
                    _root=remove_frontier(state.root, path),
                )
            except (TypeError, ValueError, RuntimeError):
                continue
            for slot in parent.open_slots():
                action = DAGAction(slot.path, removed.action_id)
                try:
                    successor = parent.step(action)
                except (IndexError, TypeError, ValueError, RuntimeError):
                    continue
                if successor.state_key != state.state_key:
                    continue
                prefix = find_to_root(parent)
                if prefix is not None:
                    return (*prefix, action)
        failed.add(state.state_key)
        return None

    path = find_to_root(terminal)
    if path is None:
        return None
    replay = GrammarState(search_space=STAGE6_SEARCH_SPACE)
    try:
        for action in path:
            replay = replay.step(action)
    except (IndexError, TypeError, ValueError, RuntimeError):
        return None
    if not replay.done:
        return None
    if replay.to_expression().canonical_key() != expression.canonical_key():
        return None
    return path


def _operator_compatibility(expression: Expression) -> tuple[list[str], str | None]:
    names: set[str] = set()
    stack = [expression.root]
    while stack:
        node = stack.pop()
        action = get_action(node.action_id)
        operator = get_operator(action.name)
        names.add(action.name)
        if action.arity != operator.arity:
            return sorted(names), "operator_arity_mismatch"
        if operator.requires_window != (action.window != 0):
            return sorted(names), "operator_window_contract_mismatch"
        if action.window != 0 and action.window not in WINDOWS:
            return sorted(names), "operator_window_unsupported"
        if action.arity == 0:
            if action.name not in FEATURE_NAMES:
                return sorted(names), "leaf_feature_unsupported"
        elif action.name not in INTERPRETER_OPERATOR_FUNCTIONS:
            return sorted(names), "interpreter_dispatch_missing"
        stack.extend(node.children)
    return sorted(names), None


def _smoke_expression(
    expression: Expression,
    interpreter: FactorInterpreter,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    current_hash = expression.structural_hash()
    if current_hash in cache:
        return cache[current_hash]
    try:
        result = interpreter.evaluate(expression)
        outcome = {
            "passed": True,
            "reason_code": None,
            "output_shape": list(result.shape),
            "finite_count": int(np.isfinite(result).sum()),
            "all_nan_allowed": True,
        }
    except Exception as error:  # The audit must record any runtime incompatibility.
        outcome = {
            "passed": False,
            "reason_code": "interpreter_runtime_incompatible",
            "error_type": type(error).__name__,
            "error_detail": str(error),
            "all_nan_allowed": True,
        }
    cache[current_hash] = outcome
    return outcome


def _audit_representation(
    representation: Mapping[str, Any],
    *,
    claimed_hash: str,
    origin_rows: list[Mapping[str, Any]],
    source_semantics: Mapping[str, Mapping[str, Any]],
    interpreter: FactorInterpreter,
    smoke_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], Expression | None]:
    reject_reasons: list[str] = []
    review_reasons: list[str] = []
    source_formula = representation.get("formula")
    prefix = representation.get("prefix_token_ids")
    try:
        expression = Expression.from_prefix(prefix)
    except (IndexError, TypeError, ValueError) as error:
        return (
            {
                "representation_digest": representation.get("representation_digest"),
                "status": AUTO_REJECT,
                "reason_codes": ["prefix_parse_invalid"],
                "error_type": type(error).__name__,
                "error_detail": str(error),
            },
            None,
        )

    current_formula = expression.to_formula()
    current_hash = expression.structural_hash()
    stats = expression.stats
    if list(expression.to_prefix()) != prefix:
        reject_reasons.append("prefix_round_trip_mismatch")
    if not isinstance(source_formula, str) or not source_formula:
        review_reasons.append("source_formula_missing")
    elif source_formula != current_formula:
        review_reasons.append("source_formula_mismatch")
    if claimed_hash != current_hash:
        review_reasons.append("source_structural_hash_mismatch")
    if representation.get("node_count") != stats.node_count:
        review_reasons.append("source_node_count_mismatch")
    if representation.get("depth") != stats.depth:
        review_reasons.append("source_depth_mismatch")

    source_ids = sorted(
        {str(row.get("provenance", {}).get("source_id")) for row in origin_rows}
    )
    action_relations = {
        source_id: source_semantics[source_id]["action_space_relation"]
        for source_id in source_ids
    }
    if "different" in action_relations.values() and (
        not isinstance(source_formula, str) or not source_formula
    ):
        review_reasons.append("action_space_changed_without_formula")

    path = _find_grammar_path(expression)
    if path is None:
        reject_reasons.append("grammar_unreachable")
    operators, operator_error = _operator_compatibility(expression)
    if operator_error is not None:
        reject_reasons.append(operator_error)
    smoke = _smoke_expression(expression, interpreter, smoke_cache)
    if not smoke["passed"]:
        reject_reasons.append(str(smoke["reason_code"]))

    status = (
        AUTO_REJECT
        if reject_reasons
        else REVIEW_REQUIRED
        if review_reasons
        else AUTO_ACCEPT
    )
    result = {
        "representation_digest": representation.get("representation_digest"),
        "status": status,
        "reason_codes": sorted(set((*reject_reasons, *review_reasons))),
        "source": {
            "formula": source_formula,
            "structural_hash": claimed_hash,
            "node_count": representation.get("node_count"),
            "depth": representation.get("depth"),
            "prefix_token_ids": prefix,
        },
        "current": {
            "formula": current_formula,
            "structural_hash": current_hash,
            "node_count": stats.node_count,
            "depth": stats.depth,
            "prefix_token_ids": list(expression.to_prefix()),
            "canonical_key_digest": _stable_hash(expression.canonical_key()),
        },
        "grammar": {
            "reachable": path is not None,
            "proof_length": len(path) if path is not None else None,
            "proof_method": "memoized_early_exit_parent_path_then_forward_replay",
        },
        "operators": operators,
        "interpreter_smoke": smoke,
        "source_action_space_relations": action_relations,
    }
    return result, expression


def _group_status(
    group: Mapping[str, Any], representation_results: list[Mapping[str, Any]]
) -> tuple[str, list[str]]:
    statuses = [result["status"] for result in representation_results]
    reasons: set[str] = {
        str(reason)
        for result in representation_results
        for reason in result.get("reason_codes", [])
    }
    conflict = bool(group.get("representation_conflict")) or len(statuses) > 1
    if statuses and all(status == AUTO_REJECT for status in statuses):
        return AUTO_REJECT, sorted(reasons)
    if conflict:
        reasons.add("representation_conflict")
        if len(set(statuses)) > 1:
            reasons.add("mixed_representation_outcomes")
        return REVIEW_REQUIRED, sorted(reasons)
    if statuses == [AUTO_ACCEPT]:
        return AUTO_ACCEPT, []
    return REVIEW_REQUIRED, sorted(reasons)


def audit_expression_compatibility(
    candidate_import_manifest: str | Path,
    source_set_manifest: str | Path,
    output_root: str | Path,
) -> Path:
    """Audit one immutable candidate registry against current expression semantics."""

    started = time.perf_counter()
    registry_path = Path(candidate_import_manifest).resolve()
    source_set_path = Path(source_set_manifest).resolve()
    output = Path(output_root).resolve()
    registry, groups, origins = _verify_candidate_registry(registry_path)
    source_set, verified_sources = _verify_source_set(source_set_path)
    if source_set.get("source_set_fingerprint") != registry.get(
        "source_set_fingerprint"
    ):
        raise RuntimeError("candidate registry 与 source set fingerprint 不一致")
    semantics = _source_semantics(verified_sources)
    target_semantics = _target_semantics_manifest()
    interpreter = FactorInterpreter(_synthetic_fixture())
    smoke_cache: dict[str, dict[str, Any]] = {}

    audits: list[dict[str, Any]] = []
    expressions_by_group: dict[str, Expression] = {}
    for group in groups:
        if group.get("schema") != CANDIDATE_REGISTRY_SCHEMA:
            raise ValueError("candidate registry group schema 不受支持")
        claimed_hash = str(group.get("source_claimed_structural_hash"))
        representation_results: list[dict[str, Any]] = []
        accepted_expression: Expression | None = None
        for representation in group.get("representations", []):
            origin_ids = representation.get("origin_ids", [])
            try:
                origin_rows = [origins[str(origin_id)] for origin_id in origin_ids]
            except KeyError as error:
                raise RuntimeError(
                    f"group 引用了不存在的 origin_id：{claimed_hash}"
                ) from error
            result, expression = _audit_representation(
                representation,
                claimed_hash=claimed_hash,
                origin_rows=origin_rows,
                source_semantics=semantics,
                interpreter=interpreter,
                smoke_cache=smoke_cache,
            )
            representation_results.append(result)
            if result["status"] == AUTO_ACCEPT and expression is not None:
                accepted_expression = expression
        status, reasons = _group_status(group, representation_results)
        if status == AUTO_ACCEPT and accepted_expression is not None:
            expressions_by_group[claimed_hash] = accepted_expression
        audits.append(
            {
                "schema": EXPRESSION_COMPATIBILITY_SCHEMA,
                "source_claimed_structural_hash": claimed_hash,
                "status": status,
                "reason_codes": reasons,
                "origin_count": group.get("origin_count"),
                "origin_ids": group.get("origin_ids"),
                "source_ids": group.get("source_ids"),
                "representations": representation_results,
                "historical_metric_reuse": "forbidden",
                "stage6_metric_recompute_required": True,
            }
        )

    current_hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in audits:
        if audit["status"] != AUTO_REJECT:
            current_hashes = {
                result.get("current", {}).get("structural_hash")
                for result in audit["representations"]
                if result.get("current", {}).get("structural_hash") is not None
            }
            for current_hash in current_hashes:
                current_hash_groups[str(current_hash)].append(audit)
    for current_hash, members in current_hash_groups.items():
        claimed = {member["source_claimed_structural_hash"] for member in members}
        if len(claimed) <= 1:
            continue
        for member in members:
            member["status"] = REVIEW_REQUIRED
            member["reason_codes"] = sorted(
                {*member["reason_codes"], "current_hash_cross_group_collision"}
            )
            expressions_by_group.pop(member["source_claimed_structural_hash"], None)

    audits.sort(key=lambda row: row["source_claimed_structural_hash"])
    accepted: list[dict[str, Any]] = []
    for audit in audits:
        if audit["status"] != AUTO_ACCEPT:
            continue
        claimed_hash = audit["source_claimed_structural_hash"]
        expression = expressions_by_group[claimed_hash]
        accepted.append(
            {
                "schema": ACCEPTED_REGISTRY_SCHEMA,
                "current_structural_hash": expression.structural_hash(),
                "source_claimed_structural_hash": claimed_hash,
                "formula": expression.to_formula(),
                "prefix_token_ids": list(expression.to_prefix()),
                "node_count": expression.stats.node_count,
                "depth": expression.stats.depth,
                "origin_ids": audit["origin_ids"],
                "source_ids": audit["source_ids"],
                "compatibility_record_fingerprint": _stable_hash(audit),
                "historical_metric_reuse": "forbidden",
                "stage6_metric_recompute_required": True,
            }
        )
    accepted.sort(key=lambda row: row["current_structural_hash"])

    status_counts = Counter(row["status"] for row in audits)
    reason_counts = Counter(
        reason for row in audits for reason in row.get("reason_codes", [])
    )
    audit_digest = _stable_hash(audits)
    accepted_digest = _stable_hash(accepted)
    review_ledger = [row for row in audits if row["status"] == REVIEW_REQUIRED]
    reject_ledger = [row for row in audits if row["status"] == AUTO_REJECT]
    fingerprint_payload = {
        "candidate_registry_fingerprint": registry["registry_fingerprint"],
        "source_set_fingerprint": source_set["source_set_fingerprint"],
        "target_semantics_fingerprint": _stable_hash(target_semantics),
        "auditor_version": AUDITOR_VERSION,
        "audit_digest": audit_digest,
        "accepted_registry_digest": accepted_digest,
        "review_ledger_digest": _stable_hash(review_ledger),
        "reject_ledger_digest": _stable_hash(reject_ledger),
    }
    audit_fingerprint = _stable_hash(fingerprint_payload)
    target = output / audit_fingerprint
    manifest_path = target / "expression_compatibility_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("audit_fingerprint") != audit_fingerprint:
            raise RuntimeError(f"既有 compatibility audit 指纹冲突：{target}")
        return manifest_path

    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=output))
    try:
        _write_jsonl(temporary / "expression_compatibility_audit.jsonl", audits)
        _write_jsonl(temporary / "auto_accepted_candidate_registry.jsonl", accepted)
        manifest = {
            "schema": EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA,
            "mode": "provisional",
            "candidate_registry_fingerprint": registry["registry_fingerprint"],
            "source_set_fingerprint": source_set["source_set_fingerprint"],
            "audit_fingerprint": audit_fingerprint,
            "accepted_registry_fingerprint": accepted_digest,
            "audit_status": (
                "incomplete" if status_counts[REVIEW_REQUIRED] else "complete"
            ),
            "downstream_eligible": status_counts[REVIEW_REQUIRED] == 0,
            "downstream_block_reasons": (
                ["unresolved_review_required"]
                if status_counts[REVIEW_REQUIRED]
                else []
            ),
            "counts": {
                "claimed_hash_groups": len(audits),
                AUTO_ACCEPT: status_counts[AUTO_ACCEPT],
                AUTO_REJECT: status_counts[AUTO_REJECT],
                REVIEW_REQUIRED: status_counts[REVIEW_REQUIRED],
                "accepted_registry_candidates": len(accepted),
                "interpreter_unique_expressions_smoked": len(smoke_cache),
            },
            "reason_counts": dict(sorted(reason_counts.items())),
            "target_semantics": target_semantics,
            "target_semantics_fingerprint": _stable_hash(target_semantics),
            "source_semantics": [semantics[key] for key in sorted(semantics)],
            "digests": {
                "compatibility_audit": audit_digest,
                "accepted_registry": accepted_digest,
                "review_ledger": _stable_hash(review_ledger),
                "reject_ledger": _stable_hash(reject_ledger),
            },
            "fingerprint_payload": fingerprint_payload,
            "elapsed_seconds_observed": time.perf_counter() - started,
            "elapsed_seconds_excluded_from_fingerprint": True,
            "artifacts": {
                name: {
                    "size_bytes": (temporary / name).stat().st_size,
                    "sha256": _sha256_file(temporary / name),
                }
                for name in (
                    "expression_compatibility_audit.jsonl",
                    "auto_accepted_candidate_registry.jsonl",
                )
            },
        }
        _write_json(temporary / "expression_compatibility_manifest.json", manifest)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return manifest_path


__all__ = [
    "ACCEPTED_REGISTRY_SCHEMA",
    "AUDITOR_VERSION",
    "AUTO_ACCEPT",
    "AUTO_REJECT",
    "EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA",
    "EXPRESSION_COMPATIBILITY_SCHEMA",
    "REVIEW_REQUIRED",
    "audit_expression_compatibility",
]
