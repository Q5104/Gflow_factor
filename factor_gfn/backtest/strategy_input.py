"""Verified Top100 strategy input derived from a frozen Baseline Factor Pool.

This module intentionally supports one policy only: the first 100 records in
the authoritative frozen pool order.  It is not a general subset or selection
framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

from .baseline_factor_pool import (
    OOS_UNTOUCHED,
    FrozenBaselineFactorRecord,
    VerifiedFrozenBaselineFactorPool,
    load_verified_baseline_factor_pool,
)


STRATEGY_INPUT_TOP_K = 100
STRATEGY_INPUT_MANIFEST_SCHEMA = "factor_gfn.strategy_input_manifest.v1"
STRATEGY_INPUT_VERSION = "frozen-baseline-pool-top100-prefix-v1"
STRATEGY_INPUT_MANIFEST_FILENAME = "strategy_input_manifest.json"


class StrategyInputIntegrityError(RuntimeError):
    """The fixed Top100 input or its parent frozen pool is inconsistent."""


@dataclass(frozen=True, slots=True)
class StrategyInputArtifact:
    manifest_path: Path
    strategy_input_fingerprint: str
    factor_count: int
    reused_existing_artifact: bool


@dataclass(frozen=True, slots=True)
class VerifiedStrategyInput:
    manifest_path: Path
    strategy_input_fingerprint: str
    factor_pool_manifest_path: Path
    factor_pool_fingerprint: str
    records: tuple[FrozenBaselineFactorRecord, ...]
    ordered_structural_hashes: tuple[str, ...]
    frozen_train_directions: tuple[int, ...]
    top_k: int
    manifest: Mapping[str, Any]
    oos_status: str


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrategyInputIntegrityError(
            f"cannot read Strategy Input manifest: {path}"
        ) from error
    if not isinstance(value, dict):
        raise StrategyInputIntegrityError("Strategy Input manifest must be an object")
    return value


def _policy() -> dict[str, Any]:
    return {
        "source": "verified_frozen_baseline_factor_pool",
        "method": "frozen_order_prefix",
        "top_k": STRATEGY_INPUT_TOP_K,
        "ranking_or_reselection": "none",
        "coverage_dependency": "none",
        "oos_dependency": "none",
        "supported_policy_count": 1,
    }


def _input_payload(pool: VerifiedFrozenBaselineFactorPool) -> dict[str, Any]:
    if not isinstance(pool, VerifiedFrozenBaselineFactorPool):
        raise TypeError("pool must be VerifiedFrozenBaselineFactorPool")
    if pool.oos_status != OOS_UNTOUCHED:
        raise StrategyInputIntegrityError("frozen pool OOS state changed")
    if len(pool.records) < STRATEGY_INPUT_TOP_K:
        raise StrategyInputIntegrityError(
            f"frozen pool has fewer than {STRATEGY_INPUT_TOP_K} records"
        )
    records = pool.records[:STRATEGY_INPUT_TOP_K]
    return {
        "factor_count": len(records),
        "ordered_structural_hashes": [record.structural_hash for record in records],
        "ordered_train_directions": [record.train_direction for record in records],
        "ordered_provisional_ranks": [record.provisional_rank for record in records],
        "ordered_stage6_sorted_ranks": [
            record.stage6_sorted_rank for record in records
        ],
    }


def _fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest.get("schema"),
        "version": manifest.get("version"),
        "factor_pool_fingerprint": manifest.get("factor_pool_fingerprint"),
        "policy": manifest.get("policy"),
        "strategy_input": manifest.get("strategy_input"),
        "oos_status": manifest.get("oos_status"),
    }


def _build_manifest(
    pool: VerifiedFrozenBaselineFactorPool, *, created_at_utc: str
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": STRATEGY_INPUT_MANIFEST_SCHEMA,
        "version": STRATEGY_INPUT_VERSION,
        "factor_pool_fingerprint": pool.baseline_factor_pool_fingerprint,
        "policy": _policy(),
        "strategy_input": _input_payload(pool),
        "upstream_manifest_paths": {
            "factor_pool_manifest": str(pool.manifest_path.resolve())
        },
        "oos_status": OOS_UNTOUCHED,
        "created_at_utc": created_at_utc,
        "created_at_excluded_from_fingerprint": True,
    }
    manifest["strategy_input_fingerprint"] = _stable_hash(
        _fingerprint_payload(manifest)
    )
    return manifest


def _verify_directory(
    manifest_path: Path, *, require_directory_identity: bool
) -> VerifiedStrategyInput:
    manifest = _read_json(manifest_path)
    fingerprint = str(manifest.get("strategy_input_fingerprint", ""))
    if (
        manifest.get("schema") != STRATEGY_INPUT_MANIFEST_SCHEMA
        or manifest.get("version") != STRATEGY_INPUT_VERSION
        or manifest.get("policy") != _policy()
        or manifest.get("oos_status") != OOS_UNTOUCHED
        or _stable_hash(_fingerprint_payload(manifest)) != fingerprint
    ):
        raise StrategyInputIntegrityError(
            "Strategy Input schema, policy, OOS state, or fingerprint mismatch"
        )
    if require_directory_identity and manifest_path.parent.name != fingerprint:
        raise StrategyInputIntegrityError("Strategy Input directory identity mismatch")
    pool_path_value = manifest.get("upstream_manifest_paths", {}).get(
        "factor_pool_manifest"
    )
    if not isinstance(pool_path_value, str) or not pool_path_value:
        raise StrategyInputIntegrityError("frozen pool manifest path is missing")
    pool_path = Path(pool_path_value).resolve()
    pool = load_verified_baseline_factor_pool(pool_path)
    expected = _input_payload(pool)
    if (
        pool.baseline_factor_pool_fingerprint
        != manifest.get("factor_pool_fingerprint")
        or manifest.get("strategy_input") != expected
    ):
        raise StrategyInputIntegrityError(
            "Strategy Input is not the exact Top100 frozen-order prefix"
        )
    records = pool.records[:STRATEGY_INPUT_TOP_K]
    return VerifiedStrategyInput(
        manifest_path=manifest_path,
        strategy_input_fingerprint=fingerprint,
        factor_pool_manifest_path=pool.manifest_path,
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        records=records,
        ordered_structural_hashes=tuple(
            record.structural_hash for record in records
        ),
        frozen_train_directions=tuple(record.train_direction for record in records),
        top_k=STRATEGY_INPUT_TOP_K,
        manifest=MappingProxyType(manifest),
        oos_status=OOS_UNTOUCHED,
    )


def freeze_top100_strategy_input(
    pool: VerifiedFrozenBaselineFactorPool, runs_root: str | Path
) -> StrategyInputArtifact:
    """Freeze the only supported strategy input: full-pool order prefix Top100."""

    manifest = _build_manifest(pool, created_at_utc=datetime.now(UTC).isoformat())
    fingerprint = str(manifest["strategy_input_fingerprint"])
    root = Path(runs_root).resolve() / "baseline_strategy_inputs"
    target = root / fingerprint
    manifest_path = target / STRATEGY_INPUT_MANIFEST_FILENAME
    if target.exists():
        verified = _verify_directory(
            manifest_path, require_directory_identity=True
        )
        return StrategyInputArtifact(
            manifest_path=verified.manifest_path,
            strategy_input_fingerprint=verified.strategy_input_fingerprint,
            factor_count=len(verified.records),
            reused_existing_artifact=True,
        )
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.tmp-", dir=root))
    try:
        temporary_manifest = temporary / STRATEGY_INPUT_MANIFEST_FILENAME
        temporary_manifest.write_bytes(_json_bytes(manifest))
        _verify_directory(temporary_manifest, require_directory_identity=False)
        os.replace(temporary, target)
        verified = _verify_directory(manifest_path, require_directory_identity=True)
        return StrategyInputArtifact(
            manifest_path=verified.manifest_path,
            strategy_input_fingerprint=verified.strategy_input_fingerprint,
            factor_count=len(verified.records),
            reused_existing_artifact=False,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_verified_strategy_input(
    manifest_path: str | Path,
) -> VerifiedStrategyInput:
    return _verify_directory(
        Path(manifest_path).resolve(), require_directory_identity=True
    )


__all__ = [
    "STRATEGY_INPUT_MANIFEST_FILENAME",
    "STRATEGY_INPUT_MANIFEST_SCHEMA",
    "STRATEGY_INPUT_TOP_K",
    "STRATEGY_INPUT_VERSION",
    "StrategyInputArtifact",
    "StrategyInputIntegrityError",
    "VerifiedStrategyInput",
    "freeze_top100_strategy_input",
    "load_verified_strategy_input",
]
