"""Incremental Stage 5 Train-only candidate artifact for Hybrid runs."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from factor_gfn.barra import STYLE_NAMES
from factor_gfn.grammar import Expression

from .real_reward import REAL_REWARD_PROVIDER_SCHEMA, RealRewardEvaluationRecord

if TYPE_CHECKING:
    from .hybrid_trainer import HybridUpdateOutput, HybridVarianceTrainer


TRAIN_CANDIDATE_ARTIFACT_SCHEMA = "factor_gfn.stage5_train_candidate_artifact.v1"
TRAIN_CANDIDATE_RECORD_SCHEMA = "factor_gfn.stage5_train_candidate_record.v1"
TRAIN_EVALUATION_CONTRACT_SCHEMA = "factor_gfn.train_evaluation_contract.v1"
TRAIN_CANDIDATE_ARTIFACT_FILENAME = "train_candidate_artifact.json"


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Train candidate artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Train candidate artifact must be a JSON object")
    return payload


def _train_scope_projection(provider_manifest: Mapping[str, Any]) -> dict[str, Any]:
    reward_config = provider_manifest.get("reward_config") or {}
    return {
        "provider_schema": provider_manifest.get("schema"),
        "data_scope": provider_manifest.get("data_scope"),
        "context_fingerprint": provider_manifest.get("context_fingerprint"),
        "reward_evaluator_context_fingerprint": provider_manifest.get(
            "reward_evaluator_context_fingerprint"
        ),
        "evaluation_config": provider_manifest.get("evaluation_config"),
        "barra_metric_config": {
            "min_common_periods": reward_config.get("barra_min_common_periods"),
            "candidate_industry_neutralization": reward_config.get(
                "candidate_industry_neutralization"
            ),
        },
        "calendar": provider_manifest.get("calendar"),
        "reward_panel": provider_manifest.get("reward_panel"),
        "interpreter": provider_manifest.get("interpreter"),
        "industry_neutralization": provider_manifest.get(
            "industry_neutralization"
        ),
    }


def supports_train_candidate_artifact(provider: Any) -> bool:
    manifest_method = getattr(provider, "manifest", None)
    if not callable(manifest_method):
        return False
    manifest = manifest_method()
    return (
        isinstance(manifest, Mapping)
        and manifest.get("schema") == REAL_REWARD_PROVIDER_SCHEMA
        and manifest.get("data_scope") == "training_only"
        and callable(getattr(provider, "evaluation_records_since", None))
        and isinstance(getattr(provider, "evaluation_record_count", None), int)
    )


def _contract(provider: Any) -> tuple[dict[str, Any], str]:
    provider_manifest = provider.manifest()
    implementation = {
        "artifact_module_sha256": _sha256_file(Path(__file__).resolve()),
        "reward_module_sha256": _sha256_file(
            Path(__file__).with_name("reward.py").resolve()
        ),
        "real_reward_module_sha256": _sha256_file(
            Path(__file__).with_name("real_reward.py").resolve()
        ),
    }
    payload = {
        "schema": TRAIN_EVALUATION_CONTRACT_SCHEMA,
        "provider_fingerprint": provider.fingerprint(),
        "train_scope_projection": _train_scope_projection(provider_manifest),
        "implementation": implementation,
    }
    return payload, _stable_hash(payload)


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _reward_payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "result"):
        value = asdict(value.result)
    if not isinstance(value, Mapping):
        raise ValueError("candidate lacks a Train RewardResult payload")
    return dict(value)


def _candidate_record(
    expression: Expression,
    reward_result: Any,
    *,
    contract_fingerprint: str,
    optimizer_step: int,
    cycle_index: int,
    condition_position: int,
) -> dict[str, Any]:
    canonical = expression.canonicalize()
    structural_hash = canonical.structural_hash()
    payload = _reward_payload(reward_result)
    if payload.get("expression_hash") not in (None, structural_hash):
        raise ValueError("RewardResult structural hash differs from candidate identity")
    if payload.get("valid") is not True:
        raise ValueError("successful Hybrid candidate lacks a valid Train evaluation")

    correlations = payload.get("barra_correlations")
    periods = payload.get("barra_valid_periods")
    if not isinstance(correlations, Mapping) or not isinstance(periods, Mapping):
        raise ValueError("Train Barra decomposition is incomplete")
    dates = payload.get("train_long_excess_dates", ())
    values = payload.get("train_long_excess_values", ())
    if not isinstance(dates, (list, tuple)) or not isinstance(values, (list, tuple)):
        raise ValueError("Train long-excess dates/values must be sequences")
    if len(dates) != len(values):
        raise ValueError("Train long-excess dates/values length mismatch")
    if expression.stats.node_count > 2 and not dates:
        raise ValueError("new Hybrid candidate lacks persisted Train long-excess")
    normalized_values = [
        None if value is None else _finite_float(value, "train_long_excess_values")
        for value in values
    ]
    direction = payload.get("long_direction")
    if direction not in (-1, 1):
        raise ValueError("Train direction must be -1 or 1")
    seen = {
        "optimizer_step": optimizer_step,
        "cycle_index": cycle_index,
        "condition_position_in_cycle": condition_position,
        "condition_N": expression.stats.node_count,
    }
    return {
        "schema": TRAIN_CANDIDATE_RECORD_SCHEMA,
        "structural_hash": structural_hash,
        "formula": canonical.to_formula(),
        "prefix_token_ids": list(canonical.to_prefix()),
        "node_count": canonical.stats.node_count,
        "depth": canonical.stats.depth,
        "train_evaluation_contract_fingerprint": contract_fingerprint,
        "train_ic": _finite_float(payload.get("train_ic"), "train_ic"),
        "train_ic_valid_periods": _non_negative_int(
            payload.get("ic_valid_periods"), "train_ic_valid_periods"
        ),
        "train_direction": direction,
        "train_long_ir": _finite_float(
            payload.get("train_long_ir"), "train_long_ir"
        ),
        "train_long_valid_periods": _non_negative_int(
            payload.get("long_ir_valid_periods"), "train_long_valid_periods"
        ),
        "train_long_excess_dates": [str(value) for value in dates],
        "train_long_excess_values": normalized_values,
        "train_barra_ts_corr": _finite_float(
            payload.get("barra_ts_corr"), "train_barra_ts_corr"
        ),
        "train_barra_correlations": {
            name: _finite_float(correlations.get(name), f"train_barra_correlations.{name}")
            for name in STYLE_NAMES
        },
        "train_barra_valid_periods_by_style": {
            name: _non_negative_int(
                periods.get(name), f"train_barra_valid_periods_by_style.{name}"
            )
            for name in STYLE_NAMES
        },
        "neutralization_diagnostics": {
            "industry_neutralized": payload.get("industry_neutralized"),
            "skipped_dates": list(payload.get("neutralization_skipped_dates", ())),
            "skipped_rate": _finite_float(
                payload.get("neutralization_skipped_rate"),
                "neutralization_diagnostics.skipped_rate",
            ),
            "details": list(payload.get("neutralization_skipped_details", ())),
        },
        "first_seen": seen,
        "last_seen": seen,
        "visit_count": 1,
    }


def _metric_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"first_seen", "last_seen", "visit_count"}
    }


def _reward_payload_from_artifact_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    neutralization = record["neutralization_diagnostics"]
    return {
        "expression_hash": record["structural_hash"],
        "valid": True,
        "train_ic": record["train_ic"],
        "ic_valid_periods": record["train_ic_valid_periods"],
        "long_direction": record["train_direction"],
        "train_long_ir": record["train_long_ir"],
        "long_ir_valid_periods": record["train_long_valid_periods"],
        "train_long_excess_dates": record["train_long_excess_dates"],
        "train_long_excess_values": record["train_long_excess_values"],
        "barra_ts_corr": record["train_barra_ts_corr"],
        "barra_correlations": record["train_barra_correlations"],
        "barra_valid_periods": record["train_barra_valid_periods_by_style"],
        "industry_neutralized": neutralization["industry_neutralized"],
        "neutralization_skipped_dates": neutralization["skipped_dates"],
        "neutralization_skipped_rate": neutralization["skipped_rate"],
        "neutralization_skipped_details": neutralization["details"],
    }


class TrainCandidateArtifactWriter:
    """Atomically rewrite a deduplicated Train-only artifact after each update."""

    def __init__(
        self,
        *,
        run_dir: Path,
        provider: Any,
        expected_optimizer_step: int,
        create: bool,
    ) -> None:
        if not supports_train_candidate_artifact(provider):
            raise TypeError("provider does not expose formal Train evaluation records")
        self.run_dir = run_dir.resolve()
        self.path = self.run_dir / TRAIN_CANDIDATE_ARTIFACT_FILENAME
        self.provider = provider
        self.contract, self.contract_fingerprint = _contract(provider)
        self._provider_record_cursor = 0
        run_config_path = self.run_dir / "hybrid_run_config.json"
        if not run_config_path.is_file():
            raise ValueError("hybrid run config must exist before Train artifact setup")
        self.source_run = {
            "run_directory_name": self.run_dir.name,
            "hybrid_run_config_sha256": _sha256_file(run_config_path),
        }
        if create:
            if self.path.exists():
                raise FileExistsError(f"Train candidate artifact already exists: {self.path}")
            self._write([], expected_optimizer_step)
        self._payload = self._load(expected_optimizer_step)

    def _load(self, expected_optimizer_step: int) -> dict[str, Any]:
        payload = _read_json(self.path)
        if payload.get("schema") != TRAIN_CANDIDATE_ARTIFACT_SCHEMA:
            raise ValueError("Train candidate artifact schema mismatch")
        if payload.get("train_evaluation_contract_fingerprint") != self.contract_fingerprint:
            raise ValueError("Train candidate artifact contract fingerprint mismatch")
        if payload.get("train_evaluation_contract") != self.contract:
            raise ValueError("Train candidate artifact contract differs from current provider")
        if payload.get("source_run") != self.source_run:
            raise ValueError("Train candidate artifact source run mismatch")
        if payload.get("committed_optimizer_step") != expected_optimizer_step:
            raise ValueError("Train candidate artifact diverges from checkpoint optimizer step")
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("Train candidate artifact records must be a list")
        hashes = [record.get("structural_hash") for record in records if isinstance(record, Mapping)]
        if len(hashes) != len(records) or hashes != sorted(hashes) or len(set(hashes)) != len(hashes):
            raise ValueError("Train candidate artifact identities are not unique and sorted")
        if payload.get("candidate_count") != len(records):
            raise ValueError("Train candidate artifact candidate count mismatch")
        return payload

    def _write(self, records: Sequence[Mapping[str, Any]], optimizer_step: int) -> None:
        ordered = sorted((dict(record) for record in records), key=lambda row: row["structural_hash"])
        payload = {
            "schema": TRAIN_CANDIDATE_ARTIFACT_SCHEMA,
            "source_run": self.source_run,
            "train_evaluation_contract": self.contract,
            "train_evaluation_contract_fingerprint": self.contract_fingerprint,
            "committed_optimizer_step": optimizer_step,
            "candidate_count": len(ordered),
            "records": ordered,
        }
        _atomic_write_json(self.path, payload)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._payload["records"])

    def commit_update(
        self,
        output: "HybridUpdateOutput",
        trainer: "HybridVarianceTrainer",
    ) -> None:
        if not output.updated or output.diagnostics is None:
            raise ValueError("Train artifact can commit only a successful Hybrid update")
        expected_previous = output.global_optimizer_step - 1
        if self._payload["committed_optimizer_step"] != expected_previous:
            raise ValueError("Train artifact is not aligned to the previous optimizer step")

        new_records = self.provider.evaluation_records_since(
            self._provider_record_cursor
        )
        next_provider_record_cursor = self.provider.evaluation_record_count
        provider_by_hash = {record.expression_hash: record for record in new_records}
        by_hash = {
            str(record["structural_hash"]): dict(record)
            for record in self._payload["records"]
        }
        diagnostics = output.diagnostics
        for trajectory in output.collection.trajectories:
            expression = trajectory.terminal_expression.canonicalize()
            structural_hash = expression.structural_hash()
            existing = by_hash.get(structural_hash)
            if existing is None:
                reward_result: Any
                node_count = expression.stats.node_count
                if node_count in trainer.config.objective.exact_tb_node_counts:
                    lookup = trainer.exhaustive_reward_lookups_by_N.get(node_count)
                    if lookup is None:
                        raise ValueError("exact candidate lacks an approved registry lookup")
                    metadata = lookup.lookup(expression).metadata
                    reward_result = metadata.get("reward_result", metadata)
                else:
                    provider_record = provider_by_hash.get(structural_hash)
                    if provider_record is None:
                        raise ValueError(
                            "successful candidate lacks an actual Train evaluation record"
                        )
                    reward_result = provider_record
                existing = _candidate_record(
                    expression,
                    reward_result,
                    contract_fingerprint=self.contract_fingerprint,
                    optimizer_step=output.global_optimizer_step,
                    cycle_index=diagnostics.cycle_index,
                    condition_position=diagnostics.condition_position_in_cycle,
                )
                by_hash[structural_hash] = existing
            else:
                identity = _candidate_record(
                    expression,
                    provider_by_hash.get(
                        structural_hash,
                        _reward_payload_from_artifact_record(existing),
                    ),
                    contract_fingerprint=self.contract_fingerprint,
                    optimizer_step=output.global_optimizer_step,
                    cycle_index=diagnostics.cycle_index,
                    condition_position=diagnostics.condition_position_in_cycle,
                )
                if _metric_identity(existing) != _metric_identity(identity):
                    raise ValueError("duplicate structural hash has conflicting Train metrics")
                existing["last_seen"] = identity["last_seen"]
                existing["visit_count"] = int(existing["visit_count"]) + 1

        self._write(list(by_hash.values()), output.global_optimizer_step)
        self._payload = self._load(output.global_optimizer_step)
        self._provider_record_cursor = next_provider_record_cursor


__all__ = [
    "TRAIN_CANDIDATE_ARTIFACT_FILENAME",
    "TRAIN_CANDIDATE_ARTIFACT_SCHEMA",
    "TRAIN_CANDIDATE_RECORD_SCHEMA",
    "TRAIN_EVALUATION_CONTRACT_SCHEMA",
    "TrainCandidateArtifactWriter",
    "supports_train_candidate_artifact",
]
