import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest

from factor_gfn.backtest.baseline_factor_pool import (
    BASELINE_FACTOR_POOL_MANIFEST_FILENAME,
    BaselineFactorPoolFreezeInputs,
    BaselineFactorPoolIntegrityError,
    _fingerprint_payload,
    freeze_baseline_factor_pool,
    load_verified_baseline_factor_pool,
)
from factor_gfn.backtest.candidate_import import (
    CANDIDATE_IMPORT_MANIFEST_SCHEMA,
    CANDIDATE_REGISTRY_SCHEMA,
    NORMALIZATION_SCHEMA,
    NORMALIZED_ORIGIN_SCHEMA,
)
from factor_gfn.backtest.expression_compatibility import (
    ACCEPTED_REGISTRY_SCHEMA,
    EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA,
)
from factor_gfn.backtest.sources import SOURCE_SET_SCHEMA, SOURCE_SNAPSHOT_SCHEMA
from factor_gfn.backtest.stage6_evaluation import (
    STAGE6_EVALUATION_RESULT_SCHEMA,
    Stage6CandidateEvaluationResult,
    _stable_hash,
)
from factor_gfn.backtest.stage6_evaluation_store import (
    EvaluationStore,
    Stage6EvaluationRunner,
)
from factor_gfn.backtest.stage6_selection import (
    STAGE6_SELECTION_RESULT_SCHEMA,
    Stage6SelectionConfig,
)
from factor_gfn.backtest.stage6_survivor_enrichment import (
    STAGE6_ENRICHED_SELECTION_MANIFEST_SCHEMA,
    STAGE6_ENRICHED_SELECTION_VERSION,
    STAGE6_LONG_EXCESS_ENRICHMENT_SCHEMA,
)
from factor_gfn.backtest.stage6_train_reuse import (
    HYBRID_TRAIN_REUSE_ADAPTER_VERSION,
    HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA,
    HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA,
)
from factor_gfn.backtest.stage6_two_phase_pipeline import (
    STAGE6_TRAIN_ENTRY_SCHEMA,
    STAGE6_TRAIN_PASS_MANIFEST_SCHEMA,
    STAGE6_TRAIN_PREPARATION_SCOPE,
    STAGE6_VALIDATION_ENTRY_SCHEMA,
    STAGE6_VALIDATION_SCOPE,
    STAGE6_TWO_PHASE_VERSION,
    TRAIN_NOT_EVALUATED_VALIDATION,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, *, count: int | None = None) -> dict:
    result = {"size_bytes": path.stat().st_size, "sha256": _sha(path)}
    if count is not None:
        result["count"] = count
    return result


def _rehash_entry(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    stable = {key: item for key, item in value.items() if key != "entry_manifest_fingerprint"}
    value["entry_manifest_fingerprint"] = _stable_hash(stable)
    _write_json(path, value)
    return value


class _SyntheticEvaluator:
    def __init__(
        self,
        *,
        candidates: list[dict],
        context_fingerprint: str,
        contract_fingerprint: str,
        compatibility_fingerprint: str,
        accepted_fingerprint: str,
        mode: str,
        frozen_train: dict[str, dict] | None = None,
    ) -> None:
        self.candidates = {row["current_structural_hash"]: row for row in candidates}
        self.context = SimpleNamespace(fingerprint=context_fingerprint)
        self.evaluation_contract_fingerprint = contract_fingerprint
        self.compatibility_audit_fingerprint = compatibility_fingerprint
        self.accepted_registry_fingerprint = accepted_fingerprint
        self.mode = mode
        self.frozen_train = frozen_train or {}

    def resolve_candidate_identity(self, candidate):
        return {
            "structural_hash": candidate["current_structural_hash"],
            "formula": candidate["formula"],
            "prefix_token_ids": candidate["prefix_token_ids"],
            "node_count": candidate["node_count"],
            "depth": candidate["depth"],
        }

    def _train(self, candidate: dict) -> dict:
        index = int(candidate["synthetic_index"])
        direction = int(candidate["synthetic_direction"])
        return {
            "requested_date_range": ["2010-01-01", "2018-12-31"],
            "actual_date_range": ["2010-01-04", "2018-12-28"],
            "rebalance_dates": ["2018-12-21", "2018-12-28"],
            "rebalance_periods": 2,
            "ic": {
                "mean": direction * (0.02 + index * 0.01),
                "std": None if index == 0 else 0.04,
                "icir": None if index == 0 else 0.75,
                "valid_periods": 100 + index,
                "total_periods": 110,
            },
            "long": {
                "direction": direction,
                "mean_period_return": None,
                "annualized_return": None,
                "annualized_ir": 0.4 + index * 0.1,
                "std": None,
                "valid_periods": 100 + index,
                "total_periods": 110,
                "excess_series": {
                    "dates": ["2018-12-21", "2018-12-28"],
                    "values": [0.01, 0.02],
                },
            },
            "barra": {
                "max_abs_correlation": 0.2 + index * 0.05,
                "correlations": {},
                "common_valid_periods": {},
            },
            "train_prefilter": {
                "status": "train_prefilter_passed",
                "condition_results": {},
                "failed_conditions": [],
            },
        }

    def evaluate(self, candidate):
        expression = self.resolve_candidate_identity(candidate)
        source_identity = {
            "compatibility_audit_fingerprint": self.compatibility_audit_fingerprint,
            "accepted_registry_fingerprint": self.accepted_registry_fingerprint,
            "compatibility_record_fingerprint": candidate[
                "compatibility_record_fingerprint"
            ],
            "source_claimed_structural_hash": candidate[
                "source_claimed_structural_hash"
            ],
            "origin_ids": list(candidate["origin_ids"]),
            "source_ids": list(candidate["source_ids"]),
        }
        if self.mode == "train":
            train = self._train(candidate)
            validation = {"availability": TRAIN_NOT_EVALUATED_VALIDATION}
            validation_seconds = 0.0
        else:
            train_result = self.frozen_train[candidate["current_structural_hash"]]
            train = train_result["train"]
            index = int(candidate["synthetic_index"])
            validation = {
                "requested_date_range": ["2019-01-01", "2020-12-31"],
                "actual_date_range": ["2019-01-02", "2020-12-31"],
                "rebalance_dates": ["2020-12-24", "2020-12-31"],
                "rebalance_periods": 2,
                "ic": {
                    "mean": candidate["synthetic_direction"] * (0.018 + index * 0.01),
                    "std": 0.03,
                    "icir": 0.7 + index * 0.1,
                    "valid_periods": 40 + index,
                    "total_periods": 42,
                },
                "long": {
                    "direction": candidate["synthetic_direction"],
                    "mean_period_return": 0.01,
                    "annualized_return": 0.2,
                    "annualized_ir": 0.35 + index * 0.1,
                    "std": 0.02,
                    "valid_periods": 40 + index,
                    "total_periods": 42,
                    "excess_series": {
                        "dates": ["2020-12-24", "2020-12-31"],
                        "values": [0.01, 0.015],
                    },
                },
                "barra": {
                    "max_abs_correlation": 0.25,
                    "correlations": {},
                    "common_valid_periods": {},
                },
            }
            validation_seconds = 0.01
            source_identity["frozen_train_result_fingerprint"] = train_result[
                "result_fingerprint"
            ]
        direction = int(candidate["synthetic_direction"])
        deterministic = {
            "schema": STAGE6_EVALUATION_RESULT_SCHEMA,
            "status": "completed",
            "invalid_reasons": [],
            "expression": expression,
            "context_fingerprint": self.context.fingerprint,
            "evaluation_contract_fingerprint": self.evaluation_contract_fingerprint,
            "train_direction": direction,
            "train": train,
            "validation": validation,
            "factor_finite_coverage": {"train": {}, "validation": {}},
        }
        return Stage6CandidateEvaluationResult(
            schema=STAGE6_EVALUATION_RESULT_SCHEMA,
            status="completed",
            invalid_reasons=(),
            expression=MappingProxyType(expression),
            source_identity=MappingProxyType(source_identity),
            context_fingerprint=self.context.fingerprint,
            evaluation_contract_fingerprint=self.evaluation_contract_fingerprint,
            train_direction=direction,
            train=MappingProxyType(train),
            validation=MappingProxyType(validation),
            factor_finite_coverage=MappingProxyType(
                {"train": {}, "validation": {}}
            ),
            factor_seconds=0.01,
            train_evaluation_seconds=0.01,
            validation_evaluation_seconds=validation_seconds,
            total_seconds=0.02,
            result_fingerprint=_stable_hash(deterministic),
        )


class _FormalFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_root = root / "published_runs"
        self.source_id = "hybrid-formal-source"
        self.context_fingerprint = "1" * 64
        self.config_fingerprint = "8" * 64
        self.reward_provider_fingerprint = "9" * 64
        self.source_train_contract = {
            "schema": "factor_gfn.train_evaluation_contract.v1",
            "provider_fingerprint": self.reward_provider_fingerprint,
            "train_scope_projection": {"context_fingerprint": self.context_fingerprint},
            "implementation": {"synthetic": True},
        }
        self.base_contract_fingerprint = _stable_hash(self.source_train_contract)
        self.train_contract_fingerprint = "3" * 64
        self.validation_contract_fingerprint = "4" * 64
        self.candidates = [
            self._candidate(0, direction=1),
            self._candidate(1, direction=-1),
        ]
        self._build_sources()
        self._build_candidate_import()
        self._build_compatibility()
        self._build_overlay()
        self._build_train()
        self._build_validation()
        self._build_selection()
        self.inputs = BaselineFactorPoolFreezeInputs(
            source_set_manifest_path=self.source_set_path,
            candidate_import_manifest_path=self.candidate_path,
            compatibility_manifest_path=self.compatibility_path,
            train_reuse_manifest_path=self.overlay_path,
            train_entry_manifest_path=self.train_entry_path,
            train_pass_manifest_path=self.train_pass_path,
            validation_entry_manifest_path=self.validation_entry_path,
            enriched_selection_manifest_path=self.selection_path,
        )

    def _candidate(self, index: int, *, direction: int) -> dict:
        structural_hash = f"{index + 10:064x}"
        return {
            "schema": ACCEPTED_REGISTRY_SCHEMA,
            "current_structural_hash": structural_hash,
            "source_claimed_structural_hash": structural_hash,
            "formula": f"synthetic_factor_{index}",
            "prefix_token_ids": [index + 1],
            "node_count": index + 1,
            "depth": index,
            "origin_ids": [f"origin-{index}"],
            "source_ids": [self.source_id],
            "compatibility_record_fingerprint": f"{index + 20:064x}",
            "historical_metric_reuse": "forbidden",
            "stage6_metric_recompute_required": True,
            "synthetic_index": index,
            "synthetic_direction": direction,
        }

    def _build_sources(self) -> None:
        base = self.root / "source_authority"
        building = base / "building"
        building.mkdir(parents=True)
        runner_state = {
            "schema": "factor_gfn.hybrid_variance_runner.v1",
            "complete": True,
            "pending_assignment": None,
            "global_optimizer_step": 100,
        }
        run_config = {
            "schema": "factor_gfn.hybrid_variance_runner.v1",
            "checkpoint_schema": "factor_gfn.checkpoint.hybrid_variance.v1",
            "objective_mode": "hybrid_variance",
            "config_fingerprint": self.config_fingerprint,
            "reward_provider_fingerprint": self.reward_provider_fingerprint,
        }
        _write_json(building / "runner_state.json", runner_state)
        _write_json(building / "hybrid_run_config.json", run_config)
        train_artifact = {
            "schema": "factor_gfn.stage5_train_candidate_artifact.v1",
            "train_evaluation_contract": self.source_train_contract,
            "train_evaluation_contract_fingerprint": self.base_contract_fingerprint,
            "committed_optimizer_step": 100,
            "candidate_count": len(self.candidates),
            "records": [
                {
                    "schema": "factor_gfn.stage5_train_candidate_record.v1",
                    "structural_hash": row["current_structural_hash"],
                    "formula": row["formula"],
                    "prefix_token_ids": row["prefix_token_ids"],
                    "node_count": row["node_count"],
                    "depth": row["depth"],
                    "train_evaluation_contract_fingerprint": self.base_contract_fingerprint,
                }
                for row in self.candidates
            ],
        }
        files = {
            "runner_state.json": runner_state,
            "hybrid_run_config.json": run_config,
            "train_candidate_artifact.json": train_artifact,
            "checkpoint_metadata.json": {
                "schema": "factor_gfn.checkpoint.hybrid_variance.v1",
                "global_optimizer_step": 100,
                "config_fingerprint": self.config_fingerprint,
                "reward_provider_fingerprint": self.reward_provider_fingerprint,
            },
        }
        for name, value in files.items():
            _write_json(building / name, value)
        checkpoint = base / "checkpoint_latest.pt"
        checkpoint.write_bytes(b"synthetic-checkpoint")
        artifacts = [
            {"name": name, **_artifact(building / name)} for name in files
        ]
        snapshot = {
            "schema": SOURCE_SNAPSHOT_SCHEMA,
            "source_id": self.source_id,
            "source_type": "hybrid_train_artifact",
            "source_role": "formal_discovery",
            "inclusion_status": "approved",
            "approval_note": "synthetic formal fixture",
            "candidate_record_policy": {
                "selection": "all_canonical_train_candidate_artifact_records",
                "included_branches": [],
                "included_record_sources": [],
                "excluded_record_sources": [],
            },
            "source_semantics": {
                "source_artifact_schema": "factor_gfn.stage5_train_candidate_artifact.v1"
            },
            "snapshot_kind": "completed_hybrid_train_artifact",
            "cutoff": {
                "committed_optimizer_step": 100,
                "candidate_count": len(self.candidates),
                "complete": True,
                "pending_assignment": None,
            },
            "record_counts": {"records": len(self.candidates)},
            "artifacts": artifacts,
            "external_artifacts": [
                {
                    "name": "checkpoint_latest.pt",
                    "source_path": str(checkpoint.resolve()),
                    **_artifact(checkpoint),
                    "metadata": {"global_optimizer_step": 100},
                }
            ],
            "logical_content_fingerprint": _stable_hash(train_artifact["records"]),
        }
        snapshot_payload = {
            key: snapshot[key]
            for key in (
                "schema",
                "source_id",
                "source_type",
                "source_role",
                "inclusion_status",
                "approval_note",
                "candidate_record_policy",
                "source_semantics",
                "snapshot_kind",
                "cutoff",
                "record_counts",
                "artifacts",
                "logical_content_fingerprint",
                "external_artifacts",
            )
        }
        self.snapshot_fingerprint = _stable_hash(snapshot_payload)
        snapshot["snapshot_fingerprint"] = self.snapshot_fingerprint
        snapshot["created_at_utc"] = "2026-01-01T00:00:00+00:00"
        snapshot_dir = (
            base / "sources" / self.source_id / self.snapshot_fingerprint
        )
        snapshot_dir.mkdir(parents=True)
        for name in files:
            (snapshot_dir / name).write_bytes((building / name).read_bytes())
        self.snapshot_path = snapshot_dir / "source_snapshot.json"
        _write_json(self.snapshot_path, snapshot)
        source_payload = {
            "schema": SOURCE_SET_SCHEMA,
            "mode": "provisional",
            "sources": [
                {
                    "source_id": self.source_id,
                    "source_type": "hybrid_train_artifact",
                    "source_role": "formal_discovery",
                    "snapshot_fingerprint": self.snapshot_fingerprint,
                }
            ],
        }
        self.source_set_fingerprint = _stable_hash(source_payload)
        source_set = {
            **source_payload,
            "source_set_fingerprint": self.source_set_fingerprint,
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "source_manifests": [
                {
                    **source_payload["sources"][0],
                    "snapshot_manifest": str(self.snapshot_path.resolve()),
                }
            ],
        }
        self.source_set_path = (
            base
            / "source_sets"
            / self.source_set_fingerprint
            / "source_set_manifest.json"
        )
        _write_json(self.source_set_path, source_set)

    def _build_candidate_import(self) -> None:
        origins = [
            {
                "schema": NORMALIZED_ORIGIN_SCHEMA,
                "origin_id": row["origin_ids"][0],
                "source_id": self.source_id,
            }
            for row in self.candidates
        ]
        groups = [
            {
                "schema": CANDIDATE_REGISTRY_SCHEMA,
                "source_claimed_structural_hash": row[
                    "source_claimed_structural_hash"
                ],
                "origin_ids": row["origin_ids"],
                "source_ids": row["source_ids"],
                "representations": [],
                "downstream_eligible": True,
            }
            for row in self.candidates
        ]
        fingerprint_payload = {
            "source_set_fingerprint": self.source_set_fingerprint,
            "normalization_schema": NORMALIZATION_SCHEMA,
            "adapter_versions": ["synthetic-hybrid-adapter"],
            "normalized_origins_digest": _stable_hash(origins),
            "claimed_hash_groups_digest": _stable_hash(groups),
            "schema_rejection_ledger_digest": _stable_hash([]),
            "representation_conflict_ledger_digest": _stable_hash([]),
        }
        self.registry_fingerprint = _stable_hash(fingerprint_payload)
        directory = self.root / "candidate_import" / self.registry_fingerprint
        origins_path = directory / "normalized_candidate_origins.jsonl"
        groups_path = directory / "candidate_registry.jsonl"
        _write_jsonl(origins_path, origins)
        _write_jsonl(groups_path, groups)
        manifest = {
            "schema": CANDIDATE_IMPORT_MANIFEST_SCHEMA,
            "mode": "provisional",
            "source_set_fingerprint": self.source_set_fingerprint,
            "normalization_schema": NORMALIZATION_SCHEMA,
            "adapter_versions": ["synthetic-hybrid-adapter"],
            "registry_fingerprint": self.registry_fingerprint,
            "registry_status": "complete",
            "downstream_eligible": True,
            "downstream_block_reasons": [],
            "counts": {
                "normalized_origins": len(origins),
                "schema_rejected": 0,
                "claimed_hash_groups": len(groups),
            },
            "digests": {
                "normalized_origins": _stable_hash(origins),
                "claimed_hash_groups": _stable_hash(groups),
            },
            "fingerprint_payload": fingerprint_payload,
            "artifacts": {
                origins_path.name: _artifact(origins_path),
                groups_path.name: _artifact(groups_path),
            },
        }
        self.candidate_path = directory / "candidate_import_manifest.json"
        _write_json(self.candidate_path, manifest)

    def _build_compatibility(self) -> None:
        audit_rows = []
        accepted_digest = _stable_hash(self.candidates)
        fingerprint_payload = {
            "candidate_registry_fingerprint": self.registry_fingerprint,
            "source_set_fingerprint": self.source_set_fingerprint,
            "target_semantics_fingerprint": "5" * 64,
            "auditor_version": "synthetic-auditor-v1",
            "audit_digest": _stable_hash(audit_rows),
            "accepted_registry_digest": accepted_digest,
            "review_ledger_digest": _stable_hash([]),
            "reject_ledger_digest": _stable_hash([]),
        }
        self.compatibility_fingerprint = _stable_hash(fingerprint_payload)
        directory = self.root / "compatibility" / self.compatibility_fingerprint
        audit_path = directory / "expression_compatibility_audit.jsonl"
        accepted_path = directory / "auto_accepted_candidate_registry.jsonl"
        _write_jsonl(audit_path, audit_rows)
        _write_jsonl(accepted_path, self.candidates)
        manifest = {
            "schema": EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA,
            "mode": "provisional",
            "candidate_registry_fingerprint": self.registry_fingerprint,
            "source_set_fingerprint": self.source_set_fingerprint,
            "audit_fingerprint": self.compatibility_fingerprint,
            "accepted_registry_fingerprint": accepted_digest,
            "audit_status": "complete",
            "downstream_eligible": True,
            "counts": {
                "claimed_hash_groups": len(self.candidates),
                "AUTO_ACCEPT": len(self.candidates),
                "AUTO_REJECT": 0,
                "REVIEW_REQUIRED": 0,
                "accepted_registry_candidates": len(self.candidates),
            },
            "digests": {
                "compatibility_audit": _stable_hash(audit_rows),
                "accepted_registry": accepted_digest,
                "review_ledger": _stable_hash([]),
                "reject_ledger": _stable_hash([]),
            },
            "fingerprint_payload": fingerprint_payload,
            "artifacts": {
                audit_path.name: _artifact(audit_path),
                accepted_path.name: _artifact(accepted_path),
            },
        }
        self.accepted_fingerprint = accepted_digest
        self.compatibility_path = directory / "expression_compatibility_manifest.json"
        _write_json(self.compatibility_path, manifest)

    def _build_overlay(self) -> None:
        overlay_rows = []
        for row in self.candidates:
            payload = {
                "schema": HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA,
                "structural_hash": row["current_structural_hash"],
                "node_count": row["node_count"],
                "train_metric_origin": "stage5_verified_reuse",
                "verification_mode": "hybrid_exact_contract",
                "train_evaluation_contract_fingerprint": self.base_contract_fingerprint,
                "train_metrics": {},
                "train_long_excess": None if row["node_count"] <= 2 else {},
                "origins": [],
            }
            overlay_rows.append(
                {**payload, "record_fingerprint": _stable_hash(payload)}
            )
        verification = {
            "verification_mode": "hybrid_exact_contract",
            "numeric_verification": "not_required_by_approved_hybrid_contract",
            "fresh_train_fallback_reason": None,
            "old_train_metrics_allowed": True,
            "result": "TRAIN_METRICS_REUSABLE",
        }
        payload = {
            "schema": HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA,
            "adapter_version": HYBRID_TRAIN_REUSE_ADAPTER_VERSION,
            "verification_mode": "hybrid_exact_contract",
            "fresh_train_fallback_reason": None,
            "source_set_fingerprint": self.source_set_fingerprint,
            "source_snapshot_fingerprint": self.snapshot_fingerprint,
            "candidate_registry_fingerprint": self.registry_fingerprint,
            "compatibility_audit_fingerprint": self.compatibility_fingerprint,
            "accepted_registry_fingerprint": self.accepted_fingerprint,
            "stage6_context_fingerprint": self.context_fingerprint,
            "stage6_evaluation_contract_fingerprint": self.base_contract_fingerprint,
            "target_provider_fingerprint": "6" * 64,
            "artifact_train_contract_fingerprint": self.base_contract_fingerprint,
            "current_train_contract_fingerprint": self.base_contract_fingerprint,
            "train_contract_verification_digest": _stable_hash(verification),
            "overlay_digest": _stable_hash(overlay_rows),
        }
        self.overlay_fingerprint = _stable_hash(payload)
        directory = self.root / "overlay" / self.overlay_fingerprint
        rows_path = directory / "train_reuse_overlay.jsonl"
        verification_path = directory / "train_reuse_contract_verification.json"
        _write_jsonl(rows_path, overlay_rows)
        _write_json(verification_path, verification)
        manifest = {
            **payload,
            "train_reuse_overlay_fingerprint": self.overlay_fingerprint,
            "counts": {
                "accepted_candidates": len(self.candidates),
                "overlay_candidates": len(self.candidates),
                "fresh_train_fallback_candidates": 0,
                "numeric_verification_candidates": 0,
            },
            "coverage_ratio": 1.0,
            "artifacts": {
                rows_path.name: _artifact(rows_path),
                verification_path.name: _artifact(verification_path),
            },
        }
        self.overlay_path = directory / "train_reuse_manifest.json"
        _write_json(self.overlay_path, manifest)

    def _entry(
        self,
        *,
        schema: str,
        scope: str,
        run,
        database: Path,
        artifacts: Path,
        extra: dict,
    ) -> dict:
        stable = {
            "schema": schema,
            "pipeline_version": STAGE6_TWO_PHASE_VERSION,
            "evaluation_run_id": run.run_id,
            "evaluation_run_scope": scope,
            "candidate_count": run.candidate_count,
            "context_fingerprint": run.manifest["context_fingerprint"],
            "evaluation_contract_fingerprint": run.manifest[
                "evaluation_contract_fingerprint"
            ],
            "database_path": str(database.resolve()),
            "run_artifact_root": str(artifacts.resolve()),
            "oos": "not_loaded_not_evaluated",
            **extra,
        }
        return {**stable, "entry_manifest_fingerprint": _stable_hash(stable)}

    def _build_train(self) -> None:
        root = self.root / "train"
        database = root / "store" / "train.sqlite"
        artifacts = root / "evaluation_runs"
        evaluator = _SyntheticEvaluator(
            candidates=self.candidates,
            context_fingerprint=self.context_fingerprint,
            contract_fingerprint=self.train_contract_fingerprint,
            compatibility_fingerprint=self.compatibility_fingerprint,
            accepted_fingerprint=self.accepted_fingerprint,
            mode="train",
        )
        with EvaluationStore(database, artifacts) as store:
            run = store.create_run(
                self.candidates, evaluator, scope=STAGE6_TRAIN_PREPARATION_SCOPE
            )
            Stage6EvaluationRunner(store, evaluator).run(run.run_id)
            verified = store.load_verified_run_results(run.run_id)
        self.train_results = {
            row["structural_hash"]: row["result"] for row in verified.records
        }
        entry = self._entry(
            schema=STAGE6_TRAIN_ENTRY_SCHEMA,
            scope=STAGE6_TRAIN_PREPARATION_SCOPE,
            run=run,
            database=database,
            artifacts=artifacts,
            extra={
                "compatibility_audit_fingerprint": self.compatibility_fingerprint,
                "accepted_registry_fingerprint": self.accepted_fingerprint,
                "train_reuse_overlay_fingerprint": self.overlay_fingerprint,
                "base_fresh_evaluation_contract_fingerprint": self.base_contract_fingerprint,
                "validation_evaluation_count": 0,
            },
        )
        self.train_entry_path = root / "train_preparation_entry_manifest.json"
        _write_json(self.train_entry_path, entry)
        decisions = [
            {
                "ordinal": index,
                "structural_hash": row["current_structural_hash"],
                "train_result_fingerprint": self.train_results[
                    row["current_structural_hash"]
                ]["result_fingerprint"],
                "train_direction": row["synthetic_direction"],
                "status": "train_prefilter_passed",
                "condition_results": {},
                "failed_conditions": [],
            }
            for index, row in enumerate(self.candidates)
        ]
        pass_root = root / "train_pass_manifest"
        decisions_path = pass_root / "train_prefilter_results.jsonl"
        candidates_path = pass_root / "train_pass_candidates.jsonl"
        _write_jsonl(decisions_path, decisions)
        _write_jsonl(candidates_path, self.candidates)
        stable = {
            "schema": STAGE6_TRAIN_PASS_MANIFEST_SCHEMA,
            "pipeline_version": STAGE6_TWO_PHASE_VERSION,
            "train_entry_manifest_fingerprint": entry[
                "entry_manifest_fingerprint"
            ],
            "train_evaluation_run_id": run.run_id,
            "train_ordered_result_set_fingerprint": verified.ordered_result_set_fingerprint,
            "accepted_registry_fingerprint": self.accepted_fingerprint,
            "selection_config_fingerprint": Stage6SelectionConfig().fingerprint,
            "candidate_count": len(self.candidates),
            "train_pass_count": len(self.candidates),
            "train_prefilter_failed_count": 0,
            "verified_stage5_train_reuse_count": len(self.candidates),
            "stage6_fresh_train_count": 0,
            "validation_evaluation_count": 0,
            "failed_condition_counts": {},
            "artifacts": {
                decisions_path.name: _artifact(
                    decisions_path, count=len(decisions)
                ),
                candidates_path.name: _artifact(
                    candidates_path, count=len(self.candidates)
                ),
            },
            "ordered_train_pass_hashes_fingerprint": _stable_hash(
                [row["current_structural_hash"] for row in self.candidates]
            ),
            "oos": "not_loaded_not_evaluated",
        }
        manifest = {
            **stable,
            "train_pass_manifest_fingerprint": _stable_hash(stable),
        }
        self.train_pass_path = pass_root / "train_pass_manifest.json"
        _write_json(self.train_pass_path, manifest)

    def _build_validation(self) -> None:
        root = self.root / "validation"
        database = root / "store" / "validation.sqlite"
        artifacts = root / "evaluation_runs"
        evaluator = _SyntheticEvaluator(
            candidates=self.candidates,
            context_fingerprint=self.context_fingerprint,
            contract_fingerprint=self.validation_contract_fingerprint,
            compatibility_fingerprint=self.compatibility_fingerprint,
            accepted_fingerprint=self.accepted_fingerprint,
            mode="validation",
            frozen_train=self.train_results,
        )
        with EvaluationStore(database, artifacts) as store:
            run = store.create_run(
                self.candidates, evaluator, scope=STAGE6_VALIDATION_SCOPE
            )
            Stage6EvaluationRunner(store, evaluator).run(run.run_id)
            verified = store.load_verified_run_results(run.run_id)
        self.validation_results = {
            row["structural_hash"]: row["result"] for row in verified.records
        }
        train_entry = json.loads(self.train_entry_path.read_text(encoding="utf-8"))
        train_pass = json.loads(self.train_pass_path.read_text(encoding="utf-8"))
        entry = self._entry(
            schema=STAGE6_VALIDATION_ENTRY_SCHEMA,
            scope=STAGE6_VALIDATION_SCOPE,
            run=run,
            database=database,
            artifacts=artifacts,
            extra={
                "compatibility_audit_fingerprint": self.compatibility_fingerprint,
                "accepted_registry_fingerprint": self.accepted_fingerprint,
                "train_entry_manifest_path": str(self.train_entry_path.resolve()),
                "train_entry_manifest_fingerprint": train_entry[
                    "entry_manifest_fingerprint"
                ],
                "train_pass_manifest_path": str(self.train_pass_path.resolve()),
                "train_pass_manifest_fingerprint": train_pass[
                    "train_pass_manifest_fingerprint"
                ],
                "base_fresh_evaluation_contract_fingerprint": self.base_contract_fingerprint,
            },
        )
        self.validation_run = verified
        self.validation_entry_path = root / "validation_evaluation_entry_manifest.json"
        _write_json(self.validation_entry_path, entry)

    def _build_selection(self) -> None:
        # Deliberately place the lower-IC factor first to prove freeze never reranks.
        alpha_order = [self.candidates[0], self.candidates[1]]
        hard_rows = []
        greedy_rows = []
        alpha_rows = []
        enrichment_rows = []
        for rank, candidate in enumerate(alpha_order, start=1):
            structural_hash = candidate["current_structural_hash"]
            result = self.validation_results[structural_hash]
            metrics = {
                "train_ic": result["train"]["ic"]["mean"],
                "validation_ic": result["validation"]["ic"]["mean"],
                "train_long_ir": result["train"]["long"]["annualized_ir"],
                "validation_long_ir": result["validation"]["long"][
                    "annualized_ir"
                ],
                "train_barra_ts_corr": result["train"]["barra"][
                    "max_abs_correlation"
                ],
            }
            hard_rows.append(
                {
                    "schema": STAGE6_SELECTION_RESULT_SCHEMA,
                    "evaluation_ordinal": rank - 1,
                    "structural_hash": structural_hash,
                    "result_fingerprint": result["result_fingerprint"],
                    "base_result_fingerprint": result["result_fingerprint"],
                    "expression": result["expression"],
                    "source_identity": result["source_identity"],
                    "evaluation_status": "completed",
                    "evaluation_ineligible": False,
                    "metrics": metrics,
                    "condition_results": {},
                    "failed_conditions": [],
                    "hard_filter_pass": True,
                    "train_direction": result["train_direction"],
                }
            )
            enrichment_payload = {
                "schema": STAGE6_LONG_EXCESS_ENRICHMENT_SCHEMA,
                "structural_hash": structural_hash,
                "status": "already_available",
                "base_result_fingerprint": result["result_fingerprint"],
            }
            enrichment = {
                **enrichment_payload,
                "result_fingerprint": _stable_hash(enrichment_payload),
            }
            enrichment_rows.append(enrichment)
            effective = _stable_hash(
                {
                    "base": result["result_fingerprint"],
                    "enrichment": enrichment["result_fingerprint"],
                }
            )
            greedy = {
                "schema": STAGE6_SELECTION_RESULT_SCHEMA,
                "sorted_rank": rank,
                "structural_hash": structural_hash,
                "base_result_fingerprint": result["result_fingerprint"],
                "effective_result_fingerprint": effective,
                "enrichment_result_fingerprint": enrichment[
                    "result_fingerprint"
                ],
                "abs_train_ic": abs(metrics["train_ic"]),
                "greedy_retained": True,
                "decorrelation_status": "retained",
                "comparison_trace": [],
            }
            greedy_rows.append(greedy)
            alpha_rows.append(
                {
                    **greedy,
                    "expression": result["expression"],
                    "source_identity": result["source_identity"],
                    "train_direction": result["train_direction"],
                    "metrics": metrics,
                }
            )
        deterministic = {
            "schema": STAGE6_ENRICHED_SELECTION_MANIFEST_SCHEMA,
            "version": STAGE6_ENRICHED_SELECTION_VERSION,
            "engineering_smoke": False,
            "evaluation_run_id": self.validation_run.run_id,
            "evaluation_ordered_result_set_fingerprint": self.validation_run.ordered_result_set_fingerprint,
            "evaluation_contract_fingerprint": self.validation_run.manifest[
                "evaluation_contract_fingerprint"
            ],
            "context_fingerprint": self.validation_run.manifest[
                "context_fingerprint"
            ],
            "selection_contract_fingerprint": Stage6SelectionConfig().fingerprint,
            "enrichment_contract_fingerprint": "7" * 64,
            "hard_filter_digest": _stable_hash(hard_rows),
            "enrichment_digest": _stable_hash(enrichment_rows),
            "greedy_digest": _stable_hash(greedy_rows),
            "alpha_pool_digest": _stable_hash(alpha_rows),
            "oos": "not_loaded_not_evaluated",
        }
        self.selection_fingerprint = _stable_hash(deterministic)
        directory = self.root / "selection" / self.selection_fingerprint
        artifacts = {}
        for name, rows in (
            ("hard_filter_results.jsonl", hard_rows),
            ("survivor_long_excess_enrichment.jsonl", enrichment_rows),
            ("greedy_decorrelation_results.jsonl", greedy_rows),
            ("alpha_pool.jsonl", alpha_rows),
        ):
            path = directory / name
            _write_jsonl(path, rows)
            artifacts[name] = _artifact(path)
        manifest = {
            **deterministic,
            "enriched_selection_fingerprint": self.selection_fingerprint,
            "counts": {
                "input_candidates": len(hard_rows),
                "hard_filter_pass": len(hard_rows),
                "hard_filter_fail": 0,
                "retained": len(alpha_rows),
            },
            "artifacts": artifacts,
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "created_at_excluded_from_fingerprint": True,
            "scope": "provisional_selection",
        }
        self.selection_path = directory / "enriched_selection_manifest.json"
        _write_json(self.selection_path, manifest)

    def freeze(self):
        return freeze_baseline_factor_pool(
            self.inputs,
            self.runs_root,
            confirmed_for_freeze=True,
            reviewed_selection_fingerprint=self.selection_fingerprint,
        )

    def selection_variant(self, mutate) -> tuple[Path, str]:
        source = self.selection_path.parent
        staging = self.root / f"selection-building-{len(list(self.root.glob('selection-building-*')))}"
        shutil.copytree(source, staging)
        names = {
            "hard": "hard_filter_results.jsonl",
            "enrichment": "survivor_long_excess_enrichment.jsonl",
            "greedy": "greedy_decorrelation_results.jsonl",
            "alpha": "alpha_pool.jsonl",
        }
        rows = {
            key: [
                json.loads(line)
                for line in (staging / name).read_text(encoding="utf-8").splitlines()
            ]
            for key, name in names.items()
        }
        mutate(rows)
        for key, name in names.items():
            _write_jsonl(staging / name, rows[key])
        manifest = json.loads(
            (staging / "enriched_selection_manifest.json").read_text(encoding="utf-8")
        )
        manifest["hard_filter_digest"] = _stable_hash(rows["hard"])
        manifest["enrichment_digest"] = _stable_hash(rows["enrichment"])
        manifest["greedy_digest"] = _stable_hash(rows["greedy"])
        manifest["alpha_pool_digest"] = _stable_hash(rows["alpha"])
        manifest["counts"]["input_candidates"] = len(rows["hard"])
        manifest["counts"]["hard_filter_pass"] = sum(
            row.get("hard_filter_pass") is True for row in rows["hard"]
        )
        manifest["counts"]["hard_filter_fail"] = len(rows["hard"]) - manifest[
            "counts"
        ]["hard_filter_pass"]
        manifest["counts"]["retained"] = len(rows["alpha"])
        manifest["artifacts"] = {
            name: _artifact(staging / name) for name in names.values()
        }
        deterministic = {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "enriched_selection_fingerprint",
                "counts",
                "artifacts",
                "created_at_utc",
                "created_at_excluded_from_fingerprint",
                "scope",
            }
        }
        fingerprint = _stable_hash(deterministic)
        manifest["enriched_selection_fingerprint"] = fingerprint
        _write_json(staging / "enriched_selection_manifest.json", manifest)
        target = self.root / "selection" / fingerprint
        staging.rename(target)
        return target / "enriched_selection_manifest.json", fingerprint


class BaselineFactorPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = _FormalFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_freeze_preserves_composition_order_direction_and_metrics(self):
        artifact = self.fixture.freeze()
        loaded = load_verified_baseline_factor_pool(artifact.manifest_path)
        expected_hashes = tuple(
            row["current_structural_hash"] for row in self.fixture.candidates
        )
        self.assertEqual(loaded.ordered_structural_hashes, expected_hashes)
        self.assertEqual(loaded.frozen_train_directions, (1, -1))
        self.assertEqual(
            tuple(row.provisional_rank for row in loaded.records), (1, 2)
        )
        self.assertEqual(tuple(row.stage6_sorted_rank for row in loaded.records), (1, 2))
        self.assertEqual(loaded.records[0].depth, 0)
        self.assertIsNone(loaded.records[0].train_metrics["ic_std"])
        self.assertEqual(loaded.oos_status, "not_loaded_not_evaluated")
        self.assertFalse(artifact.reused_existing_artifact)

    def test_verified_loader_returns_deeply_read_only_contracts(self):
        loaded = load_verified_baseline_factor_pool(
            self.fixture.freeze().manifest_path
        )
        with self.assertRaises(TypeError):
            loaded.manifest["pool"]["ordered_structural_hashes"][0] = "x"
        with self.assertRaises(AttributeError):
            loaded.records[0].source_identity["source_ids"].append("x")

    def test_explicit_authorization_and_reviewed_identity_are_required(self):
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            freeze_baseline_factor_pool(self.fixture.inputs, self.fixture.runs_root)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            freeze_baseline_factor_pool(
                self.fixture.inputs,
                self.fixture.runs_root,
                confirmed_for_freeze=True,
                reviewed_selection_fingerprint="f" * 64,
            )

    def test_same_freeze_is_idempotent(self):
        first = self.fixture.freeze()
        second = self.fixture.freeze()
        self.assertEqual(
            first.baseline_factor_pool_fingerprint,
            second.baseline_factor_pool_fingerprint,
        )
        self.assertTrue(second.reused_existing_artifact)

    def test_stage5_incomplete_fails_closed(self):
        snapshot = json.loads(self.fixture.snapshot_path.read_text(encoding="utf-8"))
        snapshot["cutoff"]["complete"] = False
        _write_json(self.fixture.snapshot_path, snapshot)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_legacy_or_mixed_source_fails_closed(self):
        source = json.loads(self.fixture.source_set_path.read_text(encoding="utf-8"))
        source["sources"][0]["source_type"] = "discovery_run"
        _write_json(self.fixture.source_set_path, source)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_validation_incomplete_fails_closed(self):
        database = Path(
            json.loads(
                self.fixture.validation_entry_path.read_text(encoding="utf-8")
            )["database_path"]
        )
        import sqlite3

        connection = sqlite3.connect(database)
        try:
            connection.execute("UPDATE runs SET status='created'")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_engineering_smoke_fails_closed(self):
        selection = json.loads(self.fixture.selection_path.read_text(encoding="utf-8"))
        selection["engineering_smoke"] = True
        selection["scope"] = "engineering_branch_coverage_not_provisional_selection"
        _write_json(self.fixture.selection_path, selection)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_resource_limited_scope_fails_closed(self):
        selection = json.loads(self.fixture.selection_path.read_text(encoding="utf-8"))
        selection["scope"] = "resource_limited_provisional_selection"
        _write_json(self.fixture.selection_path, selection)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_oos_touched_fails_closed(self):
        selection = json.loads(self.fixture.selection_path.read_text(encoding="utf-8"))
        selection["oos"] = "loaded"
        _write_json(self.fixture.selection_path, selection)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_manifest_fingerprint_and_sha_mismatch_fail_closed(self):
        alpha = self.fixture.selection_path.parent / "alpha_pool.jsonl"
        alpha.write_bytes(alpha.read_bytes() + b"\n")
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_train_pass_validation_set_mismatch_fails_closed(self):
        pass_manifest = json.loads(
            self.fixture.train_pass_path.read_text(encoding="utf-8")
        )
        pass_root = self.fixture.train_pass_path.parent
        candidate_path = pass_root / "train_pass_candidates.jsonl"
        decision_path = pass_root / "train_prefilter_results.jsonl"
        candidates = [
            json.loads(line)
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
        ][:1]
        decisions = [
            json.loads(line)
            for line in decision_path.read_text(encoding="utf-8").splitlines()
        ]
        decisions[1]["status"] = "train_prefilter_failed"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(decision_path, decisions)
        pass_manifest["train_pass_count"] = 1
        pass_manifest["train_prefilter_failed_count"] = 1
        pass_manifest["artifacts"][candidate_path.name] = _artifact(
            candidate_path, count=1
        )
        pass_manifest["artifacts"][decision_path.name] = _artifact(
            decision_path, count=2
        )
        pass_manifest["ordered_train_pass_hashes_fingerprint"] = _stable_hash(
            [candidates[0]["current_structural_hash"]]
        )
        stable = {
            key: value
            for key, value in pass_manifest.items()
            if key != "train_pass_manifest_fingerprint"
        }
        pass_manifest["train_pass_manifest_fingerprint"] = _stable_hash(stable)
        _write_json(self.fixture.train_pass_path, pass_manifest)
        entry = json.loads(self.fixture.validation_entry_path.read_text(encoding="utf-8"))
        entry["train_pass_manifest_fingerprint"] = pass_manifest[
            "train_pass_manifest_fingerprint"
        ]
        _write_json(self.fixture.validation_entry_path, entry)
        _rehash_entry(self.fixture.validation_entry_path)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_hard_pass_and_greedy_input_mismatch_fails_closed(self):
        hard_path = self.fixture.selection_path.parent / "hard_filter_results.jsonl"
        hard = [json.loads(line) for line in hard_path.read_text(encoding="utf-8").splitlines()]
        hard[0]["hard_filter_pass"] = False
        _write_jsonl(hard_path, hard)
        selection = json.loads(self.fixture.selection_path.read_text(encoding="utf-8"))
        selection["artifacts"][hard_path.name] = _artifact(hard_path)
        selection["hard_filter_digest"] = _stable_hash(hard)
        deterministic = {
            key: value
            for key, value in selection.items()
            if key
            not in {
                "enriched_selection_fingerprint",
                "counts",
                "artifacts",
                "created_at_utc",
                "created_at_excluded_from_fingerprint",
                "scope",
            }
        }
        selection["enriched_selection_fingerprint"] = _stable_hash(deterministic)
        _write_json(self.fixture.selection_path, selection)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_retained_and_alpha_pool_mismatch_fails_closed(self):
        alpha_path = self.fixture.selection_path.parent / "alpha_pool.jsonl"
        alpha = [json.loads(alpha_path.read_text(encoding="utf-8").splitlines()[0])]
        _write_jsonl(alpha_path, alpha)
        selection = json.loads(self.fixture.selection_path.read_text(encoding="utf-8"))
        selection["artifacts"][alpha_path.name] = _artifact(alpha_path)
        selection["alpha_pool_digest"] = _stable_hash(alpha)
        selection["counts"]["retained"] = 1
        deterministic = {
            key: value
            for key, value in selection.items()
            if key
            not in {
                "enriched_selection_fingerprint",
                "counts",
                "artifacts",
                "created_at_utc",
                "created_at_excluded_from_fingerprint",
                "scope",
            }
        }
        selection["enriched_selection_fingerprint"] = _stable_hash(deterministic)
        _write_json(self.fixture.selection_path, selection)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_reporting_files_do_not_affect_freeze(self):
        first = self.fixture.freeze()
        reporting = self.fixture.root / "outputs" / "stage6_reporting" / "chart.png"
        reporting.parent.mkdir(parents=True)
        reporting.write_bytes(b"one")
        reporting.write_bytes(b"two")
        second = self.fixture.freeze()
        self.assertEqual(
            first.baseline_factor_pool_fingerprint,
            second.baseline_factor_pool_fingerprint,
        )

    def test_timestamp_is_excluded_from_pool_fingerprint(self):
        first = self.fixture.freeze()
        manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
        fingerprint = manifest["baseline_factor_pool_fingerprint"]
        manifest["created_at_utc"] = "2099-01-01T00:00:00+00:00"
        _write_json(first.manifest_path, manifest)
        loaded = load_verified_baseline_factor_pool(first.manifest_path)
        self.assertEqual(loaded.baseline_factor_pool_fingerprint, fingerprint)

    def test_fingerprint_binds_identity_direction_rank_selection_and_record_digest(self):
        artifact = self.fixture.freeze()
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        base = _stable_hash(_fingerprint_payload(manifest))
        mutations = []
        changed = json.loads(json.dumps(manifest))
        changed["pool"]["ordered_structural_hashes"][0] = "f" * 64
        mutations.append(changed)
        changed = json.loads(json.dumps(manifest))
        changed["pool"]["ordered_train_directions"][0] *= -1
        mutations.append(changed)
        changed = json.loads(json.dumps(manifest))
        changed["pool"]["ordered_provisional_ranks"] = [2, 1]
        mutations.append(changed)
        changed = json.loads(json.dumps(manifest))
        changed["upstream_provenance"]["enriched_selection_fingerprint"] = "e" * 64
        mutations.append(changed)
        changed = json.loads(json.dumps(manifest))
        changed["pool"]["factor_records_digest"] = "d" * 64
        mutations.append(changed)
        self.assertTrue(
            all(_stable_hash(_fingerprint_payload(value)) != base for value in mutations)
        )
        timestamp_only = json.loads(json.dumps(manifest))
        timestamp_only["created_at_utc"] = "2099-01-01T00:00:00+00:00"
        self.assertEqual(_stable_hash(_fingerprint_payload(timestamp_only)), base)

    def test_empty_formal_pool_fails_closed(self):
        def empty(rows):
            for row in rows["hard"]:
                row["hard_filter_pass"] = False
            rows["enrichment"] = []
            rows["greedy"] = []
            rows["alpha"] = []

        path, fingerprint = self.fixture.selection_variant(empty)
        inputs = BaselineFactorPoolFreezeInputs(
            **{
                **{
                    field: getattr(self.fixture.inputs, field)
                    for field in self.fixture.inputs.__dataclass_fields__
                },
                "enriched_selection_manifest_path": path,
            }
        )
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            freeze_baseline_factor_pool(
                inputs,
                self.fixture.runs_root,
                confirmed_for_freeze=True,
                reviewed_selection_fingerprint=fingerprint,
            )

    def test_different_formal_selection_creates_new_identity_and_keeps_old_pool(self):
        first = self.fixture.freeze()

        def reverse(rows):
            rows["greedy"].reverse()
            rows["alpha"].reverse()

        path, fingerprint = self.fixture.selection_variant(reverse)
        inputs = BaselineFactorPoolFreezeInputs(
            **{
                **{
                    field: getattr(self.fixture.inputs, field)
                    for field in self.fixture.inputs.__dataclass_fields__
                },
                "enriched_selection_manifest_path": path,
            }
        )
        second = freeze_baseline_factor_pool(
            inputs,
            self.fixture.runs_root,
            confirmed_for_freeze=True,
            reviewed_selection_fingerprint=fingerprint,
        )
        self.assertNotEqual(
            first.baseline_factor_pool_fingerprint,
            second.baseline_factor_pool_fingerprint,
        )
        self.assertTrue(first.manifest_path.is_file())
        self.assertTrue(second.manifest_path.is_file())

    def test_pool_manifest_or_jsonl_tamper_fails_loader(self):
        artifact = self.fixture.freeze()
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        manifest["pool"]["ordered_train_directions"][0] = -1
        _write_json(artifact.manifest_path, manifest)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            load_verified_baseline_factor_pool(artifact.manifest_path)

    def test_fake_hash_without_artifact_is_insufficient(self):
        fake = (
            self.fixture.runs_root
            / "baseline_factor_pools"
            / ("a" * 64)
            / BASELINE_FACTOR_POOL_MANIFEST_FILENAME
        )
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            load_verified_baseline_factor_pool(fake)

    def test_upstream_selection_tamper_or_missing_fails_loader(self):
        artifact = self.fixture.freeze()
        self.fixture.selection_path.unlink()
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            load_verified_baseline_factor_pool(artifact.manifest_path)

    def test_upstream_scope_change_fails_loader(self):
        artifact = self.fixture.freeze()
        selection = json.loads(self.fixture.selection_path.read_text(encoding="utf-8"))
        selection["scope"] = "resource_limited_provisional_selection"
        _write_json(self.fixture.selection_path, selection)
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            load_verified_baseline_factor_pool(artifact.manifest_path)

    def test_conflicting_existing_content_is_not_repaired(self):
        artifact = self.fixture.freeze()
        artifact.records_path.write_bytes(artifact.records_path.read_bytes() + b"{}\n")
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()

    def test_read_only_evaluation_store_does_not_create_missing_manifest(self):
        entry = json.loads(self.fixture.train_entry_path.read_text(encoding="utf-8"))
        store_manifest = Path(entry["database_path"]).parent / "store_manifest.json"
        store_manifest.unlink()
        with self.assertRaises(BaselineFactorPoolIntegrityError):
            self.fixture.freeze()
        self.assertFalse(store_manifest.exists())


if __name__ == "__main__":
    unittest.main()
