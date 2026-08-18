"""Crash-safe Stage 6 evaluation cache and sequential resumable runner.

The same immutable cache/run ledger is used by bounded smoke runs and by the
manually started full-registry Stage 6 runner.  It never exposes OOS,
screening, ranking, or decorrelation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import psutil

from factor_gfn.grammar import get_action

from .stage6_evaluation import (
    Stage6CandidateEvaluationResult,
    Stage6CandidateEvaluator,
    _stable_hash,
)


EVALUATION_STORE_SCHEMA = "factor_gfn.stage6_evaluation_store.v1"
EVALUATION_RUN_SCHEMA = "factor_gfn.stage6_evaluation_run.v1"
EVALUATION_CACHE_KEY_SCHEMA = "factor_gfn.stage6_evaluation_cache_key.v1"
EVALUATION_RUNNER_VERSION = "stage6-bounded-sequential-runner-v1"

_COMPLETED_STATES = frozenset({"completed", "completed_invalid"})
_DETERMINISTIC_RESULT_KEYS = (
    "schema",
    "status",
    "invalid_reasons",
    "expression",
    "context_fingerprint",
    "evaluation_contract_fingerprint",
    "train_direction",
    "train",
    "validation",
    "factor_finite_coverage",
)


class EvaluationStoreIntegrityError(RuntimeError):
    """Stored state failed identity, JSON, or fingerprint verification."""


class DeterminismConflictError(RuntimeError):
    """A recomputation disagreed with an immutable cached result."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_json(text: str, label: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise EvaluationStoreIntegrityError(f"invalid {label} JSON") from error


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(_json_text(value) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class EvaluationCacheIdentity:
    cache_key: str
    structural_hash: str
    expression: Mapping[str, Any]
    context_fingerprint: str
    evaluation_contract_fingerprint: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_CACHE_KEY_SCHEMA,
            "current_structural_hash": self.structural_hash,
            "context_fingerprint": self.context_fingerprint,
            "evaluation_contract_fingerprint": self.evaluation_contract_fingerprint,
        }


def evaluation_cache_identity(
    evaluator: Stage6CandidateEvaluator,
    candidate: Mapping[str, Any],
) -> EvaluationCacheIdentity:
    expression = dict(evaluator.resolve_candidate_identity(candidate))
    payload = {
        "schema": EVALUATION_CACHE_KEY_SCHEMA,
        "current_structural_hash": expression["structural_hash"],
        "context_fingerprint": evaluator.context.fingerprint,
        "evaluation_contract_fingerprint": evaluator.evaluation_contract_fingerprint,
    }
    return EvaluationCacheIdentity(
        cache_key=_stable_hash(payload),
        structural_hash=str(expression["structural_hash"]),
        expression=MappingProxyType(expression),
        context_fingerprint=evaluator.context.fingerprint,
        evaluation_contract_fingerprint=evaluator.evaluation_contract_fingerprint,
    )


def _validate_result_record(
    record: Mapping[str, Any], identity: EvaluationCacheIdentity
) -> dict[str, Any]:
    try:
        deterministic = {key: record[key] for key in _DETERMINISTIC_RESULT_KEYS}
        result_fingerprint = str(record["result_fingerprint"])
    except KeyError as error:
        raise EvaluationStoreIntegrityError(
            f"cached result is missing field {error.args[0]!r}"
        ) from error
    if record.get("status") not in _COMPLETED_STATES:
        raise EvaluationStoreIntegrityError("cached result has a non-cacheable status")
    if dict(record.get("expression", {})) != dict(identity.expression):
        raise EvaluationStoreIntegrityError("cached expression identity mismatch")
    if record.get("context_fingerprint") != identity.context_fingerprint:
        raise EvaluationStoreIntegrityError("cached context fingerprint mismatch")
    if (
        record.get("evaluation_contract_fingerprint")
        != identity.evaluation_contract_fingerprint
    ):
        raise EvaluationStoreIntegrityError("cached evaluation contract mismatch")
    if _stable_hash(deterministic) != result_fingerprint:
        raise EvaluationStoreIntegrityError("cached result fingerprint mismatch")
    for timing in (
        "factor_seconds",
        "train_evaluation_seconds",
        "validation_evaluation_seconds",
        "total_seconds",
    ):
        value = record.get(timing)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise EvaluationStoreIntegrityError(f"cached {timing} is invalid")
    return dict(record)


@dataclass(frozen=True, slots=True)
class RssMeasurement:
    rss_before_bytes: int
    peak_rss_bytes: int
    rss_after_bytes: int
    peak_delta_bytes: int
    sampling_interval_seconds: float


class ProcessRssSampler:
    """Sample process RSS so NumPy/Numba native allocations are included."""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("RSS sampling interval must be a finite positive value")
        self.interval_seconds = float(interval_seconds)
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._before = 0
        self._peak = 0

    def __enter__(self) -> "ProcessRssSampler":
        self._before = int(self._process.memory_info().rss)
        self._peak = self._before
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._peak = max(self._peak, int(self._process.memory_info().rss))

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 5.0))
        self._peak = max(self._peak, int(self._process.memory_info().rss))

    def measurement(self) -> RssMeasurement:
        if self._thread is None:
            raise RuntimeError("RSS sampler has not been started")
        after = int(self._process.memory_info().rss)
        peak = max(self._peak, after)
        return RssMeasurement(
            rss_before_bytes=self._before,
            peak_rss_bytes=peak,
            rss_after_bytes=after,
            peak_delta_bytes=max(0, peak - self._before),
            sampling_interval_seconds=self.interval_seconds,
        )


@dataclass(frozen=True, slots=True)
class FrozenEvaluationRun:
    run_id: str
    manifest: Mapping[str, Any]
    candidate_count: int


@dataclass(frozen=True, slots=True)
class VerifiedEvaluationRun:
    """A complete frozen run whose immutable cached results were revalidated."""

    run_id: str
    manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    ordered_result_set_fingerprint: str


@dataclass(frozen=True, slots=True)
class VerifiedPartialEvaluationRun:
    """Immutable completed subset of a possibly interrupted frozen run."""

    run_id: str
    manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    completed_count: int
    candidate_count: int
    ordered_result_set_fingerprint: str


@dataclass(frozen=True, slots=True)
class RunnerInvocationSummary:
    invocation_id: str
    run_id: str
    run_status: str
    execution_limit: int | None
    recovered_interrupted: int
    resume_skipped: int
    cache_hits: int
    newly_evaluated: int
    newly_completed: int
    newly_completed_invalid: int
    newly_failed: int
    pending_after: int
    failed_after: int
    elapsed_seconds: float
    rss_before_bytes: int
    peak_rss_bytes: int
    rss_after_bytes: int


class EvaluationStore:
    """SQLite-backed immutable result cache and resumable run ledger."""

    def __init__(
        self,
        database_path: str | Path,
        run_artifact_root: str | Path,
        *,
        read_only: bool = False,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.run_artifact_root = Path(run_artifact_root).resolve()
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.database_path.is_file():
                raise EvaluationStoreIntegrityError(
                    f"read-only evaluation store is missing: {self.database_path}"
                )
            if not self.run_artifact_root.is_dir():
                raise EvaluationStoreIntegrityError(
                    f"read-only run artifact root is missing: {self.run_artifact_root}"
                )
            database_uri = f"{self.database_path.as_uri()}?mode=ro"
            self.connection = sqlite3.connect(
                database_uri,
                uri=True,
                timeout=30.0,
            )
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA query_only=ON")
            self.connection.execute("PRAGMA busy_timeout=30000")
            try:
                self._verify_existing_store()
            except BaseException:
                self.connection.close()
                raise
            return
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_artifact_root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._initialize_schema()
        self._write_store_manifest()

    def _verify_existing_store(self) -> None:
        try:
            row = self.connection.execute(
                "SELECT value FROM store_meta WHERE key='schema'"
            ).fetchone()
        except sqlite3.DatabaseError as error:
            raise EvaluationStoreIntegrityError(
                "read-only evaluation store schema cannot be verified"
            ) from error
        if row is None or row["value"] != EVALUATION_STORE_SCHEMA:
            observed = None if row is None else row["value"]
            raise EvaluationStoreIntegrityError(
                f"unsupported evaluation store schema: {observed!r}"
            )
        stable = {
            "schema": EVALUATION_STORE_SCHEMA,
            "runner_version": EVALUATION_RUNNER_VERSION,
            "sqlite_filename": self.database_path.name,
            "journal_mode": "WAL",
            "synchronous": "FULL",
            "single_writer": True,
            "cacheable_statuses": sorted(_COMPLETED_STATES),
            "failed_results_cacheable": False,
            "result_conflict_policy": "fail_closed_preserve_existing",
        }
        expected = {
            **stable,
            "store_manifest_fingerprint": _stable_hash(stable),
        }
        path = self.database_path.parent / "store_manifest.json"
        if not path.is_file():
            raise EvaluationStoreIntegrityError("read-only store manifest is missing")
        observed = _parse_json(path.read_text(encoding="utf-8"), "store manifest")
        if observed != expected:
            raise EvaluationStoreIntegrityError("read-only store manifest changed")

    def __enter__(self) -> "EvaluationStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                cache_key TEXT PRIMARY KEY,
                structural_hash TEXT NOT NULL,
                expression_json TEXT NOT NULL,
                context_fingerprint TEXT NOT NULL,
                evaluation_contract_fingerprint TEXT NOT NULL,
                result_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('completed','completed_invalid')),
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                manifest_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS run_candidates (
                run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                structural_hash TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN (
                    'pending','running','completed','completed_invalid','failed'
                )),
                resolution TEXT,
                result_fingerprint TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error_type TEXT,
                last_error_message TEXT,
                PRIMARY KEY(run_id, ordinal),
                UNIQUE(run_id, structural_hash),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                cache_key TEXT NOT NULL,
                mode TEXT NOT NULL,
                outcome TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                error_type TEXT,
                error_message TEXT,
                metrics_json TEXT,
                FOREIGN KEY(run_id, ordinal) REFERENCES run_candidates(run_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS run_events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                ordinal INTEGER,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                details_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS invocations (
                invocation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS determinism_conflicts (
                conflict_id TEXT PRIMARY KEY,
                run_id TEXT,
                ordinal INTEGER,
                cache_key TEXT NOT NULL,
                stored_result_fingerprint TEXT NOT NULL,
                observed_result_fingerprint TEXT NOT NULL,
                observed_result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        row = self.connection.execute(
            "SELECT value FROM store_meta WHERE key='schema'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO store_meta(key,value) VALUES('schema',?)",
                (EVALUATION_STORE_SCHEMA,),
            )
            self.connection.commit()
        elif row["value"] != EVALUATION_STORE_SCHEMA:
            raise EvaluationStoreIntegrityError(
                f"unsupported evaluation store schema: {row['value']!r}"
            )

    def _write_store_manifest(self) -> None:
        stable = {
            "schema": EVALUATION_STORE_SCHEMA,
            "runner_version": EVALUATION_RUNNER_VERSION,
            "sqlite_filename": self.database_path.name,
            "journal_mode": "WAL",
            "synchronous": "FULL",
            "single_writer": True,
            "cacheable_statuses": sorted(_COMPLETED_STATES),
            "failed_results_cacheable": False,
            "result_conflict_policy": "fail_closed_preserve_existing",
        }
        manifest = {
            **stable,
            "store_manifest_fingerprint": _stable_hash(stable),
        }
        path = self.database_path.parent / "store_manifest.json"
        if path.is_file():
            existing = _parse_json(path.read_text(encoding="utf-8"), "store manifest")
            if existing != manifest:
                raise EvaluationStoreIntegrityError("store manifest changed")
        else:
            _atomic_write_json(path, manifest)

    def create_run(
        self,
        candidates: Sequence[Mapping[str, Any]],
        evaluator: Stage6CandidateEvaluator,
        *,
        scope: str = "bounded_real_smoke_not_full_registry",
    ) -> FrozenEvaluationRun:
        if not candidates:
            raise ValueError("a frozen evaluation run requires at least one candidate")
        if evaluator.compatibility_audit_fingerprint is None:
            raise ValueError("run requires compatibility_audit_fingerprint")
        if evaluator.accepted_registry_fingerprint is None:
            raise ValueError("run requires accepted_registry_fingerprint")
        identities = [evaluation_cache_identity(evaluator, candidate) for candidate in candidates]
        hashes = [identity.structural_hash for identity in identities]
        if len(set(hashes)) != len(hashes):
            raise ValueError("frozen evaluation run candidates must have unique hashes")
        ordered_identity = [dict(identity.expression) for identity in identities]
        if scope not in {
            "bounded_real_smoke_not_full_registry",
            "full_accepted_registry_train_validation_evaluation",
            "train_prefilter_then_validation_full_registry",
            "train_preparation_full_accepted_registry",
            "train_preparation_resource_limited_eligible_universe",
            "validation_from_frozen_train_pass_manifest",
        }:
            raise ValueError(f"unsupported evaluation run scope: {scope!r}")
        stable_manifest = {
            "schema": EVALUATION_RUN_SCHEMA,
            "runner_version": EVALUATION_RUNNER_VERSION,
            "store_schema": EVALUATION_STORE_SCHEMA,
            "compatibility_audit_fingerprint": evaluator.compatibility_audit_fingerprint,
            "accepted_registry_fingerprint": evaluator.accepted_registry_fingerprint,
            "context_fingerprint": evaluator.context.fingerprint,
            "evaluation_contract_fingerprint": evaluator.evaluation_contract_fingerprint,
            "candidate_count": len(candidates),
            "ordered_candidate_hashes": hashes,
            "ordered_candidate_list_digest": _stable_hash(ordered_identity),
            "ordered_cache_keys_digest": _stable_hash(
                [identity.cache_key for identity in identities]
            ),
            "oos": "not_loaded",
            "scope": scope,
        }
        run_id = _stable_hash(stable_manifest)
        manifest = {**stable_manifest, "run_id": run_id}
        manifest_json = _json_text(manifest)
        now = _now_utc()
        with self.connection:
            existing = self.connection.execute(
                "SELECT manifest_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing is None:
                self.connection.execute(
                    "INSERT INTO runs(run_id,manifest_json,status,created_at,updated_at) "
                    "VALUES(?,?,?,?,?)",
                    (run_id, manifest_json, "created", now, now),
                )
                for ordinal, (candidate, identity) in enumerate(
                    zip(candidates, identities, strict=True)
                ):
                    self.connection.execute(
                        "INSERT INTO run_candidates("
                        "run_id,ordinal,structural_hash,cache_key,candidate_json,state"
                        ") VALUES(?,?,?,?,?,?)",
                        (
                            run_id,
                            ordinal,
                            identity.structural_hash,
                            identity.cache_key,
                            _json_text(dict(candidate)),
                            "pending",
                        ),
                    )
            elif existing["manifest_json"] != manifest_json:
                raise EvaluationStoreIntegrityError("run ID manifest collision")
        manifest_path = self.run_artifact_root / run_id / "run_manifest.json"
        if manifest_path.is_file():
            existing_manifest = _parse_json(
                manifest_path.read_text(encoding="utf-8"), "run manifest"
            )
            if existing_manifest != manifest:
                raise EvaluationStoreIntegrityError("materialized run manifest changed")
        else:
            _atomic_write_json(manifest_path, manifest)
        return FrozenEvaluationRun(
            run_id=run_id,
            manifest=MappingProxyType(manifest),
            candidate_count=len(candidates),
        )

    def get_run_manifest(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT manifest_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown evaluation run: {run_id}")
        manifest = _parse_json(row["manifest_json"], "run manifest")
        if manifest.get("run_id") != run_id:
            raise EvaluationStoreIntegrityError("stored run ID mismatch")
        stable = {key: value for key, value in manifest.items() if key != "run_id"}
        if _stable_hash(stable) != run_id:
            raise EvaluationStoreIntegrityError("stored run manifest fingerprint mismatch")
        return manifest

    def validate_run_evaluator(
        self, run_id: str, evaluator: Stage6CandidateEvaluator
    ) -> dict[str, Any]:
        manifest = self.get_run_manifest(run_id)
        expected = {
            "compatibility_audit_fingerprint": evaluator.compatibility_audit_fingerprint,
            "accepted_registry_fingerprint": evaluator.accepted_registry_fingerprint,
            "context_fingerprint": evaluator.context.fingerprint,
            "evaluation_contract_fingerprint": evaluator.evaluation_contract_fingerprint,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise EvaluationStoreIntegrityError(
                f"run cannot resume under a different evaluator identity: {mismatches}"
            )
        return manifest

    def run_candidates(self, run_id: str) -> list[dict[str, Any]]:
        manifest = self.get_run_manifest(run_id)
        rows = self.connection.execute(
            "SELECT ordinal,structural_hash,cache_key,candidate_json,state,resolution,"
            "result_fingerprint,attempt_count,last_error_type,last_error_message "
            "FROM run_candidates WHERE run_id=? ORDER BY ordinal",
            (run_id,),
        ).fetchall()
        if len(rows) != manifest["candidate_count"]:
            raise EvaluationStoreIntegrityError("frozen run candidate count changed")
        result: list[dict[str, Any]] = []
        identities: list[dict[str, Any]] = []
        hashes: list[str] = []
        keys: list[str] = []
        for row in rows:
            candidate = _parse_json(row["candidate_json"], "candidate")
            identity = {
                "structural_hash": candidate.get("current_structural_hash"),
                "formula": candidate.get("formula"),
                "prefix_token_ids": candidate.get("prefix_token_ids"),
                "node_count": candidate.get("node_count"),
                "depth": candidate.get("depth"),
            }
            identities.append(identity)
            hashes.append(str(row["structural_hash"]))
            keys.append(str(row["cache_key"]))
            result.append({**dict(row), "candidate": candidate})
        if hashes != manifest["ordered_candidate_hashes"]:
            raise EvaluationStoreIntegrityError("frozen candidate order changed")
        if _stable_hash(identities) != manifest["ordered_candidate_list_digest"]:
            raise EvaluationStoreIntegrityError("frozen candidate identity digest changed")
        if _stable_hash(keys) != manifest["ordered_cache_keys_digest"]:
            raise EvaluationStoreIntegrityError("frozen cache-key digest changed")
        return result

    def lookup_verified(self, identity: EvaluationCacheIdentity) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM evaluations WHERE cache_key=?", (identity.cache_key,)
        ).fetchone()
        if row is None:
            return None
        if row["structural_hash"] != identity.structural_hash:
            raise EvaluationStoreIntegrityError("cached structural hash mismatch")
        if _parse_json(row["expression_json"], "cached expression") != dict(
            identity.expression
        ):
            raise EvaluationStoreIntegrityError("cached expression payload mismatch")
        if row["context_fingerprint"] != identity.context_fingerprint:
            raise EvaluationStoreIntegrityError("cached context column mismatch")
        if (
            row["evaluation_contract_fingerprint"]
            != identity.evaluation_contract_fingerprint
        ):
            raise EvaluationStoreIntegrityError("cached contract column mismatch")
        record = _parse_json(row["result_json"], "cached result")
        validated = _validate_result_record(record, identity)
        if validated["result_fingerprint"] != row["result_fingerprint"]:
            raise EvaluationStoreIntegrityError("cached result column mismatch")
        if validated["status"] != row["status"]:
            raise EvaluationStoreIntegrityError("cached status column mismatch")
        return validated

    def load_verified_run_results(self, run_id: str) -> VerifiedEvaluationRun:
        """Load one complete run without reconstructing or evaluating expressions."""

        manifest = self.get_run_manifest(run_id)
        run = self.connection.execute(
            "SELECT status FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(f"unknown evaluation run: {run_id}")
        if run["status"] != "complete":
            raise EvaluationStoreIntegrityError(
                f"selection requires a complete evaluation run, got {run['status']!r}"
            )
        if manifest.get("oos") != "not_loaded":
            raise EvaluationStoreIntegrityError(
                "selection input must be a Train/Validation-only evaluation run"
            )
        conflict_count = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM determinism_conflicts AS dc "
                "JOIN run_candidates AS rc ON rc.cache_key=dc.cache_key "
                "WHERE rc.run_id=?",
                (run_id,),
            ).fetchone()[0]
        )
        if conflict_count:
            raise EvaluationStoreIntegrityError(
                "selection input has unresolved determinism conflicts"
            )

        verified: list[Mapping[str, Any]] = []
        seen_hashes: set[str] = set()
        result_identity: list[dict[str, Any]] = []
        for item in self.run_candidates(run_id):
            if item["state"] not in _COMPLETED_STATES:
                raise EvaluationStoreIntegrityError(
                    "selection input contains an unfinished candidate"
                )
            candidate = item["candidate"]
            expression = {
                "structural_hash": candidate.get("current_structural_hash"),
                "formula": candidate.get("formula"),
                "prefix_token_ids": candidate.get("prefix_token_ids"),
                "node_count": candidate.get("node_count"),
                "depth": candidate.get("depth"),
            }
            structural_hash = str(item["structural_hash"])
            if expression["structural_hash"] != structural_hash:
                raise EvaluationStoreIntegrityError(
                    "run candidate structural hash identity mismatch"
                )
            if structural_hash in seen_hashes:
                raise EvaluationStoreIntegrityError(
                    "selection input contains duplicate structural hashes"
                )
            seen_hashes.add(structural_hash)
            identity = EvaluationCacheIdentity(
                cache_key=str(item["cache_key"]),
                structural_hash=structural_hash,
                expression=MappingProxyType(expression),
                context_fingerprint=str(manifest["context_fingerprint"]),
                evaluation_contract_fingerprint=str(
                    manifest["evaluation_contract_fingerprint"]
                ),
            )
            if _stable_hash(identity.payload()) != identity.cache_key:
                raise EvaluationStoreIntegrityError("run item cache key is invalid")
            record = self.lookup_verified(identity)
            if record is None:
                raise EvaluationStoreIntegrityError(
                    "completed run candidate lacks an immutable cached result"
                )
            if record["result_fingerprint"] != item["result_fingerprint"]:
                raise EvaluationStoreIntegrityError(
                    "run candidate result fingerprint differs from cache"
                )
            if record["status"] != item["state"]:
                raise EvaluationStoreIntegrityError(
                    "run candidate state differs from cached result status"
                )
            payload = MappingProxyType(
                {
                    "ordinal": int(item["ordinal"]),
                    "structural_hash": structural_hash,
                    "cache_key": identity.cache_key,
                    "result_fingerprint": record["result_fingerprint"],
                    "result": record,
                }
            )
            verified.append(payload)
            result_identity.append(
                {
                    "ordinal": int(item["ordinal"]),
                    "structural_hash": structural_hash,
                    "cache_key": identity.cache_key,
                    "result_fingerprint": record["result_fingerprint"],
                }
            )
        return VerifiedEvaluationRun(
            run_id=run_id,
            manifest=MappingProxyType(manifest),
            records=tuple(verified),
            ordered_result_set_fingerprint=_stable_hash(result_identity),
        )

    def load_verified_completed_results(
        self, run_id: str
    ) -> VerifiedPartialEvaluationRun:
        """Validate and load only completed rows without requiring run completion."""

        manifest = self.get_run_manifest(run_id)
        if manifest.get("oos") != "not_loaded":
            raise EvaluationStoreIntegrityError(
                "partial seed input must be a Train/Validation-only evaluation run"
            )
        conflict_count = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM determinism_conflicts AS dc "
                "JOIN run_candidates AS rc ON rc.cache_key=dc.cache_key "
                "WHERE rc.run_id=?",
                (run_id,),
            ).fetchone()[0]
        )
        if conflict_count:
            raise EvaluationStoreIntegrityError(
                "partial seed input has unresolved determinism conflicts"
            )
        verified: list[Mapping[str, Any]] = []
        result_identity: list[dict[str, Any]] = []
        for item in self.run_candidates(run_id):
            if item["state"] not in _COMPLETED_STATES:
                continue
            candidate = item["candidate"]
            structural_hash = str(item["structural_hash"])
            expression = {
                "structural_hash": candidate.get("current_structural_hash"),
                "formula": candidate.get("formula"),
                "prefix_token_ids": candidate.get("prefix_token_ids"),
                "node_count": candidate.get("node_count"),
                "depth": candidate.get("depth"),
            }
            if expression["structural_hash"] != structural_hash:
                raise EvaluationStoreIntegrityError(
                    "partial seed candidate structural hash mismatch"
                )
            identity = EvaluationCacheIdentity(
                cache_key=str(item["cache_key"]),
                structural_hash=structural_hash,
                expression=MappingProxyType(expression),
                context_fingerprint=str(manifest["context_fingerprint"]),
                evaluation_contract_fingerprint=str(
                    manifest["evaluation_contract_fingerprint"]
                ),
            )
            if _stable_hash(identity.payload()) != identity.cache_key:
                raise EvaluationStoreIntegrityError("partial seed cache key is invalid")
            record = self.lookup_verified(identity)
            if record is None or record["result_fingerprint"] != item["result_fingerprint"]:
                raise EvaluationStoreIntegrityError(
                    "partial seed row lacks its immutable cached result"
                )
            payload = MappingProxyType(
                {
                    "ordinal": int(item["ordinal"]),
                    "structural_hash": structural_hash,
                    "cache_key": identity.cache_key,
                    "result_fingerprint": record["result_fingerprint"],
                    "result": record,
                }
            )
            verified.append(payload)
            result_identity.append(
                {
                    "ordinal": int(item["ordinal"]),
                    "structural_hash": structural_hash,
                    "cache_key": identity.cache_key,
                    "result_fingerprint": record["result_fingerprint"],
                }
            )
        return VerifiedPartialEvaluationRun(
            run_id=run_id,
            manifest=MappingProxyType(manifest),
            records=tuple(verified),
            completed_count=len(verified),
            candidate_count=int(manifest["candidate_count"]),
            ordered_result_set_fingerprint=_stable_hash(result_identity),
        )

    def recover_interrupted(self, run_id: str) -> int:
        rows = self.connection.execute(
            "SELECT ordinal FROM run_candidates WHERE run_id=? AND state='running'",
            (run_id,),
        ).fetchall()
        if not rows:
            return 0
        now = _now_utc()
        with self.connection:
            self.connection.execute(
                "UPDATE attempts SET outcome='interrupted',finished_at=? "
                "WHERE run_id=? AND outcome='running'",
                (now, run_id),
            )
            self.connection.execute(
                "UPDATE run_candidates SET state='pending',resolution='interrupted_requeued' "
                "WHERE run_id=? AND state='running'",
                (run_id,),
            )
        return len(rows)

    def record_event(
        self,
        run_id: str,
        event_type: str,
        *,
        ordinal: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO run_events(event_id,run_id,ordinal,event_type,occurred_at,details_json) "
                "VALUES(?,?,?,?,?,?)",
                (
                    uuid.uuid4().hex,
                    run_id,
                    ordinal,
                    event_type,
                    _now_utc(),
                    _json_text(dict(details or {})),
                ),
            )

    def begin_attempt(
        self,
        run_id: str,
        ordinal: int,
        cache_key: str,
        *,
        mode: str = "evaluation",
        update_item_state: bool = True,
    ) -> tuple[str, str]:
        attempt_id = uuid.uuid4().hex
        started_at = _now_utc()
        with self.connection:
            if update_item_state:
                cursor = self.connection.execute(
                    "UPDATE run_candidates SET state='running',resolution=NULL,"
                    "attempt_count=attempt_count+1,last_error_type=NULL,last_error_message=NULL "
                    "WHERE run_id=? AND ordinal=? AND state IN ('pending','failed')",
                    (run_id, ordinal),
                )
                if cursor.rowcount != 1:
                    raise EvaluationStoreIntegrityError(
                        "candidate is not eligible to begin an evaluation attempt"
                    )
            self.connection.execute(
                "INSERT INTO attempts(attempt_id,run_id,ordinal,cache_key,mode,outcome,started_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (attempt_id, run_id, ordinal, cache_key, mode, "running", started_at),
            )
        return attempt_id, started_at

    def _insert_result_or_conflict(
        self,
        identity: EvaluationCacheIdentity,
        result: Stage6CandidateEvaluationResult,
        *,
        run_id: str,
        ordinal: int,
    ) -> bool:
        record = result.to_dict()
        _validate_result_record(record, identity)
        existing = self.connection.execute(
            "SELECT result_fingerprint FROM evaluations WHERE cache_key=?",
            (identity.cache_key,),
        ).fetchone()
        if existing is None:
            self.connection.execute(
                "INSERT INTO evaluations(cache_key,structural_hash,expression_json,"
                "context_fingerprint,evaluation_contract_fingerprint,result_fingerprint,"
                "status,result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    identity.cache_key,
                    identity.structural_hash,
                    _json_text(dict(identity.expression)),
                    identity.context_fingerprint,
                    identity.evaluation_contract_fingerprint,
                    result.result_fingerprint,
                    result.status,
                    _json_text(record),
                    _now_utc(),
                ),
            )
            return False
        if existing["result_fingerprint"] == result.result_fingerprint:
            return False
        self.connection.execute(
            "INSERT INTO determinism_conflicts(conflict_id,run_id,ordinal,cache_key,"
            "stored_result_fingerprint,observed_result_fingerprint,observed_result_json,created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                uuid.uuid4().hex,
                run_id,
                ordinal,
                identity.cache_key,
                existing["result_fingerprint"],
                result.result_fingerprint,
                _json_text(record),
                _now_utc(),
            ),
        )
        return True

    def finish_success(
        self,
        *,
        attempt_id: str,
        run_id: str,
        ordinal: int,
        identity: EvaluationCacheIdentity,
        result: Stage6CandidateEvaluationResult,
        rss: RssMeasurement,
    ) -> None:
        finished = _now_utc()
        metrics = {
            "result_timings": {
                "factor_seconds": result.factor_seconds,
                "train_evaluation_seconds": result.train_evaluation_seconds,
                "validation_evaluation_seconds": result.validation_evaluation_seconds,
                "total_seconds": result.total_seconds,
            },
            "rss": asdict(rss),
        }
        with self.connection:
            conflict = self._insert_result_or_conflict(
                identity, result, run_id=run_id, ordinal=ordinal
            )
            if conflict:
                self.connection.execute(
                    "UPDATE attempts SET outcome='determinism_conflict',finished_at=?,"
                    "metrics_json=? WHERE attempt_id=? AND outcome='running'",
                    (finished, _json_text(metrics), attempt_id),
                )
                self.connection.execute(
                    "UPDATE run_candidates SET state='failed',resolution='determinism_conflict',"
                    "last_error_type='DeterminismConflictError',"
                    "last_error_message='recomputation disagreed with immutable cache' "
                    "WHERE run_id=? AND ordinal=?",
                    (run_id, ordinal),
                )
            else:
                self.connection.execute(
                    "UPDATE attempts SET outcome=?,finished_at=?,metrics_json=? "
                    "WHERE attempt_id=? AND outcome='running'",
                    (result.status, finished, _json_text(metrics), attempt_id),
                )
                self.connection.execute(
                    "UPDATE run_candidates SET state=?,resolution='evaluated',"
                    "result_fingerprint=? WHERE run_id=? AND ordinal=?",
                    (result.status, result.result_fingerprint, run_id, ordinal),
                )
        if conflict:
            raise DeterminismConflictError(
                f"cache key {identity.cache_key} produced a different result fingerprint"
            )

    def finish_failure(
        self,
        *,
        attempt_id: str,
        run_id: str,
        ordinal: int,
        error: BaseException,
        rss: RssMeasurement,
    ) -> None:
        error_type = type(error).__name__
        message = str(error)
        with self.connection:
            self.connection.execute(
                "UPDATE attempts SET outcome='failed',finished_at=?,error_type=?,"
                "error_message=?,metrics_json=? WHERE attempt_id=? AND outcome='running'",
                (
                    _now_utc(),
                    error_type,
                    message,
                    _json_text({"rss": asdict(rss)}),
                    attempt_id,
                ),
            )
            self.connection.execute(
                "UPDATE run_candidates SET state='failed',resolution='evaluation_failed',"
                "last_error_type=?,last_error_message=? WHERE run_id=? AND ordinal=?",
                (error_type, message, run_id, ordinal),
            )

    def resolve_cache_hit(
        self,
        run_id: str,
        ordinal: int,
        record: Mapping[str, Any],
    ) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE run_candidates SET state=?,resolution='cache_hit',result_fingerprint=? "
                "WHERE run_id=? AND ordinal=? AND state='pending'",
                (record["status"], record["result_fingerprint"], run_id, ordinal),
            )

    def record_determinism_result(
        self,
        *,
        attempt_id: str,
        run_id: str,
        ordinal: int,
        identity: EvaluationCacheIdentity,
        cached: Mapping[str, Any],
        observed: Stage6CandidateEvaluationResult,
        rss: RssMeasurement,
    ) -> None:
        conflict = cached["result_fingerprint"] != observed.result_fingerprint
        metrics = {
            "rss": asdict(rss),
            "observed_result_timings": {
                "factor_seconds": observed.factor_seconds,
                "train_evaluation_seconds": observed.train_evaluation_seconds,
                "validation_evaluation_seconds": observed.validation_evaluation_seconds,
                "total_seconds": observed.total_seconds,
            },
        }
        with self.connection:
            if conflict:
                self.connection.execute(
                    "INSERT INTO determinism_conflicts(conflict_id,run_id,ordinal,cache_key,"
                    "stored_result_fingerprint,observed_result_fingerprint,observed_result_json,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        run_id,
                        ordinal,
                        identity.cache_key,
                        cached["result_fingerprint"],
                        observed.result_fingerprint,
                        _json_text(observed.to_dict()),
                        _now_utc(),
                    ),
                )
                self.connection.execute(
                    "UPDATE runs SET status='incomplete',updated_at=? WHERE run_id=?",
                    (_now_utc(), run_id),
                )
            self.connection.execute(
                "UPDATE attempts SET outcome=?,finished_at=?,metrics_json=? "
                "WHERE attempt_id=? AND outcome='running'",
                (
                    "determinism_conflict" if conflict else "determinism_verified",
                    _now_utc(),
                    _json_text(metrics),
                    attempt_id,
                ),
            )
        if conflict:
            raise DeterminismConflictError(
                f"determinism verification failed for {identity.cache_key}"
            )

    def set_run_status(self, run_id: str, status: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE runs SET status=?,updated_at=? WHERE run_id=?",
                (status, _now_utc(), run_id),
            )

    def save_invocation(self, summary: RunnerInvocationSummary, started_at: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO invocations(invocation_id,run_id,started_at,finished_at,summary_json) "
                "VALUES(?,?,?,?,?)",
                (
                    summary.invocation_id,
                    summary.run_id,
                    started_at,
                    _now_utc(),
                    _json_text(asdict(summary)),
                ),
            )
        self._write_run_status(summary.run_id)

    def _write_run_status(self, run_id: str) -> None:
        run = self.connection.execute(
            "SELECT status,created_at,updated_at FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        counts = {
            row["state"]: int(row["count"])
            for row in self.connection.execute(
                "SELECT state,COUNT(*) AS count FROM run_candidates WHERE run_id=? GROUP BY state",
                (run_id,),
            ).fetchall()
        }
        payload = {
            "schema": EVALUATION_RUN_SCHEMA,
            "run_id": run_id,
            "status": run["status"],
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "candidate_state_counts": counts,
            "sqlite_is_source_of_truth": True,
        }
        _atomic_write_json(
            self.run_artifact_root / run_id / "run_status.json", payload
        )

    def database_counts(self) -> dict[str, int]:
        tables = (
            "evaluations",
            "runs",
            "run_candidates",
            "attempts",
            "run_events",
            "invocations",
            "determinism_conflicts",
        )
        return {
            table: int(
                self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in tables
        }


class Stage6EvaluationRunner:
    """Execute or resume one already-frozen run in deterministic order."""

    def __init__(
        self,
        store: EvaluationStore,
        evaluator: Stage6CandidateEvaluator,
        *,
        rss_sampling_interval_seconds: float = 0.2,
    ) -> None:
        self.store = store
        self.evaluator = evaluator
        self.rss_sampling_interval_seconds = rss_sampling_interval_seconds

    def run(
        self,
        run_id: str,
        *,
        max_new_evaluations: int | None = None,
        retry_failed: bool = False,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> RunnerInvocationSummary:
        if max_new_evaluations is not None and max_new_evaluations < 1:
            raise ValueError("max_new_evaluations must be positive or None")
        self.store.validate_run_evaluator(run_id, self.evaluator)
        recovered = self.store.recover_interrupted(run_id)
        self.store.set_run_status(run_id, "running")
        invocation_id = uuid.uuid4().hex
        started_at = _now_utc()
        started = time.perf_counter()
        process = psutil.Process(os.getpid())
        rss_before = int(process.memory_info().rss)
        peak_rss = rss_before
        resume_skipped = 0
        cache_hits = 0
        newly_evaluated = 0
        newly_completed = 0
        newly_completed_invalid = 0
        newly_failed = 0
        attempted = 0
        reused_train_resolved = 0
        fresh_train_resolved = 0
        evaluation_path_counts: dict[str, int] = {}
        train_prefilter_status_counts: dict[str, int] = {}
        train_prefilter_status_by_path: dict[str, dict[str, int]] = {}
        validation_available = 0
        candidate_count = int(self.store.get_run_manifest(run_id)["candidate_count"])

        def notify(event_type: str, **details: Any) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "event_type": event_type,
                    "run_id": run_id,
                    "candidate_count": candidate_count,
                    "elapsed_seconds": time.perf_counter() - started,
                    "resume_skipped": resume_skipped,
                    "cache_hits": cache_hits,
                    "newly_evaluated": newly_evaluated,
                    "newly_completed": newly_completed,
                    "newly_completed_invalid": newly_completed_invalid,
                    "newly_failed": newly_failed,
                    "attempted": attempted,
                    "reused_train_resolved": reused_train_resolved,
                    "fresh_train_resolved": fresh_train_resolved,
                    "evaluation_path_counts": dict(sorted(evaluation_path_counts.items())),
                    "train_prefilter_status_counts": dict(
                        sorted(train_prefilter_status_counts.items())
                    ),
                    "train_prefilter_status_by_path": {
                        path: dict(sorted(counts.items()))
                        for path, counts in sorted(
                            train_prefilter_status_by_path.items()
                        )
                    },
                    "validation_available": validation_available,
                    "peak_rss_bytes": peak_rss,
                    **details,
                }
            )

        notify("invocation_started")

        for item in self.store.run_candidates(run_id):
            ordinal = int(item["ordinal"])
            candidate = item["candidate"]
            identity = evaluation_cache_identity(self.evaluator, candidate)
            if identity.cache_key != item["cache_key"]:
                raise EvaluationStoreIntegrityError("run item cache key changed")
            state = item["state"]
            reusable_train = False
            has_reusable_train = getattr(self.evaluator, "has_reusable_train", None)
            if callable(has_reusable_train):
                reusable_train = bool(has_reusable_train(candidate))
            planned_path = None
            planned_path_resolver = getattr(self.evaluator, "planned_path", None)
            if callable(planned_path_resolver):
                planned_path = str(planned_path_resolver(candidate))

            def record_path(record: Mapping[str, Any] | None) -> str:
                nonlocal validation_available
                source = record.get("source_identity", {}) if record else {}
                path = source.get("evaluation_path") if isinstance(source, Mapping) else None
                resolved_path = str(path or planned_path or "unspecified")
                evaluation_path_counts[resolved_path] = (
                    evaluation_path_counts.get(resolved_path, 0) + 1
                )
                if record is not None:
                    train = record.get("train", {})
                    prefilter = (
                        train.get("train_prefilter", {})
                        if isinstance(train, Mapping)
                        else {}
                    )
                    prefilter_status = (
                        prefilter.get("status")
                        if isinstance(prefilter, Mapping)
                        else None
                    )
                    if prefilter_status:
                        key = str(prefilter_status)
                        train_prefilter_status_counts[key] = (
                            train_prefilter_status_counts.get(key, 0) + 1
                        )
                        path_counts = train_prefilter_status_by_path.setdefault(
                            resolved_path, {}
                        )
                        path_counts[key] = path_counts.get(key, 0) + 1
                    validation = record.get("validation", {})
                    availability = (
                        validation.get("availability")
                        if isinstance(validation, Mapping)
                        else None
                    )
                    validation_not_evaluated = (
                        isinstance(availability, str)
                        and availability.startswith("not_evaluated_")
                    )
                    if isinstance(validation, Mapping) and not validation_not_evaluated:
                        validation_available += 1
                return resolved_path
            if state in _COMPLETED_STATES:
                cached = self.store.lookup_verified(identity)
                if cached is None or cached["result_fingerprint"] != item["result_fingerprint"]:
                    raise EvaluationStoreIntegrityError(
                        "resume candidate lacks its verified immutable cache result"
                    )
                resume_skipped += 1
                self.store.record_event(
                    run_id,
                    "resume_skipped",
                    ordinal=ordinal,
                    details={"result_fingerprint": cached["result_fingerprint"]},
                )
                if reusable_train:
                    reused_train_resolved += 1
                else:
                    fresh_train_resolved += 1
                evaluation_path = record_path(cached)
                notify(
                    "candidate_resolved",
                    ordinal=ordinal,
                    structural_hash=identity.structural_hash,
                    resolution="resume_skipped",
                    reusable_train=reusable_train,
                    evaluation_path=evaluation_path,
                )
                continue
            if state == "failed" and not retry_failed:
                continue
            cached = self.store.lookup_verified(identity)
            if cached is not None:
                self.store.resolve_cache_hit(run_id, ordinal, cached)
                cache_hits += 1
                self.store.record_event(
                    run_id,
                    "cache_hit",
                    ordinal=ordinal,
                    details={"result_fingerprint": cached["result_fingerprint"]},
                )
                if reusable_train:
                    reused_train_resolved += 1
                else:
                    fresh_train_resolved += 1
                evaluation_path = record_path(cached)
                notify(
                    "candidate_resolved",
                    ordinal=ordinal,
                    structural_hash=identity.structural_hash,
                    resolution="cache_hit",
                    reusable_train=reusable_train,
                    evaluation_path=evaluation_path,
                )
                continue
            if (
                max_new_evaluations is not None
                and attempted >= max_new_evaluations
            ):
                break
            attempted += 1
            notify(
                "candidate_started",
                ordinal=ordinal,
                structural_hash=identity.structural_hash,
                reusable_train=reusable_train,
                evaluation_path=planned_path,
            )
            attempt_id, _ = self.store.begin_attempt(
                run_id, ordinal, identity.cache_key
            )
            with ProcessRssSampler(self.rss_sampling_interval_seconds) as sampler:
                try:
                    result = self.evaluator.evaluate(candidate)
                except Exception as error:  # Persist the failure before continuing.
                    measurement = sampler.measurement()
                    self.store.finish_failure(
                        attempt_id=attempt_id,
                        run_id=run_id,
                        ordinal=ordinal,
                        error=error,
                        rss=measurement,
                    )
                    newly_failed += 1
                    peak_rss = max(peak_rss, measurement.peak_rss_bytes)
                    if reusable_train:
                        reused_train_resolved += 1
                    else:
                        fresh_train_resolved += 1
                    evaluation_path = record_path(None)
                    notify(
                        "candidate_resolved",
                        ordinal=ordinal,
                        structural_hash=identity.structural_hash,
                        resolution="failed",
                        reusable_train=reusable_train,
                        evaluation_path=evaluation_path,
                    )
                    continue
            measurement = sampler.measurement()
            try:
                self.store.finish_success(
                    attempt_id=attempt_id,
                    run_id=run_id,
                    ordinal=ordinal,
                    identity=identity,
                    result=result,
                    rss=measurement,
                )
            except DeterminismConflictError:
                newly_failed += 1
                peak_rss = max(peak_rss, measurement.peak_rss_bytes)
                if reusable_train:
                    reused_train_resolved += 1
                else:
                    fresh_train_resolved += 1
                evaluation_path = record_path(None)
                notify(
                    "candidate_resolved",
                    ordinal=ordinal,
                    structural_hash=identity.structural_hash,
                    resolution="determinism_conflict",
                    reusable_train=reusable_train,
                    evaluation_path=evaluation_path,
                )
                continue
            newly_evaluated += 1
            if result.status == "completed":
                newly_completed += 1
            else:
                newly_completed_invalid += 1
            peak_rss = max(peak_rss, measurement.peak_rss_bytes)
            if reusable_train:
                reused_train_resolved += 1
            else:
                fresh_train_resolved += 1
            evaluation_path = record_path(result.to_dict())
            notify(
                "candidate_resolved",
                ordinal=ordinal,
                structural_hash=identity.structural_hash,
                resolution=result.status,
                reusable_train=reusable_train,
                evaluation_path=evaluation_path,
            )

        final_items = self.store.run_candidates(run_id)
        pending_after = sum(item["state"] == "pending" for item in final_items)
        failed_after = sum(item["state"] == "failed" for item in final_items)
        complete_after = sum(item["state"] in _COMPLETED_STATES for item in final_items)
        if complete_after == len(final_items):
            run_status = "complete"
        elif failed_after:
            run_status = "incomplete"
        else:
            run_status = "paused"
        self.store.set_run_status(run_id, run_status)
        rss_after = int(process.memory_info().rss)
        peak_rss = max(peak_rss, rss_after)
        summary = RunnerInvocationSummary(
            invocation_id=invocation_id,
            run_id=run_id,
            run_status=run_status,
            execution_limit=max_new_evaluations,
            recovered_interrupted=recovered,
            resume_skipped=resume_skipped,
            cache_hits=cache_hits,
            newly_evaluated=newly_evaluated,
            newly_completed=newly_completed,
            newly_completed_invalid=newly_completed_invalid,
            newly_failed=newly_failed,
            pending_after=pending_after,
            failed_after=failed_after,
            elapsed_seconds=time.perf_counter() - started,
            rss_before_bytes=rss_before,
            peak_rss_bytes=peak_rss,
            rss_after_bytes=rss_after,
        )
        self.store.save_invocation(summary, started_at)
        notify(
            "invocation_completed",
            run_status=run_status,
            pending_after=pending_after,
            failed_after=failed_after,
        )
        return summary

    def verify_determinism(
        self,
        run_id: str,
        ordinals: Iterable[int],
    ) -> list[dict[str, Any]]:
        self.store.validate_run_evaluator(run_id, self.evaluator)
        items = {int(item["ordinal"]): item for item in self.store.run_candidates(run_id)}
        results: list[dict[str, Any]] = []
        for ordinal in ordinals:
            if ordinal not in items:
                raise IndexError(f"run has no candidate ordinal {ordinal}")
            item = items[ordinal]
            if item["state"] not in _COMPLETED_STATES:
                raise EvaluationStoreIntegrityError(
                    "determinism verification requires a completed cached candidate"
                )
            candidate = item["candidate"]
            identity = evaluation_cache_identity(self.evaluator, candidate)
            cached = self.store.lookup_verified(identity)
            if cached is None:
                raise EvaluationStoreIntegrityError("determinism check cache is missing")
            attempt_id, _ = self.store.begin_attempt(
                run_id,
                ordinal,
                identity.cache_key,
                mode="determinism_bypass",
                update_item_state=False,
            )
            with ProcessRssSampler(self.rss_sampling_interval_seconds) as sampler:
                observed = self.evaluator.evaluate(candidate)
            measurement = sampler.measurement()
            self.store.record_determinism_result(
                attempt_id=attempt_id,
                run_id=run_id,
                ordinal=ordinal,
                identity=identity,
                cached=cached,
                observed=observed,
                rss=measurement,
            )
            results.append(
                {
                    "ordinal": ordinal,
                    "structural_hash": identity.structural_hash,
                    "stored_result_fingerprint": cached["result_fingerprint"],
                    "observed_result_fingerprint": observed.result_fingerprint,
                    "match": cached["result_fingerprint"]
                    == observed.result_fingerprint,
                    "observed_total_seconds": observed.total_seconds,
                    "rss": asdict(measurement),
                }
            )
        self.store._write_run_status(run_id)
        return results


def select_stage6_smoke_candidates(
    candidates: Sequence[Mapping[str, Any]], count: int = 12
) -> list[dict[str, Any]]:
    """Choose a deterministic structure-diverse sample without historical metrics."""

    if count < 1:
        raise ValueError("smoke candidate count must be positive")
    rows = [dict(candidate) for candidate in candidates]
    if len(rows) < count:
        raise ValueError(f"registry has only {len(rows)} candidates; need {count}")
    rows.sort(key=lambda row: str(row["current_structural_hash"]))
    profiles: dict[str, dict[str, Any]] = {}
    for row in rows:
        names = [get_action(int(action_id)).name for action_id in row["prefix_token_ids"]]
        profiles[str(row["current_structural_hash"])] = {
            "has_ts": any(name.startswith("ts_") for name in names),
            "has_cs": any(name.startswith("cs_") for name in names),
        }
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()

    def take(pool: Sequence[Mapping[str, Any]], limit: int) -> None:
        if limit <= 0 or len(selected) >= count:
            return
        for candidate in pool:
            current_hash = str(candidate["current_structural_hash"])
            if current_hash in selected_hashes:
                continue
            selected.append(dict(candidate))
            selected_hashes.add(current_hash)
            if len(selected) >= count:
                return
            limit -= 1
            if limit == 0:
                return

    by_small = sorted(
        rows,
        key=lambda row: (
            int(row["node_count"]),
            int(row["depth"]),
            str(row["current_structural_hash"]),
        ),
    )
    take([row for row in by_small if int(row["node_count"]) <= 2], 2)
    take(
        [row for row in by_small if 3 <= int(row["node_count"]) <= 6],
        2,
    )
    take(
        [
            row
            for row in rows
            if profiles[str(row["current_structural_hash"])]["has_ts"]
            and not profiles[str(row["current_structural_hash"])]["has_cs"]
        ],
        2,
    )
    take(
        [
            row
            for row in rows
            if profiles[str(row["current_structural_hash"])]["has_cs"]
            and not profiles[str(row["current_structural_hash"])]["has_ts"]
        ],
        2,
    )
    take(
        [
            row
            for row in rows
            if profiles[str(row["current_structural_hash"])]["has_ts"]
            and profiles[str(row["current_structural_hash"])]["has_cs"]
        ],
        2,
    )
    by_complexity = sorted(
        rows,
        key=lambda row: (
            -int(row["node_count"]),
            -int(row["depth"]),
            str(row["current_structural_hash"]),
        ),
    )
    take(by_complexity, 1)
    take(
        sorted(
            [row for row in rows if len(row.get("origin_ids", [])) > 1],
            key=lambda row: (
                -len(row.get("origin_ids", [])),
                str(row["current_structural_hash"]),
            ),
        ),
        1,
    )
    take(rows, count - len(selected))
    if len(selected) != count:
        raise RuntimeError(f"deterministic smoke selection produced {len(selected)} candidates")
    return selected


__all__ = [
    "EVALUATION_CACHE_KEY_SCHEMA",
    "EVALUATION_RUN_SCHEMA",
    "EVALUATION_RUNNER_VERSION",
    "EVALUATION_STORE_SCHEMA",
    "DeterminismConflictError",
    "EvaluationCacheIdentity",
    "EvaluationStore",
    "EvaluationStoreIntegrityError",
    "FrozenEvaluationRun",
    "ProcessRssSampler",
    "RssMeasurement",
    "RunnerInvocationSummary",
    "Stage6EvaluationRunner",
    "VerifiedEvaluationRun",
    "VerifiedPartialEvaluationRun",
    "evaluation_cache_identity",
    "select_stage6_smoke_candidates",
]
