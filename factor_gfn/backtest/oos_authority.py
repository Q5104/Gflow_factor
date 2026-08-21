"""Verified, leakage-separated Test access for the static baseline strategies."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from factor_gfn.evaluator import FEATURE_NAMES, FactorInterpreter
from factor_gfn.evaluator.cross_section import (
    DEFAULT_CLEANING_CONFIG,
    clean_candidate_factor_cross_sections,
    encode_industry_panel,
)
from factor_gfn.grammar import Expression

from .baseline_factor_pool import OOS_UNTOUCHED, VerifiedFrozenBaselineFactorPool
from .context import Stage5DataContext
from .development_factor_matrix import (
    FeaturesOnlyFactorMatrix,
    STRATEGY_MATRIX_MISSING_CONTRACT,
    impute_strategy_factor_nonfinite,
    strategy_matrix_base_eligibility,
    strategy_matrix_cleaning_contract,
)
from .expression_compatibility import action_registry_for_vocabulary
from .stage6_evaluation import _stable_hash
from .static_strategy_bundle import (
    STRATEGY_IDS,
    STRATEGY_OOS_LOCKED,
    StrategyScores,
    VerifiedFrozenStrategyBundle,
    score_frozen_strategy,
)
from .strategy_input import load_verified_strategy_input


TEST_FEATURE_CONTEXT_SCHEMA = "factor_gfn.verified_test_feature_context.v1"
TEST_FACTOR_MATRIX_SCHEMA = "factor_gfn.test_factor_matrix.v1"
TEST_SCORE_ARTIFACT_SCHEMA = "factor_gfn.test_score_artifact.v1"
TEST_SCORE_ARTIFACT_VERSION = "prelabel-static-scores-with-rolling-seed-v2"
TEST_SCORE_MANIFEST_FILENAME = "test_score_manifest.json"
TEST_SCORE_FILENAME = "strategy_scores_test.parquet"
TEST_LABEL_SCHEMA = "factor_gfn.verified_test_labels.v1"
LABEL_FORMULA = "open[t+6] / open[t+1] - 1"


class OOSAuthorityError(RuntimeError):
    """A verified Test authority, binding, or leakage boundary is invalid."""


def _readonly(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    values.setflags(write=False)
    return values


def _array_digest(values: npt.ArrayLike) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


def _implementation_fingerprint() -> str:
    return _sha256_file(Path(__file__).resolve())


def _verify_authorities(
    pool: VerifiedFrozenBaselineFactorPool,
    bundle: VerifiedFrozenStrategyBundle,
) -> None:
    if not isinstance(pool, VerifiedFrozenBaselineFactorPool):
        raise TypeError("pool must be VerifiedFrozenBaselineFactorPool")
    if not isinstance(bundle, VerifiedFrozenStrategyBundle):
        raise TypeError("bundle must be VerifiedFrozenStrategyBundle")
    expected_hashes = pool.ordered_structural_hashes
    expected_directions = pool.frozen_train_directions
    if bundle.strategy_input_fingerprint is not None:
        if bundle.strategy_input_manifest_path is None:
            raise OOSAuthorityError("Strategy Input manifest path missing")
        strategy_input = load_verified_strategy_input(
            bundle.strategy_input_manifest_path
        )
        if (
            strategy_input.strategy_input_fingerprint
            != bundle.strategy_input_fingerprint
            or strategy_input.factor_pool_fingerprint
            != pool.baseline_factor_pool_fingerprint
            or strategy_input.factor_pool_manifest_path.resolve()
            != pool.manifest_path.resolve()
        ):
            raise OOSAuthorityError("Strategy Input authority mismatch")
        expected_hashes = strategy_input.ordered_structural_hashes
        expected_directions = strategy_input.frozen_train_directions
    if (
        pool.oos_status != OOS_UNTOUCHED
        or bundle.oos_status != STRATEGY_OOS_LOCKED
        or bundle.factor_pool_fingerprint
        != pool.baseline_factor_pool_fingerprint
        or bundle.ordered_structural_hashes != expected_hashes
        or bundle.frozen_directions != expected_directions
        or tuple(bundle.strategies) != STRATEGY_IDS
    ):
        raise OOSAuthorityError("factor pool and strategy bundle authority mismatch")


def _strategy_records(
    pool: VerifiedFrozenBaselineFactorPool,
    bundle: VerifiedFrozenStrategyBundle,
) -> tuple[Any, ...]:
    records_by_hash = {record.structural_hash: record for record in pool.records}
    try:
        records = tuple(
            records_by_hash[structural_hash]
            for structural_hash in bundle.ordered_structural_hashes
        )
    except KeyError as error:
        raise OOSAuthorityError("Strategy Input factor is absent from frozen pool") from error
    if tuple(record.train_direction for record in records) != bundle.frozen_directions:
        raise OOSAuthorityError("Strategy Input direction differs from frozen pool")
    return records


@dataclass(frozen=True, slots=True)
class VerifiedTestFeatureContext:
    factor_pool_fingerprint: str
    strategy_bundle_fingerprint: str
    source_context_fingerprint: str
    calendar_fingerprint: str
    requested_boundary: tuple[str, str]
    actual_boundary: tuple[str, str]
    history_dates: npt.NDArray[np.datetime64]
    stocks: npt.NDArray[np.str_]
    factor_tensor: npt.NDArray[np.floating] = field(repr=False)
    signal_rows: npt.NDArray[np.int64]
    signal_dates: npt.NDArray[np.datetime64]
    universe_mask: npt.NDArray[np.bool_] = field(repr=False)
    industry_labels: npt.NDArray[np.int32] = field(repr=False)
    fingerprint: str
    ordered_feature_names: tuple[str, ...] = FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class TestFactorMatrix:
    features: FeaturesOnlyFactorMatrix
    factor_pool_fingerprint: str
    strategy_bundle_fingerprint: str
    feature_context_fingerprint: str
    source_context_fingerprint: str
    calendar_fingerprint: str
    frozen_directions: tuple[int, ...]
    raw_universe_counts: Mapping[str, int]
    eligible_counts: Mapping[str, int]
    requested_boundary: tuple[str, str]
    actual_boundary: tuple[str, str]
    contract: Mapping[str, Any]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class TestScoreArtifact:
    manifest_path: Path
    fingerprint: str
    reused_existing_artifact: bool


@dataclass(frozen=True, slots=True)
class VerifiedTestScoreArtifact:
    manifest_path: Path
    fingerprint: str
    factor_pool_fingerprint: str
    strategy_bundle_fingerprint: str
    test_factor_matrix_fingerprint: str
    common_key_fingerprint: str
    scores: pd.DataFrame
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedTestLabels:
    score_artifact_fingerprint: str
    source_context_fingerprint: str
    calendar_fingerprint: str
    dates: npt.NDArray[np.datetime64]
    symbols: npt.NDArray[np.str_]
    forward_returns: npt.NDArray[np.float64]
    fingerprint: str
    first_access_evidence: Mapping[str, Any]


def unlock_verified_test_features(
    context: Stage5DataContext,
    pool: VerifiedFrozenBaselineFactorPool,
    bundle: VerifiedFrozenStrategyBundle,
) -> VerifiedTestFeatureContext:
    """Unlock only Test inputs needed for factor interpretation and cleaning."""

    if not isinstance(context, Stage5DataContext):
        raise TypeError("context must be Stage5DataContext")
    _verify_authorities(pool, bundle)
    if context.manifest.get("label_formula") != LABEL_FORMULA:
        raise OOSAuthorityError("Test label formula contract changed")
    boundary = context.splits["oos"]
    entries = [
        entry
        for entry in context.calendar
        if entry.split == "oos" and entry.label_within_split
    ]
    if not entries:
        raise OOSAuthorityError("Test split has no globally scheduled contained periods")
    signal_rows = np.asarray([entry.signal_row for entry in entries], dtype=np.int64)
    calendar_fingerprint = str(context.manifest.get("calendar", {}).get("fingerprint", ""))
    if len(calendar_fingerprint) != 64:
        raise OOSAuthorityError("global rebalance calendar fingerprint is invalid")
    deterministic = {
        "schema": TEST_FEATURE_CONTEXT_SCHEMA,
        "factor_pool_fingerprint": pool.baseline_factor_pool_fingerprint,
        "strategy_bundle_fingerprint": bundle.bundle_fingerprint,
        "source_context_fingerprint": context.fingerprint,
        "calendar_fingerprint": calendar_fingerprint,
        "requested_boundary": [boundary.requested_start, boundary.requested_end],
        "actual_boundary": [boundary.actual_start, boundary.actual_end],
        "signal_rows": signal_rows.tolist(),
        "signal_dates": context.dates[signal_rows].astype(str).tolist(),
        "access": "features_only_no_forward_returns",
    }
    if context.ordered_feature_names != FEATURE_NAMES:
        deterministic["expression_features"] = {
            "ordered_feature_names": list(context.ordered_feature_names),
        }
    return VerifiedTestFeatureContext(
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        strategy_bundle_fingerprint=bundle.bundle_fingerprint,
        source_context_fingerprint=context.fingerprint,
        calendar_fingerprint=calendar_fingerprint,
        requested_boundary=(boundary.requested_start, boundary.requested_end),
        actual_boundary=(boundary.actual_start, boundary.actual_end),
        history_dates=_readonly(np.asarray(context.dates).copy()),
        stocks=_readonly(np.asarray(context.stocks).copy()),
        factor_tensor=context.expression_feature_tensor,
        signal_rows=_readonly(signal_rows),
        signal_dates=_readonly(np.asarray(context.dates[signal_rows]).copy()),
        universe_mask=_readonly(np.asarray(context._universe_mask[signal_rows]).copy()),
        industry_labels=_readonly(np.asarray(context._industry_labels[signal_rows]).copy()),
        fingerprint=_stable_hash(deterministic),
        ordered_feature_names=context.ordered_feature_names,
    )


def build_test_factor_matrix(
    feature_context: VerifiedTestFeatureContext,
    pool: VerifiedFrozenBaselineFactorPool,
    bundle: VerifiedFrozenStrategyBundle,
) -> TestFactorMatrix:
    """Interpret through Test with full warm-up and emit base-eligible features only."""

    if not isinstance(feature_context, VerifiedTestFeatureContext):
        raise TypeError("feature_context must be VerifiedTestFeatureContext")
    _verify_authorities(pool, bundle)
    if (
        feature_context.factor_pool_fingerprint != pool.baseline_factor_pool_fingerprint
        or feature_context.strategy_bundle_fingerprint != bundle.bundle_fingerprint
    ):
        raise OOSAuthorityError("Test feature context authority mismatch")
    cleaning = strategy_matrix_cleaning_contract(DEFAULT_CLEANING_CONFIG)
    shared_contract = bundle.manifest["shared_contract"]
    if (
        _stable_hash(cleaning) != shared_contract["cleaning_contract_fingerprint"]
        or shared_contract.get("missing") != STRATEGY_MATRIX_MISSING_CONTRACT
    ):
        raise OOSAuthorityError("Test cleaning differs from frozen Strategy Bundle")

    date_count = feature_context.signal_rows.size
    stock_count = feature_context.stocks.size
    strategy_records = _strategy_records(pool, bundle)
    factor_count = len(strategy_records)
    directional = np.full((date_count, stock_count, factor_count), np.nan)
    encoded = encode_industry_panel(
        feature_context.industry_labels, (date_count, stock_count)
    )
    base_eligible = strategy_matrix_base_eligibility(
        feature_context.universe_mask,
        encoded,
    )
    try:
        action_registry = action_registry_for_vocabulary(
            pool.manifest.get("vocabulary")
        )
    except ValueError as error:
        raise OOSAuthorityError("frozen factor pool vocabulary is invalid") from error
    if action_registry.leaf_names != feature_context.ordered_feature_names:
        raise OOSAuthorityError(
            "frozen factor pool vocabulary does not match expression feature schema"
        )
    interpreter = FactorInterpreter(
        feature_context.factor_tensor,
        ordered_feature_names=feature_context.ordered_feature_names,
    )
    for index, record in enumerate(strategy_records):
        expression = Expression.from_prefix(
            record.prefix_token_ids,
            action_registry=action_registry,
        )
        if (
            expression.to_formula() != record.formula
            or expression.structural_hash() != record.structural_hash
        ):
            raise OOSAuthorityError("frozen factor expression identity mismatch")
        raw = np.asarray(interpreter.evaluate(expression))[feature_context.signal_rows]
        cleaned = clean_candidate_factor_cross_sections(
            raw,
            None,
            feature_context.universe_mask,
            config=DEFAULT_CLEANING_CONFIG,
            encoded_industries=encoded,
        )
        imputed = impute_strategy_factor_nonfinite(cleaned, base_eligible)
        directional[:, :, index] = record.train_direction * imputed
    date_positions, stock_positions = np.nonzero(base_eligible)
    dates = _readonly(feature_context.signal_dates[date_positions].copy())
    symbols = _readonly(feature_context.stocks[stock_positions].copy())
    values = _readonly(directional[date_positions, stock_positions].copy())
    aliases = bundle.feature_aliases
    hashes = bundle.ordered_structural_hashes
    feature_fingerprint = _stable_hash(
        {
            "schema": "factor_gfn.features_only_factor_matrix.v1",
            "split": "test",
            "factor_pool_fingerprint": pool.baseline_factor_pool_fingerprint,
            "feature_aliases": list(aliases),
            "ordered_structural_hashes": list(hashes),
            "dates_digest": _array_digest(dates),
            "symbols_digest": _array_digest(symbols),
            "values_digest": _array_digest(values),
        }
    )
    features = FeaturesOnlyFactorMatrix(
        split="test",
        dates=dates,
        symbols=symbols,
        values=values,
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        feature_aliases=aliases,
        ordered_structural_hashes=hashes,
        fingerprint=feature_fingerprint,
    )
    raw_counts = {
        str(date): int(mask.sum())
        for date, mask in zip(feature_context.signal_dates, feature_context.universe_mask)
    }
    eligible_counts = {
        str(date): int(mask.sum())
        for date, mask in zip(feature_context.signal_dates, base_eligible)
    }
    contract = {
        "schema": TEST_FACTOR_MATRIX_SCHEMA,
        "factor_order": (
            "strategy_input.frozen_order_prefix_top100"
            if bundle.strategy_input_fingerprint is not None
            else "frozen_pool.ordered_structural_hashes"
        ),
        "direction": "frozen_train_direction_times_cleaned_factor",
        "cleaning": cleaning,
        "missing": STRATEGY_MATRIX_MISSING_CONTRACT,
        "calendar": "shared_global_5_day_phase",
        "warmup": "full_history_through_test_then_slice_signal_dates",
        "label_fields": "absent",
    }
    if bundle.strategy_input_fingerprint is not None:
        contract["strategy_input_fingerprint"] = (
            bundle.strategy_input_fingerprint
        )
    fingerprint = _stable_hash(
        {
            "schema": TEST_FACTOR_MATRIX_SCHEMA,
            "feature_context_fingerprint": feature_context.fingerprint,
            "factor_pool_fingerprint": pool.baseline_factor_pool_fingerprint,
            "strategy_bundle_fingerprint": bundle.bundle_fingerprint,
            "feature_fingerprint": feature_fingerprint,
            "calendar_fingerprint": feature_context.calendar_fingerprint,
            "directions": list(bundle.frozen_directions),
            "raw_universe_counts": raw_counts,
            "eligible_counts": eligible_counts,
            "contract": contract,
        }
    )
    return TestFactorMatrix(
        features=features,
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        strategy_bundle_fingerprint=bundle.bundle_fingerprint,
        feature_context_fingerprint=feature_context.fingerprint,
        source_context_fingerprint=feature_context.source_context_fingerprint,
        calendar_fingerprint=feature_context.calendar_fingerprint,
        frozen_directions=bundle.frozen_directions,
        raw_universe_counts=MappingProxyType(raw_counts),
        eligible_counts=MappingProxyType(eligible_counts),
        requested_boundary=feature_context.requested_boundary,
        actual_boundary=feature_context.actual_boundary,
        contract=MappingProxyType(contract),
        fingerprint=fingerprint,
    )


def generate_test_strategy_scores(
    bundle: VerifiedFrozenStrategyBundle,
    matrix: TestFactorMatrix,
) -> Mapping[str, StrategyScores]:
    if not isinstance(bundle, VerifiedFrozenStrategyBundle):
        raise TypeError("bundle must be VerifiedFrozenStrategyBundle")
    if not isinstance(matrix, TestFactorMatrix):
        raise TypeError("matrix must be TestFactorMatrix")
    if matrix.strategy_bundle_fingerprint != bundle.bundle_fingerprint:
        raise OOSAuthorityError("Test matrix and Strategy Bundle mismatch")
    result = {
        strategy_id: score_frozen_strategy(bundle.strategies[strategy_id], matrix.features)
        for strategy_id in STRATEGY_IDS
    }
    expected = sorted(zip(matrix.features.dates.astype(str), matrix.features.symbols))
    for strategy_id, scores in result.items():
        keys = list(zip(scores.dates.astype(str), scores.symbols))
        if keys != expected:
            raise OOSAuthorityError(f"common Test score keys mismatch: {strategy_id}")
    return MappingProxyType(result)


def _score_frame(scores: Mapping[str, StrategyScores]) -> pd.DataFrame:
    frames = []
    for strategy_id in STRATEGY_IDS:
        score = scores[strategy_id]
        frames.append(
            pd.DataFrame(
                {
                    "date": score.dates.astype("datetime64[D]"),
                    "symbol": score.symbols.astype(str),
                    "strategy_id": strategy_id,
                    "strategy_score": score.strategy_scores,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(["date", "symbol", "strategy_id"], kind="mergesort").reset_index(drop=True)


def _score_manifest_fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: manifest.get(key)
        for key in (
            "schema",
            "version",
            "factor_pool_fingerprint",
            "strategy_bundle_fingerprint",
            "test_factor_matrix_fingerprint",
            "strategy_ids",
            "common_key_fingerprint",
            "score_payload_digest",
            "score_usage_contract",
            "artifacts",
            "oos_access_state",
            "generation_identity",
        )
    }


def freeze_test_score_artifact(
    pool: VerifiedFrozenBaselineFactorPool,
    bundle: VerifiedFrozenStrategyBundle,
    matrix: TestFactorMatrix,
    scores: Mapping[str, StrategyScores],
    runs_root: str | Path,
) -> TestScoreArtifact:
    _verify_authorities(pool, bundle)
    if (
        set(scores) != set(STRATEGY_IDS)
        or matrix.factor_pool_fingerprint != pool.baseline_factor_pool_fingerprint
        or matrix.strategy_bundle_fingerprint != bundle.bundle_fingerprint
    ):
        raise OOSAuthorityError("Test score freeze authorities mismatch")
    expected_keys = sorted(
        zip(matrix.features.dates.astype(str), matrix.features.symbols.astype(str))
    )
    for strategy_id in STRATEGY_IDS:
        score = scores[strategy_id]
        keys = list(zip(score.dates.astype(str), score.symbols.astype(str)))
        if keys != expected_keys or len(keys) != len(set(keys)):
            raise OOSAuthorityError(f"common Test score keys mismatch: {strategy_id}")
    frame = _score_frame(scores)
    expected_rows = matrix.features.row_count * len(STRATEGY_IDS)
    if len(frame) != expected_rows or not np.isfinite(frame["strategy_score"]).all():
        raise OOSAuthorityError("Test scores are incomplete or nonfinite")
    key_frame = frame.loc[frame["strategy_id"] == STRATEGY_IDS[0], ["date", "symbol"]]
    common_key_fingerprint = _stable_hash(
        key_frame.assign(date=key_frame["date"].astype(str)).to_dict("records")
    )
    payload = _parquet_bytes(frame)
    artifacts = {
        TEST_SCORE_FILENAME: {"size_bytes": len(payload), "sha256": _sha256_bytes(payload)}
    }
    manifest: dict[str, Any] = {
        "schema": TEST_SCORE_ARTIFACT_SCHEMA,
        "version": TEST_SCORE_ARTIFACT_VERSION,
        "factor_pool_fingerprint": pool.baseline_factor_pool_fingerprint,
        "strategy_bundle_fingerprint": bundle.bundle_fingerprint,
        "test_factor_matrix_fingerprint": matrix.fingerprint,
        "strategy_ids": list(STRATEGY_IDS),
        "common_key_fingerprint": common_key_fingerprint,
        "score_payload_digest": _sha256_bytes(payload),
        "row_count": len(frame),
        "key_count": matrix.features.row_count,
        "key_range": {
            "start": str(matrix.features.dates.min()) if matrix.features.row_count else None,
            "end": str(matrix.features.dates.max()) if matrix.features.row_count else None,
        },
        "artifacts": artifacts,
        "oos_access_state": "scores_frozen_labels_not_accessed",
        "score_usage_contract": {
            "equal_weight": "final_prelabeled_score",
            "lightgbm": "final_prelabeled_score",
            "fixed_icir": (
                "initial_seed_score_only_replaced_by_causal_rolling_icir_after_label_gate"
            ),
        },
        "generation_identity": {
            "implementation_fingerprint": _implementation_fingerprint(),
            "score_api_label_parameters": [
                name
                for name in inspect.signature(score_frozen_strategy).parameters
                if "label" in name or "return" in name
            ],
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "created_at_excluded_from_fingerprint": True,
    }
    manifest["test_score_artifact_fingerprint"] = _stable_hash(
        _score_manifest_fingerprint_payload(manifest)
    )
    fingerprint = str(manifest["test_score_artifact_fingerprint"])
    root = Path(runs_root).resolve() / "oos_test_scores"
    target = root / fingerprint
    manifest_path = target / TEST_SCORE_MANIFEST_FILENAME
    if target.exists():
        verified = load_verified_test_score_artifact(manifest_path, pool, bundle, matrix)
        return TestScoreArtifact(manifest_path, verified.fingerprint, True)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.tmp-", dir=root))
    try:
        (temporary / TEST_SCORE_FILENAME).write_bytes(payload)
        (temporary / TEST_SCORE_MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))
        _verify_test_score_directory(
            temporary / TEST_SCORE_MANIFEST_FILENAME,
            pool,
            bundle,
            matrix,
            require_directory_identity=False,
        )
        os.replace(temporary, target)
        load_verified_test_score_artifact(manifest_path, pool, bundle, matrix)
        return TestScoreArtifact(manifest_path, fingerprint, False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _verify_test_score_directory(
    manifest_path: Path,
    pool: VerifiedFrozenBaselineFactorPool,
    bundle: VerifiedFrozenStrategyBundle,
    matrix: TestFactorMatrix,
    *,
    require_directory_identity: bool,
) -> VerifiedTestScoreArtifact:
    _verify_authorities(pool, bundle)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OOSAuthorityError("cannot read Test score manifest") from error
    fingerprint = str(manifest.get("test_score_artifact_fingerprint", ""))
    if (
        manifest.get("schema") != TEST_SCORE_ARTIFACT_SCHEMA
        or manifest.get("version") != TEST_SCORE_ARTIFACT_VERSION
        or _stable_hash(_score_manifest_fingerprint_payload(manifest)) != fingerprint
        or (require_directory_identity and manifest_path.parent.name != fingerprint)
        or manifest.get("factor_pool_fingerprint") != pool.baseline_factor_pool_fingerprint
        or manifest.get("strategy_bundle_fingerprint") != bundle.bundle_fingerprint
        or manifest.get("test_factor_matrix_fingerprint") != matrix.fingerprint
        or manifest.get("strategy_ids") != list(STRATEGY_IDS)
        or manifest.get("generation_identity", {}).get("implementation_fingerprint")
        != _implementation_fingerprint()
        or manifest.get("generation_identity", {}).get("score_api_label_parameters") != []
    ):
        raise OOSAuthorityError("Test score manifest or authority mismatch")
    metadata = manifest.get("artifacts", {}).get(TEST_SCORE_FILENAME, {})
    path = manifest_path.parent / TEST_SCORE_FILENAME
    if (
        not path.is_file()
        or path.stat().st_size != int(metadata.get("size_bytes", -1))
        or _sha256_file(path) != metadata.get("sha256")
    ):
        raise OOSAuthorityError("Test score payload changed")
    frame = pd.read_parquet(path)
    expected_columns = ["date", "symbol", "strategy_id", "strategy_score"]
    if list(frame.columns) != expected_columns or len(frame) != int(manifest.get("row_count", -1)):
        raise OOSAuthorityError("Test score schema or row count changed")
    if set(frame["strategy_id"]) != set(STRATEGY_IDS) or not np.isfinite(frame["strategy_score"]).all():
        raise OOSAuthorityError("Test score strategies or values changed")
    canonical_keys = None
    for strategy_id in STRATEGY_IDS:
        subset = frame.loc[frame["strategy_id"] == strategy_id, ["date", "symbol"]]
        keys = list(zip(pd.to_datetime(subset["date"]).dt.strftime("%Y-%m-%d"), subset["symbol"].astype(str)))
        if len(keys) != len(set(keys)) or (canonical_keys is not None and keys != canonical_keys):
            raise OOSAuthorityError("Test score common key contract changed")
        canonical_keys = keys
    expected_keys = sorted(zip(matrix.features.dates.astype(str), matrix.features.symbols.astype(str)))
    if canonical_keys != expected_keys:
        raise OOSAuthorityError("Test score keys differ from score-eligible features")
    return VerifiedTestScoreArtifact(
        manifest_path=manifest_path,
        fingerprint=fingerprint,
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        strategy_bundle_fingerprint=bundle.bundle_fingerprint,
        test_factor_matrix_fingerprint=matrix.fingerprint,
        common_key_fingerprint=str(manifest["common_key_fingerprint"]),
        scores=frame.copy(deep=True),
        manifest=MappingProxyType(manifest),
    )


def load_verified_test_score_artifact(
    manifest_path: str | Path,
    pool: VerifiedFrozenBaselineFactorPool,
    bundle: VerifiedFrozenStrategyBundle,
    matrix: TestFactorMatrix,
) -> VerifiedTestScoreArtifact:
    return _verify_test_score_directory(
        Path(manifest_path).resolve(),
        pool,
        bundle,
        matrix,
        require_directory_identity=True,
    )


def load_verified_test_labels(
    context: Stage5DataContext,
    pool: VerifiedFrozenBaselineFactorPool,
    bundle: VerifiedFrozenStrategyBundle,
    matrix: TestFactorMatrix,
    scores: VerifiedTestScoreArtifact,
) -> VerifiedTestLabels:
    """First and only E1b label gateway; requires already frozen verified scores."""

    if not isinstance(scores, VerifiedTestScoreArtifact):
        raise PermissionError("Test labels require a verified frozen Test Score Artifact")
    _verify_authorities(pool, bundle)
    reverified_scores = _verify_test_score_directory(
        scores.manifest_path,
        pool,
        bundle,
        matrix,
        require_directory_identity=True,
    )
    if reverified_scores.fingerprint != scores.fingerprint:
        raise OOSAuthorityError("Test Score Artifact verification changed before label access")
    if (
        context.fingerprint != matrix.source_context_fingerprint
        or
        scores.factor_pool_fingerprint != pool.baseline_factor_pool_fingerprint
        or scores.strategy_bundle_fingerprint != bundle.bundle_fingerprint
        or scores.test_factor_matrix_fingerprint != matrix.fingerprint
        or str(context.manifest.get("calendar", {}).get("fingerprint", ""))
        != matrix.calendar_fingerprint
    ):
        raise OOSAuthorityError("Test label authority chain mismatch")
    entries = [
        entry
        for entry in context.calendar
        if entry.split == "oos" and entry.label_within_split
    ]
    date_to_row = {str(context.dates[entry.signal_row]): entry.signal_row for entry in entries}
    symbol_to_col = {str(symbol): index for index, symbol in enumerate(context.stocks)}
    labels = np.full(matrix.features.row_count, np.nan, dtype=np.float64)
    for index, (date, symbol) in enumerate(
        zip(matrix.features.dates.astype(str), matrix.features.symbols.astype(str))
    ):
        if date not in date_to_row or symbol not in symbol_to_col:
            raise OOSAuthorityError("Test label key is outside the frozen Test context")
        labels[index] = context._forward_returns[date_to_row[date], symbol_to_col[symbol]]
    fingerprint = _stable_hash(
        {
            "schema": TEST_LABEL_SCHEMA,
            "formula": LABEL_FORMULA,
            "entry_offset": 1,
            "exit_offset": 6,
            "score_artifact_fingerprint": scores.fingerprint,
            "source_context_fingerprint": context.fingerprint,
            "calendar_fingerprint": matrix.calendar_fingerprint,
            "dates_digest": _array_digest(matrix.features.dates),
            "symbols_digest": _array_digest(matrix.features.symbols),
            "labels_digest": _array_digest(labels),
        }
    )
    evidence = {
        "event": "first_test_label_access_after_verified_score_freeze",
        "score_artifact_fingerprint": scores.fingerprint,
        "score_manifest_path": str(scores.manifest_path),
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_excluded_from_label_fingerprint": True,
    }
    return VerifiedTestLabels(
        score_artifact_fingerprint=scores.fingerprint,
        source_context_fingerprint=context.fingerprint,
        calendar_fingerprint=matrix.calendar_fingerprint,
        dates=_readonly(matrix.features.dates.copy()),
        symbols=_readonly(matrix.features.symbols.copy()),
        forward_returns=_readonly(labels),
        fingerprint=fingerprint,
        first_access_evidence=MappingProxyType(evidence),
    )


__all__ = [
    "LABEL_FORMULA",
    "OOSAuthorityError",
    "TEST_SCORE_FILENAME",
    "TEST_SCORE_MANIFEST_FILENAME",
    "TestFactorMatrix",
    "TestScoreArtifact",
    "VerifiedTestFeatureContext",
    "VerifiedTestLabels",
    "VerifiedTestScoreArtifact",
    "build_test_factor_matrix",
    "freeze_test_score_artifact",
    "generate_test_strategy_scores",
    "load_verified_test_labels",
    "load_verified_test_score_artifact",
    "unlock_verified_test_features",
]
