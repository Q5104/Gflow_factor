"""Development-only frozen-factor matrices for static baseline strategies.

The builder consumes an already verified Baseline Factor Pool and a Stage 6
context that is physically truncated at Validation.  It evaluates every frozen
expression once over the full permitted history, applies the shared
cross-sectional cleaning contract, freezes the Train direction, and only then
cuts compact Train/Validation feature rows.  Labels are stored separately from
the features-only matrices.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from factor_gfn.evaluator import FactorInterpreter
from factor_gfn.evaluator.cross_section import (
    DEFAULT_CLEANING_CONFIG,
    CrossSectionalCleaningConfig,
    clean_candidate_factor_cross_sections,
    encode_industry_panel,
)
from factor_gfn.grammar import ActionRegistry, Expression

from .baseline_factor_pool import (
    OOS_UNTOUCHED,
    VerifiedFrozenBaselineFactorPool,
    load_verified_baseline_factor_pool,
)
from .expression_compatibility import action_registry_for_vocabulary
from .strategy_input import (
    STRATEGY_INPUT_TOP_K,
    VerifiedStrategyInput,
    load_verified_strategy_input,
)
from .stage6_evaluation import (
    STAGE6_SPLIT_NAMES,
    Stage6EvaluationContext,
    Stage6SplitName,
    _stable_hash,
)


DEVELOPMENT_FACTOR_MATRIX_SCHEMA = "factor_gfn.development_factor_matrix.v1"
DEVELOPMENT_FACTOR_MATRIX_VERSION = "development-factor-matrix-v1"
DEVELOPMENT_FACTOR_MATRIX_MANIFEST_FILENAME = (
    "development_factor_matrix_manifest.json"
)
FACTOR_MAPPING_FILENAME = "factor_mapping.json"
LABEL_FORMULA = "open[t+6] / open[t+1] - 1"
STRATEGY_MATRIX_MISSING_CONTRACT = (
    "post_cleaning_factor_specific_nonfinite_to_zero"
)


class DevelopmentFactorMatrixIntegrityError(RuntimeError):
    """A development matrix or one of its frozen authorities is invalid."""


def _readonly(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    values.setflags(write=False)
    return values


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npy_bytes(values: npt.NDArray[Any]) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(values), allow_pickle=False)
    return stream.getvalue()


def _array_digest(values: npt.NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256_bytes(metadata + b"\0" + array.tobytes(order="C"))


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentFactorMatrixIntegrityError(
            f"cannot read development matrix JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise DevelopmentFactorMatrixIntegrityError(
            f"development matrix JSON must be an object: {path}"
        )
    return value


@dataclass(frozen=True, slots=True)
class FrozenFactorFeature:
    alias: str
    factor_index: int
    structural_hash: str
    formula: str
    prefix_token_ids: tuple[int, ...]
    node_count: int
    depth: int
    train_direction: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "factor_index": self.factor_index,
            "structural_hash": self.structural_hash,
            "formula": self.formula,
            "prefix_token_ids": list(self.prefix_token_ids),
            "node_count": self.node_count,
            "depth": self.depth,
            "train_direction": self.train_direction,
        }


@dataclass(frozen=True, slots=True)
class FeaturesOnlyFactorMatrix:
    """Base-eligible, label-free rows ordered by date then source stock order."""

    split: str
    dates: npt.NDArray[np.datetime64]
    symbols: npt.NDArray[np.str_]
    values: npt.NDArray[np.float64]
    factor_pool_fingerprint: str
    feature_aliases: tuple[str, ...]
    ordered_structural_hashes: tuple[str, ...]
    fingerprint: str

    @property
    def row_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def factor_count(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class DevelopmentSplitMatrix:
    features: FeaturesOnlyFactorMatrix
    forward_returns: npt.NDArray[np.float64]
    label_fingerprint: str
    boundary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DevelopmentFactorMatrices:
    artifact_manifest_path: Path | None
    artifact_fingerprint: str | None
    factor_pool_manifest_path: Path
    factor_pool_fingerprint: str
    context_fingerprint: str
    calendar_fingerprint: str
    feature_mapping: tuple[FrozenFactorFeature, ...]
    splits: Mapping[Stage6SplitName, DevelopmentSplitMatrix]
    contract: Mapping[str, Any]
    provenance: Mapping[str, Any]
    logical_fingerprint: str
    strategy_input_manifest_path: Path | None = None
    strategy_input_fingerprint: str | None = None

    @property
    def fingerprint(self) -> str:
        """Published identity when available, otherwise the logical build identity."""

        return self.artifact_fingerprint or self.logical_fingerprint


@dataclass(frozen=True, slots=True)
class DevelopmentFactorMatrixArtifact:
    manifest_path: Path
    fingerprint: str
    reused_existing_artifact: bool


def _feature_fingerprint(
    *,
    split: str,
    dates: npt.NDArray[np.datetime64],
    symbols: npt.NDArray[np.str_],
    values: npt.NDArray[np.float64],
    factor_pool_fingerprint: str,
    aliases: tuple[str, ...],
    structural_hashes: tuple[str, ...],
) -> str:
    return _stable_hash(
        {
            "schema": "factor_gfn.features_only_factor_matrix.v1",
            "split": split,
            "factor_pool_fingerprint": factor_pool_fingerprint,
            "feature_aliases": list(aliases),
            "ordered_structural_hashes": list(structural_hashes),
            "dates_digest": _array_digest(dates),
            "symbols_digest": _array_digest(symbols),
            "values_digest": _array_digest(values),
        }
    )


def _label_fingerprint(
    *,
    split: str,
    dates: npt.NDArray[np.datetime64],
    symbols: npt.NDArray[np.str_],
    labels: npt.NDArray[np.float64],
    calendar_fingerprint: str,
) -> str:
    return _stable_hash(
        {
            "schema": "factor_gfn.development_labels.v1",
            "split": split,
            "formula": LABEL_FORMULA,
            "calendar_fingerprint": calendar_fingerprint,
            "dates_digest": _array_digest(dates),
            "symbols_digest": _array_digest(symbols),
            "labels_digest": _array_digest(labels),
        }
    )


def _verify_expression_identity(
    feature: FrozenFactorFeature,
    action_registry: ActionRegistry,
) -> Expression:
    try:
        expression = Expression.from_prefix(
            feature.prefix_token_ids,
            action_registry=action_registry,
        )
    except (IndexError, TypeError, ValueError) as error:
        raise DevelopmentFactorMatrixIntegrityError(
            f"invalid frozen expression prefix for {feature.alias}"
        ) from error
    stats = expression.stats
    if (
        expression.to_prefix() != feature.prefix_token_ids
        or expression.to_formula() != feature.formula
        or expression.structural_hash() != feature.structural_hash
        or stats.node_count != feature.node_count
        or stats.depth != feature.depth
        or feature.train_direction not in (-1, 1)
    ):
        raise DevelopmentFactorMatrixIntegrityError(
            f"frozen expression identity mismatch for {feature.alias}"
        )
    return expression


def _matrix_contract(
    context: Stage6EvaluationContext,
    cleaning_config: CrossSectionalCleaningConfig,
    strategy_input: VerifiedStrategyInput | None,
) -> dict[str, Any]:
    contract = {
        "schema": "factor_gfn.development_factor_matrix_contract.v1",
        "interpretation": "one_full_history_pass_through_validation",
        "feature_axes": ["sample", "frozen_factor"],
        "sample_order": "rebalance_date_then_context_stock_order",
        "factor_order": (
            "strategy_input.frozen_order_prefix_top100"
            if strategy_input is not None
            else "frozen_pool.ordered_structural_hashes"
        ),
        "feature_alias": "factor_{zero_based_index:03d}",
        "cleaning": strategy_matrix_cleaning_contract(cleaning_config),
        "direction": "frozen_train_direction_times_cleaned_factor",
        "missing": STRATEGY_MATRIX_MISSING_CONTRACT,
        "label": {
            "formula": LABEL_FORMULA,
            "storage": "separate_from_features",
            "split_containment": "entry_and_exit_within_same_split",
        },
        "calendar": "shared_train_anchor_every_5_global_rows_no_shift",
        "evaluation_config": asdict(context.config.evaluation),
        "oos": "not_loaded_and_not_accepted",
    }
    if strategy_input is not None:
        contract["strategy_input"] = {
            "strategy_input_fingerprint": (
                strategy_input.strategy_input_fingerprint
            ),
            "factor_pool_fingerprint": strategy_input.factor_pool_fingerprint,
            "policy": "frozen_order_prefix",
            "top_k": strategy_input.top_k,
        }
    return contract


def strategy_matrix_cleaning_contract(
    cleaning_config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
) -> dict[str, Any]:
    """Return the one frozen cleaning/imputation contract shared by Dev and Test."""

    return {
        **asdict(cleaning_config),
        "order": [
            "winsorize",
            "point_in_time_industry_neutralize",
            "zscore",
            "factor_specific_nonfinite_to_zero",
        ],
        "industry": "SW_level_1_point_in_time",
        "neutralization_failure": "leave_nonfinite_then_strategy_impute_zero",
        "imputation_scope": "base_eligible_stocks_only",
        "base_eligibility": "universe_and_known_point_in_time_industry",
    }


def strategy_matrix_base_eligibility(
    universe_mask: npt.ArrayLike,
    encoded_industries: Any,
) -> npt.NDArray[np.bool_]:
    """Return the existing E-stage stock eligibility without factor availability."""

    universe = np.asarray(universe_mask, dtype=bool)
    industry_codes = np.asarray(encoded_industries.codes)
    if industry_codes.shape != universe.shape or industry_codes.dtype != np.int32:
        raise DevelopmentFactorMatrixIntegrityError(
            "strategy matrix industry eligibility shape or dtype is invalid"
        )
    return universe & (industry_codes >= 0)


def impute_strategy_factor_nonfinite(
    cleaned: npt.ArrayLike,
    base_eligible: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Fill factor-specific post-cleaning nonfinite values only for eligible stocks."""

    values = np.asarray(cleaned, dtype=np.float64)
    eligible = np.asarray(base_eligible, dtype=bool)
    if values.shape != eligible.shape:
        raise DevelopmentFactorMatrixIntegrityError(
            "strategy factor and base eligibility shapes differ"
        )
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[eligible] = np.where(np.isfinite(values[eligible]), values[eligible], 0.0)
    return result


def build_development_factor_matrices(
    frozen_pool: VerifiedFrozenBaselineFactorPool,
    context: Stage6EvaluationContext,
    *,
    strategy_input: VerifiedStrategyInput | None = None,
    cleaning_config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
) -> DevelopmentFactorMatrices:
    """Build compact Train/Validation matrices without any OOS accessor."""

    if not isinstance(frozen_pool, VerifiedFrozenBaselineFactorPool):
        raise TypeError("frozen_pool must be VerifiedFrozenBaselineFactorPool")
    if not isinstance(context, Stage6EvaluationContext):
        raise TypeError("context must be Stage6EvaluationContext")
    if frozen_pool.oos_status != OOS_UNTOUCHED:
        raise DevelopmentFactorMatrixIntegrityError("factor pool OOS lock changed")
    if context.manifest.get("label_formula") != LABEL_FORMULA:
        raise DevelopmentFactorMatrixIntegrityError("development label formula changed")
    if context.manifest.get("oos") != {
        "loaded": False,
        "exposed": False,
        "candidate_evaluation_count": 0,
    }:
        raise DevelopmentFactorMatrixIntegrityError("Stage 6 context exposes OOS")
    if strategy_input is not None:
        if not isinstance(strategy_input, VerifiedStrategyInput):
            raise TypeError("strategy_input must be VerifiedStrategyInput")
        if (
            strategy_input.oos_status != OOS_UNTOUCHED
            or strategy_input.top_k != STRATEGY_INPUT_TOP_K
            or len(strategy_input.records) != STRATEGY_INPUT_TOP_K
            or strategy_input.factor_pool_fingerprint
            != frozen_pool.baseline_factor_pool_fingerprint
            or strategy_input.factor_pool_manifest_path.resolve()
            != frozen_pool.manifest_path.resolve()
            or strategy_input.ordered_structural_hashes
            != frozen_pool.ordered_structural_hashes[: strategy_input.top_k]
            or strategy_input.frozen_train_directions
            != frozen_pool.frozen_train_directions[: strategy_input.top_k]
        ):
            raise DevelopmentFactorMatrixIntegrityError(
                "Strategy Input differs from the verified frozen pool prefix"
            )
    records = (
        strategy_input.records if strategy_input is not None else frozen_pool.records
    )
    expected_hashes = (
        strategy_input.ordered_structural_hashes
        if strategy_input is not None
        else frozen_pool.ordered_structural_hashes
    )
    if not records or len(records) != len(expected_hashes):
        raise DevelopmentFactorMatrixIntegrityError("frozen factor pool is empty or inconsistent")

    mapping = tuple(
        FrozenFactorFeature(
            alias=f"factor_{index:03d}",
            factor_index=index,
            structural_hash=record.structural_hash,
            formula=record.formula,
            prefix_token_ids=record.prefix_token_ids,
            node_count=record.node_count,
            depth=record.depth,
            train_direction=record.train_direction,
        )
        for index, record in enumerate(records)
    )
    if tuple(item.structural_hash for item in mapping) != expected_hashes:
        raise DevelopmentFactorMatrixIntegrityError("frozen factor order changed")
    try:
        action_registry = action_registry_for_vocabulary(
            frozen_pool.manifest.get("vocabulary")
        )
    except ValueError as error:
        raise DevelopmentFactorMatrixIntegrityError(
            "frozen factor pool vocabulary is invalid"
        ) from error
    if action_registry.leaf_names != context.ordered_feature_names:
        raise DevelopmentFactorMatrixIntegrityError(
            "frozen factor pool vocabulary does not match expression feature schema"
        )
    expressions = tuple(
        _verify_expression_identity(item, action_registry) for item in mapping
    )
    interpreter = FactorInterpreter(
        context.expression_feature_tensor,
        ordered_feature_names=context.ordered_feature_names,
    )
    aliases = tuple(item.alias for item in mapping)
    hashes = tuple(item.structural_hash for item in mapping)
    calendar_fingerprint = str(context.manifest.get("calendar", {}).get("fingerprint"))
    if len(calendar_fingerprint) != 64:
        raise DevelopmentFactorMatrixIntegrityError("calendar fingerprint is invalid")

    split_sources = {
        name: context.get_split_data(name) for name in STAGE6_SPLIT_NAMES
    }
    directional_by_split: dict[Stage6SplitName, npt.NDArray[np.float64]] = {}
    encoded_by_split = {}
    base_eligible_by_split: dict[Stage6SplitName, npt.NDArray[np.bool_]] = {}
    for split_name in STAGE6_SPLIT_NAMES:
        split = split_sources[split_name]
        date_count = int(split.rebalance_dates.size)
        stock_count = int(context.stocks.size)
        directional_by_split[split_name] = np.full(
            (date_count, stock_count, len(mapping)), np.nan, dtype=np.float64
        )
        encoded_by_split[split_name] = encode_industry_panel(
            split.industry_labels, (date_count, stock_count)
        )
        base_eligible_by_split[split_name] = strategy_matrix_base_eligibility(
            split.universe_mask,
            encoded_by_split[split_name],
        )

    for factor_index, (feature, expression) in enumerate(zip(mapping, expressions)):
        raw = interpreter.evaluate(expression)
        for split_name in STAGE6_SPLIT_NAMES:
            split = split_sources[split_name]
            compact_raw = np.asarray(raw)[split.global_rebalance_rows]
            cleaned = clean_candidate_factor_cross_sections(
                compact_raw,
                None,
                split.universe_mask,
                config=cleaning_config,
                encoded_industries=encoded_by_split[split_name],
            )
            imputed = impute_strategy_factor_nonfinite(
                cleaned,
                base_eligible_by_split[split_name],
            )
            directional_by_split[split_name][:, :, factor_index] = (
                feature.train_direction * imputed
            )

    split_results: dict[Stage6SplitName, DevelopmentSplitMatrix] = {}
    for split_name in STAGE6_SPLIT_NAMES:
        split = split_sources[split_name]
        directional = directional_by_split[split_name]

        base_eligible = base_eligible_by_split[split_name]
        date_positions, stock_positions = np.nonzero(base_eligible)
        dates = _readonly(
            np.asarray(split.rebalance_dates[date_positions]).astype("datetime64[D]")
        )
        symbols = _readonly(np.asarray(context.stocks[stock_positions]).astype(str))
        values = _readonly(
            np.asarray(
                directional[date_positions, stock_positions, :], dtype=np.float64
            )
        )
        labels = _readonly(
            np.asarray(
                split.forward_returns[date_positions, stock_positions],
                dtype=np.float64,
            )
        )
        if values.ndim != 2 or values.shape[1] != len(mapping):
            raise DevelopmentFactorMatrixIntegrityError("feature matrix shape is invalid")
        if not np.isfinite(values).all():
            raise DevelopmentFactorMatrixIntegrityError("imputed feature matrix is nonfinite")
        feature_fingerprint = _feature_fingerprint(
            split=split_name,
            dates=dates,
            symbols=symbols,
            values=values,
            factor_pool_fingerprint=frozen_pool.baseline_factor_pool_fingerprint,
            aliases=aliases,
            structural_hashes=hashes,
        )
        label_fingerprint = _label_fingerprint(
            split=split_name,
            dates=dates,
            symbols=symbols,
            labels=labels,
            calendar_fingerprint=calendar_fingerprint,
        )
        split_results[split_name] = DevelopmentSplitMatrix(
            features=FeaturesOnlyFactorMatrix(
                split=split_name,
                dates=dates,
                symbols=symbols,
                values=values,
                factor_pool_fingerprint=frozen_pool.baseline_factor_pool_fingerprint,
                feature_aliases=aliases,
                ordered_structural_hashes=hashes,
                fingerprint=feature_fingerprint,
            ),
            forward_returns=labels,
            label_fingerprint=label_fingerprint,
            boundary=MappingProxyType(asdict(split.boundary)),
        )

    contract = _matrix_contract(context, cleaning_config, strategy_input)
    provenance = {
        "context_fingerprint": context.fingerprint,
        "calendar_fingerprint": calendar_fingerprint,
        "context_sources": dict(context.manifest.get("sources", {})),
        "context_splits": dict(context.manifest.get("splits", {})),
    }
    if strategy_input is not None:
        provenance["strategy_input_fingerprint"] = (
            strategy_input.strategy_input_fingerprint
        )
    deterministic = {
        "schema": DEVELOPMENT_FACTOR_MATRIX_SCHEMA,
        "version": DEVELOPMENT_FACTOR_MATRIX_VERSION,
        "factor_pool_fingerprint": frozen_pool.baseline_factor_pool_fingerprint,
        "feature_mapping": [item.to_dict() for item in mapping],
        "contract": contract,
        "provenance": provenance,
        "splits": {
            name: {
                "feature_fingerprint": split_results[name].features.fingerprint,
                "label_fingerprint": split_results[name].label_fingerprint,
                "feature_row_count": split_results[name].features.row_count,
                "label_finite_count": int(
                    np.isfinite(split_results[name].forward_returns).sum()
                ),
                "boundary": dict(split_results[name].boundary),
            }
            for name in STAGE6_SPLIT_NAMES
        },
    }
    if strategy_input is not None:
        deterministic["strategy_input_fingerprint"] = (
            strategy_input.strategy_input_fingerprint
        )
    return DevelopmentFactorMatrices(
        artifact_manifest_path=None,
        artifact_fingerprint=None,
        factor_pool_manifest_path=frozen_pool.manifest_path,
        factor_pool_fingerprint=frozen_pool.baseline_factor_pool_fingerprint,
        context_fingerprint=context.fingerprint,
        calendar_fingerprint=calendar_fingerprint,
        feature_mapping=mapping,
        splits=MappingProxyType(split_results),
        contract=_deep_freeze(contract),
        provenance=_deep_freeze(provenance),
        logical_fingerprint=_stable_hash(deterministic),
        strategy_input_manifest_path=(
            strategy_input.manifest_path if strategy_input is not None else None
        ),
        strategy_input_fingerprint=(
            strategy_input.strategy_input_fingerprint
            if strategy_input is not None
            else None
        ),
    )


def _artifact_payloads(
    matrices: DevelopmentFactorMatrices,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    mapping_payload = [item.to_dict() for item in matrices.feature_mapping]
    payloads: dict[str, bytes] = {FACTOR_MAPPING_FILENAME: _json_bytes(mapping_payload)}
    split_metadata: dict[str, Any] = {}
    for name in STAGE6_SPLIT_NAMES:
        split = matrices.splits[name]
        files = {
            "features": f"{name}_features.npy",
            "dates": f"{name}_dates.npy",
            "symbols": f"{name}_symbols.npy",
            "labels": f"{name}_forward_returns.npy",
        }
        payloads[files["features"]] = _npy_bytes(split.features.values)
        payloads[files["dates"]] = _npy_bytes(split.features.dates)
        payloads[files["symbols"]] = _npy_bytes(split.features.symbols)
        payloads[files["labels"]] = _npy_bytes(split.forward_returns)
        split_metadata[name] = {
            "feature_fingerprint": split.features.fingerprint,
            "label_fingerprint": split.label_fingerprint,
            "feature_row_count": split.features.row_count,
            "factor_count": split.features.factor_count,
            "label_finite_count": int(np.isfinite(split.forward_returns).sum()),
            "boundary": dict(split.boundary),
            "files": files,
        }
    return payloads, split_metadata


def _manifest_fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": manifest.get("schema"),
        "version": manifest.get("version"),
        "factor_pool_fingerprint": manifest.get("factor_pool_fingerprint"),
        "feature_mapping": manifest.get("feature_mapping"),
        "contract": manifest.get("contract"),
        "provenance": manifest.get("provenance"),
        "splits": manifest.get("splits"),
        "artifacts": manifest.get("artifacts"),
    }
    if manifest.get("strategy_input_fingerprint") is not None:
        payload["strategy_input_fingerprint"] = manifest.get(
            "strategy_input_fingerprint"
        )
    return payload


def _build_manifest(
    matrices: DevelopmentFactorMatrices,
    payloads: Mapping[str, bytes],
    split_metadata: Mapping[str, Any],
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": DEVELOPMENT_FACTOR_MATRIX_SCHEMA,
        "version": DEVELOPMENT_FACTOR_MATRIX_VERSION,
        "factor_pool_fingerprint": matrices.factor_pool_fingerprint,
        "feature_mapping": [item.to_dict() for item in matrices.feature_mapping],
        "contract": _deep_thaw(matrices.contract),
        "provenance": _deep_thaw(matrices.provenance),
        "splits": {name: dict(split_metadata[name]) for name in STAGE6_SPLIT_NAMES},
        "artifacts": {
            name: {"size_bytes": len(payload), "sha256": _sha256_bytes(payload)}
            for name, payload in sorted(payloads.items())
        },
        "upstream_manifest_paths": {
            "factor_pool_manifest": str(matrices.factor_pool_manifest_path.resolve())
        },
        "created_at_utc": created_at_utc,
        "created_at_excluded_from_fingerprint": True,
        "oos_status": OOS_UNTOUCHED,
    }
    if matrices.strategy_input_fingerprint is not None:
        if matrices.strategy_input_manifest_path is None:
            raise DevelopmentFactorMatrixIntegrityError(
                "Strategy Input manifest path is missing"
            )
        manifest["strategy_input_fingerprint"] = (
            matrices.strategy_input_fingerprint
        )
        manifest["upstream_manifest_paths"]["strategy_input_manifest"] = str(
            matrices.strategy_input_manifest_path.resolve()
        )
    manifest["development_factor_matrix_fingerprint"] = _stable_hash(
        _manifest_fingerprint_payload(manifest)
    )
    return manifest


def _load_array(path: Path) -> npt.NDArray[Any]:
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DevelopmentFactorMatrixIntegrityError(
            f"cannot load development matrix array: {path}"
        ) from error
    return _readonly(np.asarray(values))


def _verify_directory(
    manifest_path: Path,
    *,
    require_directory_identity: bool,
    verify_factor_pool: bool,
) -> DevelopmentFactorMatrices:
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != DEVELOPMENT_FACTOR_MATRIX_SCHEMA
        or manifest.get("version") != DEVELOPMENT_FACTOR_MATRIX_VERSION
        or manifest.get("oos_status") != OOS_UNTOUCHED
    ):
        raise DevelopmentFactorMatrixIntegrityError(
            "development matrix schema, version, or OOS lock mismatch"
        )
    fingerprint = str(manifest.get("development_factor_matrix_fingerprint"))
    if _stable_hash(_manifest_fingerprint_payload(manifest)) != fingerprint:
        raise DevelopmentFactorMatrixIntegrityError("development matrix fingerprint mismatch")
    if require_directory_identity and manifest_path.parent.name != fingerprint:
        raise DevelopmentFactorMatrixIntegrityError("development matrix directory mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DevelopmentFactorMatrixIntegrityError("development artifacts metadata missing")
    for filename, metadata in artifacts.items():
        path = manifest_path.parent / str(filename)
        if (
            not isinstance(metadata, Mapping)
            or not path.is_file()
            or path.stat().st_size != int(metadata.get("size_bytes", -1))
            or _sha256_file(path) != metadata.get("sha256")
        ):
            raise DevelopmentFactorMatrixIntegrityError(
                f"development artifact changed: {filename}"
            )

    mapping_raw = json.loads(
        (manifest_path.parent / FACTOR_MAPPING_FILENAME).read_text(encoding="utf-8")
    )
    if not isinstance(mapping_raw, list) or mapping_raw != manifest.get("feature_mapping"):
        raise DevelopmentFactorMatrixIntegrityError("factor mapping changed")
    try:
        mapping = tuple(
            FrozenFactorFeature(
                alias=str(row["alias"]),
                factor_index=int(row["factor_index"]),
                structural_hash=str(row["structural_hash"]),
                formula=str(row["formula"]),
                prefix_token_ids=tuple(int(value) for value in row["prefix_token_ids"]),
                node_count=int(row["node_count"]),
                depth=int(row["depth"]),
                train_direction=int(row["train_direction"]),
            )
            for row in mapping_raw
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DevelopmentFactorMatrixIntegrityError("factor mapping is invalid") from error
    if (
        not mapping
        or tuple(item.factor_index for item in mapping) != tuple(range(len(mapping)))
        or len({item.structural_hash for item in mapping}) != len(mapping)
    ):
        raise DevelopmentFactorMatrixIntegrityError("factor mapping order is invalid")
    pool_path_value = manifest.get("upstream_manifest_paths", {}).get(
        "factor_pool_manifest"
    )
    if not isinstance(pool_path_value, str) or not pool_path_value:
        raise DevelopmentFactorMatrixIntegrityError("factor pool manifest path missing")
    pool_path = Path(pool_path_value).resolve()
    strategy_input_path: Path | None = None
    strategy_input_fingerprint = manifest.get("strategy_input_fingerprint")
    if strategy_input_fingerprint is not None:
        strategy_input_path_value = manifest.get("upstream_manifest_paths", {}).get(
            "strategy_input_manifest"
        )
        if not isinstance(strategy_input_path_value, str) or not strategy_input_path_value:
            raise DevelopmentFactorMatrixIntegrityError(
                "Strategy Input manifest path missing"
            )
        strategy_input_path = Path(strategy_input_path_value).resolve()
    pool = load_verified_baseline_factor_pool(pool_path) if verify_factor_pool else None
    try:
        action_registry = action_registry_for_vocabulary(
            pool.manifest.get("vocabulary") if pool is not None else None
        )
    except ValueError as error:
        raise DevelopmentFactorMatrixIntegrityError(
            "frozen factor pool vocabulary is invalid"
        ) from error
    for item in mapping:
        _verify_expression_identity(item, action_registry)

    if pool is not None:
        expected_hashes = pool.ordered_structural_hashes
        expected_directions = pool.frozen_train_directions
        if strategy_input_path is not None:
            strategy_input = load_verified_strategy_input(strategy_input_path)
            if (
                strategy_input.strategy_input_fingerprint
                != strategy_input_fingerprint
                or strategy_input.factor_pool_fingerprint
                != pool.baseline_factor_pool_fingerprint
                or strategy_input.factor_pool_manifest_path.resolve()
                != pool.manifest_path.resolve()
            ):
                raise DevelopmentFactorMatrixIntegrityError(
                    "Development Strategy Input authority mismatch"
                )
            expected_hashes = strategy_input.ordered_structural_hashes
            expected_directions = strategy_input.frozen_train_directions
        if (
            pool.baseline_factor_pool_fingerprint
            != manifest.get("factor_pool_fingerprint")
            or expected_hashes != tuple(item.structural_hash for item in mapping)
            or expected_directions != tuple(item.train_direction for item in mapping)
        ):
            raise DevelopmentFactorMatrixIntegrityError(
                "development matrix differs from verified factor pool"
            )

    aliases = tuple(item.alias for item in mapping)
    hashes = tuple(item.structural_hash for item in mapping)
    split_results: dict[Stage6SplitName, DevelopmentSplitMatrix] = {}
    split_manifest = manifest.get("splits")
    if not isinstance(split_manifest, Mapping) or set(split_manifest) != set(
        STAGE6_SPLIT_NAMES
    ):
        raise DevelopmentFactorMatrixIntegrityError("development split metadata mismatch")
    calendar_fingerprint = str(manifest.get("provenance", {}).get("calendar_fingerprint"))
    for name in STAGE6_SPLIT_NAMES:
        metadata = split_manifest[name]
        if not isinstance(metadata, Mapping):
            raise DevelopmentFactorMatrixIntegrityError(f"invalid split metadata: {name}")
        files = metadata.get("files")
        if not isinstance(files, Mapping):
            raise DevelopmentFactorMatrixIntegrityError(f"split files missing: {name}")
        values = np.asarray(
            _load_array(manifest_path.parent / str(files["features"])),
            dtype=np.float64,
        )
        dates = np.asarray(
            _load_array(manifest_path.parent / str(files["dates"])),
        ).astype("datetime64[D]")
        symbols = np.asarray(
            _load_array(manifest_path.parent / str(files["symbols"])),
        ).astype(str)
        labels = np.asarray(
            _load_array(manifest_path.parent / str(files["labels"])),
            dtype=np.float64,
        )
        for array in (values, dates, symbols, labels):
            _readonly(array)
        if (
            values.ndim != 2
            or values.shape != (dates.size, len(mapping))
            or symbols.shape != dates.shape
            or labels.shape != dates.shape
            or not np.isfinite(values).all()
        ):
            raise DevelopmentFactorMatrixIntegrityError(f"split array mismatch: {name}")
        feature_fingerprint = _feature_fingerprint(
            split=name,
            dates=dates,
            symbols=symbols,
            values=values,
            factor_pool_fingerprint=str(manifest["factor_pool_fingerprint"]),
            aliases=aliases,
            structural_hashes=hashes,
        )
        label_fingerprint = _label_fingerprint(
            split=name,
            dates=dates,
            symbols=symbols,
            labels=labels,
            calendar_fingerprint=calendar_fingerprint,
        )
        if (
            feature_fingerprint != metadata.get("feature_fingerprint")
            or label_fingerprint != metadata.get("label_fingerprint")
            or values.shape[0] != int(metadata.get("feature_row_count", -1))
            or len(mapping) != int(metadata.get("factor_count", -1))
            or int(np.isfinite(labels).sum())
            != int(metadata.get("label_finite_count", -1))
        ):
            raise DevelopmentFactorMatrixIntegrityError(
                f"split fingerprint or count mismatch: {name}"
            )
        split_results[name] = DevelopmentSplitMatrix(
            features=FeaturesOnlyFactorMatrix(
                split=name,
                dates=dates,
                symbols=symbols,
                values=values,
                factor_pool_fingerprint=str(manifest["factor_pool_fingerprint"]),
                feature_aliases=aliases,
                ordered_structural_hashes=hashes,
                fingerprint=feature_fingerprint,
            ),
            forward_returns=labels,
            label_fingerprint=label_fingerprint,
            boundary=_deep_freeze(dict(metadata.get("boundary", {}))),
        )

    deterministic = {
        "schema": manifest["schema"],
        "version": manifest["version"],
        "factor_pool_fingerprint": manifest["factor_pool_fingerprint"],
        "feature_mapping": manifest["feature_mapping"],
        "contract": manifest["contract"],
        "provenance": manifest["provenance"],
        "splits": {
            name: {
                "feature_fingerprint": split_results[name].features.fingerprint,
                "label_fingerprint": split_results[name].label_fingerprint,
                "feature_row_count": split_results[name].features.row_count,
                "label_finite_count": int(
                    np.isfinite(split_results[name].forward_returns).sum()
                ),
                "boundary": dict(split_results[name].boundary),
            }
            for name in STAGE6_SPLIT_NAMES
        },
    }
    if strategy_input_fingerprint is not None:
        deterministic["strategy_input_fingerprint"] = strategy_input_fingerprint
    built_fingerprint = _stable_hash(deterministic)
    # The directory identity additionally binds exact serialized artifact bytes.
    # The logical identity binds the reconstructed matrix semantics.
    if built_fingerprint != manifest.get("logical_matrix_fingerprint"):
        raise DevelopmentFactorMatrixIntegrityError(
            "logical development matrix fingerprint mismatch"
        )
    return DevelopmentFactorMatrices(
        artifact_manifest_path=manifest_path,
        artifact_fingerprint=fingerprint,
        factor_pool_manifest_path=pool_path,
        factor_pool_fingerprint=str(manifest["factor_pool_fingerprint"]),
        context_fingerprint=str(manifest.get("provenance", {}).get("context_fingerprint")),
        calendar_fingerprint=calendar_fingerprint,
        feature_mapping=mapping,
        splits=MappingProxyType(split_results),
        contract=_deep_freeze(dict(manifest["contract"])),
        provenance=_deep_freeze(dict(manifest["provenance"])),
        logical_fingerprint=str(manifest.get("logical_matrix_fingerprint")),
        strategy_input_manifest_path=strategy_input_path,
        strategy_input_fingerprint=(
            str(strategy_input_fingerprint)
            if strategy_input_fingerprint is not None
            else None
        ),
    )


def freeze_development_factor_matrices(
    matrices: DevelopmentFactorMatrices,
    runs_root: str | Path,
) -> DevelopmentFactorMatrixArtifact:
    """Publish a content-addressed development matrix artifact atomically."""

    if not isinstance(matrices, DevelopmentFactorMatrices):
        raise TypeError("matrices must be DevelopmentFactorMatrices")
    payloads, split_metadata = _artifact_payloads(matrices)
    manifest = _build_manifest(
        matrices,
        payloads,
        split_metadata,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    manifest["logical_matrix_fingerprint"] = matrices.logical_fingerprint
    fingerprint = str(manifest["development_factor_matrix_fingerprint"])
    root = Path(runs_root).resolve() / "development_factor_matrices"
    target = root / fingerprint
    manifest_path = target / DEVELOPMENT_FACTOR_MATRIX_MANIFEST_FILENAME
    if target.exists():
        if not manifest_path.is_file():
            raise DevelopmentFactorMatrixIntegrityError(
                "development matrix target exists without manifest"
            )
        verified = _verify_directory(
            manifest_path,
            require_directory_identity=True,
            verify_factor_pool=True,
        )
        if verified.fingerprint != fingerprint:
            raise DevelopmentFactorMatrixIntegrityError(
                "development matrix target conflicts with requested artifact"
            )
        return DevelopmentFactorMatrixArtifact(manifest_path, fingerprint, True)

    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.tmp-", dir=root))
    try:
        for filename, payload in payloads.items():
            (temporary / filename).write_bytes(payload)
        temporary_manifest = temporary / DEVELOPMENT_FACTOR_MATRIX_MANIFEST_FILENAME
        temporary_manifest.write_bytes(_json_bytes(manifest))
        _verify_directory(
            temporary_manifest,
            require_directory_identity=False,
            verify_factor_pool=True,
        )
        os.replace(temporary, target)
        _verify_directory(
            manifest_path,
            require_directory_identity=True,
            verify_factor_pool=True,
        )
        return DevelopmentFactorMatrixArtifact(manifest_path, fingerprint, False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_verified_development_factor_matrices(
    manifest_path: str | Path,
) -> DevelopmentFactorMatrices:
    """Load matrices after file, fingerprint, and factor-pool verification."""

    return _verify_directory(
        Path(manifest_path).resolve(),
        require_directory_identity=True,
        verify_factor_pool=True,
    )


__all__ = [
    "DEVELOPMENT_FACTOR_MATRIX_MANIFEST_FILENAME",
    "DEVELOPMENT_FACTOR_MATRIX_SCHEMA",
    "DEVELOPMENT_FACTOR_MATRIX_VERSION",
    "DevelopmentFactorMatrices",
    "DevelopmentFactorMatrixArtifact",
    "DevelopmentFactorMatrixIntegrityError",
    "DevelopmentSplitMatrix",
    "FeaturesOnlyFactorMatrix",
    "FrozenFactorFeature",
    "build_development_factor_matrices",
    "freeze_development_factor_matrices",
    "load_verified_development_factor_matrices",
]
