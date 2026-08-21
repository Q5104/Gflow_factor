"""Baseline strategy definitions and immutable Strategy Bundle freeze.

Only Train/Validation development matrices are accepted while fitting.  The
unified scoring API consumes a verified frozen strategy plus a features-only
matrix and intentionally has no label argument.
"""

from __future__ import annotations

import hashlib
import io
import inspect
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from .baseline_factor_pool import (
    OOS_UNTOUCHED,
    VerifiedFrozenBaselineFactorPool,
    load_verified_baseline_factor_pool,
)
from .development_factor_matrix import (
    DevelopmentFactorMatrices,
    FeaturesOnlyFactorMatrix,
    STRATEGY_MATRIX_MISSING_CONTRACT,
    load_verified_development_factor_matrices,
)
from .strategy_input import load_verified_strategy_input
from .rolling_icir import (
    RollingICIRConfig,
    estimate_rolling_weights,
    periodic_ic_matrix,
)
from .stage6_evaluation import STAGE6_SPLIT_NAMES, _stable_hash


STRATEGY_IDS = ("equal_weight", "fixed_icir", "lightgbm")
StrategyId = Literal["equal_weight", "fixed_icir", "lightgbm"]
STRATEGY_BUNDLE_MANIFEST_SCHEMA = "factor_gfn.frozen_strategy_bundle_manifest.v1"
STRATEGY_BUNDLE_VERSION = "three-strategy-bundle-rolling-icir-v2"
STRATEGY_BUNDLE_MANIFEST_FILENAME = "strategy_bundle_manifest.json"
EQUAL_WEIGHT_FILENAME = "equal_weight.json"
FIXED_ICIR_FILENAME = "fixed_icir.json"
LIGHTGBM_METADATA_FILENAME = "lightgbm/metadata.json"
LIGHTGBM_MODEL_FILENAME = "lightgbm/model.txt"
LIGHTGBM_TRAIN_SCORES_FILENAME = "lightgbm/development_scores_train.npy"
LIGHTGBM_VALIDATION_SCORES_FILENAME = (
    "lightgbm/development_scores_validation.npy"
)
STRATEGY_OOS_LOCKED = "locked_not_loaded_not_evaluated"
ICIR_EPSILON = 1e-12

LIGHTGBM_FIXED_PARAMS: Mapping[str, Any] = MappingProxyType(
    {
        "objective": "regression",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
)
LIGHTGBM_EVAL_METRIC = "l2"
LIGHTGBM_EARLY_STOPPING_PATIENCE = 50


class StrategyBundleIntegrityError(RuntimeError):
    """A strategy input, frozen artifact, or authority binding is invalid."""


class StrategyDependencyError(RuntimeError):
    """A required optional strategy dependency is unavailable."""


def _import_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise StrategyDependencyError(
            "Static LightGBM requires the project lightgbm dependency"
        ) from error
    return lgb


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


def _implementation_identity() -> dict[str, Any]:
    strategy_path = Path(__file__).resolve()
    matrix_path = strategy_path.with_name("development_factor_matrix.py")
    rolling_path = strategy_path.with_name("rolling_icir.py")
    return {
        "schema": "factor_gfn.strategy_implementation.v2",
        "static_strategy_bundle_sha256": _sha256_file(strategy_path),
        "development_factor_matrix_sha256": _sha256_file(matrix_path),
        "rolling_icir_sha256": _sha256_file(rolling_path),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrategyBundleIntegrityError(f"cannot read strategy JSON: {path}") from error
    if not isinstance(value, dict):
        raise StrategyBundleIntegrityError(f"strategy JSON must be an object: {path}")
    return value


@dataclass(frozen=True, slots=True)
class FrozenLinearStrategy:
    strategy_id: Literal["equal_weight", "fixed_icir"]
    factor_pool_fingerprint: str
    feature_aliases: tuple[str, ...]
    ordered_structural_hashes: tuple[str, ...]
    weights: tuple[float, ...]
    metadata: Mapping[str, Any]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class FrozenLightGBMStrategy:
    strategy_id: Literal["lightgbm"]
    factor_pool_fingerprint: str
    feature_aliases: tuple[str, ...]
    ordered_structural_hashes: tuple[str, ...]
    metadata: Mapping[str, Any]
    model_text: str
    fingerprint: str


FrozenStrategy: TypeAlias = FrozenLinearStrategy | FrozenLightGBMStrategy


@dataclass(frozen=True, slots=True)
class StrategyScores:
    strategy_id: StrategyId
    dates: npt.NDArray[np.datetime64]
    symbols: npt.NDArray[np.str_]
    strategy_scores: npt.NDArray[np.float64]
    feature_matrix_fingerprint: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class BuiltStaticStrategyBundle:
    factor_pool_manifest_path: Path
    development_matrix_manifest_path: Path
    factor_pool_fingerprint: str
    development_matrix_fingerprint: str
    feature_aliases: tuple[str, ...]
    ordered_structural_hashes: tuple[str, ...]
    frozen_directions: tuple[int, ...]
    strategies: Mapping[StrategyId, FrozenStrategy]
    lightgbm_development_scores: Mapping[str, npt.NDArray[np.float64]]
    shared_contract: Mapping[str, Any]
    implementation_identity: Mapping[str, Any]
    logical_fingerprint: str
    strategy_input_manifest_path: Path | None = None
    strategy_input_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class StrategyBundleArtifact:
    manifest_path: Path
    bundle_fingerprint: str
    reused_existing_artifact: bool


@dataclass(frozen=True, slots=True)
class VerifiedFrozenStrategyBundle:
    manifest_path: Path
    bundle_fingerprint: str
    factor_pool_fingerprint: str
    development_matrix_fingerprint: str
    feature_aliases: tuple[str, ...]
    ordered_structural_hashes: tuple[str, ...]
    frozen_directions: tuple[int, ...]
    strategies: Mapping[StrategyId, FrozenStrategy]
    manifest: Mapping[str, Any]
    oos_status: str
    strategy_input_manifest_path: Path | None = None
    strategy_input_fingerprint: str | None = None


def _common_identity(
    matrices: DevelopmentFactorMatrices,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    aliases = tuple(item.alias for item in matrices.feature_mapping)
    hashes = tuple(item.structural_hash for item in matrices.feature_mapping)
    directions = tuple(item.train_direction for item in matrices.feature_mapping)
    if not aliases or len(set(aliases)) != len(aliases) or len(set(hashes)) != len(hashes):
        raise StrategyBundleIntegrityError("development feature identity is invalid")
    for split_name in STAGE6_SPLIT_NAMES:
        features = matrices.splits[split_name].features
        if (
            features.factor_pool_fingerprint != matrices.factor_pool_fingerprint
            or features.feature_aliases != aliases
            or features.ordered_structural_hashes != hashes
            or features.factor_count != len(aliases)
            or not np.isfinite(features.values).all()
        ):
            raise StrategyBundleIntegrityError(
                f"development feature identity mismatch: {split_name}"
            )
    return aliases, hashes, directions


def _linear_payload(
    strategy_id: str,
    matrices: DevelopmentFactorMatrices,
    weights: tuple[float, ...],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    aliases, hashes, _ = _common_identity(matrices)
    return {
        "schema": "factor_gfn.frozen_linear_strategy.v1",
        "strategy_id": strategy_id,
        "factor_pool_fingerprint": matrices.factor_pool_fingerprint,
        "feature_aliases": list(aliases),
        "ordered_structural_hashes": list(hashes),
        "weights": list(weights),
        "metadata": _deep_thaw(metadata),
        "direction_contract": "frozen_directional_cleaned_features",
        "cleaning_contract_fingerprint": _stable_hash(
            dict(matrices.contract.get("cleaning", {}))
        ),
        "missing_contract": matrices.contract.get("missing"),
        "implementation_identity": _implementation_identity(),
    }


def build_equal_weight_strategy(
    matrices: DevelopmentFactorMatrices,
) -> FrozenLinearStrategy:
    aliases, hashes, _ = _common_identity(matrices)
    factor_count = len(aliases)
    weights = tuple(float(1.0 / factor_count) for _ in range(factor_count))
    metadata = {
        "K": factor_count,
        "weight_contract": "exact_1_over_K",
        "requires_development_label": False,
    }
    payload = _linear_payload("equal_weight", matrices, weights, metadata)
    return FrozenLinearStrategy(
        strategy_id="equal_weight",
        factor_pool_fingerprint=matrices.factor_pool_fingerprint,
        feature_aliases=aliases,
        ordered_structural_hashes=hashes,
        weights=weights,
        metadata=_deep_freeze(metadata),
        fingerprint=_stable_hash(payload),
    )


def _spearman(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    if left.size < 2 or np.ptp(left) <= ICIR_EPSILON or np.ptp(right) <= ICIR_EPSILON:
        return np.nan
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if not math.isfinite(denominator) or denominator <= ICIR_EPSILON:
        return np.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def _periodic_factor_ic(
    matrices: DevelopmentFactorMatrices,
    factor_index: int,
    *,
    min_cross_section_count: int,
) -> npt.NDArray[np.float64]:
    values: list[float] = []
    for split_name in STAGE6_SPLIT_NAMES:
        split = matrices.splits[split_name]
        dates = split.features.dates
        labels = split.forward_returns
        for date in np.unique(dates):
            mask = (dates == date) & np.isfinite(labels)
            if int(mask.sum()) < min_cross_section_count:
                continue
            values.append(
                _spearman(split.features.values[mask, factor_index], labels[mask])
            )
    return np.asarray(values, dtype=np.float64)


def build_fixed_icir_strategy(
    matrices: DevelopmentFactorMatrices,
    *,
    min_cross_section_count: int | None = None,
    epsilon: float = ICIR_EPSILON,
    rolling_config: RollingICIRConfig | None = None,
) -> FrozenLinearStrategy:
    """Build the causal rolling-ICIR seed stored under the legacy strategy id.

    ``fixed_icir`` remains the technical id so existing three-strategy artifact
    readers keep their key contract.  Its weights are only the first OOS
    weights; subsequent weights are updated causally by the OOS evaluator.
    """

    aliases, hashes, _ = _common_identity(matrices)
    if min_cross_section_count is None:
        min_cross_section_count = int(
            matrices.contract.get("evaluation_config", {}).get(
                "min_cross_section_count", 20
            )
        )
    if min_cross_section_count < 2 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("ICIR minimum count and epsilon are invalid")
    if rolling_config is not None and (
        rolling_config.min_cross_section_count != min_cross_section_count
        or rolling_config.epsilon != epsilon
    ):
        raise ValueError("rolling configuration conflicts with ICIR arguments")

    development_dates = np.concatenate(
        [matrices.splits[name].features.dates for name in STAGE6_SPLIT_NAMES]
    )
    development_values = np.vstack(
        [matrices.splits[name].features.values for name in STAGE6_SPLIT_NAMES]
    )
    development_labels = np.concatenate(
        [matrices.splits[name].forward_returns for name in STAGE6_SPLIT_NAMES]
    )
    ic_dates, ic_values = periodic_ic_matrix(
        development_dates,
        development_values,
        development_labels,
        min_cross_section_count=min_cross_section_count,
        epsilon=epsilon,
    )
    if rolling_config is None:
        rolling_config = RollingICIRConfig(
            min_observations=min(100, max(2, int(ic_dates.size))),
            min_cross_section_count=min_cross_section_count,
            epsilon=epsilon,
        )
    seed_dates = ic_dates[-rolling_config.window_observations :]
    seed_ic = ic_values[-rolling_config.window_observations :]
    weights_array, rolling_diagnostics = estimate_rolling_weights(
        seed_ic,
        rolling_config,
    )
    raw_icir = np.asarray(rolling_diagnostics.pop("raw_icir"), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    for factor_index, alias in enumerate(aliases):
        finite = seed_ic[np.isfinite(seed_ic[:, factor_index]), factor_index]
        mean_ic = float(finite.mean()) if finite.size else np.nan
        std_ic = float(finite.std(ddof=1)) if finite.size >= 2 else np.nan
        diagnostics.append(
            {
                "alias": alias,
                "structural_hash": hashes[factor_index],
                "observation_count": int(finite.size),
                "mean_ic": _finite_or_none(mean_ic),
                "std_ic_ddof1": _finite_or_none(std_ic),
                "icir": _finite_or_none(raw_icir[factor_index]),
                "initial_weight": float(weights_array[factor_index]),
            }
        )
    weights = tuple(float(value) for value in weights_array)
    metadata = {
        "strategy_semantics": "causal_rolling_icir_replaces_fixed_icir",
        "technical_strategy_id": "fixed_icir",
        "display_name": "Rolling ICIR",
        "weight_contract": "initial_seed_then_causal_oos_updates",
        "development_splits": list(STAGE6_SPLIT_NAMES),
        "development_feature_fingerprints": {
            name: matrices.splits[name].features.fingerprint for name in STAGE6_SPLIT_NAMES
        },
        "development_label_fingerprints": {
            name: matrices.splits[name].label_fingerprint for name in STAGE6_SPLIT_NAMES
        },
        "calendar_fingerprint": matrices.calendar_fingerprint,
        "rank_ic": "periodic_cross_sectional_spearman",
        "split_observations": "latest_rolling_window_of_Train_and_Validation_dates",
        "min_cross_section_count": min_cross_section_count,
        "std_ddof": 1,
        "epsilon": epsilon,
        "positive_clipping": True,
        "rolling_config": rolling_config.to_dict(),
        "seed_ic_dates": seed_dates.astype(str).tolist(),
        "seed_ic_values": [
            [_finite_or_none(value) for value in row]
            for row in seed_ic
        ],
        "initial_weight_diagnostics": rolling_diagnostics,
        "per_factor": diagnostics,
        "fallback_status": bool(rolling_diagnostics["fallback_status"]),
        "fallback_reason": rolling_diagnostics["fallback_reason"],
    }
    payload = _linear_payload("fixed_icir", matrices, weights, metadata)
    return FrozenLinearStrategy(
        strategy_id="fixed_icir",
        factor_pool_fingerprint=matrices.factor_pool_fingerprint,
        feature_aliases=aliases,
        ordered_structural_hashes=hashes,
        weights=weights,
        metadata=_deep_freeze(metadata),
        fingerprint=_stable_hash(payload),
    )


def equal_date_sample_weights(dates: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return row weights whose total is identical for every distinct date."""

    date_values = np.asarray(dates).astype("datetime64[D]")
    if date_values.ndim != 1 or not date_values.size or np.isnat(date_values).any():
        raise ValueError("dates must be a non-empty finite one-dimensional array")
    unique_dates, inverse, counts = np.unique(
        date_values, return_inverse=True, return_counts=True
    )
    weights = date_values.size / (unique_dates.size * counts[inverse])
    return _readonly(np.asarray(weights, dtype=np.float64))


def _labelled_rows(
    matrices: DevelopmentFactorMatrices, split_name: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.datetime64]]:
    split = matrices.splits[split_name]  # type: ignore[index]
    valid = np.isfinite(split.forward_returns)
    if int(valid.sum()) < 2:
        raise StrategyBundleIntegrityError(
            f"{split_name} contains fewer than two finite development labels"
        )
    return (
        np.asarray(split.features.values[valid], dtype=np.float64),
        np.asarray(split.forward_returns[valid], dtype=np.float64),
        np.asarray(split.features.dates[valid]).astype("datetime64[D]"),
    )


def _lightgbm_payload(
    matrices: DevelopmentFactorMatrices,
    metadata: Mapping[str, Any],
    model_text: str,
) -> dict[str, Any]:
    aliases, hashes, _ = _common_identity(matrices)
    return {
        "schema": "factor_gfn.frozen_static_lightgbm_strategy.v1",
        "strategy_id": "lightgbm",
        "factor_pool_fingerprint": matrices.factor_pool_fingerprint,
        "feature_aliases": list(aliases),
        "ordered_structural_hashes": list(hashes),
        "metadata": _deep_thaw(metadata),
        "model_sha256": _sha256_bytes(model_text.encode("utf-8")),
        "implementation_identity": _implementation_identity(),
    }


def build_static_lightgbm_strategy(
    matrices: DevelopmentFactorMatrices,
) -> tuple[FrozenLightGBMStrategy, Mapping[str, npt.NDArray[np.float64]]]:
    """Fit one selection model, freeze best iteration, then refit Train+Validation."""

    lgb = _import_lightgbm()
    aliases, hashes, _ = _common_identity(matrices)
    x_train, y_train, train_dates = _labelled_rows(matrices, "train")
    x_validation, y_validation, validation_dates = _labelled_rows(
        matrices, "validation"
    )
    train_weights = equal_date_sample_weights(train_dates)
    validation_weights = equal_date_sample_weights(validation_dates)
    params = dict(LIGHTGBM_FIXED_PARAMS)
    selection_model = lgb.LGBMRegressor(**params)
    selection_model.fit(
        x_train,
        y_train,
        sample_weight=train_weights,
        eval_X=x_validation,
        eval_y=y_validation,
        eval_sample_weight=[validation_weights],
        eval_metric=LIGHTGBM_EVAL_METRIC,
        callbacks=[
            lgb.early_stopping(
                LIGHTGBM_EARLY_STOPPING_PATIENCE,
                verbose=False,
            )
        ],
    )
    best_iteration = int(selection_model.best_iteration_ or 0)
    if best_iteration < 1 or best_iteration > int(params["n_estimators"]):
        raise StrategyBundleIntegrityError("LightGBM best_iteration is invalid")
    selection_model_text = selection_model.booster_.model_to_string(
        num_iteration=best_iteration
    )
    development_scores = {
        name: _readonly(
            np.asarray(
                selection_model.predict(
                    matrices.splits[name].features.values,
                    num_iteration=best_iteration,
                ),
                dtype=np.float64,
            )
        )
        for name in STAGE6_SPLIT_NAMES
    }
    if any(not np.isfinite(values).all() for values in development_scores.values()):
        raise StrategyBundleIntegrityError("selection-model development score is nonfinite")

    x_combined = np.concatenate([x_train, x_validation], axis=0)
    y_combined = np.concatenate([y_train, y_validation], axis=0)
    combined_dates = np.concatenate([train_dates, validation_dates], axis=0)
    combined_weights = equal_date_sample_weights(combined_dates)
    final_params = {**params, "n_estimators": best_iteration}
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(x_combined, y_combined, sample_weight=combined_weights)
    final_model_text = final_model.booster_.model_to_string(
        num_iteration=best_iteration
    )
    if not final_model_text.strip():
        raise StrategyBundleIntegrityError("final LightGBM model serialization is empty")

    score_fingerprints = {
        name: _stable_hash(
            {
                "schema": "factor_gfn.lightgbm_development_scores.v1",
                "split": name,
                "source": "Train_only_selection_model_at_best_iteration",
                "feature_fingerprint": matrices.splits[name].features.fingerprint,
                "score_digest": _array_digest(development_scores[name]),
            }
        )
        for name in STAGE6_SPLIT_NAMES
    }
    metadata = {
        "feature_aliases": list(aliases),
        "ordered_structural_hashes": list(hashes),
        "direction_contract": "frozen_directional_cleaned_features",
        "missing_contract": matrices.contract.get("missing"),
        "train_matrix_fingerprint": matrices.splits["train"].features.fingerprint,
        "validation_matrix_fingerprint": matrices.splits[
            "validation"
        ].features.fingerprint,
        "development_label_fingerprints": {
            name: matrices.splits[name].label_fingerprint for name in STAGE6_SPLIT_NAMES
        },
        "sample_weight_contract": "equal_total_weight_per_rebalance_date",
        "library": "lightgbm",
        "library_version": str(lgb.__version__),
        "fixed_parameters": params,
        "fixed_parameter_fingerprint": _stable_hash(params),
        "eval_metric": LIGHTGBM_EVAL_METRIC,
        "early_stopping_patience": LIGHTGBM_EARLY_STOPPING_PATIENCE,
        "best_iteration": best_iteration,
        "selection_model_fingerprint": _sha256_bytes(
            selection_model_text.encode("utf-8")
        ),
        "development_score_fingerprints": score_fingerprints,
        "development_score_source": "selection_model_not_final_refit_model",
        "final_refit_splits": list(STAGE6_SPLIT_NAMES),
        "final_n_estimators": best_iteration,
        "final_model_fingerprint": _sha256_bytes(final_model_text.encode("utf-8")),
        "complex_tuning": "forbidden",
    }
    payload = _lightgbm_payload(matrices, metadata, final_model_text)
    strategy = FrozenLightGBMStrategy(
        strategy_id="lightgbm",
        factor_pool_fingerprint=matrices.factor_pool_fingerprint,
        feature_aliases=aliases,
        ordered_structural_hashes=hashes,
        metadata=_deep_freeze(metadata),
        model_text=final_model_text,
        fingerprint=_stable_hash(payload),
    )
    return strategy, MappingProxyType(development_scores)


def _validate_features_for_strategy(
    strategy: FrozenStrategy, features: FeaturesOnlyFactorMatrix
) -> None:
    if not isinstance(features, FeaturesOnlyFactorMatrix):
        raise TypeError("features must be FeaturesOnlyFactorMatrix")
    if (
        features.factor_pool_fingerprint != strategy.factor_pool_fingerprint
        or features.feature_aliases != strategy.feature_aliases
        or features.ordered_structural_hashes != strategy.ordered_structural_hashes
        or features.factor_count != len(strategy.feature_aliases)
        or features.dates.shape != features.symbols.shape
        or features.row_count != features.dates.size
        or not np.isfinite(features.values).all()
    ):
        raise StrategyBundleIntegrityError("features-only matrix identity mismatch")
    keys = list(zip(features.dates.astype(str).tolist(), features.symbols.tolist()))
    if len(set(keys)) != len(keys):
        raise StrategyBundleIntegrityError("features-only matrix contains duplicate keys")


def score_frozen_strategy(
    strategy: FrozenStrategy,
    features: FeaturesOnlyFactorMatrix,
) -> StrategyScores:
    """Score a features-only matrix.  Labels are deliberately absent from the API."""

    _validate_features_for_strategy(strategy, features)
    if isinstance(strategy, FrozenLinearStrategy):
        scores = features.values @ np.asarray(strategy.weights, dtype=np.float64)
    elif isinstance(strategy, FrozenLightGBMStrategy):
        lgb = _import_lightgbm()
        booster = lgb.Booster(model_str=strategy.model_text)
        scores = booster.predict(
            features.values,
            num_iteration=int(strategy.metadata["best_iteration"]),
        )
    else:
        raise TypeError("unsupported frozen strategy type")
    raw_scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort(
        (
            np.asarray(features.symbols).astype(str),
            np.asarray(features.dates).astype("datetime64[D]").astype(np.int64),
        )
    )
    score_values = _readonly(raw_scores[order])
    if score_values.shape != (features.row_count,) or not np.isfinite(score_values).all():
        raise StrategyBundleIntegrityError("strategy score output is invalid")
    fingerprint = _stable_hash(
        {
            "schema": "factor_gfn.strategy_scores.v1",
            "strategy_id": strategy.strategy_id,
            "strategy_fingerprint": strategy.fingerprint,
            "feature_matrix_fingerprint": features.fingerprint,
            "score_digest": _array_digest(score_values),
            "label_dependency": "none",
        }
    )
    return StrategyScores(
        strategy_id=strategy.strategy_id,
        dates=_readonly(np.asarray(features.dates)[order].copy()),
        symbols=_readonly(np.asarray(features.symbols)[order].copy()),
        strategy_scores=score_values,
        feature_matrix_fingerprint=features.fingerprint,
        fingerprint=fingerprint,
    )


def _shared_contract(matrices: DevelopmentFactorMatrices) -> dict[str, Any]:
    if matrices.contract.get("missing") != STRATEGY_MATRIX_MISSING_CONTRACT:
        raise StrategyBundleIntegrityError("development missing-value contract changed")
    contract = {
        "schema": "factor_gfn.static_strategy_bundle_contract.v1",
        "strategy_ids": list(STRATEGY_IDS),
        "factor_order": (
            "strategy_input.frozen_order_prefix_top100"
            if matrices.strategy_input_fingerprint is not None
            else "frozen_pool.ordered_structural_hashes"
        ),
        "direction": "frozen_train_direction_times_cleaned_factor",
        "cleaning_contract_fingerprint": _stable_hash(
            dict(matrices.contract.get("cleaning", {}))
        ),
        "missing": matrices.contract.get("missing"),
        "score_semantics": "higher_is_more_bullish",
        "score_api": "verified_strategy_plus_features_only_matrix",
        "score_label_argument": "absent",
        "development_splits": list(STAGE6_SPLIT_NAMES),
        "oos": "locked",
    }
    if matrices.strategy_input_fingerprint is not None:
        contract["strategy_input"] = {
            "strategy_input_fingerprint": matrices.strategy_input_fingerprint,
            "factor_pool_fingerprint": matrices.factor_pool_fingerprint,
            "policy": "frozen_order_prefix",
            "top_k": len(matrices.feature_mapping),
        }
    return contract


def build_static_strategy_bundle(
    frozen_pool: VerifiedFrozenBaselineFactorPool,
    matrices: DevelopmentFactorMatrices,
) -> BuiltStaticStrategyBundle:
    """Build all three development-only strategies; no Test accessor exists."""

    if not isinstance(frozen_pool, VerifiedFrozenBaselineFactorPool):
        raise TypeError("frozen_pool must be VerifiedFrozenBaselineFactorPool")
    if not isinstance(matrices, DevelopmentFactorMatrices):
        raise TypeError("matrices must be DevelopmentFactorMatrices")
    if matrices.artifact_manifest_path is None:
        raise StrategyBundleIntegrityError(
            "strategies require a verified frozen development matrix artifact"
        )
    aliases, hashes, directions = _common_identity(matrices)
    expected_hashes = frozen_pool.ordered_structural_hashes
    expected_directions = frozen_pool.frozen_train_directions
    if matrices.strategy_input_fingerprint is not None:
        if matrices.strategy_input_manifest_path is None:
            raise StrategyBundleIntegrityError("Strategy Input manifest path missing")
        strategy_input = load_verified_strategy_input(
            matrices.strategy_input_manifest_path
        )
        if (
            strategy_input.strategy_input_fingerprint
            != matrices.strategy_input_fingerprint
            or strategy_input.factor_pool_fingerprint
            != frozen_pool.baseline_factor_pool_fingerprint
            or strategy_input.factor_pool_manifest_path.resolve()
            != frozen_pool.manifest_path.resolve()
        ):
            raise StrategyBundleIntegrityError("Strategy Input authority mismatch")
        expected_hashes = strategy_input.ordered_structural_hashes
        expected_directions = strategy_input.frozen_train_directions
    if (
        frozen_pool.oos_status != OOS_UNTOUCHED
        or matrices.factor_pool_fingerprint
        != frozen_pool.baseline_factor_pool_fingerprint
        or hashes != expected_hashes
        or directions != expected_directions
    ):
        raise StrategyBundleIntegrityError(
            "development matrices differ from verified frozen factor pool"
        )
    equal = build_equal_weight_strategy(matrices)
    fixed = build_fixed_icir_strategy(matrices)
    lightgbm, development_scores = build_static_lightgbm_strategy(matrices)
    strategies: dict[StrategyId, FrozenStrategy] = {
        "equal_weight": equal,
        "fixed_icir": fixed,
        "lightgbm": lightgbm,
    }
    contract = _shared_contract(matrices)
    implementation = _implementation_identity()
    logical_payload = {
        "schema": STRATEGY_BUNDLE_MANIFEST_SCHEMA,
        "version": STRATEGY_BUNDLE_VERSION,
        "factor_pool_fingerprint": matrices.factor_pool_fingerprint,
        "development_matrix_fingerprint": matrices.fingerprint,
        "feature_aliases": list(aliases),
        "ordered_structural_hashes": list(hashes),
        "frozen_directions": list(directions),
        "shared_contract": contract,
        "strategy_fingerprints": {
            name: strategies[name].fingerprint for name in STRATEGY_IDS
        },
        "implementation_identity": implementation,
        "oos_status": STRATEGY_OOS_LOCKED,
    }
    if matrices.strategy_input_fingerprint is not None:
        logical_payload["strategy_input_fingerprint"] = (
            matrices.strategy_input_fingerprint
        )
    return BuiltStaticStrategyBundle(
        factor_pool_manifest_path=frozen_pool.manifest_path,
        development_matrix_manifest_path=matrices.artifact_manifest_path,
        factor_pool_fingerprint=matrices.factor_pool_fingerprint,
        development_matrix_fingerprint=matrices.fingerprint,
        feature_aliases=aliases,
        ordered_structural_hashes=hashes,
        frozen_directions=directions,
        strategies=MappingProxyType(strategies),
        lightgbm_development_scores=development_scores,
        shared_contract=_deep_freeze(contract),
        implementation_identity=_deep_freeze(implementation),
        logical_fingerprint=_stable_hash(logical_payload),
        strategy_input_manifest_path=matrices.strategy_input_manifest_path,
        strategy_input_fingerprint=matrices.strategy_input_fingerprint,
    )


def build_static_strategy_bundle_from_verified_artifacts(
    factor_pool_manifest_path: str | Path,
    development_matrix_manifest_path: str | Path,
) -> BuiltStaticStrategyBundle:
    """Production entry: both upstream inputs must pass their verified loaders."""

    pool = load_verified_baseline_factor_pool(factor_pool_manifest_path)
    matrices = load_verified_development_factor_matrices(
        development_matrix_manifest_path
    )
    built = build_static_strategy_bundle(pool, matrices)
    return BuiltStaticStrategyBundle(
        factor_pool_manifest_path=built.factor_pool_manifest_path,
        development_matrix_manifest_path=Path(development_matrix_manifest_path).resolve(),
        factor_pool_fingerprint=built.factor_pool_fingerprint,
        development_matrix_fingerprint=built.development_matrix_fingerprint,
        feature_aliases=built.feature_aliases,
        ordered_structural_hashes=built.ordered_structural_hashes,
        frozen_directions=built.frozen_directions,
        strategies=built.strategies,
        lightgbm_development_scores=built.lightgbm_development_scores,
        shared_contract=built.shared_contract,
        implementation_identity=built.implementation_identity,
        logical_fingerprint=built.logical_fingerprint,
        strategy_input_manifest_path=built.strategy_input_manifest_path,
        strategy_input_fingerprint=built.strategy_input_fingerprint,
    )


def _linear_artifact_payload(strategy: FrozenLinearStrategy) -> dict[str, Any]:
    return {
        "schema": "factor_gfn.frozen_linear_strategy_artifact.v1",
        "strategy_id": strategy.strategy_id,
        "factor_pool_fingerprint": strategy.factor_pool_fingerprint,
        "feature_aliases": list(strategy.feature_aliases),
        "ordered_structural_hashes": list(strategy.ordered_structural_hashes),
        "weights": list(strategy.weights),
        "metadata": _deep_thaw(strategy.metadata),
        "strategy_fingerprint": strategy.fingerprint,
    }


def _lightgbm_artifact_payload(strategy: FrozenLightGBMStrategy) -> dict[str, Any]:
    return {
        "schema": "factor_gfn.frozen_static_lightgbm_artifact.v1",
        "strategy_id": strategy.strategy_id,
        "factor_pool_fingerprint": strategy.factor_pool_fingerprint,
        "feature_aliases": list(strategy.feature_aliases),
        "ordered_structural_hashes": list(strategy.ordered_structural_hashes),
        "metadata": _deep_thaw(strategy.metadata),
        "model_sha256": _sha256_bytes(strategy.model_text.encode("utf-8")),
        "strategy_fingerprint": strategy.fingerprint,
    }


def _bundle_payloads(
    bundle: BuiltStaticStrategyBundle,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    equal = bundle.strategies["equal_weight"]
    fixed = bundle.strategies["fixed_icir"]
    lightgbm = bundle.strategies["lightgbm"]
    assert isinstance(equal, FrozenLinearStrategy)
    assert isinstance(fixed, FrozenLinearStrategy)
    assert isinstance(lightgbm, FrozenLightGBMStrategy)
    payloads = {
        EQUAL_WEIGHT_FILENAME: _json_bytes(_linear_artifact_payload(equal)),
        FIXED_ICIR_FILENAME: _json_bytes(_linear_artifact_payload(fixed)),
        LIGHTGBM_METADATA_FILENAME: _json_bytes(
            _lightgbm_artifact_payload(lightgbm)
        ),
        LIGHTGBM_MODEL_FILENAME: lightgbm.model_text.encode("utf-8"),
        LIGHTGBM_TRAIN_SCORES_FILENAME: _npy_bytes(
            bundle.lightgbm_development_scores["train"]
        ),
        LIGHTGBM_VALIDATION_SCORES_FILENAME: _npy_bytes(
            bundle.lightgbm_development_scores["validation"]
        ),
    }
    artifact_metadata = {
        filename: {"size_bytes": len(payload), "sha256": _sha256_bytes(payload)}
        for filename, payload in sorted(payloads.items())
    }
    return payloads, artifact_metadata


def _bundle_fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": manifest.get("schema"),
        "version": manifest.get("version"),
        "factor_pool_fingerprint": manifest.get("factor_pool_fingerprint"),
        "development_matrix_fingerprint": manifest.get(
            "development_matrix_fingerprint"
        ),
        "feature_aliases": manifest.get("feature_aliases"),
        "ordered_structural_hashes": manifest.get("ordered_structural_hashes"),
        "frozen_directions": manifest.get("frozen_directions"),
        "shared_contract": manifest.get("shared_contract"),
        "strategy_fingerprints": manifest.get("strategy_fingerprints"),
        "artifacts": manifest.get("artifacts"),
        "implementation_identity": manifest.get("implementation_identity"),
        "oos_status": manifest.get("oos_status"),
    }
    if manifest.get("strategy_input_fingerprint") is not None:
        payload["strategy_input_fingerprint"] = manifest.get(
            "strategy_input_fingerprint"
        )
    return payload


def _build_bundle_manifest(
    bundle: BuiltStaticStrategyBundle,
    artifact_metadata: Mapping[str, Any],
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": STRATEGY_BUNDLE_MANIFEST_SCHEMA,
        "version": STRATEGY_BUNDLE_VERSION,
        "factor_pool_fingerprint": bundle.factor_pool_fingerprint,
        "development_matrix_fingerprint": bundle.development_matrix_fingerprint,
        "feature_aliases": list(bundle.feature_aliases),
        "ordered_structural_hashes": list(bundle.ordered_structural_hashes),
        "frozen_directions": list(bundle.frozen_directions),
        "shared_contract": _deep_thaw(bundle.shared_contract),
        "strategy_fingerprints": {
            name: bundle.strategies[name].fingerprint for name in STRATEGY_IDS
        },
        "artifacts": dict(artifact_metadata),
        "implementation_identity": _deep_thaw(bundle.implementation_identity),
        "logical_bundle_fingerprint": bundle.logical_fingerprint,
        "upstream_manifest_paths": {
            "factor_pool_manifest": str(bundle.factor_pool_manifest_path.resolve()),
            "development_matrix_manifest": str(
                bundle.development_matrix_manifest_path.resolve()
            ),
        },
        "oos_status": STRATEGY_OOS_LOCKED,
        "created_at_utc": created_at_utc,
        "created_at_excluded_from_fingerprint": True,
    }
    if bundle.strategy_input_fingerprint is not None:
        if bundle.strategy_input_manifest_path is None:
            raise StrategyBundleIntegrityError("Strategy Input manifest path missing")
        manifest["strategy_input_fingerprint"] = bundle.strategy_input_fingerprint
        manifest["upstream_manifest_paths"]["strategy_input_manifest"] = str(
            bundle.strategy_input_manifest_path.resolve()
        )
    manifest["strategy_bundle_fingerprint"] = _stable_hash(
        _bundle_fingerprint_payload(manifest)
    )
    return manifest


def _load_linear_strategy(
    path: Path,
    expected_id: str,
    *,
    cleaning_contract_fingerprint: str,
    missing_contract: str,
) -> FrozenLinearStrategy:
    payload = _read_json(path)
    if (
        payload.get("schema") != "factor_gfn.frozen_linear_strategy_artifact.v1"
        or payload.get("strategy_id") != expected_id
    ):
        raise StrategyBundleIntegrityError(f"linear strategy schema mismatch: {expected_id}")
    try:
        weights = tuple(float(value) for value in payload["weights"])
        strategy = FrozenLinearStrategy(
            strategy_id=expected_id,  # type: ignore[arg-type]
            factor_pool_fingerprint=str(payload["factor_pool_fingerprint"]),
            feature_aliases=tuple(str(value) for value in payload["feature_aliases"]),
            ordered_structural_hashes=tuple(
                str(value) for value in payload["ordered_structural_hashes"]
            ),
            weights=weights,
            metadata=_deep_freeze(dict(payload["metadata"])),
            fingerprint=str(payload["strategy_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise StrategyBundleIntegrityError(f"linear strategy invalid: {expected_id}") from error
    if (
        not weights
        or any(not math.isfinite(value) or value < 0 for value in weights)
        or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise StrategyBundleIntegrityError(f"linear weights invalid: {expected_id}")
    if expected_id == "equal_weight" and any(
        not math.isclose(value, 1.0 / len(weights), rel_tol=0.0, abs_tol=1e-15)
        for value in weights
    ):
        raise StrategyBundleIntegrityError("equal strategy weights are not exact 1/K")
    logical_payload = {
        "schema": "factor_gfn.frozen_linear_strategy.v1",
        "strategy_id": strategy.strategy_id,
        "factor_pool_fingerprint": strategy.factor_pool_fingerprint,
        "feature_aliases": list(strategy.feature_aliases),
        "ordered_structural_hashes": list(strategy.ordered_structural_hashes),
        "weights": list(strategy.weights),
        "metadata": _deep_thaw(strategy.metadata),
        "direction_contract": "frozen_directional_cleaned_features",
        "cleaning_contract_fingerprint": cleaning_contract_fingerprint,
        "missing_contract": missing_contract,
        "implementation_identity": _implementation_identity(),
    }
    if _stable_hash(logical_payload) != strategy.fingerprint:
        raise StrategyBundleIntegrityError(
            f"linear strategy fingerprint mismatch: {expected_id}"
        )
    return strategy


def _verify_bundle_directory(
    manifest_path: Path,
    *,
    require_directory_identity: bool,
    verify_upstreams: bool,
) -> VerifiedFrozenStrategyBundle:
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != STRATEGY_BUNDLE_MANIFEST_SCHEMA
        or manifest.get("version") != STRATEGY_BUNDLE_VERSION
        or manifest.get("oos_status") != STRATEGY_OOS_LOCKED
    ):
        raise StrategyBundleIntegrityError("strategy bundle schema, version, or OOS lock mismatch")
    fingerprint = str(manifest.get("strategy_bundle_fingerprint"))
    if _stable_hash(_bundle_fingerprint_payload(manifest)) != fingerprint:
        raise StrategyBundleIntegrityError("strategy bundle fingerprint mismatch")
    if require_directory_identity and manifest_path.parent.name != fingerprint:
        raise StrategyBundleIntegrityError("strategy bundle directory identity mismatch")
    if manifest.get("implementation_identity") != _implementation_identity():
        raise StrategyBundleIntegrityError("strategy implementation identity changed")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StrategyBundleIntegrityError("strategy artifact metadata missing")
    expected_files = {
        EQUAL_WEIGHT_FILENAME,
        FIXED_ICIR_FILENAME,
        LIGHTGBM_METADATA_FILENAME,
        LIGHTGBM_MODEL_FILENAME,
        LIGHTGBM_TRAIN_SCORES_FILENAME,
        LIGHTGBM_VALIDATION_SCORES_FILENAME,
    }
    if set(artifacts) != expected_files:
        raise StrategyBundleIntegrityError("strategy artifact inventory mismatch")
    for filename, metadata in artifacts.items():
        path = manifest_path.parent / str(filename)
        if (
            not isinstance(metadata, Mapping)
            or not path.is_file()
            or path.stat().st_size != int(metadata.get("size_bytes", -1))
            or _sha256_file(path) != metadata.get("sha256")
        ):
            raise StrategyBundleIntegrityError(f"strategy artifact changed: {filename}")

    shared_contract = manifest.get("shared_contract")
    if not isinstance(shared_contract, Mapping):
        raise StrategyBundleIntegrityError("shared strategy contract missing")
    cleaning_fingerprint = str(
        shared_contract.get("cleaning_contract_fingerprint", "")
    )
    missing_contract = str(shared_contract.get("missing", ""))
    equal = _load_linear_strategy(
        manifest_path.parent / EQUAL_WEIGHT_FILENAME,
        "equal_weight",
        cleaning_contract_fingerprint=cleaning_fingerprint,
        missing_contract=missing_contract,
    )
    fixed = _load_linear_strategy(
        manifest_path.parent / FIXED_ICIR_FILENAME,
        "fixed_icir",
        cleaning_contract_fingerprint=cleaning_fingerprint,
        missing_contract=missing_contract,
    )
    lightgbm_payload = _read_json(manifest_path.parent / LIGHTGBM_METADATA_FILENAME)
    model_text = (manifest_path.parent / LIGHTGBM_MODEL_FILENAME).read_text(
        encoding="utf-8"
    )
    if (
        lightgbm_payload.get("schema")
        != "factor_gfn.frozen_static_lightgbm_artifact.v1"
        or lightgbm_payload.get("strategy_id") != "lightgbm"
        or _sha256_bytes(model_text.encode("utf-8"))
        != lightgbm_payload.get("model_sha256")
    ):
        raise StrategyBundleIntegrityError("LightGBM artifact or model digest mismatch")
    metadata = lightgbm_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise StrategyBundleIntegrityError("LightGBM metadata missing")
    best_iteration = int(metadata.get("best_iteration", 0))
    if (
        best_iteration < 1
        or int(metadata.get("final_n_estimators", 0)) != best_iteration
        or metadata.get("fixed_parameters") != dict(LIGHTGBM_FIXED_PARAMS)
        or metadata.get("fixed_parameter_fingerprint")
        != _stable_hash(dict(LIGHTGBM_FIXED_PARAMS))
        or metadata.get("development_score_source")
        != "selection_model_not_final_refit_model"
    ):
        raise StrategyBundleIntegrityError("LightGBM frozen training contract changed")
    lightgbm = FrozenLightGBMStrategy(
        strategy_id="lightgbm",
        factor_pool_fingerprint=str(lightgbm_payload["factor_pool_fingerprint"]),
        feature_aliases=tuple(str(v) for v in lightgbm_payload["feature_aliases"]),
        ordered_structural_hashes=tuple(
            str(v) for v in lightgbm_payload["ordered_structural_hashes"]
        ),
        metadata=_deep_freeze(dict(metadata)),
        model_text=model_text,
        fingerprint=str(lightgbm_payload["strategy_fingerprint"]),
    )
    lightgbm_logical_payload = {
        "schema": "factor_gfn.frozen_static_lightgbm_strategy.v1",
        "strategy_id": "lightgbm",
        "factor_pool_fingerprint": lightgbm.factor_pool_fingerprint,
        "feature_aliases": list(lightgbm.feature_aliases),
        "ordered_structural_hashes": list(lightgbm.ordered_structural_hashes),
        "metadata": _deep_thaw(lightgbm.metadata),
        "model_sha256": _sha256_bytes(model_text.encode("utf-8")),
        "implementation_identity": _implementation_identity(),
    }
    if _stable_hash(lightgbm_logical_payload) != lightgbm.fingerprint:
        raise StrategyBundleIntegrityError("LightGBM strategy fingerprint mismatch")
    strategies: dict[StrategyId, FrozenStrategy] = {
        "equal_weight": equal,
        "fixed_icir": fixed,
        "lightgbm": lightgbm,
    }
    if {
        name: strategies[name].fingerprint for name in STRATEGY_IDS
    } != manifest.get("strategy_fingerprints"):
        raise StrategyBundleIntegrityError("strategy fingerprints differ from bundle")

    aliases = tuple(str(value) for value in manifest.get("feature_aliases", []))
    hashes = tuple(str(value) for value in manifest.get("ordered_structural_hashes", []))
    directions = tuple(int(value) for value in manifest.get("frozen_directions", []))
    if (
        not aliases
        or len(aliases) != len(hashes)
        or len(hashes) != len(directions)
        or any(
            strategy.factor_pool_fingerprint != manifest.get("factor_pool_fingerprint")
            or strategy.feature_aliases != aliases
            or strategy.ordered_structural_hashes != hashes
            for strategy in strategies.values()
        )
    ):
        raise StrategyBundleIntegrityError("shared strategy identity mismatch")
    lgb = _import_lightgbm()
    if str(metadata.get("library_version")) != str(lgb.__version__):
        raise StrategyBundleIntegrityError("LightGBM library version changed")
    try:
        booster = lgb.Booster(model_str=model_text)
    except Exception as error:
        raise StrategyBundleIntegrityError("LightGBM model cannot be loaded") from error
    if int(booster.num_feature()) != len(aliases):
        raise StrategyBundleIntegrityError("LightGBM feature count changed")

    logical_payload = {
        "schema": STRATEGY_BUNDLE_MANIFEST_SCHEMA,
        "version": STRATEGY_BUNDLE_VERSION,
        "factor_pool_fingerprint": manifest.get("factor_pool_fingerprint"),
        "development_matrix_fingerprint": manifest.get(
            "development_matrix_fingerprint"
        ),
        "feature_aliases": list(aliases),
        "ordered_structural_hashes": list(hashes),
        "frozen_directions": list(directions),
        "shared_contract": _deep_thaw(shared_contract),
        "strategy_fingerprints": {
            name: strategies[name].fingerprint for name in STRATEGY_IDS
        },
        "implementation_identity": _implementation_identity(),
        "oos_status": STRATEGY_OOS_LOCKED,
    }
    strategy_input_fingerprint = manifest.get("strategy_input_fingerprint")
    if strategy_input_fingerprint is not None:
        logical_payload["strategy_input_fingerprint"] = strategy_input_fingerprint
    if _stable_hash(logical_payload) != manifest.get("logical_bundle_fingerprint"):
        raise StrategyBundleIntegrityError("logical strategy bundle fingerprint mismatch")

    paths = manifest.get("upstream_manifest_paths")
    if not isinstance(paths, Mapping):
        raise StrategyBundleIntegrityError("strategy upstream paths missing")
    pool_path = Path(str(paths.get("factor_pool_manifest", ""))).resolve()
    matrix_path = Path(str(paths.get("development_matrix_manifest", ""))).resolve()
    strategy_input_path: Path | None = None
    if strategy_input_fingerprint is not None:
        strategy_input_path_value = paths.get("strategy_input_manifest")
        if not isinstance(strategy_input_path_value, str) or not strategy_input_path_value:
            raise StrategyBundleIntegrityError("Strategy Input upstream path missing")
        strategy_input_path = Path(strategy_input_path_value).resolve()
    if verify_upstreams:
        pool = load_verified_baseline_factor_pool(pool_path)
        matrices = load_verified_development_factor_matrices(matrix_path)
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
                raise StrategyBundleIntegrityError(
                    "Strategy Input differs from verified full pool"
                )
            expected_hashes = strategy_input.ordered_structural_hashes
            expected_directions = strategy_input.frozen_train_directions
        if (
            pool.baseline_factor_pool_fingerprint
            != manifest.get("factor_pool_fingerprint")
            or expected_hashes != hashes
            or expected_directions != directions
            or matrices.fingerprint != manifest.get("development_matrix_fingerprint")
            or matrices.factor_pool_fingerprint
            != pool.baseline_factor_pool_fingerprint
            or matrices.strategy_input_fingerprint
            != strategy_input_fingerprint
        ):
            raise StrategyBundleIntegrityError(
                "strategy bundle differs from verified upstream artifacts"
            )
        score_files = {
            "train": LIGHTGBM_TRAIN_SCORES_FILENAME,
            "validation": LIGHTGBM_VALIDATION_SCORES_FILENAME,
        }
        for name, filename in score_files.items():
            scores = np.asarray(
                np.load(manifest_path.parent / filename, allow_pickle=False),
                dtype=np.float64,
            )
            expected = matrices.splits[name].features.row_count
            score_fingerprint = _stable_hash(
                {
                    "schema": "factor_gfn.lightgbm_development_scores.v1",
                    "split": name,
                    "source": "Train_only_selection_model_at_best_iteration",
                    "feature_fingerprint": matrices.splits[name].features.fingerprint,
                    "score_digest": _array_digest(scores),
                }
            )
            if (
                scores.shape != (expected,)
                or not np.isfinite(scores).all()
                or score_fingerprint
                != metadata.get("development_score_fingerprints", {}).get(name)
            ):
                raise StrategyBundleIntegrityError(
                    f"LightGBM development scores changed: {name}"
                )

    return VerifiedFrozenStrategyBundle(
        manifest_path=manifest_path,
        bundle_fingerprint=fingerprint,
        factor_pool_fingerprint=str(manifest["factor_pool_fingerprint"]),
        development_matrix_fingerprint=str(
            manifest["development_matrix_fingerprint"]
        ),
        feature_aliases=aliases,
        ordered_structural_hashes=hashes,
        frozen_directions=directions,
        strategies=MappingProxyType(strategies),
        manifest=_deep_freeze(manifest),
        oos_status=STRATEGY_OOS_LOCKED,
        strategy_input_manifest_path=strategy_input_path,
        strategy_input_fingerprint=(
            str(strategy_input_fingerprint)
            if strategy_input_fingerprint is not None
            else None
        ),
    )


def freeze_static_strategy_bundle(
    bundle: BuiltStaticStrategyBundle,
    runs_root: str | Path,
) -> StrategyBundleArtifact:
    """Atomically publish the three-strategy bundle without granting OOS access."""

    if not isinstance(bundle, BuiltStaticStrategyBundle):
        raise TypeError("bundle must be BuiltStaticStrategyBundle")
    payloads, artifact_metadata = _bundle_payloads(bundle)
    manifest = _build_bundle_manifest(
        bundle,
        artifact_metadata,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    fingerprint = str(manifest["strategy_bundle_fingerprint"])
    root = Path(runs_root).resolve() / "baseline_strategy_bundles"
    target = root / fingerprint
    manifest_path = target / STRATEGY_BUNDLE_MANIFEST_FILENAME
    if target.exists():
        if not manifest_path.is_file():
            raise StrategyBundleIntegrityError(
                "strategy bundle target exists without manifest"
            )
        verified = _verify_bundle_directory(
            manifest_path,
            require_directory_identity=True,
            verify_upstreams=True,
        )
        if verified.bundle_fingerprint != fingerprint:
            raise StrategyBundleIntegrityError(
                "strategy bundle target conflicts with requested artifact"
            )
        return StrategyBundleArtifact(manifest_path, fingerprint, True)

    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.tmp-", dir=root))
    try:
        for filename, payload in payloads.items():
            path = temporary / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        temporary_manifest = temporary / STRATEGY_BUNDLE_MANIFEST_FILENAME
        temporary_manifest.write_bytes(_json_bytes(manifest))
        _verify_bundle_directory(
            temporary_manifest,
            require_directory_identity=False,
            verify_upstreams=True,
        )
        os.replace(temporary, target)
        _verify_bundle_directory(
            manifest_path,
            require_directory_identity=True,
            verify_upstreams=True,
        )
        return StrategyBundleArtifact(manifest_path, fingerprint, False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_verified_strategy_bundle(
    manifest_path: str | Path,
) -> VerifiedFrozenStrategyBundle:
    """Load a deeply read-only bundle after artifact and upstream verification."""

    return _verify_bundle_directory(
        Path(manifest_path).resolve(),
        require_directory_identity=True,
        verify_upstreams=True,
    )


def score_api_accepts_labels() -> bool:
    """Machine-checkable leakage guard used by focused tests."""

    parameters = inspect.signature(score_frozen_strategy).parameters
    return any("label" in name or "return" in name for name in parameters)


__all__ = [
    "EQUAL_WEIGHT_FILENAME",
    "FIXED_ICIR_FILENAME",
    "LIGHTGBM_EARLY_STOPPING_PATIENCE",
    "LIGHTGBM_FIXED_PARAMS",
    "LIGHTGBM_METADATA_FILENAME",
    "LIGHTGBM_MODEL_FILENAME",
    "STRATEGY_BUNDLE_MANIFEST_FILENAME",
    "STRATEGY_BUNDLE_MANIFEST_SCHEMA",
    "STRATEGY_BUNDLE_VERSION",
    "STRATEGY_IDS",
    "STRATEGY_OOS_LOCKED",
    "BuiltStaticStrategyBundle",
    "FrozenLightGBMStrategy",
    "FrozenLinearStrategy",
    "FrozenStrategy",
    "StrategyBundleArtifact",
    "StrategyBundleIntegrityError",
    "StrategyDependencyError",
    "StrategyScores",
    "VerifiedFrozenStrategyBundle",
    "build_equal_weight_strategy",
    "build_fixed_icir_strategy",
    "build_static_lightgbm_strategy",
    "build_static_strategy_bundle",
    "build_static_strategy_bundle_from_verified_artifacts",
    "equal_date_sample_weights",
    "freeze_static_strategy_bundle",
    "load_verified_strategy_bundle",
    "score_api_accepts_labels",
    "score_frozen_strategy",
]
