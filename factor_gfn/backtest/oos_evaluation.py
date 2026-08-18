"""Frozen-contract OOS evaluator and immutable evaluation artifact store."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .baseline_factor_pool import VerifiedFrozenBaselineFactorPool
from .development_factor_matrix import load_verified_development_factor_matrices
from .oos_authority import (
    LABEL_FORMULA,
    OOSAuthorityError,
    TestFactorMatrix,
    VerifiedTestLabels,
    VerifiedTestScoreArtifact,
)
from .stage6_evaluation import _stable_hash
from .static_strategy_bundle import (
    LIGHTGBM_TRAIN_SCORES_FILENAME,
    LIGHTGBM_VALIDATION_SCORES_FILENAME,
    STRATEGY_IDS,
    VerifiedFrozenStrategyBundle,
)


OOS_EVALUATION_SCHEMA = "factor_gfn.oos_baseline_evaluation.v1"
OOS_EVALUATION_VERSION = "static-three-strategy-oos-evaluation-v1"
OOS_EVALUATION_MANIFEST_FILENAME = "evaluation_manifest.json"
PERIODS_PER_YEAR = 50.4
MIN_DECILE_STOCK_COUNT = 20
DDOF = 1

PARQUET_FILES = {
    "coverage_by_date": "coverage_by_date.parquet",
    "strategy_scores_test": "strategy_scores_test.parquet",
    "strategy_rank_ic_by_date": "strategy_rank_ic_by_date.parquet",
    "decile_assignments": "decile_assignments.parquet",
    "decile_returns": "decile_returns.parquet",
    "portfolio_returns": "portfolio_returns.parquet",
    "turnover_by_date": "turnover_by_date.parquet",
    "nav_series": "nav_series.parquet",
}
JSON_FILES = {
    "performance_summary": "performance_summary.json",
    "lightgbm_split_effectiveness": "lightgbm_split_effectiveness.json",
    "strategy_score_correlation": "strategy_score_correlation.json",
}


class OOSBaselineEvaluationIntegrityError(RuntimeError):
    """The evaluation contract, artifact, or upstream binding is invalid."""


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


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_deep_thaw(item) for item in value]
    return value


def _implementation_fingerprint() -> str:
    return _sha256_file(Path(__file__).resolve())


def spearman_average_ties(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return math.nan
    x_rank = rankdata(x[valid], method="average")
    y_rank = rankdata(y[valid], method="average")
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def deterministic_deciles(scores: Sequence[float], symbols: Sequence[str]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    names = np.asarray(symbols).astype(str)
    count = values.size
    if count < MIN_DECILE_STOCK_COUNT or names.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("a decile cross-section requires at least 20 finite keyed scores")
    order = np.lexsort((names, values))
    quotient, remainder = divmod(count, 10)
    groups = np.empty(count, dtype=np.int8)
    cursor = 0
    for group in range(1, 11):
        size = quotient + (1 if group <= remainder else 0)
        groups[order[cursor : cursor + size]] = group
        cursor += size
    if cursor != count or set(groups.tolist()) != set(range(1, 11)):
        raise ValueError("cannot form complete deterministic G1-G10")
    return groups


def compound_nav(returns: Sequence[float]) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    return np.cumprod(1.0 + values)


def geometric_annualized_return(
    returns: Sequence[float], periods_per_year: float = PERIODS_PER_YEAR
) -> float:
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return math.nan
    growth = float(np.prod(1.0 + values))
    if growth <= 0:
        return math.nan
    return growth ** (periods_per_year / values.size) - 1.0


def annualized_ratio(
    returns: Sequence[float], periods_per_year: float = PERIODS_PER_YEAR
) -> float:
    values = np.asarray(returns, dtype=np.float64)
    if values.size < 2:
        return math.nan
    standard_deviation = float(np.std(values, ddof=DDOF))
    if not math.isfinite(standard_deviation) or standard_deviation == 0:
        return math.nan
    return float(np.mean(values) / standard_deviation * math.sqrt(periods_per_year))


def max_drawdown_from_returns(returns: Sequence[float]) -> float:
    nav = np.concatenate(([1.0], compound_nav(returns)))
    running_maximum = np.maximum.accumulate(nav)
    return float(np.min(nav / running_maximum - 1.0))


def drift_adjusted_one_way_turnover(
    previous_symbols: Sequence[str],
    previous_returns: Sequence[float],
    current_symbols: Sequence[str],
) -> tuple[float, float, str]:
    """Return one-way turnover, replacement rate, and a fail-closed diagnostic."""

    previous = np.asarray(previous_symbols).astype(str)
    returns = np.asarray(previous_returns, dtype=np.float64)
    current = np.asarray(current_symbols).astype(str)
    if (
        previous.size == 0
        or current.size == 0
        or previous.shape != returns.shape
        or len(set(previous)) != previous.size
        or len(set(current)) != current.size
    ):
        return math.nan, math.nan, "invalid_holding_keys"
    if not np.isfinite(returns).all() or np.any(1.0 + returns <= 0):
        return math.nan, math.nan, "invalid_previous_holding_drift_return"
    denominator = float(np.sum(1.0 + returns))
    if not math.isfinite(denominator) or denominator <= 0:
        return math.nan, math.nan, "invalid_drift_denominator"
    drift = {symbol: float((1.0 + value) / denominator) for symbol, value in zip(previous, returns)}
    current_set = set(current)
    previous_set = set(previous)
    target_weight = 1.0 / current.size
    turnover = 0.5 * sum(
        abs((target_weight if symbol in current_set else 0.0) - drift.get(symbol, 0.0))
        for symbol in previous_set | current_set
    )
    replacement = 1.0 - len(previous_set & current_set) / current.size
    return float(turnover), float(replacement), "ok"


def summarize_rank_ic(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean_rank_ic": None,
            "ic_std": None,
            "icir": None,
            "ic_positive_ratio": None,
            "valid_ic_periods": 0,
        }
    standard_deviation = float(np.std(finite, ddof=DDOF)) if finite.size >= 2 else math.nan
    icir = float(np.mean(finite) / standard_deviation) if standard_deviation > 0 else math.nan
    return {
        "mean_rank_ic": float(np.mean(finite)),
        "ic_std": _finite_or_none(standard_deviation),
        "icir": _finite_or_none(icir),
        "ic_positive_ratio": float(np.mean(finite > 0)),
        "valid_ic_periods": int(finite.size),
    }


def _performance(values: np.ndarray, *, win_name: str) -> dict[str, Any]:
    return {
        "geometric_annualized_return": _finite_or_none(geometric_annualized_return(values)),
        "sharpe": _finite_or_none(annualized_ratio(values)),
        "max_drawdown": _finite_or_none(max_drawdown_from_returns(values)),
        win_name: float(np.mean(values > 0)) if values.size else None,
    }


@dataclass(frozen=True, slots=True)
class OOSBaselineEvaluation:
    factor_pool_fingerprint: str
    strategy_bundle_fingerprint: str
    test_feature_context_fingerprint: str
    test_factor_matrix_fingerprint: str
    test_score_artifact_fingerprint: str
    test_label_artifact_fingerprint: str
    calendar_fingerprint: str
    split_fingerprint: str
    coverage_by_date: pd.DataFrame
    strategy_scores_test: pd.DataFrame
    strategy_rank_ic_by_date: pd.DataFrame
    decile_assignments: pd.DataFrame
    decile_returns: pd.DataFrame
    portfolio_returns: pd.DataFrame
    turnover_by_date: pd.DataFrame
    nav_series: pd.DataFrame
    performance_summary: Mapping[str, Any]
    lightgbm_split_effectiveness: Mapping[str, Any]
    strategy_score_correlation: Mapping[str, Any]
    strategy_freeze_summary: Mapping[str, Any]
    invalid_period_diagnostics: Mapping[str, Any]
    first_access_evidence: Mapping[str, Any]
    contract: Mapping[str, Any]
    logical_fingerprint: str


@dataclass(frozen=True, slots=True)
class OOSBaselineEvaluationArtifact:
    manifest_path: Path
    fingerprint: str
    reused_existing_artifact: bool


@dataclass(frozen=True, slots=True)
class VerifiedOOSBaselineEvaluation:
    manifest_path: Path
    fingerprint: str
    factor_pool_fingerprint: str
    strategy_bundle_fingerprint: str
    test_score_artifact_fingerprint: str
    manifest: Mapping[str, Any]
    payloads: Mapping[str, Any]


def _split_diagnostics(
    dates: np.ndarray,
    symbols: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    rank_ics: list[float] = []
    spreads: list[float] = []
    for date in np.unique(dates):
        mask = dates == date
        valid = mask & np.isfinite(scores) & np.isfinite(labels)
        if int(valid.sum()) < MIN_DECILE_STOCK_COUNT or np.ptp(scores[valid]) == 0:
            continue
        rank_ics.append(spearman_average_ties(scores[valid], labels[valid]))
        groups = deterministic_deciles(scores[valid], symbols[valid])
        spreads.append(float(np.mean(labels[valid][groups == 10]) - np.mean(labels[valid][groups == 1])))
    summary = summarize_rank_ic(rank_ics)
    summary["mean_5d_g10_g1"] = float(np.mean(spreads)) if spreads else None
    return summary


def _lightgbm_split_effectiveness(
    bundle: VerifiedFrozenStrategyBundle,
    test_rank_ic: Sequence[float],
    test_spreads: Sequence[float],
) -> dict[str, Any]:
    matrix_path = Path(bundle.manifest["upstream_manifest_paths"]["development_matrix_manifest"])
    matrices = load_verified_development_factor_matrices(matrix_path)
    rows: dict[str, Any] = {}
    filenames = {
        "Train": LIGHTGBM_TRAIN_SCORES_FILENAME,
        "Validation": LIGHTGBM_VALIDATION_SCORES_FILENAME,
    }
    for display, filename in filenames.items():
        split_name = display.lower()
        split = matrices.splits[split_name]
        scores = np.asarray(np.load(bundle.manifest_path.parent / filename, allow_pickle=False), dtype=np.float64)
        rows[display] = {
            **_split_diagnostics(
                split.features.dates,
                split.features.symbols,
                scores,
                split.forward_returns,
            ),
            "model_source": "selection_model",
            "interpretation": (
                "development_in_sample" if display == "Train" else "development_early_stopping_holdout"
            ),
            "is_true_oos": False,
        }
    rows["Test"] = {
        **summarize_rank_ic(test_rank_ic),
        "mean_5d_g10_g1": float(np.mean(test_spreads)) if test_spreads else None,
        "model_source": "final_train_validation_refit_frozen_model",
        "interpretation": "true_untouched_oos",
        "is_true_oos": True,
    }
    return {
        "schema": "factor_gfn.lightgbm_split_effectiveness.v1",
        "rows": rows,
        "development_diagnostics_source": "stored_selection_model_scores",
        "final_model_backfill_for_development": False,
    }


def evaluate_oos_baselines(
    pool: VerifiedFrozenBaselineFactorPool,
    bundle: VerifiedFrozenStrategyBundle,
    matrix: TestFactorMatrix,
    scores: VerifiedTestScoreArtifact,
    labels: VerifiedTestLabels,
    *,
    cost_rate: float = 0.0,
) -> OOSBaselineEvaluation:
    """Evaluate the common Test sample after labels have crossed the score-freeze gate."""

    if not isinstance(scores, VerifiedTestScoreArtifact) or not isinstance(labels, VerifiedTestLabels):
        raise TypeError("verified Test score and label artifacts are required")
    if not math.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("cost_rate must be finite and nonnegative")
    if (
        bundle.factor_pool_fingerprint != pool.baseline_factor_pool_fingerprint
        or matrix.factor_pool_fingerprint != pool.baseline_factor_pool_fingerprint
        or matrix.strategy_bundle_fingerprint != bundle.bundle_fingerprint
        or scores.test_factor_matrix_fingerprint != matrix.fingerprint
        or labels.score_artifact_fingerprint != scores.fingerprint
        or labels.source_context_fingerprint != matrix.source_context_fingerprint
        or labels.calendar_fingerprint != matrix.calendar_fingerprint
    ):
        raise OOSAuthorityError("OOS evaluator authority chain mismatch")

    key_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(matrix.features.dates.astype(str)),
            "symbol": matrix.features.symbols.astype(str),
            "forward_return_5d": labels.forward_returns,
        }
    )
    score_frame = scores.scores.copy()
    score_frame["date"] = pd.to_datetime(score_frame["date"])
    wide = score_frame.pivot(index=["date", "symbol"], columns="strategy_id", values="strategy_score")
    wide = wide.reset_index().merge(key_frame, on=["date", "symbol"], how="left", validate="one_to_one")
    if wide[list(STRATEGY_IDS)].isna().any().any():
        raise OOSBaselineEvaluationIntegrityError("common strategy keys are incomplete")

    coverage_rows = []
    valid_dates: list[pd.Timestamp] = []
    invalid: dict[str, list[str]] = {}
    for date_string, raw_count in matrix.raw_universe_counts.items():
        date = pd.Timestamp(date_string)
        subset = wide.loc[wide["date"] == date]
        eligible_count = int(matrix.eligible_counts[date_string])
        label_count = int(np.isfinite(subset["forward_return_5d"]).sum())
        reasons: list[str] = []
        if label_count < MIN_DECILE_STOCK_COUNT:
            reasons.append("evaluation_eligible_count_below_20")
        for strategy_id in STRATEGY_IDS:
            values = subset.loc[np.isfinite(subset["forward_return_5d"]), strategy_id].to_numpy(dtype=float)
            if values.size and np.ptp(values) == 0:
                reasons.append(f"{strategy_id}_score_has_no_cross_sectional_variation")
        valid = not reasons
        if valid:
            valid_dates.append(date)
        else:
            invalid[date_string] = reasons
        coverage_rows.append(
            {
                "date": date,
                "raw_universe_count": int(raw_count),
                "eligible_stock_count": eligible_count,
                "coverage_ratio": eligible_count / raw_count if raw_count else math.nan,
                "label_eligible_count": label_count,
                "decile_evaluable_count": label_count if valid else 0,
                "valid_period": valid,
                "invalid_reasons": "|".join(reasons),
            }
        )
    coverage = pd.DataFrame(coverage_rows).sort_values("date").reset_index(drop=True)

    assignment_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    for date in valid_dates:
        subset = wide.loc[(wide["date"] == date) & np.isfinite(wide["forward_return_5d"])].copy()
        benchmark = float(subset["forward_return_5d"].mean())
        for strategy_id in STRATEGY_IDS:
            groups = deterministic_deciles(subset[strategy_id], subset["symbol"])
            subset_groups = subset.assign(group=groups)
            returns = {
                group: float(subset_groups.loc[subset_groups["group"] == group, "forward_return_5d"].mean())
                for group in range(1, 11)
            }
            for row in subset_groups.itertuples(index=False):
                assignment_rows.append(
                    {
                        "date": date,
                        "symbol": row.symbol,
                        "strategy_id": strategy_id,
                        "group": int(row.group),
                        "strategy_score": float(getattr(row, strategy_id)),
                        "forward_return_5d": float(row.forward_return_5d),
                    }
                )
            spread = returns[10] - returns[1]
            decile_rows.append(
                {"date": date, "strategy_id": strategy_id, **{f"G{i}": returns[i] for i in range(1, 11)}, "G10_G1": spread}
            )
            rank_ic = spearman_average_ties(subset[strategy_id], subset["forward_return_5d"])
            ic_rows.append({"date": date, "strategy_id": strategy_id, "rank_ic": rank_ic})
            long_return = returns[10]
            excess = long_return - benchmark
            portfolio_rows.append(
                {
                    "date": date,
                    "strategy_id": strategy_id,
                    "g10_return": long_return,
                    "benchmark_return": benchmark,
                    "excess_return": excess,
                    "long_short_return": spread,
                    "cost_rate": cost_rate,
                    "estimated_cost": math.nan,
                    "net_long_return": math.nan,
                }
            )

    assignments = pd.DataFrame(assignment_rows)
    deciles = pd.DataFrame(decile_rows)
    rank_ic = pd.DataFrame(ic_rows)
    if not rank_ic.empty:
        rank_ic = rank_ic.sort_values(["strategy_id", "date"]).reset_index(drop=True)
        rank_ic["cumulative_ic"] = rank_ic.groupby("strategy_id", sort=False)["rank_ic"].cumsum()
    portfolio = pd.DataFrame(portfolio_rows)

    turnover_rows: list[dict[str, Any]] = []
    coverage_dates = coverage["date"].tolist()
    valid_set = set(valid_dates)
    for strategy_id in STRATEGY_IDS:
        previous_date: pd.Timestamp | None = None
        previous_holdings: pd.DataFrame | None = None
        for date in valid_dates:
            current = assignments.loc[
                (assignments["strategy_id"] == strategy_id)
                & (assignments["date"] == date)
                & (assignments["group"] == 10),
                ["symbol", "forward_return_5d"],
            ]
            turnover = math.nan
            replacement = math.nan
            diagnostic = "first_or_reset_period"
            adjacent = (
                previous_date is not None
                and coverage_dates.index(date) == coverage_dates.index(previous_date) + 1
                and previous_date in valid_set
            )
            if adjacent and previous_holdings is not None:
                turnover, replacement, diagnostic = drift_adjusted_one_way_turnover(
                    previous_holdings["symbol"],
                    previous_holdings["forward_return_5d"],
                    current["symbol"],
                )
            turnover_rows.append(
                {
                    "date": date,
                    "strategy_id": strategy_id,
                    "one_way_turnover": turnover,
                    "constituent_replacement_rate": replacement,
                    "diagnostic_reason": diagnostic,
                }
            )
            previous_date = date
            previous_holdings = current
    turnover = pd.DataFrame(turnover_rows)
    portfolio = portfolio.merge(turnover[["date", "strategy_id", "one_way_turnover"]], on=["date", "strategy_id"], how="left")
    portfolio["estimated_cost"] = portfolio["one_way_turnover"] * cost_rate
    portfolio["net_long_return"] = portfolio["g10_return"] - portfolio["estimated_cost"]

    nav_rows: list[dict[str, Any]] = []
    performance: dict[str, Any] = {}
    for strategy_id in STRATEGY_IDS:
        values = portfolio.loc[portfolio["strategy_id"] == strategy_id].sort_values("date")
        long_values = values["g10_return"].to_numpy(dtype=float)
        benchmark_values = values["benchmark_return"].to_numpy(dtype=float)
        excess_values = values["excess_return"].to_numpy(dtype=float)
        spread_values = values["long_short_return"].to_numpy(dtype=float)
        long_nav = compound_nav(long_values)
        benchmark_nav = compound_nav(benchmark_values)
        excess_nav = compound_nav(excess_values)
        spread_nav = compound_nav(spread_values)
        if len(values):
            nav_rows.append(
                {
                    "date": values["date"].iloc[0] - pd.Timedelta(days=1),
                    "strategy_id": strategy_id,
                    "g10_nav": 1.0,
                    "benchmark_nav": 1.0,
                    "excess_nav": 1.0,
                    "long_short_nav": 1.0,
                    "is_initial": True,
                }
            )
        for index, date in enumerate(values["date"]):
            nav_rows.append(
                {
                    "date": date,
                    "strategy_id": strategy_id,
                    "g10_nav": long_nav[index],
                    "benchmark_nav": benchmark_nav[index],
                    "excess_nav": excess_nav[index],
                    "long_short_nav": spread_nav[index],
                    "is_initial": False,
                }
            )
        ic_summary = summarize_rank_ic(
            rank_ic.loc[rank_ic["strategy_id"] == strategy_id, "rank_ic"]
        )
        long_summary = _performance(long_values, win_name="long_win_rate")
        excess_summary = {
            "geometric_annualized_excess_return": _finite_or_none(geometric_annualized_return(excess_values)),
            "excess_ir": _finite_or_none(annualized_ratio(excess_values)),
            "excess_max_drawdown": _finite_or_none(max_drawdown_from_returns(excess_values)),
            "excess_win_rate": float(np.mean(excess_values > 0)) if excess_values.size else None,
        }
        spread_summary = _performance(spread_values, win_name="long_short_win_rate")
        strategy_turnover = turnover.loc[turnover["strategy_id"] == strategy_id]
        performance[strategy_id] = {
            **ic_summary,
            "g10_long": long_summary,
            "excess": excess_summary,
            "long_short": spread_summary,
            "turnover": {
                "mean_one_way_turnover": _finite_or_none(float(strategy_turnover["one_way_turnover"].mean())),
                "median_one_way_turnover": _finite_or_none(float(strategy_turnover["one_way_turnover"].median())),
                "p75_one_way_turnover": _finite_or_none(float(strategy_turnover["one_way_turnover"].quantile(0.75))),
                "max_one_way_turnover": _finite_or_none(float(strategy_turnover["one_way_turnover"].max())),
                "turnover_observation_count": int(strategy_turnover["one_way_turnover"].notna().sum()),
                "mean_constituent_replacement_rate": _finite_or_none(float(strategy_turnover["constituent_replacement_rate"].mean())),
            },
        }
    mean_deciles = {}
    for strategy_id in STRATEGY_IDS:
        subset = deciles.loc[deciles["strategy_id"] == strategy_id]
        mean_deciles[strategy_id] = {
            **{f"G{index}": _finite_or_none(float(subset[f"G{index}"].mean())) for index in range(1, 11)},
            "G10_G1": _finite_or_none(float(subset["G10_G1"].mean())),
        }
    performance["reporting_summary"] = {
        "mean_decile_returns": mean_deciles,
        "coverage": {
            "mean_raw_universe_count": _finite_or_none(float(coverage["raw_universe_count"].mean())),
            "mean_complete_case_eligible_count": _finite_or_none(float(coverage["eligible_stock_count"].mean())),
            "mean_label_eligible_count": _finite_or_none(float(coverage["label_eligible_count"].mean())),
            "min_eligible_count": int(coverage["eligible_stock_count"].min()),
            "mean_coverage_ratio": _finite_or_none(float(coverage["coverage_ratio"].mean())),
            "min_coverage_ratio": _finite_or_none(float(coverage["coverage_ratio"].min())),
            "invalid_rebalance_periods": int((~coverage["valid_period"]).sum()),
        },
    }
    nav = pd.DataFrame(nav_rows)

    correlation_matrix = np.eye(len(STRATEGY_IDS), dtype=float)
    correlation_counts = np.zeros((len(STRATEGY_IDS), len(STRATEGY_IDS)), dtype=int)
    for left_index, left in enumerate(STRATEGY_IDS):
        for right_index, right in enumerate(STRATEGY_IDS):
            periodic = []
            for date in valid_dates:
                subset = wide.loc[wide["date"] == date]
                value = spearman_average_ties(subset[left], subset[right])
                if math.isfinite(value):
                    periodic.append(value)
            correlation_matrix[left_index, right_index] = float(np.mean(periodic)) if periodic else math.nan
            correlation_counts[left_index, right_index] = len(periodic)
    score_correlation = {
        "schema": "factor_gfn.mean_periodic_strategy_score_correlation.v1",
        "strategy_ids": list(STRATEGY_IDS),
        "matrix": [[_finite_or_none(v) for v in row] for row in correlation_matrix],
        "finite_period_counts": correlation_counts.tolist(),
        "method": "mean_periodic_cross_sectional_spearman_average_rank_ties",
    }
    lgb_test_ic = rank_ic.loc[rank_ic["strategy_id"] == "lightgbm", "rank_ic"].tolist()
    lgb_test_spreads = deciles.loc[deciles["strategy_id"] == "lightgbm", "G10_G1"].tolist()
    lgb_effectiveness = _lightgbm_split_effectiveness(bundle, lgb_test_ic, lgb_test_spreads)
    equal = bundle.strategies["equal_weight"]
    fixed = bundle.strategies["fixed_icir"]
    lightgbm = bundle.strategies["lightgbm"]
    freeze_summary = {
        "factor_pool_fingerprint": pool.baseline_factor_pool_fingerprint,
        "strategy_bundle_fingerprint": bundle.bundle_fingerprint,
        "test_score_artifact_fingerprint": scores.fingerprint,
        "equal_weight": {
            "K": len(bundle.feature_aliases),
            "weights_identity": _stable_hash({"weights": list(equal.weights)}),
        },
        "fixed_icir": {
            "weights_fingerprint": _stable_hash({"weights": list(fixed.weights)}),
            "development_ic_contract": {
                key: _deep_thaw(fixed.metadata.get(key))
                for key in (
                    "development_splits",
                    "rank_ic",
                    "split_observations",
                    "min_cross_section_count",
                    "std_ddof",
                    "epsilon",
                    "positive_clipping",
                    "fallback_status",
                    "fallback_reason",
                )
            },
        },
        "lightgbm": {
            "parameter_fingerprint": lightgbm.metadata.get("fixed_parameter_fingerprint"),
            "best_iteration": lightgbm.metadata.get("best_iteration"),
            "selection_model_fingerprint": lightgbm.metadata.get("selection_model_fingerprint"),
            "final_model_fingerprint": lightgbm.metadata.get("final_model_fingerprint"),
            "lightgbm_version": lightgbm.metadata.get("library_version"),
        },
        "strategy_bundle_freeze_timestamp": bundle.manifest.get("created_at_utc"),
        "freeze_time_oos_status": bundle.oos_status,
        "first_test_access_evidence": dict(labels.first_access_evidence),
    }
    contract = {
        "label_formula": LABEL_FORMULA,
        "entry_offset": 1,
        "exit_offset": 6,
        "complete_case": "all_K_directional_cleaned_factors_finite",
        "min_decile_stock_count": MIN_DECILE_STOCK_COUNT,
        "deciles": "score_ascending_symbol_tiebreak_equal_count_G1_low_G10_high",
        "rank_ic_ties": "spearman_average_rank",
        "periods_per_year": PERIODS_PER_YEAR,
        "ddof": DDOF,
        "annualization": "geometric_compounded",
        "nav": "product_of_one_plus_arithmetic_period_return",
        "excess": "G10_minus_common_equal_weight_benchmark",
        "long_short": "G10_minus_G1",
        "mdd": "drawdown_of_corresponding_compounded_arithmetic_return_nav",
        "turnover": "drift_adjusted_one_way_union_holdings",
        "cost_rate": cost_rate,
        "headline_performance": "gross",
    }
    split_fingerprint = _stable_hash(
        {"requested": list(matrix.requested_boundary), "actual": list(matrix.actual_boundary)}
    )
    logical = {
        "schema": OOS_EVALUATION_SCHEMA,
        "version": OOS_EVALUATION_VERSION,
        "factor_pool_fingerprint": pool.baseline_factor_pool_fingerprint,
        "strategy_bundle_fingerprint": bundle.bundle_fingerprint,
        "test_feature_context_fingerprint": matrix.feature_context_fingerprint,
        "test_factor_matrix_fingerprint": matrix.fingerprint,
        "test_score_artifact_fingerprint": scores.fingerprint,
        "test_label_artifact_fingerprint": labels.fingerprint,
        "calendar_fingerprint": matrix.calendar_fingerprint,
        "split_fingerprint": split_fingerprint,
        "contract": contract,
        "payload_digests": {
            "coverage": _sha256_bytes(_parquet_bytes(coverage)),
            "scores": _sha256_bytes(_parquet_bytes(score_frame)),
            "rank_ic": _sha256_bytes(_parquet_bytes(rank_ic)),
            "deciles": _sha256_bytes(_parquet_bytes(deciles)),
            "portfolio": _sha256_bytes(_parquet_bytes(portfolio)),
            "turnover": _sha256_bytes(_parquet_bytes(turnover)),
            "performance": _sha256_bytes(_json_bytes(performance)),
        },
    }
    return OOSBaselineEvaluation(
        factor_pool_fingerprint=pool.baseline_factor_pool_fingerprint,
        strategy_bundle_fingerprint=bundle.bundle_fingerprint,
        test_feature_context_fingerprint=matrix.feature_context_fingerprint,
        test_factor_matrix_fingerprint=matrix.fingerprint,
        test_score_artifact_fingerprint=scores.fingerprint,
        test_label_artifact_fingerprint=labels.fingerprint,
        calendar_fingerprint=matrix.calendar_fingerprint,
        split_fingerprint=split_fingerprint,
        coverage_by_date=coverage,
        strategy_scores_test=score_frame,
        strategy_rank_ic_by_date=rank_ic,
        decile_assignments=assignments,
        decile_returns=deciles,
        portfolio_returns=portfolio,
        turnover_by_date=turnover,
        nav_series=nav,
        performance_summary=MappingProxyType(performance),
        lightgbm_split_effectiveness=MappingProxyType(lgb_effectiveness),
        strategy_score_correlation=MappingProxyType(score_correlation),
        strategy_freeze_summary=MappingProxyType(freeze_summary),
        invalid_period_diagnostics=MappingProxyType(invalid),
        first_access_evidence=labels.first_access_evidence,
        contract=MappingProxyType(contract),
        logical_fingerprint=_stable_hash(logical),
    )


def _payloads(evaluation: OOSBaselineEvaluation) -> dict[str, bytes]:
    payloads = {
        PARQUET_FILES[name]: _parquet_bytes(getattr(evaluation, name))
        for name in PARQUET_FILES
    }
    payloads.update(
        {
            JSON_FILES[name]: _json_bytes(dict(getattr(evaluation, name)))
            for name in JSON_FILES
        }
    )
    return payloads


def _manifest_fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"evaluation_fingerprint", "created_at_utc", "created_at_excluded_from_fingerprint"}
    payload = {key: value for key, value in manifest.items() if key not in excluded}
    evidence = dict(payload.get("oos_first_access_evidence", {}))
    evidence.pop("accessed_at_utc", None)
    payload["oos_first_access_evidence"] = evidence
    return payload


def freeze_oos_baseline_evaluation(
    evaluation: OOSBaselineEvaluation,
    runs_root: str | Path,
) -> OOSBaselineEvaluationArtifact:
    if not isinstance(evaluation, OOSBaselineEvaluation):
        raise TypeError("evaluation must be OOSBaselineEvaluation")
    payloads = _payloads(evaluation)
    row_counts = {
        name: len(getattr(evaluation, name)) for name in PARQUET_FILES
    }
    manifest: dict[str, Any] = {
        "schema": OOS_EVALUATION_SCHEMA,
        "version": OOS_EVALUATION_VERSION,
        "evaluation_status": "complete_verified_oos",
        "factor_pool_fingerprint": evaluation.factor_pool_fingerprint,
        "strategy_bundle_fingerprint": evaluation.strategy_bundle_fingerprint,
        "test_feature_context_fingerprint": evaluation.test_feature_context_fingerprint,
        "test_factor_matrix_fingerprint": evaluation.test_factor_matrix_fingerprint,
        "test_score_artifact_fingerprint": evaluation.test_score_artifact_fingerprint,
        "test_label_artifact_fingerprint": evaluation.test_label_artifact_fingerprint,
        "universe_fingerprint": _stable_hash(
            evaluation.coverage_by_date[["date", "raw_universe_count", "eligible_stock_count"]]
            .assign(date=lambda frame: frame["date"].astype(str))
            .to_dict("records")
        ),
        "calendar_fingerprint": evaluation.calendar_fingerprint,
        "split_fingerprint": evaluation.split_fingerprint,
        "contract": dict(evaluation.contract),
        "evaluator_implementation_fingerprint": _implementation_fingerprint(),
        "logical_evaluation_fingerprint": evaluation.logical_fingerprint,
        "artifacts": {
            name: {"size_bytes": len(payload), "sha256": _sha256_bytes(payload)}
            for name, payload in sorted(payloads.items())
        },
        "row_counts": row_counts,
        "key_ranges": {
            name: {
                "start": str(getattr(evaluation, name)["date"].min()) if len(getattr(evaluation, name)) and "date" in getattr(evaluation, name) else None,
                "end": str(getattr(evaluation, name)["date"].max()) if len(getattr(evaluation, name)) and "date" in getattr(evaluation, name) else None,
            }
            for name in PARQUET_FILES
        },
        "invalid_period_diagnostics": dict(evaluation.invalid_period_diagnostics),
        "strategy_freeze_summary": dict(evaluation.strategy_freeze_summary),
        "oos_first_access_evidence": dict(evaluation.first_access_evidence),
        "upstream_manifest_paths": {
            "test_score_manifest": str(evaluation.first_access_evidence["score_manifest_path"]),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "created_at_excluded_from_fingerprint": True,
    }
    manifest["evaluation_fingerprint"] = _stable_hash(_manifest_fingerprint_payload(manifest))
    fingerprint = str(manifest["evaluation_fingerprint"])
    root = Path(runs_root).resolve() / "oos_baseline_evaluations"
    target = root / fingerprint
    manifest_path = target / OOS_EVALUATION_MANIFEST_FILENAME
    if target.exists():
        verified = load_verified_oos_baseline_evaluation(
            manifest_path,
            expected_factor_pool_fingerprint=evaluation.factor_pool_fingerprint,
            expected_strategy_bundle_fingerprint=evaluation.strategy_bundle_fingerprint,
            expected_test_score_artifact_fingerprint=evaluation.test_score_artifact_fingerprint,
        )
        return OOSBaselineEvaluationArtifact(manifest_path, verified.fingerprint, True)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.tmp-", dir=root))
    try:
        for name, payload in payloads.items():
            (temporary / name).write_bytes(payload)
        (temporary / OOS_EVALUATION_MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))
        _verify_evaluation_directory(
            temporary / OOS_EVALUATION_MANIFEST_FILENAME,
            expected_factor_pool_fingerprint=evaluation.factor_pool_fingerprint,
            expected_strategy_bundle_fingerprint=evaluation.strategy_bundle_fingerprint,
            expected_test_score_artifact_fingerprint=evaluation.test_score_artifact_fingerprint,
            require_directory_identity=False,
        )
        os.replace(temporary, target)
        load_verified_oos_baseline_evaluation(
            manifest_path,
            expected_factor_pool_fingerprint=evaluation.factor_pool_fingerprint,
            expected_strategy_bundle_fingerprint=evaluation.strategy_bundle_fingerprint,
            expected_test_score_artifact_fingerprint=evaluation.test_score_artifact_fingerprint,
        )
        return OOSBaselineEvaluationArtifact(manifest_path, fingerprint, False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _verify_evaluation_directory(
    manifest_path: Path,
    *,
    expected_factor_pool_fingerprint: str,
    expected_strategy_bundle_fingerprint: str,
    expected_test_score_artifact_fingerprint: str,
    require_directory_identity: bool,
) -> VerifiedOOSBaselineEvaluation:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OOSBaselineEvaluationIntegrityError("cannot read OOS evaluation manifest") from error
    fingerprint = str(manifest.get("evaluation_fingerprint", ""))
    if (
        manifest.get("schema") != OOS_EVALUATION_SCHEMA
        or manifest.get("version") != OOS_EVALUATION_VERSION
        or manifest.get("evaluation_status") != "complete_verified_oos"
        or _stable_hash(_manifest_fingerprint_payload(manifest)) != fingerprint
        or (require_directory_identity and manifest_path.parent.name != fingerprint)
        or manifest.get("factor_pool_fingerprint") != expected_factor_pool_fingerprint
        or manifest.get("strategy_bundle_fingerprint") != expected_strategy_bundle_fingerprint
        or manifest.get("test_score_artifact_fingerprint") != expected_test_score_artifact_fingerprint
        or manifest.get("evaluator_implementation_fingerprint") != _implementation_fingerprint()
    ):
        raise OOSBaselineEvaluationIntegrityError("OOS evaluation manifest or authority mismatch")
    expected_files = set(PARQUET_FILES.values()) | set(JSON_FILES.values())
    if set(manifest.get("artifacts", {})) != expected_files:
        raise OOSBaselineEvaluationIntegrityError("OOS evaluation artifact inventory mismatch")
    payloads: dict[str, Any] = {}
    reverse = {value: key for key, value in {**PARQUET_FILES, **JSON_FILES}.items()}
    for filename, metadata in manifest["artifacts"].items():
        path = manifest_path.parent / filename
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata.get("size_bytes", -1))
            or _sha256_file(path) != metadata.get("sha256")
        ):
            raise OOSBaselineEvaluationIntegrityError(f"OOS evaluation payload changed: {filename}")
        name = reverse[filename]
        if filename.endswith(".parquet"):
            frame = pd.read_parquet(path)
            if len(frame) != int(manifest["row_counts"].get(name, -1)):
                raise OOSBaselineEvaluationIntegrityError(f"OOS row count changed: {filename}")
            payloads[name] = frame
        else:
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    score_frame = payloads["strategy_scores_test"]
    key_sets = []
    for strategy_id in STRATEGY_IDS:
        subset = score_frame.loc[score_frame["strategy_id"] == strategy_id, ["date", "symbol"]]
        key_sets.append(list(map(tuple, subset.astype({"symbol": str}).to_numpy())))
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise OOSBaselineEvaluationIntegrityError("verified evaluation common score keys changed")
    return VerifiedOOSBaselineEvaluation(
        manifest_path=manifest_path,
        fingerprint=fingerprint,
        factor_pool_fingerprint=expected_factor_pool_fingerprint,
        strategy_bundle_fingerprint=expected_strategy_bundle_fingerprint,
        test_score_artifact_fingerprint=expected_test_score_artifact_fingerprint,
        manifest=MappingProxyType(manifest),
        payloads=MappingProxyType(payloads),
    )


def load_verified_oos_baseline_evaluation(
    manifest_path: str | Path,
    *,
    expected_factor_pool_fingerprint: str,
    expected_strategy_bundle_fingerprint: str,
    expected_test_score_artifact_fingerprint: str,
) -> VerifiedOOSBaselineEvaluation:
    return _verify_evaluation_directory(
        Path(manifest_path).resolve(),
        expected_factor_pool_fingerprint=expected_factor_pool_fingerprint,
        expected_strategy_bundle_fingerprint=expected_strategy_bundle_fingerprint,
        expected_test_score_artifact_fingerprint=expected_test_score_artifact_fingerprint,
        require_directory_identity=True,
    )


__all__ = [
    "DDOF",
    "MIN_DECILE_STOCK_COUNT",
    "OOSBaselineEvaluation",
    "OOSBaselineEvaluationArtifact",
    "OOSBaselineEvaluationIntegrityError",
    "OOS_EVALUATION_MANIFEST_FILENAME",
    "PERIODS_PER_YEAR",
    "VerifiedOOSBaselineEvaluation",
    "annualized_ratio",
    "compound_nav",
    "deterministic_deciles",
    "drift_adjusted_one_way_turnover",
    "evaluate_oos_baselines",
    "freeze_oos_baseline_evaluation",
    "geometric_annualized_return",
    "load_verified_oos_baseline_evaluation",
    "max_drawdown_from_returns",
    "spearman_average_ties",
    "summarize_rank_ic",
]
