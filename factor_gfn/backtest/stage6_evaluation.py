"""Stage 6 Train/Validation-only context and single-candidate evaluation.

This module deliberately has no batch runner, cache, resume, hard-screen,
decorrelation, or OOS evaluation surface.  It answers only whether one
AUTO_ACCEPT candidate can be recomputed under the frozen Stage 6 contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
import numpy.typing as npt

from factor_gfn.barra import (
    STYLE_NAMES,
    BarraConfig,
    BarraFactorSet,
    LongShortSeries,
    build_barra_long_short_returns,
    calculate_barra_ts_corr,
)
from factor_gfn.data.industry import load_sw_industry_panel
from factor_gfn.evaluator import (
    EvaluationConfig,
    FactorInterpreter,
    NeutralizationDiagnostics,
    build_forward_returns,
    clean_candidate_factor_cross_sections,
    encode_industry_panel,
    infer_long_direction,
    long_portfolio_series_from_cleaned,
    long_short_portfolio_series_from_cleaned,
    rank_ic_values_from_cleaned,
    summarize_excess_returns,
    summarize_ic,
)
from factor_gfn.grammar import Expression
from factor_gfn.gfn.real_data import RealRewardDataPaths

from .expression_compatibility import ACCEPTED_REGISTRY_SCHEMA


STAGE6_EVALUATION_CONTEXT_SCHEMA = "factor_gfn.stage6_evaluation_context.v1"
STAGE6_EVALUATION_RESULT_SCHEMA = "factor_gfn.stage6_candidate_evaluation.v1"
STAGE6_EVALUATION_CONTRACT_SCHEMA = "factor_gfn.stage6_evaluation_contract.v1"
STAGE6_EVALUATOR_VERSION = "stage6-single-candidate-v1"
Stage6SplitName = Literal["train", "validation"]
STAGE6_SPLIT_NAMES: tuple[Stage6SplitName, ...] = ("train", "validation")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npy_row_prefix(path: Path, row_count: int) -> npt.NDArray[Any]:
    """Memory-map only the C-order row prefix, excluding later/OOS payload rows."""

    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version for prefix mapping: {version}")
        data_offset = stream.tell()
    if not shape or row_count < 1 or row_count > shape[0]:
        raise ValueError(f"invalid row prefix {row_count} for {path} with shape {shape}")
    if fortran_order:
        raise ValueError(f"Stage 6 prefix mapping requires C-order NPY input: {path}")
    if dtype.hasobject:
        raise ValueError(f"Stage 6 prefix mapping rejects object NPY input: {path}")
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=(row_count, *shape[1:]),
        order="C",
    )


def _readonly(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    values.setflags(write=False)
    return values


def _finite_or_none(value: float | int | np.number | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    return number if math.isfinite(number) else None


def _float_series(values: npt.ArrayLike) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    return [_finite_or_none(value) for value in array]


def _valid_fingerprint(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class Stage6EvaluationConfig:
    """Frozen Train/Validation boundaries and numerical conventions."""

    train_start: str = "2010-01-01"
    train_end: str = "2018-12-31"
    validation_start: str = "2019-01-01"
    validation_end: str = "2020-12-31"
    evaluation: EvaluationConfig = EvaluationConfig()
    barra: BarraConfig = BarraConfig()

    def __post_init__(self) -> None:
        boundaries: list[tuple[np.datetime64, np.datetime64]] = []
        for name in STAGE6_SPLIT_NAMES:
            try:
                start = np.datetime64(getattr(self, f"{name}_start"), "D")
                end = np.datetime64(getattr(self, f"{name}_end"), "D")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} boundaries must be valid dates") from error
            if np.isnat(start) or np.isnat(end) or start > end:
                raise ValueError(f"{name} boundaries are invalid or reversed")
            boundaries.append((start, end))
        if boundaries[0][1] >= boundaries[1][0]:
            raise ValueError("train and validation must be ordered and non-overlapping")
        if boundaries[0][1] + np.timedelta64(1, "D") != boundaries[1][0]:
            raise ValueError("train and validation requested ranges must be calendar-contiguous")
        if self.evaluation.horizon != 5 or self.evaluation.entry_lag != 1:
            raise ValueError("Stage 6 fixes the label to open[t+6] / open[t+1] - 1")
        if self.evaluation.rebalance_interval != 5:
            raise ValueError("Stage 6 fixes a five-trading-day rebalance interval")
        if self.evaluation.rebalance_offset != 0:
            raise ValueError("Stage 6 global calendar does not permit a local offset")
        if self.evaluation.min_cross_section_count != self.barra.min_cross_section_count:
            raise ValueError("Evaluation and Barra minimum cross-section counts must match")
        if not np.isclose(self.evaluation.long_quantile, self.barra.long_short_quantile):
            raise ValueError("Evaluation and Barra long quantiles must match")


@dataclass(frozen=True, slots=True)
class Stage6SplitBoundary:
    name: Stage6SplitName
    requested_start: str
    requested_end: str
    actual_start: str
    actual_end: str
    start_row: int
    end_row: int
    scheduled_periods: int
    included_periods: int


@dataclass(frozen=True, slots=True)
class Stage6CalendarEntry:
    sequence_id: int
    split: Stage6SplitName
    signal_row: int
    entry_row: int | None
    exit_row: int | None
    signal_date: str
    entry_date: str | None
    exit_date: str | None
    label_within_split: bool
    barra_eligible: bool
    included: bool
    exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Stage6SplitData:
    boundary: Stage6SplitBoundary
    global_rebalance_rows: npt.NDArray[np.int64]
    rebalance_dates: npt.NDArray[np.datetime64]
    forward_returns: npt.NDArray[np.float64]
    universe_mask: npt.NDArray[np.bool_]
    industry_labels: npt.NDArray[np.int32]
    barra_long_short: Mapping[str, LongShortSeries]


@dataclass(frozen=True, slots=True)
class Stage6EvaluationContext:
    """Read-only data through Validation only; OOS has no data accessor."""

    config: Stage6EvaluationConfig
    factor_tensor: npt.NDArray[np.floating]
    dates: npt.NDArray[np.datetime64]
    stocks: npt.NDArray[np.str_]
    splits: Mapping[Stage6SplitName, Stage6SplitBoundary]
    calendar: tuple[Stage6CalendarEntry, ...]
    fingerprint: str
    manifest: Mapping[str, Any]
    _forward_returns: npt.NDArray[np.float64] = field(repr=False)
    _universe_mask: npt.NDArray[np.bool_] = field(repr=False)
    _industry_labels: npt.NDArray[np.int32] = field(repr=False)
    _barra_long_short: Mapping[str, LongShortSeries] = field(repr=False)

    def get_split_data(self, split: str) -> Stage6SplitData:
        if split == "oos":
            raise PermissionError("OOS is not loaded and cannot be accessed by Stage 6 evaluation")
        if split not in STAGE6_SPLIT_NAMES:
            raise ValueError(f"unknown Stage 6 split: {split!r}")
        typed_split: Stage6SplitName = split  # type: ignore[assignment]
        rows = np.asarray(
            [entry.signal_row for entry in self.calendar if entry.split == split and entry.included],
            dtype=np.int64,
        )
        compact_references: dict[str, LongShortSeries] = {}
        for name in STYLE_NAMES:
            series = self._barra_long_short[name]
            compact_references[name] = LongShortSeries(
                long_return=_readonly(np.asarray(series.long_return)[rows].copy()),
                short_return=_readonly(np.asarray(series.short_return)[rows].copy()),
                long_short_return=_readonly(
                    np.asarray(series.long_short_return)[rows].copy()
                ),
                universe_count=_readonly(np.asarray(series.universe_count)[rows].copy()),
                leg_count=_readonly(np.asarray(series.leg_count)[rows].copy()),
            )
        return Stage6SplitData(
            boundary=self.splits[typed_split],
            global_rebalance_rows=_readonly(rows),
            rebalance_dates=_readonly(self.dates[rows].copy()),
            forward_returns=_readonly(self._forward_returns[rows].copy()),
            universe_mask=_readonly(self._universe_mask[rows].copy()),
            industry_labels=_readonly(self._industry_labels[rows].copy()),
            barra_long_short=MappingProxyType(compact_references),
        )


def _split_rows(
    dates: npt.NDArray[np.datetime64], config: Stage6EvaluationConfig
) -> dict[Stage6SplitName, npt.NDArray[np.int64]]:
    result: dict[Stage6SplitName, npt.NDArray[np.int64]] = {}
    for name in STAGE6_SPLIT_NAMES:
        start = np.datetime64(getattr(config, f"{name}_start"), "D")
        end = np.datetime64(getattr(config, f"{name}_end"), "D")
        rows = np.flatnonzero((dates >= start) & (dates <= end)).astype(np.int64)
        if rows.size == 0:
            raise ValueError(f"processed data does not cover requested {name} range")
        if not np.all(np.diff(rows) == 1):
            raise ValueError(f"{name} must be a contiguous date-list slice")
        result[name] = rows
    return result


def build_stage6_evaluation_context_from_arrays(
    *,
    dates: npt.ArrayLike,
    stocks: npt.ArrayLike,
    factor_tensor: npt.ArrayLike,
    universe_mask: npt.ArrayLike,
    industry_labels: npt.ArrayLike,
    barra_exposures: Mapping[str, npt.ArrayLike],
    config: Stage6EvaluationConfig = Stage6EvaluationConfig(),
    source_manifest: Mapping[str, Any] | None = None,
) -> Stage6EvaluationContext:
    """Build a context after physically truncating every matrix at Validation."""

    all_dates = np.asarray(dates).astype("datetime64[D]")
    stock_values = np.asarray(stocks).astype(str)
    tensor_all = np.asarray(factor_tensor)
    universe_all = np.asarray(universe_mask, dtype=bool)
    industry_all = np.asarray(industry_labels, dtype=np.int32)
    if all_dates.ndim != 1 or stock_values.ndim != 1 or not all_dates.size or not stock_values.size:
        raise ValueError("dates and stocks must be non-empty one-dimensional arrays")
    if np.isnat(all_dates).any() or np.unique(all_dates).size != all_dates.size or not np.all(
        all_dates[:-1] < all_dates[1:]
    ):
        raise ValueError("dates must be unique, finite, and strictly increasing")
    if np.unique(stock_values).size != stock_values.size:
        raise ValueError("stocks must be unique")
    expected = (all_dates.size, stock_values.size)
    if tensor_all.ndim != 3 or tensor_all.shape[0] != expected[0] or tensor_all.shape[2] != expected[1]:
        raise ValueError("factor_tensor must use (date, feature, stock) axes")
    if tensor_all.shape[1] < 1:
        raise ValueError("factor_tensor must contain the open feature")
    if universe_all.shape != expected or industry_all.shape != expected:
        raise ValueError("universe and industry arrays must align to date/stock")
    if set(barra_exposures) != set(STYLE_NAMES):
        raise ValueError("barra_exposures must contain exactly the five frozen styles")
    exposure_all = {name: np.asarray(barra_exposures[name]) for name in STYLE_NAMES}
    if any(values.shape != expected for values in exposure_all.values()):
        raise ValueError("all Barra exposures must align to date/stock")

    requested_validation_end = np.datetime64(config.validation_end, "D")
    eligible_prefix = np.flatnonzero(all_dates <= requested_validation_end)
    if eligible_prefix.size == 0:
        raise ValueError("no data exists on or before requested Validation end")
    cutoff = int(eligible_prefix[-1]) + 1
    date_values = _readonly(all_dates[:cutoff].copy())
    tensor = _readonly(tensor_all[:cutoff])
    universe = _readonly(universe_all[:cutoff])
    industry = _readonly(industry_all[:cutoff])
    exposures = {
        name: _readonly(exposure_all[name][:cutoff]) for name in STYLE_NAMES
    }
    rows_by_split = _split_rows(date_values, config)

    forward_returns = build_forward_returns(tensor[:, 0, :], config.evaluation)
    exit_offset = config.evaluation.entry_lag + config.evaluation.horizon
    label_within = np.zeros(date_values.size, dtype=bool)
    split_by_row: dict[int, Stage6SplitName] = {}
    for name, rows in rows_by_split.items():
        split_by_row.update({int(row): name for row in rows})
        end_row = int(rows[-1])
        contained = rows + exit_offset <= end_row
        label_within[rows[contained]] = True
        forward_returns[rows[~contained]] = np.nan
    _readonly(forward_returns)

    barra_eligible = np.ones(date_values.size, dtype=bool)
    for values in exposures.values():
        counts = np.sum(
            universe & np.isfinite(forward_returns) & np.isfinite(values),
            axis=1,
            dtype=np.int64,
        )
        barra_eligible &= counts >= config.evaluation.min_cross_section_count
    train_rows = rows_by_split["train"]
    anchors = train_rows[label_within[train_rows] & barra_eligible[train_rows]]
    if anchors.size == 0:
        raise ValueError("Train contains no shared label-and-Barra-eligible anchor")
    anchor = int(anchors[0])
    scheduled_rows = np.arange(
        anchor,
        int(rows_by_split["validation"][-1]) + 1,
        config.evaluation.rebalance_interval,
        dtype=np.int64,
    )
    calendar: list[Stage6CalendarEntry] = []
    for sequence_id, signal_row in enumerate(scheduled_rows):
        row = int(signal_row)
        split = split_by_row[row]
        entry_row = row + config.evaluation.entry_lag
        exit_row = row + exit_offset
        reasons: list[str] = []
        if not bool(label_within[row]):
            reasons.append("label_crosses_split")
        if not bool(barra_eligible[row]):
            reasons.append("barra_cross_section_insufficient")
        calendar.append(
            Stage6CalendarEntry(
                sequence_id=sequence_id,
                split=split,
                signal_row=row,
                entry_row=entry_row if entry_row < date_values.size else None,
                exit_row=exit_row if exit_row < date_values.size else None,
                signal_date=str(date_values[row]),
                entry_date=str(date_values[entry_row]) if entry_row < date_values.size else None,
                exit_date=str(date_values[exit_row]) if exit_row < date_values.size else None,
                label_within_split=bool(label_within[row]),
                barra_eligible=bool(barra_eligible[row]),
                included=not reasons,
                exclusion_reasons=tuple(reasons),
            )
        )
    included_rows = np.asarray(
        [entry.signal_row for entry in calendar if entry.included], dtype=np.int64
    )
    factor_set = BarraFactorSet(
        exposures=exposures,
        market_return=_readonly(np.full(date_values.size, np.nan)),
    )
    barra_long_short = build_barra_long_short_returns(
        factor_set,
        forward_returns,
        included_rows,
        config.barra,
        universe_mask=universe,
    )
    for series in barra_long_short.values():
        for values in (
            series.long_return,
            series.short_return,
            series.long_short_return,
            series.universe_count,
            series.leg_count,
        ):
            _readonly(values)

    boundaries: dict[Stage6SplitName, Stage6SplitBoundary] = {}
    for name, rows in rows_by_split.items():
        entries = [entry for entry in calendar if entry.split == name]
        boundaries[name] = Stage6SplitBoundary(
            name=name,
            requested_start=getattr(config, f"{name}_start"),
            requested_end=getattr(config, f"{name}_end"),
            actual_start=str(date_values[rows[0]]),
            actual_end=str(date_values[rows[-1]]),
            start_row=int(rows[0]),
            end_row=int(rows[-1]),
            scheduled_periods=len(entries),
            included_periods=sum(entry.included for entry in entries),
        )
    calendar_payload = [asdict(entry) for entry in calendar]
    manifest: dict[str, Any] = {
        "schema": STAGE6_EVALUATION_CONTEXT_SCHEMA,
        "mode": "provisional",
        "config": asdict(config),
        "requested_validation_end": config.validation_end,
        "actual_latest_loaded_trade_date": str(date_values[-1]),
        "label_formula": "open[t+6] / open[t+1] - 1",
        "calendar": {
            "rule": "single_train_anchor_then_every_5_global_trading_rows_no_shift",
            "anchor_row": anchor,
            "anchor_date": str(date_values[anchor]),
            "scheduled_periods": len(calendar),
            "included_periods": sum(entry.included for entry in calendar),
            "fingerprint": _stable_hash(calendar_payload),
        },
        "splits": {name: asdict(boundary) for name, boundary in boundaries.items()},
        "shape": {
            "factor_tensor": list(map(int, tensor.shape)),
            "date_stock": list(map(int, universe.shape)),
        },
        "oos": {
            "loaded": False,
            "exposed": False,
            "candidate_evaluation_count": 0,
        },
        "sources": dict(source_manifest or {}),
    }
    fingerprint = _stable_hash(manifest)
    return Stage6EvaluationContext(
        config=config,
        factor_tensor=tensor,
        dates=date_values,
        stocks=_readonly(stock_values.copy()),
        splits=MappingProxyType(boundaries),
        calendar=tuple(calendar),
        fingerprint=fingerprint,
        manifest=MappingProxyType(manifest),
        _forward_returns=forward_returns,
        _universe_mask=universe,
        _industry_labels=industry,
        _barra_long_short=MappingProxyType(barra_long_short),
    )


def build_stage6_evaluation_context(
    config: Stage6EvaluationConfig = Stage6EvaluationConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> Stage6EvaluationContext:
    """Load only the prefix through requested Validation end from real inputs."""

    barra_paths = paths.barra_paths
    required = [
        paths.tensor_path,
        paths.universe_mask_path,
        paths.date_list_path,
        paths.stock_list_path,
        paths.processed_metadata_path,
        paths.industry_path,
        paths.industry_metadata_path,
        barra_paths.metadata_path,
        *[barra_paths.exposure_path(name) for name in STYLE_NAMES],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 6 evaluation context is missing inputs: {missing}")
    all_dates = np.load(paths.date_list_path, allow_pickle=False).astype("datetime64[D]")
    requested_end = np.datetime64(config.validation_end, "D")
    prefix_rows = np.flatnonzero(all_dates <= requested_end)
    if prefix_rows.size == 0:
        raise ValueError("date list has no row on or before requested Validation end")
    cutoff = int(prefix_rows[-1]) + 1
    dates = all_dates[:cutoff]
    stocks = np.load(paths.stock_list_path, allow_pickle=False).astype(str)
    tensor = _load_npy_row_prefix(paths.tensor_path, cutoff)
    universe = _load_npy_row_prefix(paths.universe_mask_path, cutoff)
    industry = load_sw_industry_panel(dates, stocks, level=1, path=paths.industry_path)
    exposures = {
        name: _load_npy_row_prefix(barra_paths.exposure_path(name), cutoff)
        for name in STYLE_NAMES
    }
    sources = {
        "processed_metadata_sha256": _sha256_file(paths.processed_metadata_path),
        "industry_metadata_sha256": _sha256_file(paths.industry_metadata_path),
        "barra_metadata_sha256": _sha256_file(barra_paths.metadata_path),
        "date_list_sha256": _sha256_file(paths.date_list_path),
        "stock_list_sha256": _sha256_file(paths.stock_list_path),
        "matrix_load_policy": "read_only_limited_npy_memmap_prefix_through_validation",
        "requested_prefix_rows": cutoff,
    }
    return build_stage6_evaluation_context_from_arrays(
        dates=dates,
        stocks=stocks,
        factor_tensor=tensor,
        universe_mask=universe,
        industry_labels=industry,
        barra_exposures=exposures,
        config=config,
        source_manifest=sources,
    )


@dataclass(frozen=True, slots=True)
class Stage6CandidateEvaluationResult:
    schema: str
    status: str
    invalid_reasons: tuple[str, ...]
    expression: Mapping[str, Any]
    source_identity: Mapping[str, Any]
    context_fingerprint: str
    evaluation_contract_fingerprint: str
    train_direction: int | None
    train: Mapping[str, Any]
    validation: Mapping[str, Any]
    factor_finite_coverage: Mapping[str, Any]
    factor_seconds: float
    train_evaluation_seconds: float
    validation_evaluation_seconds: float
    total_seconds: float
    result_fingerprint: str

    def deterministic_payload(self) -> dict[str, Any]:
        """Return exactly the payload bound by ``result_fingerprint``."""

        return {
            "schema": self.schema,
            "status": self.status,
            "invalid_reasons": list(self.invalid_reasons),
            "expression": dict(self.expression),
            "context_fingerprint": self.context_fingerprint,
            "evaluation_contract_fingerprint": self.evaluation_contract_fingerprint,
            "train_direction": self.train_direction,
            "train": dict(self.train),
            "validation": dict(self.validation),
            "factor_finite_coverage": dict(self.factor_finite_coverage),
        }

    def to_dict(self) -> dict[str, Any]:
        record = self.deterministic_payload()
        record.update(
            {
                "source_identity": dict(self.source_identity),
                "factor_seconds": self.factor_seconds,
                "train_evaluation_seconds": self.train_evaluation_seconds,
                "validation_evaluation_seconds": self.validation_evaluation_seconds,
                "total_seconds": self.total_seconds,
                "result_fingerprint": self.result_fingerprint,
            }
        )
        return record


@dataclass(slots=True)
class _PreparedSplit:
    result: dict[str, Any]
    cleaned_factor: npt.NDArray[np.float64]
    forward_returns: npt.NDArray[np.float64]


def _diagnostics_payload(
    diagnostics: NeutralizationDiagnostics, split: Stage6SplitData
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for local_row in sorted(diagnostics.skipped_details):
        detail = diagnostics.skipped_details[local_row]
        details.append(
            {
                "date": str(split.rebalance_dates[local_row]),
                "compact_row": int(local_row),
                "global_row": int(split.global_rebalance_rows[local_row]),
                "factor_valid_count": int(detail.factor_valid_count),
                "known_industry_count": int(detail.known_industry_count),
                "industry_count": int(detail.industry_count),
                "required_regression_count": int(detail.required_regression_count),
                "reason": detail.reason,
            }
        )
    count = int(split.rebalance_dates.size)
    return {
        "skipped_dates": [detail["date"] for detail in details],
        "skipped_count": len(details),
        "skipped_rate": len(details) / count if count else 0.0,
        "details": details,
    }


class Stage6CandidateEvaluator:
    """Recompute one accepted expression on Train and Validation."""

    _FORBIDDEN_HISTORICAL_FIELDS = frozenset(
        {
            "reward",
            "raw_reward",
            "train_ic",
            "validation_ic",
            "train_long_ir",
            "validation_long_ir",
            "barra_ts_corr",
            "metrics",
        }
    )

    def __init__(
        self,
        context: Stage6EvaluationContext,
        *,
        compatibility_audit_fingerprint: str | None = None,
        accepted_registry_fingerprint: str | None = None,
    ) -> None:
        self.context = context
        self.compatibility_audit_fingerprint = _valid_fingerprint(
            compatibility_audit_fingerprint, "compatibility_audit_fingerprint"
        )
        self.accepted_registry_fingerprint = _valid_fingerprint(
            accepted_registry_fingerprint, "accepted_registry_fingerprint"
        )
        implementation_sha256 = _sha256_file(Path(__file__).resolve())
        self.evaluation_contract: Mapping[str, Any] = MappingProxyType(
            {
                "schema": STAGE6_EVALUATION_CONTRACT_SCHEMA,
                "evaluator_version": STAGE6_EVALUATOR_VERSION,
                "implementation_sha256": implementation_sha256,
                "candidate_schema": ACCEPTED_REGISTRY_SCHEMA,
                "splits": list(STAGE6_SPLIT_NAMES),
                "historical_metric_reuse": "forbidden",
                "interpretation": "one_full_history_pass_through_validation",
                "label_formula": "open[t+6] / open[t+1] - 1",
                "calendar": "shared_train_anchor_every_5_global_rows_no_shift",
                "cleaning": {
                    "winsor_quantiles": [0.01, 0.99],
                    "industry": "SW_level_1_point_in_time",
                    "zscore_ddof": 0,
                    "neutralization_failure": "exclude_entire_cross_section",
                },
                "direction": {
                    "source": "train_rank_ic_only",
                    "positive": 1,
                    "negative": -1,
                    "zero_or_nonfinite": None,
                    "validation_refit_forbidden": True,
                },
                "validation_ic_sign": "raw",
                "directional_long": "train_direction_for_both_splits",
                "barra_candidate_series": "raw_top_minus_bottom",
                "barra_styles": list(STYLE_NAMES),
                "evaluation_config": asdict(context.config.evaluation),
                "barra_config": asdict(context.config.barra),
                "oos": "not_loaded_and_interface_rejected",
            }
        )
        self.evaluation_contract_fingerprint = _stable_hash(
            dict(self.evaluation_contract)
        )
        self._interpreter = FactorInterpreter(context.factor_tensor)

    def _expression(self, candidate: Mapping[str, Any]) -> tuple[Expression, dict[str, Any]]:
        forbidden = sorted(self._FORBIDDEN_HISTORICAL_FIELDS.intersection(candidate))
        if forbidden:
            raise ValueError(f"accepted candidate contains forbidden historical metrics: {forbidden}")
        if candidate.get("schema") != ACCEPTED_REGISTRY_SCHEMA:
            raise ValueError("candidate is not from the AUTO_ACCEPT registry schema")
        if candidate.get("historical_metric_reuse") != "forbidden":
            raise ValueError("candidate does not forbid historical metric reuse")
        if candidate.get("stage6_metric_recompute_required") is not True:
            raise ValueError("candidate does not require Stage 6 metric recomputation")
        try:
            expression = Expression.from_prefix(candidate["prefix_token_ids"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("accepted candidate prefix is invalid") from error
        stats = expression.stats
        identity = {
            "structural_hash": expression.structural_hash(),
            "formula": expression.to_formula(),
            "prefix_token_ids": list(expression.to_prefix()),
            "node_count": stats.node_count,
            "depth": stats.depth,
        }
        expected = {
            "structural_hash": candidate.get("current_structural_hash"),
            "formula": candidate.get("formula"),
            "prefix_token_ids": candidate.get("prefix_token_ids"),
            "node_count": candidate.get("node_count"),
            "depth": candidate.get("depth"),
        }
        if identity != expected:
            raise ValueError("accepted candidate identity no longer matches current Expression semantics")
        return expression, identity

    def resolve_candidate_identity(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate an accepted candidate and expose its deterministic identity."""

        _, identity = self._expression(candidate)
        return MappingProxyType(identity)

    def _prepare_split(
        self,
        factor: npt.NDArray[np.float64],
        split_name: Stage6SplitName,
    ) -> _PreparedSplit:
        split = self.context.get_split_data(split_name)
        compact_factor = np.asarray(factor[split.global_rebalance_rows], dtype=np.float64)
        diagnostics = NeutralizationDiagnostics()
        encoded = encode_industry_panel(split.industry_labels, compact_factor.shape)
        cleaned = clean_candidate_factor_cross_sections(
            compact_factor,
            split.industry_labels,
            split.universe_mask,
            diagnostics=diagnostics,
            encoded_industries=encoded,
        )
        ic_values = rank_ic_values_from_cleaned(
            cleaned,
            split.forward_returns,
            self.context.config.evaluation.min_cross_section_count,
        )
        ic = summarize_ic(
            ic_values, ddof=self.context.config.evaluation.performance_ddof
        )
        raw_ls = long_short_portfolio_series_from_cleaned(
            cleaned, split.forward_returns, self.context.config.evaluation
        )
        barra = calculate_barra_ts_corr(
            raw_ls.long_short_return,
            dict(split.barra_long_short),
            min_periods=self.context.config.barra.min_common_periods,
        )
        eligible_count = int(np.sum(split.universe_mask))
        finite_count = int(np.sum(split.universe_mask & np.isfinite(compact_factor)))
        result = {
            "requested_date_range": [
                split.boundary.requested_start,
                split.boundary.requested_end,
            ],
            "actual_date_range": [split.boundary.actual_start, split.boundary.actual_end],
            "rebalance_dates": [str(value) for value in split.rebalance_dates],
            "rebalance_periods": int(split.rebalance_dates.size),
            "ic": {
                "mean": _finite_or_none(ic.mean),
                "std": _finite_or_none(ic.std),
                "icir": _finite_or_none(ic.icir),
                "valid_periods": int(ic.valid_periods),
                "total_periods": int(ic.total_periods),
            },
            "long": {
                "direction": None,
                "mean_period_return": None,
                "annualized_return": None,
                "annualized_ir": None,
                "std": None,
                "valid_periods": 0,
                "total_periods": int(split.rebalance_dates.size),
                "excess_series": {
                    "dates": [str(value) for value in split.rebalance_dates],
                    "values": None,
                },
            },
            "barra": {
                "max_abs_correlation": _finite_or_none(barra.barra_ts_corr),
                "correlations": {
                    name: _finite_or_none(barra.correlations[name]) for name in STYLE_NAMES
                },
                "common_valid_periods": {
                    name: int(barra.valid_periods[name]) for name in STYLE_NAMES
                },
            },
            "raw_long_short_valid_periods": int(
                np.isfinite(raw_ls.long_short_return).sum()
            ),
            "factor_finite_coverage": {
                "finite_universe_values": finite_count,
                "eligible_universe_values": eligible_count,
                "rate": finite_count / eligible_count if eligible_count else None,
            },
            "neutralization": _diagnostics_payload(diagnostics, split),
        }
        return _PreparedSplit(result=result, cleaned_factor=cleaned, forward_returns=split.forward_returns)

    def _apply_direction(self, prepared: _PreparedSplit, direction: int | None) -> None:
        if direction is None:
            return
        series = long_portfolio_series_from_cleaned(
            prepared.cleaned_factor,
            prepared.forward_returns,
            direction,
            self.context.config.evaluation,
        )
        summary = summarize_excess_returns(series.excess_return, self.context.config.evaluation)
        prepared.result["long"] = {
            "direction": direction,
            "mean_period_return": _finite_or_none(summary.mean_period_return),
            "annualized_return": _finite_or_none(summary.annualized_return),
            "annualized_ir": _finite_or_none(summary.annualized_ir),
            "std": _finite_or_none(summary.std),
            "valid_periods": int(summary.valid_periods),
            "total_periods": int(summary.total_periods),
            "excess_series": {
                "dates": list(prepared.result["rebalance_dates"]),
                "values": _float_series(series.excess_return),
            },
        }

    def evaluate(self, candidate: Mapping[str, Any]) -> Stage6CandidateEvaluationResult:
        total_started = time.perf_counter()
        expression, expression_identity = self._expression(candidate)
        factor_started = time.perf_counter()
        factor = np.asarray(self._interpreter.evaluate(expression), dtype=np.float64)
        factor_seconds = time.perf_counter() - factor_started
        expected_shape = (self.context.dates.size, self.context.stocks.size)
        if factor.shape != expected_shape:
            raise RuntimeError(
                f"FactorInterpreter returned {factor.shape}; expected {expected_shape}"
            )

        train_started = time.perf_counter()
        train = self._prepare_split(factor, "train")
        train_ic = train.result["ic"]["mean"]
        direction = (
            infer_long_direction(float(train_ic)) if train_ic is not None and train_ic != 0.0 else None
        )
        self._apply_direction(train, direction)
        train_seconds = time.perf_counter() - train_started

        validation_started = time.perf_counter()
        validation = self._prepare_split(factor, "validation")
        self._apply_direction(validation, direction)
        validation_seconds = time.perf_counter() - validation_started

        invalid_reasons = () if direction is not None else ("train_direction_unavailable",)
        status = "completed" if direction is not None else "completed_invalid"
        coverage = {
            "train": dict(train.result["factor_finite_coverage"]),
            "validation": dict(validation.result["factor_finite_coverage"]),
        }
        deterministic_payload = {
            "schema": STAGE6_EVALUATION_RESULT_SCHEMA,
            "status": status,
            "invalid_reasons": list(invalid_reasons),
            "expression": expression_identity,
            "context_fingerprint": self.context.fingerprint,
            "evaluation_contract_fingerprint": self.evaluation_contract_fingerprint,
            "train_direction": direction,
            "train": train.result,
            "validation": validation.result,
            "factor_finite_coverage": coverage,
        }
        result_fingerprint = _stable_hash(deterministic_payload)
        total_seconds = time.perf_counter() - total_started
        return Stage6CandidateEvaluationResult(
            schema=STAGE6_EVALUATION_RESULT_SCHEMA,
            status=status,
            invalid_reasons=invalid_reasons,
            expression=MappingProxyType(expression_identity),
            source_identity=MappingProxyType(
                {
                    "compatibility_audit_fingerprint": self.compatibility_audit_fingerprint,
                    "accepted_registry_fingerprint": self.accepted_registry_fingerprint,
                    "compatibility_record_fingerprint": candidate.get(
                        "compatibility_record_fingerprint"
                    ),
                    "source_claimed_structural_hash": candidate.get(
                        "source_claimed_structural_hash"
                    ),
                    "origin_ids": list(candidate.get("origin_ids", [])),
                    "source_ids": list(candidate.get("source_ids", [])),
                }
            ),
            context_fingerprint=self.context.fingerprint,
            evaluation_contract_fingerprint=self.evaluation_contract_fingerprint,
            train_direction=direction,
            train=MappingProxyType(train.result),
            validation=MappingProxyType(validation.result),
            factor_finite_coverage=MappingProxyType(coverage),
            factor_seconds=float(factor_seconds),
            train_evaluation_seconds=float(train_seconds),
            validation_evaluation_seconds=float(validation_seconds),
            total_seconds=float(total_seconds),
            result_fingerprint=result_fingerprint,
        )


__all__ = [
    "STAGE6_EVALUATION_CONTEXT_SCHEMA",
    "STAGE6_EVALUATION_CONTRACT_SCHEMA",
    "STAGE6_EVALUATION_RESULT_SCHEMA",
    "STAGE6_EVALUATOR_VERSION",
    "STAGE6_SPLIT_NAMES",
    "Stage6CalendarEntry",
    "Stage6CandidateEvaluationResult",
    "Stage6CandidateEvaluator",
    "Stage6EvaluationConfig",
    "Stage6EvaluationContext",
    "Stage6SplitBoundary",
    "Stage6SplitData",
    "build_stage6_evaluation_context",
    "build_stage6_evaluation_context_from_arrays",
]
