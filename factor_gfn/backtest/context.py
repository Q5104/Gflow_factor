"""Stage-five data splits, shared rebalance calendar, and leakage guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
import numpy.typing as npt

from factor_gfn.barra import STYLE_NAMES, BarraConfig
from factor_gfn.data.industry import load_sw_industry_panel
from factor_gfn.evaluator import (
    FEATURE_NAMES,
    EvaluationConfig,
    build_forward_returns,
)
from factor_gfn.gfn.real_data import (
    RAW_DAILY_FEATURE_SPACE_ID,
    RealRewardDataPaths,
    validate_expression_feature_artifact,
)


STAGE5_CONTEXT_SCHEMA = "factor_gfn.stage5_context.v1"
SplitName = Literal["train", "validation", "oos"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "oos")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    values.setflags(write=False)
    return values


@dataclass(frozen=True, slots=True)
class Stage5DataConfig:
    """Frozen stage-five split and label conventions."""

    train_start: str = "2010-01-01"
    train_end: str = "2018-12-31"
    validation_start: str = "2019-01-01"
    validation_end: str = "2020-12-31"
    oos_start: str = "2021-01-01"
    oos_end: str = "2025-12-31"
    evaluation: EvaluationConfig = EvaluationConfig()
    barra: BarraConfig = BarraConfig()

    def __post_init__(self) -> None:
        boundaries: list[tuple[np.datetime64, np.datetime64]] = []
        for name in SPLIT_NAMES:
            try:
                start = np.datetime64(getattr(self, f"{name}_start"), "D")
                end = np.datetime64(getattr(self, f"{name}_end"), "D")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} 日期边界必须是有效日期") from error
            if np.isnat(start) or np.isnat(end) or start > end:
                raise ValueError(f"{name} 日期边界无效或顺序颠倒")
            boundaries.append((start, end))
        if not (
            boundaries[0][1] < boundaries[1][0]
            and boundaries[1][1] < boundaries[2][0]
        ):
            raise ValueError("train、validation 和 oos 必须严格有序且互不重叠")
        one_day = np.timedelta64(1, "D")
        if not (
            boundaries[0][1] + one_day == boundaries[1][0]
            and boundaries[1][1] + one_day == boundaries[2][0]
        ):
            raise ValueError("三个请求区间必须按自然日连续，不能留下未归属日期")
        if self.evaluation.horizon != 5 or self.evaluation.entry_lag != 1:
            raise ValueError("阶段 5 第一版固定使用 open[t+6] / open[t+1] - 1")
        if self.evaluation.rebalance_interval != 5:
            raise ValueError("阶段 5 第一版固定使用每 5 个交易日调仓")
        if self.evaluation.rebalance_offset != 0:
            raise ValueError("阶段 5 全局日历锚点固定后不再应用额外 offset")
        if self.evaluation.min_cross_section_count != self.barra.min_cross_section_count:
            raise ValueError("Evaluation 与 Barra 的最小截面样本数必须一致")
        if not np.isclose(
            self.evaluation.long_quantile, self.barra.long_short_quantile
        ):
            raise ValueError("Evaluation 与 Barra 的多空分位比例必须一致")


@dataclass(frozen=True, slots=True)
class Stage5SplitBoundary:
    name: SplitName
    requested_start: str
    requested_end: str
    actual_start: str
    actual_end: str
    start_row: int
    end_row: int
    scheduled_periods: int
    included_periods: int


@dataclass(frozen=True, slots=True)
class RebalanceCalendarEntry:
    sequence_id: int
    split: SplitName
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
class Stage5SplitData:
    """Arrays exposed for one split; factor history always begins at row zero."""

    boundary: Stage5SplitBoundary
    factor_tensor: npt.NDArray[np.floating]
    history_dates: npt.NDArray[np.datetime64]
    evaluation_factor_rows: npt.NDArray[np.int64]
    evaluation_dates: npt.NDArray[np.datetime64]
    forward_returns: npt.NDArray[np.float64]
    universe_mask: npt.NDArray[np.bool_]
    industry_labels: npt.NDArray[np.int32]
    barra_exposures: Mapping[str, npt.NDArray[np.floating]]
    rebalance_indices: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Stage5DataContext:
    config: Stage5DataConfig
    dates: npt.NDArray[np.datetime64]
    stocks: npt.NDArray[np.str_]
    splits: Mapping[SplitName, Stage5SplitBoundary]
    calendar: tuple[RebalanceCalendarEntry, ...]
    fingerprint: str
    manifest: Mapping[str, Any]
    _factor_tensor: npt.NDArray[np.floating] = field(repr=False)
    _forward_returns: npt.NDArray[np.float64] = field(repr=False)
    _universe_mask: npt.NDArray[np.bool_] = field(repr=False)
    _industry_labels: npt.NDArray[np.int32] = field(repr=False)
    _barra_exposures: Mapping[str, npt.NDArray[np.floating]] = field(repr=False)
    ordered_feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def expression_feature_tensor(self) -> npt.NDArray[np.floating]:
        return self._factor_tensor

    def get_split_data(
        self,
        split: SplitName,
        *,
        frozen_selection_fingerprint: str | None = None,
    ) -> Stage5SplitData:
        """Return a split view; OOS stays locked until a selection is frozen."""

        if split not in SPLIT_NAMES:
            raise ValueError(f"未知阶段 5 分段：{split!r}")
        if split == "oos":
            fingerprint = frozen_selection_fingerprint or ""
            if len(fingerprint) != 64 or any(
                character not in "0123456789abcdef" for character in fingerprint
            ):
                raise PermissionError("最终样本外要求已冻结的 64 位选择指纹")
        boundary = self.splits[split]
        start = boundary.start_row
        stop = boundary.end_row + 1
        history_stop = stop
        evaluation_rows = _readonly(np.arange(start, stop, dtype=np.int64))
        rebalance_rows = np.array(
            [
                entry.signal_row - start
                for entry in self.calendar
                if entry.split == split and entry.included
            ],
            dtype=np.int64,
        )
        return Stage5SplitData(
            boundary=boundary,
            factor_tensor=self._factor_tensor[:history_stop],
            history_dates=_readonly(self.dates[:history_stop].copy()),
            evaluation_factor_rows=evaluation_rows,
            evaluation_dates=_readonly(self.dates[start:stop].copy()),
            forward_returns=self._forward_returns[start:stop],
            universe_mask=self._universe_mask[start:stop],
            industry_labels=self._industry_labels[start:stop],
            barra_exposures=MappingProxyType(
                {name: values[start:stop] for name, values in self._barra_exposures.items()}
            ),
            rebalance_indices=_readonly(rebalance_rows),
        )


def _split_rows(
    dates: npt.NDArray[np.datetime64], config: Stage5DataConfig
) -> dict[SplitName, npt.NDArray[np.int64]]:
    result: dict[SplitName, npt.NDArray[np.int64]] = {}
    for name in SPLIT_NAMES:
        start = np.datetime64(getattr(config, f"{name}_start"), "D")
        end = np.datetime64(getattr(config, f"{name}_end"), "D")
        rows = np.flatnonzero((dates >= start) & (dates <= end)).astype(np.int64)
        if rows.size == 0:
            raise ValueError(f"处理后数据不覆盖 {name} 请求区间")
        if not np.all(np.diff(rows) == 1):
            raise ValueError(f"{name} 在 date_list 中必须是连续切片")
        result[name] = rows
    return result


def build_stage5_context_from_arrays(
    *,
    dates: npt.ArrayLike,
    stocks: npt.ArrayLike,
    factor_tensor: npt.ArrayLike,
    raw_open: npt.ArrayLike | None = None,
    ordered_feature_names: tuple[str, ...] = FEATURE_NAMES,
    universe_mask: npt.ArrayLike,
    industry_labels: npt.ArrayLike,
    barra_exposures: Mapping[str, npt.ArrayLike],
    config: Stage5DataConfig = Stage5DataConfig(),
    source_manifest: Mapping[str, Any] | None = None,
) -> Stage5DataContext:
    """Build the stage-five contract from already aligned arrays."""

    date_values = np.asarray(dates).astype("datetime64[D]")
    stock_values = np.asarray(stocks).astype(str)
    tensor = np.asarray(factor_tensor)
    feature_names = tuple(ordered_feature_names)
    universe = np.asarray(universe_mask, dtype=bool)
    industry = np.asarray(industry_labels, dtype=np.int32)
    if date_values.ndim != 1 or stock_values.ndim != 1:
        raise ValueError("dates 和 stocks 必须是一维")
    if date_values.size == 0 or stock_values.size == 0:
        raise ValueError("dates 和 stocks 不能为空")
    if np.unique(date_values).size != date_values.size or not np.all(
        date_values[:-1] < date_values[1:]
    ):
        raise ValueError("dates 必须唯一且严格升序")
    if np.unique(stock_values).size != stock_values.size:
        raise ValueError("stocks 必须唯一")
    expected = (date_values.size, stock_values.size)
    if tensor.ndim != 3 or tensor.shape[0] != expected[0] or tensor.shape[2] != expected[1]:
        raise ValueError("factor_tensor 必须使用 (date, feature, stock) 轴")
    if (
        not feature_names
        or len(set(feature_names)) != len(feature_names)
        or any(not isinstance(name, str) or not name for name in feature_names)
    ):
        raise ValueError("ordered_feature_names 必须是唯一的非空字符串")
    if tensor.shape[1] != len(feature_names):
        raise ValueError("factor_tensor feature count 必须匹配 ordered_feature_names")
    if raw_open is None:
        if feature_names != FEATURE_NAMES:
            raise ValueError("非 Raw expression tensor 必须显式提供 raw_open labels")
        raw_open_values = tensor[:, 0, :]
    else:
        raw_open_values = np.asarray(raw_open)
        if raw_open_values.shape != expected:
            raise ValueError("raw_open 必须与 Raw date/stock 轴对齐")
    if universe.shape != expected or industry.shape != expected:
        raise ValueError("universe、industry 与 date/stock 轴不一致")
    if set(barra_exposures) != set(STYLE_NAMES):
        raise ValueError("barra_exposures 必须且只能包含五个第一版风格")
    exposures = {
        name: np.asarray(barra_exposures[name]) for name in STYLE_NAMES
    }
    if any(values.shape != expected for values in exposures.values()):
        raise ValueError("所有 Barra 暴露必须与 date/stock 轴一致")

    rows_by_split = _split_rows(date_values, config)
    oos_end_row = int(rows_by_split["oos"][-1])
    date_values = _readonly(date_values[: oos_end_row + 1].copy())
    tensor = _readonly(tensor[: oos_end_row + 1])
    universe = _readonly(universe[: oos_end_row + 1])
    industry = _readonly(industry[: oos_end_row + 1])
    exposures = {
        name: _readonly(values[: oos_end_row + 1])
        for name, values in exposures.items()
    }
    rows_by_split = _split_rows(date_values, config)

    forward_returns = build_forward_returns(
        raw_open_values[: oos_end_row + 1], config.evaluation
    )
    split_by_row: dict[int, SplitName] = {}
    for name, rows in rows_by_split.items():
        split_by_row.update({int(row): name for row in rows})

    exit_offset = config.evaluation.entry_lag + config.evaluation.horizon
    label_within = np.zeros(date_values.size, dtype=bool)
    for name, rows in rows_by_split.items():
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
    anchor_candidates = train_rows[label_within[train_rows] & barra_eligible[train_rows]]
    if anchor_candidates.size == 0:
        raise ValueError("训练期不存在五类 Barra 和区间内标签均有效的日历锚点")
    anchor = int(anchor_candidates[0])
    scheduled_rows = np.arange(
        anchor,
        int(rows_by_split["oos"][-1]) + 1,
        config.evaluation.rebalance_interval,
        dtype=np.int64,
    )
    calendar: list[RebalanceCalendarEntry] = []
    for sequence_id, signal_row in enumerate(scheduled_rows):
        row = int(signal_row)
        split = split_by_row[row]
        entry_row = row + config.evaluation.entry_lag
        exit_row = row + exit_offset
        contained = bool(label_within[row])
        reasons: list[str] = []
        if not contained:
            reasons.append("label_crosses_split")
        if not bool(barra_eligible[row]):
            reasons.append("barra_cross_section_insufficient")
        calendar.append(
            RebalanceCalendarEntry(
                sequence_id=sequence_id,
                split=split,
                signal_row=row,
                entry_row=entry_row if entry_row < date_values.size else None,
                exit_row=exit_row if exit_row < date_values.size else None,
                signal_date=str(date_values[row]),
                entry_date=(
                    str(date_values[entry_row]) if entry_row < date_values.size else None
                ),
                exit_date=(
                    str(date_values[exit_row]) if exit_row < date_values.size else None
                ),
                label_within_split=contained,
                barra_eligible=bool(barra_eligible[row]),
                included=not reasons,
                exclusion_reasons=tuple(reasons),
            )
        )

    boundaries: dict[SplitName, Stage5SplitBoundary] = {}
    for name, rows in rows_by_split.items():
        entries = [entry for entry in calendar if entry.split == name]
        boundaries[name] = Stage5SplitBoundary(
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
    calendar_fingerprint = _stable_hash(calendar_payload)
    manifest: dict[str, Any] = {
        "schema": STAGE5_CONTEXT_SCHEMA,
        "config": asdict(config),
        "label_formula": "open[t+6] / open[t+1] - 1",
        "calendar": {
            "rule": "single_train_anchor_then_every_5_global_trading_rows_no_shift",
            "anchor_row": anchor,
            "anchor_date": str(date_values[anchor]),
            "scheduled_periods": len(calendar),
            "included_periods": sum(entry.included for entry in calendar),
            "fingerprint": calendar_fingerprint,
        },
        "splits": {name: asdict(boundary) for name, boundary in boundaries.items()},
        "shape": {
            "factor_tensor": list(map(int, tensor.shape)),
            "date_stock": list(map(int, universe.shape)),
        },
        "oos": {
            "locked": True,
            "candidate_evaluation_count": 0,
        },
        "sources": dict(source_manifest or {}),
    }
    if feature_names != FEATURE_NAMES:
        manifest["expression_features"] = {
            "ordered_feature_names": list(feature_names),
            "label_source": "explicit_raw_open",
        }
    fingerprint = _stable_hash(manifest)
    return Stage5DataContext(
        config=config,
        dates=date_values,
        stocks=_readonly(stock_values.copy()),
        splits=MappingProxyType(boundaries),
        calendar=tuple(calendar),
        fingerprint=fingerprint,
        manifest=MappingProxyType(manifest),
        _factor_tensor=tensor,
        _forward_returns=forward_returns,
        _universe_mask=universe,
        _industry_labels=industry,
        _barra_exposures=MappingProxyType(exposures),
        ordered_feature_names=feature_names,
    )


def build_stage5_data_context(
    config: Stage5DataConfig = Stage5DataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> Stage5DataContext:
    """Load the existing processed inputs read-only and build stage-five context."""

    barra_paths = paths.barra_paths
    required = [
        paths.tensor_path,
        paths.expression_tensor_path,
        paths.expression_metadata_path,
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
        raise FileNotFoundError(f"阶段 5 数据上下文缺少输入：{missing}")

    dates = np.load(paths.date_list_path, allow_pickle=False).astype("datetime64[D]")
    stocks = np.load(paths.stock_list_path, allow_pickle=False).astype(str)
    raw_tensor = np.load(paths.tensor_path, mmap_mode="r", allow_pickle=False)
    expression_tensor = np.load(
        paths.expression_tensor_path, mmap_mode="r", allow_pickle=False
    )
    validate_expression_feature_artifact(paths, expression_tensor)
    if (
        expression_tensor.shape[0] != dates.size
        or expression_tensor.shape[2] != stocks.size
    ):
        raise ValueError("expression tensor 与 Raw date/stock 轴不一致")
    tensor = (
        raw_tensor
        if paths.expression_tensor_path == paths.tensor_path
        else expression_tensor
    )
    universe = np.load(paths.universe_mask_path, mmap_mode="r", allow_pickle=False)
    industry = load_sw_industry_panel(dates, stocks, level=1, path=paths.industry_path)
    exposures = {
        name: np.load(
            barra_paths.exposure_path(name), mmap_mode="r", allow_pickle=False
        )
        for name in STYLE_NAMES
    }
    sources = {
        "processed_metadata_sha256": _sha256_file(paths.processed_metadata_path),
        "industry_metadata_sha256": _sha256_file(paths.industry_metadata_path),
        "barra_metadata_sha256": _sha256_file(barra_paths.metadata_path),
        "date_list_sha256": _sha256_file(paths.date_list_path),
        "stock_list_sha256": _sha256_file(paths.stock_list_path),
        "data_tensor_bytes": paths.tensor_path.stat().st_size,
        "universe_mask_bytes": paths.universe_mask_path.stat().st_size,
    }
    spec = paths.expression_features
    assert spec is not None
    if spec.feature_space_id != RAW_DAILY_FEATURE_SPACE_ID:
        sources.update(
            {
                "expression_metadata_sha256": _sha256_file(
                    paths.expression_metadata_path
                ),
                "expression_feature_space_id": spec.feature_space_id,
                "expression_schema_fingerprint": spec.expected_schema_fingerprint,
                "expression_data_tensor_bytes": (
                    paths.expression_tensor_path.stat().st_size
                ),
            }
        )
    return build_stage5_context_from_arrays(
        dates=dates,
        stocks=stocks,
        factor_tensor=tensor,
        raw_open=raw_tensor[:, 0, :],
        ordered_feature_names=spec.ordered_feature_names,
        universe_mask=universe,
        industry_labels=industry,
        barra_exposures=exposures,
        config=config,
        source_manifest=sources,
    )


__all__ = [
    "STAGE5_CONTEXT_SCHEMA",
    "SPLIT_NAMES",
    "RebalanceCalendarEntry",
    "Stage5DataConfig",
    "Stage5DataContext",
    "Stage5SplitBoundary",
    "Stage5SplitData",
    "build_stage5_context_from_arrays",
    "build_stage5_data_context",
]
