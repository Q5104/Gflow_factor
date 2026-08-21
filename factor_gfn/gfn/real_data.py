"""Read-only real-data context used by the stage-four Reward adapter.

This module only aligns inputs and constructs the global five-day calendar and
five Barra long-short reference series.  It does not evaluate expressions or
train a GFlowNet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from factor_gfn.barra import (
    STYLE_NAMES,
    BarraConfig,
    BarraFactorSet,
    BarraPaths,
    LongShortSeries,
    build_barra_long_short_returns,
    summarize_long_short,
)
from factor_gfn.data.daily_derived import DAILY_DERIVED_FEATURE_NAMES
from factor_gfn.data.daily_derived_artifact import daily_derived_schema_fingerprint
from factor_gfn.data.industry import (
    INDUSTRY_SW_DAILY_METADATA_PATH,
    INDUSTRY_SW_DAILY_PATH,
    load_sw_industry_panel,
)
from factor_gfn.data.preprocess import PROCESSED_DATA_DIR
from factor_gfn.evaluator import FEATURE_NAMES, EvaluationConfig, build_forward_returns


REAL_REWARD_CONTEXT_SCHEMA = "factor_gfn.real_reward_context.v1"
RAW_DAILY_FEATURE_SPACE_ID = "raw_daily"
DAILY_DERIVED_FEATURE_SPACE_ID = "daily_derived_v1"


@dataclass(frozen=True, slots=True)
class ExpressionFeatureSpec:
    """One explicit expression-tensor artifact and its ordered leaf schema."""

    feature_space_id: str
    artifact_dir: Path
    ordered_feature_names: tuple[str, ...]
    expected_schema_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir).resolve())
        names = tuple(self.ordered_feature_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("ordered_feature_names 必须是非空有序特征名")
        if len(set(names)) != len(names):
            raise ValueError("ordered_feature_names 不允许重复")
        object.__setattr__(self, "ordered_feature_names", names)
        if self.feature_space_id == RAW_DAILY_FEATURE_SPACE_ID:
            if names != FEATURE_NAMES or self.expected_schema_fingerprint is not None:
                raise ValueError("Raw Daily 必须使用冻结的 6-feature schema")
        elif self.feature_space_id == DAILY_DERIVED_FEATURE_SPACE_ID:
            if names != DAILY_DERIVED_FEATURE_NAMES:
                raise ValueError("Daily-Derived 必须使用冻结的 16-feature order")
            if self.expected_schema_fingerprint != daily_derived_schema_fingerprint():
                raise ValueError("Daily-Derived expected schema fingerprint 不匹配")
        else:
            raise ValueError(f"未知 expression feature space：{self.feature_space_id!r}")

    @property
    def tensor_path(self) -> Path:
        return self.artifact_dir / "data_tensor.npy"

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / "metadata.json"

    @classmethod
    def raw_daily(
        cls, processed_dir: Path = PROCESSED_DATA_DIR
    ) -> ExpressionFeatureSpec:
        return cls(
            feature_space_id=RAW_DAILY_FEATURE_SPACE_ID,
            artifact_dir=processed_dir,
            ordered_feature_names=FEATURE_NAMES,
        )

    @classmethod
    def daily_derived(
        cls,
        artifact_dir: Path = PROCESSED_DATA_DIR / DAILY_DERIVED_FEATURE_SPACE_ID,
    ) -> ExpressionFeatureSpec:
        return cls(
            feature_space_id=DAILY_DERIVED_FEATURE_SPACE_ID,
            artifact_dir=artifact_dir,
            ordered_feature_names=DAILY_DERIVED_FEATURE_NAMES,
            expected_schema_fingerprint=daily_derived_schema_fingerprint(),
        )


@dataclass(frozen=True, slots=True)
class RealRewardDataPaths:
    """Files required to assemble a real Reward context."""

    processed_dir: Path = PROCESSED_DATA_DIR
    industry_path: Path = INDUSTRY_SW_DAILY_PATH
    industry_metadata_path: Path = INDUSTRY_SW_DAILY_METADATA_PATH
    expression_features: ExpressionFeatureSpec | None = None

    def __post_init__(self) -> None:
        for name in ("processed_dir", "industry_path", "industry_metadata_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.expression_features is None:
            object.__setattr__(
                self,
                "expression_features",
                ExpressionFeatureSpec.raw_daily(self.processed_dir),
            )
        elif not isinstance(self.expression_features, ExpressionFeatureSpec):
            raise TypeError("expression_features 必须是 ExpressionFeatureSpec")

    @property
    def tensor_path(self) -> Path:
        """Frozen Raw Daily tensor used for raw evaluation inputs."""
        return self.processed_dir / "data_tensor.npy"

    @property
    def expression_tensor_path(self) -> Path:
        assert self.expression_features is not None
        return self.expression_features.tensor_path

    @property
    def expression_metadata_path(self) -> Path:
        assert self.expression_features is not None
        return self.expression_features.metadata_path

    @property
    def universe_mask_path(self) -> Path:
        return self.processed_dir / "universe_mask.npy"

    @property
    def date_list_path(self) -> Path:
        return self.processed_dir / "date_list.npy"

    @property
    def stock_list_path(self) -> Path:
        return self.processed_dir / "stock_list.npy"

    @property
    def processed_metadata_path(self) -> Path:
        return self.processed_dir / "metadata.json"

    @property
    def barra_paths(self) -> BarraPaths:
        return BarraPaths(processed_dir=self.processed_dir)


@dataclass(frozen=True, slots=True)
class RealRewardDataConfig:
    """Frozen first-pass real Reward data and calendar conventions."""

    train_start: str = "2010-01-01"
    train_end: str = "2018-12-31"
    evaluation: EvaluationConfig = EvaluationConfig()
    barra: BarraConfig = BarraConfig()

    def __post_init__(self) -> None:
        try:
            start = np.datetime64(self.train_start, "D")
            end = np.datetime64(self.train_end, "D")
        except (TypeError, ValueError) as error:
            raise ValueError("train_start/train_end 必须是有效日期") from error
        if np.isnat(start) or np.isnat(end) or start > end:
            raise ValueError("训练日期边界无效或顺序颠倒")
        if self.evaluation.rebalance_interval != 5:
            raise ValueError("第一版真实 Reward 要求 rebalance_interval=5")
        if self.evaluation.min_cross_section_count != self.barra.min_cross_section_count:
            raise ValueError("Evaluation 与 Barra 的最小截面样本数必须一致")
        if not np.isclose(self.evaluation.long_quantile, self.barra.long_short_quantile):
            raise ValueError("Evaluation 与 Barra 的多空分位比例必须一致")


@dataclass(frozen=True, slots=True)
class RealRewardDataContext:
    """Aligned arrays and diagnostics for a real Reward run.

    ``expression_feature_tensor`` contains the selected Feature Space history
    through the training end. ``factor_tensor`` remains its compatibility alias.
    ``evaluation_factor_rows`` selects the rows that enter Reward evaluation.
    No validation-period feature or return is exposed through this context.
    """

    factor_tensor: npt.NDArray[np.floating]
    config: RealRewardDataConfig
    history_dates: npt.NDArray[np.datetime64]
    evaluation_factor_rows: npt.NDArray[np.int64]
    evaluation_dates: npt.NDArray[np.datetime64]
    stocks: npt.NDArray[np.str_]
    forward_returns: npt.NDArray[np.float64]
    universe_mask: npt.NDArray[np.bool_]
    industry_labels: npt.NDArray[np.int32]
    rebalance_indices: npt.NDArray[np.int64]
    barra_long_short: Mapping[str, LongShortSeries]
    fingerprint: str
    manifest: Mapping[str, Any]
    ordered_feature_names: tuple[str, ...] = FEATURE_NAMES
    expression_feature_space_id: str = RAW_DAILY_FEATURE_SPACE_ID

    @property
    def expression_feature_tensor(self) -> npt.NDArray[np.floating]:
        return self.factor_tensor


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"真实 Reward 数据上下文缺少输入：{missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    values.setflags(write=False)
    return values


def validate_expression_feature_artifact(
    paths: RealRewardDataPaths,
    tensor: npt.NDArray[Any],
) -> Mapping[str, Any]:
    spec = paths.expression_features
    assert spec is not None
    metadata = json.loads(paths.expression_metadata_path.read_text(encoding="utf-8"))
    if tuple(metadata.get("feature_order", ())) != spec.ordered_feature_names:
        raise ValueError("expression metadata feature order 与显式 schema 不一致")
    if tensor.ndim != 3:
        raise ValueError("expression tensor 必须使用 (date, feature, stock) 三维布局")
    if tensor.shape[1] != len(spec.ordered_feature_names):
        raise ValueError("expression tensor feature count 与显式 schema 不一致")
    if spec.feature_space_id == DAILY_DERIVED_FEATURE_SPACE_ID:
        if metadata.get("status") != "completed":
            raise ValueError("Daily-Derived metadata 未标记 completed")
        if metadata.get("feature_space_id") != DAILY_DERIVED_FEATURE_SPACE_ID:
            raise ValueError("Daily-Derived metadata feature_space_id 不匹配")
        if metadata.get("feature_count") != len(spec.ordered_feature_names):
            raise ValueError("Daily-Derived metadata feature_count 不匹配")
        if tuple(metadata.get("shape", ())) != tensor.shape:
            raise ValueError("Daily-Derived metadata shape 与 tensor 不一致")
        if metadata.get("builder_schema_fingerprint") != spec.expected_schema_fingerprint:
            raise ValueError("Daily-Derived metadata schema fingerprint 不匹配")
        axes = metadata.get("axes", {})
        if axes.get("date", {}).get("sha256") != _sha256_file(paths.date_list_path):
            raise ValueError("Daily-Derived date axis fingerprint 不匹配")
        if axes.get("stock", {}).get("sha256") != _sha256_file(paths.stock_list_path):
            raise ValueError("Daily-Derived stock axis fingerprint 不匹配")
    return metadata


def build_global_rebalance_indices(
    exposures: Mapping[str, npt.ArrayLike],
    forward_returns: npt.ArrayLike,
    universe_mask: npt.ArrayLike,
    config: RealRewardDataConfig = RealRewardDataConfig(),
) -> npt.NDArray[np.int64]:
    """Build one calendar shared by every candidate and all five Barra styles."""

    returns = np.asarray(forward_returns, dtype=np.float64)
    universe = np.asarray(universe_mask, dtype=bool)
    if returns.ndim != 2 or returns.shape != universe.shape or 0 in returns.shape:
        raise ValueError("forward_returns 与 universe_mask 必须是同形非空二维矩阵")
    if set(exposures) != set(STYLE_NAMES):
        raise ValueError("exposures 必须且只能包含五个第一版 Barra 风格")

    eligible = np.ones(returns.shape[0], dtype=bool)
    for name in STYLE_NAMES:
        values = np.asarray(exposures[name])
        if values.shape != returns.shape:
            raise ValueError(f"{name} 暴露形状与 forward_returns 不一致")
        counts = np.sum(
            universe & np.isfinite(returns) & np.isfinite(values),
            axis=1,
            dtype=np.int64,
        )
        eligible &= counts >= config.evaluation.min_cross_section_count

    candidates = np.flatnonzero(eligible)
    if candidates.size == 0:
        raise ValueError("训练区间不存在五个 Barra 风格同时可评价的日期")
    first = int(candidates[0]) + config.evaluation.rebalance_offset
    if first >= returns.shape[0]:
        raise ValueError("rebalance_offset 使首个全局评价日越界")
    last = int(candidates[-1])
    indices = np.arange(
        first,
        last + 1,
        config.evaluation.rebalance_interval,
        dtype=np.int64,
    )
    return _freeze(indices)


def _series_diagnostics(
    series_by_style: Mapping[str, LongShortSeries],
    rebalance_indices: npt.NDArray[np.int64],
    config: RealRewardDataConfig,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    annualization = 252.0 / config.evaluation.rebalance_interval
    for name in STYLE_NAMES:
        values = series_by_style[name].long_short_return[rebalance_indices]
        summary = summarize_long_short(
            values,
            annualization=annualization,
            ddof=config.evaluation.performance_ddof,
        )
        finite = values[np.isfinite(values)]
        if summary.valid_periods < config.barra.min_common_periods:
            raise ValueError(
                f"{name} Barra LS 仅有 {summary.valid_periods} 个有效期，"
                f"少于 {config.barra.min_common_periods}"
            )
        if finite.size < 2 or float(np.std(finite, ddof=0)) <= np.finfo(np.float64).eps:
            raise ValueError(f"{name} Barra LS 为常数或有效样本不足")
        wealth = float(np.prod(1.0 + finite) - 1.0)
        result[name] = {
            "valid_periods": summary.valid_periods,
            "mean_period_return": summary.mean_period_return,
            "period_std": summary.period_std,
            "annualized_ir": summary.annualized_ir,
            "cumulative_return": wealth,
        }
    return result


def build_real_reward_data_context(
    config: RealRewardDataConfig = RealRewardDataConfig(),
    paths: RealRewardDataPaths = RealRewardDataPaths(),
) -> RealRewardDataContext:
    """Load and validate all real Reward inputs without evaluating expressions."""

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
        barra_paths.market_return_path,
        *[barra_paths.exposure_path(name) for name in STYLE_NAMES],
    ]
    _require_files(required)

    raw_tensor = np.load(paths.tensor_path, mmap_mode="r", allow_pickle=False)
    expression_tensor = (
        raw_tensor
        if paths.expression_tensor_path == paths.tensor_path
        else np.load(paths.expression_tensor_path, mmap_mode="r", allow_pickle=False)
    )
    universe_all = np.load(paths.universe_mask_path, mmap_mode="r", allow_pickle=False)
    dates = np.load(paths.date_list_path, allow_pickle=False).astype("datetime64[D]")
    stocks = np.load(paths.stock_list_path, allow_pickle=False).astype(str)
    expected = (dates.size, stocks.size)
    if raw_tensor.ndim != 3 or raw_tensor.shape != (dates.size, 6, stocks.size):
        raise ValueError("data_tensor 必须与 date/stock 轴形成 (date, 6, stock)")
    validate_expression_feature_artifact(paths, expression_tensor)
    spec = paths.expression_features
    assert spec is not None
    if (
        expression_tensor.shape[0] != dates.size
        or expression_tensor.shape[2] != stocks.size
    ):
        raise ValueError("expression tensor 与 Raw date/stock 轴不一致")
    if universe_all.shape != expected:
        raise ValueError("universe_mask 与 date/stock 轴不一致")
    if np.unique(dates).size != dates.size or not np.all(dates[:-1] < dates[1:]):
        raise ValueError("date_list 必须唯一且严格升序")
    if np.unique(stocks).size != stocks.size or not np.all(stocks[:-1] < stocks[1:]):
        raise ValueError("stock_list 必须唯一且严格升序")

    start = np.datetime64(config.train_start, "D")
    end = np.datetime64(config.train_end, "D")
    evaluation_rows_full = np.flatnonzero((dates >= start) & (dates <= end))
    if evaluation_rows_full.size == 0:
        raise ValueError("处理后数据不覆盖指定训练区间")
    if not np.all(np.diff(evaluation_rows_full) == 1):
        raise ValueError("训练区间在 date_list 中必须是连续切片")
    start_row = int(evaluation_rows_full[0])
    end_row = int(evaluation_rows_full[-1])

    history_dates = _freeze(dates[: end_row + 1].copy())
    factor_tensor = expression_tensor[: end_row + 1]
    evaluation_factor_rows = _freeze(
        np.arange(start_row, end_row + 1, dtype=np.int64)
    )
    evaluation_dates = _freeze(dates[start_row : end_row + 1].copy())
    universe = universe_all[start_row : end_row + 1]
    forward_returns = build_forward_returns(
        raw_tensor[start_row : end_row + 1, 0, :], config.evaluation
    )
    _freeze(forward_returns)
    industry = _freeze(
        load_sw_industry_panel(
            evaluation_dates,
            stocks,
            level=1,
            path=paths.industry_path,
        )
    )
    if industry.shape != forward_returns.shape or universe.shape != forward_returns.shape:
        raise ValueError("行业、股票池和未来收益矩阵未严格对齐")

    exposures: dict[str, npt.NDArray[np.floating]] = {}
    for name in STYLE_NAMES:
        values = np.load(
            barra_paths.exposure_path(name), mmap_mode="r", allow_pickle=False
        )
        if values.shape != expected:
            raise ValueError(f"{name} Barra 暴露与 date/stock 轴不一致")
        exposures[name] = values[start_row : end_row + 1]
    market_return = np.load(
        barra_paths.market_return_path, mmap_mode="r", allow_pickle=False
    )
    if market_return.shape != (dates.size,):
        raise ValueError("market_return 与 date 轴不一致")

    rebalance_indices = build_global_rebalance_indices(
        exposures,
        forward_returns,
        universe,
        config,
    )
    factor_set = BarraFactorSet(
        exposures=exposures,
        market_return=market_return[start_row : end_row + 1],
    )
    barra_long_short = build_barra_long_short_returns(
        factor_set,
        forward_returns,
        rebalance_indices,
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
            _freeze(values)
    diagnostics = _series_diagnostics(barra_long_short, rebalance_indices, config)

    universe_count = int(np.sum(universe))
    universe_missing_count = int(np.sum((industry < 0) & universe))

    sources = {
        "processed_metadata_sha256": _sha256_file(paths.processed_metadata_path),
        "industry_metadata_sha256": _sha256_file(paths.industry_metadata_path),
        "barra_metadata_sha256": _sha256_file(barra_paths.metadata_path),
        "date_list_sha256": _sha256_file(paths.date_list_path),
        "stock_list_sha256": _sha256_file(paths.stock_list_path),
        "data_tensor_bytes": paths.tensor_path.stat().st_size,
        "universe_mask_bytes": paths.universe_mask_path.stat().st_size,
    }
    manifest: dict[str, Any] = {
        "schema": REAL_REWARD_CONTEXT_SCHEMA,
        "requested_train_start": config.train_start,
        "requested_train_end": config.train_end,
        "actual_train_start": str(evaluation_dates[0]),
        "actual_train_end": str(evaluation_dates[-1]),
        "label_formula": "open[t+6] / open[t+1] - 1",
        "label_exit_must_not_exceed": config.train_end,
        "calendar": {
            "rule": "global_first_date_all_five_barra_styles_evaluable_then_every_5_trading_days_through_last_evaluable_date",
            "first_rebalance_date": str(evaluation_dates[rebalance_indices[0]]),
            "last_rebalance_date": str(evaluation_dates[rebalance_indices[-1]]),
            "rebalance_periods": int(rebalance_indices.size),
        },
        "shape": {
            "history_tensor": list(map(int, factor_tensor.shape)),
            "evaluation": list(map(int, forward_returns.shape)),
        },
        "industry": {
            "missing_code": -1,
            "dense_grid_missing_count": int(np.sum(industry < 0)),
            "dense_grid_missing_rate": float(np.mean(industry < 0)),
            "universe_rows": universe_count,
            "universe_missing_count": universe_missing_count,
            "universe_missing_rate": (
                universe_missing_count / universe_count if universe_count else None
            ),
        },
        "evaluation_config": asdict(config.evaluation),
        "barra_config": asdict(config.barra),
        "barra_long_short": diagnostics,
        "sources": sources,
    }
    if spec.feature_space_id != RAW_DAILY_FEATURE_SPACE_ID:
        sources.update(
            {
                "expression_metadata_sha256": _sha256_file(
                    paths.expression_metadata_path
                ),
                "expression_data_tensor_bytes": (
                    paths.expression_tensor_path.stat().st_size
                ),
            }
        )
        manifest["expression_features"] = {
            "feature_space_id": spec.feature_space_id,
            "ordered_feature_names": list(spec.ordered_feature_names),
            "schema_fingerprint": spec.expected_schema_fingerprint,
        }
    payload = json.dumps(
        manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return RealRewardDataContext(
        factor_tensor=factor_tensor,
        config=config,
        history_dates=history_dates,
        evaluation_factor_rows=evaluation_factor_rows,
        evaluation_dates=evaluation_dates,
        stocks=_freeze(stocks.copy()),
        forward_returns=forward_returns,
        universe_mask=universe,
        industry_labels=industry,
        rebalance_indices=rebalance_indices,
        barra_long_short=barra_long_short,
        fingerprint=fingerprint,
        manifest=manifest,
        ordered_feature_names=spec.ordered_feature_names,
        expression_feature_space_id=spec.feature_space_id,
    )


__all__ = [
    "DAILY_DERIVED_FEATURE_SPACE_ID",
    "REAL_REWARD_CONTEXT_SCHEMA",
    "RAW_DAILY_FEATURE_SPACE_ID",
    "ExpressionFeatureSpec",
    "RealRewardDataConfig",
    "RealRewardDataContext",
    "RealRewardDataPaths",
    "build_global_rebalance_indices",
    "build_real_reward_data_context",
    "validate_expression_feature_artifact",
]
