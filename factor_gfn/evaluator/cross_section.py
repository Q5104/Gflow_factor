"""Shared cross-sectional cleaning for candidate and Barra factor evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class CrossSectionalCleaningConfig:
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    zscore_ddof: int = 0
    min_count: int = 2
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.winsor_lower < self.winsor_upper <= 1.0:
            raise ValueError("winsor 分位数必须满足 0 <= lower < upper <= 1")
        if self.zscore_ddof < 0:
            raise ValueError("zscore_ddof 不能为负数")
        if self.min_count < 2:
            raise ValueError("min_count 至少为 2")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon 必须是有限正数")


DEFAULT_CLEANING_CONFIG = CrossSectionalCleaningConfig()


class IndustryNeutralizationWarning(UserWarning):
    """A candidate date was excluded because industry neutralization failed."""


@dataclass(frozen=True, slots=True)
class NeutralizationSkipDetail:
    """Auditable reason and sample counts for one excluded matrix row."""

    row_index: int
    factor_valid_count: int
    known_industry_count: int
    industry_count: int
    required_regression_count: int
    reason: str


@dataclass(slots=True)
class NeutralizationDiagnostics:
    """Collect unique matrix rows where requested neutralization failed closed."""

    skipped_details: dict[int, NeutralizationSkipDetail] = field(default_factory=dict)

    @property
    def skipped_rows(self) -> set[int]:
        return set(self.skipped_details)

    def record_skip(
        self,
        row_index: int,
        *,
        factor_valid_count: int,
        known_industry_count: int,
        industry_count: int,
        reason: str,
    ) -> None:
        detail = NeutralizationSkipDetail(
            row_index=int(row_index),
            factor_valid_count=int(factor_valid_count),
            known_industry_count=int(known_industry_count),
            industry_count=int(industry_count),
            required_regression_count=int(industry_count) + 1,
            reason=str(reason),
        )
        existing = self.skipped_details.get(detail.row_index)
        if existing is not None and existing != detail:
            raise RuntimeError(
                f"日期索引 {detail.row_index} 出现不一致的行业中性化失败诊断"
            )
        self.skipped_details[detail.row_index] = detail


@dataclass(frozen=True, slots=True)
class EncodedIndustryPanel:
    """Point-in-time industry labels encoded once for repeated candidates."""

    codes: npt.NDArray[np.int32]


def encode_industry_panel(
    industry_labels: npt.ArrayLike,
    shape: tuple[int, int],
) -> EncodedIndustryPanel:
    """Normalize labels once; ``-1`` denotes an unknown point-in-time label."""

    labels = _industry_matrix(industry_labels, shape)
    codes = np.full(shape, -1, dtype=np.int32)
    for date_index in range(shape[0]):
        normalized = [_valid_industry_label(value) for value in labels[date_index]]
        known_positions = np.fromiter(
            (index for index, label in enumerate(normalized) if label is not None),
            dtype=np.int64,
        )
        if known_positions.size == 0:
            continue
        known_labels = np.asarray(
            [normalized[index] for index in known_positions],
            dtype=object,
        )
        _, inverse = np.unique(known_labels, return_inverse=True)
        codes[date_index, known_positions] = inverse.astype(np.int32, copy=False)
    codes.setflags(write=False)
    return EncodedIndustryPanel(codes=codes)


def _validate_inputs(
    factor: npt.ArrayLike,
    universe_mask: npt.ArrayLike | None,
    row_indices: npt.ArrayLike | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    values = np.asarray(factor, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("factor 必须是非空 (date, stock) 二维矩阵")
    if universe_mask is None:
        universe = np.ones(values.shape, dtype=bool)
    else:
        universe = np.asarray(universe_mask, dtype=bool)
        if universe.shape != values.shape:
            raise ValueError("universe_mask 必须与 factor 同形")
    if row_indices is None:
        rows = np.arange(values.shape[0], dtype=np.int64)
    else:
        rows = np.asarray(row_indices)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("row_indices 必须是一维整数数组")
        rows = rows.astype(np.int64, copy=False)
        if rows.size and ((rows < 0).any() or (rows >= values.shape[0]).any()):
            raise IndexError("row_indices 包含越界日期")
        if np.unique(rows).size != rows.size:
            raise ValueError("row_indices 不能重复")
    return values, universe, rows


def _winsorize(
    values: npt.NDArray[np.float64],
    config: CrossSectionalCleaningConfig,
) -> npt.NDArray[np.float64]:
    lower, upper = np.quantile(
        values,
        [config.winsor_lower, config.winsor_upper],
    )
    return np.clip(values, lower, upper)


def _zscore(
    values: npt.NDArray[np.float64],
    config: CrossSectionalCleaningConfig,
) -> npt.NDArray[np.float64] | None:
    mean = float(values.mean())
    std = float(values.std(ddof=config.zscore_ddof))
    if not np.isfinite(std) or std <= config.epsilon:
        return None
    return (values - mean) / std


def _industry_matrix(
    industry_labels: npt.ArrayLike,
    shape: tuple[int, int],
) -> npt.NDArray:
    labels = np.asarray(industry_labels)
    if labels.ndim == 1:
        if labels.shape[0] != shape[1]:
            raise ValueError("一维 industry_labels 长度必须等于股票数")
        return np.broadcast_to(labels[None, :], shape)
    if labels.ndim == 2 and labels.shape == shape:
        return labels
    raise ValueError("industry_labels 必须为 (stock,) 或与 factor 同形的 (date, stock)")


def _valid_industry_label(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(numeric) or numeric < 0:
            return None
    try:
        if bool(value != value):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    return text


def _industry_group_residuals(
    values: npt.NDArray[np.float64],
    codes: npt.NDArray[np.int32],
    epsilon: float,
) -> npt.NDArray[np.float64]:
    """Project onto industry groups without constructing a dense OLS design.

    An intercept plus a full set of industry dummies fits each observation to
    its industry mean.  Computing that projection directly preserves the
    neutralization definition while removing one rank-deficient ``lstsq`` per
    candidate and date.
    """

    _, inverse = np.unique(codes, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    residuals = values - sums[inverse] / counts[inverse]
    residuals[np.abs(residuals) <= epsilon] = 0.0
    return residuals


def clean_factor_cross_sections(
    factor: npt.ArrayLike,
    universe_mask: npt.ArrayLike | None = None,
    *,
    row_indices: npt.ArrayLike | None = None,
    config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
) -> npt.NDArray[np.float64]:
    """Winsorize then z-score each requested date without filling NaNs."""
    values, universe, rows = _validate_inputs(factor, universe_mask, row_indices)

    result = np.full(values.shape, np.nan, dtype=np.float64)
    for date_index in rows:
        valid = universe[date_index] & np.isfinite(values[date_index])
        count = int(valid.sum())
        if count < max(config.min_count, config.zscore_ddof + 1):
            continue
        cross_section = values[date_index, valid]
        standardized = _zscore(_winsorize(cross_section, config), config)
        if standardized is None:
            continue
        result[date_index, valid] = standardized
    return result


def clean_candidate_factor_cross_sections(
    factor: npt.ArrayLike,
    industry_labels: npt.ArrayLike | None,
    universe_mask: npt.ArrayLike | None = None,
    *,
    row_indices: npt.ArrayLike | None = None,
    config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
    diagnostics: NeutralizationDiagnostics | None = None,
    encoded_industries: EncodedIndustryPanel | None = None,
) -> npt.NDArray[np.float64]:
    """Winsorize, industry-neutralize, then z-score candidate factors.

    Industry labels may be a static ``(stock,)`` vector or a point-in-time
    ``(date, stock)`` matrix. Stocks without a known industry are excluded from
    the cleaned row. If the industry regression cannot run, the entire date is
    left NaN; the global evaluation calendar is never shifted or backfilled.
    """
    values, universe, rows = _validate_inputs(factor, universe_mask, row_indices)
    if encoded_industries is None:
        if industry_labels is None:
            raise ValueError("industry_labels 与 encoded_industries 不能同时为空")
        encoded_industries = encode_industry_panel(industry_labels, values.shape)
    industry_codes = np.asarray(encoded_industries.codes)
    if industry_codes.shape != values.shape or industry_codes.dtype != np.int32:
        raise ValueError("encoded_industries 必须是与 factor 同形的 int32 编码")
    result = np.full(values.shape, np.nan, dtype=np.float64)

    for date_index in rows:
        valid = universe[date_index] & np.isfinite(values[date_index])
        count = int(valid.sum())
        if count < max(config.min_count, config.zscore_ddof + 1):
            continue

        clipped = _winsorize(values[date_index, valid], config)
        row_codes = industry_codes[date_index, valid]
        known = row_codes >= 0
        known_count = int(known.sum())
        known_codes = row_codes[known]
        industry_count = int(np.unique(known_codes).size) if known_count else 0
        skip_reason: str | None = None
        if known_count == 0:
            skip_reason = "no_known_industry_labels"
        elif known_count < industry_count + 1:
            skip_reason = "insufficient_industry_regression_samples"

        residuals: npt.NDArray[np.float64] | None = None
        if skip_reason is None:
            try:
                residuals = _industry_group_residuals(
                    clipped[known],
                    known_codes,
                    config.epsilon,
                )
            except (ArithmeticError, np.linalg.LinAlgError):
                skip_reason = "industry_ols_failure"

        if skip_reason is not None:
            if diagnostics is not None:
                diagnostics.record_skip(
                    int(date_index),
                    factor_valid_count=count,
                    known_industry_count=known_count,
                    industry_count=industry_count,
                    reason=skip_reason,
                )
            warnings.warn(
                f"日期索引 {date_index} 的行业中性化失败 "
                f"(reason={skip_reason}, factor_valid={count}, "
                f"known_industry={known_count}, industries={industry_count})，"
                "已排除当日候选截面",
                IndustryNeutralizationWarning,
                stacklevel=2,
            )
            continue

        assert residuals is not None
        standardized = _zscore(residuals, config)
        if standardized is not None:
            valid_positions = np.flatnonzero(valid)
            result[date_index, valid_positions[known]] = standardized
    return result


__all__ = [
    "DEFAULT_CLEANING_CONFIG",
    "CrossSectionalCleaningConfig",
    "EncodedIndustryPanel",
    "IndustryNeutralizationWarning",
    "NeutralizationDiagnostics",
    "NeutralizationSkipDetail",
    "clean_candidate_factor_cross_sections",
    "clean_factor_cross_sections",
    "encode_industry_panel",
]
