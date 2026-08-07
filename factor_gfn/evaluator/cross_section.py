"""Shared cross-sectional cleaning for candidate and Barra factor evaluation."""

from __future__ import annotations

from dataclasses import dataclass
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
    """A date skipped industry neutralization and continued to z-score."""


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
) -> npt.NDArray[np.object_]:
    labels = np.asarray(industry_labels, dtype=object)
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
    try:
        if bool(value != value):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    return text


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
    industry_labels: npt.ArrayLike,
    universe_mask: npt.ArrayLike | None = None,
    *,
    row_indices: npt.ArrayLike | None = None,
    config: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG,
) -> npt.NDArray[np.float64]:
    """Winsorize, industry-neutralize, then z-score candidate factors.

    Industry labels may be a static ``(stock,)`` vector or a point-in-time
    ``(date, stock)`` matrix. Missing-industry stocks keep their winsorized
    value and join the final z-score, but do not enter the OLS fit.
    """
    values, universe, rows = _validate_inputs(factor, universe_mask, row_indices)
    industries = _industry_matrix(industry_labels, values.shape)
    result = np.full(values.shape, np.nan, dtype=np.float64)

    for date_index in rows:
        valid = universe[date_index] & np.isfinite(values[date_index])
        count = int(valid.sum())
        if count < max(config.min_count, config.zscore_ddof + 1):
            continue

        clipped = _winsorize(values[date_index, valid], config)
        raw_labels = industries[date_index, valid]
        normalized_labels = [_valid_industry_label(value) for value in raw_labels]
        known = np.fromiter(
            (label is not None for label in normalized_labels),
            dtype=bool,
            count=count,
        )
        adjusted = clipped.copy()
        if known.any():
            labels = np.asarray(
                [label for label in normalized_labels if label is not None],
                dtype=object,
            )
            categories = np.unique(labels)
            known_count = int(known.sum())
            if known_count < len(categories) + 1:
                warnings.warn(
                    f"日期索引 {date_index} 的行业回归样本数 {known_count} 少于"
                    f"行业数+1 ({len(categories) + 1})，已跳过行业中性化",
                    IndustryNeutralizationWarning,
                    stacklevel=2,
                )
            else:
                design = np.column_stack(
                    [np.ones(known_count)]
                    + [(labels == category).astype(np.float64) for category in categories]
                )
                try:
                    coefficients = np.linalg.lstsq(
                        design,
                        clipped[known],
                        rcond=None,
                    )[0]
                    residuals = clipped[known] - design @ coefficients
                    residuals[np.abs(residuals) <= config.epsilon] = 0.0
                    adjusted[known] = residuals
                except np.linalg.LinAlgError:
                    warnings.warn(
                        f"日期索引 {date_index} 的行业 OLS 求解失败，已跳过行业中性化",
                        IndustryNeutralizationWarning,
                        stacklevel=2,
                    )

        standardized = _zscore(adjusted, config)
        if standardized is not None:
            result[date_index, valid] = standardized
    return result


__all__ = [
    "DEFAULT_CLEANING_CONFIG",
    "CrossSectionalCleaningConfig",
    "IndustryNeutralizationWarning",
    "clean_candidate_factor_cross_sections",
    "clean_factor_cross_sections",
]
