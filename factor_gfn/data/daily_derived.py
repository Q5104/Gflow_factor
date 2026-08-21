"""Pure numerical builder for the frozen Daily-Derived v1 feature space.

All inputs must already be aligned as ``(date, stock)`` matrices. This module
does not read data, align share history, apply a universe mask, or write an
artifact.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from factor_gfn.feature_spaces import DAILY_DERIVED_FEATURE_NAMES

CLV_TOL = 1e-12
ILLIQ_SCALE = 1e8


def _aligned_float64_inputs(
    values: Mapping[str, npt.ArrayLike],
) -> dict[str, npt.NDArray[np.float64]]:
    arrays: dict[str, npt.NDArray[np.float64]] = {}
    expected_shape: tuple[int, int] | None = None
    for name, value in values.items():
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须能够转换为 float64 数组") from exc
        if array.ndim != 2:
            raise ValueError(f"{name} 必须是二维 (date, stock) 数组，实际形状为 {array.shape}")
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                "所有 Daily-Derived 输入 shape 必须一致："
                f"期望 {expected_shape}，但 {name} 为 {array.shape}"
            )
        arrays[name] = array
    return arrays


def _lag(values: npt.NDArray[np.float64], periods: int) -> npt.NDArray[np.float64]:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if periods < values.shape[0]:
        result[periods:] = values[:-periods]
    return result


def _safe_ratio(
    numerator: npt.NDArray[np.float64],
    denominator: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.float64]:
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    ratio_valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    if valid is not None:
        ratio_valid &= valid
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(numerator, denominator, out=result, where=ratio_valid)
    result[~np.isfinite(result)] = np.nan
    return result


def build_daily_derived_features(
    *,
    open_adjusted: npt.ArrayLike,
    high_adjusted: npt.ArrayLike,
    low_adjusted: npt.ArrayLike,
    close_adjusted: npt.ArrayLike,
    vwap_adjusted: npt.ArrayLike,
    volume_raw: npt.ArrayLike,
    amount_raw: npt.ArrayLike,
    list_a_shares: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Build the 16 frozen Daily-Derived v1 features as ``(date, 16, stock)``.

    Lagged inputs use exact row offsets on the supplied date axis. Missing
    history remains missing; this function never searches for an earlier
    finite observation.
    """

    arrays = _aligned_float64_inputs(
        {
            "open_adjusted": open_adjusted,
            "high_adjusted": high_adjusted,
            "low_adjusted": low_adjusted,
            "close_adjusted": close_adjusted,
            "vwap_adjusted": vwap_adjusted,
            "volume_raw": volume_raw,
            "amount_raw": amount_raw,
            "list_a_shares": list_a_shares,
        }
    )
    open_ = arrays["open_adjusted"]
    high = arrays["high_adjusted"]
    low = arrays["low_adjusted"]
    close = arrays["close_adjusted"]
    vwap = arrays["vwap_adjusted"]
    volume = arrays["volume_raw"]
    amount = arrays["amount_raw"]
    shares = arrays["list_a_shares"]

    close_lag1 = _lag(close, 1)
    volume_lag1 = _lag(volume, 1)
    volume_lag5 = _lag(volume, 5)
    amount_lag5 = _lag(amount, 5)

    finite_ohlc = (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
    )
    positive_ohlc = (
        finite_ohlc
        & (open_ > 0.0)
        & (high > 0.0)
        & (low > 0.0)
        & (close > 0.0)
    )
    geometry_valid = (
        positive_ohlc
        & (high >= np.maximum(open_, close))
        & (low <= np.minimum(open_, close))
        & (high >= low)
    )

    with np.errstate(invalid="ignore", over="ignore"):
        ret_gap = _safe_ratio(open_, close_lag1, open_ > 0.0) - 1.0
        ret_cc1 = _safe_ratio(close, close_lag1, close > 0.0) - 1.0
        ret_co = _safe_ratio(close, open_, close > 0.0) - 1.0
        ret_hl = _safe_ratio(high, low, high > 0.0) - 1.0
        ret_range = _safe_ratio(high - low, close_lag1, geometry_valid)
        ret_body = _safe_ratio(np.abs(close - open_), close_lag1, geometry_valid)
        ret_upper_shadow = _safe_ratio(
            high - np.maximum(open_, close), close_lag1, geometry_valid
        )
        ret_lower_shadow = _safe_ratio(
            np.minimum(open_, close) - low, close_lag1, geometry_valid
        )
        ret_close_vwap = _safe_ratio(close, vwap, close > 0.0) - 1.0
        ret_open_vwap = _safe_ratio(open_, vwap, open_ > 0.0) - 1.0
        ret_vol_chg1 = _safe_ratio(volume, volume_lag1) - 1.0
        ret_vol_chg5 = _safe_ratio(volume, volume_lag5) - 1.0
        turnover = _safe_ratio(volume, shares)
        illiq = _safe_ratio(ILLIQ_SCALE * np.abs(ret_cc1), amount)
        ret_amt_chg5 = _safe_ratio(amount, amount_lag5) - 1.0

        clv_denominator = high - low
        clv = _safe_ratio(
            2.0 * close - high - low,
            clv_denominator,
            geometry_valid,
        )

    clv_abs = np.abs(clv)
    near_boundary = (clv_abs > 1.0) & (clv_abs <= 1.0 + CLV_TOL)
    clv[near_boundary] = np.sign(clv[near_boundary])
    clv[np.abs(clv) > 1.0 + CLV_TOL] = np.nan

    result = np.stack(
        (
            ret_gap,
            ret_cc1,
            ret_co,
            ret_hl,
            ret_range,
            ret_body,
            ret_upper_shadow,
            ret_lower_shadow,
            ret_close_vwap,
            ret_open_vwap,
            ret_vol_chg1,
            ret_vol_chg5,
            turnover,
            illiq,
            ret_amt_chg5,
            clv,
        ),
        axis=1,
    )
    result[~np.isfinite(result)] = np.nan
    return result


__all__ = [
    "CLV_TOL",
    "DAILY_DERIVED_FEATURE_NAMES",
    "ILLIQ_SCALE",
    "build_daily_derived_features",
]
