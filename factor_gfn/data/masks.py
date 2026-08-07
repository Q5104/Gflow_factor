"""构造六特征有效性 mask 与股票池资格 mask。

本模块只处理内存中的 DataFrame，不读取或写入项目数据文件。数据有效性与
股票池资格刻意分离：前者决定六特征是否应统一设为 NaN，后者只决定某日
某股票是否参与因子评价和交易。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


KEY_COLUMNS = ["trade_date", "stock_code"]
PRICE_COLUMNS = ["open", "high", "low", "close", "vwap"]
FEATURE_COLUMNS = [*PRICE_COLUMNS, "volume"]
STOCK_LIST_COLUMNS = ["stock_code", "short_name", "list_date"]

# 覆盖 ST、*ST、S*ST、SST；标记后不能紧跟英文字母或数字，避免误判 STAR。
CURRENT_ST_NAME_PATTERN = re.compile(
    r"^(?:\*?ST|S\*?ST)(?=[^A-Z0-9]|$)", re.IGNORECASE
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} 缺少字段：{sorted(missing)}")


def _normalize_stock_code(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def _normalized_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    _require_columns(frame, KEY_COLUMNS, label)
    keys = frame.loc[:, KEY_COLUMNS].copy()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"], errors="coerce").dt.normalize()
    keys["stock_code"] = _normalize_stock_code(keys["stock_code"])
    if keys[KEY_COLUMNS].isna().any(axis=None):
        raise ValueError(f"{label} 的股票代码或交易日期存在空值/非法值")
    duplicated = keys.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = keys.loc[duplicated, KEY_COLUMNS].head(10).to_dict("records")
        raise ValueError(f"{label} 存在重复股票-日期键，示例：{examples}")
    return keys


def is_current_st_name(short_names: pd.Series) -> pd.Series:
    """按当前简称识别 ST、*ST、S*ST 和 SST 前缀。"""
    names = short_names.astype("string").str.strip()
    return names.str.match(CURRENT_ST_NAME_PATTERN, na=False).astype(bool)


def build_feature_validity(
    frame: pd.DataFrame,
    *,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """检查六特征的数值与 OHLC/VWAP 逻辑，返回逐行原因和总 mask。

    `volume=0` 会使 VWAP 无法定义，因此在六特征共同有效样本中视为无效。
    本函数不会修改输入 DataFrame。
    """
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("容差必须为非负数")
    _require_columns(frame, [*KEY_COLUMNS, *FEATURE_COLUMNS], "六特征数据")
    keys = _normalized_keys(frame, "六特征数据")

    numeric = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
        index=frame.index,
        columns=FEATURE_COLUMNS,
    )
    prices_finite = finite[PRICE_COLUMNS].all(axis=1)
    volume_finite = finite["volume"]
    prices_positive = numeric[PRICE_COLUMNS].gt(0).all(axis=1)
    volume_positive = numeric["volume"].gt(0)

    high_reference = numeric[["open", "close", "low"]].max(axis=1, skipna=False)
    low_reference = numeric[["open", "close", "high"]].min(axis=1, skipna=False)
    price_scale = numeric[PRICE_COLUMNS].abs().max(axis=1, skipna=False)
    tolerance = absolute_tolerance + relative_tolerance * price_scale

    ohlc_consistent = (
        numeric["high"].add(tolerance).ge(high_reference)
        & numeric["low"].sub(tolerance).le(low_reference)
    )
    vwap_in_range = (
        numeric["vwap"].ge(numeric["low"].sub(tolerance))
        & numeric["vwap"].le(numeric["high"].add(tolerance))
    )
    feature_valid = (
        prices_finite
        & volume_finite
        & prices_positive
        & volume_positive
        & ohlc_consistent
        & vwap_in_range
    )

    result = keys
    result["prices_finite"] = prices_finite.to_numpy(dtype=bool)
    result["volume_finite"] = volume_finite.to_numpy(dtype=bool)
    result["prices_positive"] = prices_positive.to_numpy(dtype=bool)
    result["volume_positive"] = volume_positive.to_numpy(dtype=bool)
    result["ohlc_consistent"] = ohlc_consistent.fillna(False).to_numpy(dtype=bool)
    result["vwap_in_range"] = vwap_in_range.fillna(False).to_numpy(dtype=bool)
    result["feature_valid"] = feature_valid.fillna(False).to_numpy(dtype=bool)
    return result


def apply_feature_valid_mask(
    frame: pd.DataFrame,
    validity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """将数据质量无效行的六个特征统一设为 NaN，保留键和其他字段。"""
    _require_columns(frame, [*KEY_COLUMNS, *FEATURE_COLUMNS], "六特征数据")
    expected_keys = _normalized_keys(frame, "六特征数据")
    checked = build_feature_validity(frame) if validity is None else validity.copy()
    _require_columns(checked, [*KEY_COLUMNS, "feature_valid"], "特征有效性 mask")
    checked_keys = _normalized_keys(checked, "特征有效性 mask")
    if not expected_keys.reset_index(drop=True).equals(checked_keys.reset_index(drop=True)):
        raise ValueError("特征有效性 mask 与六特征数据的键或行顺序不一致")

    valid = checked["feature_valid"]
    if valid.isna().any():
        raise ValueError("feature_valid 不能包含空值")
    valid = valid.astype(bool).to_numpy()

    result = frame.copy()
    result.loc[:, FEATURE_COLUMNS] = result.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    result.loc[~valid, FEATURE_COLUMNS] = np.nan
    return result


def build_universe_eligibility(
    daily_keys: pd.DataFrame,
    stock_list: pd.DataFrame,
    *,
    min_listing_days: int = 180,
) -> pd.DataFrame:
    """构造当前 ST 与上市自然日数共同决定的股票池资格 mask。

    当前 ST 判断来自静态股票主表，因此对应股票在输入的全部历史日期均为
    `universe_eligible=False`。本函数不会修改或清空基础六特征。
    """
    if min_listing_days < 0:
        raise ValueError("min_listing_days 必须为非负整数")
    keys = _normalized_keys(daily_keys, "日频键")
    _require_columns(stock_list, STOCK_LIST_COLUMNS, "股票主表")

    stocks = stock_list.loc[:, STOCK_LIST_COLUMNS].copy()
    stocks["stock_code"] = _normalize_stock_code(stocks["stock_code"])
    if stocks["stock_code"].isna().any():
        raise ValueError("股票主表存在空股票代码")
    duplicated = stocks["stock_code"].duplicated(keep=False)
    if duplicated.any():
        examples = stocks.loc[duplicated, "stock_code"].astype(str).head(10).tolist()
        raise ValueError(f"股票主表存在重复代码，示例：{examples}")

    stocks["short_name"] = stocks["short_name"].astype("string").str.strip()
    stocks["list_date"] = pd.to_datetime(stocks["list_date"], errors="coerce").dt.normalize()
    stocks["name_known"] = stocks["short_name"].notna() & stocks["short_name"].ne("")
    stocks["list_date_known"] = stocks["list_date"].notna()
    stocks["is_current_st"] = is_current_st_name(stocks["short_name"])

    merged = keys.merge(
        stocks,
        on="stock_code",
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    merged["stock_record_present"] = merged["_merge"].eq("both")
    merged = merged.drop(columns="_merge")
    merged["name_known"] = merged["name_known"].fillna(False).astype(bool)
    merged["list_date_known"] = merged["list_date_known"].fillna(False).astype(bool)
    merged["is_current_st"] = merged["is_current_st"].fillna(False).astype(bool)

    days_since_list = (merged["trade_date"] - merged["list_date"]).dt.days
    merged["days_since_list"] = days_since_list.astype("Int64")
    merged["listing_age_eligible"] = days_since_list.ge(min_listing_days).fillna(False)
    merged["universe_eligible"] = (
        merged["stock_record_present"]
        & merged["name_known"]
        & merged["list_date_known"]
        & ~merged["is_current_st"]
        & merged["listing_age_eligible"]
    )

    output_columns = [
        *KEY_COLUMNS,
        "short_name",
        "list_date",
        "days_since_list",
        "stock_record_present",
        "name_known",
        "list_date_known",
        "is_current_st",
        "listing_age_eligible",
        "universe_eligible",
    ]
    return merged.loc[:, output_columns]


def combine_masks(
    feature_validity: pd.DataFrame,
    universe_eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """按股票-日期严格对齐两个 mask，生成最终 `usable_mask`。"""
    _require_columns(
        feature_validity, [*KEY_COLUMNS, "feature_valid"], "特征有效性 mask"
    )
    _require_columns(
        universe_eligibility,
        [*KEY_COLUMNS, "universe_eligible"],
        "股票池资格 mask",
    )
    feature_keys = _normalized_keys(feature_validity, "特征有效性 mask")
    universe_keys = _normalized_keys(universe_eligibility, "股票池资格 mask")

    left = feature_keys.copy()
    left["feature_valid"] = feature_validity["feature_valid"].astype("boolean").to_numpy()
    right = universe_keys.copy()
    right["universe_eligible"] = (
        universe_eligibility["universe_eligible"].astype("boolean").to_numpy()
    )
    result = left.merge(right, on=KEY_COLUMNS, how="outer", validate="one_to_one", indicator=True)
    if not result["_merge"].eq("both").all():
        examples = result.loc[result["_merge"].ne("both"), KEY_COLUMNS + ["_merge"]]
        raise ValueError(f"两个 mask 的股票-日期键不完全一致，示例：{examples.head(10).to_dict('records')}")
    if result[["feature_valid", "universe_eligible"]].isna().any(axis=None):
        raise ValueError("两个 mask 的布尔值均不能包含空值")

    result = result.drop(columns="_merge")
    result["feature_valid"] = result["feature_valid"].astype(bool)
    result["universe_eligible"] = result["universe_eligible"].astype(bool)
    result["usable_mask"] = result["feature_valid"] & result["universe_eligible"]
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)
