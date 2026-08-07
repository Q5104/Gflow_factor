"""Load and align the static Shenwan level-1 industry cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from .downloader import INDUSTRY_SW_PATH


def load_sw_level1_industries(
    stock_list: npt.ArrayLike,
    path: str | Path = INDUSTRY_SW_PATH,
) -> npt.NDArray[np.object_]:
    """Return level-1 industry labels aligned to ``stock_list`` order.

    Missing classifications are returned as ``None``. The current adata
    interface does not expose an effective date, so this loader represents a
    static mapping rather than a point-in-time industry history.
    """
    stocks = np.asarray(stock_list)
    if stocks.ndim != 1 or stocks.size == 0:
        raise ValueError("stock_list 必须是非空一维数组")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"申万行业文件不存在：{resolved}")

    frame = pd.read_parquet(
        resolved,
        columns=["stock_code", "industry_name", "industry_type"],
    )
    frame["stock_code"] = (
        frame["stock_code"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    frame["industry_name"] = frame["industry_name"].astype("string").str.strip()
    level1 = frame.loc[
        frame["industry_type"].astype("string").str.strip().eq("申万一级")
        & frame["stock_code"].str.fullmatch(r"\d{6}", na=False)
        & frame["industry_name"].notna()
        & frame["industry_name"].ne("")
    ].copy()

    conflicts = (
        level1.groupby("stock_code", dropna=False)["industry_name"]
        .nunique(dropna=False)
        .gt(1)
    )
    if conflicts.any():
        examples = conflicts.index[conflicts].astype(str).tolist()[:10]
        raise ValueError(f"申万一级行业存在同代码冲突，示例：{examples}")
    mapping = (
        level1.drop_duplicates("stock_code", keep="last")
        .set_index("stock_code")["industry_name"]
        .to_dict()
    )
    labels = [mapping.get(str(code).strip().zfill(6)) for code in stocks]
    return np.asarray(labels, dtype=object)


__all__ = ["load_sw_level1_industries"]
