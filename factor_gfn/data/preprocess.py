"""将下载完成的日频行情转换为清洗长表和 GFlowNet 数组。

本模块不在导入时读取数据。所有长任务只能由用户显式调用 ``run_preprocess`` 启动，
并且不会修改 ``data/raw`` 或 ``data/download_parts`` 中的任何文件。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from .downloader import (
    LISTING_DATES_PATH,
    MARKET_DATA_PATH,
    PROJECT_ROOT,
    RAW_CLOSE_PATH,
)
from .masks import FEATURE_COLUMNS, is_current_st_name


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    market_path: Path = MARKET_DATA_PATH
    raw_close_path: Path = RAW_CLOSE_PATH
    listing_path: Path = LISTING_DATES_PATH
    output_dir: Path = PROCESSED_DATA_DIR
    min_listing_days: int = 180
    relative_tolerance: float = 1e-6
    absolute_tolerance: float = 1e-12
    array_dtype: str = "float32"
    batch_rows: int = 250_000

    def __post_init__(self) -> None:
        for field_name in ("market_path", "raw_close_path", "listing_path", "output_dir"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).resolve())
        if self.min_listing_days < 0:
            raise ValueError("min_listing_days 不能为负数")
        if self.relative_tolerance < 0 or self.absolute_tolerance < 0:
            raise ValueError("价格容差不能为负数")
        if np.dtype(self.array_dtype).kind != "f":
            raise ValueError("array_dtype 必须是浮点类型")
        if self.batch_rows <= 0:
            raise ValueError("batch_rows 必须为正整数")

    @property
    def daily_clean_path(self) -> Path:
        return self.output_dir / "daily_clean.parquet"

    @property
    def data_tensor_path(self) -> Path:
        return self.output_dir / "data_tensor.npy"

    @property
    def valid_mask_path(self) -> Path:
        return self.output_dir / "valid_mask.npy"

    @property
    def universe_mask_path(self) -> Path:
        return self.output_dir / "universe_mask.npy"

    @property
    def date_list_path(self) -> Path:
        return self.output_dir / "date_list.npy"

    @property
    def stock_list_path(self) -> Path:
        return self.output_dir / "stock_list.npy"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _require_inputs(config: PreprocessConfig) -> None:
    missing = [
        str(path)
        for path in (config.market_path, config.raw_close_path, config.listing_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"预处理输入不存在：{missing}")


def inspect_inputs(config: PreprocessConfig = PreprocessConfig()) -> dict:
    """只读返回三份输入的行数、股票数和日期范围，不生成输出。"""

    _require_inputs(config)
    connection = duckdb.connect()
    try:
        market = connection.execute(
            f"""
            SELECT count(*) AS rows, count(DISTINCT stock_code) AS stocks,
                   min(trade_date) AS start_date, max(trade_date) AS end_date
            FROM read_parquet('{_sql_path(config.market_path)}')
            """
        ).fetchone()
        raw_close = connection.execute(
            f"""
            SELECT count(*) AS rows, count(DISTINCT stock_code) AS stocks,
                   min(trade_date) AS start_date, max(trade_date) AS end_date
            FROM read_parquet('{_sql_path(config.raw_close_path)}')
            """
        ).fetchone()
        listing = connection.execute(
            f"""
            SELECT count(*) AS rows, count(DISTINCT stock_code) AS stocks
            FROM read_parquet('{_sql_path(config.listing_path)}')
            """
        ).fetchone()
    finally:
        connection.close()
    return {
        "market_data": dict(zip(("rows", "stocks", "start_date", "end_date"), market)),
        "raw_close": dict(zip(("rows", "stocks", "start_date", "end_date"), raw_close)),
        "stock_list": dict(zip(("rows", "stocks"), listing)),
    }


def _daily_clean_sql(config: PreprocessConfig) -> str:
    market = _sql_path(config.market_path)
    raw_close = _sql_path(config.raw_close_path)
    relative = float(config.relative_tolerance)
    absolute = float(config.absolute_tolerance)
    return f"""
        WITH joined AS (
            SELECT
                CAST(m.trade_date AS DATE) AS trade_date,
                lpad(CAST(m.stock_code AS VARCHAR), 6, '0') AS stock_code,
                CAST(m.open AS DOUBLE) AS open,
                CAST(m.high AS DOUBLE) AS high,
                CAST(m.low AS DOUBLE) AS low,
                CAST(m.close AS DOUBLE) AS close,
                CAST(m.volume AS DOUBLE) AS volume,
                CAST(m.amount AS DOUBLE) AS amount,
                CAST(r.close AS DOUBLE) AS close_raw
            FROM read_parquet('{market}') m
            INNER JOIN read_parquet('{raw_close}') r
                USING (trade_date, stock_code)
        ),
        derived AS (
            SELECT *,
                CASE
                    WHEN isfinite(close) AND isfinite(close_raw)
                         AND close > 0 AND close_raw > 0
                    THEN close / close_raw
                    ELSE NULL
                END AS adj_factor
            FROM joined
        ),
        priced AS (
            SELECT *,
                CASE
                    WHEN isfinite(amount) AND isfinite(volume) AND volume > 0
                         AND isfinite(adj_factor)
                    THEN amount * adj_factor / volume
                    ELSE NULL
                END AS vwap
            FROM derived
        ),
        checked AS (
            SELECT *,
                {absolute} + {relative} * greatest(
                    abs(open), abs(high), abs(low), abs(close), abs(vwap)
                ) AS tolerance
            FROM priced
        ),
        validity AS (
            SELECT *,
                isfinite(open) AND isfinite(high) AND isfinite(low)
                AND isfinite(close) AND isfinite(vwap) AND isfinite(volume)
                AND open > 0 AND high > 0 AND low > 0 AND close > 0 AND vwap > 0
                AND volume > 0
                AND high + tolerance >= greatest(open, close, low)
                AND low - tolerance <= least(open, close, high)
                AND vwap >= low - tolerance AND vwap <= high + tolerance
                AS feature_valid
            FROM checked
        )
        SELECT trade_date, stock_code,
            CASE WHEN feature_valid THEN open ELSE NULL END AS open,
            CASE WHEN feature_valid THEN high ELSE NULL END AS high,
            CASE WHEN feature_valid THEN low ELSE NULL END AS low,
            CASE WHEN feature_valid THEN close ELSE NULL END AS close,
            CASE WHEN feature_valid THEN vwap ELSE NULL END AS vwap,
            CASE WHEN feature_valid THEN volume ELSE NULL END AS volume
        FROM validity
    """


def build_daily_clean(config: PreprocessConfig = PreprocessConfig()) -> Path:
    """以内连接口径生成清洗长表；无效行保留键并将六特征统一设为 NULL。"""

    _require_inputs(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = config.daily_clean_path.with_suffix(".parquet.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = duckdb.connect()
    try:
        connection.execute(
            f"COPY ({_daily_clean_sql(config)}) TO '{_sql_path(temporary)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()
    try:
        os.replace(temporary, config.daily_clean_path)
    except PermissionError as exc:
        raise PermissionError(
            f"无法替换 {config.daily_clean_path}；请关闭占用该文件的 Notebook 变量或程序后重试"
        ) from exc
    return config.daily_clean_path


def _load_stock_metadata(config: PreprocessConfig, stock_list: np.ndarray) -> tuple:
    stocks = pd.read_parquet(config.listing_path)
    required = {"stock_code", "short_name", "list_date"}
    missing = required.difference(stocks.columns)
    if missing:
        raise ValueError(f"股票主表缺少字段：{sorted(missing)}")
    stocks = stocks.loc[:, sorted(required)].copy()
    stocks["stock_code"] = (
        stocks["stock_code"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    if stocks["stock_code"].duplicated().any():
        raise ValueError("股票主表存在重复 stock_code")
    stocks["short_name"] = stocks["short_name"].astype("string").str.strip()
    stocks["list_date"] = pd.to_datetime(stocks["list_date"], errors="coerce").dt.normalize()
    stocks["is_current_st"] = is_current_st_name(stocks["short_name"])
    aligned = stocks.set_index("stock_code").reindex(stock_list.astype(str))
    base_eligible = (
        aligned["short_name"].notna()
        & aligned["short_name"].ne("")
        & aligned["list_date"].notna()
        & ~aligned["is_current_st"].fillna(False)
    ).to_numpy(dtype=bool)
    list_dates = aligned["list_date"].to_numpy(dtype="datetime64[D]")
    return base_eligible, list_dates


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


def build_processed_arrays(config: PreprocessConfig = PreprocessConfig()) -> dict:
    """从 ``daily_clean.parquet`` 分批写出张量、两个 mask 及索引。"""

    if not config.daily_clean_path.exists():
        raise FileNotFoundError(f"请先生成 {config.daily_clean_path}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        dates = connection.execute(
            f"SELECT DISTINCT trade_date FROM read_parquet('{_sql_path(config.daily_clean_path)}') ORDER BY trade_date"
        ).fetchnumpy()["trade_date"].astype("datetime64[D]")
        stocks = connection.execute(
            f"SELECT DISTINCT stock_code FROM read_parquet('{_sql_path(config.daily_clean_path)}') ORDER BY stock_code"
        ).fetchnumpy()["stock_code"].astype(str)
    finally:
        connection.close()
    if dates.size == 0 or stocks.size == 0:
        raise ValueError("daily_clean.parquet 没有可用于矩阵化的股票或日期")

    _atomic_save_array(config.date_list_path, dates)
    _atomic_save_array(config.stock_list_path, stocks)
    shape = (int(dates.size), len(FEATURE_COLUMNS), int(stocks.size))
    temporary_paths = {
        "tensor": config.data_tensor_path.with_suffix(".npy.tmp"),
        "valid": config.valid_mask_path.with_suffix(".npy.tmp"),
        "universe": config.universe_mask_path.with_suffix(".npy.tmp"),
    }
    for path in temporary_paths.values():
        if path.exists():
            path.unlink()
    tensor = open_memmap(temporary_paths["tensor"], mode="w+", dtype=config.array_dtype, shape=shape)
    valid_mask = open_memmap(
        temporary_paths["valid"], mode="w+", dtype=np.bool_, shape=(shape[0], shape[2])
    )
    universe_mask = open_memmap(
        temporary_paths["universe"], mode="w+", dtype=np.bool_, shape=(shape[0], shape[2])
    )
    tensor[:] = np.nan
    valid_mask[:] = False
    universe_mask[:] = False
    base_eligible, list_dates = _load_stock_metadata(config, stocks)

    connection = duckdb.connect()
    try:
        reader = connection.execute(
            f"SELECT trade_date, stock_code, {', '.join(FEATURE_COLUMNS)} "
            f"FROM read_parquet('{_sql_path(config.daily_clean_path)}')"
        ).fetch_record_batch(config.batch_rows)
        for batch in reader:
            frame = batch.to_pandas()
            batch_dates = pd.to_datetime(frame["trade_date"]).to_numpy(dtype="datetime64[D]")
            batch_stocks = frame["stock_code"].astype(str).to_numpy()
            date_index = np.searchsorted(dates, batch_dates)
            stock_index = np.searchsorted(stocks, batch_stocks)
            values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=config.array_dtype)
            row_valid = np.isfinite(values).all(axis=1)
            tensor[date_index, :, stock_index] = values
            valid_mask[date_index, stock_index] = row_valid
            aligned_list_dates = list_dates[stock_index]
            list_date_known = ~np.isnat(aligned_list_dates)
            age_days = (batch_dates - aligned_list_dates).astype("timedelta64[D]").astype(float)
            universe_mask[date_index, stock_index] = (
                base_eligible[stock_index]
                & list_date_known
                & (age_days >= config.min_listing_days)
            )
    finally:
        connection.close()
        tensor.flush()
        valid_mask.flush()
        universe_mask.flush()
        del tensor, valid_mask, universe_mask

    os.replace(temporary_paths["tensor"], config.data_tensor_path)
    os.replace(temporary_paths["valid"], config.valid_mask_path)
    os.replace(temporary_paths["universe"], config.universe_mask_path)
    return {
        "shape": shape,
        "dates": int(dates.size),
        "stocks": int(stocks.size),
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
    }


def run_preprocess(config: PreprocessConfig = PreprocessConfig()) -> dict:
    """显式执行完整预处理，并写出可复现元数据。"""

    input_summary = inspect_inputs(config)
    build_daily_clean(config)
    array_summary = build_processed_arrays(config)
    metadata = {
        "schema": "factor_gfn.processed_daily.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_order": FEATURE_COLUMNS,
        "tensor_layout": "date,feature,stock",
        "vwap_formula": "amount * (close_adjusted / close_raw) / volume",
        "join_rule": "inner join on trade_date,stock_code",
        "invalid_row_rule": "keep key and set all six features to NaN",
        "st_rule": "current short-name prefix; universe mask only",
        "listing_rule": f"at least {config.min_listing_days} natural days; universe mask only",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "input_summary": input_summary,
        "array_summary": array_summary,
    }
    temporary = config.metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, config.metadata_path)
    return metadata


__all__ = [
    "PROCESSED_DATA_DIR",
    "PreprocessConfig",
    "build_daily_clean",
    "build_processed_arrays",
    "inspect_inputs",
    "run_preprocess",
]
