"""Disk-backed data preparation and persistence for Barra-style factors."""

from __future__ import annotations

import json
import os
import gc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
from numpy.lib.format import open_memmap

from factor_gfn.data.downloader import PROJECT_ROOT, RAW_CLOSE_PATH, STOCK_SHARES_PATH
from factor_gfn.data.preprocess import PROCESSED_DATA_DIR

from .config import BarraConfig, DEFAULT_BARRA_CONFIG
from .factors import BarraFactorSet, STYLE_NAMES, calculate_barra_factors


@dataclass(frozen=True, slots=True)
class BarraPaths:
    processed_dir: Path = PROCESSED_DATA_DIR
    raw_close_path: Path = RAW_CLOSE_PATH
    shares_path: Path = STOCK_SHARES_PATH

    def __post_init__(self) -> None:
        for name in ("processed_dir", "raw_close_path", "shares_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())

    @property
    def output_dir(self) -> Path:
        return self.processed_dir / "barra"

    @property
    def tensor_path(self) -> Path:
        return self.processed_dir / "data_tensor.npy"

    @property
    def universe_mask_path(self) -> Path:
        return self.processed_dir / "universe_mask.npy"

    @property
    def date_list_path(self) -> Path:
        return self.processed_dir / "date_list.npy"

    @property
    def stock_list_path(self) -> Path:
        return self.processed_dir / "stock_list.npy"

    def auxiliary_path(self, name: str) -> Path:
        return self.output_dir / f"{name}.npy"

    def exposure_path(self, name: str) -> Path:
        if name not in STYLE_NAMES:
            raise KeyError(name)
        return self.output_dir / f"{name}.npy"

    @property
    def market_return_path(self) -> Path:
        return self.output_dir / "market_return.npy"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"


@dataclass(frozen=True, slots=True)
class BarraInputs:
    dates: np.ndarray
    stocks: np.ndarray
    open: np.ndarray
    adjusted_close: np.ndarray
    volume: np.ndarray
    universe_mask: np.ndarray
    float_market_cap: np.ndarray
    total_market_cap: np.ndarray
    list_a_shares: np.ndarray


DEFAULT_BARRA_PATHS = BarraPaths()


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


def _require(paths: list[Path], message: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{message}：{missing}")


def build_barra_auxiliary_arrays(
    paths: BarraPaths = DEFAULT_BARRA_PATHS,
    *,
    array_dtype: str = "float32",
    batch_rows: int = 250_000,
) -> dict:
    """Align historical shares and raw close to the processed date-stock grid."""
    _require(
        [paths.date_list_path, paths.stock_list_path, paths.raw_close_path, paths.shares_path],
        "Barra 辅助数组输入不存在",
    )
    if np.dtype(array_dtype).kind != "f" or batch_rows <= 0:
        raise ValueError("array_dtype 必须为浮点类型且 batch_rows 必须为正")
    dates = np.load(paths.date_list_path, allow_pickle=False).astype("datetime64[D]")
    stocks = np.load(paths.stock_list_path, allow_pickle=False).astype(str)
    shape = (dates.size, stocks.size)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    names = ("float_market_cap", "total_market_cap", "list_a_shares")
    temporary_paths = {
        name: paths.auxiliary_path(name).with_suffix(".npy.tmp") for name in names
    }
    arrays = {}
    for name, temporary in temporary_paths.items():
        temporary.unlink(missing_ok=True)
        arrays[name] = open_memmap(temporary, mode="w+", dtype=array_dtype, shape=shape)
        arrays[name][:] = np.nan

    raw = _sql_path(paths.raw_close_path)
    shares = _sql_path(paths.shares_path)
    connection = duckdb.connect()
    try:
        reader = connection.execute(
            f"""
            SELECT p.trade_date, p.stock_code,
                   CAST(s.list_a_shares AS DOUBLE) AS list_a_shares,
                   CAST(p.close AS DOUBLE) * CAST(s.list_a_shares AS DOUBLE)
                       AS float_market_cap,
                   CAST(p.close AS DOUBLE) * CAST(s.total_shares AS DOUBLE)
                       AS total_market_cap
            FROM (
                SELECT CAST(trade_date AS DATE) AS trade_date,
                       lpad(CAST(stock_code AS VARCHAR), 6, '0') AS stock_code,
                       close
                FROM read_parquet('{raw}')
            ) p
            ASOF LEFT JOIN (
                SELECT CAST(change_date AS DATE) AS change_date,
                       lpad(CAST(stock_code AS VARCHAR), 6, '0') AS stock_code,
                       total_shares, list_a_shares
                FROM read_parquet('{shares}')
            ) s
            ON p.stock_code = s.stock_code AND p.trade_date >= s.change_date
            """
        ).fetch_record_batch(batch_rows)
        for batch in reader:
            frame = batch.to_pandas()
            batch_dates = frame["trade_date"].to_numpy(dtype="datetime64[D]")
            batch_stocks = frame["stock_code"].astype(str).to_numpy()
            date_index = np.searchsorted(dates, batch_dates)
            stock_index = np.searchsorted(stocks, batch_stocks)
            in_bounds = (date_index < dates.size) & (stock_index < stocks.size)
            matched = np.zeros(frame.shape[0], dtype=bool)
            matched[in_bounds] = (
                (dates[date_index[in_bounds]] == batch_dates[in_bounds])
                & (stocks[stock_index[in_bounds]] == batch_stocks[in_bounds])
            )
            if not matched.any():
                continue
            for name in names:
                arrays[name][date_index[matched], stock_index[matched]] = frame.loc[
                    matched, name
                ].to_numpy(dtype=array_dtype)
    finally:
        connection.close()
        for array in arrays.values():
            array.flush()
        del array
        arrays.clear()
        gc.collect()

    for name in names:
        os.replace(temporary_paths[name], paths.auxiliary_path(name))
    return {
        "shape": tuple(map(int, shape)),
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
        "stock_count": int(stocks.size),
        "dtype": array_dtype,
    }


def load_barra_inputs(
    paths: BarraPaths = DEFAULT_BARRA_PATHS,
    mmap_mode: str | None = "r",
) -> BarraInputs:
    required = [
        paths.tensor_path,
        paths.universe_mask_path,
        paths.date_list_path,
        paths.stock_list_path,
        paths.auxiliary_path("float_market_cap"),
        paths.auxiliary_path("total_market_cap"),
        paths.auxiliary_path("list_a_shares"),
    ]
    _require(required, "Barra 输入不存在，请先完成日频预处理和辅助数组构建")
    tensor = np.load(paths.tensor_path, mmap_mode=mmap_mode, allow_pickle=False)
    if tensor.ndim != 3 or tensor.shape[1] != 6:
        raise ValueError("data_tensor 必须采用 (date, 6, stock) 布局")
    dates = np.load(paths.date_list_path, allow_pickle=False)
    stocks = np.load(paths.stock_list_path, allow_pickle=False)
    universe = np.load(paths.universe_mask_path, mmap_mode=mmap_mode, allow_pickle=False)
    expected = (dates.size, stocks.size)
    if tensor.shape[::2] != expected or universe.shape != expected:
        raise ValueError("处理后行情、日期、股票和 universe_mask 形状不一致")
    return BarraInputs(
        dates=dates,
        stocks=stocks,
        open=tensor[:, 0, :],
        adjusted_close=tensor[:, 3, :],
        volume=tensor[:, 5, :],
        universe_mask=universe,
        float_market_cap=np.load(paths.auxiliary_path("float_market_cap"), mmap_mode=mmap_mode),
        total_market_cap=np.load(paths.auxiliary_path("total_market_cap"), mmap_mode=mmap_mode),
        list_a_shares=np.load(paths.auxiliary_path("list_a_shares"), mmap_mode=mmap_mode),
    )


def run_barra_factor_pipeline(
    config: BarraConfig = DEFAULT_BARRA_CONFIG,
    paths: BarraPaths = DEFAULT_BARRA_PATHS,
) -> dict:
    """Calculate and persist the five exposures; never modifies raw data."""
    inputs = load_barra_inputs(paths)
    factor_set = calculate_barra_factors(
        inputs.adjusted_close,
        inputs.volume,
        inputs.float_market_cap,
        inputs.total_market_cap,
        inputs.list_a_shares,
        inputs.universe_mask,
        config,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = {}
    for name in STYLE_NAMES:
        values = factor_set.exposures[name]
        coverage[name] = float(np.isfinite(values).mean())
        _atomic_save(paths.exposure_path(name), values.astype("float32"))
    _atomic_save(paths.market_return_path, factor_set.market_return.astype("float64"))
    metadata = {
        "schema": "factor_gfn.barra_five_style.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "styles": list(STYLE_NAMES),
        "config": asdict(config),
        "shape": [int(inputs.dates.size), int(inputs.stocks.size)],
        "coverage": coverage,
        "market_return": "lagged market-cap-weighted all-A return",
        "liquidity": "rolling mean of volume_shares / list_a_shares",
    }
    temporary = paths.metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, paths.metadata_path)
    return metadata


def load_barra_factor_set(
    paths: BarraPaths = DEFAULT_BARRA_PATHS,
    mmap_mode: str | None = "r",
) -> BarraFactorSet:
    required = [paths.exposure_path(name) for name in STYLE_NAMES] + [paths.market_return_path]
    _require(required, "Barra 因子结果不存在，请先运行 run_barra_factor_pipeline")
    exposures = {
        name: np.load(paths.exposure_path(name), mmap_mode=mmap_mode, allow_pickle=False)
        for name in STYLE_NAMES
    }
    market_return = np.load(paths.market_return_path, mmap_mode=mmap_mode, allow_pickle=False)
    return BarraFactorSet(exposures=exposures, market_return=market_return)


__all__ = [
    "DEFAULT_BARRA_PATHS",
    "BarraInputs",
    "BarraPaths",
    "build_barra_auxiliary_arrays",
    "load_barra_factor_set",
    "load_barra_inputs",
    "run_barra_factor_pipeline",
]
