"""Real-input alignment and artifact writing for Daily-Derived v1."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import duckdb
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from .daily_derived import DAILY_DERIVED_FEATURE_NAMES, build_daily_derived_features
from .downloader import MARKET_DATA_PATH, RAW_CLOSE_PATH
from .preprocess import PROCESSED_DATA_DIR


FEATURE_SPACE_ID = "daily_derived_v1"
SCHEMA_VERSION = "factor_gfn.daily_derived.v1"
MAX_LAG = 5
FORMULAS = (
    "O_t/C_{t-1}-1", "C_t/C_{t-1}-1", "C_t/O_t-1", "H_t/L_t-1",
    "(H_t-L_t)/C_{t-1}", "abs(C_t-O_t)/C_{t-1}",
    "(H_t-max(O_t,C_t))/C_{t-1}", "(min(O_t,C_t)-L_t)/C_{t-1}",
    "C_t/VWAP_t-1", "O_t/VWAP_t-1", "V_t/V_{t-1}-1",
    "V_t/V_{t-5}-1", "V_t/S_t", "1e8*abs(ret_cc1_t)/A_t",
    "A_t/A_{t-5}-1", "(2*C_t-H_t-L_t)/(H_t-L_t)",
)


@dataclass(frozen=True, slots=True)
class DailyDerivedArtifactConfig:
    market_path: Path = MARKET_DATA_PATH
    raw_close_path: Path = RAW_CLOSE_PATH
    date_list_path: Path = PROCESSED_DATA_DIR / "date_list.npy"
    stock_list_path: Path = PROCESSED_DATA_DIR / "stock_list.npy"
    universe_mask_path: Path = PROCESSED_DATA_DIR / "universe_mask.npy"
    shares_path: Path = PROCESSED_DATA_DIR / "barra" / "list_a_shares.npy"
    shares_metadata_path: Path = PROCESSED_DATA_DIR / "barra" / "metadata.json"
    output_dir: Path = PROCESSED_DATA_DIR / FEATURE_SPACE_ID
    date_block_size: int = 128
    relative_tolerance: float = 1e-6
    absolute_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "market_path", "raw_close_path", "date_list_path", "stock_list_path",
            "universe_mask_path", "shares_path", "shares_metadata_path", "output_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.date_block_size <= 0:
            raise ValueError("date_block_size 必须为正整数")
        if self.relative_tolerance < 0 or self.absolute_tolerance < 0:
            raise ValueError("VWAP 容差不能为负数")

    @property
    def tensor_path(self) -> Path:
        return self.output_dir / "data_tensor.npy"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"


def daily_derived_schema_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_space_id": FEATURE_SPACE_ID,
        "feature_order": list(DAILY_DERIVED_FEATURE_NAMES),
        "formulas": list(FORMULAS),
        "layout": "(date, feature, stock)",
        "storage_dtype": "float32",
        "calculation_dtype": "float64",
        "lag_contract": "exact global date-axis offsets t-1 and t-5; no fill or nearest-valid search",
        "shares_pit_contract": "latest same-stock list_a_shares with change_date <= trade_date",
        "amount_contract": "raw actual trading amount in RMB; never adjusted",
        "vwap_contract": "amount_raw * (close_adjusted / close_raw) / volume_raw",
        "missing_value_contract": "feature-specific NaN; every required price must be finite and > 0; other dependencies must be finite; every denominator must be > 0",
    }


def daily_derived_schema_fingerprint(contract: Mapping[str, Any] | None = None) -> str:
    payload = daily_derived_schema_contract() if contract is None else dict(contract)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _parquet_summary(path: Path) -> dict[str, Any]:
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT stock_code) AS stocks,
                   min(trade_date) AS start_date,
                   max(trade_date) AS end_date
            FROM read_parquet('{_sql_path(path)}')
            """
        ).fetchone()
    finally:
        connection.close()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "rows": int(row[0]),
        "stocks": int(row[1]),
        "start_date": str(row[2]),
        "end_date": str(row[3]),
    }


def _load_authority_axes(config: DailyDerivedArtifactConfig) -> tuple[np.ndarray, np.ndarray]:
    dates = np.load(config.date_list_path, allow_pickle=False).astype("datetime64[D]")
    stocks = np.load(config.stock_list_path, allow_pickle=False).astype(str)
    if dates.ndim != 1 or stocks.ndim != 1 or dates.size == 0 or stocks.size == 0:
        raise ValueError("Raw date_list/stock_list 必须是一维非空 authority axis")
    if np.any(dates[1:] <= dates[:-1]) or np.any(stocks[1:] <= stocks[:-1]):
        raise ValueError("Raw date_list/stock_list 必须严格递增且无重复")
    return dates, stocks


def inspect_daily_derived_inputs(
    config: DailyDerivedArtifactConfig = DailyDerivedArtifactConfig(),
) -> dict[str, Any]:
    required = (
        config.market_path, config.raw_close_path, config.date_list_path,
        config.stock_list_path, config.universe_mask_path, config.shares_path,
        config.shares_metadata_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Daily-Derived 输入不存在：{missing}")
    dates, stocks = _load_authority_axes(config)
    shares = np.load(config.shares_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (dates.size, stocks.size)
    if shares.shape != expected_shape or np.dtype(shares.dtype).kind != "f":
        raise ValueError(
            f"PIT shares artifact 必须是浮点 {expected_shape}，实际为 {shares.shape}/{shares.dtype}"
        )
    bad_shares = 0
    finite_shares = 0
    for start in range(0, dates.size, config.date_block_size):
        block = np.asarray(shares[start : start + config.date_block_size])
        finite = np.isfinite(block)
        finite_shares += int(finite.sum())
        bad_shares += int((finite & (block <= 0)).sum())
    if bad_shares:
        raise ValueError(f"PIT shares artifact 含 {bad_shares} 个非正有限值")
    shares_metadata = json.loads(config.shares_metadata_path.read_text(encoding="utf-8"))
    if tuple(shares_metadata.get("shape", ())) != expected_shape:
        raise ValueError("Barra metadata shape 与 Raw authority axes 不一致")
    universe = np.load(config.universe_mask_path, mmap_mode="r", allow_pickle=False)
    if universe.shape != expected_shape or universe.dtype != np.bool_:
        raise ValueError("Raw universe_mask 与 authority axes 不一致")
    return {
        "date_count": int(dates.size),
        "stock_count": int(stocks.size),
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
        "shares_shape": list(map(int, shares.shape)),
        "shares_dtype": str(shares.dtype),
        "shares_finite_count": finite_shares,
        "shares_missing_count": int(shares.size - finite_shares),
        "shares_source_schema": shares_metadata.get("schema"),
        "shares_reuse_contract": daily_derived_schema_contract()["shares_pit_contract"],
        "schema_fingerprint": daily_derived_schema_fingerprint(),
    }


def _aligned_market_block(
    config: DailyDerivedArtifactConfig,
    dates: np.ndarray,
    stocks: np.ndarray,
) -> dict[str, np.ndarray]:
    shape = (dates.size, stocks.size)
    output = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in (
            "open_adjusted", "high_adjusted", "low_adjusted", "close_adjusted",
            "vwap_adjusted", "volume_raw", "amount_raw",
        )
    }
    start_date, end_date = str(dates[0]), str(dates[-1])
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            f"""
            SELECT CAST(m.trade_date AS DATE) AS trade_date,
                   lpad(CAST(m.stock_code AS VARCHAR), 6, '0') AS stock_code,
                   CAST(m.open AS DOUBLE) AS open_adjusted,
                   CAST(m.high AS DOUBLE) AS high_adjusted,
                   CAST(m.low AS DOUBLE) AS low_adjusted,
                   CAST(m.close AS DOUBLE) AS close_adjusted,
                   CAST(m.volume AS DOUBLE) AS volume_raw,
                   CAST(m.amount AS DOUBLE) AS amount_raw,
                   CAST(r.close AS DOUBLE) AS close_raw
            FROM read_parquet('{_sql_path(config.market_path)}') m
            LEFT JOIN read_parquet('{_sql_path(config.raw_close_path)}') r
              USING (trade_date, stock_code)
            WHERE CAST(m.trade_date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            """
        ).fetchdf()
    finally:
        connection.close()
    if frame.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError("Raw market/raw-close join 产生重复 date-stock key")
    if frame.empty:
        return output
    frame_dates = pd.to_datetime(frame["trade_date"]).to_numpy(dtype="datetime64[D]")
    frame_stocks = frame["stock_code"].astype(str).to_numpy()
    date_index = np.searchsorted(dates, frame_dates)
    stock_index = np.searchsorted(stocks, frame_stocks)
    in_bounds = (date_index < dates.size) & (stock_index < stocks.size)
    matched = np.zeros(frame.shape[0], dtype=bool)
    matched[in_bounds] = (
        (dates[date_index[in_bounds]] == frame_dates[in_bounds])
        & (stocks[stock_index[in_bounds]] == frame_stocks[in_bounds])
    )
    if not matched.all():
        raise ValueError("Raw market 包含不属于冻结 authority axes 的 key")
    rows, cols = date_index, stock_index
    for name in (
        "open_adjusted", "high_adjusted", "low_adjusted", "close_adjusted",
        "volume_raw", "amount_raw",
    ):
        output[name][rows, cols] = pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=np.float64
        )
    close_raw = pd.to_numeric(frame["close_raw"], errors="coerce").to_numpy(dtype=np.float64)
    close_adjusted = output["close_adjusted"][rows, cols]
    volume = output["volume_raw"][rows, cols]
    amount = output["amount_raw"][rows, cols]
    high = output["high_adjusted"][rows, cols]
    low = output["low_adjusted"][rows, cols]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        adj_factor = close_adjusted / close_raw
        vwap = amount * adj_factor / volume
    tolerance = config.absolute_tolerance + config.relative_tolerance * np.maximum.reduce(
        [np.abs(high), np.abs(low), np.abs(vwap)]
    )
    valid_vwap = (
        np.isfinite(close_adjusted) & np.isfinite(close_raw) & (close_adjusted > 0)
        & (close_raw > 0) & np.isfinite(volume) & (volume > 0)
        & np.isfinite(amount) & np.isfinite(vwap) & (vwap > 0)
        & np.isfinite(high) & np.isfinite(low)
        & (vwap >= low - tolerance) & (vwap <= high + tolerance)
    )
    output["vwap_adjusted"][rows[valid_vwap], cols[valid_vwap]] = vwap[valid_vwap]
    return output


def _artifact_qa(tensor: np.ndarray) -> dict[str, Any]:
    features = []
    total = tensor.shape[0] * tensor.shape[2]
    for index, name in enumerate(DAILY_DERIVED_FEATURE_NAMES):
        values = np.asarray(tensor[:, index, :])
        finite_values = values[np.isfinite(values)]
        features.append({
            "name": name,
            "finite_count": int(finite_values.size),
            "finite_rate": float(finite_values.size / total),
            "nan_rate": float(1.0 - finite_values.size / total),
            "min": float(np.min(finite_values)) if finite_values.size else None,
            "median": float(np.median(finite_values)) if finite_values.size else None,
            "max": float(np.max(finite_values)) if finite_values.size else None,
        })
    lag1_indices = (0, 1, 4, 5, 6, 7, 10, 13)
    lag5_indices = (11, 14)
    sample = np.asarray(tensor[:: max(1, tensor.shape[0] // 64), :, :: max(1, tensor.shape[2] // 64)])
    identity = sample[:, 4, :] - sample[:, 5, :] - sample[:, 6, :] - sample[:, 7, :]
    finite_identity = np.abs(identity[np.isfinite(identity)])
    return {
        "features": features,
        "warmup": {
            "lag1_first_row_all_nan": bool(np.isnan(tensor[0, lag1_indices, :]).all()),
            "lag5_first_five_rows_all_nan": bool(
                np.isnan(tensor[: min(5, tensor.shape[0]), lag5_indices, :]).all()
            ),
        },
        "kline_identity_sample_max_abs_error": (
            float(np.max(finite_identity)) if finite_identity.size else None
        ),
    }


def build_daily_derived_artifact(
    config: DailyDerivedArtifactConfig = DailyDerivedArtifactConfig(),
) -> dict[str, Any]:
    """Build a new isolated float32 artifact; existing formal outputs are never overwritten."""

    if config.tensor_path.exists() or config.metadata_path.exists():
        raise FileExistsError("Daily-Derived 正式 artifact 已存在；本 builder 不覆盖已有结果")
    preflight = inspect_daily_derived_inputs(config)
    dates, stocks = _load_authority_axes(config)
    shape = (dates.size, len(DAILY_DERIVED_FEATURE_NAMES), stocks.size)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_temporary = config.output_dir / "data_tensor.npy.tmp"
    metadata_temporary = config.output_dir / "metadata.json.tmp"
    tensor_temporary.unlink(missing_ok=True)
    metadata_temporary.unlink(missing_ok=True)
    tensor = None
    try:
        tensor = open_memmap(tensor_temporary, mode="w+", dtype="float32", shape=shape)
        tensor[:] = np.nan
        shares = np.load(config.shares_path, mmap_mode="r", allow_pickle=False)
        for start in range(0, dates.size, config.date_block_size):
            end = min(start + config.date_block_size, dates.size)
            input_start = max(0, start - MAX_LAG)
            block_inputs = _aligned_market_block(config, dates[input_start:end], stocks)
            block_inputs["list_a_shares"] = np.asarray(
                shares[input_start:end], dtype=np.float64
            )
            derived = build_daily_derived_features(**block_inputs)
            output_offset = start - input_start
            tensor[start:end] = derived[output_offset:].astype(np.float32)
        tensor.flush()
        qa = _artifact_qa(tensor)
        del tensor
        tensor = None
        verified = np.load(tensor_temporary, mmap_mode="r", allow_pickle=False)
        if verified.shape != shape or verified.dtype != np.float32:
            raise RuntimeError("临时 Derived tensor shape/dtype 校验失败")
        del verified
        contract = daily_derived_schema_contract()
        metadata = {
            "status": "completed",
            **contract,
            "feature_count": len(DAILY_DERIVED_FEATURE_NAMES),
            "shape": list(map(int, shape)),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "builder_schema_fingerprint": daily_derived_schema_fingerprint(contract),
            "axes": {
                "date": {"reference": str(config.date_list_path), "sha256": _sha256_file(config.date_list_path)},
                "stock": {"reference": str(config.stock_list_path), "sha256": _sha256_file(config.stock_list_path)},
            },
            "universe_mask": {"reference": str(config.universe_mask_path), "contract": "shared read-only; not rebuilt or modified"},
            "raw_source_summary": {
                "market_data": _parquet_summary(config.market_path),
                "raw_close": _parquet_summary(config.raw_close_path),
                "pit_shares": {"path": str(config.shares_path), "shape": preflight["shares_shape"], "dtype": preflight["shares_dtype"], "source_schema": preflight["shares_source_schema"]},
            },
            "price_adjustment_contract": "OHLC adjust_type=2; all price features use one adjusted scale",
            "vwap_validation": {"relative_tolerance": config.relative_tolerance, "absolute_tolerance": config.absolute_tolerance, "range_rule": "low-tolerance <= adjusted VWAP <= high+tolerance"},
            "date_block_size": config.date_block_size,
            "max_lag_overlap": MAX_LAG,
            "qa": qa,
        }
        metadata_temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tensor_temporary, config.tensor_path)
        os.replace(metadata_temporary, config.metadata_path)
        return metadata
    except Exception:
        if tensor is not None:
            tensor.flush()
            del tensor
        tensor_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
        raise


def inspect_daily_derived_artifact(
    config: DailyDerivedArtifactConfig = DailyDerivedArtifactConfig(),
) -> dict[str, Any]:
    if not config.tensor_path.exists() or not config.metadata_path.exists():
        raise FileNotFoundError("Daily-Derived tensor/metadata 尚未完整生成")
    metadata = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    tensor = np.load(config.tensor_path, mmap_mode="r", allow_pickle=False)
    if metadata.get("status") != "completed":
        raise ValueError("Daily-Derived metadata 未标记 completed")
    if metadata.get("builder_schema_fingerprint") != daily_derived_schema_fingerprint():
        raise ValueError("Daily-Derived schema fingerprint 不匹配")
    axes = metadata.get("axes", {})
    if axes.get("date", {}).get("sha256") != _sha256_file(config.date_list_path):
        raise ValueError("Daily-Derived date axis identity 不匹配")
    if axes.get("stock", {}).get("sha256") != _sha256_file(config.stock_list_path):
        raise ValueError("Daily-Derived stock axis identity 不匹配")
    if tuple(metadata.get("shape", ())) != tensor.shape or tensor.dtype != np.float32:
        raise ValueError("Daily-Derived tensor 与 metadata shape/dtype 不一致")
    if tuple(metadata.get("feature_order", ())) != DAILY_DERIVED_FEATURE_NAMES:
        raise ValueError("Daily-Derived feature order 不匹配")
    return metadata


__all__ = [
    "DailyDerivedArtifactConfig", "build_daily_derived_artifact",
    "daily_derived_schema_contract", "daily_derived_schema_fingerprint",
    "inspect_daily_derived_artifact", "inspect_daily_derived_inputs",
]
