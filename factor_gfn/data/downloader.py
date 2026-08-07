"""使用 adata 下载 A 股股票主表、后复权日行情和不复权收盘价。"""

from __future__ import annotations

import shutil
import time
from datetime import date
from pathlib import Path

import adata
import duckdb
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DOWNLOAD_PARTS_DIR = PROJECT_ROOT / "data" / "download_parts"

LISTING_DATES_PATH = RAW_DATA_DIR / "adata_listing_dates.parquet"
MARKET_DATA_PATH = RAW_DATA_DIR / "market_data.parquet"
RAW_CLOSE_PATH = RAW_DATA_DIR / "raw_close.parquet"
STOCK_SHARES_PATH = RAW_DATA_DIR / "stock_shares_history.parquet"
INDUSTRY_SW_PATH = RAW_DATA_DIR / "industry_sw.parquet"

DEFAULT_START_DATE = "2010-01-01"
PART_SIZE = 50
MAX_RETRIES = 3
MAX_CONSECUTIVE_FAILURES = 50

STOCK_COLUMNS = ["stock_code", "short_name", "exchange", "list_date"]
MARKET_COLUMNS = [
    "trade_date",
    "stock_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]
RAW_CLOSE_COLUMNS = ["trade_date", "stock_code", "close"]
STOCK_SHARES_COLUMNS = [
    "stock_code",
    "change_date",
    "total_shares",
    "limit_shares",
    "list_a_shares",
    "change_reason",
]
INDUSTRY_SW_COLUMNS = [
    "stock_code",
    "sw_code",
    "industry_name",
    "industry_type",
    "source",
]


def _normalize_stock_code(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def _atomic_to_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _retry_call(function, label: str, attempts: int = MAX_RETRIES):
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            result = function()
            if result is None or result.empty:
                raise ValueError("接口返回空数据")
            return result
        except Exception as exc:  # 网络和上游接口异常统一进入重试
            errors.append(f"第 {attempt} 次：{exc}")
            if attempt < attempts:
                time.sleep(attempt)
    raise RuntimeError(f"{label}失败；" + "；".join(errors))


def _prepare_stock_list(raw: pd.DataFrame, min_stock_count: int) -> pd.DataFrame:
    missing = set(STOCK_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"all_code() 缺少字段：{sorted(missing)}")

    result = raw.loc[:, STOCK_COLUMNS].copy()
    result["stock_code"] = _normalize_stock_code(result["stock_code"])
    result["short_name"] = result["short_name"].astype("string").str.strip()
    result["exchange"] = result["exchange"].astype("string").str.strip()
    result["list_date"] = pd.to_datetime(
        result["list_date"], errors="coerce"
    ).dt.normalize()
    result = result.dropna(subset=["stock_code"]).copy()
    result = result.loc[result["stock_code"].str.fullmatch(r"\d{6}", na=False)]

    conflicts = result.groupby("stock_code", dropna=False)[STOCK_COLUMNS[1:]].nunique(
        dropna=False
    ).gt(1).any(axis=1)
    if conflicts.any():
        examples = conflicts.index[conflicts].tolist()[:10]
        raise ValueError(f"all_code() 存在同代码冲突记录，示例：{examples}")

    result = (
        result.drop_duplicates("stock_code", keep="last")
        .sort_values("stock_code")
        .reset_index(drop=True)
    )
    if len(result) < int(min_stock_count):
        raise ValueError(
            f"all_code() 仅得到 {len(result):,} 只，少于最低要求 "
            f"{int(min_stock_count):,} 只"
        )
    return result


def download_stock_list(
    force_update: bool = False,
    min_stock_count: int = 5000,
) -> pd.DataFrame:
    """下载并缓存 all_code() 的完整股票主表。"""
    if LISTING_DATES_PATH.exists() and not force_update:
        cached = pd.read_parquet(LISTING_DATES_PATH)
        return _prepare_stock_list(cached, min_stock_count=min_stock_count)

    raw = _retry_call(
        lambda: adata.stock.info.all_code(),
        "adata.stock.info.all_code()",
        attempts=5,
    )
    result = _prepare_stock_list(raw, min_stock_count=min_stock_count)
    _atomic_to_parquet(result, LISTING_DATES_PATH)
    print(f"股票主表已保存：{LISTING_DATES_PATH}")
    print(f"股票数：{len(result):,}；上市日期缺失：{result['list_date'].isna().sum():,}")
    print(result["exchange"].value_counts(dropna=False).to_string())
    return result


def _part_directory(dataset_name: str) -> Path:
    return DOWNLOAD_PARTS_DIR / dataset_name


def _part_files(dataset_name: str) -> list[Path]:
    return sorted(_part_directory(dataset_name).glob("part_*.parquet"))


def _next_part_number(files: list[Path]) -> int:
    if not files:
        return 1
    return max(int(path.stem.removeprefix("part_")) for path in files) + 1


def _codes_in_parquet(path: Path) -> set[str]:
    parquet = pq.ParquetFile(path)
    codes: set[str] = set()
    for row_group in range(parquet.num_row_groups):
        values = parquet.read_row_group(
            row_group, columns=["stock_code"]
        ).column("stock_code").to_pandas()
        codes.update(_normalize_stock_code(pd.Series(values)).dropna().astype(str))
    return codes


def _industry_level1_codes(path: Path) -> set[str]:
    parquet = pq.ParquetFile(path)
    codes: set[str] = set()
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(
            row_group,
            columns=["stock_code", "industry_type"],
        ).to_pandas()
        level1 = frame.loc[
            frame["industry_type"].astype("string").str.strip().eq("申万一级"),
            "stock_code",
        ]
        codes.update(_normalize_stock_code(pd.Series(level1)).dropna().astype(str))
    return codes


def _prepare_market_response(
    response: pd.DataFrame,
    stock_code: str,
    output_columns: list[str],
) -> pd.DataFrame:
    required_source = set(output_columns).difference({"stock_code"})
    missing = required_source.difference(response.columns)
    if missing:
        raise ValueError(f"行情接口缺少字段：{sorted(missing)}")

    result = response.loc[:, [c for c in output_columns if c != "stock_code"]].copy()
    result.insert(1, "stock_code", str(stock_code).zfill(6))
    result["trade_date"] = pd.to_datetime(
        result["trade_date"], errors="coerce"
    ).dt.normalize()
    numeric_columns = [
        column for column in output_columns if column not in {"trade_date", "stock_code"}
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["trade_date", "stock_code"])

    duplicate_dates = result["trade_date"].duplicated(keep=False)
    if duplicate_dates.any():
        duplicate_rows = result.loc[duplicate_dates, output_columns]
        conflicts = duplicate_rows.groupby("trade_date")[numeric_columns].nunique(
            dropna=False
        ).gt(1).any(axis=1)
        if conflicts.any():
            examples = conflicts.index[conflicts].astype(str).tolist()[:10]
            raise ValueError(f"同一股票同日行情存在冲突，示例：{examples}")
        result = result.drop_duplicates("trade_date", keep="last")

    return result.loc[:, output_columns].sort_values("trade_date").reset_index(drop=True)


def _prepare_stock_shares_response(
    response: pd.DataFrame,
    stock_code: str,
) -> pd.DataFrame:
    """Normalize one stock's historical share-capital change records."""
    required_source = {"change_date", "total_shares", "list_a_shares"}
    missing = required_source.difference(response.columns)
    if missing:
        raise ValueError(f"股本接口缺少字段：{sorted(missing)}")

    result = response.loc[:, sorted(required_source)].copy()
    result["limit_shares"] = (
        response["limit_shares"] if "limit_shares" in response.columns else pd.NA
    )
    result["change_reason"] = (
        response["change_reason"] if "change_reason" in response.columns else pd.NA
    )
    result.insert(0, "stock_code", str(stock_code).zfill(6))
    result["change_date"] = pd.to_datetime(
        result["change_date"], errors="coerce"
    ).dt.normalize()
    share_columns = ["total_shares", "limit_shares", "list_a_shares"]
    for column in share_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["change_reason"] = result["change_reason"].astype("string").str.strip()

    core_valid = (
        result["change_date"].notna()
        & result["total_shares"].notna()
        & result["list_a_shares"].notna()
        & result["total_shares"].gt(0)
        & result["list_a_shares"].gt(0)
        & result["total_shares"].ge(result["list_a_shares"])
    )
    invalid_count = int((~core_valid).sum())
    result = result.loc[core_valid, STOCK_SHARES_COLUMNS].copy()
    if result.empty:
        raise ValueError(
            "接口没有可用历史股本记录；要求日期有效、total_shares/list_a_shares "
            "为正且 total_shares >= list_a_shares"
        )

    duplicate_dates = result["change_date"].duplicated(keep=False)
    if duplicate_dates.any():
        duplicate_rows = result.loc[duplicate_dates]
        conflict_columns = [
            "total_shares",
            "limit_shares",
            "list_a_shares",
            "change_reason",
        ]
        conflicts = (
            duplicate_rows.groupby("change_date", dropna=False)[conflict_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            examples = conflicts.index[conflicts].astype(str).tolist()[:10]
            raise ValueError(f"同一股票同一股本变更日存在冲突，示例：{examples}")
        result = result.drop_duplicates("change_date", keep="last")

    normalized = result.sort_values("change_date").reset_index(drop=True)
    normalized.attrs["dropped_invalid_rows"] = invalid_count
    return normalized


def _prepare_industry_sw_response(
    response: pd.DataFrame,
    stock_code: str,
) -> pd.DataFrame:
    """Normalize one stock's current Shenwan level-1/level-2 classifications."""
    required_source = {"sw_code", "industry_name", "industry_type"}
    missing = required_source.difference(response.columns)
    if missing:
        raise ValueError(f"申万行业接口缺少字段：{sorted(missing)}")

    result = response.loc[:, sorted(required_source)].copy()
    result["source"] = response["source"] if "source" in response.columns else pd.NA
    result.insert(0, "stock_code", str(stock_code).zfill(6))
    for column in ("sw_code", "industry_name", "industry_type", "source"):
        result[column] = result[column].astype("string").str.strip()

    valid_types = {"申万一级", "申万二级"}
    valid = (
        result["sw_code"].notna()
        & result["sw_code"].ne("")
        & result["industry_name"].notna()
        & result["industry_name"].ne("")
        & result["industry_type"].isin(valid_types)
    )
    result = result.loc[valid, INDUSTRY_SW_COLUMNS].copy()
    if result.empty or not result["industry_type"].eq("申万一级").any():
        raise ValueError("接口没有有效的申万一级行业记录")

    duplicate_types = result["industry_type"].duplicated(keep=False)
    if duplicate_types.any():
        conflicts = (
            result.loc[duplicate_types]
            .groupby("industry_type", dropna=False)[["sw_code", "industry_name"]]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            examples = conflicts.index[conflicts].astype(str).tolist()
            raise ValueError(f"同一股票同一行业层级存在冲突：{examples}")
        result = result.drop_duplicates("industry_type", keep="last")

    return result.sort_values("industry_type").reset_index(drop=True)


def _save_part(dataset_name: str, frames: list[pd.DataFrame], number: int) -> Path:
    directory = _part_directory(dataset_name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"part_{number:04d}.parquet"
    _atomic_to_parquet(pd.concat(frames, ignore_index=True), path)
    return path


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _consolidate_parts(
    sources: list[Path],
    output_path: Path,
    columns: list[str],
    key_columns: tuple[str, ...] = ("trade_date", "stock_code"),
) -> None:
    if not sources:
        raise RuntimeError("没有可合并的下载分段")

    source_sql = "[" + ",".join(f"'{_sql_path(path)}'" for path in sources) + "]"
    value_columns = [column for column in columns if column not in set(key_columns)]
    key_sql = ", ".join(f'"{column}"' for column in key_columns)
    hash_values = ", ".join(f'"{column}"' for column in value_columns)
    connection = duckdb.connect()
    try:
        conflict_count = connection.execute(
            f"""
            SELECT count(*)
            FROM (
                SELECT {key_sql}
                FROM read_parquet({source_sql}, union_by_name=true)
                GROUP BY {key_sql}
                HAVING count(DISTINCT hash({hash_values})) > 1
            )
            """
        ).fetchone()[0]
        if conflict_count:
            raise ValueError(f"分段中存在 {conflict_count:,} 个同键冲突记录")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        selected = ", ".join(f'"{column}"' for column in columns)
        connection.execute(
            f"""
            COPY (
                SELECT {selected}
                FROM read_parquet({source_sql}, union_by_name=true)
                QUALIFY row_number() OVER (
                    PARTITION BY {key_sql}
                    ORDER BY {key_sql}
                ) = 1
                ORDER BY {key_sql}
            ) TO '{_sql_path(temporary)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
            """
        )
    finally:
        connection.close()

    # Windows 不允许替换仍被 DuckDB 读取的源文件；连接关闭后再原子替换。
    temporary.replace(output_path)


def _download_market_variant(
    *,
    dataset_name: str,
    output_path: Path,
    output_columns: list[str],
    adjust_type: int,
    start_date: str,
    end_date: str | None,
    force_update: bool,
) -> dict:
    stocks = download_stock_list(force_update=False)
    codes = stocks["stock_code"].astype(str).tolist()
    resolved_end = end_date or date.today().isoformat()
    parts_directory = _part_directory(dataset_name)

    if force_update:
        shutil.rmtree(parts_directory, ignore_errors=True)
        output_path.unlink(missing_ok=True)

    files = _part_files(dataset_name)
    completed: set[str] = set()
    sources: list[Path] = []
    if output_path.exists():
        completed.update(_codes_in_parquet(output_path))
        sources.append(output_path)
    for path in files:
        completed.update(_codes_in_parquet(path))
        sources.append(path)

    pending = [code for code in codes if code not in completed]
    print(
        f"{dataset_name}：股票池 {len(codes):,} 只，已完成 {len(completed):,} 只，"
        f"待下载 {len(pending):,} 只。"
    )

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    next_part = _next_part_number(files)
    consecutive_failures = 0

    for code in tqdm(pending, desc=f"下载 {dataset_name}"):
        try:
            response = _retry_call(
                lambda code=code: adata.stock.market.get_market(
                    stock_code=code,
                    start_date=start_date,
                    end_date=resolved_end,
                    k_type=1,
                    adjust_type=adjust_type,
                ),
                f"{code} 行情",
            )
            prepared = _prepare_market_response(response, code, output_columns)
            if prepared.empty:
                raise ValueError("接口返回空数据")
            frames.append(prepared)
            completed.add(code)
            consecutive_failures = 0
        except Exception as exc:
            failures.append((code, str(exc)))
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"连续失败 {MAX_CONSECUTIVE_FAILURES} 只，已停止本轮。"
                    "稍后重新运行同一单元即可续传。"
                )
                break

        if len(frames) >= PART_SIZE:
            path = _save_part(dataset_name, frames, next_part)
            sources.append(path)
            next_part += 1
            frames.clear()

    if frames:
        path = _save_part(dataset_name, frames, next_part)
        sources.append(path)
        frames.clear()

    # sources 中可能包含旧最终文件；重新扫描分段，避免漏掉本轮落盘文件。
    sources = ([output_path] if output_path.exists() else []) + _part_files(dataset_name)
    if sources:
        _consolidate_parts(sources, output_path, output_columns)

    final_codes = _codes_in_parquet(output_path) if output_path.exists() else set()
    missing_codes = [code for code in codes if code not in final_codes]
    print(f"已保存：{output_path}")
    print(
        f"最终覆盖 {len(final_codes):,}/{len(codes):,} 只；"
        f"仍缺失 {len(missing_codes):,} 只。"
    )
    if failures:
        print("本轮失败示例：", failures[:20])
    if missing_codes:
        print("待重试代码示例：", missing_codes[:30])
    return {
        "path": output_path,
        "stock_count": len(final_codes),
        "missing_codes": missing_codes,
        "failures": failures,
    }


def download_adjusted_market(
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    force_update: bool = False,
) -> dict:
    """下载 k_type=1、adjust_type=2 的后复权日频 OHLCVA。"""
    return _download_market_variant(
        dataset_name="market_adjusted",
        output_path=MARKET_DATA_PATH,
        output_columns=MARKET_COLUMNS,
        adjust_type=2,
        start_date=start_date,
        end_date=end_date,
        force_update=force_update,
    )


def download_raw_close(
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    force_update: bool = False,
) -> dict:
    """下载 k_type=1、adjust_type=0 的不复权收盘价。"""
    return _download_market_variant(
        dataset_name="market_raw_close",
        output_path=RAW_CLOSE_PATH,
        output_columns=RAW_CLOSE_COLUMNS,
        adjust_type=0,
        start_date=start_date,
        end_date=end_date,
        force_update=force_update,
    )


def download_stock_shares(force_update: bool = False) -> dict:
    """Download historical share-capital changes with stock-level checkpoints.

    Successful stocks are saved in ``data/download_parts/stock_shares_history``.
    Failed or empty responses are not marked complete, so rerunning with
    ``force_update=False`` retries only the still-missing stocks.
    """
    dataset_name = "stock_shares_history"
    stocks = download_stock_list(force_update=False)
    codes = stocks["stock_code"].astype(str).tolist()
    parts_directory = _part_directory(dataset_name)

    if force_update:
        shutil.rmtree(parts_directory, ignore_errors=True)
        STOCK_SHARES_PATH.unlink(missing_ok=True)

    files = _part_files(dataset_name)
    completed: set[str] = set()
    if STOCK_SHARES_PATH.exists():
        completed.update(_codes_in_parquet(STOCK_SHARES_PATH))
    for path in files:
        completed.update(_codes_in_parquet(path))

    pending = [code for code in codes if code not in completed]
    print(
        f"{dataset_name}：股票池 {len(codes):,} 只，已完成 {len(completed):,} 只，"
        f"待下载 {len(pending):,} 只。"
    )

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    filtered_rows = 0
    filtered_examples: list[tuple[str, int]] = []
    next_part = _next_part_number(files)
    consecutive_failures = 0

    for code in tqdm(pending, desc="下载历史股本"):
        try:
            response = _retry_call(
                lambda code=code: adata.stock.info.get_stock_shares(
                    stock_code=code,
                    is_history=True,
                ),
                f"{code} 历史股本",
            )
            prepared = _prepare_stock_shares_response(response, code)
            dropped = int(prepared.attrs.get("dropped_invalid_rows", 0))
            filtered_rows += dropped
            if dropped and len(filtered_examples) < 20:
                filtered_examples.append((code, dropped))
            frames.append(prepared)
            completed.add(code)
            consecutive_failures = 0
        except Exception as exc:
            failures.append((code, str(exc)))
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"连续失败 {MAX_CONSECUTIVE_FAILURES} 只，已停止本轮。"
                    "稍后重新运行同一单元即可续传。"
                )
                break

        if len(frames) >= PART_SIZE:
            _save_part(dataset_name, frames, next_part)
            next_part += 1
            frames.clear()

    if frames:
        _save_part(dataset_name, frames, next_part)
        frames.clear()

    sources = ([STOCK_SHARES_PATH] if STOCK_SHARES_PATH.exists() else []) + _part_files(
        dataset_name
    )
    if sources:
        _consolidate_parts(
            sources,
            STOCK_SHARES_PATH,
            STOCK_SHARES_COLUMNS,
            key_columns=("stock_code", "change_date"),
        )

    final_codes = (
        _codes_in_parquet(STOCK_SHARES_PATH)
        if STOCK_SHARES_PATH.exists()
        else set()
    )
    missing_codes = [code for code in codes if code not in final_codes]
    if STOCK_SHARES_PATH.exists():
        print(f"已保存：{STOCK_SHARES_PATH}")
    else:
        print("本轮没有可用于生成最终历史股本文件的有效分段。")
    print(
        f"最终覆盖 {len(final_codes):,}/{len(codes):,} 只；"
        f"仍缺失 {len(missing_codes):,} 只。"
    )
    if failures:
        print("本轮失败示例：", failures[:20])
    if filtered_rows:
        print(f"本轮过滤核心字段无效的历史记录：{filtered_rows:,} 条。")
        print("过滤记录按股票示例：", filtered_examples)
    if missing_codes:
        print("待重试代码示例：", missing_codes[:30])
    return {
        "path": STOCK_SHARES_PATH,
        "stock_count": len(final_codes),
        "missing_codes": missing_codes,
        "failures": failures,
        "filtered_invalid_rows": filtered_rows,
        "filtered_examples": filtered_examples,
    }


def download_industry_sw(force_update: bool = False) -> dict:
    """Download current Shenwan level-1/level-2 industries with checkpoints.

    Each successful stock must contain a valid ``申万一级`` record. Successful
    responses are checkpointed under ``data/download_parts/industry_sw``;
    rerunning with ``force_update=False`` retries only missing stocks.
    """
    dataset_name = "industry_sw"
    stocks = download_stock_list(force_update=False)
    codes = stocks["stock_code"].astype(str).tolist()
    parts_directory = _part_directory(dataset_name)

    if force_update:
        shutil.rmtree(parts_directory, ignore_errors=True)
        INDUSTRY_SW_PATH.unlink(missing_ok=True)

    files = _part_files(dataset_name)
    completed: set[str] = set()
    if INDUSTRY_SW_PATH.exists():
        completed.update(_industry_level1_codes(INDUSTRY_SW_PATH))
    for path in files:
        completed.update(_industry_level1_codes(path))

    pending = [code for code in codes if code not in completed]
    print(
        f"{dataset_name}：股票池 {len(codes):,} 只，已完成 {len(completed):,} 只，"
        f"待下载 {len(pending):,} 只。"
    )

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    next_part = _next_part_number(files)
    consecutive_failures = 0

    for code in tqdm(pending, desc="下载申万行业"):
        try:
            response = _retry_call(
                lambda code=code: adata.stock.info.get_industry_sw(
                    stock_code=code,
                ),
                f"{code} 申万行业",
            )
            prepared = _prepare_industry_sw_response(response, code)
            frames.append(prepared)
            completed.add(code)
            consecutive_failures = 0
        except Exception as exc:
            failures.append((code, str(exc)))
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"连续失败 {MAX_CONSECUTIVE_FAILURES} 只，已停止本轮。"
                    "稍后重新运行同一单元即可续传。"
                )
                break

        if len(frames) >= PART_SIZE:
            _save_part(dataset_name, frames, next_part)
            next_part += 1
            frames.clear()

    if frames:
        _save_part(dataset_name, frames, next_part)
        frames.clear()

    sources = ([INDUSTRY_SW_PATH] if INDUSTRY_SW_PATH.exists() else []) + _part_files(
        dataset_name
    )
    if sources:
        _consolidate_parts(
            sources,
            INDUSTRY_SW_PATH,
            INDUSTRY_SW_COLUMNS,
            key_columns=("stock_code", "industry_type"),
        )

    final_codes = (
        _industry_level1_codes(INDUSTRY_SW_PATH)
        if INDUSTRY_SW_PATH.exists()
        else set()
    )
    missing_codes = [code for code in codes if code not in final_codes]
    if INDUSTRY_SW_PATH.exists():
        print(f"已保存：{INDUSTRY_SW_PATH}")
    else:
        print("本轮没有可用于生成最终申万行业文件的有效分段。")
    print(
        f"最终覆盖 {len(final_codes):,}/{len(codes):,} 只；"
        f"仍缺失 {len(missing_codes):,} 只。"
    )
    if failures:
        print("本轮失败示例：", failures[:20])
    if missing_codes:
        print("待重试代码示例：", missing_codes[:30])
    return {
        "path": INDUSTRY_SW_PATH,
        "stock_count": len(final_codes),
        "missing_codes": missing_codes,
        "failures": failures,
    }


def _summarize_parquet(path: Path, required_columns: list[str]) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}

    selected = ", ".join(f'"{column}"' for column in required_columns)
    null_terms = ", ".join(
        f'sum(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) AS "{column}"'
        for column in required_columns
    )
    connection = duckdb.connect()
    try:
        base = connection.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT stock_code) AS stocks,
                   min(trade_date) AS start_date,
                   max(trade_date) AS end_date,
                   count(*) - count(DISTINCT (trade_date, stock_code)) AS duplicates
            FROM read_parquet('{_sql_path(path)}')
            """
        ).fetchone()
        null_values = connection.execute(
            f"SELECT {null_terms} FROM read_parquet('{_sql_path(path)}')"
        ).fetchone()
        return {
            "path": str(path),
            "exists": True,
            "rows": int(base[0]),
            "stocks": int(base[1]),
            "start_date": base[2],
            "end_date": base[3],
            "duplicates": int(base[4]),
            "nulls": dict(zip(required_columns, map(int, null_values))),
            "columns": selected,
        }
    finally:
        connection.close()


def _summarize_stock_shares(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}

    null_terms = ", ".join(
        f'sum(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) AS "{column}"'
        for column in STOCK_SHARES_COLUMNS
    )
    connection = duckdb.connect()
    try:
        base = connection.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT stock_code) AS stocks,
                   min(change_date) AS start_date,
                   max(change_date) AS end_date,
                   count(*) - count(DISTINCT (stock_code, change_date)) AS duplicates,
                   sum(CASE WHEN total_shares <= 0 THEN 1 ELSE 0 END) AS bad_total,
                   sum(CASE WHEN limit_shares < 0 THEN 1 ELSE 0 END) AS bad_limit,
                   sum(CASE WHEN list_a_shares <= 0 THEN 1 ELSE 0 END) AS bad_list_a,
                   sum(CASE WHEN total_shares < list_a_shares THEN 1 ELSE 0 END)
                       AS total_less_than_list_a
            FROM read_parquet('{_sql_path(path)}')
            """
        ).fetchone()
        null_values = connection.execute(
            f"SELECT {null_terms} FROM read_parquet('{_sql_path(path)}')"
        ).fetchone()
        null_counts = dict(zip(STOCK_SHARES_COLUMNS, map(int, null_values)))
        row_count = int(base[0])
        return {
            "path": str(path),
            "exists": True,
            "rows": row_count,
            "stocks": int(base[1]),
            "start_date": base[2],
            "end_date": base[3],
            "duplicates": int(base[4]),
            "invalid_share_rows": {
                "total_shares_le_0": int(base[5] or 0),
                "limit_shares_lt_0": int(base[6] or 0),
                "list_a_shares_le_0": int(base[7] or 0),
                "total_less_than_list_a": int(base[8] or 0),
            },
            "nulls": null_counts,
            "limit_shares_null_rate": (
                null_counts["limit_shares"] / row_count if row_count else None
            ),
        }
    finally:
        connection.close()


def _summarize_industry_sw(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}

    null_terms = ", ".join(
        f'sum(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) AS "{column}"'
        for column in INDUSTRY_SW_COLUMNS
    )
    connection = duckdb.connect()
    try:
        base = connection.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT stock_code) AS stocks,
                   count(DISTINCT stock_code) FILTER (
                       WHERE industry_type = '申万一级'
                   ) AS level1_stocks,
                   count(DISTINCT stock_code) FILTER (
                       WHERE industry_type = '申万二级'
                   ) AS level2_stocks,
                   count(*) - count(DISTINCT (stock_code, industry_type))
                       AS duplicates,
                   count(DISTINCT industry_name) FILTER (
                       WHERE industry_type = '申万一级'
                   ) AS level1_industries
            FROM read_parquet('{_sql_path(path)}')
            """
        ).fetchone()
        null_values = connection.execute(
            f"SELECT {null_terms} FROM read_parquet('{_sql_path(path)}')"
        ).fetchone()
        return {
            "path": str(path),
            "exists": True,
            "rows": int(base[0]),
            "stocks": int(base[1]),
            "level1_stocks": int(base[2]),
            "level2_stocks": int(base[3]),
            "duplicates": int(base[4]),
            "level1_industries": int(base[5]),
            "nulls": dict(zip(INDUSTRY_SW_COLUMNS, map(int, null_values))),
        }
    finally:
        connection.close()


def print_download_summary() -> dict:
    """Print lightweight QA summaries for all raw download outputs."""
    summary = {
        "market_data": _summarize_parquet(MARKET_DATA_PATH, MARKET_COLUMNS),
        "raw_close": _summarize_parquet(RAW_CLOSE_PATH, RAW_CLOSE_COLUMNS),
        "stock_shares_history": _summarize_stock_shares(STOCK_SHARES_PATH),
        "industry_sw": _summarize_industry_sw(INDUSTRY_SW_PATH),
    }
    for name, values in summary.items():
        print(f"\n[{name}]")
        for key, value in values.items():
            if key != "columns":
                print(f"{key}: {value}")
    return summary


__all__ = [
    "DEFAULT_START_DATE",
    "LISTING_DATES_PATH",
    "MARKET_DATA_PATH",
    "RAW_CLOSE_PATH",
    "STOCK_SHARES_PATH",
    "INDUSTRY_SW_PATH",
    "download_stock_list",
    "download_adjusted_market",
    "download_raw_close",
    "download_stock_shares",
    "download_industry_sw",
    "print_download_summary",
]
