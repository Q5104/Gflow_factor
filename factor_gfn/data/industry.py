"""Prepare and align point-in-time Shenwan industry classifications.

The external ``参考文件/swind/`` directory contains one UTF-8 CSV per trading day.
Long-running conversion is never started on import; callers must explicitly
invoke :func:`build_sw_industry_daily`.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd

from .downloader import PROJECT_ROOT


SWIND_SOURCE_DIR = PROJECT_ROOT / "参考文件" / "swind"
INDUSTRY_SW_DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "industry_sw_daily.parquet"
INDUSTRY_SW_DAILY_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "industry_sw_daily_metadata.json"
)
DAILY_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "daily_clean.parquet"

SWIND_INPUT_COLUMNS = (
    "TradingDay",
    "StockCode",
    "StockName",
    "SWCode1",
    "SWName1",
    "SWCode2",
    "SWName2",
    "SWCode3",
    "SWName3",
)
INDUSTRY_SW_DAILY_COLUMNS = (
    "trade_date",
    "stock_code",
    "stock_name",
    "sw_code_1",
    "sw_name_1",
    "sw_code_2",
    "sw_name_2",
    "sw_code_3",
    "sw_name_3",
)

_SOURCE_FILE_RE = re.compile(r"^swind_(\d{8})\.csv$")


@dataclass(frozen=True, slots=True)
class IndustryBuildConfig:
    """Paths and batching options for the point-in-time industry pipeline."""

    source_dir: Path = SWIND_SOURCE_DIR
    market_keys_path: Path = DAILY_CLEAN_PATH
    output_path: Path = INDUSTRY_SW_DAILY_PATH
    metadata_path: Path = INDUSTRY_SW_DAILY_METADATA_PATH
    source_glob: str = "swind_*.csv"
    batch_rows: int = 250_000

    def __post_init__(self) -> None:
        for field_name in ("source_dir", "market_keys_path", "output_path", "metadata_path"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).resolve())
        if not self.source_glob or Path(self.source_glob).name != self.source_glob:
            raise ValueError("source_glob 必须是 source_dir 下的文件名模式")
        if self.batch_rows <= 0:
            raise ValueError("batch_rows 必须为正整数")


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _source_files(config: IndustryBuildConfig) -> list[Path]:
    if not config.source_dir.is_dir():
        raise FileNotFoundError(f"申万逐日 CSV 目录不存在：{config.source_dir}")
    files = sorted(config.source_dir.glob(config.source_glob))
    if not files:
        raise FileNotFoundError(
            f"{config.source_dir} 中没有匹配 {config.source_glob!r} 的 CSV"
        )
    return files


def _validate_source_headers(files: list[Path]) -> tuple[str, str]:
    seen_dates: set[str] = set()
    first_date: str | None = None
    last_date: str | None = None
    expected = list(SWIND_INPUT_COLUMNS)
    for path in files:
        match = _SOURCE_FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"申万源文件名不符合 swind_YYYYMMDD.csv：{path.name}")
        compact_date = match.group(1)
        if compact_date in seen_dates:
            raise ValueError(f"申万源文件日期重复：{compact_date}")
        seen_dates.add(compact_date)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), None)
        if header != expected:
            raise ValueError(
                f"申万源文件表头不一致：{path.name}；实际={header}；期望={expected}"
            )
        first_date = compact_date if first_date is None else min(first_date, compact_date)
        last_date = compact_date if last_date is None else max(last_date, compact_date)
    assert first_date is not None and last_date is not None
    return first_date, last_date


def _source_scan_sql(config: IndustryBuildConfig) -> str:
    source_glob = _sql_path(config.source_dir / config.source_glob)
    return (
        "read_csv_auto("
        f"'{source_glob}', header=true, all_varchar=true, "
        "union_by_name=true, filename=true)"
    )


def _parsed_source_sql(config: IndustryBuildConfig) -> str:
    scan = _source_scan_sql(config)
    return f"""
        SELECT
            CAST(try_strptime(trim(TradingDay), '%Y-%m-%d') AS DATE) AS trade_date,
            CAST(
                try_strptime(
                    regexp_extract(filename, 'swind_([0-9]{{8}})\\.csv$', 1),
                    '%Y%m%d'
                ) AS DATE
            ) AS file_date,
            trim(StockCode) AS raw_stock_code,
            CASE
                WHEN regexp_full_match(trim(StockCode), '[0-9]{{6}}\\.(SH|SZ|BJ)')
                THEN left(trim(StockCode), 6)
                ELSE NULL
            END AS stock_code,
            nullif(trim(StockName), '') AS stock_name,
            CASE WHEN regexp_full_match(trim(SWCode1), '[0-9]{{6}}\\.SI')
                 THEN left(trim(SWCode1), 6) ELSE NULL END AS sw_code_1,
            nullif(trim(SWName1), '') AS sw_name_1,
            CASE WHEN regexp_full_match(trim(SWCode2), '[0-9]{{6}}\\.SI')
                 THEN left(trim(SWCode2), 6) ELSE NULL END AS sw_code_2,
            nullif(trim(SWName2), '') AS sw_name_2,
            CASE WHEN regexp_full_match(trim(SWCode3), '[0-9]{{6}}\\.SI')
                 THEN left(trim(SWCode3), 6) ELSE NULL END AS sw_code_3,
            nullif(trim(SWName3), '') AS sw_name_3
        FROM {scan}
    """


def _market_key_summary(config: IndustryBuildConfig, connection: duckdb.DuckDBPyConnection) -> dict:
    if not config.market_keys_path.is_file():
        raise FileNotFoundError(f"行情键文件不存在：{config.market_keys_path}")
    path = _sql_path(config.market_keys_path)
    summary = connection.execute(
        f"""
        SELECT count(*) AS row_count,
               count(DISTINCT stock_code) AS stock_count,
               count(DISTINCT trade_date) AS date_count,
               min(CAST(trade_date AS DATE)) AS start_date,
               max(CAST(trade_date AS DATE)) AS end_date,
               count(*) - count(DISTINCT (trade_date, stock_code)) AS duplicate_count
        FROM read_parquet('{path}')
        """
    ).fetchone()
    result = dict(
        zip(
            (
                "row_count",
                "stock_count",
                "date_count",
                "start_date",
                "end_date",
                "duplicate_count",
            ),
            summary,
        )
    )
    if int(result["row_count"]) <= 0:
        raise ValueError("行情键文件为空")
    if int(result["duplicate_count"]) != 0:
        raise ValueError("行情键文件存在重复 (trade_date, stock_code)")
    return result


def inspect_sw_industry_source(
    config: IndustryBuildConfig = IndustryBuildConfig(),
) -> dict:
    """Scan source contracts and return QA without writing any output."""

    files = _source_files(config)
    file_start, file_end = _validate_source_headers(files)
    connection = duckdb.connect()
    try:
        market = _market_key_summary(config, connection)
        parsed = _parsed_source_sql(config)
        start_date = market["start_date"]
        end_date = market["end_date"]
        source_values = connection.execute(
            f"""
            WITH parsed AS ({parsed}), target AS (
                SELECT * FROM parsed
                WHERE file_date BETWEEN ? AND ?
            )
            SELECT
                count(*) AS row_count,
                count(DISTINCT file_date) AS date_count,
                count(*) - count(DISTINCT (trade_date, stock_code))
                    AS duplicate_normalized_keys,
                sum(CASE WHEN trade_date IS NULL THEN 1 ELSE 0 END) AS invalid_trade_date,
                sum(CASE WHEN trade_date IS DISTINCT FROM file_date THEN 1 ELSE 0 END)
                    AS filename_date_mismatch,
                sum(CASE WHEN stock_code IS NULL THEN 1 ELSE 0 END) AS invalid_stock_code,
                sum(CASE WHEN stock_name IS NULL THEN 1 ELSE 0 END) AS missing_stock_name,
                sum(CASE WHEN sw_code_1 IS NULL OR sw_name_1 IS NULL THEN 1 ELSE 0 END)
                    AS invalid_level1,
                sum(CASE WHEN sw_code_2 IS NULL OR sw_name_2 IS NULL THEN 1 ELSE 0 END)
                    AS invalid_level2,
                sum(CASE WHEN sw_code_3 IS NULL OR sw_name_3 IS NULL THEN 1 ELSE 0 END)
                    AS invalid_level3,
                sum(CASE WHEN ends_with(raw_stock_code, '.SH') THEN 1 ELSE 0 END) AS sh_rows,
                sum(CASE WHEN ends_with(raw_stock_code, '.SZ') THEN 1 ELSE 0 END) AS sz_rows,
                sum(CASE WHEN ends_with(raw_stock_code, '.BJ') THEN 1 ELSE 0 END) AS bj_rows
            FROM target
            """,
            [start_date, end_date],
        ).fetchone()
        source = dict(
            zip(
                (
                    "row_count",
                    "date_count",
                    "duplicate_normalized_keys",
                    "invalid_trade_date",
                    "filename_date_mismatch",
                    "invalid_stock_code",
                    "missing_stock_name",
                    "invalid_level1",
                    "invalid_level2",
                    "invalid_level3",
                    "sh_rows",
                    "sz_rows",
                    "bj_rows",
                ),
                map(int, source_values),
            )
        )
        market_dates = {
            value[0].strftime("%Y%m%d")
            for value in connection.execute(
                f"SELECT DISTINCT CAST(trade_date AS DATE) "
                f"FROM read_parquet('{_sql_path(config.market_keys_path)}')"
            ).fetchall()
        }
    finally:
        connection.close()
    file_dates = {_SOURCE_FILE_RE.fullmatch(path.name).group(1) for path in files}
    source["market_dates_missing_in_source"] = len(market_dates.difference(file_dates))
    return {
        "source_files": {
            "count": len(files),
            "start_date": file_start,
            "end_date": file_end,
        },
        "market_keys": market,
        "source_in_market_date_range": source,
    }


def _validate_source_summary(summary: dict) -> None:
    source = summary["source_in_market_date_range"]
    required_zero = (
        "invalid_trade_date",
        "filename_date_mismatch",
        "invalid_stock_code",
        "missing_stock_name",
        "invalid_level1",
        "invalid_level2",
        "invalid_level3",
        "duplicate_normalized_keys",
        "market_dates_missing_in_source",
    )
    failures = {key: source[key] for key in required_zero if int(source[key]) != 0}
    if failures:
        raise ValueError(f"申万逐日行业源数据未通过严格校验：{failures}")


def inspect_sw_industry_output(path: str | Path = INDUSTRY_SW_DAILY_PATH) -> dict:
    """Return key coverage, null rates and worst dates for a built Parquet."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"申万点时行业长表不存在：{resolved}")
    sql_path = _sql_path(resolved)
    connection = duckdb.connect()
    try:
        values = connection.execute(
            f"""
            SELECT count(*) AS row_count,
                   count(DISTINCT stock_code) AS stock_count,
                   count(DISTINCT trade_date) AS date_count,
                   min(trade_date) AS start_date,
                   max(trade_date) AS end_date,
                   count(*) - count(DISTINCT (trade_date, stock_code)) AS duplicate_count,
                   sum(CASE WHEN sw_code_1 IS NULL OR sw_name_1 IS NULL THEN 1 ELSE 0 END)
                       AS missing_level1,
                   sum(CASE WHEN sw_code_2 IS NULL OR sw_name_2 IS NULL THEN 1 ELSE 0 END)
                       AS missing_level2,
                   sum(CASE WHEN sw_code_3 IS NULL OR sw_name_3 IS NULL THEN 1 ELSE 0 END)
                       AS missing_level3,
                   sum(CASE WHEN stock_name IS NULL THEN 1 ELSE 0 END) AS missing_source_rows,
                   count(DISTINCT sw_code_1) AS level1_codes,
                   count(DISTINCT sw_code_2) AS level2_codes,
                   count(DISTINCT sw_code_3) AS level3_codes
            FROM read_parquet('{sql_path}')
            """
        ).fetchone()
        keys = (
            "row_count",
            "stock_count",
            "date_count",
            "start_date",
            "end_date",
            "duplicate_count",
            "missing_level1",
            "missing_level2",
            "missing_level3",
            "missing_source_rows",
            "level1_codes",
            "level2_codes",
            "level3_codes",
        )
        summary = dict(zip(keys, values))
        row_count = int(summary["row_count"])
        for level in (1, 2, 3):
            missing = int(summary[f"missing_level{level}"])
            summary[f"missing_level{level}_rate"] = missing / row_count if row_count else None
        worst_dates = connection.execute(
            f"""
            SELECT trade_date, count(*) AS rows,
                   avg(CASE WHEN sw_code_1 IS NULL THEN 1.0 ELSE 0.0 END)
                       AS missing_level1_rate,
                   count(DISTINCT sw_code_1) AS level1_codes
            FROM read_parquet('{sql_path}')
            GROUP BY trade_date
            ORDER BY missing_level1_rate DESC, trade_date
            LIMIT 10
            """
        ).fetchdf()
        conflicts = {}
        for level in (1, 2, 3):
            conflicts[f"level{level}_code_name_conflicts"] = int(
                connection.execute(
                    f"""
                    SELECT count(*) FROM (
                        SELECT trade_date, sw_code_{level}
                        FROM read_parquet('{sql_path}')
                        WHERE sw_code_{level} IS NOT NULL
                        GROUP BY trade_date, sw_code_{level}
                        HAVING count(DISTINCT sw_name_{level}) > 1
                    )
                    """
                ).fetchone()[0]
            )
    finally:
        connection.close()
    summary["worst_level1_dates"] = worst_dates.to_dict(orient="records")
    summary.update(conflicts)
    return summary


def build_sw_industry_daily(
    config: IndustryBuildConfig = IndustryBuildConfig(),
) -> dict:
    """Build a point-in-time industry table aligned exactly to market keys.

    The source is scanned with DuckDB. Market keys form the left side of the
    join, so pre-listing, delisted-only and otherwise out-of-scope source rows
    cannot create extra observations in the output.
    """

    source_summary = inspect_sw_industry_source(config)
    _validate_source_summary(source_summary)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_path.with_suffix(config.output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    parsed = _parsed_source_sql(config)
    market_path = _sql_path(config.market_keys_path)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            COPY (
                WITH source AS ({parsed}), market_keys AS (
                    SELECT CAST(trade_date AS DATE) AS trade_date,
                           lpad(CAST(stock_code AS VARCHAR), 6, '0') AS stock_code
                    FROM read_parquet('{market_path}')
                )
                SELECT
                    market_keys.trade_date,
                    market_keys.stock_code,
                    source.stock_name,
                    source.sw_code_1,
                    source.sw_name_1,
                    source.sw_code_2,
                    source.sw_name_2,
                    source.sw_code_3,
                    source.sw_name_3
                FROM market_keys
                LEFT JOIN source USING (trade_date, stock_code)
                ORDER BY market_keys.trade_date, market_keys.stock_code
            ) TO '{_sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        connection.close()

    output_summary = inspect_sw_industry_output(temporary)
    market_rows = int(source_summary["market_keys"]["row_count"])
    if int(output_summary["row_count"]) != market_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"行业输出行数 {output_summary['row_count']} 与行情键 {market_rows} 不一致"
        )
    if int(output_summary["duplicate_count"]) != 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("行业输出存在重复 (trade_date, stock_code)")
    conflicts = {
        key: output_summary[key]
        for key in (
            "level1_code_name_conflicts",
            "level2_code_name_conflicts",
            "level3_code_name_conflicts",
        )
        if int(output_summary[key]) != 0
    }
    if conflicts:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"行业输出存在同日同代码多名称冲突：{conflicts}")
    matched_rows = market_rows - int(output_summary["missing_source_rows"])
    output_summary["matched_source_rows"] = matched_rows
    output_summary["excluded_source_rows"] = (
        int(source_summary["source_in_market_date_range"]["row_count"]) - matched_rows
    )
    try:
        os.replace(temporary, config.output_path)
    except PermissionError as exc:
        raise PermissionError(
            f"无法替换 {config.output_path}；请关闭占用该文件的 Notebook 变量或程序后重试"
        ) from exc

    metadata = {
        "schema": "factor_gfn.industry_sw_daily.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_rule": "daily_clean market keys LEFT JOIN normalized daily SW industry",
        "neutralization_label": "sw_code_1",
        "missing_rule": "keep market key and leave all industry fields null",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "source_summary": source_summary,
        "output_summary": output_summary,
    }
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary = config.metadata_path.with_suffix(config.metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(metadata_temporary, config.metadata_path)
    return metadata


def load_sw_industry_panel(
    date_list: npt.ArrayLike,
    stock_list: npt.ArrayLike,
    *,
    level: int = 1,
    path: str | Path = INDUSTRY_SW_DAILY_PATH,
    batch_rows: int = 250_000,
) -> npt.NDArray[np.int32]:
    """Load SW codes as a ``(date, stock)`` int32 panel; ``-1`` means missing."""

    if level not in (1, 2, 3):
        raise ValueError("level 必须是 1、2 或 3")
    if batch_rows <= 0:
        raise ValueError("batch_rows 必须为正整数")
    dates = np.asarray(date_list).astype("datetime64[D]")
    stocks = np.asarray(stock_list).astype(str)
    if dates.ndim != 1 or dates.size == 0 or np.unique(dates).size != dates.size:
        raise ValueError("date_list 必须是非空且不重复的一维日期数组")
    if stocks.ndim != 1 or stocks.size == 0 or np.unique(stocks).size != stocks.size:
        raise ValueError("stock_list 必须是非空且不重复的一维数组")
    if not np.all(dates[:-1] < dates[1:]):
        raise ValueError("date_list 必须严格升序")
    if not np.all(stocks[:-1] < stocks[1:]):
        raise ValueError("stock_list 必须严格升序")
    if not np.char.isnumeric(stocks).all() or any(len(value) != 6 for value in stocks):
        raise ValueError("stock_list 必须全部是六位数字字符串")

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"申万点时行业长表不存在：{resolved}")
    result = np.full((dates.size, stocks.size), -1, dtype=np.int32)
    connection = duckdb.connect()
    try:
        reader = connection.execute(
            f"""
            SELECT trade_date, stock_code, sw_code_{level} AS sw_code
            FROM read_parquet('{_sql_path(resolved)}')
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date, stock_code
            """,
            [pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date()],
        ).to_arrow_reader(batch_rows)
        for batch in reader:
            frame = batch.to_pandas()
            batch_dates = pd.to_datetime(frame["trade_date"]).to_numpy(dtype="datetime64[D]")
            batch_stocks = frame["stock_code"].astype(str).to_numpy()
            date_index = np.searchsorted(dates, batch_dates)
            stock_index = np.searchsorted(stocks, batch_stocks)
            valid_date = (date_index < dates.size) & (dates[np.minimum(date_index, dates.size - 1)] == batch_dates)
            valid_stock = (stock_index < stocks.size) & (
                stocks[np.minimum(stock_index, stocks.size - 1)] == batch_stocks
            )
            valid_code = frame["sw_code"].astype("string").str.fullmatch(r"\d{6}", na=False).to_numpy()
            valid = valid_date & valid_stock & valid_code
            if valid.any():
                result[date_index[valid], stock_index[valid]] = (
                    frame.loc[valid, "sw_code"].astype(str).to_numpy(dtype=np.int32)
                )
    finally:
        connection.close()
    return result


__all__ = [
    "DAILY_CLEAN_PATH",
    "INDUSTRY_SW_DAILY_COLUMNS",
    "INDUSTRY_SW_DAILY_METADATA_PATH",
    "INDUSTRY_SW_DAILY_PATH",
    "SWIND_INPUT_COLUMNS",
    "SWIND_SOURCE_DIR",
    "IndustryBuildConfig",
    "build_sw_industry_daily",
    "inspect_sw_industry_output",
    "inspect_sw_industry_source",
    "load_sw_industry_panel",
]
