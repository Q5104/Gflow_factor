from __future__ import annotations

from pathlib import Path

import pandas as pd

from factor_gfn.data import downloader


def _share_response(code: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": [code, code],
            "change_date": ["2020-01-02", "2021-03-04"],
            "total_shares": [1_000_000, 1_200_000],
            "limit_shares": [400_000, 300_000],
            "list_a_shares": [600_000, 900_000],
            "change_reason": ["首次记录", "股份变动"],
        }
    )


def test_prepare_stock_shares_response_normalizes_and_sorts() -> None:
    raw = _share_response("1").iloc[::-1].reset_index(drop=True)
    result = downloader._prepare_stock_shares_response(raw, "1")

    assert result.columns.tolist() == downloader.STOCK_SHARES_COLUMNS
    assert result["stock_code"].tolist() == ["000001", "000001"]
    assert result["change_date"].is_monotonic_increasing
    assert (result["total_shares"] >= result["list_a_shares"]).all()


def test_prepare_stock_shares_allows_missing_limit_and_filters_bad_core_row() -> None:
    raw = pd.DataFrame(
        {
            "stock_code": ["000001", "000001"],
            "change_date": ["2000-01-01", "2001-01-01"],
            "total_shares": [1_000_000, 100],
            "limit_shares": [None, None],
            "list_a_shares": [600_000, 200],
            "change_reason": ["早期记录", None],
        }
    )
    result = downloader._prepare_stock_shares_response(raw, "000001")

    assert len(result) == 1
    assert pd.isna(result.loc[0, "limit_shares"])
    assert result.attrs["dropped_invalid_rows"] == 1


def test_summarize_stock_shares_reports_optional_limit_null_rate(tmp_path: Path) -> None:
    path = tmp_path / "stock_shares_history.parquet"
    frame = _share_response("000001")
    frame.loc[0, "limit_shares"] = None
    frame.to_parquet(path, index=False)

    summary = downloader._summarize_stock_shares(path)

    assert summary["rows"] == 2
    assert summary["nulls"]["limit_shares"] == 1
    assert summary["limit_shares_null_rate"] == 0.5


def test_stock_shares_download_resumes_from_successful_parts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    parts_dir = tmp_path / "download_parts"
    output_path = raw_dir / "stock_shares_history.parquet"
    stocks = pd.DataFrame({"stock_code": ["000001", "000002"]})
    calls: list[str] = []

    monkeypatch.setattr(downloader, "DOWNLOAD_PARTS_DIR", parts_dir)
    monkeypatch.setattr(downloader, "STOCK_SHARES_PATH", output_path)
    monkeypatch.setattr(
        downloader,
        "download_stock_list",
        lambda force_update=False: stocks.copy(),
    )

    def fake_get_stock_shares(stock_code: str, is_history: bool) -> pd.DataFrame:
        assert is_history is True
        calls.append(stock_code)
        return _share_response(stock_code)

    monkeypatch.setattr(
        downloader.adata.stock.info,
        "get_stock_shares",
        fake_get_stock_shares,
    )

    first = downloader.download_stock_shares(force_update=False)
    assert first["stock_count"] == 2
    assert first["missing_codes"] == []
    assert calls == ["000001", "000002"]
    assert list((parts_dir / "stock_shares_history").glob("part_*.parquet"))

    calls.clear()
    second = downloader.download_stock_shares(force_update=False)
    assert second["stock_count"] == 2
    assert second["missing_codes"] == []
    assert calls == []
