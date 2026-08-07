"""旧下载入口的兼容层。

下载实现已迁入 ``factor_gfn.data.downloader``。当前文件暂时保留，避免已经打开的
Notebook 或旧命令在本轮下载结束前失效；新代码应直接从包内模块导入。
"""

from factor_gfn.data.downloader import (
    DEFAULT_START_DATE,
    INDUSTRY_SW_PATH,
    LISTING_DATES_PATH,
    MARKET_DATA_PATH,
    RAW_CLOSE_PATH,
    STOCK_SHARES_PATH,
    download_adjusted_market,
    download_industry_sw,
    download_raw_close,
    download_stock_shares,
    download_stock_list,
    print_download_summary,
)

__all__ = [
    "DEFAULT_START_DATE",
    "INDUSTRY_SW_PATH",
    "LISTING_DATES_PATH",
    "MARKET_DATA_PATH",
    "RAW_CLOSE_PATH",
    "STOCK_SHARES_PATH",
    "download_stock_list",
    "download_adjusted_market",
    "download_industry_sw",
    "download_raw_close",
    "download_stock_shares",
    "print_download_summary",
]
