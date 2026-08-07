from __future__ import annotations

import numpy as np

from factor_gfn.barra import (
    DEFAULT_BARRA_CONFIG,
    BarraConfig,
    build_barra_long_short_returns,
    calculate_barra_factors,
    calculate_barra_ts_corr,
)
from factor_gfn.barra.factors import market_cap_weighted_return
from factor_gfn.barra.factors import rolling_beta_strict, rolling_std_strict


def test_market_return_uses_lagged_market_cap() -> None:
    returns = np.array([[np.nan, np.nan], [0.10, 0.20], [0.30, 0.40]])
    caps = np.array([[1.0, 3.0], [9.0, 1.0], [1_000.0, 1.0]])
    universe = np.ones_like(caps, dtype=bool)
    market = market_cap_weighted_return(returns, caps, universe)

    assert np.isclose(market[1], 0.10 * 0.25 + 0.20 * 0.75)
    assert np.isclose(market[2], 0.30 * 0.90 + 0.40 * 0.10)


def test_five_styles_stay_separate_and_penalty_is_max_abs_corr() -> None:
    rng = np.random.default_rng(17)
    date_count, stock_count = 30, 12
    daily_returns = rng.normal(0.0005, 0.015, size=(date_count, stock_count))
    close = 100.0 * np.cumprod(1.0 + daily_returns, axis=0)
    volume = rng.integers(1_000, 10_000, size=close.shape).astype(float)
    shares = np.broadcast_to(np.linspace(1e6, 2e6, stock_count), close.shape).copy()
    float_cap = close / 3.0 * shares
    total_cap = float_cap * 1.4
    universe = np.ones(close.shape, dtype=bool)
    config = BarraConfig(
        beta_window=5,
        beta_min_periods=3,
        momentum_lookback=6,
        momentum_skip=1,
        volatility_window=5,
        volatility_min_periods=3,
        liquidity_window=3,
        min_cross_section_count=4,
        stock_chunk_size=4,
    )
    factors = calculate_barra_factors(
        close, volume, float_cap, total_cap, shares, universe, config
    )
    forward = rng.normal(0.0, 0.03, size=close.shape)
    indices = np.arange(6, date_count, 5)
    series = build_barra_long_short_returns(factors, forward, indices, config)

    assert set(series) == {
        "market_beta",
        "size",
        "momentum",
        "volatility",
        "liquidity",
    }
    assert len({id(item.long_short_return) for item in series.values()}) == 5
    candidate = series["momentum"].long_short_return.copy()
    penalty = calculate_barra_ts_corr(candidate, series, min_periods=2)
    assert np.isclose(penalty.correlations["momentum"], 1.0)
    assert np.isclose(penalty.barra_ts_corr, 1.0)

    # The production default requires 60 common rebalance periods. This small
    # synthetic sample remains unscored unless the test lowers the threshold.
    default_penalty = calculate_barra_ts_corr(candidate, series)
    assert DEFAULT_BARRA_CONFIG.min_common_periods == 60
    assert np.isnan(default_penalty.barra_ts_corr)
    assert all(np.isnan(value) for value in default_penalty.correlations.values())


def test_beta_and_volatility_use_min_periods_inside_rolling_window() -> None:
    market = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    stocks = (2.0 * market)[:, None]
    beta = rolling_beta_strict(stocks, market, window=5, min_periods=3)
    assert np.isnan(beta[1, 0])
    assert np.isclose(beta[3, 0], 2.0)

    values = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
    volatility = rolling_std_strict(values, window=5, min_periods=3)
    assert np.isnan(volatility[1, 0])
    assert np.isclose(volatility[3, 0], np.std([1.0, 2.0, 4.0], ddof=0))
