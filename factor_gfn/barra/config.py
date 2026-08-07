"""Configuration for the first reproducible five-style Barra proxy set."""

from __future__ import annotations

from dataclasses import dataclass

from factor_gfn.evaluator.cross_section import (
    DEFAULT_CLEANING_CONFIG,
    CrossSectionalCleaningConfig,
)


@dataclass(frozen=True, slots=True)
class BarraConfig:
    """Engineering defaults; these are not claimed as the report's internals."""

    beta_window: int = 252
    beta_min_periods: int = 120
    momentum_lookback: int = 252
    momentum_skip: int = 21
    volatility_window: int = 252
    volatility_min_periods: int = 120
    liquidity_window: int = 60
    market_cap_type: str = "float"
    long_short_quantile: float = 0.10
    min_cross_section_count: int = 20
    min_common_periods: int = 60
    stock_chunk_size: int = 256
    performance_ddof: int = 1
    cleaning: CrossSectionalCleaningConfig = DEFAULT_CLEANING_CONFIG

    def __post_init__(self) -> None:
        for name in ("beta_window", "momentum_lookback", "volatility_window", "liquidity_window"):
            if getattr(self, name) <= 1:
                raise ValueError(f"{name} 必须大于 1")
        if not 0 <= self.momentum_skip < self.momentum_lookback:
            raise ValueError("momentum_skip 必须位于 [0, momentum_lookback) 内")
        if not 2 <= self.beta_min_periods <= self.beta_window:
            raise ValueError("beta_min_periods 必须位于 [2, beta_window] 内")
        if not 2 <= self.volatility_min_periods <= self.volatility_window:
            raise ValueError("volatility_min_periods 必须位于 [2, volatility_window] 内")
        if self.market_cap_type not in {"float", "total"}:
            raise ValueError("market_cap_type 只能为 'float' 或 'total'")
        if not 0.0 < self.long_short_quantile < 0.5:
            raise ValueError("long_short_quantile 必须位于 (0, 0.5) 内")
        if self.min_cross_section_count < 2:
            raise ValueError("min_cross_section_count 至少为 2")
        if self.min_common_periods < 2:
            raise ValueError("min_common_periods 至少为 2")
        if self.stock_chunk_size <= 0:
            raise ValueError("stock_chunk_size 必须为正整数")
        if self.performance_ddof < 0:
            raise ValueError("performance_ddof 不能为负数")


DEFAULT_BARRA_CONFIG = BarraConfig()


__all__ = ["BarraConfig", "DEFAULT_BARRA_CONFIG"]
