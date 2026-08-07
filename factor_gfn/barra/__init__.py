"""Five-style Barra proxy factors used by the GFlowNet reward penalty."""

from .config import BarraConfig, DEFAULT_BARRA_CONFIG
from .factors import BarraFactorSet, STYLE_NAMES, calculate_barra_factors
from .pipeline import (
    DEFAULT_BARRA_PATHS,
    BarraInputs,
    BarraPaths,
    build_barra_auxiliary_arrays,
    load_barra_factor_set,
    load_barra_inputs,
    run_barra_factor_pipeline,
)
from .portfolio import (
    BarraPenaltyResult,
    LongShortSeries,
    LongShortSummary,
    build_barra_long_short_returns,
    calculate_barra_ts_corr,
    cumulative_return,
    equal_weight_long_short,
    summarize_long_short,
)

__all__ = [
    "DEFAULT_BARRA_CONFIG",
    "DEFAULT_BARRA_PATHS",
    "STYLE_NAMES",
    "BarraConfig",
    "BarraFactorSet",
    "BarraInputs",
    "BarraPaths",
    "BarraPenaltyResult",
    "LongShortSeries",
    "LongShortSummary",
    "build_barra_auxiliary_arrays",
    "build_barra_long_short_returns",
    "calculate_barra_factors",
    "calculate_barra_ts_corr",
    "cumulative_return",
    "equal_weight_long_short",
    "load_barra_factor_set",
    "load_barra_inputs",
    "run_barra_factor_pipeline",
    "summarize_long_short",
]
