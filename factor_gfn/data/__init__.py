"""数据预处理与股票池 mask。"""

from .industry import load_sw_level1_industries

from .masks import (
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    apply_feature_valid_mask,
    build_feature_validity,
    build_universe_eligibility,
    combine_masks,
    is_current_st_name,
)
from .preprocess import (
    PROCESSED_DATA_DIR,
    PreprocessConfig,
    build_daily_clean,
    build_processed_arrays,
    inspect_inputs,
    run_preprocess,
)

__all__ = [
    "FEATURE_COLUMNS",
    "KEY_COLUMNS",
    "PROCESSED_DATA_DIR",
    "PreprocessConfig",
    "apply_feature_valid_mask",
    "build_feature_validity",
    "build_universe_eligibility",
    "build_daily_clean",
    "build_processed_arrays",
    "combine_masks",
    "is_current_st_name",
    "inspect_inputs",
    "load_sw_level1_industries",
    "run_preprocess",
]
