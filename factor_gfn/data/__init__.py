"""数据下载、预处理、行业对齐与股票池 mask。"""

from .daily_derived import (
    CLV_TOL,
    DAILY_DERIVED_FEATURE_NAMES,
    ILLIQ_SCALE,
    build_daily_derived_features,
)
from .daily_derived_artifact import (
    DailyDerivedArtifactConfig,
    build_daily_derived_artifact,
    daily_derived_schema_contract,
    daily_derived_schema_fingerprint,
    inspect_daily_derived_artifact,
    inspect_daily_derived_inputs,
)

from .industry import (
    INDUSTRY_SW_DAILY_PATH,
    SWIND_SOURCE_DIR,
    IndustryBuildConfig,
    build_sw_industry_daily,
    inspect_sw_industry_output,
    inspect_sw_industry_source,
    load_sw_industry_panel,
)

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
    "CLV_TOL",
    "DAILY_DERIVED_FEATURE_NAMES",
    "DailyDerivedArtifactConfig",
    "FEATURE_COLUMNS",
    "ILLIQ_SCALE",
    "KEY_COLUMNS",
    "INDUSTRY_SW_DAILY_PATH",
    "PROCESSED_DATA_DIR",
    "SWIND_SOURCE_DIR",
    "IndustryBuildConfig",
    "PreprocessConfig",
    "apply_feature_valid_mask",
    "build_feature_validity",
    "build_universe_eligibility",
    "build_daily_clean",
    "build_daily_derived_features",
    "build_daily_derived_artifact",
    "build_processed_arrays",
    "build_sw_industry_daily",
    "combine_masks",
    "daily_derived_schema_contract",
    "daily_derived_schema_fingerprint",
    "is_current_st_name",
    "inspect_inputs",
    "inspect_daily_derived_artifact",
    "inspect_daily_derived_inputs",
    "inspect_sw_industry_output",
    "inspect_sw_industry_source",
    "load_sw_industry_panel",
    "run_preprocess",
]
