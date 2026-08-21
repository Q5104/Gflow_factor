"""Frozen identities for the two supported daily expression feature spaces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


FEATURE_SPACE_SCHEMA = "factor_gfn.feature_space.v1"
RAW_DAILY_FEATURE_NAMES = ("open", "high", "low", "close", "vwap", "volume")
DAILY_DERIVED_FEATURE_NAMES = (
    "ret_gap",
    "ret_cc1",
    "ret_co",
    "ret_hl",
    "ret_range",
    "ret_body",
    "ret_upper_shadow",
    "ret_lower_shadow",
    "ret_close_vwap",
    "ret_open_vwap",
    "ret_vol_chg1",
    "ret_vol_chg5",
    "turnover",
    "illiq",
    "ret_amt_chg5",
    "clv",
)


@dataclass(frozen=True, slots=True)
class FeatureSpaceSpec:
    feature_space_id: str
    ordered_leaf_names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(self.ordered_leaf_names)
        if not self.feature_space_id:
            raise ValueError("feature_space_id 不能为空")
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("ordered_leaf_names 必须是非空有序名称")
        if len(set(names)) != len(names):
            raise ValueError("ordered_leaf_names 不允许重复")
        object.__setattr__(self, "ordered_leaf_names", names)

    def manifest(self) -> dict[str, object]:
        return {
            "schema": FEATURE_SPACE_SCHEMA,
            "feature_space_id": self.feature_space_id,
            "ordered_leaf_names": list(self.ordered_leaf_names),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.manifest(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


RAW_DAILY_FEATURE_SPACE = FeatureSpaceSpec(
    feature_space_id="raw_daily",
    ordered_leaf_names=RAW_DAILY_FEATURE_NAMES,
)
DAILY_DERIVED_V1_FEATURE_SPACE = FeatureSpaceSpec(
    feature_space_id="daily_derived_v1",
    ordered_leaf_names=DAILY_DERIVED_FEATURE_NAMES,
)


__all__ = [
    "DAILY_DERIVED_FEATURE_NAMES",
    "DAILY_DERIVED_V1_FEATURE_SPACE",
    "FEATURE_SPACE_SCHEMA",
    "FeatureSpaceSpec",
    "RAW_DAILY_FEATURE_NAMES",
    "RAW_DAILY_FEATURE_SPACE",
]
