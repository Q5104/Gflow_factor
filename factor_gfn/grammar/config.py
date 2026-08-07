"""阶段二文法与阶段四策略共同使用的搜索空间配置。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Integral


DEFAULT_MAX_DEPTH = 10
DEFAULT_MAX_NODES = 30
SEARCH_SPACE_CONFIG_SCHEMA = "factor_gfn.search_space_config.v1"


def _integer_limit(value: int, name: str, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} 必须是整数")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} 必须大于或等于 {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class SearchSpaceConfig:
    """规范部分 AST 的运行时结构约束；跨阶段唯一真值源。"""

    max_depth: int = DEFAULT_MAX_DEPTH
    max_nodes: int = DEFAULT_MAX_NODES

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_depth", _integer_limit(self.max_depth, "max_depth", 0)
        )
        object.__setattr__(
            self, "max_nodes", _integer_limit(self.max_nodes, "max_nodes", 1)
        )

    def manifest(self) -> dict[str, object]:
        return {"schema": SEARCH_SPACE_CONFIG_SCHEMA, **asdict(self)}

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.manifest(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_SEARCH_SPACE = SearchSpaceConfig()


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NODES",
    "DEFAULT_SEARCH_SPACE",
    "SEARCH_SPACE_CONFIG_SCHEMA",
    "SearchSpaceConfig",
]
