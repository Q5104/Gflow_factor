"""日频因子表达式的叶子与算子签名注册表。

本模块只描述表达式文法，不包含任何数值计算实现。算子的实际计算逻辑属于
后续 ``factor_gfn.evaluator`` 阶段。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class OperatorCategory(StrEnum):
    """研报日频文法中的叶子和五类算子。"""

    LEAF = "LEAF"
    UNARY = "UNARY_OP"
    TS_UNARY = "TS_UNARY_OP"
    BINARY = "BINARY_OP"
    TS_BINARY = "TS_BINARY_OP"
    CROSS_SECTIONAL = "CS_OP"


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """一个叶子或算子的不可变文法签名。"""

    name: str
    category: OperatorCategory
    arity: int
    requires_window: bool
    commutative: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"非法算子名称：{self.name!r}")
        if self.arity not in (0, 1, 2):
            raise ValueError(f"{self.name} 的元数必须为 0、1 或 2")
        if self.category is OperatorCategory.LEAF:
            if self.arity != 0 or self.requires_window:
                raise ValueError(f"叶子 {self.name} 必须是无窗口的 0 元节点")
        elif self.arity == 0:
            raise ValueError(f"非叶子算子 {self.name} 的元数不能为 0")
        if self.commutative and self.arity != 2:
            raise ValueError(f"交换律标记只适用于二元算子：{self.name}")


def _specs(
    names: tuple[str, ...],
    category: OperatorCategory,
    arity: int,
    requires_window: bool,
    commutative_names: frozenset[str] = frozenset(),
) -> tuple[OperatorSpec, ...]:
    return tuple(
        OperatorSpec(
            name=name,
            category=category,
            arity=arity,
            requires_window=requires_window,
            commutative=name in commutative_names,
        )
        for name in names
    )


LEAVES = _specs(
    ("open", "high", "low", "close", "vwap", "volume"),
    OperatorCategory.LEAF,
    arity=0,
    requires_window=False,
)

UNARY_OPERATORS = _specs(
    (
        "abs",
        "neg",
        "sign",
        "log",
        "inv",
        "sqrt",
        "tanh",
        "relu",
        "softsign",
        "signed_power2",
        "signed_power3",
        "signed_log1p",
    ),
    OperatorCategory.UNARY,
    arity=1,
    requires_window=False,
)

TS_UNARY_OPERATORS = _specs(
    (
        "ts_mean",
        "ts_std",
        "ts_max",
        "ts_min",
        "ts_rank",
        "ts_delay",
        "ts_delta",
        "ts_sum",
        "ts_argmax",
        "ts_argmin",
        "ts_wma",
        "ts_ema",
        "ts_slope",
        "ts_residual",
        "ts_zscore",
        "ts_position",
        "ts_range",
    ),
    OperatorCategory.TS_UNARY,
    arity=1,
    requires_window=True,
)

BINARY_OPERATORS = _specs(
    (
        "add",
        "sub",
        "mul",
        "div",
        "max2",
        "min2",
        "greater",
        "less",
        "signed_ratio",
        "log_ratio",
    ),
    OperatorCategory.BINARY,
    arity=2,
    requires_window=False,
    commutative_names=frozenset({"add", "mul", "max2", "min2"}),
)

TS_BINARY_OPERATORS = _specs(
    ("ts_corr", "ts_cov", "ts_beta", "ts_orth"),
    OperatorCategory.TS_BINARY,
    arity=2,
    requires_window=True,
)

CROSS_SECTIONAL_OPERATORS = _specs(
    (
        "cs_rank",
        "cs_zscore",
        "cs_demean",
        "cs_scale",
        "cs_normalize",
        "cs_winsorize",
        "cs_truncate",
        "cs_quantile",
        "cs_rank_gauss",
    ),
    OperatorCategory.CROSS_SECTIONAL,
    arity=1,
    requires_window=False,
)

NON_LEAF_OPERATORS = (
    UNARY_OPERATORS
    + TS_UNARY_OPERATORS
    + BINARY_OPERATORS
    + TS_BINARY_OPERATORS
    + CROSS_SECTIONAL_OPERATORS
)
ALL_SYMBOLS = LEAVES + NON_LEAF_OPERATORS
SYMBOL_BY_NAME = MappingProxyType({symbol.name: symbol for symbol in ALL_SYMBOLS})


def get_operator(name: str) -> OperatorSpec:
    """按名称获取文法签名；未知名称会给出明确错误。"""

    try:
        return SYMBOL_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"未知叶子或算子：{name!r}") from exc


def _validate_registry() -> None:
    if len(LEAVES) != 6:
        raise RuntimeError(f"叶子数量应为 6，实际为 {len(LEAVES)}")
    if len(NON_LEAF_OPERATORS) != 52:
        raise RuntimeError(
            f"非叶子算子数量应为 52，实际为 {len(NON_LEAF_OPERATORS)}"
        )
    if len(SYMBOL_BY_NAME) != len(ALL_SYMBOLS):
        raise RuntimeError("叶子与算子名称必须全局唯一")


_validate_registry()


__all__ = [
    "ALL_SYMBOLS",
    "BINARY_OPERATORS",
    "CROSS_SECTIONAL_OPERATORS",
    "LEAVES",
    "NON_LEAF_OPERATORS",
    "OperatorCategory",
    "OperatorSpec",
    "SYMBOL_BY_NAME",
    "TS_BINARY_OPERATORS",
    "TS_UNARY_OPERATORS",
    "UNARY_OPERATORS",
    "get_operator",
]
