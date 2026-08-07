"""不可变表达式树、序列转换、结构统计与保守规范化。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from .operators import get_operator
from .tokens import ActionSpec, get_action


HASH_SCHEMA = "factor_gfn.expression.v1"


class ExpressionParseError(ValueError):
    """表达式 Token 序列无法构成唯一完整语法树。"""


def _normalize_token_ids(token_ids: Iterable[int], label: str) -> tuple[int, ...]:
    try:
        raw_tokens = tuple(token_ids)
    except TypeError as exc:
        raise TypeError(f"{label}必须是动作 ID 的可迭代对象") from exc
    if not raw_tokens:
        raise ExpressionParseError(f"{label}不能为空")

    normalized: list[int] = []
    for position, token_id in enumerate(raw_tokens):
        try:
            action = get_action(token_id)
        except (TypeError, IndexError) as exc:
            raise ExpressionParseError(
                f"{label}第 {position} 个动作 ID 非法：{token_id!r}"
            ) from exc
        normalized.append(int(token_id))
        # 调用保留在这里，确保注册表异常不会被静默忽略。
        if action.arity not in (0, 1, 2):
            raise RuntimeError(f"注册表包含非法元数：{action}")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ExpressionNode:
    """一个动作及其按语义顺序排列的不可变子节点。"""

    action_id: int
    children: tuple[ExpressionNode, ...] = ()

    def __post_init__(self) -> None:
        try:
            action = get_action(self.action_id)
        except (TypeError, IndexError) as exc:
            raise ValueError(f"表达式节点包含非法动作 ID：{self.action_id!r}") from exc
        children = tuple(self.children)
        if any(not isinstance(child, ExpressionNode) for child in children):
            raise TypeError("表达式节点的 children 必须全部是 ExpressionNode")
        if len(children) != action.arity:
            raise ValueError(
                f"{action.name} 需要 {action.arity} 个子节点，实际为 {len(children)}"
            )
        object.__setattr__(self, "action_id", int(self.action_id))
        object.__setattr__(self, "children", children)

    @property
    def action(self) -> ActionSpec:
        return get_action(self.action_id)


@dataclass(frozen=True, slots=True)
class ExpressionStats:
    """一棵表达式树的结构统计；深度按根节点为 0 计算。"""

    node_count: int
    operator_count: int
    leaf_count: int
    depth: int
    operator_ratio: float


@dataclass(slots=True)
class _PendingPrefixNode:
    action_id: int
    children: list[ExpressionNode]


@dataclass(frozen=True, slots=True)
class Expression:
    """一棵完整、合法且不可变的因子表达式树。"""

    root: ExpressionNode

    def __post_init__(self) -> None:
        if not isinstance(self.root, ExpressionNode):
            raise TypeError("root 必须是 ExpressionNode")

    @classmethod
    def from_prefix(cls, token_ids: Iterable[int]) -> Expression:
        """严格解析前序动作序列，拒绝缺失子节点和多余根节点。"""

        tokens = _normalize_token_ids(token_ids, "前序序列")
        pending: list[_PendingPrefixNode] = []
        root: ExpressionNode | None = None

        for position, action_id in enumerate(tokens):
            if root is not None and not pending:
                raise ExpressionParseError(
                    f"前序序列在位置 {position} 存在多余动作，完整表达式已提前结束"
                )

            action = get_action(action_id)
            if action.arity > 0:
                pending.append(_PendingPrefixNode(action_id=action_id, children=[]))
                continue

            completed = ExpressionNode(action_id)
            while True:
                if not pending:
                    root = completed
                    break

                parent = pending[-1]
                parent.children.append(completed)
                parent_arity = get_action(parent.action_id).arity
                if len(parent.children) < parent_arity:
                    break

                pending.pop()
                completed = ExpressionNode(parent.action_id, tuple(parent.children))

        if pending:
            parent = pending[-1]
            action = get_action(parent.action_id)
            missing = action.arity - len(parent.children)
            raise ExpressionParseError(
                f"前序序列不完整：{action.name} 仍缺少 {missing} 个子节点"
            )
        if root is None:
            raise ExpressionParseError("前序序列未生成完整表达式")
        return cls(root)

    @classmethod
    def from_postfix(cls, token_ids: Iterable[int]) -> Expression:
        """严格解析后序动作序列，并保持二元子节点的左右顺序。"""

        tokens = _normalize_token_ids(token_ids, "后序序列")
        stack: list[ExpressionNode] = []
        for position, action_id in enumerate(tokens):
            action = get_action(action_id)
            if len(stack) < action.arity:
                raise ExpressionParseError(
                    f"后序序列在位置 {position} 的 {action.name} 缺少操作数："
                    f"需要 {action.arity}，当前只有 {len(stack)}"
                )
            if action.arity == 0:
                children: tuple[ExpressionNode, ...] = ()
            else:
                children = tuple(stack[-action.arity :])
                del stack[-action.arity :]
            stack.append(ExpressionNode(action_id, children))

        if len(stack) != 1:
            raise ExpressionParseError(
                f"后序序列包含 {len(stack)} 个独立结果，必须恰好为 1 个"
            )
        return cls(stack[0])

    def to_prefix(self) -> tuple[int, ...]:
        """输出动作 ID 前序序列。"""

        output: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            output.append(node.action_id)
            stack.extend(reversed(node.children))
        return tuple(output)

    def to_postfix(self) -> tuple[int, ...]:
        """输出供第三阶段栈式解释器使用的动作 ID 后序序列。"""

        output: list[int] = []
        stack: list[tuple[ExpressionNode, bool]] = [(self.root, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                output.append(node.action_id)
                continue
            stack.append((node, True))
            for child in reversed(node.children):
                stack.append((child, False))
        return tuple(output)

    def to_formula(self) -> str:
        """输出无歧义、便于检查的函数式公式字符串。"""

        def render(node: ExpressionNode) -> str:
            action = node.action
            if action.arity == 0:
                return action.name
            arguments = [render(child) for child in node.children]
            if action.window != 0:
                arguments.append(str(action.window))
            return f"{action.name}({', '.join(arguments)})"

        return render(self.root)

    def __str__(self) -> str:
        return self.to_formula()

    @property
    def stats(self) -> ExpressionStats:
        """计算节点数、深度及非叶子算子占全部节点的比例。"""

        node_count = 0
        operator_count = 0
        max_depth = 0
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            node_count += 1
            operator_count += int(node.action.arity > 0)
            max_depth = max(max_depth, depth)
            stack.extend((child, depth + 1) for child in node.children)
        leaf_count = node_count - operator_count
        return ExpressionStats(
            node_count=node_count,
            operator_count=operator_count,
            leaf_count=leaf_count,
            depth=max_depth,
            operator_ratio=operator_count / node_count,
        )

    def canonicalize(self) -> Expression:
        """仅按已登记交换律对子树排序，不执行任何代数化简。"""

        def normalize(node: ExpressionNode) -> ExpressionNode:
            children = tuple(normalize(child) for child in node.children)
            operator = get_operator(node.action.name)
            if operator.commutative:
                children = tuple(sorted(children, key=_node_canonical_json))
            return ExpressionNode(node.action_id, children)

        return Expression(normalize(self.root))

    def canonical_key(self) -> str:
        """返回不依赖 action ID 的规范结构 JSON，可直接用于精确去重。"""

        return _node_canonical_json(self.canonicalize().root)

    def structural_hash(self) -> str:
        """返回跨进程稳定的 SHA-256 规范结构哈希。"""

        payload = f"{HASH_SCHEMA}\0{self.canonical_key()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _node_payload(node: ExpressionNode) -> list[object]:
    action = node.action
    return [
        action.name,
        action.window,
        [_node_payload(child) for child in node.children],
    ]


def _node_canonical_json(node: ExpressionNode) -> str:
    return json.dumps(
        _node_payload(node),
        ensure_ascii=True,
        separators=(",", ":"),
    )


def validate_postfix(token_ids: Iterable[int]) -> None:
    """验证后序序列能否恰好构造一个完整结果；成功时不返回内容。"""

    Expression.from_postfix(token_ids)


__all__ = [
    "HASH_SCHEMA",
    "Expression",
    "ExpressionNode",
    "ExpressionParseError",
    "ExpressionStats",
    "validate_postfix",
]
