"""路径条件化的部分 AST Transformer 前向策略。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from factor_gfn.grammar import (
    CATEGORY_TO_INDEX,
    OPERATOR_TO_INDEX,
    OperatorCategory,
    SearchSpaceConfig,
    TOTAL_ACTIONS,
    WINDOWS,
    WINDOW_TO_INDEX,
    get_action,
    get_token_indices,
)

from .config import ModelConfig
from .state_adapter import MODEL_TOKEN_COUNT, ROLE_COUNT, StateBatch


TOKEN_GROUP_NAMES = ("leaf", "unary", "binary")
TOKEN_GROUP_COUNT = len(TOKEN_GROUP_NAMES)
GRAMMAR_CATEGORY_NAMES = (
    "feature",
    "unary",
    "ts_unary",
    "binary",
    "ts_binary",
    "cross_sectional",
)
GRAMMAR_CATEGORY_COUNT = len(GRAMMAR_CATEGORY_NAMES)
WINDOW_NAMES = tuple(str(window) for window in WINDOWS)
WINDOW_COUNT = len(WINDOW_NAMES)


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    slot_logits: Tensor
    token_logits: Tensor
    slot_log_probs: Tensor
    token_log_probs: Tensor
    slot_mask: Tensor
    legal_token_mask: Tensor
    group_logits: Tensor | None = None
    group_log_probs: Tensor | None = None
    legal_group_mask: Tensor | None = None
    grammar_category_log_probs: Tensor | None = None
    legal_grammar_category_mask: Tensor | None = None
    operator_log_probs: Tensor | None = None
    legal_operator_mask: Tensor | None = None
    operator_category_lookup: Tensor | None = None
    window_log_probs: Tensor | None = None
    legal_window_mask: Tensor | None = None


def _masked_log_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError("logits 与 mask 形状必须一致")
    if not bool(mask.any(dim=dim).all()):
        raise ValueError("每个有效分布至少需要一个合法动作")
    return torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=dim)


class ForwardPolicyNetwork(nn.Module):
    """输出 ``P(slot|state)`` 及 ``P(token|state,slot)``。"""

    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        search_space: SearchSpaceConfig = SearchSpaceConfig(),
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.search_space = search_space
        d_model = model_config.d_model
        max_sequence_nodes = 2 * search_space.max_nodes + 1

        category_special = len(CATEGORY_TO_INDEX)
        operator_special = len(OPERATOR_TO_INDEX)
        window_special = len(WINDOW_TO_INDEX)
        category_lookup: list[int] = []
        operator_lookup: list[int] = []
        window_lookup: list[int] = []
        for token_id in range(TOTAL_ACTIONS):
            category, operator, window = get_token_indices(token_id)
            category_lookup.append(category)
            operator_lookup.append(operator)
            window_lookup.append(window)
        category_lookup.extend((category_special, category_special + 1))
        operator_lookup.extend((operator_special, operator_special + 1))
        window_lookup.extend((window_special, window_special + 1))
        self.register_buffer(
            "token_category_lookup", torch.tensor(category_lookup, dtype=torch.long)
        )
        self.register_buffer(
            "token_operator_lookup", torch.tensor(operator_lookup, dtype=torch.long)
        )
        self.register_buffer(
            "token_window_lookup", torch.tensor(window_lookup, dtype=torch.long)
        )
        self.category_embedding = nn.Embedding(category_special + 2, d_model)
        self.operator_embedding = nn.Embedding(operator_special + 2, d_model)
        self.window_embedding = nn.Embedding(window_special + 2, d_model)
        self.depth_embedding = nn.Embedding(search_space.max_depth + 1, d_model)
        self.role_embedding = nn.Embedding(ROLE_COUNT, d_model)
        self.path_level_embedding = nn.Embedding(max(search_space.max_depth, 1), d_model)
        self.position_embedding = nn.Embedding(max_sequence_nodes, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=model_config.num_heads,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=model_config.num_layers,
            enable_nested_tensor=False,
        )
        self.state_aux_projection = nn.Linear(3, d_model)
        self.budget_projection = nn.Linear(3, d_model)
        self.slot_fusion = nn.Sequential(
            nn.Linear(7 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.slot_head = nn.Linear(d_model, 1)
        if model_config.token_policy_mode == "grammar_hierarchical":
            self.token_head: nn.Sequential | None = None
        else:
            self.token_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, TOTAL_ACTIONS),
            )
        token_groups = [get_action(token_id).arity for token_id in range(TOTAL_ACTIONS)]
        token_categories = [
            CATEGORY_TO_INDEX[get_action(token_id).category]
            for token_id in range(TOTAL_ACTIONS)
        ]
        token_operators = [
            OPERATOR_TO_INDEX[get_action(token_id).name]
            for token_id in range(TOTAL_ACTIONS)
        ]
        token_windows = [
            (WINDOWS.index(get_action(token_id).window) if get_action(token_id).window else -1)
            for token_id in range(TOTAL_ACTIONS)
        ]
        self.register_buffer(
            "token_group_lookup",
            torch.tensor(token_groups, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "action_category_lookup",
            torch.tensor(token_categories, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "action_operator_lookup",
            torch.tensor(token_operators, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "action_window_lookup",
            torch.tensor(token_windows, dtype=torch.long),
            persistent=False,
        )
        category_groups = [
            0 if category is OperatorCategory.LEAF else (
                2 if category in (OperatorCategory.BINARY, OperatorCategory.TS_BINARY) else 1
            )
            for category in OperatorCategory
        ]
        operator_categories = [0] * len(OPERATOR_TO_INDEX)
        operator_requires_window = [False] * len(OPERATOR_TO_INDEX)
        for action in (get_action(token_id) for token_id in range(TOTAL_ACTIONS)):
            operator_categories[OPERATOR_TO_INDEX[action.name]] = CATEGORY_TO_INDEX[action.category]
            operator_requires_window[OPERATOR_TO_INDEX[action.name]] = bool(action.window)
        self.register_buffer(
            "category_group_lookup",
            torch.tensor(category_groups, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "operator_category_lookup",
            torch.tensor(operator_categories, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "operator_requires_window",
            torch.tensor(operator_requires_window, dtype=torch.bool),
            persistent=False,
        )
        if model_config.token_policy_mode in ("arity_hierarchical", "grammar_hierarchical"):
            self.group_head: nn.Linear | None = nn.Linear(d_model, TOKEN_GROUP_COUNT)
            nn.init.zeros_(self.group_head.weight)
            nn.init.zeros_(self.group_head.bias)
        else:
            self.group_head = None
        if model_config.token_policy_mode == "grammar_hierarchical":
            self.grammar_category_head: nn.Linear | None = nn.Linear(
                d_model, GRAMMAR_CATEGORY_COUNT
            )
            self.operator_head: nn.Linear | None = nn.Linear(
                d_model, len(OPERATOR_TO_INDEX)
            )
            self.window_head: nn.Linear | None = nn.Linear(
                d_model, len(OPERATOR_TO_INDEX) * WINDOW_COUNT
            )
            for head in (
                self.grammar_category_head,
                self.operator_head,
                self.window_head,
            ):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
            self.grammar_category_head = None
            self.operator_head = None
            self.window_head = None

    def _hierarchical_token_distribution(
        self,
        token_logits: Tensor,
        legal_token_mask: Tensor,
        slot_mask: Tensor,
        slot_hidden: Tensor,
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.group_head is None:
            raise RuntimeError("分组 Token 策略缺少 group_head")
        group_logits = self.group_head(slot_hidden) / temperature
        group_membership = torch.nn.functional.one_hot(
            self.token_group_lookup,
            num_classes=TOKEN_GROUP_COUNT,
        ).to(dtype=torch.bool)
        legal_by_group = legal_token_mask.unsqueeze(-1) & group_membership
        legal_group_mask = legal_by_group.any(dim=2)

        safe_group_mask = legal_group_mask.clone()
        padding_slots = ~slot_mask
        safe_group_mask[..., 0] |= padding_slots
        group_log_probs = _masked_log_softmax(group_logits, safe_group_mask, dim=2)
        group_log_probs = group_log_probs.masked_fill(
            padding_slots.unsqueeze(-1), -torch.inf
        )

        within_group_log_probs = torch.full_like(token_logits, -torch.inf)
        for group_id in range(TOKEN_GROUP_COUNT):
            group_token_mask = legal_by_group[..., group_id].clone()
            missing_group = ~group_token_mask.any(dim=2)
            sentinel_token = int(
                torch.nonzero(
                    self.token_group_lookup == group_id,
                    as_tuple=False,
                )[0, 0]
            )
            group_token_mask[..., sentinel_token] |= missing_group
            group_distribution = _masked_log_softmax(
                token_logits,
                group_token_mask,
                dim=2,
            )
            actual_members = legal_by_group[..., group_id]
            within_group_log_probs = torch.where(
                actual_members,
                group_distribution,
                within_group_log_probs,
            )

        token_group_log_probs = group_log_probs.gather(
            2,
            self.token_group_lookup.view(1, 1, -1).expand_as(token_logits),
        )
        token_log_probs = token_group_log_probs + within_group_log_probs
        token_log_probs = token_log_probs.masked_fill(
            ~legal_token_mask | padding_slots.unsqueeze(-1),
            -torch.inf,
        )
        return group_logits, group_log_probs, legal_group_mask, token_log_probs

    def _grammar_hierarchical_token_distribution(
        self,
        legal_token_mask: Tensor,
        slot_mask: Tensor,
        slot_hidden: Tensor,
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if any(
            head is None
            for head in (
                self.group_head,
                self.grammar_category_head,
                self.operator_head,
                self.window_head,
            )
        ):
            raise RuntimeError("完整文法分层策略缺少概率头")
        padding_slots = ~slot_mask

        action_categories = torch.nn.functional.one_hot(
            self.action_category_lookup, num_classes=GRAMMAR_CATEGORY_COUNT
        ).to(dtype=torch.bool)
        action_operators = torch.nn.functional.one_hot(
            self.action_operator_lookup, num_classes=len(OPERATOR_TO_INDEX)
        ).to(dtype=torch.bool)
        legal_by_category = legal_token_mask.unsqueeze(-1) & action_categories
        legal_category_mask = legal_by_category.any(dim=2)
        legal_by_operator = legal_token_mask.unsqueeze(-1) & action_operators
        legal_operator_mask = legal_by_operator.any(dim=2)

        group_logits = self.group_head(slot_hidden) / temperature
        legal_group_mask = (
            legal_category_mask.unsqueeze(-1)
            & torch.nn.functional.one_hot(
                self.category_group_lookup, num_classes=TOKEN_GROUP_COUNT
            ).to(dtype=torch.bool)
        ).any(dim=2)
        safe_group_mask = legal_group_mask.clone()
        safe_group_mask[..., 0] |= padding_slots
        group_log_probs = _masked_log_softmax(group_logits, safe_group_mask, dim=2)
        group_log_probs = group_log_probs.masked_fill(padding_slots.unsqueeze(-1), -torch.inf)

        category_logits = self.grammar_category_head(slot_hidden) / temperature
        category_conditional = torch.full_like(category_logits, -torch.inf)
        for group_id in range(TOKEN_GROUP_COUNT):
            mask = legal_category_mask & (self.category_group_lookup == group_id)
            missing = ~mask.any(dim=2)
            sentinel = int(torch.nonzero(self.category_group_lookup == group_id)[0, 0])
            safe = mask.clone()
            safe[..., sentinel] |= missing
            distribution = _masked_log_softmax(category_logits, safe, dim=2)
            category_conditional = torch.where(mask, distribution, category_conditional)
        category_group_logs = group_log_probs.gather(
            2, self.category_group_lookup.view(1, 1, -1).expand_as(category_logits)
        )
        category_joint_log_probs = category_group_logs + category_conditional
        category_joint_log_probs = category_joint_log_probs.masked_fill(
            ~legal_category_mask | padding_slots.unsqueeze(-1), -torch.inf
        )

        operator_logits = self.operator_head(slot_hidden) / temperature
        operator_log_probs = torch.full_like(operator_logits, -torch.inf)
        for category_id in range(GRAMMAR_CATEGORY_COUNT):
            mask = legal_operator_mask & (self.operator_category_lookup == category_id)
            missing = ~mask.any(dim=2)
            sentinel = int(torch.nonzero(self.operator_category_lookup == category_id)[0, 0])
            safe = mask.clone()
            safe[..., sentinel] |= missing
            distribution = _masked_log_softmax(operator_logits, safe, dim=2)
            operator_log_probs = torch.where(mask, distribution, operator_log_probs)
        operator_log_probs = operator_log_probs.masked_fill(
            ~legal_operator_mask | padding_slots.unsqueeze(-1), -torch.inf
        )

        window_logits = self.window_head(slot_hidden).view(
            *slot_hidden.shape[:2], len(OPERATOR_TO_INDEX), WINDOW_COUNT
        ) / temperature
        # 当前固定文法中，同一个时序 Operator 的五个窗口具有相同结构合法性；
        # 用 Operator 合法掩码一次性展开，避免每次 forward 发起 142 次 GPU 小操作。
        legal_window_mask = (
            legal_operator_mask.unsqueeze(-1)
            & self.operator_requires_window.view(1, 1, -1, 1)
        ).expand(-1, -1, -1, WINDOW_COUNT)
        safe_window_mask = legal_window_mask.clone()
        missing_windows = ~safe_window_mask.any(dim=3)
        safe_window_mask[..., 0] |= missing_windows
        window_log_probs = _masked_log_softmax(window_logits, safe_window_mask, dim=3)
        window_log_probs = window_log_probs.masked_fill(~legal_window_mask, -torch.inf)

        category_for_action = category_joint_log_probs.gather(
            2, self.action_category_lookup.view(1, 1, -1).expand_as(legal_token_mask)
        )
        operator_for_action = operator_log_probs.gather(
            2, self.action_operator_lookup.view(1, 1, -1).expand_as(legal_token_mask)
        )
        window_indices = self.action_window_lookup.clamp_min(0)
        operator_indices = self.action_operator_lookup
        window_for_action = window_log_probs[
            ...,
            operator_indices,
            window_indices,
        ]
        has_window = self.action_window_lookup >= 0
        window_for_action = torch.where(
            has_window.view(1, 1, -1),
            window_for_action,
            torch.zeros_like(window_for_action),
        )
        token_log_probs = category_for_action + operator_for_action + window_for_action
        token_log_probs = token_log_probs.masked_fill(
            ~legal_token_mask | padding_slots.unsqueeze(-1), -torch.inf
        )
        return (
            group_logits,
            group_log_probs,
            legal_group_mask,
            category_joint_log_probs,
            legal_category_mask,
            operator_log_probs,
            legal_operator_mask,
            window_log_probs,
            legal_window_mask,
            token_log_probs,
        )

    def _token_embeddings(self, token_ids: Tensor) -> Tensor:
        """按研报口径叠加类别、算子/特征和窗口三组 embedding。"""

        if token_ids.min().item() < 0 or token_ids.max().item() >= MODEL_TOKEN_COUNT:
            raise ValueError("模型 Token ID 超出动作/Hole/PAD 词表")
        return (
            self.category_embedding(self.token_category_lookup[token_ids])
            + self.operator_embedding(self.token_operator_lookup[token_ids])
            + self.window_embedding(self.token_window_lookup[token_ids])
        )

    def _path_embeddings(self, batch: StateBatch) -> Tensor:
        batch_size, node_count, max_depth = batch.path_parent_token_ids.shape
        if max_depth == 0:
            return torch.zeros(
                (batch_size, node_count, self.model_config.d_model),
                device=batch.token_ids.device,
                dtype=self.category_embedding.weight.dtype,
            )
        levels = torch.arange(max_depth, device=batch.token_ids.device)
        levels = levels.view(1, 1, max_depth)
        path = (
            self._token_embeddings(batch.path_parent_token_ids)
            + self.role_embedding(batch.path_role_ids)
            + self.path_level_embedding(levels)
        )
        path = path * batch.path_mask.unsqueeze(-1)
        denominator = batch.path_mask.sum(dim=-1, keepdim=True).clamp_min(1).sqrt()
        return path.sum(dim=2) / denominator

    def forward(self, batch: StateBatch, *, temperature: float = 1.0) -> PolicyOutput:
        if temperature <= 0:
            raise ValueError("temperature 必须为正数")
        if batch.token_ids.ndim != 2 or batch.legal_token_mask.ndim != 3:
            raise ValueError("StateBatch 张量维度不正确")
        if any(
            state.search_space != self.search_space
            for state in batch.states
        ):
            raise ValueError("StateBatch 与模型的 SearchSpaceConfig 必须等价")
        if batch.depths.max().item() > self.search_space.max_depth:
            raise ValueError("输入状态深度超过模型配置")

        batch_size, sequence_length = batch.token_ids.shape
        positions = torch.arange(sequence_length, device=batch.token_ids.device)
        path_embeddings = self._path_embeddings(batch)
        node_inputs = (
            self._token_embeddings(batch.token_ids)
            + self.depth_embedding(batch.depths)
            + self.role_embedding(batch.role_ids)
            + self.position_embedding(positions).unsqueeze(0)
            + path_embeddings
        )
        encoded = self.encoder(
            self.input_norm(node_inputs),
            src_key_padding_mask=~batch.node_mask,
        )
        mask_float = batch.node_mask.unsqueeze(-1).to(encoded.dtype)
        global_state = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1)
        global_state = global_state + self.state_aux_projection(batch.auxiliary_features)

        slot_count = batch.slot_node_indices.shape[1]
        gather_indices = batch.slot_node_indices.unsqueeze(-1).expand(
            batch_size, slot_count, self.model_config.d_model
        )
        hole_context = encoded.gather(1, gather_indices)
        path_context = path_embeddings.gather(1, gather_indices)
        state_context = global_state.unsqueeze(1).expand(-1, slot_count, -1)
        parent_context = self._token_embeddings(batch.slot_parent_token_ids)
        role_context = self.role_embedding(batch.slot_role_ids)
        depth_context = self.depth_embedding(batch.slot_depths)
        budget_context = self.budget_projection(batch.slot_budget_features)
        slot_hidden = self.slot_fusion(
            torch.cat(
                (
                    state_context,
                    hole_context,
                    path_context,
                    parent_context,
                    role_context,
                    depth_context,
                    budget_context,
                ),
                dim=-1,
            )
        )

        slot_logits = self.slot_head(slot_hidden).squeeze(-1) / temperature
        token_logits = (
            slot_hidden.new_zeros((*slot_hidden.shape[:2], TOTAL_ACTIONS))
            if self.token_head is None
            else self.token_head(slot_hidden) / temperature
        )
        slot_log_probs = _masked_log_softmax(slot_logits, batch.slot_mask, dim=1)

        # Padding 槽位没有合法 Token。先为其放置一个仅用于数值计算的哨兵动作，
        # 计算后再整体设回 -inf；真实槽位仍严格使用联合规范边 mask。
        safe_token_mask = batch.legal_token_mask.clone()
        padding_slots = ~batch.slot_mask
        safe_token_mask[..., 0] |= padding_slots
        if self.model_config.token_policy_mode == "grammar_hierarchical":
            (
                group_logits,
                group_log_probs,
                legal_group_mask,
                grammar_category_log_probs,
                legal_grammar_category_mask,
                operator_log_probs,
                legal_operator_mask,
                window_log_probs,
                legal_window_mask,
                token_log_probs,
            ) = self._grammar_hierarchical_token_distribution(
                batch.legal_token_mask,
                batch.slot_mask,
                slot_hidden,
                temperature,
            )
            # grammar_hierarchical 没有冗余的142维独立输出头；以联合 log
            # 概率作为可审计的最终动作 logits，避免每次 forward 做无效矩阵乘法。
            token_logits = token_log_probs
        elif self.model_config.token_policy_mode == "arity_hierarchical":
            (
                group_logits,
                group_log_probs,
                legal_group_mask,
                token_log_probs,
            ) = self._hierarchical_token_distribution(
                token_logits,
                batch.legal_token_mask,
                batch.slot_mask,
                slot_hidden,
                temperature,
            )
        else:
            token_log_probs = _masked_log_softmax(token_logits, safe_token_mask, dim=2)
            token_log_probs = token_log_probs.masked_fill(
                padding_slots.unsqueeze(-1), -torch.inf
            )
            group_logits = None
            group_log_probs = None
            legal_group_mask = None
            grammar_category_log_probs = None
            legal_grammar_category_mask = None
            operator_log_probs = None
            legal_operator_mask = None
            window_log_probs = None
            legal_window_mask = None
        if self.model_config.token_policy_mode == "arity_hierarchical":
            grammar_category_log_probs = None
            legal_grammar_category_mask = None
            operator_log_probs = None
            legal_operator_mask = None
            window_log_probs = None
            legal_window_mask = None
        masked_slot_logits = slot_logits.masked_fill(~batch.slot_mask, -torch.inf)
        masked_token_logits = token_logits.masked_fill(~batch.legal_token_mask, -torch.inf)

        return PolicyOutput(
            slot_logits=masked_slot_logits,
            token_logits=masked_token_logits,
            slot_log_probs=slot_log_probs,
            token_log_probs=token_log_probs,
            slot_mask=batch.slot_mask,
            legal_token_mask=batch.legal_token_mask,
            group_logits=(
                group_logits.masked_fill(~legal_group_mask, -torch.inf)
                if group_logits is not None and legal_group_mask is not None
                else None
            ),
            group_log_probs=group_log_probs,
            legal_group_mask=legal_group_mask,
            grammar_category_log_probs=grammar_category_log_probs,
            legal_grammar_category_mask=legal_grammar_category_mask,
            operator_log_probs=operator_log_probs,
            legal_operator_mask=legal_operator_mask,
            operator_category_lookup=(
                self.operator_category_lookup
                if operator_log_probs is not None else None
            ),
            window_log_probs=window_log_probs,
            legal_window_mask=legal_window_mask,
        )


__all__ = [
    "ForwardPolicyNetwork",
    "PolicyOutput",
    "TOKEN_GROUP_COUNT",
    "TOKEN_GROUP_NAMES",
    "GRAMMAR_CATEGORY_COUNT",
    "GRAMMAR_CATEGORY_NAMES",
    "WINDOW_COUNT",
    "WINDOW_NAMES",
]
