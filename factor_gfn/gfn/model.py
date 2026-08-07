"""路径条件化的部分 AST Transformer 前向策略。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from factor_gfn.grammar import (
    CATEGORY_TO_INDEX,
    OPERATOR_TO_INDEX,
    SearchSpaceConfig,
    TOTAL_ACTIONS,
    WINDOW_TO_INDEX,
    get_token_indices,
)

from .config import ModelConfig
from .state_adapter import MODEL_TOKEN_COUNT, ROLE_COUNT, StateBatch


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    slot_logits: Tensor
    token_logits: Tensor
    slot_log_probs: Tensor
    token_log_probs: Tensor
    slot_mask: Tensor
    legal_token_mask: Tensor


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
        self.token_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, TOTAL_ACTIONS),
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
        token_logits = self.token_head(slot_hidden) / temperature
        slot_log_probs = _masked_log_softmax(slot_logits, batch.slot_mask, dim=1)

        # Padding 槽位没有合法 Token。先为其放置一个仅用于数值计算的哨兵动作，
        # 计算后再整体设回 -inf；真实槽位仍严格使用联合规范边 mask。
        safe_token_mask = batch.legal_token_mask.clone()
        padding_slots = ~batch.slot_mask
        safe_token_mask[..., 0] |= padding_slots
        token_log_probs = _masked_log_softmax(token_logits, safe_token_mask, dim=2)
        token_log_probs = token_log_probs.masked_fill(padding_slots.unsqueeze(-1), -torch.inf)
        masked_slot_logits = slot_logits.masked_fill(~batch.slot_mask, -torch.inf)
        masked_token_logits = token_logits.masked_fill(~batch.legal_token_mask, -torch.inf)

        return PolicyOutput(
            slot_logits=masked_slot_logits,
            token_logits=masked_token_logits,
            slot_log_probs=slot_log_probs,
            token_log_probs=token_log_probs,
            slot_mask=batch.slot_mask,
            legal_token_mask=batch.legal_token_mask,
        )


__all__ = ["ForwardPolicyNetwork", "PolicyOutput"]
