"""阶段四 GFlowNet 的集中配置和运行统计数据合同。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

from factor_gfn.grammar import (
    SearchSpaceConfig,
    action_space_fingerprint,
    state_space_fingerprint,
    transition_space_fingerprint,
)


CONFIG_SCHEMA = "factor_gfn.gfn_config.v1"
STATE_ADAPTER_SCHEMA = "factor_gfn.state_adapter.v1"


def state_adapter_manifest() -> dict[str, Any]:
    """锁定模型状态输入语义，防止检查点跨口径误用。"""

    return {
        "schema": STATE_ADAPTER_SCHEMA,
        "state_source": "canonical_partial_ast_not_action_history",
        "node_embeddings": ["category", "operator_or_feature", "window"],
        "manual_features": [
            "max(filled_node_depth, open_slot_depth)/max_depth",
            "operator_count/max_nodes",
            "node_count/max_nodes",
        ],
        "legal_mask_source": "GrammarState.legal_transitions",
    }


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数")


def _non_negative_float(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} 必须是有限非负数")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("d_model", "num_heads", "num_layers", "dim_feedforward"):
            _positive_int(getattr(self, name), name)
        if self.d_model % self.num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1)")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 1.0
    greedy: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.temperature, Real)
            or isinstance(self.temperature, bool)
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0
        ):
            raise ValueError("temperature 必须是有限的正实数且不能是 bool")
        if not isinstance(self.greedy, bool):
            raise ValueError("greedy 必须严格为 bool")
        object.__setattr__(self, "temperature", float(self.temperature))


@dataclass(frozen=True, slots=True)
class RewardConfig:
    long_ir_lambda: float = 0.3
    long_ir_cap: float = 2.0
    barra_ts_penalty_mu: float = 0.2
    barra_min_common_periods: int = 60
    reward_floor: float = 1e-8
    candidate_industry_neutralization: bool = False
    reward_clip_min: float | None = None
    reward_clip_max: float | None = None

    def __post_init__(self) -> None:
        _non_negative_float(self.long_ir_lambda, "long_ir_lambda")
        _non_negative_float(self.long_ir_cap, "long_ir_cap")
        _non_negative_float(self.barra_ts_penalty_mu, "barra_ts_penalty_mu")
        _positive_int(self.barra_min_common_periods, "barra_min_common_periods")
        if not 0.0 <= self.barra_ts_penalty_mu <= 1.0:
            raise ValueError("barra_ts_penalty_mu 必须位于 [0, 1] 内")
        if not math.isfinite(self.reward_floor) or self.reward_floor <= 0:
            raise ValueError("reward_floor 必须是有限正数")
        if not isinstance(self.candidate_industry_neutralization, bool):
            raise ValueError("candidate_industry_neutralization 必须严格为 bool")
        # 字段目前仅作为兼容性预留。避免用户设置后被静默忽略。
        if self.reward_clip_min is not None or self.reward_clip_max is not None:
            raise NotImplementedError("reward 截断逻辑尚未实现；两个 clip 字段当前必须为 None")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    log_z_learning_rate: float = 1e-3
    max_steps: int = 1_000
    gradient_clip_norm: float = 1.0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    weight_decay: float = 0.0
    max_sampling_multiplier: int = 10
    deterministic_algorithms: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.max_steps, "max_steps")
        _positive_int(self.max_sampling_multiplier, "max_sampling_multiplier")
        if (
            not math.isfinite(self.learning_rate)
            or not math.isfinite(self.log_z_learning_rate)
            or self.learning_rate <= 0
            or self.log_z_learning_rate <= 0
        ):
            raise ValueError("学习率必须是有限正数")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm 必须是有限正数")
        if not 0.0 <= self.optimizer_beta1 < 1.0:
            raise ValueError("optimizer_beta1 必须位于 [0, 1) 内")
        if not 0.0 <= self.optimizer_beta2 < 1.0:
            raise ValueError("optimizer_beta2 必须位于 [0, 1) 内")
        if not math.isfinite(self.optimizer_eps) or self.optimizer_eps <= 0.0:
            raise ValueError("optimizer_eps 必须是有限正数")
        _non_negative_float(self.weight_decay, "weight_decay")
        if not isinstance(self.deterministic_algorithms, bool):
            raise ValueError("deterministic_algorithms 必须严格为 bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed 必须是整数")


@dataclass(frozen=True, slots=True)
class GFNConfig:
    search_space: SearchSpaceConfig = SearchSpaceConfig()
    model: ModelConfig = ModelConfig()
    sampling: SamplingConfig = SamplingConfig()
    reward: RewardConfig = RewardConfig()
    training: TrainingConfig = TrainingConfig()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": CONFIG_SCHEMA,
            "config": asdict(self),
            "state_adapter": state_adapter_manifest(),
            "token_space_fingerprint": action_space_fingerprint(),
            "state_space_fingerprint": state_space_fingerprint(),
            "transition_space_fingerprint": transition_space_fingerprint(),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.manifest(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class TrainingStats:
    step: int = 0
    optimizer_step: int = 0
    loss: float | None = None
    log_z: float | None = None
    reward_mean: float | None = None
    reward_median: float | None = None
    log_reward_mean: float | None = None
    expression_unique_rate: float | None = None
    trajectory_length_mean: float | None = None
    trajectory_length_max: int | None = None
    policy_entropy_mean: float | None = None
    policy_entropy_normalized_mean: float | None = None
    gradient_norm: float | None = None
    sampled_count: int = 0
    effective_batch_size: int = 0
    invalid_reward_count: int = 0
    batch_rejection_rate: float | None = None
    resample_rounds: int = 0
    skipped_update: bool = False
    illegal_action_rate: float = 0.0
    batch_corr_mean: float | None = None
    batch_corr_median: float | None = None


__all__ = [
    "CONFIG_SCHEMA",
    "GFNConfig",
    "ModelConfig",
    "RewardConfig",
    "SamplingConfig",
    "STATE_ADAPTER_SCHEMA",
    "SearchSpaceConfig",
    "TrainingConfig",
    "TrainingStats",
    "state_adapter_manifest",
]
