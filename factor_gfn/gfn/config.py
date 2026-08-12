"""阶段四 GFlowNet 的集中配置和运行统计数据合同。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal

from factor_gfn.grammar import (
    SearchSpaceConfig,
    action_space_fingerprint,
    state_space_fingerprint,
    transition_space_fingerprint,
)


CONFIG_SCHEMA = "factor_gfn.gfn_config.v5"
STATE_ADAPTER_SCHEMA = "factor_gfn.state_adapter.v1"
TokenPolicyMode = Literal["flat", "arity_hierarchical", "grammar_hierarchical"]
TOKEN_POLICY_MODES = ("flat", "arity_hierarchical", "grammar_hierarchical")
LEGACY_TOKEN_POLICY_MODES = ("flat", "arity_hierarchical")
STAGE5_TOKEN_POLICY_MODE: TokenPolicyMode = "grammar_hierarchical"


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
    token_policy_mode: TokenPolicyMode = "flat"

    def __post_init__(self) -> None:
        for name in ("d_model", "num_heads", "num_layers", "dim_feedforward"):
            _positive_int(getattr(self, name), name)
        if self.d_model % self.num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.token_policy_mode not in TOKEN_POLICY_MODES:
            raise ValueError(
                f"token_policy_mode 必须属于 {TOKEN_POLICY_MODES}"
            )


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
    initial_log_z: float = 0.0
    max_steps: int = 1_000
    model_gradient_clip_norm: float = 1.0
    log_z_gradient_clip_norm: float = 1.0
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
        if (
            not isinstance(self.initial_log_z, Real)
            or isinstance(self.initial_log_z, bool)
            or not math.isfinite(float(self.initial_log_z))
        ):
            raise ValueError("initial_log_z 必须是有限实数且不能是 bool")
        object.__setattr__(self, "initial_log_z", float(self.initial_log_z))
        for name in ("model_gradient_clip_norm", "log_z_gradient_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
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


STAGE5_REAL_SEARCH_SPACE = SearchSpaceConfig(max_depth=6, max_nodes=15)
STAGE5_REAL_MODEL = ModelConfig(
    d_model=128,
    num_heads=4,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.0,
    token_policy_mode=STAGE5_TOKEN_POLICY_MODE,
)
STAGE5_REAL_SAMPLING = SamplingConfig(temperature=1.0, greedy=False)
STAGE5_REAL_REWARD = RewardConfig(candidate_industry_neutralization=True)


def build_stage5_real_training_config(*, max_steps: int, seed: int = 42) -> GFNConfig:
    """Build the frozen stage-five real-search preset for one recorded run budget."""

    return GFNConfig(
        search_space=STAGE5_REAL_SEARCH_SPACE,
        model=STAGE5_REAL_MODEL,
        sampling=STAGE5_REAL_SAMPLING,
        reward=STAGE5_REAL_REWARD,
        training=TrainingConfig(
            batch_size=8,
            learning_rate=1e-4,
            log_z_learning_rate=1e-2,
            initial_log_z=39.0,
            max_steps=max_steps,
            model_gradient_clip_norm=5.0,
            log_z_gradient_clip_norm=5.0,
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            optimizer_eps=1e-8,
            weight_decay=0.0,
            max_sampling_multiplier=10,
            deterministic_algorithms=True,
            seed=seed,
        ),
    )


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
    terminal_node_count_p50: float | None = None
    terminal_node_count_p90: float | None = None
    max_node_terminal_rate: float | None = None
    policy_entropy_mean: float | None = None
    policy_entropy_normalized_mean: float | None = None
    group_entropy_mean: float | None = None
    group_entropy_normalized_mean: float | None = None
    leaf_group_probability_mean: float | None = None
    unary_group_probability_mean: float | None = None
    binary_group_probability_mean: float | None = None
    leaf_action_rate: float | None = None
    unary_action_rate: float | None = None
    binary_action_rate: float | None = None
    grammar_category_entropy_mean: float | None = None
    grammar_category_entropy_normalized_mean: float | None = None
    operator_entropy_mean: float | None = None
    operator_entropy_normalized_mean: float | None = None
    window_entropy_mean: float | None = None
    window_entropy_normalized_mean: float | None = None
    feature_category_probability_mean: float | None = None
    unary_category_probability_mean: float | None = None
    ts_unary_category_probability_mean: float | None = None
    binary_category_probability_mean: float | None = None
    ts_binary_category_probability_mean: float | None = None
    cross_sectional_category_probability_mean: float | None = None
    feature_category_action_rate: float | None = None
    unary_category_action_rate: float | None = None
    ts_unary_category_action_rate: float | None = None
    binary_category_action_rate: float | None = None
    ts_binary_category_action_rate: float | None = None
    cross_sectional_category_action_rate: float | None = None
    window_5_probability_mean: float | None = None
    window_10_probability_mean: float | None = None
    window_20_probability_mean: float | None = None
    window_40_probability_mean: float | None = None
    window_60_probability_mean: float | None = None
    window_5_action_rate: float | None = None
    window_10_action_rate: float | None = None
    window_20_action_rate: float | None = None
    window_40_action_rate: float | None = None
    window_60_action_rate: float | None = None
    temporal_operator_action_rate: float | None = None
    gradient_norm: float | None = None
    tb_delta_mean: float | None = None
    tb_delta_std: float | None = None
    tb_delta_rms: float | None = None
    tb_delta_mean_square_ratio: float | None = None
    tb_delta_std_square_ratio: float | None = None
    mean_log_pf: float | None = None
    mean_log_pb: float | None = None
    model_gradient_norm_before_clip: float | None = None
    log_z_gradient_before_clip: float | None = None
    model_gradient_clip_coefficient: float | None = None
    log_z_gradient_clip_coefficient: float | None = None
    # 仅用于恢复旧联合裁剪统计；独立裁剪的新步骤保持 None。
    gradient_clip_coefficient: float | None = None
    model_parameter_update_norm: float | None = None
    model_relative_update_norm: float | None = None
    log_z_update: float | None = None
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
    "LEGACY_TOKEN_POLICY_MODES",
    "ModelConfig",
    "RewardConfig",
    "SamplingConfig",
    "STAGE5_REAL_MODEL",
    "STAGE5_REAL_REWARD",
    "STAGE5_REAL_SAMPLING",
    "STAGE5_REAL_SEARCH_SPACE",
    "STAGE5_TOKEN_POLICY_MODE",
    "STATE_ADAPTER_SCHEMA",
    "SearchSpaceConfig",
    "TrainingConfig",
    "TrainingStats",
    "TOKEN_POLICY_MODES",
    "TokenPolicyMode",
    "build_stage5_real_training_config",
    "state_adapter_manifest",
]
