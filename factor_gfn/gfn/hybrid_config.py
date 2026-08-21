"""Frozen configuration contract for Stage 5 hybrid-variance training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Literal

from factor_gfn.grammar import (
    DAILY_DERIVED_V1_FEATURE_SPACE,
    RAW_DAILY_FEATURE_SPACE,
    ActionRegistry,
    FeatureSpaceSpec,
    SearchSpaceConfig,
    action_space_fingerprint,
    resolve_exact_node_strata,
    state_space_fingerprint,
    transition_space_fingerprint,
    build_action_registry,
)

from .config import ModelConfig, RewardConfig, SamplingConfig, state_adapter_manifest
from .no_anchor_config import (
    ExhaustiveRegistryReuseConfig,
    NoAnchorComplexityConfig,
)


HYBRID_VARIANCE_CONFIG_SCHEMA = "factor_gfn.gfn_config.hybrid_variance.v1"
STAGE5_HYBRID_MAX_DEPTH = 5
STAGE5_HYBRID_MAX_NODES = 15
STAGE5_HYBRID_CONDITIONS = tuple(range(1, STAGE5_HYBRID_MAX_NODES + 1))
STAGE5_HYBRID_EXACT_TB_CONDITIONS = (1, 2)
STAGE5_HYBRID_LPV_CONDITIONS = tuple(range(3, STAGE5_HYBRID_MAX_NODES + 1))
STAGE5_HYBRID_DEFAULT_TRAJECTORIES_PER_BATCH = 16
STAGE5_HYBRID_SEED = 42
STAGE5_HYBRID_SEARCH_SPACE = SearchSpaceConfig(
    max_depth=STAGE5_HYBRID_MAX_DEPTH,
    max_nodes=STAGE5_HYBRID_MAX_NODES,
)
STAGE5_HYBRID_MODEL = ModelConfig(
    d_model=128,
    num_heads=4,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.0,
    token_policy_mode="grammar_hierarchical",
)
STAGE5_HYBRID_SAMPLING = SamplingConfig(temperature=1.0, greedy=False)
STAGE5_HYBRID_REWARD = RewardConfig(candidate_industry_neutralization=True)


def _positive_int(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return normalized


@dataclass(frozen=True, slots=True)
class HybridObjectiveConfig:
    """Frozen exact-TB/LPV condition partition for the hybrid mode."""

    objective_mode: Literal["hybrid_variance"] = "hybrid_variance"
    exact_tb_node_counts: tuple[int, ...] = STAGE5_HYBRID_EXACT_TB_CONDITIONS
    lpv_node_counts: tuple[int, ...] = STAGE5_HYBRID_LPV_CONDITIONS

    def __post_init__(self) -> None:
        if self.objective_mode != "hybrid_variance":
            raise ValueError("hybrid objective_mode must be hybrid_variance")
        if self.exact_tb_node_counts != STAGE5_HYBRID_EXACT_TB_CONDITIONS:
            raise ValueError("hybrid exact-TB conditions must be exactly (1, 2)")
        if self.lpv_node_counts != STAGE5_HYBRID_LPV_CONDITIONS:
            raise ValueError("hybrid LPV conditions must be exactly 3..15")
        if self.condition_node_counts != STAGE5_HYBRID_CONDITIONS:
            raise ValueError("hybrid objective conditions must partition 1..15")

    @property
    def condition_node_counts(self) -> tuple[int, ...]:
        return tuple(sorted(self.exact_tb_node_counts + self.lpv_node_counts))


@dataclass(frozen=True, slots=True)
class HybridTrainingConfig:
    """Policy-only optimizer and cycle budget for the hybrid mode."""

    max_cycles: int
    trajectories_per_batch: int = STAGE5_HYBRID_DEFAULT_TRAJECTORIES_PER_BATCH
    optimizer: Literal["adam"] = "adam"
    learning_rate: float = 1e-4
    model_gradient_clip_norm: float = 5.0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    weight_decay: float = 0.0
    max_sampling_multiplier: int = 10
    deterministic_algorithms: bool = True
    seed: int = STAGE5_HYBRID_SEED

    def __post_init__(self) -> None:
        _positive_int(self.max_cycles, "max_cycles")
        _positive_int(
            self.trajectories_per_batch,
            "trajectories_per_batch",
            minimum=2,
        )
        _positive_int(self.max_sampling_multiplier, "max_sampling_multiplier")
        if self.optimizer != "adam":
            raise ValueError("hybrid policy optimizer must be adam")

        frozen_reals = {
            "learning_rate": (self.learning_rate, 1e-4),
            "model_gradient_clip_norm": (self.model_gradient_clip_norm, 5.0),
            "optimizer_beta1": (self.optimizer_beta1, 0.9),
            "optimizer_beta2": (self.optimizer_beta2, 0.999),
            "optimizer_eps": (self.optimizer_eps, 1e-8),
            "weight_decay": (self.weight_decay, 0.0),
        }
        for name, (value, expected) in frozen_reals.items():
            normalized = _finite_real(value, name)
            if normalized != expected:
                raise ValueError(f"hybrid {name} must remain {expected}")
            object.__setattr__(self, name, normalized)
        if self.deterministic_algorithms is not True:
            raise ValueError("hybrid training requires deterministic_algorithms=True")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    @property
    def conditions_per_cycle(self) -> int:
        return len(STAGE5_HYBRID_CONDITIONS)

    @property
    def optimizer_steps_per_cycle(self) -> int:
        return self.conditions_per_cycle

    @property
    def trajectories_per_cycle(self) -> int:
        return self.conditions_per_cycle * self.trajectories_per_batch

    @property
    def total_optimizer_steps(self) -> int:
        return self.max_cycles * self.optimizer_steps_per_cycle

    @property
    def total_training_trajectories(self) -> int:
        return self.max_cycles * self.trajectories_per_cycle


@dataclass(frozen=True, slots=True)
class HybridVarianceGFNConfig:
    """Isolated Stage 5 5/15 config with no learned-logZ training fields."""

    training: HybridTrainingConfig
    feature_space: FeatureSpaceSpec = RAW_DAILY_FEATURE_SPACE
    search_space: SearchSpaceConfig = STAGE5_HYBRID_SEARCH_SPACE
    model: ModelConfig = STAGE5_HYBRID_MODEL
    sampling: SamplingConfig = STAGE5_HYBRID_SAMPLING
    reward: RewardConfig = STAGE5_HYBRID_REWARD
    objective: HybridObjectiveConfig = HybridObjectiveConfig()
    complexity: NoAnchorComplexityConfig = NoAnchorComplexityConfig(
        exact_normalizer_node_counts=STAGE5_HYBRID_EXACT_TB_CONDITIONS,
        exact_node_retry_budget=3,
    )
    exhaustive_registry_reuse: ExhaustiveRegistryReuseConfig = (
        ExhaustiveRegistryReuseConfig()
    )

    def __post_init__(self) -> None:
        if self.feature_space not in (
            RAW_DAILY_FEATURE_SPACE,
            DAILY_DERIVED_V1_FEATURE_SPACE,
        ):
            raise ValueError("hybrid config 仅支持 Raw Daily 或 Daily-Derived v1")
        if self.search_space != STAGE5_HYBRID_SEARCH_SPACE:
            raise ValueError("hybrid search space must remain max_depth=5, max_nodes=15")
        if self.model != STAGE5_HYBRID_MODEL:
            raise ValueError("hybrid policy architecture must remain frozen")
        if self.sampling != STAGE5_HYBRID_SAMPLING:
            raise ValueError("hybrid sampling settings must remain frozen")
        if self.reward != STAGE5_HYBRID_REWARD:
            raise ValueError("hybrid Reward settings must remain frozen")
        if (
            self.complexity.exact_normalizer_node_counts
            != STAGE5_HYBRID_EXACT_TB_CONDITIONS
        ):
            raise ValueError("hybrid exact normalizer conditions must be (1, 2)")
        if self.resolved_condition_node_counts != self.objective.condition_node_counts:
            raise ValueError("hybrid grammar and objective conditions must both be 1..15")

    @property
    def action_registry(self) -> ActionRegistry:
        return build_action_registry(self.feature_space)

    @property
    def resolved_condition_node_counts(self) -> tuple[int, ...]:
        return resolve_exact_node_strata(
            self.search_space,
            self.action_registry,
        ).resolved_feasible_node_counts

    def manifest(self) -> dict[str, Any]:
        config_payload = asdict(self)
        config_payload.pop("feature_space")
        registry = self.action_registry
        manifest = {
            "schema": HYBRID_VARIANCE_CONFIG_SCHEMA,
            "config": config_payload,
            "resolved_condition_node_counts": self.resolved_condition_node_counts,
            "training_units": {
                "conditions_per_cycle": self.training.conditions_per_cycle,
                "optimizer_steps_per_cycle": self.training.optimizer_steps_per_cycle,
                "trajectories_per_cycle": self.training.trajectories_per_cycle,
                "total_optimizer_steps": self.training.total_optimizer_steps,
                "total_training_trajectories": (
                    self.training.total_training_trajectories
                ),
            },
            "state_adapter": state_adapter_manifest(registry),
            "token_space_fingerprint": action_space_fingerprint(registry),
            "state_space_fingerprint": state_space_fingerprint(),
            "transition_space_fingerprint": transition_space_fingerprint(registry),
        }
        if self.feature_space != RAW_DAILY_FEATURE_SPACE:
            manifest["feature_space"] = self.feature_space.manifest()
            manifest["feature_space_fingerprint"] = self.feature_space.fingerprint()
        return manifest

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.manifest(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_stage5_hybrid_variance_5_15_config(
    *,
    max_cycles: int,
    trajectories_per_batch: int = STAGE5_HYBRID_DEFAULT_TRAJECTORIES_PER_BATCH,
    seed: int = STAGE5_HYBRID_SEED,
    feature_space: FeatureSpaceSpec = RAW_DAILY_FEATURE_SPACE,
) -> HybridVarianceGFNConfig:
    """Build the frozen 5/15 hybrid contract for one explicit cycle budget."""

    return HybridVarianceGFNConfig(
        feature_space=feature_space,
        training=HybridTrainingConfig(
            max_cycles=max_cycles,
            trajectories_per_batch=trajectories_per_batch,
            seed=seed,
        )
    )


__all__ = [
    "HYBRID_VARIANCE_CONFIG_SCHEMA",
    "STAGE5_HYBRID_CONDITIONS",
    "STAGE5_HYBRID_DEFAULT_TRAJECTORIES_PER_BATCH",
    "STAGE5_HYBRID_EXACT_TB_CONDITIONS",
    "STAGE5_HYBRID_LPV_CONDITIONS",
    "STAGE5_HYBRID_MAX_DEPTH",
    "STAGE5_HYBRID_MAX_NODES",
    "STAGE5_HYBRID_MODEL",
    "STAGE5_HYBRID_REWARD",
    "STAGE5_HYBRID_SAMPLING",
    "STAGE5_HYBRID_SEARCH_SPACE",
    "STAGE5_HYBRID_SEED",
    "HybridObjectiveConfig",
    "HybridTrainingConfig",
    "HybridVarianceGFNConfig",
    "build_stage5_hybrid_variance_5_15_config",
]
