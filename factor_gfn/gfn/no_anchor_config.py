"""Formal no-anchor conditional Stage 5 configuration.

This module is intentionally not exported from :mod:`factor_gfn.gfn` until the
Trainer and checkpoint migrations are complete.  It therefore cannot be mixed
accidentally with the historical anchor-aware runtime during the staged cutover.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Literal

from factor_gfn.grammar import (
    SearchSpaceConfig,
    action_space_fingerprint,
    resolve_exact_node_strata,
    state_space_fingerprint,
    transition_space_fingerprint,
)

from .config import (
    ModelConfig,
    RewardConfig,
    SamplingConfig,
    TrainingConfig,
    state_adapter_manifest,
)
from .no_anchor_contract import NoAnchorStrataContract


NO_ANCHOR_CONFIG_SCHEMA = "factor_gfn.gfn_config.no_anchor.v1"
FORMAL_STAGE5_MAX_DEPTH = 6
FORMAL_STAGE5_MAX_NODES = 20
FORMAL_STAGE5_SEARCH_SPACE = SearchSpaceConfig(
    max_depth=FORMAL_STAGE5_MAX_DEPTH,
    max_nodes=FORMAL_STAGE5_MAX_NODES,
)
FORMAL_STAGE5_NO_ANCHOR_MAX_STEPS = 1_000
FORMAL_STAGE5_NO_ANCHOR_SEED = 42
FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT = (
    "b6453816d90f89609e506e02d6c8c0a9d3eda37571ea64079ecd91c9ad341789"
)
STAGE5_LOGZ_ADAM_LR2E2_AB_EXPERIMENT_ID = "logz_adam_lr2e2_seed42"
STAGE5_LOGZ_ADAM_LR2E2_AB_CONFIG_FINGERPRINT = (
    "8435eb605b29b59af02f60d0ee68392f2ecee6303aae9da14464c749cd14fa8a"
)
STAGE5_LOGZ_SGD_LR1E1_B1_EXPERIMENT_ID = "logz_sgd_lr1e1_seed42"
STAGE5_LOGZ_SGD_LR1E1_B1_CONFIG_FINGERPRINT = (
    "936e4e185b12a827bc584e0bdc6feb3b87a542ce97590e1df5276b6bf93ce5df"
)


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class NoAnchorComplexityConfig:
    """Exact-N scheduling with E separated from the discovery allocation."""

    exact_normalizer_node_counts: tuple[int, ...] = (1, 2)
    exact_node_retry_budget: int = 2
    low_effective_update_rate_warning_threshold: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exact_normalizer_node_counts, tuple):
            raise TypeError("exact_normalizer_node_counts must be a tuple")
        values = self.exact_normalizer_node_counts
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        ):
            raise ValueError(
                "exact_normalizer_node_counts must contain positive integers"
            )
        if len(set(values)) != len(values):
            raise ValueError("exact_normalizer_node_counts must not contain duplicates")
        object.__setattr__(self, "exact_normalizer_node_counts", tuple(sorted(values)))
        if (
            isinstance(self.exact_node_retry_budget, bool)
            or not isinstance(self.exact_node_retry_budget, int)
            or self.exact_node_retry_budget < 0
        ):
            raise ValueError("exact_node_retry_budget must be a non-negative integer")
        threshold = self.low_effective_update_rate_warning_threshold
        if threshold is not None:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not 0.0 <= float(threshold) <= 1.0
            ):
                raise ValueError(
                    "low_effective_update_rate_warning_threshold must be in [0, 1]"
                )
            object.__setattr__(
                self,
                "low_effective_update_rate_warning_threshold",
                float(threshold),
            )


@dataclass(frozen=True, slots=True)
class HistoricalLogZInitializationConfig:
    """Strict provenance contract for importing diagnostic medians as constants."""

    mode: Literal["verified_historical_median"] = "verified_historical_median"
    source_scope: Literal["training_only_diagnostic"] = "training_only_diagnostic"
    reuse_scope: Literal["initialization_constants_only"] = (
        "initialization_constants_only"
    )
    require_semantics_equivalence: bool = True
    restore_parameter_state: bool = False
    restore_model_state: bool = False
    restore_optimizer_state: bool = False
    restore_scheduler_state: bool = False
    restore_checkpoint_state: bool = False

    def __post_init__(self) -> None:
        if self.mode != "verified_historical_median":
            raise ValueError("formal no-anchor initialization requires verified medians")
        if self.source_scope != "training_only_diagnostic":
            raise ValueError("historical logZ initialization must be training-only")
        if self.reuse_scope != "initialization_constants_only":
            raise ValueError("historical reuse is limited to initialization constants")
        if self.require_semantics_equivalence is not True:
            raise ValueError("historical initialization requires semantics equivalence")
        forbidden_restores = (
            self.restore_parameter_state,
            self.restore_model_state,
            self.restore_optimizer_state,
            self.restore_scheduler_state,
            self.restore_checkpoint_state,
        )
        if any(value is not False for value in forbidden_restores):
            raise ValueError("historical initialization must not restore training state")


@dataclass(frozen=True, slots=True)
class NoAnchorCalibrationConfig:
    """Problem-driven targeted fallback, never an all-L formal-run precheck."""

    enabled: bool = False
    target_node_counts: tuple[int, ...] = ()
    minimum_valid_samples: int = 64
    maximum_requested_slots_per_N: int = 128
    comparison_window: int = 16
    median_absolute_tolerance: float = 0.25
    iqr_absolute_tolerance: float = 0.50
    requires_fresh_training_state: bool = True
    allow_mid_training_log_z_reset: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("targeted calibration enabled must be bool")
        if not isinstance(self.target_node_counts, tuple):
            raise TypeError("target_node_counts must be a tuple")
        targets = tuple(sorted(self.target_node_counts))
        if len(targets) != len(set(targets)):
            raise ValueError("target_node_counts must not contain duplicates")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in targets
        ):
            raise ValueError("target_node_counts must contain positive integers")
        object.__setattr__(self, "target_node_counts", targets)
        if self.enabled and not targets:
            raise ValueError("enabled targeted calibration requires target_node_counts")
        if not self.enabled and targets:
            raise ValueError("disabled targeted calibration must not name targets")
        if self.minimum_valid_samples != 64:
            raise ValueError("targeted calibration minimum_valid_samples must be 64")
        if self.maximum_requested_slots_per_N != 128:
            raise ValueError(
                "targeted calibration maximum_requested_slots_per_N must be 128"
            )
        if self.comparison_window != 16:
            raise ValueError("targeted calibration comparison_window must be 16")
        if self.median_absolute_tolerance != 0.25:
            raise ValueError("targeted calibration median_absolute_tolerance must be 0.25")
        if self.iqr_absolute_tolerance != 0.50:
            raise ValueError("targeted calibration iqr_absolute_tolerance must be 0.50")
        if self.requires_fresh_training_state is not True:
            raise ValueError("targeted calibration requires a fresh training state")
        if self.allow_mid_training_log_z_reset is not False:
            raise ValueError("mid-training logZ reset is forbidden")


@dataclass(frozen=True, slots=True)
class ExhaustiveRegistryReuseConfig:
    """Run-initialization equivalence proof followed by hash-only lookup."""

    equivalence_verification_phase: str = "run_initialization_once"
    discovery_lookup_key: str = "structural_hash"
    reenumerate_during_candidate_lookup: bool = False

    def __post_init__(self) -> None:
        if self.equivalence_verification_phase != "run_initialization_once":
            raise ValueError("equivalence verification must run once at initialization")
        if self.discovery_lookup_key != "structural_hash":
            raise ValueError("exhaustive discovery lookup must use structural_hash")
        if self.reenumerate_during_candidate_lookup is not False:
            raise ValueError("candidate lookup must not re-enumerate canonical strata")


@dataclass(frozen=True, slots=True)
class NoAnchorGFNConfig:
    """No-anchor config whose resolved scheduler contract is D=F."""

    search_space: SearchSpaceConfig = FORMAL_STAGE5_SEARCH_SPACE
    model: ModelConfig = ModelConfig(token_policy_mode="grammar_hierarchical")
    sampling: SamplingConfig = SamplingConfig()
    reward: RewardConfig = RewardConfig(candidate_industry_neutralization=True)
    training: TrainingConfig = TrainingConfig()
    complexity: NoAnchorComplexityConfig = NoAnchorComplexityConfig()
    initialization: HistoricalLogZInitializationConfig = (
        HistoricalLogZInitializationConfig()
    )
    calibration: NoAnchorCalibrationConfig = NoAnchorCalibrationConfig()
    exhaustive_registry_reuse: ExhaustiveRegistryReuseConfig = (
        ExhaustiveRegistryReuseConfig()
    )

    def __post_init__(self) -> None:
        if self.model.token_policy_mode != "grammar_hierarchical":
            raise ValueError(
                "formal no-anchor conditional training requires grammar_hierarchical"
            )
        learned = set(self.resolved_strata().learned_normalizer_node_counts)
        targets = set(self.calibration.target_node_counts)
        if not targets.issubset(learned):
            raise ValueError("targeted calibration node counts must be a subset of L")

    def resolved_strata(self) -> NoAnchorStrataContract:
        feasible = resolve_exact_node_strata(
            self.search_space
        ).resolved_feasible_node_counts
        return NoAnchorStrataContract.resolve(
            feasible,
            self.complexity.exact_normalizer_node_counts,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": NO_ANCHOR_CONFIG_SCHEMA,
            "config": asdict(self),
            "strata": self.resolved_strata().manifest(),
            "normalizer_initialization": {
                "mode": self.initialization.mode,
                "source_scope": self.initialization.source_scope,
                "reuse_scope": self.initialization.reuse_scope,
                "semantics_equivalence_required": True,
                "training_state_restore_forbidden": True,
            },
            "normalizer_calibration_scope": "targeted_problem_strata_only",
            "mid_training_log_z_reset": "forbidden",
            "state_adapter": state_adapter_manifest(),
            "token_space_fingerprint": action_space_fingerprint(),
            "state_space_fingerprint": state_space_fingerprint(),
            "transition_space_fingerprint": transition_space_fingerprint(),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.manifest(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_formal_stage5_no_anchor_6_20_config(
    *,
    training: TrainingConfig,
    seed: int | None = None,
) -> NoAnchorGFNConfig:
    """Build the frozen 6/20 boundary without freezing Step 12 hyperparameters."""

    if seed is not None:
        training_payload = asdict(training)
        training_payload["seed"] = seed
        training = TrainingConfig(**training_payload)
    return NoAnchorGFNConfig(
        search_space=FORMAL_STAGE5_SEARCH_SPACE,
        model=ModelConfig(
            d_model=128,
            num_heads=4,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.0,
            token_policy_mode="grammar_hierarchical",
        ),
        sampling=SamplingConfig(temperature=1.0, greedy=False),
        reward=RewardConfig(candidate_industry_neutralization=True),
        training=training,
    )


def build_frozen_stage5_no_anchor_6_20_config() -> NoAnchorGFNConfig:
    """Return the first formal Stage 5 run contract frozen after final health QA."""

    base = build_formal_stage5_no_anchor_6_20_config(
        training=TrainingConfig(
            batch_size=8,
            learning_rate=1e-4,
            log_z_learning_rate=1e-2,
            max_steps=FORMAL_STAGE5_NO_ANCHOR_MAX_STEPS,
            model_gradient_clip_norm=5.0,
            log_z_gradient_clip_norm=5.0,
            optimizer_beta1=0.9,
            optimizer_beta2=0.999,
            optimizer_eps=1e-8,
            weight_decay=0.0,
            max_sampling_multiplier=10,
            deterministic_algorithms=True,
            seed=FORMAL_STAGE5_NO_ANCHOR_SEED,
        )
    )
    config = NoAnchorGFNConfig(
        search_space=base.search_space,
        model=base.model,
        sampling=base.sampling,
        reward=base.reward,
        training=base.training,
        complexity=NoAnchorComplexityConfig(
            exact_normalizer_node_counts=(1, 2),
            exact_node_retry_budget=3,
        ),
        initialization=base.initialization,
        calibration=NoAnchorCalibrationConfig(
            enabled=True,
            target_node_counts=(17, 18),
        ),
        exhaustive_registry_reuse=base.exhaustive_registry_reuse,
    )
    if config.fingerprint() != FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT:
        raise RuntimeError("frozen formal Stage 5 config fingerprint drift")
    return config


def build_stage5_logz_adam_lr2e2_ab_config() -> NoAnchorGFNConfig:
    """Return experiment A: the formal seed42 contract with only logZ LR changed."""

    baseline = build_frozen_stage5_no_anchor_6_20_config()
    training_payload = asdict(baseline.training)
    training_payload["log_z_learning_rate"] = 2e-2
    config = NoAnchorGFNConfig(
        search_space=baseline.search_space,
        model=baseline.model,
        sampling=baseline.sampling,
        reward=baseline.reward,
        training=TrainingConfig(**training_payload),
        complexity=baseline.complexity,
        initialization=baseline.initialization,
        calibration=baseline.calibration,
        exhaustive_registry_reuse=baseline.exhaustive_registry_reuse,
    )
    if config.fingerprint() != STAGE5_LOGZ_ADAM_LR2E2_AB_CONFIG_FINGERPRINT:
        raise RuntimeError("Stage 5 logZ Adam LR=2e-2 experiment A fingerprint drift")
    return config


def build_stage5_logz_sgd_lr1e1_b1_config() -> NoAnchorGFNConfig:
    """Return B1 with the frozen baseline parameters and learned-logZ LR=0.1."""

    baseline = build_frozen_stage5_no_anchor_6_20_config()
    training_payload = asdict(baseline.training)
    training_payload["log_z_learning_rate"] = 1e-1
    config = NoAnchorGFNConfig(
        search_space=baseline.search_space,
        model=baseline.model,
        sampling=baseline.sampling,
        reward=baseline.reward,
        training=TrainingConfig(**training_payload),
        complexity=baseline.complexity,
        initialization=baseline.initialization,
        calibration=baseline.calibration,
        exhaustive_registry_reuse=baseline.exhaustive_registry_reuse,
    )
    if config.fingerprint() != STAGE5_LOGZ_SGD_LR1E1_B1_CONFIG_FINGERPRINT:
        raise RuntimeError("Stage 5 logZ SGD LR=0.1 experiment B1 fingerprint drift")
    return config


__all__ = [
    "ExhaustiveRegistryReuseConfig",
    "FORMAL_STAGE5_MAX_DEPTH",
    "FORMAL_STAGE5_MAX_NODES",
    "FORMAL_STAGE5_NO_ANCHOR_MAX_STEPS",
    "FORMAL_STAGE5_NO_ANCHOR_SEED",
    "FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT",
    "STAGE5_LOGZ_ADAM_LR2E2_AB_EXPERIMENT_ID",
    "STAGE5_LOGZ_ADAM_LR2E2_AB_CONFIG_FINGERPRINT",
    "STAGE5_LOGZ_SGD_LR1E1_B1_EXPERIMENT_ID",
    "STAGE5_LOGZ_SGD_LR1E1_B1_CONFIG_FINGERPRINT",
    "FORMAL_STAGE5_SEARCH_SPACE",
    "NO_ANCHOR_CONFIG_SCHEMA",
    "HistoricalLogZInitializationConfig",
    "NoAnchorCalibrationConfig",
    "NoAnchorComplexityConfig",
    "NoAnchorGFNConfig",
    "build_formal_stage5_no_anchor_6_20_config",
    "build_frozen_stage5_no_anchor_6_20_config",
    "build_stage5_logz_adam_lr2e2_ab_config",
    "build_stage5_logz_sgd_lr1e1_b1_config",
]
