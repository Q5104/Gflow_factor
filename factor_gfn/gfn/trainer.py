"""GFlowNet 参数更新闭环与不依赖行情的合成 Reward。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Protocol
from uuid import uuid4

import numpy as np
import torch

from factor_gfn.grammar import (
    WINDOWS,
    Expression,
    action_space_fingerprint,
    get_action,
    state_space_fingerprint,
    transition_space_fingerprint,
)

from .config import GFNConfig, TrainingStats
from .no_anchor_config import NoAnchorGFNConfig
from .calibration_stability import CalibrationStabilityConfig
from .calibration import (
    CalibrationStatistics,
    NormalizerCalibration,
    require_training_only_provider_manifest,
)
from .complexity_scheduler import BalancedNodeCountScheduler
from .exhaustive import (
    ExactMassResult,
    ExhaustiveRegistry,
    count_canonical_terminals,
)
from .exhaustive_registry_reuse import (
    ExhaustiveReuseSemantics,
    ExhaustiveStratumReuseProof,
    ProvenExhaustiveRewardLookup,
    prove_exhaustive_stratum_reuse,
)
from .historical_log_z import (
    HistoricalLogZInitialization,
    load_verified_historical_log_z_initialization,
)
from .targeted_calibration import (
    TargetedLogZInitialization,
    load_verified_targeted_calibration_artifact,
)
from .loss import TrajectoryBalanceLoss
from .model import ForwardPolicyNetwork
from .policy_sampler import sample_trajectories
from .state_adapter import StateAdapter


TRAINER_SCHEMA = "factor_gfn.trainer.no_anchor.v1"
LEGACY_TRAINER_SCHEMA = "factor_gfn.trainer.legacy_read_only.v1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True, slots=True)
class RewardAssignment:
    valid: bool
    reward: float | None = None
    log_reward: float | None = None
    rejection_reason: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ValueError("valid 必须严格为 bool")
        if self.valid:
            if self.reward is None or self.log_reward is None:
                raise ValueError("有效 RewardAssignment 必须包含 reward 和 log_reward")
            reward = float(self.reward)
            log_reward = float(self.log_reward)
            if not math.isfinite(reward) or reward <= 0.0 or not math.isfinite(log_reward):
                raise ValueError("有效 RewardAssignment 必须包含有限正 reward")
            if not math.isclose(log_reward, math.log(reward), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("RewardAssignment 的 log_reward 与 reward 不一致")
        elif self.reward is not None or self.log_reward is not None:
            raise ValueError("无效 RewardAssignment 不得携带 reward 或 log_reward")


class RewardProvider(Protocol):
    def evaluate(self, expression: Expression) -> RewardAssignment: ...

    def fingerprint(self) -> str: ...

    def manifest(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SyntheticRewardConfig:
    close_bonus: float = 0.75
    ts_mean_bonus: float = 0.50
    node_penalty: float = 0.10

    def __post_init__(self) -> None:
        for name in ("close_bonus", "ts_mean_bonus", "node_penalty"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限实数")
        if self.node_penalty < 0.0:
            raise ValueError("node_penalty 不能为负数")


class SyntheticRewardProvider:
    """确定、始终有效且具有明确 Token 偏好的合成目标。"""

    def __init__(self, config: SyntheticRewardConfig = SyntheticRewardConfig()) -> None:
        self.config = config

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "factor_gfn.synthetic_reward.v2",
            "data_scope": "training_only",
            "validation_oos_loaded": False,
            "context_fingerprint": "synthetic_training_only",
            "config": asdict(self.config),
            "formula": (
                "log_reward = close_bonus*contains(close) + "
                "ts_mean_bonus*contains(ts_mean) - node_penalty*node_count"
            ),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.manifest(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def evaluate(self, expression: Expression) -> RewardAssignment:
        if not isinstance(expression, Expression):
            raise TypeError("expression 必须是 Expression")
        names = {get_action(token_id).name for token_id in expression.to_prefix()}
        log_reward = (
            self.config.close_bonus * float("close" in names)
            + self.config.ts_mean_bonus * float("ts_mean" in names)
            - self.config.node_penalty * expression.stats.node_count
        )
        reward = math.exp(log_reward)
        return RewardAssignment(
            valid=True,
            reward=reward,
            log_reward=log_reward,
            metadata={"synthetic": True},
        )


def configure_cuda_determinism(
    device: str | torch.device,
    *,
    deterministic_algorithms: bool,
) -> str | None:
    """Establish the CuBLAS contract before the first CUDA matrix operation."""

    resolved = torch.device(device)
    if resolved.type != "cuda" or not deterministic_algorithms:
        return None
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif existing != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUDA 确定性训练要求 "
            f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG}，"
            f"当前值为 {existing!r}；请修正后重启 Python/Jupyter Kernel"
        )
    return CUBLAS_WORKSPACE_CONFIG


def seed_everything(seed: int, *, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic_algorithms
        torch.backends.cudnn.benchmark = not deterministic_algorithms


class GFNTrainer:
    """采样、Reward、TB Loss、反传和参数更新的最小训练器。"""

    def __init__(
        self,
        config: GFNConfig | NoAnchorGFNConfig,
        reward_provider: RewardProvider,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(config, (GFNConfig, NoAnchorGFNConfig)):
            raise TypeError("config must be GFNConfig or NoAnchorGFNConfig")
        self.no_anchor_mode = isinstance(config, NoAnchorGFNConfig)
        for method in ("evaluate", "fingerprint", "manifest"):
            if not callable(getattr(reward_provider, method, None)):
                raise TypeError(f"reward_provider 缺少 {method}()")
        provider_manifest = reward_provider.manifest()
        if not isinstance(provider_manifest, dict):
            raise TypeError("reward_provider.manifest() 必须返回 dict")
        declared_reward_config = provider_manifest.get("reward_config")
        if (
            declared_reward_config is not None
            and declared_reward_config != asdict(config.reward)
        ):
            raise ValueError(
                "GFNConfig.reward 与 RewardProvider 声明的 reward_config 不一致"
            )
        self.config = config
        self.reward_provider = reward_provider
        self.device = torch.device(device)
        self.cublas_workspace_config = configure_cuda_determinism(
            self.device,
            deterministic_algorithms=config.training.deterministic_algorithms,
        )
        seed_everything(
            config.training.seed,
            deterministic_algorithms=config.training.deterministic_algorithms,
        )
        if self.no_anchor_mode:
            no_anchor_strata = config.resolved_strata()
            resolved_complexity = {
                "resolved_feasible_node_counts": no_anchor_strata.feasible_node_counts,
                "resolved_exhaustive_node_counts": (
                    no_anchor_strata.exact_normalizer_node_counts
                ),
                "resolved_discovery_node_counts": no_anchor_strata.discovery_node_counts,
                "resolved_learned_node_counts": (
                    no_anchor_strata.learned_normalizer_node_counts
                ),
            }
            conditioned_enabled = True
        else:
            resolved_complexity = (
                config.resolved_complexity_strata()
                if config.complexity_scheduler.enabled else None
            )
            conditioned_enabled = config.complexity_scheduler.enabled
        self.adapter = StateAdapter(config.search_space)
        self.model = ForwardPolicyNetwork(config.model, config.search_space).to(self.device)
        training = config.training
        if resolved_complexity is None:
            self.tb_loss = TrajectoryBalanceLoss(
                initial_log_z=training.initial_log_z
            ).to(self.device)
        else:
            self.tb_loss = TrajectoryBalanceLoss(
                initial_log_z=training.initial_log_z,
                max_nodes=config.search_space.max_nodes,
                exact_node_counts=resolved_complexity[
                    "resolved_exhaustive_node_counts"
                ],
            ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "name": "policy",
                    "params": self.model.parameters(),
                    "lr": training.learning_rate,
                    "weight_decay": training.weight_decay,
                },
                {
                    "name": "normalizer",
                    "params": self.tb_loss.parameters(),
                    "lr": training.log_z_learning_rate,
                    # Per-N sparsity requires absent scalars to remain unchanged.
                    "weight_decay": 0.0,
                },
            ],
            betas=(training.optimizer_beta1, training.optimizer_beta2),
            eps=training.optimizer_eps,
        )
        self.step = 0
        self.optimizer_step = 0
        self.history: list[TrainingStats] = []
        self.last_step_timings: dict[str, float | None] = {}
        # Read-only diagnostics for the most recent discovery update.  This is
        # deliberately not part of the optimization/checkpoint state: callers
        # may persist it in a run-specific audit without changing training.
        self.last_discovery_trajectory_diagnostics: list[dict[str, Any]] = []
        self.run_id = uuid4().hex
        self.created_at_utc = datetime.now(timezone.utc).isoformat()
        self.complexity_scheduler: BalancedNodeCountScheduler | None = None
        self.resolved_feasible_node_counts: tuple[int, ...] = ()
        self.resolved_exhaustive_node_counts: tuple[int, ...] = ()
        self.resolved_discovery_node_counts: tuple[int, ...] = ()
        self.resolved_learned_node_counts: tuple[int, ...] = ()
        self.requested_count_by_N: dict[int, int] = {}
        self.sampled_attempt_count_by_N: dict[int, int] = {}
        self.valid_count_by_N: dict[int, int] = {}
        self.successful_update_count_by_N: dict[int, int] = {}
        self.retry_exhausted_count_by_N: dict[int, int] = {}
        self.calibration: NormalizerCalibration | None = None
        self.registered_exact_masses_by_N: dict[int, ExactMassResult] = {}
        self.exhaustive_reuse_proofs_by_N: dict[
            int, ExhaustiveStratumReuseProof
        ] = {}
        self.exhaustive_reward_lookups_by_N: dict[
            int, ProvenExhaustiveRewardLookup
        ] = {}
        self.historical_log_z_initialization: HistoricalLogZInitialization | None = (
            None
        )
        self.targeted_log_z_initialization: TargetedLogZInitialization | None = None
        if conditioned_enabled:
            if config.model.token_policy_mode != "grammar_hierarchical":
                raise ValueError(
                    "conditioned Trainer requires token_policy_mode="
                    "'grammar_hierarchical'"
                )
            if resolved_complexity is None:
                raise RuntimeError("enabled scheduler must resolve F/E/S")
            feasible = resolved_complexity["resolved_feasible_node_counts"]
            exhaustive = resolved_complexity["resolved_exhaustive_node_counts"]
            discovery = resolved_complexity["resolved_discovery_node_counts"]
            learned = resolved_complexity.get(
                "resolved_learned_node_counts", discovery
            )
            self.resolved_feasible_node_counts = feasible
            self.resolved_exhaustive_node_counts = exhaustive
            self.resolved_discovery_node_counts = discovery
            self.resolved_learned_node_counts = learned
            self.complexity_scheduler = BalancedNodeCountScheduler(
                discovery,
                seed=config.training.seed,
            )
            for counter in self._node_count_counters():
                counter.update({node_count: 0 for node_count in discovery})
            if config.calibration.enabled:
                require_training_only_provider_manifest(provider_manifest)
                if self.no_anchor_mode:
                    calibration_node_counts = config.calibration.target_node_counts
                    calibration_exhaustive_node_counts: tuple[int, ...] = ()
                    minimum_valid_samples = config.calibration.minimum_valid_samples
                    maximum_requested_slots = (
                        config.calibration.maximum_requested_slots_per_N
                    )
                    stability_config = CalibrationStabilityConfig(
                        minimum_valid_samples=minimum_valid_samples,
                        maximum_requested_slots=maximum_requested_slots,
                        comparison_window=config.calibration.comparison_window,
                        median_absolute_tolerance=(
                            config.calibration.median_absolute_tolerance
                        ),
                        iqr_absolute_tolerance=(
                            config.calibration.iqr_absolute_tolerance
                        ),
                    )
                else:
                    calibration_node_counts = feasible
                    calibration_exhaustive_node_counts = exhaustive
                    minimum_valid_samples = (
                        config.calibration.minimum_valid_calibration_samples
                    )
                    maximum_requested_slots = (
                        config.calibration.maximum_requested_calibration_slots_per_N
                    )
                    stability_config = None
                self.calibration = NormalizerCalibration(
                    node_counts=calibration_node_counts,
                    exhaustive_node_counts=calibration_exhaustive_node_counts,
                    minimum_valid_samples=minimum_valid_samples,
                    maximum_requested_slots_per_N=maximum_requested_slots,
                    seed=config.training.seed,
                    stability_config=stability_config,
                )
        elif config.calibration.enabled:
            raise ValueError("normalizer calibration requires conditioned scheduling")

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.model.parameters()) + list(self.tb_loss.parameters())

    def _normalizer_monitor_value(self) -> float:
        if self.tb_loss.normalizer_mode == "legacy_scalar":
            assert self.tb_loss.log_z is not None
            return float(self.tb_loss.log_z.detach())
        assert self.tb_loss.log_z_by_node_count is not None
        return float(self.tb_loss.log_z_by_node_count.detach().mean())

    @staticmethod
    def _tensor_l2_norm(tensors: list[torch.Tensor]) -> float:
        if not tensors:
            return 0.0
        total = torch.zeros((), dtype=torch.float64, device=tensors[0].device)
        for tensor in tensors:
            total = total + torch.sum(torch.square(tensor.detach().to(torch.float64)))
        return float(torch.sqrt(total))

    @classmethod
    def _gradient_l2_norm(cls, parameters: list[torch.nn.Parameter]) -> float:
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        return cls._tensor_l2_norm(gradients)

    @staticmethod
    def _parameter_update_l2_norm(
        parameters: list[torch.nn.Parameter],
        before: list[torch.Tensor],
    ) -> float:
        if len(parameters) != len(before):
            raise ValueError("参数快照长度与当前参数不一致")
        if not parameters:
            return 0.0
        total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
        for parameter, old_value in zip(parameters, before, strict=True):
            difference = parameter.detach().to(torch.float64) - old_value.to(torch.float64)
            total = total + torch.sum(torch.square(difference))
        return float(torch.sqrt(total))

    def run_metadata(self) -> dict[str, Any]:
        metadata = {
            "schema": TRAINER_SCHEMA if self.no_anchor_mode else LEGACY_TRAINER_SCHEMA,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "config_fingerprint": self.config.fingerprint(),
            "config_manifest": self.config.manifest(),
            "reward_provider": self.reward_provider.manifest(),
            "reward_provider_fingerprint": self.reward_provider.fingerprint(),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(self.device),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cublas_workspace_config": self.cublas_workspace_config,
            },
            "parameter_semantics": {
                "normalizer": self.tb_loss.normalizer_state_manifest(),
                "initial_log_z": {
                    "value": self.config.training.initial_log_z,
                    "meaning": "TB 全局归一化常数 logZ 的工程初值",
                },
                "optimizer_eps": {
                    "value": self.config.training.optimizer_eps,
                    "meaning": "Adam 分母数值稳定参数",
                },
                "reward_floor": {
                    "value": self.config.reward.reward_floor,
                    "meaning": "TB 对数所需的正值 Reward 下界",
                },
            },
            "complexity_scheduler": self.complexity_state_dict(),
            "normalizer_calibration": self.calibration_state_dict(),
            "historical_log_z_initialization": (
                None
                if self.historical_log_z_initialization is None
                else self.historical_log_z_initialization.manifest()
            ),
            "targeted_log_z_initialization": (
                None
                if self.targeted_log_z_initialization is None
                else self.targeted_log_z_initialization.manifest()
            ),
            "optimizer_steps": {
                "discovery": self.optimizer_step,
            },
        }
        return metadata

    def initialize_verified_historical_log_z(
        self,
        source_directory: str | Path,
    ) -> HistoricalLogZInitialization:
        """Import verified diagnostic medians once, without historical state."""

        if not self.no_anchor_mode or not isinstance(self.config, NoAnchorGFNConfig):
            raise RuntimeError("historical logZ initialization requires no-anchor mode")
        if self.step != 0 or self.optimizer_step != 0 or self.history:
            raise RuntimeError("historical logZ initialization requires fresh training state")
        if self.optimizer.state:
            raise RuntimeError("historical logZ initialization forbids optimizer state")
        if self.historical_log_z_initialization is not None:
            raise RuntimeError("historical logZ initialization is already configured")
        targeted = set(self.config.calibration.target_node_counts)
        imported = tuple(
            node_count
            for node_count in self.resolved_learned_node_counts
            if node_count not in targeted
        )
        record = load_verified_historical_log_z_initialization(
            source_directory,
            target_search_space=asdict(self.config.search_space),
            target_feasible_node_counts=self.resolved_feasible_node_counts,
            target_exact_node_counts=self.resolved_exhaustive_node_counts,
            target_learned_node_counts=self.resolved_learned_node_counts,
            target_semantics=self.target_exhaustive_reuse_semantics(),
            import_node_counts=imported,
        )
        for node_count in imported:
            self.tb_loss.initialize_learned_log_z(
                node_count,
                record.median_log_z_by_N[node_count],
            )
        self.historical_log_z_initialization = record
        return record

    def initialize_verified_targeted_log_z(
        self,
        artifact_path: str | Path,
    ) -> TargetedLogZInitialization:
        """Import stable medians for only the configured targets before training."""

        return load_verified_targeted_calibration_artifact(artifact_path, self)

    def _require_no_anchor_normalizer_initialized(self) -> None:
        if not self.no_anchor_mode:
            return
        assert self.tb_loss.learned_log_z_initialized_mask is not None
        missing = tuple(
            node_count
            for node_count in self.resolved_learned_node_counts
            if not bool(self.tb_loss.learned_log_z_initialized_mask[node_count - 1])
        )
        if missing:
            raise RuntimeError(
                "no-anchor learned logZ initialization is incomplete: "
                f"missing={missing}"
            )

    def _node_count_counters(self) -> tuple[dict[int, int], ...]:
        return (
            self.requested_count_by_N,
            self.sampled_attempt_count_by_N,
            self.valid_count_by_N,
            self.successful_update_count_by_N,
            self.retry_exhausted_count_by_N,
        )

    def _exact_node_retry_budget(self) -> int:
        if self.no_anchor_mode:
            return self.config.complexity.exact_node_retry_budget
        return self.config.complexity_scheduler.exact_node_retry_budget

    def _low_effective_update_rate_warning_threshold(self) -> float | None:
        if self.no_anchor_mode:
            return self.config.complexity.low_effective_update_rate_warning_threshold
        return (
            self.config.complexity_scheduler
            .low_effective_update_rate_warning_threshold
        )

    def _effective_update_rate_by_N(self) -> dict[int, float]:
        return {
            node_count: (
                self.successful_update_count_by_N[node_count] / requested
                if requested else 0.0
            )
            for node_count, requested in self.requested_count_by_N.items()
        }

    def _low_effective_update_rate_node_counts(self) -> tuple[int, ...]:
        threshold = self._low_effective_update_rate_warning_threshold()
        if threshold is None:
            return ()
        rates = self._effective_update_rate_by_N()
        return tuple(
            node_count
            for node_count in self.resolved_discovery_node_counts
            if self.requested_count_by_N[node_count] > 0
            and rates[node_count] < threshold
        )

    def _warn_low_effective_update_rates(self) -> tuple[int, ...]:
        low = self._low_effective_update_rate_node_counts()
        if low:
            warnings.warn(
                "uniform requested distribution has not translated into uniform "
                f"gradient exposure for node counts {low}",
                RuntimeWarning,
                stacklevel=2,
            )
        return low

    def _node_count_stats_fields(self) -> dict[str, object]:
        if self.complexity_scheduler is None:
            return {}
        low = self._warn_low_effective_update_rates()
        return {
            "requested_count_by_N": dict(self.requested_count_by_N),
            "sampled_attempt_count_by_N": dict(self.sampled_attempt_count_by_N),
            "valid_count_by_N": dict(self.valid_count_by_N),
            "successful_update_count_by_N": dict(
                self.successful_update_count_by_N
            ),
            "retry_exhausted_count_by_N": dict(self.retry_exhausted_count_by_N),
            "effective_update_rate_by_N": self._effective_update_rate_by_N(),
            "low_effective_update_rate_node_counts": low,
        }

    def complexity_state_dict(self) -> dict[str, Any] | None:
        if self.complexity_scheduler is None:
            return None
        state = {
            "resolved_feasible_node_counts": self.resolved_feasible_node_counts,
            "resolved_exhaustive_node_counts": self.resolved_exhaustive_node_counts,
            "resolved_discovery_node_counts": self.resolved_discovery_node_counts,
            "scheduler": self.complexity_scheduler.state_dict(),
            "requested_count_by_N": dict(self.requested_count_by_N),
            "sampled_attempt_count_by_N": dict(self.sampled_attempt_count_by_N),
            "valid_count_by_N": dict(self.valid_count_by_N),
            "successful_update_count_by_N": dict(
                self.successful_update_count_by_N
            ),
            "retry_exhausted_count_by_N": dict(self.retry_exhausted_count_by_N),
        }
        if self.no_anchor_mode:
            state["schema"] = "factor_gfn.no_anchor_complexity_state.v1"
            state["resolved_learned_node_counts"] = self.resolved_learned_node_counts
            state["normal_discovery_equals_feasible"] = True
        return state

    def load_complexity_state_dict(self, state: dict[str, Any] | None) -> None:
        if self.complexity_scheduler is None:
            if state is not None:
                raise ValueError("legacy trainer cannot load conditioned scheduler state")
            return
        if not isinstance(state, dict):
            raise ValueError("conditioned trainer checkpoint lacks scheduler state")
        names = [
            "resolved_feasible_node_counts",
            "resolved_exhaustive_node_counts",
            "resolved_discovery_node_counts",
        ]
        expected_values = [
            self.resolved_feasible_node_counts,
            self.resolved_exhaustive_node_counts,
            self.resolved_discovery_node_counts,
        ]
        if self.no_anchor_mode:
            if state.get("schema") != "factor_gfn.no_anchor_complexity_state.v1":
                raise ValueError("no-anchor complexity state schema is incompatible")
            names.append("resolved_learned_node_counts")
            expected_values.append(self.resolved_learned_node_counts)
        expected = tuple(expected_values)
        actual = tuple(tuple(state[name]) for name in names)
        if actual != expected:
            raise ValueError("checkpoint resolved F/D/E/L do not match current config")
        self.complexity_scheduler.load_state_dict(state["scheduler"])
        for name, counter in zip(
            (
                "requested_count_by_N",
                "sampled_attempt_count_by_N",
                "valid_count_by_N",
                "successful_update_count_by_N",
                "retry_exhausted_count_by_N",
            ),
            self._node_count_counters(),
            strict=True,
        ):
            values = {int(key): int(value) for key, value in state[name].items()}
            if set(values) != set(self.resolved_discovery_node_counts):
                raise ValueError(f"checkpoint {name} strata do not match")
            if any(value < 0 for value in values.values()):
                raise ValueError(f"checkpoint {name} contains negative counts")
            counter.clear()
            counter.update(values)

    def register_exact_mass_result(self, result: ExactMassResult) -> None:
        """Bind one fully audited exhaustive mass to the fixed TB buffer."""

        if not isinstance(result, ExactMassResult):
            raise TypeError("result must be ExactMassResult")
        if result.node_count not in self.resolved_exhaustive_node_counts:
            raise ValueError("exact mass can only register a resolved exhaustive stratum")
        if result.valid_candidate_count < 1 or not math.isfinite(result.exact_tb_log_z):
            raise ValueError("exact mass requires valid candidates and finite TB logZ")
        if result.reward_floor != self.config.reward.reward_floor:
            raise ValueError("exact mass reward_floor does not match GFNConfig")
        if result.provider_fingerprint != self.reward_provider.fingerprint():
            raise ValueError("exact mass provider fingerprint does not match Trainer")
        provider_manifest = self.reward_provider.manifest()
        if result.context_fingerprint != provider_manifest.get("context_fingerprint"):
            raise ValueError("exact mass context fingerprint does not match Trainer")
        current = self.registered_exact_masses_by_N.get(result.node_count)
        if current is not None and current != result:
            raise ValueError(f"exact mass for N={result.node_count} is already registered")
        self.tb_loss.set_exact_log_z(result.node_count, result.exact_tb_log_z)
        self.registered_exact_masses_by_N[result.node_count] = result

    def configure_no_anchor_exhaustive_registry(
        self,
        registry: ExhaustiveRegistry,
        *,
        source_semantics_by_N: Mapping[int, ExhaustiveReuseSemantics],
    ) -> dict[int, ExhaustiveStratumReuseProof]:
        """Verify each E stratum once, then retain structural-hash lookups."""

        if not self.no_anchor_mode:
            raise RuntimeError("registry reuse configuration requires no-anchor mode")
        if self.exhaustive_reward_lookups_by_N:
            raise RuntimeError("exhaustive registry equivalence is already verified")
        expected = set(self.resolved_exhaustive_node_counts)
        if set(source_semantics_by_N) != expected:
            raise ValueError("source semantics strata must exactly match E")
        target_semantics = self.target_exhaustive_reuse_semantics()
        proofs: dict[int, ExhaustiveStratumReuseProof] = {}
        lookups: dict[int, ProvenExhaustiveRewardLookup] = {}
        exact_results: dict[int, ExactMassResult] = {}
        for node_count in self.resolved_exhaustive_node_counts:
            target_expressions = count_canonical_terminals(
                search_space=self.config.search_space,
                target_node_count=node_count,
                canonical_count_cap=None,
            ).expressions
            proof = prove_exhaustive_stratum_reuse(
                registry,
                node_count=node_count,
                target_expressions=target_expressions,
                source_semantics=source_semantics_by_N[node_count],
                target_semantics=target_semantics,
            )
            proofs[node_count] = proof
            lookups[node_count] = ProvenExhaustiveRewardLookup(registry, proof)
            exact_results[node_count] = registry.exact_mass_result(node_count)
        # Mutate Trainer state only after every stratum has passed equivalence.
        for result in exact_results.values():
            self.register_exact_mass_result(result)
        self.exhaustive_reuse_proofs_by_N = proofs
        self.exhaustive_reward_lookups_by_N = lookups
        return dict(proofs)

    def target_exhaustive_reuse_semantics(self) -> ExhaustiveReuseSemantics:
        """Derive target semantics from the active runtime, never caller claims."""

        if not self.no_anchor_mode:
            raise RuntimeError("target reuse semantics require no-anchor mode")
        provider_manifest = self.reward_provider.manifest()
        interpreter_payload = provider_manifest.get("interpreter")
        if interpreter_payload is None:
            interpreter_payload = {
                "provider_schema": provider_manifest.get("schema"),
                "declared_interpreter": False,
            }

        def fingerprint(payload: Any) -> str:
            encoded = json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

        return ExhaustiveReuseSemantics(
            grammar_semantics_fingerprint=fingerprint(
                {
                    "state_space": state_space_fingerprint(),
                    "transition_space": transition_space_fingerprint(),
                }
            ),
            operator_semantics_fingerprint=action_space_fingerprint(),
            interpreter_semantics_fingerprint=fingerprint(interpreter_payload),
            provider_fingerprint=self.reward_provider.fingerprint(),
            data_context_fingerprint=str(provider_manifest["context_fingerprint"]),
            reward_config_fingerprint=fingerprint(asdict(self.config.reward)),
            reward_floor=float(self.config.reward.reward_floor),
        )

    def _evaluate_reward(self, expression: Expression) -> RewardAssignment:
        node_count = expression.stats.node_count
        if self.no_anchor_mode and node_count in self.resolved_exhaustive_node_counts:
            lookup = self.exhaustive_reward_lookups_by_N.get(node_count)
            if lookup is None:
                raise RuntimeError(
                    f"no-anchor exact N={node_count} lacks initialization-time "
                    "registry equivalence verification"
                )
            result = lookup.lookup(expression)
            return RewardAssignment(
                valid=result.valid,
                reward=result.reward,
                log_reward=result.log_reward,
                rejection_reason=result.rejection_reason,
                metadata=result.metadata,
            )
        assignment = self.reward_provider.evaluate(expression)
        if not isinstance(assignment, RewardAssignment):
            raise TypeError("reward_provider.evaluate() must return RewardAssignment")
        return assignment

    def _require_no_anchor_exact_ready(self) -> None:
        if not self.no_anchor_mode:
            return
        expected = set(self.resolved_exhaustive_node_counts)
        if set(self.registered_exact_masses_by_N) != expected:
            raise RuntimeError("no-anchor Trainer requires exact masses for every E stratum")
        if set(self.exhaustive_reward_lookups_by_N) != expected:
            raise RuntimeError(
                "no-anchor Trainer requires verified registry lookup for every E stratum"
            )

    def calibration_state_dict(self) -> dict[str, Any] | None:
        if self.calibration is None:
            return None
        return {
            "engine": self.calibration.state_dict(),
            "registered_exact_masses_by_N": {
                node_count: asdict(result)
                for node_count, result in self.registered_exact_masses_by_N.items()
            },
        }

    def load_calibration_state_dict(self, state: dict[str, Any] | None) -> None:
        if self.calibration is None:
            if state is not None:
                raise ValueError("calibration-disabled Trainer cannot load calibration state")
            return
        if not isinstance(state, dict):
            raise ValueError("calibration-enabled checkpoint lacks calibration state")
        exact = {
            int(key): ExactMassResult(**value)
            for key, value in state["registered_exact_masses_by_N"].items()
        }
        if not set(exact).issubset(self.resolved_exhaustive_node_counts):
            raise ValueError("checkpoint exact masses do not match resolved E")
        self.registered_exact_masses_by_N = {}
        for result in exact.values():
            self.register_exact_mass_result(result)
        self.calibration.load_state_dict(state["engine"])

    def _fixed_exact_log_z_by_N(self) -> dict[int, float]:
        if self.tb_loss.normalizer_mode != "conditional_vector":
            return {}
        assert self.tb_loss.exact_log_z_mask is not None
        assert self.tb_loss.exact_tb_log_z_by_node_count is not None
        values: dict[int, float] = {}
        for node_count in self.resolved_exhaustive_node_counts:
            index = node_count - 1
            if not bool(self.tb_loss.exact_log_z_mask[index]):
                raise RuntimeError(
                    f"exhaustive N={node_count} lacks registered exact TB logZ"
                )
            if node_count not in self.registered_exact_masses_by_N:
                raise RuntimeError(
                    f"exhaustive N={node_count} lacks audited exact mass metadata"
                )
            fixed_value = float(
                self.tb_loss.exact_tb_log_z_by_node_count[index]
            )
            if fixed_value != self.registered_exact_masses_by_N[node_count].exact_tb_log_z:
                raise RuntimeError(
                    f"exhaustive N={node_count} buffer and audited exact mass disagree"
                )
            values[node_count] = fixed_value
        return values

    def _finalize_calibration(self) -> dict[int, CalibrationStatistics]:
        if self.calibration is None:
            raise RuntimeError("normalizer calibration is disabled")
        if self.optimizer_step != 0:
            raise RuntimeError("learned logZ initialization requires optimizer_step == 0")
        if self.optimizer.state:
            raise RuntimeError("calibration cannot initialize after optimizer state exists")
        exact_values = self._fixed_exact_log_z_by_N()
        statistics = self.calibration.finalize(
            exact_tb_log_z_by_N=exact_values
        )
        exhaustive_node_counts = set(self.calibration.exhaustive_node_counts)
        for node_count in self.calibration.node_counts:
            if node_count in exhaustive_node_counts:
                continue
            self.tb_loss.initialize_learned_log_z(
                node_count,
                statistics[node_count].median,
            )
        return statistics

    def calibration_step(self) -> dict[int, CalibrationStatistics] | None:
        """Collect one balanced slot without gradients or optimizer/scheduler effects."""

        if self.calibration is None:
            raise RuntimeError("normalizer calibration is disabled")
        if self.optimizer_step != 0:
            raise RuntimeError("calibration is only allowed at optimizer_step == 0")
        require_training_only_provider_manifest(self.reward_provider.manifest())
        self._fixed_exact_log_z_by_N()
        if self.config.sampling.greedy:
            raise ValueError("calibration requires stochastic policy sampling")
        node_count = self.calibration.next_node_count()
        maximum_attempts = 1 + self._exact_node_retry_budget()
        implied_log_z: float | None = None
        sampled_attempts = 0
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for _ in range(maximum_attempts):
                    trajectory = sample_trajectories(
                        self.model,
                        self.adapter,
                        num_trajectories=1,
                        sampling_config=self.config.sampling,
                        target_node_counts=(node_count,),
                        batched_policy_diagnostics=True,
                    )[0]
                    sampled_attempts += 1
                    assignment = self._evaluate_reward(
                        trajectory.terminal_expression
                    )
                    if not assignment.valid:
                        continue
                    trajectory.attach_reward(assignment.reward, assignment.log_reward)
                    _, log_reward = trajectory.require_valid_reward()
                    implied_log_z = (
                        log_reward
                        + trajectory.sum_log_pb
                        - float(trajectory.sum_log_pf.detach())
                    )
                    if not math.isfinite(implied_log_z):
                        raise ValueError("calibration implied logZ is not finite")
                    break
        finally:
            self.model.train(was_training)
        self.calibration.record_slot(
            node_count,
            sampled_attempts=sampled_attempts,
            implied_log_z=implied_log_z,
        )
        if self.calibration.ready_to_finalize:
            return self._finalize_calibration()
        return None

    def calibration_report(self) -> dict[int, dict[str, Any]]:
        if self.calibration is None:
            return {}
        return {
            node_count: asdict(statistics)
            for node_count, statistics in self.calibration.statistics_by_N.items()
        }

    def _collect_training_batch(self):
        if self.complexity_scheduler is not None:
            return self._collect_conditioned_training_batch()
        target = self.config.training.batch_size
        limit = target * self.config.training.max_sampling_multiplier
        accepted = []
        sampled_count = 0
        invalid_count = 0
        rounds = 0
        sampling_seconds = 0.0
        reward_provider_seconds = 0.0
        while len(accepted) < target and sampled_count < limit:
            request = min(target - len(accepted), limit - sampled_count)
            rounds += 1
            sampling_started = perf_counter()
            candidates = sample_trajectories(
                self.model,
                self.adapter,
                num_trajectories=request,
                sampling_config=self.config.sampling,
                batched_policy_diagnostics=True,
            )
            sampling_seconds += perf_counter() - sampling_started
            sampled_count += len(candidates)
            for trajectory in candidates:
                reward_started = perf_counter()
                assignment = self._evaluate_reward(trajectory.terminal_expression)
                reward_provider_seconds += perf_counter() - reward_started
                if not assignment.valid:
                    invalid_count += 1
                    continue
                trajectory.attach_reward(assignment.reward, assignment.log_reward)
                accepted.append(trajectory)
        return (
            accepted,
            sampled_count,
            invalid_count,
            rounds,
            sampling_seconds,
            reward_provider_seconds,
            None,
        )

    def _collect_conditioned_training_batch(self):
        if self.complexity_scheduler is None:
            raise RuntimeError("conditioned batch collection requires scheduler")
        target = self.config.training.batch_size
        target_node_counts = self.complexity_scheduler.next_batch(target)
        for node_count in target_node_counts:
            self.requested_count_by_N[node_count] += 1
        accepted: list[Any | None] = [None] * target
        pending = list(range(target))
        sampled_count = 0
        invalid_count = 0
        rounds = 0
        sampling_seconds = 0.0
        reward_provider_seconds = 0.0
        maximum_attempts = 1 + self._exact_node_retry_budget()
        for _ in range(maximum_attempts):
            if not pending:
                break
            rounds += 1
            pending_targets = [target_node_counts[index] for index in pending]
            sampling_started = perf_counter()
            candidates = sample_trajectories(
                self.model,
                self.adapter,
                num_trajectories=len(pending),
                sampling_config=self.config.sampling,
                target_node_counts=pending_targets,
                batched_policy_diagnostics=True,
            )
            sampling_seconds += perf_counter() - sampling_started
            sampled_count += len(candidates)
            next_pending: list[int] = []
            for slot_index, node_count, trajectory in zip(
                pending, pending_targets, candidates, strict=True
            ):
                self.sampled_attempt_count_by_N[node_count] += 1
                reward_started = perf_counter()
                assignment = self._evaluate_reward(
                    trajectory.terminal_expression
                )
                reward_provider_seconds += perf_counter() - reward_started
                if not assignment.valid:
                    invalid_count += 1
                    next_pending.append(slot_index)
                    continue
                trajectory.attach_reward(assignment.reward, assignment.log_reward)
                accepted[slot_index] = trajectory
                self.valid_count_by_N[node_count] += 1
            pending = next_pending
        for slot_index in pending:
            node_count = target_node_counts[slot_index]
            self.retry_exhausted_count_by_N[node_count] += 1
        return (
            [trajectory for trajectory in accepted if trajectory is not None],
            sampled_count,
            invalid_count,
            rounds,
            sampling_seconds,
            reward_provider_seconds,
            target_node_counts,
        )

    def train_step(self) -> TrainingStats:
        if self.calibration is not None and self.calibration.status != "complete":
            raise RuntimeError("normalizer calibration must complete before training")
        self._require_no_anchor_normalizer_initialized()
        if self.config.sampling.greedy:
            raise ValueError("Trainer 禁止使用 greedy=True 进行参数更新")
        self._require_no_anchor_exact_ready()
        if self.step >= self.config.training.max_steps:
            raise RuntimeError("已达到 TrainingConfig.max_steps")
        self.model.train()
        self.last_discovery_trajectory_diagnostics = []
        (
            trajectories,
            sampled_count,
            invalid_count,
            rounds,
            sampling_seconds,
            reward_provider_seconds,
            batch_target_node_counts,
        ) = self._collect_training_batch()
        self.step += 1
        effective = len(trajectories)
        rejection_rate = invalid_count / sampled_count if sampled_count else None
        if effective < self.config.training.batch_size:
            self.optimizer.zero_grad(set_to_none=True)
            stats = TrainingStats(
                step=self.step,
                optimizer_step=self.optimizer_step,
                log_z=self._normalizer_monitor_value(),
                sampled_count=sampled_count,
                effective_batch_size=effective,
                invalid_reward_count=invalid_count,
                batch_rejection_rate=rejection_rate,
                resample_rounds=rounds,
                skipped_update=True,
                illegal_action_rate=0.0,
                **self._node_count_stats_fields(),
            )
            self.last_step_timings = {
                "sampling_seconds": sampling_seconds,
                "reward_provider_seconds": reward_provider_seconds,
                "training_update_seconds": 0.0,
                "tb_loss_forward_cuda_seconds": None,
                "backward_cuda_seconds": None,
                "optimizer_cuda_seconds": None,
            }
            self.history.append(stats)
            return stats

        training_update_started = perf_counter()
        rewards = np.asarray([trajectory.reward for trajectory in trajectories], dtype=np.float64)
        log_rewards = np.asarray([trajectory.log_reward for trajectory in trajectories], dtype=np.float64)
        lengths = np.asarray([len(trajectory.steps) for trajectory in trajectories], dtype=np.int64)
        entropies = [
            step.policy_entropy
            for trajectory in trajectories
            for step in trajectory.steps
        ]
        if any(value is None or not math.isfinite(float(value)) for value in entropies):
            raise ValueError("训练轨迹缺少有限的策略熵")
        normalized_entropies = [
            step.normalized_policy_entropy
            for trajectory in trajectories
            for step in trajectory.steps
            if step.normalized_policy_entropy is not None
        ]
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in normalized_entropies
        ):
            raise ValueError("训练轨迹包含无效的归一化策略熵")
        grouped_steps = [
            step
            for trajectory in trajectories
            for step in trajectory.steps
            if step.selected_token_group is not None
        ]
        if self.config.model.token_policy_mode in (
            "arity_hierarchical",
            "grammar_hierarchical",
        ):
            if len(grouped_steps) != int(lengths.sum()):
                raise ValueError("分组策略训练轨迹缺少完整组诊断")
            group_entropies = np.asarray(
                [float(step.group_entropy) for step in grouped_steps],
                dtype=np.float64,
            )
            normalized_group_entropies = np.asarray(
                [float(step.normalized_group_entropy) for step in grouped_steps],
                dtype=np.float64,
            )
            group_probabilities = np.asarray(
                [step.group_probabilities for step in grouped_steps],
                dtype=np.float64,
            )
            selected_groups = np.asarray(
                [int(step.selected_token_group) for step in grouped_steps],
                dtype=np.int64,
            )
            if (
                not np.isfinite(group_entropies).all()
                or not np.isfinite(normalized_group_entropies).all()
                or not np.isfinite(group_probabilities).all()
            ):
                raise ValueError("分组策略诊断出现非有限值")
        else:
            group_entropies = None
            normalized_group_entropies = None
            group_probabilities = None
            selected_groups = None

        grammar_steps = [
            step
            for trajectory in trajectories
            for step in trajectory.steps
            if step.selected_grammar_category is not None
        ]
        if self.config.model.token_policy_mode == "grammar_hierarchical":
            if len(grammar_steps) != int(lengths.sum()):
                raise ValueError("完整文法策略训练轨迹缺少分层诊断")
            grammar_category_probabilities = np.asarray(
                [step.grammar_category_probabilities for step in grammar_steps],
                dtype=np.float64,
            )
            selected_categories = np.asarray(
                [int(step.selected_grammar_category) for step in grammar_steps],
                dtype=np.int64,
            )
            grammar_category_entropies = np.asarray(
                [float(step.grammar_category_entropy) for step in grammar_steps],
                dtype=np.float64,
            )
            normalized_grammar_category_entropies = np.asarray(
                [float(step.normalized_grammar_category_entropy) for step in grammar_steps],
                dtype=np.float64,
            )
            operator_entropies = np.asarray(
                [float(step.operator_entropy) for step in grammar_steps],
                dtype=np.float64,
            )
            normalized_operator_entropies = np.asarray(
                [float(step.normalized_operator_entropy) for step in grammar_steps],
                dtype=np.float64,
            )
            window_steps = [
                step for step in grammar_steps if step.window_probabilities is not None
            ]
            window_probabilities = np.asarray(
                [step.window_probabilities for step in window_steps], dtype=np.float64
            )
            window_entropies = np.asarray(
                [float(step.window_entropy) for step in window_steps], dtype=np.float64
            )
            normalized_window_entropies = np.asarray(
                [float(step.normalized_window_entropy) for step in window_steps],
                dtype=np.float64,
            )
            selected_windows = np.asarray(
                [WINDOWS.index(get_action(step.selected_token_id).window) for step in window_steps],
                dtype=np.int64,
            )
            diagnostics = (
                grammar_category_probabilities,
                grammar_category_entropies,
                normalized_grammar_category_entropies,
                operator_entropies,
                normalized_operator_entropies,
                window_probabilities,
                window_entropies,
                normalized_window_entropies,
            )
            if any(not np.isfinite(values).all() for values in diagnostics):
                raise ValueError("完整文法策略诊断出现非有限值")
        else:
            grammar_category_probabilities = None
            selected_categories = None
            grammar_category_entropies = None
            normalized_grammar_category_entropies = None
            operator_entropies = None
            normalized_operator_entropies = None
            window_probabilities = None
            window_entropies = None
            normalized_window_entropies = None
            selected_windows = None
            window_steps = []

        self.optimizer.zero_grad(set_to_none=True)
        use_cuda_events = self.device.type == "cuda"
        if use_cuda_events:
            loss_start_event = torch.cuda.Event(enable_timing=True)
            loss_end_event = torch.cuda.Event(enable_timing=True)
            backward_end_event = torch.cuda.Event(enable_timing=True)
            optimizer_start_event = torch.cuda.Event(enable_timing=True)
            optimizer_end_event = torch.cuda.Event(enable_timing=True)
            loss_start_event.record()
        output = self.tb_loss(trajectories)
        selected_log_z = output.log_z.detach().cpu().reshape(-1).tolist()
        if len(selected_log_z) == 1 and len(trajectories) > 1:
            selected_log_z *= len(trajectories)
        if len(selected_log_z) != len(trajectories):
            raise RuntimeError("TB diagnostic logZ selection length mismatch")
        deltas = output.deltas.detach().cpu().tolist()
        self.last_discovery_trajectory_diagnostics = [
            {
                "source": "discovery",
                "target_node_count": (
                    None
                    if trajectory.target_node_count is None
                    else int(trajectory.target_node_count)
                ),
                "terminal_node_count": int(trajectory.terminal_expression.stats.node_count),
                "terminal_depth": int(trajectory.terminal_expression.stats.depth),
                "structural_hash": trajectory.terminal_expression.structural_hash(),
                "reward": float(trajectory.reward),
                "log_reward": float(trajectory.log_reward),
                "sum_log_pf": float(trajectory.sum_log_pf.detach().cpu()),
                "sum_log_pb": float(trajectory.sum_log_pb),
                "selected_log_z": float(log_z),
                "tb_delta": float(delta),
            }
            for trajectory, log_z, delta in zip(
                trajectories, selected_log_z, deltas, strict=True
            )
        ]
        if use_cuda_events:
            loss_end_event.record()
        output.loss.backward()
        if use_cuda_events:
            backward_end_event.record()
        model_parameters = list(self.model.parameters())
        model_gradient_norm = self._gradient_l2_norm(model_parameters)
        normalizer_parameters = list(self.tb_loss.parameters())
        log_z_gradient_tensor = normalizer_parameters[0].grad
        if (
            log_z_gradient_tensor is None
            and self.tb_loss.normalizer_mode == "legacy_scalar"
        ):
            raise RuntimeError("TB Loss 反传后 logZ 缺少梯度")
        log_z_gradient = self._gradient_l2_norm(normalizer_parameters)
        model_before = [parameter.detach().clone() for parameter in model_parameters]
        model_parameter_norm = self._tensor_l2_norm(model_before)
        normalizer_before = [
            parameter.detach().clone() for parameter in normalizer_parameters
        ]
        active_learned_indices = self.tb_loss.active_learned_indices(trajectories)
        model_gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model_parameters,
            max_norm=self.config.training.model_gradient_clip_norm,
            error_if_nonfinite=True,
        )
        log_z_gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            normalizer_parameters,
            max_norm=self.config.training.log_z_gradient_clip_norm,
            error_if_nonfinite=True,
        )
        model_gradient_norm_value = float(model_gradient_norm_tensor)
        log_z_gradient_norm_value = float(log_z_gradient_norm_tensor)
        gradient_norm_value = math.hypot(
            model_gradient_norm_value,
            log_z_gradient_norm_value,
        )
        model_gradient_clip_coefficient = min(
            1.0,
            self.config.training.model_gradient_clip_norm
            / (model_gradient_norm_value + 1e-6),
        )
        log_z_gradient_clip_coefficient = min(
            1.0,
            self.config.training.log_z_gradient_clip_norm
            / (log_z_gradient_norm_value + 1e-6),
        )
        if use_cuda_events:
            optimizer_start_event.record()
        self.optimizer.step()
        if self.tb_loss.normalizer_mode == "conditional_vector":
            assert self.tb_loss.log_z_by_node_count is not None
            active = set(active_learned_indices)
            with torch.no_grad():
                for index in range(self.tb_loss.log_z_by_node_count.numel()):
                    if index not in active:
                        self.tb_loss.log_z_by_node_count[index].copy_(
                            normalizer_before[0][index]
                        )
        if use_cuda_events:
            optimizer_end_event.record()
        model_parameter_update_norm = self._parameter_update_l2_norm(
            model_parameters,
            model_before,
        )
        model_relative_update_norm = (
            model_parameter_update_norm / model_parameter_norm
            if model_parameter_norm > 0.0
            else None
        )
        normalizer_update_norm = self._parameter_update_l2_norm(
            normalizer_parameters,
            normalizer_before,
        )
        log_z_after = self._normalizer_monitor_value()
        if use_cuda_events:
            optimizer_end_event.synchronize()
            tb_loss_forward_cuda_seconds = (
                loss_start_event.elapsed_time(loss_end_event) / 1000.0
            )
            backward_cuda_seconds = (
                loss_end_event.elapsed_time(backward_end_event) / 1000.0
            )
            optimizer_cuda_seconds = (
                optimizer_start_event.elapsed_time(optimizer_end_event) / 1000.0
            )
        else:
            tb_loss_forward_cuda_seconds = None
            backward_cuda_seconds = None
            optimizer_cuda_seconds = None
        training_update_seconds = perf_counter() - training_update_started
        self.last_step_timings = {
            "sampling_seconds": sampling_seconds,
            "reward_provider_seconds": reward_provider_seconds,
            "training_update_seconds": training_update_seconds,
            "tb_loss_forward_cuda_seconds": tb_loss_forward_cuda_seconds,
            "backward_cuda_seconds": backward_cuda_seconds,
            "optimizer_cuda_seconds": optimizer_cuda_seconds,
        }
        loss_value = float(output.loss.detach())
        delta_mean = float(output.delta_mean.detach())
        delta_std = float(output.delta_std.detach())
        delta_rms = math.sqrt(max(loss_value, 0.0))
        delta_mean_square_ratio = (
            delta_mean * delta_mean / loss_value if loss_value > 0.0 else None
        )
        delta_std_square_ratio = (
            delta_std * delta_std / loss_value if loss_value > 0.0 else None
        )
        self.optimizer_step += 1
        if batch_target_node_counts is not None:
            for node_count in batch_target_node_counts:
                self.successful_update_count_by_N[node_count] += 1
        hashes = {
            trajectory.terminal_expression.structural_hash() for trajectory in trajectories
        }
        stats = TrainingStats(
            step=self.step,
            optimizer_step=self.optimizer_step,
            loss=loss_value,
            log_z=log_z_after,
            reward_mean=float(rewards.mean()),
            reward_median=float(np.median(rewards)),
            log_reward_mean=float(log_rewards.mean()),
            expression_unique_rate=len(hashes) / effective,
            trajectory_length_mean=float(lengths.mean()),
            trajectory_length_max=int(lengths.max()),
            terminal_node_count_p50=float(np.quantile(lengths, 0.50)),
            terminal_node_count_p90=float(np.quantile(lengths, 0.90)),
            max_node_terminal_rate=float(
                np.mean(lengths == self.config.search_space.max_nodes)
            ),
            policy_entropy_mean=float(np.mean(entropies)),
            policy_entropy_normalized_mean=(
                float(np.mean(normalized_entropies)) if normalized_entropies else None
            ),
            group_entropy_mean=(
                float(group_entropies.mean())
                if group_entropies is not None else None
            ),
            group_entropy_normalized_mean=(
                float(normalized_group_entropies.mean())
                if normalized_group_entropies is not None else None
            ),
            leaf_group_probability_mean=(
                float(group_probabilities[:, 0].mean())
                if group_probabilities is not None else None
            ),
            unary_group_probability_mean=(
                float(group_probabilities[:, 1].mean())
                if group_probabilities is not None else None
            ),
            binary_group_probability_mean=(
                float(group_probabilities[:, 2].mean())
                if group_probabilities is not None else None
            ),
            leaf_action_rate=(
                float(np.mean(selected_groups == 0))
                if selected_groups is not None else None
            ),
            unary_action_rate=(
                float(np.mean(selected_groups == 1))
                if selected_groups is not None else None
            ),
            binary_action_rate=(
                float(np.mean(selected_groups == 2))
                if selected_groups is not None else None
            ),
            grammar_category_entropy_mean=(
                float(grammar_category_entropies.mean())
                if grammar_category_entropies is not None else None
            ),
            grammar_category_entropy_normalized_mean=(
                float(normalized_grammar_category_entropies.mean())
                if normalized_grammar_category_entropies is not None else None
            ),
            operator_entropy_mean=(
                float(operator_entropies.mean()) if operator_entropies is not None else None
            ),
            operator_entropy_normalized_mean=(
                float(normalized_operator_entropies.mean())
                if normalized_operator_entropies is not None else None
            ),
            window_entropy_mean=(
                float(window_entropies.mean()) if window_entropies is not None and window_entropies.size else None
            ),
            window_entropy_normalized_mean=(
                float(normalized_window_entropies.mean())
                if normalized_window_entropies is not None and normalized_window_entropies.size else None
            ),
            **(
                {
                    "feature_category_probability_mean": float(grammar_category_probabilities[:, 0].mean()),
                    "unary_category_probability_mean": float(grammar_category_probabilities[:, 1].mean()),
                    "ts_unary_category_probability_mean": float(grammar_category_probabilities[:, 2].mean()),
                    "binary_category_probability_mean": float(grammar_category_probabilities[:, 3].mean()),
                    "ts_binary_category_probability_mean": float(grammar_category_probabilities[:, 4].mean()),
                    "cross_sectional_category_probability_mean": float(grammar_category_probabilities[:, 5].mean()),
                    "feature_category_action_rate": float(np.mean(selected_categories == 0)),
                    "unary_category_action_rate": float(np.mean(selected_categories == 1)),
                    "ts_unary_category_action_rate": float(np.mean(selected_categories == 2)),
                    "binary_category_action_rate": float(np.mean(selected_categories == 3)),
                    "ts_binary_category_action_rate": float(np.mean(selected_categories == 4)),
                    "cross_sectional_category_action_rate": float(np.mean(selected_categories == 5)),
                    "temporal_operator_action_rate": float(len(window_steps) / len(grammar_steps)),
                }
                if grammar_category_probabilities is not None else {}
            ),
            **(
                {
                    **{
                        f"window_{window}_probability_mean": float(window_probabilities[:, index].mean())
                        for index, window in enumerate(WINDOWS)
                    },
                    **{
                        f"window_{window}_action_rate": float(np.mean(selected_windows == index))
                        for index, window in enumerate(WINDOWS)
                    },
                }
                if window_probabilities is not None and window_probabilities.size else {}
            ),
            gradient_norm=gradient_norm_value,
            tb_delta_mean=delta_mean,
            tb_delta_std=delta_std,
            tb_delta_rms=delta_rms,
            tb_delta_mean_square_ratio=delta_mean_square_ratio,
            tb_delta_std_square_ratio=delta_std_square_ratio,
            mean_log_pf=float(output.mean_log_pf.detach()),
            mean_log_pb=float(output.mean_log_pb.detach()),
            model_gradient_norm_before_clip=model_gradient_norm,
            log_z_gradient_before_clip=log_z_gradient,
            model_gradient_clip_coefficient=model_gradient_clip_coefficient,
            log_z_gradient_clip_coefficient=log_z_gradient_clip_coefficient,
            model_parameter_update_norm=model_parameter_update_norm,
            model_relative_update_norm=model_relative_update_norm,
            log_z_update=(
                float(normalizer_parameters[0].detach() - normalizer_before[0])
                if self.tb_loss.normalizer_mode == "legacy_scalar"
                else normalizer_update_norm
            ),
            sampled_count=sampled_count,
            effective_batch_size=effective,
            invalid_reward_count=invalid_count,
            batch_rejection_rate=rejection_rate,
            resample_rounds=rounds,
            skipped_update=False,
            illegal_action_rate=0.0,
            **self._node_count_stats_fields(),
        )
        self.history.append(stats)
        return stats

    def train(self, steps: int) -> list[TrainingStats]:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps 必须是正整数")
        return [self.train_step() for _ in range(steps)]

    def save_checkpoint(self, path) -> None:
        if not self.no_anchor_mode:
            raise RuntimeError(
                "legacy Trainer checkpoint writes are disabled (read-only)"
            )
        from .checkpoint import save_checkpoint

        save_checkpoint(path, self)

    def load_checkpoint(self, path) -> dict[str, Any]:
        from .checkpoint import load_checkpoint

        return load_checkpoint(path, self)


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "GFNTrainer",
    "LEGACY_TRAINER_SCHEMA",
    "RewardAssignment",
    "RewardProvider",
    "SyntheticRewardConfig",
    "SyntheticRewardProvider",
    "TRAINER_SCHEMA",
    "configure_cuda_determinism",
    "seed_everything",
]
