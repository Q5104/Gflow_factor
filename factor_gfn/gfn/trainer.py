"""GFlowNet 参数更新闭环与不依赖行情的合成 Reward。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import torch

from factor_gfn.grammar import WINDOWS, Expression, get_action

from .config import GFNConfig, TrainingStats
from .loss import TrajectoryBalanceLoss
from .model import ForwardPolicyNetwork
from .policy_sampler import sample_trajectories
from .state_adapter import StateAdapter


TRAINER_SCHEMA = "factor_gfn.trainer.v1"
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
            "schema": "factor_gfn.synthetic_reward.v1",
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
        config: GFNConfig,
        reward_provider: RewardProvider,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(config, GFNConfig):
            raise TypeError("config 必须是 GFNConfig")
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
        self.adapter = StateAdapter(config.search_space)
        self.model = ForwardPolicyNetwork(config.model, config.search_space).to(self.device)
        training = config.training
        self.tb_loss = TrajectoryBalanceLoss(
            initial_log_z=training.initial_log_z
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters(), "lr": training.learning_rate},
                {"params": self.tb_loss.parameters(), "lr": training.log_z_learning_rate},
            ],
            betas=(training.optimizer_beta1, training.optimizer_beta2),
            eps=training.optimizer_eps,
            weight_decay=training.weight_decay,
        )
        self.step = 0
        self.optimizer_step = 0
        self.history: list[TrainingStats] = []
        self.last_step_timings: dict[str, float | None] = {}
        self.run_id = uuid4().hex
        self.created_at_utc = datetime.now(timezone.utc).isoformat()

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.model.parameters()) + list(self.tb_loss.parameters())

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
        return {
            "schema": TRAINER_SCHEMA,
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
        }

    def _collect_training_batch(self):
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
                assignment = self.reward_provider.evaluate(trajectory.terminal_expression)
                reward_provider_seconds += perf_counter() - reward_started
                if not isinstance(assignment, RewardAssignment):
                    raise TypeError("reward_provider.evaluate() 必须返回 RewardAssignment")
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
        )

    def train_step(self) -> TrainingStats:
        if self.config.sampling.greedy:
            raise ValueError("Trainer 禁止使用 greedy=True 进行参数更新")
        if self.step >= self.config.training.max_steps:
            raise RuntimeError("已达到 TrainingConfig.max_steps")
        self.model.train()
        (
            trajectories,
            sampled_count,
            invalid_count,
            rounds,
            sampling_seconds,
            reward_provider_seconds,
        ) = self._collect_training_batch()
        self.step += 1
        effective = len(trajectories)
        rejection_rate = invalid_count / sampled_count if sampled_count else None
        if effective < self.config.training.batch_size:
            self.optimizer.zero_grad(set_to_none=True)
            stats = TrainingStats(
                step=self.step,
                optimizer_step=self.optimizer_step,
                log_z=float(self.tb_loss.log_z.detach()),
                sampled_count=sampled_count,
                effective_batch_size=effective,
                invalid_reward_count=invalid_count,
                batch_rejection_rate=rejection_rate,
                resample_rounds=rounds,
                skipped_update=True,
                illegal_action_rate=0.0,
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
        if use_cuda_events:
            loss_end_event.record()
        output.loss.backward()
        if use_cuda_events:
            backward_end_event.record()
        model_parameters = list(self.model.parameters())
        model_gradient_norm = self._gradient_l2_norm(model_parameters)
        log_z_gradient_tensor = self.tb_loss.log_z.grad
        if log_z_gradient_tensor is None:
            raise RuntimeError("TB Loss 反传后 logZ 缺少梯度")
        log_z_gradient = float(log_z_gradient_tensor.detach())
        model_before = [parameter.detach().clone() for parameter in model_parameters]
        model_parameter_norm = self._tensor_l2_norm(model_before)
        log_z_before = float(self.tb_loss.log_z.detach())
        model_gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model_parameters,
            max_norm=self.config.training.model_gradient_clip_norm,
            error_if_nonfinite=True,
        )
        log_z_parameters = list(self.tb_loss.parameters())
        log_z_gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            log_z_parameters,
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
        log_z_after = float(self.tb_loss.log_z.detach())
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
            log_z_update=log_z_after - log_z_before,
            sampled_count=sampled_count,
            effective_batch_size=effective,
            invalid_reward_count=invalid_count,
            batch_rejection_rate=rejection_rate,
            resample_rounds=rounds,
            skipped_update=False,
            illegal_action_rate=0.0,
        )
        self.history.append(stats)
        return stats

    def train(self, steps: int) -> list[TrainingStats]:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps 必须是正整数")
        return [self.train_step() for _ in range(steps)]

    def save_checkpoint(self, path) -> None:
        from .checkpoint import save_checkpoint

        save_checkpoint(path, self)

    def load_checkpoint(self, path) -> dict[str, Any]:
        from .checkpoint import load_checkpoint

        return load_checkpoint(path, self)


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "GFNTrainer",
    "RewardAssignment",
    "RewardProvider",
    "SyntheticRewardConfig",
    "SyntheticRewardProvider",
    "TRAINER_SCHEMA",
    "configure_cuda_determinism",
    "seed_everything",
]
