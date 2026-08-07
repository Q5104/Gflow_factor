"""GFlowNet 参数更新闭环与不依赖行情的合成 Reward。"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import torch

from factor_gfn.grammar import Expression, get_action

from .config import GFNConfig, TrainingStats
from .loss import TrajectoryBalanceLoss
from .model import ForwardPolicyNetwork
from .policy_sampler import sample_trajectories
from .state_adapter import StateAdapter


TRAINER_SCHEMA = "factor_gfn.trainer.v1"


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
        seed_everything(
            config.training.seed,
            deterministic_algorithms=config.training.deterministic_algorithms,
        )
        self.adapter = StateAdapter(config.search_space)
        self.model = ForwardPolicyNetwork(config.model, config.search_space).to(self.device)
        self.tb_loss = TrajectoryBalanceLoss(initial_log_z=0.0).to(self.device)
        training = config.training
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
        self.run_id = uuid4().hex
        self.created_at_utc = datetime.now(timezone.utc).isoformat()

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.model.parameters()) + list(self.tb_loss.parameters())

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
            },
            "parameter_semantics": {
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
        while len(accepted) < target and sampled_count < limit:
            request = min(target - len(accepted), limit - sampled_count)
            rounds += 1
            candidates = sample_trajectories(
                self.model,
                self.adapter,
                num_trajectories=request,
                sampling_config=self.config.sampling,
            )
            sampled_count += len(candidates)
            for trajectory in candidates:
                assignment = self.reward_provider.evaluate(trajectory.terminal_expression)
                if not isinstance(assignment, RewardAssignment):
                    raise TypeError("reward_provider.evaluate() 必须返回 RewardAssignment")
                if not assignment.valid:
                    invalid_count += 1
                    continue
                trajectory.attach_reward(assignment.reward, assignment.log_reward)
                accepted.append(trajectory)
        return accepted, sampled_count, invalid_count, rounds

    def train_step(self) -> TrainingStats:
        if self.config.sampling.greedy:
            raise ValueError("Trainer 禁止使用 greedy=True 进行参数更新")
        if self.step >= self.config.training.max_steps:
            raise RuntimeError("已达到 TrainingConfig.max_steps")
        self.model.train()
        trajectories, sampled_count, invalid_count, rounds = self._collect_training_batch()
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
            self.history.append(stats)
            return stats

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

        self.optimizer.zero_grad(set_to_none=True)
        output = self.tb_loss(trajectories)
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters,
            max_norm=self.config.training.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.optimizer_step += 1
        hashes = {
            trajectory.terminal_expression.structural_hash() for trajectory in trajectories
        }
        stats = TrainingStats(
            step=self.step,
            optimizer_step=self.optimizer_step,
            loss=float(output.loss.detach()),
            log_z=float(self.tb_loss.log_z.detach()),
            reward_mean=float(rewards.mean()),
            reward_median=float(np.median(rewards)),
            log_reward_mean=float(log_rewards.mean()),
            expression_unique_rate=len(hashes) / effective,
            trajectory_length_mean=float(lengths.mean()),
            trajectory_length_max=int(lengths.max()),
            policy_entropy_mean=float(np.mean(entropies)),
            policy_entropy_normalized_mean=(
                float(np.mean(normalized_entropies)) if normalized_entropies else None
            ),
            gradient_norm=float(gradient_norm),
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
    "GFNTrainer",
    "RewardAssignment",
    "RewardProvider",
    "SyntheticRewardConfig",
    "SyntheticRewardProvider",
    "TRAINER_SCHEMA",
    "seed_everything",
]
