"""Direct minibatch log-partition variance objective for fixed-N trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .hybrid_config import (
    STAGE5_HYBRID_LPV_CONDITIONS,
)
from .trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class LogPartitionVarianceOutput:
    """Differentiable LPV loss plus per-batch diagnostic tensors."""

    loss: Tensor
    zeta: Tensor
    zeta_mean: Tensor
    zeta_std: Tensor
    zeta_variance: Tensor
    centered_zeta_rms: Tensor
    mean_log_pf: Tensor
    mean_log_pb: Tensor
    mean_log_reward: Tensor


def _require_vector(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional Tensor")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a floating-point Tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def direct_log_partition_variance(
    *,
    sum_log_pf: Tensor,
    sum_log_pb: Tensor,
    log_reward: Tensor,
) -> LogPartitionVarianceOutput:
    """Compute ``mean((zeta - mean(zeta)) ** 2)`` with the frozen gradient scale."""

    _require_vector(sum_log_pf, "sum_log_pf")
    _require_vector(sum_log_pb, "sum_log_pb")
    _require_vector(log_reward, "log_reward")
    if len(sum_log_pf) < 2:
        raise ValueError("direct LPV requires a fixed-condition batch with K >= 2")
    if sum_log_pb.shape != sum_log_pf.shape or log_reward.shape != sum_log_pf.shape:
        raise ValueError("sum_log_pf, sum_log_pb, and log_reward must share one shape")
    if sum_log_pb.device != sum_log_pf.device or log_reward.device != sum_log_pf.device:
        raise ValueError("LPV inputs must share one device")

    forward = sum_log_pf.to(dtype=torch.float64)
    backward = sum_log_pb.detach().to(dtype=torch.float64)
    rewards = log_reward.detach().to(dtype=torch.float64)
    zeta = rewards + backward - forward
    if not bool(torch.isfinite(zeta).all()):
        raise FloatingPointError("LPV zeta contains NaN or Inf")

    zeta_mean = zeta.mean()
    centered = zeta - zeta_mean
    loss = centered.square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("LPV loss contains NaN or Inf")

    zeta_variance = loss
    zeta_std = torch.sqrt(zeta_variance)
    return LogPartitionVarianceOutput(
        loss=loss,
        zeta=zeta,
        zeta_mean=zeta_mean,
        zeta_std=zeta_std,
        zeta_variance=zeta_variance,
        centered_zeta_rms=zeta_std,
        mean_log_pf=forward.mean(),
        mean_log_pb=backward.mean(),
        mean_log_reward=rewards.mean(),
    )


class LogPartitionVarianceLoss(nn.Module):
    """Validate one fixed LPV condition batch and compute the direct objective."""

    def forward(
        self,
        trajectories: Sequence[Trajectory],
    ) -> LogPartitionVarianceOutput:
        if len(trajectories) < 2:
            raise ValueError("direct LPV requires a fixed-condition batch with K >= 2")

        node_counts: list[int] = []
        sum_log_pf: list[Tensor] = []
        sum_log_pb: list[float] = []
        log_rewards: list[float] = []
        device: torch.device | None = None
        for index, trajectory in enumerate(trajectories):
            if not isinstance(trajectory, Trajectory):
                raise TypeError(f"trajectories[{index}] must be a Trajectory")
            trajectory.validate()
            trajectory.require_training_eligible()
            if trajectory.target_node_count is None:
                raise ValueError("direct LPV requires conditioned trajectories")
            node_counts.append(int(trajectory.target_node_count))

            _, log_reward = trajectory.require_valid_reward()
            forward_log_probability = trajectory.sum_log_pf
            if device is None:
                device = forward_log_probability.device
            elif forward_log_probability.device != device:
                raise ValueError("all trajectory probabilities must share one device")
            sum_log_pf.append(forward_log_probability)
            sum_log_pb.append(trajectory.sum_log_pb)
            log_rewards.append(log_reward)

        unique_node_counts = set(node_counts)
        if len(unique_node_counts) != 1:
            raise ValueError("direct LPV batch must contain exactly one fixed condition N")
        node_count = node_counts[0]
        if node_count not in STAGE5_HYBRID_LPV_CONDITIONS:
            raise ValueError("direct LPV condition N must be in 3..15")

        assert device is not None
        return direct_log_partition_variance(
            sum_log_pf=torch.stack(sum_log_pf),
            sum_log_pb=torch.tensor(sum_log_pb, dtype=torch.float64, device=device),
            log_reward=torch.tensor(log_rewards, dtype=torch.float64, device=device),
        )


__all__ = [
    "LogPartitionVarianceLoss",
    "LogPartitionVarianceOutput",
    "direct_log_partition_variance",
]
