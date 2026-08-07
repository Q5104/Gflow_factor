"""Trajectory Balance 目标与全局可学习 ``logZ``。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import torch
from torch import Tensor, nn

from .trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class TBLossOutput:
    loss: Tensor
    deltas: Tensor
    delta_mean: Tensor
    delta_std: Tensor
    log_z: Tensor
    mean_log_pf: Tensor
    mean_log_pb: Tensor
    mean_log_reward: Tensor


class TrajectoryBalanceLoss(nn.Module):
    """计算 ``mean((logZ + ΣlogPF - logR - ΣlogPB)²)``。

    Delta 始终使用 float64 累加，但前向策略及 ``log_z`` 的梯度仍回传到
    它们原有的参数 dtype。固定均匀后向概率不参与求导。
    """

    def __init__(self, initial_log_z: float = 0.0) -> None:
        super().__init__()
        if (
            not isinstance(initial_log_z, Real)
            or isinstance(initial_log_z, bool)
            or not math.isfinite(float(initial_log_z))
        ):
            raise ValueError("initial_log_z 必须是有限实数")
        self.log_z = nn.Parameter(torch.tensor(float(initial_log_z), dtype=torch.float32))

    @property
    def estimated_z(self) -> Tensor:
        """仅供监控使用；TB 计算本身始终直接使用 ``log_z``。"""

        return torch.exp(self.log_z.detach())

    def forward(self, trajectories: Sequence[Trajectory]) -> TBLossOutput:
        if not trajectories:
            raise ValueError("TB Loss 不接受空 batch")

        sum_log_pf: list[Tensor] = []
        sum_log_pb: list[float] = []
        log_rewards: list[float] = []
        device = self.log_z.device
        for index, trajectory in enumerate(trajectories):
            if not isinstance(trajectory, Trajectory):
                raise TypeError(f"trajectories[{index}] 必须是 Trajectory")
            trajectory.validate()
            trajectory.require_training_eligible()
            _, log_reward = trajectory.require_valid_reward()
            forward_log_probability = trajectory.sum_log_pf
            if forward_log_probability.device != device:
                raise ValueError("轨迹前向概率与 log_z 必须位于同一设备")
            if not bool(torch.isfinite(forward_log_probability)):
                raise ValueError("轨迹 sum_log_pf 必须有限")
            backward_log_probability = trajectory.sum_log_pb
            if not math.isfinite(backward_log_probability):
                raise ValueError("轨迹 sum_log_pb 必须有限")
            sum_log_pf.append(forward_log_probability)
            sum_log_pb.append(backward_log_probability)
            log_rewards.append(log_reward)

        forward = torch.stack(sum_log_pf).to(dtype=torch.float64)
        backward = torch.tensor(sum_log_pb, dtype=torch.float64, device=device)
        rewards = torch.tensor(log_rewards, dtype=torch.float64, device=device)
        log_z = self.log_z.to(dtype=torch.float64)
        deltas = log_z + forward - rewards - backward
        if not bool(torch.isfinite(deltas).all()):
            raise FloatingPointError("TB delta 出现 NaN 或 Inf")
        loss = torch.mean(torch.square(deltas))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("TB Loss 出现 NaN 或 Inf")

        return TBLossOutput(
            loss=loss,
            deltas=deltas,
            delta_mean=deltas.mean(),
            delta_std=deltas.std(unbiased=False),
            log_z=log_z,
            mean_log_pf=forward.mean(),
            mean_log_pb=backward.mean(),
            mean_log_reward=rewards.mean(),
        )


__all__ = ["TBLossOutput", "TrajectoryBalanceLoss"]
