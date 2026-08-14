"""Trajectory Balance objective and legacy/conditional normalizers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class TBLossOutput:
    loss: Tensor
    deltas: Tensor
    delta_mean: Tensor
    delta_std: Tensor
    # One selected logZ per trajectory in conditional mode; a scalar in legacy mode.
    log_z: Tensor
    mean_log_pf: Tensor
    mean_log_pb: Tensor
    mean_log_reward: Tensor


class TrajectoryBalanceLoss(nn.Module):
    """Compute ``mean((logZ_N + sum_log_pf - logR - sum_log_pb) ** 2)``."""

    def __init__(
        self,
        initial_log_z: float = 0.0,
        *,
        max_nodes: int | None = None,
        exact_node_counts: Iterable[int] = (),
    ) -> None:
        super().__init__()
        if (
            not isinstance(initial_log_z, Real)
            or isinstance(initial_log_z, bool)
            or not math.isfinite(float(initial_log_z))
        ):
            raise ValueError("initial_log_z must be a finite real number")
        initial = float(initial_log_z)
        self.max_nodes = self._validate_max_nodes(max_nodes)
        self.exact_node_counts = self._validate_exact_node_counts(exact_node_counts)
        if self.max_nodes is None:
            if self.exact_node_counts:
                raise ValueError("legacy scalar normalizer cannot declare exact strata")
            self.log_z = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
            self.register_parameter("log_z_by_node_count", None)
            self.register_buffer("exact_tb_log_z_by_node_count", None)
            self.register_buffer("exact_log_z_mask", None)
            self.register_buffer("learned_log_z_initialized_mask", None)
        else:
            self.register_parameter("log_z", None)
            self.log_z_by_node_count = nn.Parameter(
                torch.full((self.max_nodes,), initial, dtype=torch.float32)
            )
            self.register_buffer(
                "exact_tb_log_z_by_node_count",
                torch.zeros(self.max_nodes, dtype=torch.float64),
            )
            self.register_buffer(
                "exact_log_z_mask",
                torch.zeros(self.max_nodes, dtype=torch.bool),
            )
            self.register_buffer(
                "learned_log_z_initialized_mask",
                torch.zeros(self.max_nodes, dtype=torch.bool),
            )

    @staticmethod
    def _validate_max_nodes(max_nodes: int | None) -> int | None:
        if max_nodes is None:
            return None
        if not isinstance(max_nodes, Integral) or isinstance(max_nodes, bool):
            raise ValueError("max_nodes must be a positive integer")
        value = int(max_nodes)
        if value <= 0:
            raise ValueError("max_nodes must be a positive integer")
        return value

    def _validate_exact_node_counts(self, values: Iterable[int]) -> tuple[int, ...]:
        try:
            resolved = tuple(values)
        except TypeError as error:
            raise ValueError("exact_node_counts must be an iterable of integers") from error
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in resolved):
            raise ValueError("exact_node_counts must contain integers")
        normalized = tuple(sorted(int(value) for value in resolved))
        if len(set(normalized)) != len(normalized):
            raise ValueError("exact_node_counts cannot contain duplicates")
        if self.max_nodes is not None and any(
            value < 1 or value > self.max_nodes for value in normalized
        ):
            raise ValueError("exact_node_counts must lie within [1, max_nodes]")
        return normalized

    @property
    def normalizer_mode(self) -> str:
        return "legacy_scalar" if self.max_nodes is None else "conditional_vector"

    def normalizer_manifest(self) -> dict[str, object]:
        return {
            "mode": self.normalizer_mode,
            "vector_length": self.max_nodes,
            "exact_node_counts": self.exact_node_counts,
            "learned_parameter_dtype": (
                None if self.max_nodes is None else "float32"
            ),
            "exact_buffer_dtype": (
                None if self.max_nodes is None else "float64"
            ),
        }

    def normalizer_state_manifest(self) -> dict[str, object]:
        manifest = self.normalizer_manifest()
        if self.max_nodes is None:
            assert self.log_z is not None
            manifest["learned_log_z"] = float(self.log_z.detach())
            return manifest
        assert self.log_z_by_node_count is not None
        assert self.exact_tb_log_z_by_node_count is not None
        assert self.exact_log_z_mask is not None
        assert self.learned_log_z_initialized_mask is not None
        manifest.update(
            {
                "learned_log_z_by_node_count": self.log_z_by_node_count.detach()
                .cpu()
                .tolist(),
                "exact_tb_log_z_by_node_count": self.exact_tb_log_z_by_node_count.detach()
                .cpu()
                .tolist(),
                "exact_log_z_mask": self.exact_log_z_mask.detach().cpu().tolist(),
                "learned_log_z_initialized_mask": self.learned_log_z_initialized_mask.detach()
                .cpu()
                .tolist(),
            }
        )
        return manifest

    @property
    def estimated_z(self) -> Tensor:
        """Return detached Z values for monitoring only."""

        parameter = self.log_z if self.max_nodes is None else self.log_z_by_node_count
        assert parameter is not None
        return torch.exp(parameter.detach())

    def _node_count_index(self, node_count: int) -> int:
        if self.max_nodes is None:
            raise RuntimeError("node-count indexing is unavailable in legacy scalar mode")
        if not isinstance(node_count, Integral) or isinstance(node_count, bool):
            raise ValueError("target_node_count must be an integer")
        value = int(node_count)
        if value < 1 or value > self.max_nodes:
            raise ValueError("target_node_count must lie within [1, max_nodes]")
        return value - 1

    def set_exact_log_z(self, node_count: int, value: float) -> None:
        """Register an authoritative exact TB partition value for one declared stratum."""

        index = self._node_count_index(node_count)
        if int(node_count) not in self.exact_node_counts:
            raise ValueError("exact logZ can only be registered for a declared exact stratum")
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("exact logZ must be a finite real number")
        assert self.exact_tb_log_z_by_node_count is not None
        assert self.exact_log_z_mask is not None
        normalized = torch.tensor(
            float(value),
            dtype=self.exact_tb_log_z_by_node_count.dtype,
        ).item()
        if bool(self.exact_log_z_mask[index]):
            current = float(self.exact_tb_log_z_by_node_count[index])
            if current != normalized:
                raise ValueError(
                    f"exact logZ for N={int(node_count)} is already registered"
                )
            return
        with torch.no_grad():
            self.exact_tb_log_z_by_node_count[index] = normalized
            self.exact_log_z_mask[index] = True

    def initialize_learned_log_z(self, node_count: int, value: float) -> None:
        """Initialize one non-exhaustive scalar exactly once before optimization."""

        index = self._node_count_index(node_count)
        if int(node_count) in self.exact_node_counts:
            raise ValueError("exact strata cannot initialize a learned logZ scalar")
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("learned logZ initialization must be a finite real number")
        assert self.log_z_by_node_count is not None
        assert self.learned_log_z_initialized_mask is not None
        normalized = torch.tensor(
            float(value), dtype=self.log_z_by_node_count.dtype
        ).item()
        if bool(self.learned_log_z_initialized_mask[index]):
            current = float(self.log_z_by_node_count[index])
            if current != normalized:
                raise ValueError(
                    f"learned logZ for N={int(node_count)} is already initialized"
                )
            return
        with torch.no_grad():
            self.log_z_by_node_count[index] = normalized
            self.learned_log_z_initialized_mask[index] = True

    def _selected_conditional_log_z(
        self,
        trajectories: Sequence[Trajectory],
        *,
        device: torch.device,
    ) -> Tensor:
        assert self.max_nodes is not None
        assert self.log_z_by_node_count is not None
        assert self.exact_tb_log_z_by_node_count is not None
        assert self.exact_log_z_mask is not None
        selected: list[Tensor] = []
        exact = set(self.exact_node_counts)
        for index, trajectory in enumerate(trajectories):
            node_count = trajectory.target_node_count
            if node_count is None:
                raise ValueError(
                    f"trajectories[{index}] lacks target_node_count for conditional TB"
                )
            scalar_index = self._node_count_index(node_count)
            if node_count in exact:
                if not bool(self.exact_log_z_mask[scalar_index]):
                    raise RuntimeError(
                        f"exact stratum N={node_count} has no registered exact TB logZ"
                    )
                selected.append(self.exact_tb_log_z_by_node_count[scalar_index])
            else:
                selected.append(self.log_z_by_node_count[scalar_index])
        return torch.stack(selected).to(device=device, dtype=torch.float64)

    def active_learned_indices(self, trajectories: Sequence[Trajectory]) -> tuple[int, ...]:
        """Return learned vector indices used by this batch, excluding exact strata."""

        if self.max_nodes is None:
            return ()
        exact = set(self.exact_node_counts)
        indices = {
            self._node_count_index(trajectory.target_node_count)
            for trajectory in trajectories
            if trajectory.target_node_count is not None
            and trajectory.target_node_count not in exact
        }
        return tuple(sorted(indices))

    def forward(self, trajectories: Sequence[Trajectory]) -> TBLossOutput:
        if not trajectories:
            raise ValueError("TB Loss does not accept an empty batch (空 batch)")

        sum_log_pf: list[Tensor] = []
        sum_log_pb: list[float] = []
        log_rewards: list[float] = []
        parameter = self.log_z if self.max_nodes is None else self.log_z_by_node_count
        assert parameter is not None
        device = parameter.device
        for index, trajectory in enumerate(trajectories):
            if not isinstance(trajectory, Trajectory):
                raise TypeError(f"trajectories[{index}] must be a Trajectory")
            trajectory.validate()
            trajectory.require_training_eligible()
            _, log_reward = trajectory.require_valid_reward()
            forward_log_probability = trajectory.sum_log_pf
            if forward_log_probability.device != device:
                raise ValueError("trajectory probabilities and normalizer must share a device")
            if not bool(torch.isfinite(forward_log_probability)):
                raise ValueError("trajectory sum_log_pf must be finite")
            backward_log_probability = trajectory.sum_log_pb
            if not math.isfinite(backward_log_probability):
                raise ValueError("trajectory sum_log_pb must be finite")
            sum_log_pf.append(forward_log_probability)
            sum_log_pb.append(backward_log_probability)
            log_rewards.append(log_reward)

        forward = torch.stack(sum_log_pf).to(dtype=torch.float64)
        backward = torch.tensor(sum_log_pb, dtype=torch.float64, device=device)
        rewards = torch.tensor(log_rewards, dtype=torch.float64, device=device)
        if self.max_nodes is None:
            assert self.log_z is not None
            log_z = self.log_z.to(dtype=torch.float64)
        else:
            log_z = self._selected_conditional_log_z(trajectories, device=device)
        deltas = log_z + forward - rewards - backward
        if not bool(torch.isfinite(deltas).all()):
            raise FloatingPointError("TB delta contains NaN or Inf")
        loss = torch.mean(torch.square(deltas))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("TB Loss contains NaN or Inf")

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
