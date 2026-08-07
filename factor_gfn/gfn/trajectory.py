"""Trajectory Balance 所需的可微轨迹数据结构。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import torch
from torch import Tensor

from factor_gfn.grammar import (
    DAGAction,
    Expression,
    GrammarState,
    get_action,
    state_space_fingerprint,
)


def _validate_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} 必须是 64 位 SHA-256 十六进制字符串")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是 64 位 SHA-256 十六进制字符串") from exc


def state_hash(state: GrammarState) -> str:
    """哈希完整 DAG 状态身份，包括结构和搜索空间约束。"""

    payload = json.dumps(
        {
            "state_key": state.state_key,
            "state_space_fingerprint": state_space_fingerprint(),
            "search_space_fingerprint": state.search_space.fingerprint(),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    state_hash: str
    selected_slot_index: int
    selected_slot_path: tuple[int, ...]
    selected_slot_orbit_key: str
    selected_token_id: int
    log_p_slot: Tensor
    log_p_token: Tensor
    log_pf: Tensor
    child_state_hash: str
    n_parents: int
    log_pb: float
    policy_entropy: float | None = None
    normalized_policy_entropy: float | None = None

    def validate(self) -> None:
        _validate_hash(self.state_hash, "state_hash")
        _validate_hash(self.child_state_hash, "child_state_hash")
        if (
            not isinstance(self.selected_slot_index, Integral)
            or isinstance(self.selected_slot_index, bool)
            or self.selected_slot_index < 0
        ):
            raise ValueError("selected_slot_index 必须是非负整数")
        if not isinstance(self.selected_slot_path, tuple) or any(
            not isinstance(index, Integral)
            or isinstance(index, bool)
            or index < 0
            for index in self.selected_slot_path
        ):
            raise ValueError("selected_slot_path 必须是非负整数元组")
        if not isinstance(self.selected_slot_orbit_key, str) or not self.selected_slot_orbit_key:
            raise ValueError("selected_slot_orbit_key 不能为空")
        if not isinstance(self.selected_token_id, Integral) or isinstance(self.selected_token_id, bool):
            raise ValueError("selected_token_id 必须是合法整数 Token ID")
        get_action(int(self.selected_token_id))
        if (
            not isinstance(self.n_parents, Integral)
            or isinstance(self.n_parents, bool)
            or self.n_parents < 1
        ):
            raise ValueError("n_parents 必须至少为 1")
        for name, value in (
            ("log_p_slot", self.log_p_slot),
            ("log_p_token", self.log_p_token),
            ("log_pf", self.log_pf),
        ):
            if not isinstance(value, Tensor) or value.ndim:
                raise ValueError(f"{name} 必须是标量 Tensor")
            if not torch.is_floating_point(value) or not bool(torch.isfinite(value)):
                raise ValueError(f"{name} 必须是有限浮点标量 Tensor")
            if float(value.detach()) > 1e-6:
                raise ValueError(f"{name} 作为 log 概率不能大于 0")
        if not isinstance(self.log_pb, Real) or isinstance(self.log_pb, bool):
            raise ValueError("log_pb 必须是实数")
        if not math.isfinite(float(self.log_pb)) or self.log_pb > 1e-12:
            raise ValueError("log_pb 必须是有限且不大于 0 的数")
        if self.policy_entropy is not None:
            if (
                not isinstance(self.policy_entropy, Real)
                or isinstance(self.policy_entropy, bool)
                or not math.isfinite(float(self.policy_entropy))
                or self.policy_entropy < -1e-10
            ):
                raise ValueError("policy_entropy 必须是有限非负实数或 None")
        if self.normalized_policy_entropy is not None:
            if (
                not isinstance(self.normalized_policy_entropy, Real)
                or isinstance(self.normalized_policy_entropy, bool)
                or not math.isfinite(float(self.normalized_policy_entropy))
                or not -1e-10 <= float(self.normalized_policy_entropy) <= 1.0 + 1e-10
            ):
                raise ValueError("normalized_policy_entropy 必须位于 [0, 1] 或为 None")
        if (
            self.log_p_slot.device != self.log_p_token.device
            or self.log_p_slot.device != self.log_pf.device
        ):
            raise ValueError("三个前向 log 概率 Tensor 必须位于同一设备")
        torch.testing.assert_close(self.log_pf, self.log_p_slot + self.log_p_token)
        if not math.isclose(self.log_pb, -math.log(self.n_parents), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("log_pb 与固定均匀后向概率不一致")


@dataclass(slots=True)
class Trajectory:
    steps: list[TrajectoryStep]
    terminal_state_hash: str
    terminal_expression: Expression
    sampling_mode: Literal["stochastic", "greedy"]
    reward: float | None = None
    log_reward: float | None = None

    @property
    def training_eligible(self) -> bool:
        return self.sampling_mode == "stochastic"

    def require_training_eligible(self) -> None:
        if not self.training_eligible:
            raise ValueError("greedy 诊断轨迹禁止用于 TB Loss 或参数更新")

    def attach_reward(self, reward: float, log_reward: float | None = None) -> None:
        """挂载经过校验的正值 Reward；不允许静默覆盖不同结果。"""

        if not isinstance(reward, Real) or isinstance(reward, bool):
            raise ValueError("reward 必须是正的有限实数")
        reward_value = float(reward)
        if not math.isfinite(reward_value) or reward_value <= 0.0:
            raise ValueError("reward 必须是正的有限实数")
        expected_log = math.log(reward_value)
        if log_reward is None:
            log_value = expected_log
        else:
            if not isinstance(log_reward, Real) or isinstance(log_reward, bool):
                raise ValueError("log_reward 必须是有限实数")
            log_value = float(log_reward)
            if not math.isfinite(log_value):
                raise ValueError("log_reward 必须是有限实数")
            if not math.isclose(log_value, expected_log, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("log_reward 必须与 log(reward) 一致")
        if self.reward is not None or self.log_reward is not None:
            if (
                self.reward is None
                or self.log_reward is None
                or not math.isclose(self.reward, reward_value, rel_tol=1e-12, abs_tol=0.0)
                or not math.isclose(self.log_reward, log_value, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError("轨迹已挂载不同的 Reward，禁止静默覆盖")
            return
        self.reward = reward_value
        self.log_reward = log_value

    def require_valid_reward(self) -> tuple[float, float]:
        """返回一致的 ``(reward, log_reward)``，否则拒绝进入训练。"""

        if self.reward is None or self.log_reward is None:
            raise ValueError("轨迹尚未挂载 Reward")
        reward_value = float(self.reward)
        log_value = float(self.log_reward)
        if not math.isfinite(reward_value) or reward_value <= 0.0:
            raise ValueError("轨迹 reward 必须是正的有限值")
        if not math.isfinite(log_value):
            raise ValueError("轨迹 log_reward 必须是有限值")
        if not math.isclose(log_value, math.log(reward_value), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("轨迹 log_reward 与 log(reward) 不一致")
        return reward_value, log_value

    @property
    def sum_log_pf(self) -> Tensor:
        if not self.steps:
            raise ValueError("轨迹至少需要一个步骤")
        return torch.stack([step.log_pf for step in self.steps]).sum()

    @property
    def sum_log_pb(self) -> float:
        return math.fsum(step.log_pb for step in self.steps)

    def validate(self) -> None:
        if not self.steps:
            raise ValueError("轨迹至少需要一个步骤")
        if self.sampling_mode not in ("stochastic", "greedy"):
            raise ValueError("sampling_mode 必须是 stochastic 或 greedy")
        for step in self.steps:
            step.validate()
        for current, following in zip(self.steps, self.steps[1:]):
            if current.child_state_hash != following.state_hash:
                raise ValueError("相邻轨迹步骤的状态哈希不连续")
        if self.steps[-1].child_state_hash != self.terminal_state_hash:
            raise ValueError("最后一步没有到达 terminal_state_hash")
        expected_pf = torch.stack([step.log_pf for step in self.steps]).sum()
        torch.testing.assert_close(self.sum_log_pf, expected_pf)
        expected_pb = math.fsum(step.log_pb for step in self.steps)
        if not math.isclose(self.sum_log_pb, expected_pb, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("sum_log_pb 与步骤求和不一致")
        if (self.reward is None) != (self.log_reward is None):
            raise ValueError("reward 与 log_reward 必须同时为空或同时存在")
        if self.reward is not None:
            self.require_valid_reward()

    def replay(self, initial_state: GrammarState) -> GrammarState:
        """按记录的规范槽位和 Token 重放轨迹并核对每个状态。"""

        self.validate()
        state = initial_state
        for step in self.steps:
            if state_hash(state) != step.state_hash:
                raise ValueError("重放时当前状态哈希与轨迹记录不一致")
            slots = state.open_slots()
            if step.selected_slot_index >= len(slots):
                raise ValueError("重放时槽位索引越界")
            slot = slots[step.selected_slot_index]
            if slot.path != step.selected_slot_path or slot.orbit_key != step.selected_slot_orbit_key:
                raise ValueError("重放时规范槽位身份与轨迹记录不一致")
            state = state.step(DAGAction(slot.path, step.selected_token_id))
            if state_hash(state) != step.child_state_hash:
                raise ValueError("重放后的子状态哈希与轨迹记录不一致")
            actual_n_parents = state.count_parents()
            if actual_n_parents != step.n_parents:
                raise ValueError("重放得到的真实父状态数与轨迹记录不一致")
            if not math.isclose(
                state.log_backward_probability(),
                step.log_pb,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("重放得到的固定后向概率与轨迹记录不一致")
        if not state.done or state_hash(state) != self.terminal_state_hash:
            raise ValueError("轨迹重放没有到达记录的终止状态")
        if state.to_expression().structural_hash() != self.terminal_expression.structural_hash():
            raise ValueError("轨迹重放得到的终止表达式不一致")
        return state


__all__ = ["Trajectory", "TrajectoryStep", "state_hash"]
