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
    CATEGORY_TO_INDEX,
    DAGAction,
    Expression,
    ExactNodeGrammarState,
    GrammarState,
    get_action,
    state_space_fingerprint,
)


CONDITION_SCHEMA = "factor_gfn.exact_node_condition.v1"
ReplayState = GrammarState | ExactNodeGrammarState


def _validate_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} 必须是 64 位 SHA-256 十六进制字符串")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是 64 位 SHA-256 十六进制字符串") from exc


def target_condition_fingerprint(
    target_node_count: int,
    search_space_fingerprint: str,
) -> str:
    if not isinstance(target_node_count, Integral) or isinstance(target_node_count, bool):
        raise ValueError("target_node_count 必须是正整数")
    if int(target_node_count) < 1:
        raise ValueError("target_node_count 必须是正整数")
    _validate_hash(search_space_fingerprint, "search_space_fingerprint")
    payload = json.dumps(
        {
            "schema": CONDITION_SCHEMA,
            "target_node_count": int(target_node_count),
            "search_space_fingerprint": search_space_fingerprint,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def state_hash(state: ReplayState) -> str:
    """哈希完整 DAG 状态身份，包括结构和搜索空间约束。"""

    structural_state = (
        state.state if isinstance(state, ExactNodeGrammarState) else state
    )
    manifest: dict[str, object] = {
        "state_key": structural_state.state_key,
        "state_space_fingerprint": state_space_fingerprint(),
        "search_space_fingerprint": structural_state.search_space.fingerprint(),
    }
    if isinstance(state, ExactNodeGrammarState):
        manifest["condition_fingerprint"] = target_condition_fingerprint(
            state.target_node_count,
            state.search_space.fingerprint(),
        )
    payload = json.dumps(
        manifest,
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
    selected_token_group: int | None = None
    group_probabilities: tuple[float, float, float] | None = None
    group_entropy: float | None = None
    normalized_group_entropy: float | None = None
    selected_grammar_category: int | None = None
    grammar_category_probabilities: tuple[float, ...] | None = None
    grammar_category_entropy: float | None = None
    normalized_grammar_category_entropy: float | None = None
    operator_entropy: float | None = None
    normalized_operator_entropy: float | None = None
    window_probabilities: tuple[float, ...] | None = None
    window_entropy: float | None = None
    normalized_window_entropy: float | None = None

    def validate(self, *, check_tensor_values: bool = True) -> None:
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
            if not torch.is_floating_point(value):
                raise ValueError(f"{name} 必须是浮点标量 Tensor")
            if check_tensor_values and not bool(torch.isfinite(value)):
                raise ValueError(f"{name} 必须是有限浮点标量 Tensor")
            if check_tensor_values and float(value.detach()) > 1e-6:
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
        group_fields = (
            self.selected_token_group,
            self.group_probabilities,
            self.group_entropy,
            self.normalized_group_entropy,
        )
        if any(value is not None for value in group_fields):
            if any(value is None for value in group_fields):
                raise ValueError("分组策略轨迹字段必须同时存在或同时为空")
            if self.selected_token_group not in (0, 1, 2):
                raise ValueError("selected_token_group 必须属于 {0, 1, 2}")
            if (
                not isinstance(self.group_probabilities, tuple)
                or len(self.group_probabilities) != 3
                or any(
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                    for value in self.group_probabilities
                )
                or not math.isclose(
                    math.fsum(float(value) for value in self.group_probabilities),
                    1.0,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("group_probabilities 必须是和为1的三元概率")
            if self.selected_token_group != get_action(int(self.selected_token_id)).arity:
                raise ValueError("selected_token_group 与 Token 元数不一致")
            if (
                not isinstance(self.group_entropy, Real)
                or isinstance(self.group_entropy, bool)
                or not math.isfinite(float(self.group_entropy))
                or float(self.group_entropy) < -1e-10
            ):
                raise ValueError("group_entropy 必须是有限非负实数")
            if (
                not isinstance(self.normalized_group_entropy, Real)
                or isinstance(self.normalized_group_entropy, bool)
                or not math.isfinite(float(self.normalized_group_entropy))
                or not -1e-10 <= float(self.normalized_group_entropy) <= 1.0 + 1e-10
            ):
                raise ValueError("normalized_group_entropy 必须位于 [0, 1]")
        grammar_fields = (
            self.selected_grammar_category,
            self.grammar_category_probabilities,
            self.grammar_category_entropy,
            self.normalized_grammar_category_entropy,
            self.operator_entropy,
            self.normalized_operator_entropy,
        )
        if any(value is not None for value in grammar_fields):
            if any(value is None for value in grammar_fields):
                raise ValueError("完整文法策略轨迹字段必须同时存在或同时为空")
            action = get_action(int(self.selected_token_id))
            if self.selected_grammar_category != CATEGORY_TO_INDEX[action.category]:
                raise ValueError("selected_grammar_category 与 Token 类别不一致")
            if (
                not isinstance(self.grammar_category_probabilities, tuple)
                or len(self.grammar_category_probabilities) != 6
                or any(
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                    for value in self.grammar_category_probabilities
                )
                or not math.isclose(
                    math.fsum(float(value) for value in self.grammar_category_probabilities),
                    1.0,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("grammar_category_probabilities 必须是和为1的六元概率")
            for name, value in (
                ("grammar_category_entropy", self.grammar_category_entropy),
                ("operator_entropy", self.operator_entropy),
            ):
                if not isinstance(value, Real) or not math.isfinite(float(value)) or value < -1e-10:
                    raise ValueError(f"{name} 必须是有限非负实数")
            for name, value in (
                ("normalized_grammar_category_entropy", self.normalized_grammar_category_entropy),
                ("normalized_operator_entropy", self.normalized_operator_entropy),
            ):
                if not isinstance(value, Real) or not -1e-10 <= float(value) <= 1.0 + 1e-10:
                    raise ValueError(f"{name} 必须位于 [0, 1]")
            expects_window = action.window != 0
            window_fields = (
                self.window_probabilities,
                self.window_entropy,
                self.normalized_window_entropy,
            )
            if expects_window != all(value is not None for value in window_fields):
                raise ValueError("Window 诊断必须且只能出现在时序算子动作上")
            if expects_window:
                if (
                    not isinstance(self.window_probabilities, tuple)
                    or len(self.window_probabilities) != 5
                    or any(
                        not isinstance(value, Real)
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        or not 0.0 <= float(value) <= 1.0
                        for value in self.window_probabilities
                    )
                    or not math.isclose(
                        math.fsum(float(value) for value in self.window_probabilities),
                        1.0,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                ):
                    raise ValueError("window_probabilities 必须是和为1的五元概率")
                if not isinstance(self.window_entropy, Real) or not math.isfinite(float(self.window_entropy)) or self.window_entropy < -1e-10:
                    raise ValueError("window_entropy 必须是有限非负实数")
                if not isinstance(self.normalized_window_entropy, Real) or not -1e-10 <= float(self.normalized_window_entropy) <= 1.0 + 1e-10:
                    raise ValueError("normalized_window_entropy 必须位于 [0, 1]")
        elif any(
            value is not None
            for value in (
                self.window_probabilities,
                self.window_entropy,
                self.normalized_window_entropy,
            )
        ):
            raise ValueError("Window 诊断不能脱离完整文法策略字段存在")
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
    target_node_count: int | None = None
    terminal_node_count: int | None = None
    condition_fingerprint: str | None = None
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

    def validate(self, *, check_tensor_values: bool = True) -> None:
        if not self.steps:
            raise ValueError("轨迹至少需要一个步骤")
        if self.sampling_mode not in ("stochastic", "greedy"):
            raise ValueError("sampling_mode 必须是 stochastic 或 greedy")
        for step in self.steps:
            step.validate(check_tensor_values=check_tensor_values)
        for current, following in zip(self.steps, self.steps[1:]):
            if current.child_state_hash != following.state_hash:
                raise ValueError("相邻轨迹步骤的状态哈希不连续")
        if self.steps[-1].child_state_hash != self.terminal_state_hash:
            raise ValueError("最后一步没有到达 terminal_state_hash")
        if self.terminal_node_count is not None:
            if (
                not isinstance(self.terminal_node_count, Integral)
                or isinstance(self.terminal_node_count, bool)
                or int(self.terminal_node_count) < 1
            ):
                raise ValueError("terminal_node_count 必须是正整数或 None")
            if self.terminal_expression.stats.node_count != self.terminal_node_count:
                raise ValueError("terminal_node_count 与终止表达式不一致")
        conditioned = (
            self.target_node_count is not None
            or self.condition_fingerprint is not None
        )
        if conditioned:
            if self.target_node_count is None or self.condition_fingerprint is None:
                raise ValueError(
                    "conditioned 轨迹必须同时保存目标 N 与 condition fingerprint"
                )
            if self.terminal_node_count is None:
                raise ValueError("conditioned 轨迹必须保存 terminal_node_count")
            if (
                not isinstance(self.target_node_count, Integral)
                or isinstance(self.target_node_count, bool)
                or int(self.target_node_count) < 1
            ):
                raise ValueError("target_node_count 必须是正整数")
            _validate_hash(self.condition_fingerprint, "condition_fingerprint")
            if self.terminal_node_count != self.target_node_count:
                raise ValueError("terminal_node_count 必须等于 target_node_count")
        expected_pf = torch.stack([step.log_pf for step in self.steps]).sum()
        torch.testing.assert_close(self.sum_log_pf, expected_pf)
        expected_pb = math.fsum(step.log_pb for step in self.steps)
        if not math.isclose(self.sum_log_pb, expected_pb, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("sum_log_pb 与步骤求和不一致")
        if (self.reward is None) != (self.log_reward is None):
            raise ValueError("reward 与 log_reward 必须同时为空或同时存在")
        if self.reward is not None:
            self.require_valid_reward()

    def replay(self, initial_state: ReplayState) -> ReplayState:
        """按记录的规范槽位和 Token 重放轨迹并核对每个状态。"""

        self.validate()
        if self.target_node_count is None:
            if isinstance(initial_state, ExactNodeGrammarState):
                raise ValueError("无条件轨迹不能用 conditioned 初始状态重放")
            state: ReplayState = initial_state
        else:
            if isinstance(initial_state, ExactNodeGrammarState):
                if initial_state.target_node_count != self.target_node_count:
                    raise ValueError("重放初始状态的目标 N 与轨迹不一致")
                state = initial_state
            else:
                state = ExactNodeGrammarState(
                    initial_state,
                    target_node_count=self.target_node_count,
                )
            expected_condition_fingerprint = target_condition_fingerprint(
                self.target_node_count,
                state.search_space.fingerprint(),
            )
            if expected_condition_fingerprint != self.condition_fingerprint:
                raise ValueError("重放 condition fingerprint 与轨迹不一致")
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
        if self.terminal_node_count is not None and state.node_count != self.terminal_node_count:
            raise ValueError("轨迹重放得到的终止节点数不一致")
        return state


__all__ = [
    "CONDITION_SCHEMA",
    "Trajectory",
    "TrajectoryStep",
    "state_hash",
    "target_condition_fingerprint",
]
