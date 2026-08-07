"""在阶段二 DAG 状态机之上按前向策略采样可微训练轨迹。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from factor_gfn.grammar import DAGAction, GrammarState

from .config import SamplingConfig
from .model import ForwardPolicyNetwork
from .state_adapter import StateAdapter
from .trajectory import Trajectory, TrajectoryStep, state_hash


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _select(log_probs: torch.Tensor, *, greedy: bool) -> int:
    if log_probs.ndim != 1:
        raise ValueError("动作 log_probs 必须是一维")
    if greedy:
        return int(torch.argmax(log_probs).item())
    probabilities = log_probs.detach().exp()
    return int(torch.multinomial(probabilities, num_samples=1).item())


def _joint_policy_entropy(
    slot_log_probs: torch.Tensor,
    token_log_probs: torch.Tensor,
    slot_mask: torch.Tensor,
    legal_token_mask: torch.Tensor,
) -> float:
    """精确计算 ``H(slot) + E_slot[H(token|slot)]``。"""

    def entropy_from_logs(logs: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(logs)
        if not bool(finite.any()):
            raise FloatingPointError("合法动作分布没有有限 log 概率")
        finite_logs = logs[finite]
        return -(finite_logs.exp() * finite_logs).sum()

    valid_slot_log_probs = slot_log_probs[slot_mask]
    slot_probabilities = valid_slot_log_probs.exp()
    slot_entropy = entropy_from_logs(valid_slot_log_probs)
    token_entropies = []
    for probability, slot_index in zip(
        slot_probabilities,
        torch.nonzero(slot_mask, as_tuple=False).flatten().tolist(),
    ):
        if float(probability.detach()) == 0.0:
            token_entropies.append(torch.zeros_like(probability))
            continue
        valid = legal_token_mask[slot_index]
        logs = token_log_probs[slot_index, valid]
        token_entropies.append(entropy_from_logs(logs))
    conditional_entropy = torch.stack(token_entropies)
    joint = slot_entropy + torch.dot(slot_probabilities, conditional_entropy)
    if not bool(torch.isfinite(joint)) or float(joint.detach()) < -1e-10:
        raise FloatingPointError("联合策略熵出现无效值")
    return max(0.0, float(joint.detach()))


def _normalized_joint_policy_entropy(
    entropy: float,
    slot_mask: torch.Tensor,
    legal_token_mask: torch.Tensor,
) -> float | None:
    """按当前状态的合法 ``(slot, token)`` 联合动作数归一化策略熵。

    仅有一个合法动作时最大熵为 0，比值无定义，因此返回 ``None``
    并在 batch 均值中排除该强制动作状态。
    """

    legal_joint_actions = int(legal_token_mask[slot_mask].sum().item())
    if legal_joint_actions < 1:
        raise FloatingPointError("当前状态没有合法联合动作")
    if legal_joint_actions == 1:
        return None
    normalized = entropy / math.log(legal_joint_actions)
    if not math.isfinite(normalized) or not -1e-8 <= normalized <= 1.0 + 1e-6:
        raise FloatingPointError("归一化策略熵超出合法范围")
    return min(1.0, max(0.0, normalized))


def sample_trajectories(
    model: ForwardPolicyNetwork,
    adapter: StateAdapter,
    *,
    num_trajectories: int,
    sampling_config: SamplingConfig = SamplingConfig(),
    initial_states: Sequence[GrammarState] | None = None,
) -> list[Trajectory]:
    """批量推进所有未终止状态，并保留前向概率的计算图。"""

    if isinstance(num_trajectories, bool) or not isinstance(num_trajectories, int) or num_trajectories < 1:
        raise ValueError("num_trajectories 必须是正整数")
    if initial_states is None:
        states = [
            GrammarState(
                search_space=adapter.search_space,
            )
            for _ in range(num_trajectories)
        ]
    else:
        if len(initial_states) != num_trajectories:
            raise ValueError("initial_states 数量必须等于 num_trajectories")
        states = list(initial_states)
        if any(state.done for state in states):
            raise ValueError("初始状态必须全部为非终止状态")

    device = _model_device(model)
    step_lists: list[list[TrajectoryStep]] = [[] for _ in states]
    active = list(range(num_trajectories))
    was_training = model.training
    model.eval()  # PF 必须是给定状态的确定函数；不把 dropout mask 当成隐藏状态。
    try:
        while active:
            if any(len(step_lists[index]) >= adapter.search_space.max_nodes for index in active):
                raise RuntimeError("轨迹超过 max_nodes 仍未终止")
            active_states = [states[index] for index in active]
            batch = adapter.batch(active_states).to(device)
            output = model(batch, temperature=float(sampling_config.temperature))
            next_active: list[int] = []

            for row, trajectory_index in enumerate(active):
                state = states[trajectory_index]
                slot_index = _select(output.slot_log_probs[row], greedy=sampling_config.greedy)
                if not bool(batch.slot_mask[row, slot_index]):
                    raise RuntimeError("策略采样到了非法槽位")
                token_id = _select(
                    output.token_log_probs[row, slot_index],
                    greedy=sampling_config.greedy,
                )
                if not bool(batch.legal_token_mask[row, slot_index, token_id]):
                    raise RuntimeError("策略采样到了非法 Token")

                slot = batch.open_slots[row][slot_index]
                log_p_slot = output.slot_log_probs[row, slot_index]
                log_p_token = output.token_log_probs[row, slot_index, token_id]
                policy_entropy = _joint_policy_entropy(
                    output.slot_log_probs[row],
                    output.token_log_probs[row],
                    batch.slot_mask[row],
                    batch.legal_token_mask[row],
                )
                normalized_policy_entropy = _normalized_joint_policy_entropy(
                    policy_entropy,
                    batch.slot_mask[row],
                    batch.legal_token_mask[row],
                )
                child = state.step(DAGAction(slot.path, token_id))
                n_parents = child.count_parents()
                log_pb = -math.log(n_parents)
                step_lists[trajectory_index].append(
                    TrajectoryStep(
                        state_hash=state_hash(state),
                        selected_slot_index=slot_index,
                        selected_slot_path=slot.path,
                        selected_slot_orbit_key=slot.orbit_key,
                        selected_token_id=token_id,
                        log_p_slot=log_p_slot,
                        log_p_token=log_p_token,
                        log_pf=log_p_slot + log_p_token,
                        child_state_hash=state_hash(child),
                        n_parents=n_parents,
                        log_pb=log_pb,
                        policy_entropy=policy_entropy,
                        normalized_policy_entropy=normalized_policy_entropy,
                    )
                )
                states[trajectory_index] = child
                if not child.done:
                    next_active.append(trajectory_index)
            active = next_active
    finally:
        model.train(was_training)

    trajectories = [
        Trajectory(
            steps=step_lists[index],
            terminal_state_hash=state_hash(state),
            terminal_expression=state.to_expression(),
            sampling_mode="greedy" if sampling_config.greedy else "stochastic",
        )
        for index, state in enumerate(states)
    ]
    for trajectory in trajectories:
        trajectory.validate()
    return trajectories


def sample_trajectory(
    model: ForwardPolicyNetwork,
    adapter: StateAdapter,
    *,
    sampling_config: SamplingConfig = SamplingConfig(),
    initial_state: GrammarState | None = None,
) -> Trajectory:
    initial_states = None if initial_state is None else [initial_state]
    return sample_trajectories(
        model,
        adapter,
        num_trajectories=1,
        sampling_config=sampling_config,
        initial_states=initial_states,
    )[0]


__all__ = ["sample_trajectories", "sample_trajectory"]
