"""在阶段二 DAG 状态机之上按前向策略采样可微训练轨迹。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import torch

from factor_gfn.grammar import (
    CATEGORY_TO_INDEX,
    OPERATOR_TO_INDEX,
    WINDOWS,
    DAGAction,
    ExactNodeGrammarState,
    GrammarState,
    get_action,
)

from .config import SamplingConfig
from .model import ForwardPolicyNetwork
from .state_adapter import PolicyGrammarState, StateAdapter
from .trajectory import (
    Trajectory,
    TrajectoryStep,
    state_hash,
    target_condition_fingerprint,
)


_COMPACT_DIAGNOSTIC_SIZE = 24


def _masked_entropy_tensor(logs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe_logs = logs.masked_fill(~mask, 0.0)
    return -(
        safe_logs.exp() * safe_logs * mask.to(dtype=safe_logs.dtype)
    ).sum()


def _compact_policy_diagnostics(
    output,
    row: int,
    slot_index: int,
    token_id: int,
    slot_mask: torch.Tensor,
    legal_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep hot-path diagnostics on device until the sampled batch is complete."""

    dtype = output.slot_log_probs.dtype
    device = output.slot_log_probs.device
    nan = torch.full((), torch.nan, dtype=dtype, device=device)
    nan3 = torch.full((3,), torch.nan, dtype=dtype, device=device)
    nan5 = torch.full((5,), torch.nan, dtype=dtype, device=device)
    nan6 = torch.full((6,), torch.nan, dtype=dtype, device=device)

    slot_logs = output.slot_log_probs[row]
    token_logs = output.token_log_probs[row]
    safe_slot_logs = slot_logs.masked_fill(~slot_mask, 0.0)
    slot_probabilities = safe_slot_logs.exp() * slot_mask.to(dtype=dtype)
    slot_entropy = _masked_entropy_tensor(slot_logs, slot_mask)
    safe_token_logs = token_logs.masked_fill(~legal_token_mask, 0.0)
    token_entropies = -(
        safe_token_logs.exp()
        * safe_token_logs
        * legal_token_mask.to(dtype=dtype)
    ).sum(dim=1)
    policy_entropy = slot_entropy + torch.dot(slot_probabilities, token_entropies)
    legal_joint_count = legal_token_mask[slot_mask].sum().to(dtype=dtype)
    normalized_policy_entropy = torch.where(
        legal_joint_count > 1,
        policy_entropy / torch.log(legal_joint_count.clamp_min(2)),
        nan,
    )

    if output.group_log_probs is None or output.legal_group_mask is None:
        group_probabilities = nan3
        group_entropy = nan
        normalized_group_entropy = nan
    else:
        group_logs = output.group_log_probs[row, slot_index]
        group_legal = output.legal_group_mask[row, slot_index]
        group_probabilities = (
            group_logs.masked_fill(~group_legal, 0.0).exp()
            * group_legal.to(dtype=dtype)
        )
        group_entropy = _masked_entropy_tensor(group_logs, group_legal)
        group_count = group_legal.sum().to(dtype=dtype)
        normalized_group_entropy = torch.where(
            group_count > 1,
            group_entropy / torch.log(group_count.clamp_min(2)),
            torch.zeros((), dtype=dtype, device=device),
        )

    if output.grammar_category_log_probs is None:
        grammar_probabilities = nan6
        grammar_entropy = nan
        normalized_grammar_entropy = nan
        operator_entropy = nan
        normalized_operator_entropy = nan
        window_probabilities = nan5
        window_entropy = nan
        normalized_window_entropy = nan
    else:
        action = get_action(token_id)
        category_id = CATEGORY_TO_INDEX[action.category]
        category_logs = output.grammar_category_log_probs[row, slot_index]
        category_legal = output.legal_grammar_category_mask[row, slot_index]
        grammar_probabilities = (
            category_logs.masked_fill(~category_legal, 0.0).exp()
            * category_legal.to(dtype=dtype)
        )
        grammar_entropy = _masked_entropy_tensor(category_logs, category_legal)
        category_count = category_legal.sum().to(dtype=dtype)
        normalized_grammar_entropy = torch.where(
            category_count > 1,
            grammar_entropy / torch.log(category_count.clamp_min(2)),
            torch.zeros((), dtype=dtype, device=device),
        )

        operator_id = OPERATOR_TO_INDEX[action.name]
        operator_logs = output.operator_log_probs[row, slot_index]
        operator_legal = output.legal_operator_mask[row, slot_index]
        category_operator_mask = output.operator_category_lookup == category_id
        selected_operator_mask = operator_legal & category_operator_mask
        operator_entropy = _masked_entropy_tensor(
            operator_logs, selected_operator_mask
        )
        operator_count = selected_operator_mask.sum().to(dtype=dtype)
        normalized_operator_entropy = torch.where(
            operator_count > 1,
            operator_entropy / torch.log(operator_count.clamp_min(2)),
            torch.zeros((), dtype=dtype, device=device),
        )

        if action.window:
            window_logs = output.window_log_probs[row, slot_index, operator_id]
            window_legal = output.legal_window_mask[row, slot_index, operator_id]
            window_probabilities = (
                window_logs.masked_fill(~window_legal, 0.0).exp()
                * window_legal.to(dtype=dtype)
            )
            window_entropy = _masked_entropy_tensor(window_logs, window_legal)
            window_count = window_legal.sum().to(dtype=dtype)
            normalized_window_entropy = torch.where(
                window_count > 1,
                window_entropy / torch.log(window_count.clamp_min(2)),
                torch.zeros((), dtype=dtype, device=device),
            )
        else:
            window_probabilities = nan5
            window_entropy = nan
            normalized_window_entropy = nan

    compact = torch.cat(
        (
            policy_entropy.reshape(1),
            normalized_policy_entropy.reshape(1),
            group_probabilities,
            group_entropy.reshape(1),
            normalized_group_entropy.reshape(1),
            grammar_probabilities,
            grammar_entropy.reshape(1),
            normalized_grammar_entropy.reshape(1),
            operator_entropy.reshape(1),
            normalized_operator_entropy.reshape(1),
            window_probabilities,
            window_entropy.reshape(1),
            normalized_window_entropy.reshape(1),
        )
    )
    if compact.shape != (_COMPACT_DIAGNOSTIC_SIZE,):
        raise RuntimeError("紧凑策略诊断向量长度错误")
    return compact.detach()


def _decode_compact_policy_diagnostics(
    values: Sequence[float], token_id: int
) -> dict[str, object]:
    if len(values) != _COMPACT_DIAGNOSTIC_SIZE:
        raise ValueError("紧凑策略诊断向量长度错误")
    numbers = tuple(float(value) for value in values)
    policy_entropy = numbers[0]
    normalized_policy_entropy = numbers[1]
    if not math.isfinite(policy_entropy) or policy_entropy < -1e-10:
        raise FloatingPointError("联合策略熵出现无效值")
    if math.isfinite(normalized_policy_entropy):
        if not -1e-8 <= normalized_policy_entropy <= 1.0 + 1e-6:
            raise FloatingPointError("归一化策略熵超出合法范围")
        normalized_value: float | None = min(
            1.0, max(0.0, normalized_policy_entropy)
        )
    else:
        normalized_value = None

    result: dict[str, object] = {
        "policy_entropy": max(0.0, policy_entropy),
        "normalized_policy_entropy": normalized_value,
    }
    group_probabilities = numbers[2:5]
    if all(math.isfinite(value) for value in group_probabilities):
        result.update(
            selected_token_group=get_action(token_id).arity,
            group_probabilities=group_probabilities,
            group_entropy=max(0.0, numbers[5]),
            normalized_group_entropy=min(1.0, max(0.0, numbers[6])),
        )

    grammar_probabilities = numbers[7:13]
    if all(math.isfinite(value) for value in grammar_probabilities):
        action = get_action(token_id)
        result.update(
            selected_grammar_category=CATEGORY_TO_INDEX[action.category],
            grammar_category_probabilities=grammar_probabilities,
            grammar_category_entropy=max(0.0, numbers[13]),
            normalized_grammar_category_entropy=min(
                1.0, max(0.0, numbers[14])
            ),
            operator_entropy=max(0.0, numbers[15]),
            normalized_operator_entropy=min(1.0, max(0.0, numbers[16])),
        )
        window_probabilities = numbers[17:22]
        if action.window:
            if not all(math.isfinite(value) for value in window_probabilities):
                raise FloatingPointError("时序动作缺少有限 Window 诊断")
            result.update(
                window_probabilities=window_probabilities,
                window_entropy=max(0.0, numbers[22]),
                normalized_window_entropy=min(1.0, max(0.0, numbers[23])),
            )
    return result


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


def _group_diagnostics(
    output,
    row: int,
    slot_index: int,
) -> tuple[tuple[float, float, float], float, float] | None:
    if output.group_log_probs is None or output.legal_group_mask is None:
        return None
    logs = output.group_log_probs[row, slot_index]
    legal = output.legal_group_mask[row, slot_index]
    legal_logs = logs[legal]
    if not bool(torch.isfinite(legal_logs).all()):
        raise FloatingPointError("合法 Token 组缺少有限 log 概率")
    probabilities = logs.exp().masked_fill(~legal, 0.0)
    if not torch.isclose(
        probabilities.sum(),
        torch.ones((), device=probabilities.device),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise FloatingPointError("Token 组概率没有归一化")
    entropy = float((-(legal_logs.exp() * legal_logs).sum()).detach())
    legal_count = int(legal.sum().item())
    normalized = entropy / math.log(legal_count) if legal_count > 1 else 0.0
    return (
        tuple(float(value) for value in probabilities.detach().cpu().tolist()),
        max(0.0, entropy),
        min(1.0, max(0.0, normalized)),
    )


def _entropy_diagnostics(logs: torch.Tensor, legal: torch.Tensor) -> tuple[float, float]:
    legal_logs = logs[legal]
    if not bool(torch.isfinite(legal_logs).all()):
        raise FloatingPointError("合法条件分布缺少有限 log 概率")
    entropy = float((-(legal_logs.exp() * legal_logs).sum()).detach())
    count = int(legal.sum().item())
    normalized = entropy / math.log(count) if count > 1 else 0.0
    return max(0.0, entropy), min(1.0, max(0.0, normalized))


def _grammar_diagnostics(
    output,
    row: int,
    slot_index: int,
    token_id: int,
) -> dict[str, object] | None:
    if output.grammar_category_log_probs is None:
        return None
    action = get_action(token_id)
    category_id = CATEGORY_TO_INDEX[action.category]
    category_logs = output.grammar_category_log_probs[row, slot_index]
    category_legal = output.legal_grammar_category_mask[row, slot_index]
    category_probabilities = category_logs.exp().masked_fill(~category_legal, 0.0)
    if not torch.isclose(
        category_probabilities.sum(),
        torch.ones((), device=category_probabilities.device),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise FloatingPointError("文法类别联合概率没有归一化")
    category_entropy = _entropy_diagnostics(category_logs, category_legal)

    operator_id = OPERATOR_TO_INDEX[action.name]
    operator_logs = output.operator_log_probs[row, slot_index]
    operator_legal = output.legal_operator_mask[row, slot_index]
    selected_operator_log = operator_logs[operator_id]
    if not bool(torch.isfinite(selected_operator_log)):
        raise FloatingPointError("所选 Operator 缺少有限条件概率")
    if output.operator_category_lookup is None:
        raise FloatingPointError("完整文法策略缺少 Operator 类别映射")
    category_operator_mask = output.operator_category_lookup == category_id
    finite_operator_logs = operator_logs[operator_legal & category_operator_mask]
    operator_entropy_value = float(
        (-(finite_operator_logs.exp() * finite_operator_logs).sum()).detach()
    )
    operator_count = int(finite_operator_logs.numel())
    operator_entropy = (
        max(0.0, operator_entropy_value),
        operator_entropy_value / math.log(operator_count) if operator_count > 1 else 0.0,
    )

    result: dict[str, object] = {
        "selected_grammar_category": category_id,
        "grammar_category_probabilities": tuple(
            float(value) for value in category_probabilities.detach().cpu().tolist()
        ),
        "grammar_category_entropy": category_entropy[0],
        "normalized_grammar_category_entropy": category_entropy[1],
        "operator_entropy": operator_entropy[0],
        "normalized_operator_entropy": min(1.0, max(0.0, operator_entropy[1])),
        "window_probabilities": None,
        "window_entropy": None,
        "normalized_window_entropy": None,
    }
    if action.window:
        window_logs = output.window_log_probs[row, slot_index, operator_id]
        window_legal = output.legal_window_mask[row, slot_index, operator_id]
        window_probabilities = window_logs.exp().masked_fill(~window_legal, 0.0)
        if not torch.isclose(
            window_probabilities.sum(),
            torch.ones((), device=window_probabilities.device),
            rtol=1e-6,
            atol=1e-6,
        ):
            raise FloatingPointError("Window 条件概率没有归一化")
        window_entropy = _entropy_diagnostics(window_logs, window_legal)
        result.update(
            window_probabilities=tuple(
                float(value) for value in window_probabilities.detach().cpu().tolist()
            ),
            window_entropy=window_entropy[0],
            normalized_window_entropy=window_entropy[1],
        )
        if not bool(window_legal[WINDOWS.index(action.window)]):
            raise FloatingPointError("所选 Window 在条件分布中不合法")
    return result


def sample_trajectories(
    model: ForwardPolicyNetwork,
    adapter: StateAdapter,
    *,
    num_trajectories: int,
    sampling_config: SamplingConfig = SamplingConfig(),
    initial_states: Sequence[PolicyGrammarState] | None = None,
    target_node_counts: Sequence[int] | None = None,
    batched_policy_diagnostics: bool = False,
) -> list[Trajectory]:
    """批量推进所有未终止状态，并保留前向概率的计算图。"""

    if isinstance(num_trajectories, bool) or not isinstance(num_trajectories, int) or num_trajectories < 1:
        raise ValueError("num_trajectories 必须是正整数")
    if target_node_counts is not None and len(target_node_counts) != num_trajectories:
        raise ValueError("target_node_counts 数量必须等于 num_trajectories")
    if initial_states is None:
        structural_sources = [
            GrammarState(search_space=adapter.search_space)
            for _ in range(num_trajectories)
        ]
        if target_node_counts is None:
            states: list[PolicyGrammarState] = structural_sources
        else:
            states = [
                ExactNodeGrammarState(source, target_node_count=target)
                for source, target in zip(
                    structural_sources, target_node_counts, strict=True
                )
            ]
    else:
        if len(initial_states) != num_trajectories:
            raise ValueError("initial_states 数量必须等于 num_trajectories")
        if target_node_counts is None:
            states = list(initial_states)
        else:
            states = []
            for initial_state, target in zip(
                initial_states, target_node_counts, strict=True
            ):
                if isinstance(initial_state, ExactNodeGrammarState):
                    if initial_state.target_node_count != target:
                        raise ValueError(
                            "initial_state 与 target_node_counts 的目标 N 不一致"
                        )
                    states.append(initial_state)
                else:
                    states.append(
                        ExactNodeGrammarState(
                            initial_state,
                            target_node_count=target,
                        )
                    )
        if any(state.done for state in states):
            raise ValueError("初始状态必须全部为非终止状态")

    device = _model_device(model)
    step_lists: list[list[TrajectoryStep]] = [[] for _ in states]
    compact_diagnostic_lists: list[list[torch.Tensor]] = [[] for _ in states]
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
                if batched_policy_diagnostics:
                    compact_diagnostic_lists[trajectory_index].append(
                        _compact_policy_diagnostics(
                            output,
                            row,
                            slot_index,
                            token_id,
                            batch.slot_mask[row],
                            batch.legal_token_mask[row],
                        )
                    )
                    policy_entropy = None
                    normalized_policy_entropy = None
                    group_diagnostics = None
                    grammar_diagnostics = None
                else:
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
                    group_diagnostics = _group_diagnostics(output, row, slot_index)
                    grammar_diagnostics = _grammar_diagnostics(
                        output, row, slot_index, token_id
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
                        selected_token_group=(
                            get_action(token_id).arity
                            if group_diagnostics is not None else None
                        ),
                        group_probabilities=(
                            group_diagnostics[0]
                            if group_diagnostics is not None else None
                        ),
                        group_entropy=(
                            group_diagnostics[1]
                            if group_diagnostics is not None else None
                        ),
                        normalized_group_entropy=(
                            group_diagnostics[2]
                            if group_diagnostics is not None else None
                        ),
                        **(grammar_diagnostics or {}),
                    )
                )
                states[trajectory_index] = child
                if not child.done:
                    next_active.append(trajectory_index)
            active = next_active
    finally:
        model.train(was_training)

    if batched_policy_diagnostics:
        packed_tensors = []
        locations: list[tuple[int, int]] = []
        for trajectory_index, (steps, diagnostics) in enumerate(
            zip(step_lists, compact_diagnostic_lists, strict=True)
        ):
            if len(steps) != len(diagnostics):
                raise RuntimeError("轨迹步骤与紧凑策略诊断未对齐")
            for step_index, (step, diagnostic) in enumerate(
                zip(steps, diagnostics, strict=True)
            ):
                packed_tensors.append(
                    torch.cat(
                        (
                            torch.stack(
                                (
                                    step.log_p_slot.detach(),
                                    step.log_p_token.detach(),
                                    step.log_pf.detach(),
                                )
                            ),
                            diagnostic,
                        )
                    )
                )
                locations.append((trajectory_index, step_index))
        packed_values = torch.stack(packed_tensors).cpu().tolist()
        for (trajectory_index, step_index), values in zip(
            locations, packed_values, strict=True
        ):
            log_probabilities = tuple(float(value) for value in values[:3])
            if any(
                not math.isfinite(value) or value > 1e-6
                for value in log_probabilities
            ):
                raise FloatingPointError("采样轨迹包含无效 log 概率")
            step = step_lists[trajectory_index][step_index]
            step_lists[trajectory_index][step_index] = replace(
                step,
                **_decode_compact_policy_diagnostics(
                    values[3:], step.selected_token_id
                ),
            )

    trajectories = [
        Trajectory(
            steps=step_lists[index],
            terminal_state_hash=state_hash(state),
            terminal_expression=state.to_expression(),
            sampling_mode="greedy" if sampling_config.greedy else "stochastic",
            target_node_count=(
                state.target_node_count
                if isinstance(state, ExactNodeGrammarState)
                else None
            ),
            terminal_node_count=state.node_count,
            condition_fingerprint=(
                target_condition_fingerprint(
                    state.target_node_count,
                    state.search_space.fingerprint(),
                )
                if isinstance(state, ExactNodeGrammarState)
                else None
            ),
        )
        for index, state in enumerate(states)
    ]
    for trajectory in trajectories:
        trajectory.validate(check_tensor_values=not batched_policy_diagnostics)
    return trajectories


def sample_trajectory(
    model: ForwardPolicyNetwork,
    adapter: StateAdapter,
    *,
    sampling_config: SamplingConfig = SamplingConfig(),
    initial_state: PolicyGrammarState | None = None,
    target_node_count: int | None = None,
) -> Trajectory:
    initial_states = None if initial_state is None else [initial_state]
    return sample_trajectories(
        model,
        adapter,
        num_trajectories=1,
        sampling_config=sampling_config,
        initial_states=initial_states,
        target_node_counts=(
            None if target_node_count is None else [target_node_count]
        ),
    )[0]


__all__ = ["sample_trajectories", "sample_trajectory"]
