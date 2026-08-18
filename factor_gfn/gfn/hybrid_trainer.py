"""Isolated Trainer for the frozen Stage 5 hybrid-variance objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from os import PathLike
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

from factor_gfn.grammar import (
    action_space_fingerprint,
    state_space_fingerprint,
    transition_space_fingerprint,
)

from .complexity_scheduler import BalancedNodeCountScheduler, ConditionAssignment
from .exhaustive import ExactMassResult, ExhaustiveRegistry, count_canonical_terminals
from .exhaustive_registry_reuse import (
    ExhaustiveReuseSemantics,
    ExhaustiveStratumReuseProof,
    ProvenExhaustiveRewardLookup,
    prove_exhaustive_stratum_reuse,
)
from .hybrid_config import HybridVarianceGFNConfig
from .log_partition_variance import (
    LogPartitionVarianceLoss,
    LogPartitionVarianceOutput,
)
from .loss import FixedExactTrajectoryBalanceLoss, TBLossOutput
from .model import ForwardPolicyNetwork
from .state_adapter import StateAdapter
from .trainer import (
    GFNTrainer,
    RewardAssignment,
    RewardProvider,
    SingleConditionBatchCollection,
    configure_cuda_determinism,
    seed_everything,
)
from .trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class HybridObjectiveOutput:
    condition_N: int
    objective_kind: Literal["exact_tb", "log_partition_variance"]
    loss: Tensor
    exact_tb: TBLossOutput | None = None
    log_partition_variance: LogPartitionVarianceOutput | None = None


@dataclass(frozen=True, slots=True)
class HybridCommonDiagnostics:
    """Fields shared by every successful hybrid policy update.

    ``reward_std`` uses the population convention (``ddof=0``), and
    ``trajectory_length`` is the arithmetic mean number of actions in the
    successful batch.
    """

    cycle_index: int
    condition_position_in_cycle: int
    condition_N: int
    objective_kind: Literal["exact_tb", "log_partition_variance"]
    global_optimizer_step: int
    trajectories_in_batch: int
    total_trajectories_seen: int
    requested_count: int
    accepted_count: int
    invalid_count: int
    retry_count: int
    retry_exhausted_count: int
    reward_mean: float
    reward_std: float
    sum_log_pf_mean: float
    sum_log_pb_mean: float
    trajectory_length: float
    terminal_success_rate: float
    policy_grad_norm: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HybridExactTBDiagnostics(HybridCommonDiagnostics):
    exact_log_z: float
    tb_loss: float
    tb_delta_mean: float
    tb_delta_std: float
    tb_delta_rms: float


@dataclass(frozen=True, slots=True)
class HybridLPVDiagnostics(HybridCommonDiagnostics):
    zeta_mean: float
    zeta_std: float
    zeta_variance: float
    variance_loss: float
    centered_zeta_rms: float
    unique_terminal_count: int
    unique_terminal_fraction: float


HybridUpdateDiagnostics = HybridExactTBDiagnostics | HybridLPVDiagnostics


@dataclass(frozen=True, slots=True)
class HybridUpdateOutput:
    assignment: ConditionAssignment
    collection: SingleConditionBatchCollection
    objective: HybridObjectiveOutput | None
    updated: bool
    global_optimizer_step: int
    policy_grad_norm: float | None
    diagnostics: HybridUpdateDiagnostics | None


class HybridVarianceTrainer(GFNTrainer):
    """Shared-policy Trainer with exact TB for N=1/2 and direct LPV otherwise."""

    def __init__(
        self,
        config: HybridVarianceGFNConfig,
        reward_provider: RewardProvider,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(config, HybridVarianceGFNConfig):
            raise TypeError("config must be HybridVarianceGFNConfig")
        for method in ("evaluate", "fingerprint", "manifest"):
            if not callable(getattr(reward_provider, method, None)):
                raise TypeError(f"reward_provider lacks {method}()")
        provider_manifest = reward_provider.manifest()
        if not isinstance(provider_manifest, dict):
            raise TypeError("reward_provider.manifest() must return dict")
        declared_reward_config = provider_manifest.get("reward_config")
        if (
            declared_reward_config is not None
            and declared_reward_config != asdict(config.reward)
        ):
            raise ValueError(
                "hybrid config Reward does not match RewardProvider declaration"
            )

        self.config = config
        self.reward_provider = reward_provider
        self.device = torch.device(device)
        self.no_anchor_mode = False
        self.hybrid_mode = True
        self.cublas_workspace_config = configure_cuda_determinism(
            self.device,
            deterministic_algorithms=config.training.deterministic_algorithms,
        )
        seed_everything(
            config.training.seed,
            deterministic_algorithms=config.training.deterministic_algorithms,
        )
        self.adapter = StateAdapter(config.search_space)
        self.model = ForwardPolicyNetwork(config.model, config.search_space).to(
            self.device
        )
        self.exact_tb_loss = FixedExactTrajectoryBalanceLoss(
            max_nodes=config.search_space.max_nodes,
            exact_node_counts=config.objective.exact_tb_node_counts,
        ).to(self.device)
        self.tb_loss = self.exact_tb_loss
        self.log_partition_variance_loss = LogPartitionVarianceLoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "name": "policy",
                    "params": self.model.parameters(),
                    "lr": config.training.learning_rate,
                    "weight_decay": config.training.weight_decay,
                }
            ],
            betas=(
                config.training.optimizer_beta1,
                config.training.optimizer_beta2,
            ),
            eps=config.training.optimizer_eps,
        )
        self.complexity_scheduler = BalancedNodeCountScheduler(
            config.resolved_condition_node_counts,
            seed=config.training.seed,
        )
        self.optimizer_step = 0
        self.total_trajectories_seen = 0
        self.diagnostic_history: list[HybridUpdateDiagnostics] = []
        self.registered_exact_masses_by_N: dict[int, ExactMassResult] = {}
        self.exhaustive_reuse_proofs_by_N: dict[
            int, ExhaustiveStratumReuseProof
        ] = {}
        self.exhaustive_reward_lookups_by_N: dict[
            int, ProvenExhaustiveRewardLookup
        ] = {}

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.model.parameters())

    def optimizer_contract(self) -> dict[str, Any]:
        return {
            "schema": "factor_gfn.hybrid_policy_optimizer.v1",
            "optimizer": "adam",
            "parameter_group_names": ("policy",),
            "learning_rate": self.config.training.learning_rate,
            "gradient_clip_norm": self.config.training.model_gradient_clip_norm,
            "weight_decay": self.config.training.weight_decay,
            "betas": (
                self.config.training.optimizer_beta1,
                self.config.training.optimizer_beta2,
            ),
            "eps": self.config.training.optimizer_eps,
            "learned_log_z": False,
            "normalizer_optimizer": None,
        }

    def _exact_node_retry_budget(self) -> int:
        return self.config.complexity.exact_node_retry_budget

    def target_exhaustive_reuse_semantics(self) -> ExhaustiveReuseSemantics:
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

    def register_exact_mass_result(self, result: ExactMassResult) -> None:
        if not isinstance(result, ExactMassResult):
            raise TypeError("result must be ExactMassResult")
        if result.node_count not in self.config.objective.exact_tb_node_counts:
            raise ValueError("exact mass can only register hybrid exact-TB strata")
        if result.valid_candidate_count < 1 or not math.isfinite(result.exact_tb_log_z):
            raise ValueError("exact mass requires valid candidates and finite TB logZ")
        if result.reward_floor != self.config.reward.reward_floor:
            raise ValueError("exact mass reward_floor does not match hybrid config")
        if result.provider_fingerprint != self.reward_provider.fingerprint():
            raise ValueError("exact mass provider fingerprint does not match Trainer")
        context = self.reward_provider.manifest().get("context_fingerprint")
        if result.context_fingerprint != context:
            raise ValueError("exact mass context fingerprint does not match Trainer")
        current = self.registered_exact_masses_by_N.get(result.node_count)
        if current is not None and current != result:
            raise ValueError(f"exact mass for N={result.node_count} is already registered")
        self.exact_tb_loss.set_exact_log_z(
            result.node_count,
            result.exact_tb_log_z,
        )
        self.registered_exact_masses_by_N[result.node_count] = result

    def configure_hybrid_exhaustive_registry(
        self,
        registry: ExhaustiveRegistry,
        *,
        source_semantics_by_N: Mapping[int, ExhaustiveReuseSemantics],
    ) -> dict[int, ExhaustiveStratumReuseProof]:
        if self.exhaustive_reward_lookups_by_N:
            raise RuntimeError("hybrid exhaustive registry equivalence is already verified")
        expected = set(self.config.objective.exact_tb_node_counts)
        if set(source_semantics_by_N) != expected:
            raise ValueError("source semantics strata must exactly match hybrid exact TB")
        target_semantics = self.target_exhaustive_reuse_semantics()
        proofs: dict[int, ExhaustiveStratumReuseProof] = {}
        lookups: dict[int, ProvenExhaustiveRewardLookup] = {}
        exact_results: dict[int, ExactMassResult] = {}
        for node_count in self.config.objective.exact_tb_node_counts:
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
        for result in exact_results.values():
            self.register_exact_mass_result(result)
        self.exhaustive_reuse_proofs_by_N = proofs
        self.exhaustive_reward_lookups_by_N = lookups
        return dict(proofs)

    def _evaluate_reward(self, expression) -> RewardAssignment:
        node_count = expression.stats.node_count
        if node_count in self.config.objective.exact_tb_node_counts:
            lookup = self.exhaustive_reward_lookups_by_N.get(node_count)
            if lookup is None:
                raise RuntimeError(
                    f"hybrid exact N={node_count} lacks registry equivalence verification"
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

    def route_objective(
        self,
        trajectories: Sequence[Trajectory],
    ) -> HybridObjectiveOutput:
        if len(trajectories) != self.config.training.trajectories_per_batch:
            raise ValueError("hybrid objective requires exactly configured K trajectories")
        conditions = {trajectory.target_node_count for trajectory in trajectories}
        if len(conditions) != 1:
            raise ValueError("hybrid objective requires one fixed condition")
        condition_N = next(iter(conditions))
        if condition_N in self.config.objective.exact_tb_node_counts:
            if set(self.registered_exact_masses_by_N) != set(
                self.config.objective.exact_tb_node_counts
            ):
                raise RuntimeError("hybrid exact masses are not fully configured")
            output = self.exact_tb_loss(trajectories)
            return HybridObjectiveOutput(
                condition_N=condition_N,
                objective_kind="exact_tb",
                loss=output.loss,
                exact_tb=output,
            )
        if condition_N in self.config.objective.lpv_node_counts:
            output = self.log_partition_variance_loss(trajectories)
            return HybridObjectiveOutput(
                condition_N=condition_N,
                objective_kind="log_partition_variance",
                loss=output.loss,
                log_partition_variance=output,
            )
        raise ValueError("trajectory condition is outside hybrid N=1..15")

    def _build_update_diagnostics(
        self,
        *,
        assignment: ConditionAssignment,
        collection: SingleConditionBatchCollection,
        objective: HybridObjectiveOutput,
        policy_grad_norm: float,
    ) -> HybridUpdateDiagnostics:
        trajectories = collection.trajectories
        rewards = torch.tensor(
            [trajectory.require_valid_reward()[0] for trajectory in trajectories],
            dtype=torch.float64,
        )
        common: dict[str, Any] = {
            "cycle_index": assignment.cycle_index,
            "condition_position_in_cycle": (
                assignment.condition_position_in_cycle
            ),
            "condition_N": assignment.condition_N,
            "objective_kind": objective.objective_kind,
            "global_optimizer_step": self.optimizer_step,
            "trajectories_in_batch": len(trajectories),
            "total_trajectories_seen": self.total_trajectories_seen,
            "requested_count": collection.requested_count,
            "accepted_count": collection.accepted_count,
            "invalid_count": collection.invalid_count,
            "retry_count": collection.retry_count,
            "retry_exhausted_count": collection.retry_exhausted_count,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "trajectory_length": float(
                math.fsum(len(trajectory.steps) for trajectory in trajectories)
                / len(trajectories)
            ),
            "terminal_success_rate": (
                collection.accepted_count / collection.requested_count
            ),
            "policy_grad_norm": policy_grad_norm,
        }
        if objective.exact_tb is not None:
            exact = objective.exact_tb
            return HybridExactTBDiagnostics(
                **common,
                sum_log_pf_mean=float(exact.mean_log_pf.detach()),
                sum_log_pb_mean=float(exact.mean_log_pb.detach()),
                exact_log_z=float(exact.log_z.detach().mean()),
                tb_loss=float(exact.loss.detach()),
                tb_delta_mean=float(exact.delta_mean.detach()),
                tb_delta_std=float(exact.delta_std.detach()),
                tb_delta_rms=float(
                    torch.sqrt(exact.deltas.detach().square().mean())
                ),
            )
        if objective.log_partition_variance is not None:
            lpv = objective.log_partition_variance
            terminal_identities = {
                trajectory.terminal_expression.canonicalize().structural_hash()
                for trajectory in trajectories
            }
            unique_terminal_count = len(terminal_identities)
            return HybridLPVDiagnostics(
                **common,
                sum_log_pf_mean=float(lpv.mean_log_pf.detach()),
                sum_log_pb_mean=float(lpv.mean_log_pb.detach()),
                zeta_mean=float(lpv.zeta_mean.detach()),
                zeta_std=float(lpv.zeta_std.detach()),
                zeta_variance=float(lpv.zeta_variance.detach()),
                variance_loss=float(lpv.loss.detach()),
                centered_zeta_rms=float(lpv.centered_zeta_rms.detach()),
                unique_terminal_count=unique_terminal_count,
                unique_terminal_fraction=(
                    unique_terminal_count / len(trajectories)
                ),
            )
        raise RuntimeError("hybrid objective lacks routed diagnostic output")

    def train_step(self) -> HybridUpdateOutput:
        if self.optimizer_step >= self.config.training.total_optimizer_steps:
            raise RuntimeError("hybrid training cycle budget is exhausted")
        if self.config.sampling.greedy:
            raise ValueError("hybrid Trainer forbids greedy parameter updates")
        self.model.train()
        assignment = self.complexity_scheduler.peek()
        collection = self.collect_single_condition_batch(
            condition_N=assignment.condition_N,
            trajectories_per_batch=self.config.training.trajectories_per_batch,
        )
        if not collection.complete:
            self.optimizer.zero_grad(set_to_none=True)
            return HybridUpdateOutput(
                assignment=assignment,
                collection=collection,
                objective=None,
                updated=False,
                global_optimizer_step=self.optimizer_step,
                policy_grad_norm=None,
                diagnostics=None,
            )

        self.optimizer.zero_grad(set_to_none=True)
        objective = self.route_objective(collection.trajectories)
        objective.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.training.model_gradient_clip_norm,
        )
        self.optimizer.step()
        self.complexity_scheduler.commit(assignment)
        self.optimizer_step += 1
        self.total_trajectories_seen += collection.accepted_count
        policy_grad_norm = float(gradient_norm)
        diagnostics = self._build_update_diagnostics(
            assignment=assignment,
            collection=collection,
            objective=objective,
            policy_grad_norm=policy_grad_norm,
        )
        self.diagnostic_history.append(diagnostics)
        return HybridUpdateOutput(
            assignment=assignment,
            collection=collection,
            objective=objective,
            updated=True,
            global_optimizer_step=self.optimizer_step,
            policy_grad_norm=policy_grad_norm,
            diagnostics=diagnostics,
        )

    def train(self, steps: int) -> list[HybridUpdateOutput]:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        return [self.train_step() for _ in range(steps)]

    def save_checkpoint(self, path: str | PathLike[str]) -> None:
        from .hybrid_checkpoint import save_hybrid_checkpoint

        save_hybrid_checkpoint(path, self)

    def load_checkpoint(self, path: str | PathLike[str]) -> dict[str, Any]:
        from .hybrid_checkpoint import load_hybrid_checkpoint

        return load_hybrid_checkpoint(path, self)


__all__ = [
    "HybridCommonDiagnostics",
    "HybridExactTBDiagnostics",
    "HybridLPVDiagnostics",
    "HybridObjectiveOutput",
    "HybridUpdateOutput",
    "HybridUpdateDiagnostics",
    "HybridVarianceTrainer",
]
