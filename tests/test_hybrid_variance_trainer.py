import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from factor_gfn.gfn import (
    ExhaustivePlanningConfig,
    ExhaustiveRegistry,
    HybridVarianceTrainer,
    SingleConditionBatchCollection,
    SyntheticRewardProvider,
    TrajectoryBalanceLoss,
    build_stage5_hybrid_variance_5_15_config,
    resolve_exhaustive_plan,
)
from factor_gfn.gfn.trajectory import (
    Trajectory,
    TrajectoryStep,
    target_condition_fingerprint,
)
from factor_gfn.grammar import Expression, SearchSpaceConfig, get_action_id


class _CountingSyntheticRewardProvider(SyntheticRewardProvider):
    def __init__(self) -> None:
        super().__init__()
        self.evaluate_count = 0

    def evaluate(self, expression):
        self.evaluate_count += 1
        return super().evaluate(expression)


def _trajectory(
    trainer: HybridVarianceTrainer,
    *,
    index: int,
    condition_N: int,
    reward: float = 1.0,
) -> Trajectory:
    if condition_N == 1:
        prefix = [get_action_id("close")]
    elif condition_N == 3:
        leaves = (
            (get_action_id("close"), get_action_id("open"))
            if index % 2
            else (get_action_id("open"), get_action_id("close"))
        )
        prefix = [get_action_id("add"), *leaves]
    else:
        raise ValueError("test helper supports N=1 or N=3")
    parameter = next(trainer.model.parameters()).reshape(-1)[0]
    log_p_slot = torch.zeros((), dtype=parameter.dtype, device=parameter.device)
    log_p_token = -torch.nn.functional.softplus(parameter + float(index))
    step = TrajectoryStep(
        state_hash=f"{2 * index:064x}",
        selected_slot_index=0,
        selected_slot_path=(),
        selected_slot_orbit_key="hybrid-routing-test",
        selected_token_id=get_action_id("close"),
        log_p_slot=log_p_slot,
        log_p_token=log_p_token,
        log_pf=log_p_slot + log_p_token,
        child_state_hash=f"{2 * index + 1:064x}",
        n_parents=1,
        log_pb=0.0,
    )
    trajectory = Trajectory(
        steps=[step],
        terminal_state_hash=step.child_state_hash,
        terminal_expression=Expression.from_prefix(prefix),
        sampling_mode="stochastic",
        target_node_count=condition_N,
        terminal_node_count=condition_N,
        condition_fingerprint=target_condition_fingerprint(
            condition_N,
            trainer.config.search_space.fingerprint(),
        ),
    )
    trajectory.attach_reward(reward)
    return trajectory


class HybridVarianceTrainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.provider = _CountingSyntheticRewardProvider()
        self.config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=1,
            trajectories_per_batch=2,
        )
        self.trainer = HybridVarianceTrainer(
            self.config,
            self.provider,
            device="cpu",
        )
        source_space = SearchSpaceConfig(max_depth=6, max_nodes=20)
        plan = resolve_exhaustive_plan(
            source_space,
            ExhaustivePlanningConfig(
                explicit_include_node_counts=(1, 2),
                explicit_exclude_node_counts=tuple(range(3, 21)),
            ),
        )
        self.registry = ExhaustiveRegistry(
            Path(self.temporary.name) / "legacy_registry.sqlite3"
        )
        self.registry.register_plan(
            plan,
            provider_fingerprint=self.provider.fingerprint(),
            context_fingerprint=self.provider.manifest()["context_fingerprint"],
        )
        for node_count in (1, 2):
            for candidate in self.registry.pending_candidates(node_count):
                expression = Expression.from_prefix(candidate.prefix_token_ids)
                assignment = self.provider.evaluate(expression)
                self.registry.record_evaluation(
                    candidate.structural_hash,
                    valid=True,
                    reward_details={
                        "reward_result": {
                            "raw_reward": assignment.reward,
                            "reward": assignment.reward,
                            "log_reward": assignment.log_reward,
                        }
                    },
                    target_mass=assignment.reward,
                )
            self.registry.compute_exact_masses(
                node_count,
                reward_floor=self.config.reward.reward_floor,
            )
        self.provider.evaluate_count = 0
        semantics = self.trainer.target_exhaustive_reuse_semantics()
        self.trainer.configure_hybrid_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics, 2: semantics},
        )

    def tearDown(self):
        self.registry.close()
        self.temporary.cleanup()

    def _batch(
        self,
        condition_N: int,
        *,
        rewards: tuple[float, ...] | None = None,
    ) -> tuple[Trajectory, ...]:
        if rewards is None:
            rewards = (1.0,) * self.config.training.trajectories_per_batch
        self.assertEqual(
            len(rewards),
            self.config.training.trajectories_per_batch,
        )
        return tuple(
            _trajectory(
                self.trainer,
                index=index,
                condition_N=condition_N,
                reward=rewards[index - 1],
            )
            for index in range(1, self.config.training.trajectories_per_batch + 1)
        )

    @staticmethod
    def _complete_collection(
        condition_N: int,
        trajectories: tuple[Trajectory, ...],
        *,
        invalid_count: int = 0,
        retry_count: int = 0,
    ) -> SingleConditionBatchCollection:
        return SingleConditionBatchCollection(
            condition_N=condition_N,
            requested_count=len(trajectories),
            trajectories=trajectories,
            sampled_count=len(trajectories) + invalid_count,
            invalid_count=invalid_count,
            retry_count=retry_count,
            retry_exhausted_count=0,
            sampling_rounds=1 + int(retry_count > 0),
            sampling_seconds=0.0,
            reward_provider_seconds=0.0,
        )

    def _force_pending_condition(self, condition_N: int) -> None:
        state = self.trainer.complexity_scheduler.state_dict()
        permutation = (condition_N,) + tuple(
            value
            for value in self.config.resolved_condition_node_counts
            if value != condition_N
        )
        state["current_permutation"] = permutation
        state["position"] = 0
        self.trainer.complexity_scheduler.load_state_dict(state)

    def test_policy_only_optimizer_has_no_learned_logz_state(self):
        optimizer_groups = self.trainer.optimizer.param_groups
        self.assertEqual([group["name"] for group in optimizer_groups], ["policy"])
        optimizer_ids = {
            id(parameter)
            for group in optimizer_groups
            for parameter in group["params"]
        }
        model_ids = {id(parameter) for parameter in self.trainer.model.parameters()}
        self.assertEqual(optimizer_ids, model_ids)
        self.assertEqual(list(self.trainer.exact_tb_loss.parameters()), [])
        self.assertEqual(
            list(self.trainer.log_partition_variance_loss.parameters()),
            [],
        )
        contract = self.trainer.optimizer_contract()
        self.assertFalse(contract["learned_log_z"])
        self.assertIsNone(contract["normalizer_optimizer"])
        all_named_parameters = [
            *self.trainer.model.named_parameters(),
            *self.trainer.exact_tb_loss.named_parameters(),
            *self.trainer.log_partition_variance_loss.named_parameters(),
        ]
        self.assertFalse(
            any("log_z" in name.lower() for name, _ in all_named_parameters)
        )

    def test_routes_N1_to_existing_exact_tb_formula_and_N3_to_direct_lpv(self):
        exact_batch = self._batch(1)
        exact = self.trainer.route_objective(exact_batch)
        self.assertEqual(exact.objective_kind, "exact_tb")
        self.assertIsNotNone(exact.exact_tb)
        self.assertIsNone(exact.log_partition_variance)

        legacy_exact = TrajectoryBalanceLoss(
            max_nodes=15,
            exact_node_counts=(1, 2),
        )
        for node_count, result in self.trainer.registered_exact_masses_by_N.items():
            legacy_exact.set_exact_log_z(node_count, result.exact_tb_log_z)
        legacy_output = legacy_exact(exact_batch)
        torch.testing.assert_close(exact.loss, legacy_output.loss)
        torch.testing.assert_close(exact.exact_tb.deltas, legacy_output.deltas)

        lpv = self.trainer.route_objective(self._batch(3))
        self.assertEqual(lpv.objective_kind, "log_partition_variance")
        self.assertIsNone(lpv.exact_tb)
        self.assertIsNotNone(lpv.log_partition_variance)
        torch.testing.assert_close(
            lpv.loss,
            lpv.log_partition_variance.zeta.var(correction=0),
        )

    def test_both_objectives_backpropagate_to_the_same_shared_policy(self):
        parameter = next(self.trainer.model.parameters())
        optimizer_identity = id(self.trainer.optimizer)
        before_exact = parameter.detach().clone()
        self.trainer.optimizer.zero_grad(set_to_none=True)
        self.trainer.route_objective(self._batch(1)).loss.backward()
        self.assertIsNotNone(parameter.grad)
        self.trainer.optimizer.step()
        self.assertFalse(torch.equal(parameter.detach(), before_exact))

        before_lpv = parameter.detach().clone()
        self.trainer.optimizer.zero_grad(set_to_none=True)
        self.trainer.route_objective(self._batch(3)).loss.backward()
        self.assertIsNotNone(parameter.grad)
        self.trainer.optimizer.step()
        self.assertFalse(torch.equal(parameter.detach(), before_lpv))
        self.assertEqual(id(self.trainer.optimizer), optimizer_identity)

    def test_successful_update_steps_policy_then_commits_condition(self):
        self._force_pending_condition(3)
        pending = self.trainer.complexity_scheduler.peek()
        trajectories = self._batch(3)
        collection = self._complete_collection(3, trajectories)
        parameter = next(self.trainer.model.parameters())
        before = parameter.detach().clone()

        with patch.object(
            self.trainer,
            "collect_single_condition_batch",
            return_value=collection,
        ):
            output = self.trainer.train_step()

        self.assertTrue(output.updated)
        self.assertEqual(output.assignment, pending)
        self.assertEqual(output.objective.objective_kind, "log_partition_variance")
        self.assertEqual(output.global_optimizer_step, 1)
        self.assertFalse(torch.equal(parameter.detach(), before))
        self.assertEqual(self.trainer.complexity_scheduler.position, 1)
        self.assertIsNotNone(output.diagnostics)
        self.assertEqual(self.trainer.total_trajectories_seen, 2)
        self.assertEqual(self.trainer.diagnostic_history, [output.diagnostics])

    def test_exact_update_diagnostics_have_exact_fields_only(self):
        self._force_pending_condition(1)
        collection = self._complete_collection(1, self._batch(1))

        with patch.object(
            self.trainer,
            "collect_single_condition_batch",
            return_value=collection,
        ):
            output = self.trainer.train_step()

        diagnostics = output.diagnostics
        self.assertIsNotNone(diagnostics)
        manifest = diagnostics.to_dict()
        self.assertEqual(manifest["objective_kind"], "exact_tb")
        self.assertEqual(manifest["global_optimizer_step"], 1)
        self.assertEqual(manifest["trajectories_in_batch"], 2)
        self.assertEqual(manifest["total_trajectories_seen"], 2)
        self.assertEqual(manifest["requested_count"], 2)
        self.assertEqual(manifest["accepted_count"], 2)
        self.assertEqual(manifest["trajectory_length"], 1.0)
        self.assertEqual(manifest["terminal_success_rate"], 1.0)
        self.assertIn("exact_log_z", manifest)
        self.assertIn("tb_loss", manifest)
        self.assertIn("tb_delta_mean", manifest)
        self.assertIn("tb_delta_std", manifest)
        self.assertIn("tb_delta_rms", manifest)
        self.assertNotIn("zeta_mean", manifest)
        self.assertNotIn("unique_terminal_count", manifest)
        self.assertAlmostEqual(
            manifest["tb_delta_rms"],
            math.sqrt(manifest["tb_loss"]),
        )

    def test_lpv_diagnostics_use_population_stats_and_canonical_diversity(self):
        self._force_pending_condition(3)
        trajectories = self._batch(3, rewards=(1.0, 3.0))
        self.assertNotEqual(
            trajectories[0].terminal_state_hash,
            trajectories[1].terminal_state_hash,
        )
        collection = self._complete_collection(
            3,
            trajectories,
            invalid_count=1,
            retry_count=1,
        )

        with patch.object(
            self.trainer,
            "collect_single_condition_batch",
            return_value=collection,
        ):
            output = self.trainer.train_step()

        diagnostics = output.diagnostics
        self.assertIsNotNone(diagnostics)
        manifest = diagnostics.to_dict()
        self.assertEqual(manifest["objective_kind"], "log_partition_variance")
        self.assertEqual(manifest["invalid_count"], 1)
        self.assertEqual(manifest["retry_count"], 1)
        self.assertEqual(manifest["reward_mean"], 2.0)
        self.assertEqual(manifest["reward_std"], 1.0)
        self.assertEqual(manifest["unique_terminal_count"], 1)
        self.assertEqual(manifest["unique_terminal_fraction"], 0.5)
        self.assertAlmostEqual(manifest["variance_loss"], manifest["zeta_variance"])
        self.assertAlmostEqual(
            manifest["centered_zeta_rms"],
            manifest["zeta_std"],
        )
        for forbidden in (
            "exact_log_z",
            "selected_logZ",
            "learned_logZ",
            "tb_delta_mean",
            "tb_delta_std",
            "tb_delta_rms",
            "mean_expression_depth",
            "mean_node_count",
            "fraction_depth_eq_5",
            "fraction_nodes_eq_15",
        ):
            self.assertNotIn(forbidden, manifest)

    def test_incomplete_collection_does_not_step_or_commit(self):
        self._force_pending_condition(3)
        scheduler_before = self.trainer.complexity_scheduler.state_dict()
        parameter = next(self.trainer.model.parameters())
        parameter_before = parameter.detach().clone()
        collection = SingleConditionBatchCollection(
            condition_N=3,
            requested_count=2,
            trajectories=(),
            sampled_count=4,
            invalid_count=4,
            retry_count=2,
            retry_exhausted_count=2,
            sampling_rounds=2,
            sampling_seconds=0.0,
            reward_provider_seconds=0.0,
        )

        with patch.object(
            self.trainer,
            "collect_single_condition_batch",
            return_value=collection,
        ):
            output = self.trainer.train_step()

        self.assertFalse(output.updated)
        self.assertIsNone(output.objective)
        self.assertIsNone(output.diagnostics)
        self.assertEqual(output.global_optimizer_step, 0)
        self.assertEqual(
            self.trainer.complexity_scheduler.state_dict(),
            scheduler_before,
        )
        self.assertTrue(torch.equal(parameter.detach(), parameter_before))
        self.assertEqual(self.trainer.optimizer.state, {})
        self.assertEqual(self.trainer.total_trajectories_seen, 0)
        self.assertEqual(self.trainer.diagnostic_history, [])

    def test_exact_registry_configuration_uses_no_new_reward_evaluation(self):
        self.assertEqual(self.provider.evaluate_count, 0)
        self.assertEqual(set(self.trainer.registered_exact_masses_by_N), {1, 2})
        self.assertEqual(set(self.trainer.exhaustive_reuse_proofs_by_N), {1, 2})
        self.assertEqual(
            self.trainer.exact_tb_loss.exact_log_z_mask[:2].tolist(),
            [True, True],
        )


if __name__ == "__main__":
    unittest.main()
