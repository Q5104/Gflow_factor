import math
import unittest

import torch

from factor_gfn.gfn import (
    ComplexitySchedulerConfig,
    ExactMassResult,
    ForwardPolicyNetwork,
    GFNConfig,
    GFNTrainer,
    ModelConfig,
    NormalizerCalibrationConfig,
    RewardAssignment,
    SamplingConfig,
    StateAdapter,
    SyntheticRewardProvider,
    TrainingConfig,
    TrajectoryBalanceLoss,
    sample_trajectories,
)
from factor_gfn.grammar import ExactNodeGrammarState, SearchSpaceConfig, resolve_exact_node_strata
from factor_gfn.grammar.exact_node import _reachable_subtree_sizes_by_depth


MODEL = ModelConfig(
    d_model=16,
    num_heads=4,
    num_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    token_policy_mode="grammar_hierarchical",
)


class OriginalPlanCoverageTests(unittest.TestCase):
    def test_exact_n_no_dead_state_and_dynamic_max_nodes_depths(self):
        cases = (
            SearchSpaceConfig(max_depth=0, max_nodes=5),
            SearchSpaceConfig(max_depth=1, max_nodes=15),
            SearchSpaceConfig(max_depth=3, max_nodes=15),
            SearchSpaceConfig(max_depth=6, max_nodes=30),
        )
        for search_space in cases:
            with self.subTest(search_space=search_space):
                strata = resolve_exact_node_strata(search_space)
                objective = TrajectoryBalanceLoss(
                    max_nodes=search_space.max_nodes,
                    exact_node_counts=(strata.resolved_feasible_node_counts[0],),
                )
                self.assertEqual(objective.log_z_by_node_count.shape, (search_space.max_nodes,))
                for target in {
                    strata.resolved_feasible_node_counts[0],
                    strata.resolved_feasible_node_counts[-1],
                }:
                    state = ExactNodeGrammarState.source(
                        target_node_count=target,
                        search_space=search_space,
                    )
                    while not state.done:
                        legal = state.legal_transitions()
                        self.assertTrue(legal)
                        state = state.step(legal[-1])
                    self.assertEqual(state.node_count, target)
                    self.assertLessEqual(state.max_depth_seen, search_space.max_depth)

    def test_non_contiguous_feasible_strata_for_binary_only_toy_grammar(self):
        reachable = _reachable_subtree_sizes_by_depth(
            max_depth=3,
            max_nodes=15,
            arities=(0, 2),
        )[0]
        feasible = tuple(index for index in range(1, 16) if index in reachable)
        self.assertEqual(feasible, (1, 3, 5, 7, 9, 11, 13, 15))

    def test_exact_and_learned_mixed_n_batch_selects_gradients_correctly(self):
        search_space = SearchSpaceConfig(max_depth=1, max_nodes=2)
        model = ForwardPolicyNetwork(MODEL, search_space)
        trajectories = sample_trajectories(
            model,
            StateAdapter(search_space),
            num_trajectories=2,
            target_node_counts=(1, 2),
        )
        for trajectory, reward in zip(trajectories, (1.5, 2.5), strict=True):
            trajectory.attach_reward(reward, math.log(reward))
        objective = TrajectoryBalanceLoss(max_nodes=2, exact_node_counts=(1,))
        objective.set_exact_log_z(1, math.log(6.0))
        exact_before = objective.exact_tb_log_z_by_node_count.clone()
        output = objective(trajectories)
        output.loss.backward()
        self.assertEqual(float(objective.log_z_by_node_count.grad[0]), 0.0)
        self.assertNotEqual(float(objective.log_z_by_node_count.grad[1]), 0.0)
        self.assertTrue(torch.equal(exact_before, objective.exact_tb_log_z_by_node_count))


class FailureAndIsolationIntegrationTests(unittest.TestCase):
    def test_skipped_mixed_n_update_counts_every_slot_without_success(self):
        class RejectN2Provider(SyntheticRewardProvider):
            def evaluate(self, expression):
                if expression.stats.node_count == 2:
                    return RewardAssignment(valid=False, rejection_reason="reject N=2")
                return super().evaluate(expression)

        config = GFNConfig(
            search_space=SearchSpaceConfig(max_depth=1, max_nodes=2),
            model=MODEL,
            complexity_scheduler=ComplexitySchedulerConfig(
                enabled=True,
                exact_node_retry_budget=1,
            ),
            training=TrainingConfig(batch_size=2, max_steps=1, seed=913),
        )
        trainer = GFNTrainer(config, RejectN2Provider(), device="cpu")
        stats = trainer.train_step()
        self.assertTrue(stats.skipped_update)
        self.assertEqual(trainer.requested_count_by_N, {1: 1, 2: 1})
        self.assertEqual(trainer.sampled_attempt_count_by_N, {1: 1, 2: 2})
        self.assertEqual(trainer.successful_update_count_by_N, {1: 0, 2: 0})
        self.assertEqual(trainer.optimizer_step, 0)

    def test_trainer_calibration_insufficient_samples_fails_closed(self):
        class RejectN2Provider(SyntheticRewardProvider):
            def evaluate(self, expression):
                if expression.stats.node_count == 2:
                    return RewardAssignment(
                        valid=False,
                        rejection_reason="reject calibration N=2",
                    )
                return super().evaluate(expression)

        provider = RejectN2Provider()
        config = GFNConfig(
            search_space=SearchSpaceConfig(max_depth=1, max_nodes=2),
            model=MODEL,
            complexity_scheduler=ComplexitySchedulerConfig(
                enabled=True,
                exhaustive_node_counts=(1,),
            ),
            calibration=NormalizerCalibrationConfig(
                enabled=True,
                minimum_valid_calibration_samples=1,
                maximum_requested_calibration_slots_per_N=1,
            ),
            training=TrainingConfig(batch_size=1, max_steps=1, seed=914),
        )
        trainer = GFNTrainer(config, provider, device="cpu")
        trainer.register_exact_mass_result(
            ExactMassResult(
                node_count=1,
                valid_candidate_count=6,
                invalid_candidate_count=0,
                exact_raw_reward_log_mass=1.0,
                raw_reward_mass_status="positive_mass",
                exact_tb_log_z=1.0,
                reward_floor=config.reward.reward_floor,
                provider_fingerprint=provider.fingerprint(),
                context_fingerprint=provider.manifest()["context_fingerprint"],
                aggregation_fingerprint="synthetic-calibration-integration",
            )
        )
        with self.assertRaisesRegex(RuntimeError, "only 0 valid samples"):
            while True:
                trainer.calibration_step()
        self.assertEqual(trainer.calibration.status, "failed")


if __name__ == "__main__":
    unittest.main()
