import math
import tempfile
import unittest
from pathlib import Path

import torch

from factor_gfn.gfn import (
    ExhaustivePlanningConfig,
    ExhaustiveRegistry,
    HybridVarianceTrainer,
    SyntheticRewardProvider,
    build_stage5_hybrid_variance_5_15_config,
    create_hybrid_variance_runner,
    resolve_exhaustive_plan,
    resume_hybrid_variance_runner,
)
from factor_gfn.grammar import Expression, SearchSpaceConfig


class HybridEndToEndSyntheticSmokeTests(unittest.TestCase):
    def test_one_real_sampler_cycle_checkpoints_and_resumes(self):
        """Exercise sampler -> objectives -> updates -> checkpoint -> resume."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = SyntheticRewardProvider()
            config = build_stage5_hybrid_variance_5_15_config(
                max_cycles=1,
                trajectories_per_batch=2,
            )
            registry = ExhaustiveRegistry(root / "synthetic_exact_registry.sqlite3")
            try:
                source_space = SearchSpaceConfig(max_depth=6, max_nodes=20)
                plan = resolve_exhaustive_plan(
                    source_space,
                    ExhaustivePlanningConfig(
                        explicit_include_node_counts=(1, 2),
                        explicit_exclude_node_counts=tuple(range(3, 21)),
                    ),
                )
                registry.register_plan(
                    plan,
                    provider_fingerprint=provider.fingerprint(),
                    context_fingerprint=(
                        provider.manifest()["context_fingerprint"]
                    ),
                )
                for node_count in (1, 2):
                    for candidate in registry.pending_candidates(node_count):
                        expression = Expression.from_prefix(
                            candidate.prefix_token_ids
                        )
                        assignment = provider.evaluate(expression)
                        registry.record_evaluation(
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
                    registry.compute_exact_masses(
                        node_count,
                        reward_floor=config.reward.reward_floor,
                    )

                trainer = HybridVarianceTrainer(config, provider, device="cpu")
                semantics = trainer.target_exhaustive_reuse_semantics()
                trainer.configure_hybrid_exhaustive_registry(
                    registry,
                    source_semantics_by_N={1: semantics, 2: semantics},
                )
                runner = create_hybrid_variance_runner(
                    trainer,
                    root / "synthetic_hybrid_run",
                )
                outputs = runner.run_attempts(
                    config.training.total_optimizer_steps
                )

                self.assertTrue(all(output.updated for output in outputs))
                self.assertTrue(runner.complete)
                self.assertEqual(trainer.optimizer_step, 15)
                self.assertEqual(trainer.total_trajectories_seen, 30)
                self.assertEqual(len(trainer.diagnostic_history), 15)
                self.assertEqual(
                    {item.condition_N for item in trainer.diagnostic_history},
                    set(range(1, 16)),
                )
                self.assertEqual(
                    [item.condition_position_in_cycle for item in trainer.diagnostic_history],
                    list(range(15)),
                )
                self.assertEqual(
                    sum(item.objective_kind == "exact_tb" for item in trainer.diagnostic_history),
                    2,
                )
                self.assertEqual(
                    sum(
                        item.objective_kind == "log_partition_variance"
                        for item in trainer.diagnostic_history
                    ),
                    13,
                )
                self.assertTrue(
                    all(
                        math.isfinite(float(item.policy_grad_norm))
                        for item in trainer.diagnostic_history
                    )
                )
                self.assertTrue(runner.latest_checkpoint_path.is_file())
                payload = torch.load(
                    runner.latest_checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(payload["global_optimizer_step"], 15)
                self.assertEqual(payload["total_trajectories_seen"], 30)
                self.assertEqual(payload["scheduler_state"]["position"], 15)

                resumed_provider = SyntheticRewardProvider()
                resumed = HybridVarianceTrainer(
                    config,
                    resumed_provider,
                    device="cpu",
                )
                resumed_semantics = resumed.target_exhaustive_reuse_semantics()
                resumed.configure_hybrid_exhaustive_registry(
                    registry,
                    source_semantics_by_N={
                        1: resumed_semantics,
                        2: resumed_semantics,
                    },
                )
                resumed_runner = resume_hybrid_variance_runner(
                    runner.run_dir,
                    resumed,
                )
                self.assertTrue(resumed_runner.complete)
                self.assertEqual(resumed.optimizer_step, 15)
                self.assertEqual(resumed.total_trajectories_seen, 30)
                self.assertEqual(
                    resumed.diagnostic_history,
                    trainer.diagnostic_history,
                )
                self.assertEqual(
                    resumed.complexity_scheduler.state_dict(),
                    trainer.complexity_scheduler.state_dict(),
                )
            finally:
                registry.close()


if __name__ == "__main__":
    unittest.main()
