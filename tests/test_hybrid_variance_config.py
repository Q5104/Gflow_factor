from __future__ import annotations

import unittest

from factor_gfn.gfn import (
    HYBRID_VARIANCE_CONFIG_SCHEMA,
    STAGE5_HYBRID_CONDITIONS,
    STAGE5_HYBRID_DEFAULT_TRAJECTORIES_PER_BATCH,
    STAGE5_HYBRID_EXACT_TB_CONDITIONS,
    STAGE5_HYBRID_LPV_CONDITIONS,
    HybridObjectiveConfig,
    HybridTrainingConfig,
    HybridVarianceGFNConfig,
    build_stage5_hybrid_variance_5_15_config,
)
from factor_gfn.gfn.no_anchor_config import (
    FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT,
    build_frozen_stage5_no_anchor_6_20_config,
)
from factor_gfn.grammar import ExactNodeGrammarState, SearchSpaceConfig


class HybridVarianceConfigTests(unittest.TestCase):
    def test_formal_builder_freezes_5_15_contract_and_training_units(self):
        config = build_stage5_hybrid_variance_5_15_config(max_cycles=100)

        self.assertEqual(config.search_space.max_depth, 5)
        self.assertEqual(config.search_space.max_nodes, 15)
        self.assertEqual(config.resolved_condition_node_counts, tuple(range(1, 16)))
        self.assertEqual(config.objective.condition_node_counts, tuple(range(1, 16)))
        self.assertEqual(
            config.objective.exact_tb_node_counts,
            STAGE5_HYBRID_EXACT_TB_CONDITIONS,
        )
        self.assertEqual(config.objective.lpv_node_counts, STAGE5_HYBRID_LPV_CONDITIONS)
        self.assertEqual(config.training.trajectories_per_batch, 16)
        self.assertEqual(config.training.optimizer_steps_per_cycle, 15)
        self.assertEqual(config.training.trajectories_per_cycle, 240)
        self.assertEqual(config.training.total_optimizer_steps, 1_500)
        self.assertEqual(config.training.total_training_trajectories, 24_000)
        self.assertEqual(config.manifest()["schema"], HYBRID_VARIANCE_CONFIG_SCHEMA)

    def test_k_is_configurable_and_not_part_of_the_condition_partition(self):
        config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=3,
            trajectories_per_batch=32,
        )

        self.assertEqual(config.training.trajectories_per_batch, 32)
        self.assertEqual(config.training.trajectories_per_cycle, 480)
        self.assertEqual(config.training.total_training_trajectories, 1_440)
        self.assertEqual(config.objective.condition_node_counts, STAGE5_HYBRID_CONDITIONS)
        self.assertEqual(STAGE5_HYBRID_DEFAULT_TRAJECTORIES_PER_BATCH, 16)

    def test_k_requires_an_integer_at_least_two(self):
        for invalid in (1, 0, -1, True, 1.5, "16"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    HybridTrainingConfig(
                        max_cycles=1,
                        trajectories_per_batch=invalid,  # type: ignore[arg-type]
                    )

    def test_cycle_budget_requires_a_positive_integer(self):
        for invalid in (0, -1, True, 1.5, "10"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    HybridTrainingConfig(max_cycles=invalid)  # type: ignore[arg-type]

    def test_objective_partition_is_frozen(self):
        with self.assertRaisesRegex(ValueError, "exact-TB conditions"):
            HybridObjectiveConfig(exact_tb_node_counts=(1,))
        with self.assertRaisesRegex(ValueError, "LPV conditions"):
            HybridObjectiveConfig(lpv_node_counts=tuple(range(3, 15)))
        with self.assertRaisesRegex(ValueError, "objective_mode"):
            HybridObjectiveConfig(objective_mode="legacy_tb")  # type: ignore[arg-type]

    def test_policy_optimizer_contract_is_frozen_without_logz_fields(self):
        config = build_stage5_hybrid_variance_5_15_config(max_cycles=1)
        training_manifest = config.manifest()["config"]["training"]

        self.assertEqual(training_manifest["optimizer"], "adam")
        self.assertEqual(training_manifest["learning_rate"], 1e-4)
        self.assertEqual(training_manifest["model_gradient_clip_norm"], 5.0)
        self.assertFalse(any("log_z" in key for key in training_manifest))
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            HybridTrainingConfig(max_cycles=1, learning_rate=2e-4)
        with self.assertRaisesRegex(ValueError, "gradient_clip"):
            HybridTrainingConfig(max_cycles=1, model_gradient_clip_norm=1.0)

    def test_hybrid_config_rejects_search_or_architecture_drift(self):
        training = HybridTrainingConfig(max_cycles=1)
        with self.assertRaisesRegex(ValueError, "max_depth=5, max_nodes=15"):
            HybridVarianceGFNConfig(
                training=training,
                search_space=SearchSpaceConfig(max_depth=6, max_nodes=15),
            )

    def test_exact_node_conditions_respect_frozen_boundaries(self):
        config = build_stage5_hybrid_variance_5_15_config(max_cycles=1)
        for target in config.resolved_condition_node_counts:
            state = ExactNodeGrammarState.source(
                target_node_count=target,
                search_space=config.search_space,
            )
            while not state.done:
                state = state.step(state.legal_transitions()[0])
                self.assertLessEqual(state.node_count, 15)
                self.assertLessEqual(state.max_depth_seen, 5)
            self.assertEqual(state.node_count, target)

    def test_fingerprint_is_stable_and_budget_sensitive(self):
        first = build_stage5_hybrid_variance_5_15_config(max_cycles=10)
        second = build_stage5_hybrid_variance_5_15_config(max_cycles=10)
        changed_budget = build_stage5_hybrid_variance_5_15_config(max_cycles=11)
        changed_k = build_stage5_hybrid_variance_5_15_config(
            max_cycles=10,
            trajectories_per_batch=32,
        )

        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertNotEqual(first.fingerprint(), changed_budget.fingerprint())
        self.assertNotEqual(first.fingerprint(), changed_k.fingerprint())

    def test_legacy_frozen_fingerprint_is_unchanged(self):
        legacy = build_frozen_stage5_no_anchor_6_20_config()
        self.assertEqual(
            legacy.fingerprint(),
            FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT,
        )


if __name__ == "__main__":
    unittest.main()
