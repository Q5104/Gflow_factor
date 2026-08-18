import unittest
from dataclasses import asdict, fields

from factor_gfn.gfn.config import TrainingConfig
from factor_gfn.gfn.no_anchor_config import (
    FORMAL_STAGE5_MAX_DEPTH,
    FORMAL_STAGE5_MAX_NODES,
    FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT,
    FORMAL_STAGE5_NO_ANCHOR_MAX_STEPS,
    FORMAL_STAGE5_NO_ANCHOR_SEED,
    STAGE5_LOGZ_ADAM_LR2E2_AB_CONFIG_FINGERPRINT,
    STAGE5_LOGZ_ADAM_LR2E2_AB_EXPERIMENT_ID,
    STAGE5_LOGZ_SGD_LR1E1_B1_CONFIG_FINGERPRINT,
    STAGE5_LOGZ_SGD_LR1E1_B1_EXPERIMENT_ID,
    NO_ANCHOR_CONFIG_SCHEMA,
    ExhaustiveRegistryReuseConfig,
    HistoricalLogZInitializationConfig,
    NoAnchorCalibrationConfig,
    NoAnchorComplexityConfig,
    NoAnchorGFNConfig,
    build_formal_stage5_no_anchor_6_20_config,
    build_frozen_stage5_no_anchor_6_20_config,
    build_stage5_logz_adam_lr2e2_ab_config,
    build_stage5_logz_sgd_lr1e1_b1_config,
)
from factor_gfn.grammar import SearchSpaceConfig


def _contains_anchor_key(value) -> bool:
    if isinstance(value, dict):
        return any(
            "anchor" in str(key).lower() or _contains_anchor_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_anchor_key(item) for item in value)
    return False


class NoAnchorConfigTests(unittest.TestCase):
    def test_frozen_formal_stage5_contract_and_fingerprint(self):
        config = build_frozen_stage5_no_anchor_6_20_config()
        self.assertEqual(config.fingerprint(), FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT)
        self.assertEqual(config.training.max_steps, FORMAL_STAGE5_NO_ANCHOR_MAX_STEPS)
        self.assertEqual(config.training.seed, FORMAL_STAGE5_NO_ANCHOR_SEED)
        self.assertEqual(config.training.batch_size, 8)
        self.assertEqual(config.training.learning_rate, 1e-4)
        self.assertEqual(config.training.log_z_learning_rate, 1e-2)
        self.assertEqual(config.training.model_gradient_clip_norm, 5.0)
        self.assertEqual(config.training.log_z_gradient_clip_norm, 5.0)
        self.assertEqual(config.complexity.exact_node_retry_budget, 3)
        self.assertEqual(config.calibration.target_node_counts, (17, 18))

    def test_logz_adam_lr2e2_ab_changes_only_logz_learning_rate(self):
        baseline = build_frozen_stage5_no_anchor_6_20_config()
        experiment = build_stage5_logz_adam_lr2e2_ab_config()
        self.assertEqual(
            experiment.fingerprint(),
            STAGE5_LOGZ_ADAM_LR2E2_AB_CONFIG_FINGERPRINT,
        )
        self.assertEqual(STAGE5_LOGZ_ADAM_LR2E2_AB_EXPERIMENT_ID, "logz_adam_lr2e2_seed42")
        baseline_payload = asdict(baseline)
        experiment_payload = asdict(experiment)
        baseline_training = baseline_payload.pop("training")
        experiment_training = experiment_payload.pop("training")
        self.assertEqual(experiment_payload, baseline_payload)
        changed_training_fields = {
            name
            for name in baseline_training
            if baseline_training[name] != experiment_training[name]
        }
        self.assertEqual(changed_training_fields, {"log_z_learning_rate"})
        self.assertEqual(baseline_training["log_z_learning_rate"], 1e-2)
        self.assertEqual(experiment_training["log_z_learning_rate"], 2e-2)

    def test_logz_sgd_lr1e1_b1_freezes_parameter_contract(self):
        baseline = build_frozen_stage5_no_anchor_6_20_config()
        experiment = build_stage5_logz_sgd_lr1e1_b1_config()
        self.assertEqual(
            experiment.fingerprint(),
            STAGE5_LOGZ_SGD_LR1E1_B1_CONFIG_FINGERPRINT,
        )
        self.assertEqual(
            STAGE5_LOGZ_SGD_LR1E1_B1_EXPERIMENT_ID,
            "logz_sgd_lr1e1_seed42",
        )
        baseline_payload = asdict(baseline)
        experiment_payload = asdict(experiment)
        baseline_training = baseline_payload.pop("training")
        experiment_training = experiment_payload.pop("training")
        self.assertEqual(experiment_payload, baseline_payload)
        self.assertEqual(
            {
                name
                for name in baseline_training
                if baseline_training[name] != experiment_training[name]
            },
            {"log_z_learning_rate"},
        )
        self.assertEqual(experiment_training["log_z_learning_rate"], 1e-1)

    def test_formal_factory_freezes_boundary_but_not_training_hyperparameters(self):
        training = TrainingConfig(
            batch_size=7,
            learning_rate=3e-4,
            log_z_learning_rate=4e-3,
            max_steps=19,
            seed=11,
        )
        config = build_formal_stage5_no_anchor_6_20_config(
            training=training,
            seed=23,
        )
        self.assertEqual(config.search_space.max_depth, FORMAL_STAGE5_MAX_DEPTH)
        self.assertEqual(config.search_space.max_nodes, FORMAL_STAGE5_MAX_NODES)
        self.assertEqual((FORMAL_STAGE5_MAX_DEPTH, FORMAL_STAGE5_MAX_NODES), (6, 20))
        self.assertEqual(config.training.batch_size, 7)
        self.assertEqual(config.training.learning_rate, 3e-4)
        self.assertEqual(config.training.log_z_learning_rate, 4e-3)
        self.assertEqual(config.training.seed, 23)

    def test_resolved_contract_is_F_equals_D_with_E_and_L_partition(self):
        config = build_formal_stage5_no_anchor_6_20_config(
            training=TrainingConfig(max_steps=1)
        )
        strata = config.resolved_strata()
        self.assertEqual(strata.feasible_node_counts, tuple(range(1, 21)))
        self.assertEqual(strata.discovery_node_counts, tuple(range(1, 21)))
        self.assertEqual(strata.exact_normalizer_node_counts, (1, 2))
        self.assertEqual(strata.learned_normalizer_node_counts, tuple(range(3, 21)))

    def test_formal_config_and_manifest_have_no_anchor_state(self):
        config = build_formal_stage5_no_anchor_6_20_config(
            training=TrainingConfig(max_steps=1)
        )
        field_names = {field.name for field in fields(NoAnchorGFNConfig)}
        self.assertFalse(any("anchor" in name.lower() for name in field_names))
        manifest = config.manifest()
        self.assertEqual(manifest["schema"], NO_ANCHOR_CONFIG_SCHEMA)
        self.assertFalse(_contains_anchor_key(manifest))

    def test_historical_initialization_reuses_only_verified_constants(self):
        initialization = HistoricalLogZInitializationConfig()
        payload = asdict(initialization)
        self.assertEqual(payload["mode"], "verified_historical_median")
        self.assertEqual(payload["source_scope"], "training_only_diagnostic")
        self.assertEqual(payload["reuse_scope"], "initialization_constants_only")
        self.assertTrue(payload["require_semantics_equivalence"])
        self.assertFalse(any(
            value for key, value in payload.items() if key.startswith("restore_")
        ))
        with self.assertRaisesRegex(ValueError, "must not restore training state"):
            HistoricalLogZInitializationConfig(restore_optimizer_state=True)

    def test_calibration_contract_is_targeted_fallback_only(self):
        calibration = NoAnchorCalibrationConfig()
        payload = asdict(calibration)
        self.assertEqual(
            payload,
            {
                "enabled": False,
                "target_node_counts": (),
                "minimum_valid_samples": 64,
                "maximum_requested_slots_per_N": 128,
                "comparison_window": 16,
                "median_absolute_tolerance": 0.25,
                "iqr_absolute_tolerance": 0.5,
                "requires_fresh_training_state": True,
                "allow_mid_training_log_z_reset": False,
            },
        )
        self.assertFalse(any("relative" in key for key in payload))
        with self.assertRaisesRegex(ValueError, "must be 0.25"):
            NoAnchorCalibrationConfig(median_absolute_tolerance=0.2)
        enabled = NoAnchorCalibrationConfig(enabled=True, target_node_counts=(18,))
        self.assertEqual(enabled.target_node_counts, (18,))
        with self.assertRaisesRegex(ValueError, "requires target_node_counts"):
            NoAnchorCalibrationConfig(enabled=True)
        with self.assertRaisesRegex(ValueError, "mid-training logZ reset is forbidden"):
            NoAnchorCalibrationConfig(allow_mid_training_log_z_reset=True)

    def test_targeted_recalibration_must_only_name_L_strata(self):
        with self.assertRaisesRegex(ValueError, "subset of L"):
            NoAnchorGFNConfig(
                search_space=SearchSpaceConfig(max_depth=2, max_nodes=2),
                complexity=NoAnchorComplexityConfig(
                    exact_normalizer_node_counts=(1,),
                ),
                calibration=NoAnchorCalibrationConfig(
                    enabled=True,
                    target_node_counts=(1,),
                ),
                training=TrainingConfig(max_steps=1),
            )

    def test_manifest_records_historical_initialization_and_targeted_fallback(self):
        manifest = build_formal_stage5_no_anchor_6_20_config(
            training=TrainingConfig(max_steps=1)
        ).manifest()
        self.assertEqual(
            manifest["normalizer_initialization"]["mode"],
            "verified_historical_median",
        )
        self.assertTrue(
            manifest["normalizer_initialization"]["training_state_restore_forbidden"]
        )
        self.assertEqual(
            manifest["normalizer_calibration_scope"],
            "targeted_problem_strata_only",
        )
        self.assertEqual(manifest["mid_training_log_z_reset"], "forbidden")

    def test_registry_equivalence_is_once_then_hash_only_lookup(self):
        config = ExhaustiveRegistryReuseConfig()
        self.assertEqual(
            config.equivalence_verification_phase,
            "run_initialization_once",
        )
        self.assertEqual(config.discovery_lookup_key, "structural_hash")
        self.assertFalse(config.reenumerate_during_candidate_lookup)
        manifest = NoAnchorGFNConfig(
            search_space=SearchSpaceConfig(max_depth=6, max_nodes=20),
            training=TrainingConfig(max_steps=1),
        ).resolved_strata().manifest()
        self.assertEqual(
            manifest["exact_registry_equivalence_verification"],
            "run_initialization_once",
        )
        self.assertFalse(manifest["canonical_reenumeration_per_discovery_candidate"])

    def test_generic_no_anchor_config_remains_search_space_configurable(self):
        config = NoAnchorGFNConfig(
            search_space=SearchSpaceConfig(max_depth=5, max_nodes=15),
            training=TrainingConfig(max_steps=1),
        )
        strata = config.resolved_strata()
        self.assertEqual(strata.feasible_node_counts, tuple(range(1, 16)))
        self.assertEqual(strata.discovery_node_counts, strata.feasible_node_counts)
        self.assertEqual(len(config.fingerprint()), 64)


if __name__ == "__main__":
    unittest.main()
