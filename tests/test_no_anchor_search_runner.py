import unittest
from pathlib import Path
from unittest.mock import patch

from factor_gfn.gfn.no_anchor_config import (
    FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT,
    STAGE5_LOGZ_ADAM_LR2E2_AB_CONFIG_FINGERPRINT,
    STAGE5_LOGZ_SGD_LR1E1_B1_CONFIG_FINGERPRINT,
)
from factor_gfn.gfn.no_anchor_search_runner import (
    NO_ANCHOR_LOGZ_ADAM_LR2E2_AB_SEARCH_SCHEMA,
    NO_ANCHOR_LOGZ_SGD_LR1E1_B1_SEARCH_SCHEMA,
    NO_ANCHOR_REAL_SEARCH_SCHEMA,
    _require_frozen_settings,
    create_no_anchor_logz_adam_lr2e2_ab_runner,
    create_no_anchor_logz_sgd_lr1e1_b1_runner,
    create_no_anchor_real_search_runner,
)
from factor_gfn.gfn.search_runner import RealSearchSettings


class NoAnchorSearchRunnerContractTests(unittest.TestCase):
    def test_schema_is_distinct_from_legacy_real_search(self):
        self.assertEqual(
            NO_ANCHOR_REAL_SEARCH_SCHEMA,
            "factor_gfn.no_anchor_real_search.v1",
        )
        self.assertEqual(
            NO_ANCHOR_LOGZ_ADAM_LR2E2_AB_SEARCH_SCHEMA,
            "factor_gfn.no_anchor_logz_adam_lr2e2_ab_search.v1",
        )
        self.assertNotEqual(
            NO_ANCHOR_LOGZ_ADAM_LR2E2_AB_SEARCH_SCHEMA,
            NO_ANCHOR_REAL_SEARCH_SCHEMA,
        )
        self.assertEqual(
            NO_ANCHOR_LOGZ_SGD_LR1E1_B1_SEARCH_SCHEMA,
            "factor_gfn.no_anchor_logz_sgd_lr1e1_b1_search.v1",
        )

    def test_formal_settings_freeze_1000_steps_and_seed42(self):
        valid = RealSearchSettings(
            max_steps=1000,
            seed=42,
            device="cuda:0",
            run_root=Path("runs/stage5_no_anchor_formal_6_20"),
        )
        _require_frozen_settings(valid)
        with self.assertRaisesRegex(ValueError, "max_steps"):
            _require_frozen_settings(
                RealSearchSettings(max_steps=10, seed=42, device="cuda:0")
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            _require_frozen_settings(
                RealSearchSettings(max_steps=1000, seed=7, device="cuda:0")
            )

    @patch("factor_gfn.gfn.no_anchor_search_runner._create_initialized_no_anchor_runner")
    def test_formal_and_experiment_a_factories_are_schema_isolated(self, create):
        settings = RealSearchSettings(max_steps=1000, seed=42, device="cuda:0")
        common = {
            "registry_path": Path("registry.sqlite3"),
            "historical_diagnostic_root": Path("diagnostic"),
            "targeted_artifact_path": Path("targeted.json"),
        }
        create_no_anchor_real_search_runner(settings, **common)
        formal = create.call_args.kwargs
        self.assertEqual(formal["search_schema"], NO_ANCHOR_REAL_SEARCH_SCHEMA)
        self.assertEqual(
            formal["config"].fingerprint(),
            FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT,
        )

        create_no_anchor_logz_adam_lr2e2_ab_runner(settings, **common)
        experiment = create.call_args.kwargs
        self.assertEqual(
            experiment["search_schema"],
            NO_ANCHOR_LOGZ_ADAM_LR2E2_AB_SEARCH_SCHEMA,
        )
        self.assertEqual(
            experiment["config"].fingerprint(),
            STAGE5_LOGZ_ADAM_LR2E2_AB_CONFIG_FINGERPRINT,
        )
        self.assertEqual(
            experiment["experiment_contract"]["single_changed_training_field"],
            "log_z_learning_rate",
        )

        create_no_anchor_logz_sgd_lr1e1_b1_runner(settings, **common)
        b1 = create.call_args.kwargs
        self.assertEqual(
            b1["search_schema"],
            NO_ANCHOR_LOGZ_SGD_LR1E1_B1_SEARCH_SCHEMA,
        )
        self.assertEqual(
            b1["config"].fingerprint(),
            STAGE5_LOGZ_SGD_LR1E1_B1_CONFIG_FINGERPRINT,
        )
        self.assertEqual(b1["normalizer_optimizer"], "sgd")
        self.assertEqual(b1["experiment_contract"]["normalizer_learning_rate"], 0.1)
        self.assertEqual(
            b1["experiment_contract"]["safety_gate_successful_optimizer_updates"],
            [20, 30],
        )


if __name__ == "__main__":
    unittest.main()
