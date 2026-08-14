import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NoAnchorNotebookTests(unittest.TestCase):
    def _read(self, name: str):
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{name}:{index}")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        return notebook, code

    def test_synthetic_smoke_has_no_anchor_and_covers_checkpoint_retry_and_progress(self):
        _, code = self._read("run_no_anchor_integration_smoke_6_20.ipynb")
        for marker in (
            "max_depth=6, max_nodes=20",
            "resolved_discovery_node_counts == trainer.resolved_feasible_node_counts",
            "factor_gfn.checkpoint.no_anchor.v1",
            "NO_ANCHOR_SYNTHETIC_SMOKE_OK",
            "sampled_attempt_count_by_N",
            "[calibration-synthetic]",
            "run_training_with_progress",
        ):
            self.assertIn(marker, code)
        self.assertNotIn("ExhaustiveAnchor", code)

    def test_step12_is_safety_locked_training_only_and_reports_required_health(self):
        notebook, code = self._read("run_step12_no_anchor_training_health_6_20.ipynb")
        for marker in (
            "RUN_REAL_STEP12 = False",
            "data_scope",
            "validation_oos_loaded",
            "initialize_verified_historical_log_z",
            "ExhaustiveRegistry(SOURCE_REGISTRY, read_only=True)",
            "run_training_with_progress",
            "progress_heartbeat",
            "initialization_pre_update",
            "successful_gradient_exposures",
            "targeted_recalibration_review",
            "insufficient_evidence_node_counts",
            "model_gradient_norm_before_clip",
            "log_z_gradient_before_clip",
            "retry_exhausted_count_by_N",
            "cuda_peak_memory_bytes",
            "checkpoint_resume_exact",
        ):
            self.assertIn(marker, code)
        self.assertNotIn("run_calibration_with_progress", code)
        self.assertNotIn("build_or_resume_n1_n2_registry", code)
        self.assertNotIn("build_depth_boundary_diagnostic", code)
        self.assertNotIn("ExhaustiveAnchor", code)
        self.assertNotIn("validation_ic", code)
        self.assertNotIn("oos_ic", code)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertGreater(index, 0)
            previous = notebook["cells"][index - 1]
            self.assertEqual(previous["cell_type"], "markdown")
            explanation = "".join(previous["source"])
            self.assertIn("### 第", explanation)

    def test_targeted_calibration_is_locked_resumable_and_only_targets_17_18(self):
        notebook, code = self._read(
            "run_targeted_logz_calibration_n17_n18_6_20.ipynb"
        )
        for marker in (
            "RUN_REAL_TARGETED_CALIBRATION = False",
            "TARGET_NODE_COUNTS = (17, 18)",
            "exact_node_retry_budget=3",
            "minimum_valid_samples=64",
            "maximum_requested_slots_per_N=128",
            "comparison_window=16",
            "median_absolute_tolerance=0.25",
            "iqr_absolute_tolerance=0.50",
            "ExhaustiveRegistry(SOURCE_REGISTRY, read_only=True)",
            "initialize_verified_historical_log_z",
            "load_targeted_calibration_progress",
            "run_targeted_calibration_with_progress",
            "write_targeted_calibration_artifact",
            "initialize_verified_targeted_log_z",
            "TARGETED_CALIBRATION_ACCEPTANCE_OK",
        ):
            self.assertIn(marker, code)
        self.assertNotIn("train_step()", code)
        self.assertNotIn("run_training_with_progress", code)
        self.assertNotIn("validation_ic", code)
        self.assertNotIn("oos_ic", code)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertGreater(index, 0)
            previous = notebook["cells"][index - 1]
            self.assertEqual(previous["cell_type"], "markdown")
            self.assertIn("### 第", "".join(previous["source"]))

    def test_policy_clip_comparison_is_locked_fair_and_progress_visible(self):
        notebook, code = self._read(
            "run_policy_clip_comparison_5_vs_20_6_20.ipynb"
        )
        for marker in (
            "RUN_REAL_CLIP_COMPARISON = False",
            "CLIP_VALUES = (5.0, 20.0)",
            "TARGET_SUCCESSFUL_UPDATES = 16",
            "MAX_LOGICAL_BATCHES_PER_BRANCH = 32",
            "exact_node_retry_budget=3",
            "target_node_counts=(17, 18)",
            "targeted_log_z_engineering_initialization.json",
            "high_variance_engineering_estimate",
            "policy_state_fingerprint",
            "scheduler_state_fingerprint",
            "run_training_with_progress",
            "trainer.load_checkpoint(branch['checkpoint'])",
            "selection_uses_reward_or_ic': False",
            "manual_review_required",
        ):
            self.assertIn(marker, code)
        self.assertNotIn("validation_ic", code)
        self.assertNotIn("oos_ic", code)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertGreater(index, 0)
            previous = notebook["cells"][index - 1]
            self.assertEqual(previous["cell_type"], "markdown")
            self.assertIn("### ", "".join(previous["source"]))

    def test_final_health_confirmation_is_fresh_locked_and_auditable(self):
        notebook, code = self._read(
            "run_final_no_anchor_health_confirmation_6_20.ipynb"
        )
        for marker in (
            "RUN_REAL_FINAL_HEALTH = False",
            "LOGICAL_BATCHES = 32",
            "model_gradient_clip_norm=5.0",
            "exact_node_retry_budget=3",
            "target_node_counts=(17, 18)",
            "targeted_log_z_engineering_initialization.json",
            "high_variance_engineering_estimate",
            "run_training_with_progress",
            "progress_heartbeat",
            "initialization_pre_update",
            "successful_gradient_exposures",
            "insufficient_evidence_node_counts",
            "checkpoint_resume_exact",
            "manual_review_required",
            "FINAL_NO_ANCHOR_HEALTH_CONFIRMATION_COMPLETE",
        ):
            self.assertIn(marker, code)
        self.assertNotIn("validation_ic", code)
        self.assertNotIn("oos_ic", code)
        self.assertNotIn("run_calibration_with_progress", code)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertGreater(index, 0)
            previous = notebook["cells"][index - 1]
            self.assertEqual(previous["cell_type"], "markdown")
            self.assertIn("### ", "".join(previous["source"]))

    def test_formal_stage5_notebook_is_locked_segmented_and_compact(self):
        notebook, code = self._read("run_stage5_no_anchor_formal_6_20.ipynb")
        for marker in (
            "RUN_FORMAL_STAGE5=True",
            "MODE='resume'",
            "TARGET_STEP=1000",
            "PLANNED_MAX_STEPS=1000",
            "c3a1c2747cbb41dbbb3f8f23e6ddddcb",
            "FORMAL_STAGE5_NO_ANCHOR_CONFIG_FINGERPRINT",
            "create_no_anchor_real_search_runner",
            "resume_no_anchor_real_search_runner",
            "runner.run_until(TARGET_STEP)",
            "validation_oos_loaded",
            "subexpression_cache_max_bytes=0",
            "每完成一个step只输出一行",
            "先把上述run目录交给Codex检查",
            "FORMAL_STAGE5_SEGMENT_COMPLETE",
        ):
            self.assertIn(marker, code + "\n" + "\n".join(
                "".join(cell["source"]) for cell in notebook["cells"]
            ))
        self.assertNotIn("validation_ic", code)
        self.assertNotIn("oos_ic", code)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertGreater(index, 0)
            previous = notebook["cells"][index - 1]
            self.assertEqual(previous["cell_type"], "markdown")
            self.assertIn("### ", "".join(previous["source"]))


if __name__ == "__main__":
    unittest.main()
