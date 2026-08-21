import ast
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "run_stage5_daily_derived_v1_hybrid_variance_real_5_15.ipynb"
)


class DailyDerivedStage5TrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_cells = [
            cell for cell in cls.notebook["cells"] if cell["cell_type"] == "code"
        ]
        cls.code_by_id = {
            cell["id"]: "".join(cell["source"]) for cell in cls.code_cells
        }
        cls.all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_notebook_is_clean_and_all_code_parses(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for cell in self.code_cells:
            self.assertIsNone(cell["execution_count"], cell["id"])
            self.assertEqual(cell["outputs"], [], cell["id"])
            ast.parse("".join(cell["source"]), filename=cell["id"])

    def test_first_code_cell_bootstraps_project_imports(self):
        first = self.code_cells[0]
        self.assertEqual(first["id"], "environment-check")
        source = "".join(first["source"])
        self.assertIn("(path / 'factor_gfn').is_dir()", source)
        self.assertIn("sys.path.insert(0, str(PROJECT_ROOT))", source)

    def test_derived_config_freezes_raw_baseline_training_parameters(self):
        source = self.code_by_id["frozen-config"]
        markers = (
            "feature_space=DAILY_DERIVED_V1_FEATURE_SPACE",
            "FORMAL_MAX_CYCLES = 100",
            "K = 16",
            "assert CONTRACT['max_cycles'] == 100",
            "assert CONTRACT['max_depth'] == 5",
            "assert CONTRACT['max_nodes'] == 15",
            "assert CONTRACT['K'] == 16",
            "assert CONTRACT['conditions'] == tuple(range(1, 16))",
            "assert CONTRACT['exact_conditions'] == (1, 2)",
            "assert CONTRACT['lpv_conditions'] == tuple(range(3, 16))",
            "assert CONTRACT['policy_lr'] == 1e-4",
            "assert CONTRACT['gradient_clip'] == 5.0",
            "assert CONTRACT['seed'] == 42",
            "assert CONTRACT['planned_total_optimizer_steps'] == 1500",
            "assert CONTRACT['planned_total_trajectories'] == 24000",
        )
        for marker in markers:
            self.assertIn(marker, source)

    def test_preflight_is_derived_and_reasserts_frozen_contract(self):
        source = self.code_by_id["real-preflight"]
        markers = (
            "ExpressionFeatureSpec.daily_derived()",
            "'daily_derived_v1' / 'exact_tb_n1_n2'",
            "'stage5_hybrid_variance_real_5_15'",
            "get_action_id('ret_gap', registry=config.action_registry)",
            "action_registry=config.action_registry",
            "assert config.search_space.max_depth == 5",
            "assert config.search_space.max_nodes == 15",
            "assert config.resolved_condition_node_counts == tuple(range(1, 16))",
            "assert config.training.trajectories_per_batch == 16",
            "assert config.training.max_cycles == 100",
            "assert config.training.learning_rate == 1e-4",
            "assert config.training.model_gradient_clip_norm == 5.0",
            "assert config.objective.exact_tb_node_counts == (1, 2)",
            "assert config.objective.lpv_node_counts == tuple(range(3, 16))",
            "assert config.training.seed == 42",
            "context.manifest['label_formula'] == 'open[t+6] / open[t+1] - 1'",
            "ExhaustiveRegistry(SOURCE_REGISTRY, read_only=True, action_registry=config.action_registry)",
            "configure_hybrid_exhaustive_registry",
            "create_hybrid_variance_runner",
            "optimizer_step_after_preflight",
        )
        for marker in markers:
            self.assertIn(marker, source)
        self.assertNotIn("train_step(", source)
        self.assertNotIn("run_attempts(", source)

    def test_manual_gates_one_cycle_artifact_and_resume_are_present(self):
        runner_init = self.code_by_id["runner-init"]
        self.assertIn("RUN_REAL_ONE_CYCLE = False", runner_init)
        self.assertIn("MODE = 'new'", runner_init)
        self.assertIn("RESUME_RUN_DIR = None", runner_init)
        self.assertNotIn("RUN_REAL_ONE_CYCLE = True", self.all_source)
        one_cycle = self.code_by_id["one-cycle"]
        self.assertIn("runner.run_attempts(1)", one_cycle)
        self.assertIn("list(range(1, 16))", one_cycle)
        artifact = self.code_by_id["artifact-check"]
        self.assertIn("artifact['vocabulary']['feature_space_id']", artifact)
        self.assertIn("config.action_registry.fingerprint()", artifact)
        resume = self.code_by_id["resume-check"]
        self.assertIn("resume_hybrid_variance_runner", resume)
        self.assertIn("action_registry=config.action_registry", resume)

    def test_continuation_is_cumulative_and_disabled(self):
        source = self.code_by_id["continuation-definition"]
        markers = (
            "TARGET_CYCLE = 100",
            "RUN_TO_TARGET_CYCLE = False",
            "target_optimizer_step = target_cycle * steps_per_cycle",
            "active_runner.trainer.optimizer_step < target_optimizer_step",
            "runner = active_runner",
        )
        for marker in markers:
            self.assertIn(marker, source)
        self.assertNotIn("RUN_TO_TARGET_CYCLE = True", self.all_source)

    def test_raw_outputs_and_stage6_are_not_referenced(self):
        lowered = self.all_source.lower()
        self.assertNotIn("complexity_diagnostic_6_20", lowered)
        self.assertNotIn("stage5_hybrid_variance_real_5_15/hybrid_", lowered)
        self.assertNotIn("factor_gfn.backtest", lowered)
        self.assertNotIn("stage6_", lowered)
        self.assertNotIn("validation_metrics", lowered)


if __name__ == "__main__":
    unittest.main()
