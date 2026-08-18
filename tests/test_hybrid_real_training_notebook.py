import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "run_stage5_hybrid_variance_real_5_15.ipynb"
)


class HybridRealTrainingNotebookTests(unittest.TestCase):
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

    def test_frozen_config_and_manual_training_gate_are_explicit(self):
        config = self.code_by_id["frozen-config"]
        self.assertIn("build_stage5_hybrid_variance_5_15_config", config)
        self.assertIn("FORMAL_MAX_CYCLES = 100", config)
        self.assertIn("assert CONTRACT['max_cycles'] == 100", config)
        self.assertIn("K = 16", config)
        self.assertIn("assert CONTRACT['max_depth'] == 5", config)
        self.assertIn("assert CONTRACT['optimizer_steps_per_cycle'] == 15", config)
        self.assertIn("assert CONTRACT['planned_total_optimizer_steps'] == 1500", config)
        self.assertIn("assert CONTRACT['planned_total_trajectories'] == 24000", config)
        runner_init = self.code_by_id["runner-init"]
        self.assertIn("RUN_REAL_ONE_CYCLE = False", runner_init)
        self.assertNotIn("RUN_REAL_ONE_CYCLE = True", self.all_source)
        self.assertIn(
            "Initial one-cycle cell requires a fresh step-0 runner",
            self.code_by_id["one-cycle"],
        )
        continuation = self.code_by_id["continuation-definition"]
        self.assertIn("TARGET_CYCLE = 100", continuation)
        self.assertIn("RUN_TO_TARGET_CYCLE = False", continuation)
        self.assertNotIn("RUN_TO_TARGET_CYCLE = True", self.all_source)

    def test_continuation_uses_a_cumulative_cycle_target(self):
        continuation = self.code_by_id["continuation-definition"]
        for marker in (
            "def run_until_cycle(active_runner, target_cycle)",
            "target_cycle > training.max_cycles",
            "target_optimizer_step < start_optimizer_step",
            "target_optimizer_step = target_cycle * steps_per_cycle",
            "active_runner.trainer.optimizer_step < target_optimizer_step",
            "artifact committed step differs from the runner optimizer step",
            "runner = active_runner",
        ):
            self.assertIn(marker, continuation)
        self.assertNotIn("ADDITIONAL_CYCLES", continuation)
        self.assertNotIn("run_additional_cycles", continuation)

    def test_run_until_cycle_advances_to_total_not_by_an_additional_amount(self):
        continuation = self.code_by_id["continuation-definition"]
        tree = ast.parse(continuation, filename="continuation-definition")
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_until_cycle"
        )
        namespace = {
            "json": json,
            "perf_counter": lambda: 0.0,
            "print": lambda *args, **kwargs: None,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                filename="continuation-definition",
                mode="exec",
            ),
            namespace,
        )
        run_until_cycle = namespace["run_until_cycle"]

        class FakeArtifactPath:
            def __init__(self, owner):
                self.owner = owner

            def read_text(self, encoding):
                self.owner.artifact_reads += 1
                return json.dumps(
                    {
                        "committed_optimizer_step": self.owner.trainer.optimizer_step,
                        "candidate_count": 0,
                    }
                )

        class FakeRunner:
            def __init__(self, start_step=15):
                training = SimpleNamespace(
                    max_cycles=100,
                    optimizer_steps_per_cycle=15,
                    trajectories_per_cycle=240,
                )
                self.trainer = SimpleNamespace(
                    config=SimpleNamespace(training=training),
                    optimizer_step=start_step,
                    total_trajectories_seen=start_step * 16,
                )
                self.run_dir = Path("fake-run")
                self.artifact_reads = 0
                self.train_candidate_artifact_path = FakeArtifactPath(self)

            @property
            def complete(self):
                return self.trainer.optimizer_step == 1500

            def run_attempts(self, attempts):
                self.trainer.optimizer_step += 1
                self.trainer.total_trajectories_seen += 16
                diagnostics = SimpleNamespace(
                    cycle_index=(self.trainer.optimizer_step - 1) // 15,
                    condition_N=1,
                    objective_kind="exact_tb",
                    policy_grad_norm=0.0,
                )
                return [
                    SimpleNamespace(
                        updated=True,
                        diagnostics=diagnostics,
                        global_optimizer_step=self.trainer.optimizer_step,
                    )
                ]

        runner = FakeRunner()
        summary = run_until_cycle(runner, target_cycle=6)
        self.assertEqual(summary["start_cycle"], 1)
        self.assertEqual(summary["start_position_in_cycle"], 0)
        self.assertEqual(summary["equivalent_cycles_completed_this_call"], 5)
        self.assertEqual(runner.trainer.optimizer_step, 90)
        self.assertEqual(runner.trainer.total_trajectories_seen, 1440)
        summary = run_until_cycle(runner, target_cycle=100)
        self.assertEqual(summary["start_cycle"], 6)
        self.assertEqual(summary["start_position_in_cycle"], 0)
        self.assertEqual(summary["equivalent_cycles_completed_this_call"], 94)
        self.assertEqual(runner.trainer.optimizer_step, 1500)
        self.assertEqual(runner.trainer.total_trajectories_seen, 24000)
        self.assertTrue(runner.complete)
        with self.assertRaises(ValueError):
            run_until_cycle(runner, target_cycle=101)
        with self.assertRaises(ValueError):
            run_until_cycle(runner, target_cycle=99)

        interrupted_runner = FakeRunner(start_step=283)
        summary = run_until_cycle(interrupted_runner, target_cycle=20)
        self.assertEqual(summary["start_cycle"], 18)
        self.assertEqual(summary["start_position_in_cycle"], 13)
        self.assertEqual(summary["successful_optimizer_updates"], 17)
        self.assertEqual(interrupted_runner.trainer.optimizer_step, 300)
        self.assertEqual(interrupted_runner.trainer.total_trajectories_seen, 4800)

    def test_all_instruction_cells_are_written_in_chinese(self):
        markdown_cells = [
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        self.assertTrue(markdown_cells)
        for source in markdown_cells:
            self.assertRegex(source, r"[\u4e00-\u9fff]", source)

    def test_preflight_uses_real_production_components_without_training(self):
        preflight = self.code_by_id["real-preflight"]
        for marker in (
            "build_real_reward_data_context",
            "RealRewardProvider",
            "HybridVarianceTrainer",
            "ExhaustiveRegistry(SOURCE_REGISTRY, read_only=True)",
            "configure_hybrid_exhaustive_registry",
            "create_hybrid_variance_runner",
            "optimizer_step_after_preflight",
        ):
            self.assertIn(marker, preflight)
        self.assertNotIn("train_step(", preflight)
        self.assertNotIn("run_attempts(", preflight)

    def test_one_cycle_diagnostics_artifact_and_resume_are_present(self):
        cycle = self.code_by_id["one-cycle"]
        self.assertIn("runner.run_attempts(1)", cycle)
        self.assertIn("optimizer_steps_per_cycle", cycle)
        diagnostics = self.code_by_id["diagnostics"]
        for marker in (
            "exact_log_z",
            "tb_delta_rms",
            "zeta_variance",
            "variance_loss",
            "unique_terminal_fraction",
            "policy_grad_norm",
        ):
            self.assertIn(marker, diagnostics)
        self.assertIn("train_long_excess_values", self.code_by_id["artifact-check"])
        self.assertIn("resume_hybrid_variance_runner", self.code_by_id["resume-check"])

    def test_stage6_validation_and_phase_b_continuation_are_absent(self):
        lowered = self.all_source.lower()
        self.assertNotIn("factor_gfn.backtest", lowered)
        self.assertNotIn("stage6_", lowered)
        self.assertNotIn("validation_metrics", lowered)
        self.assertNotIn("candidate_freeze", lowered)


if __name__ == "__main__":
    unittest.main()
