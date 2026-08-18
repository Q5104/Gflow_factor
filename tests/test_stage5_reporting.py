import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import nbformat
import torch

from factor_gfn.gfn.hybrid_checkpoint import HYBRID_CHECKPOINT_SCHEMA
from factor_gfn.gfn.hybrid_config import build_stage5_hybrid_variance_5_15_config
from factor_gfn.gfn.hybrid_search_runner import HYBRID_VARIANCE_RUNNER_SCHEMA
from factor_gfn.gfn.train_candidate_artifact import (
    TRAIN_CANDIDATE_ARTIFACT_SCHEMA,
    TRAIN_CANDIDATE_RECORD_SCHEMA,
    TRAIN_EVALUATION_CONTRACT_SCHEMA,
)
from factor_gfn.grammar import get_action_id
from factor_gfn.reporting import Stage5ReportRenderer, load_stage5_report_data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage5ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = build_stage5_hybrid_variance_5_15_config(max_cycles=1)

    def tearDown(self):
        self.temporary.cleanup()

    def _diagnostic(self, step: int, node_count: int) -> dict:
        objective = "exact_tb" if node_count in (1, 2) else "log_partition_variance"
        row = {
            "cycle_index": 0,
            "condition_position_in_cycle": step - 1,
            "condition_N": node_count,
            "objective_kind": objective,
            "global_optimizer_step": step,
            "trajectories_in_batch": 16,
            "total_trajectories_seen": step * 16,
            "requested_count": 16,
            "accepted_count": 16,
            "invalid_count": node_count % 2,
            "retry_count": node_count % 3,
            "retry_exhausted_count": 0,
            "reward_mean": 0.01 * node_count,
            "reward_std": 0.001,
            "sum_log_pf_mean": -1.0,
            "sum_log_pb_mean": -0.5,
            "trajectory_length": float(node_count),
            "terminal_success_rate": 1.0,
            "policy_grad_norm": float(node_count),
        }
        if objective == "exact_tb":
            row.update(
                exact_log_z=1.0,
                tb_loss=0.1 * node_count,
                tb_delta_mean=0.0,
                tb_delta_std=0.1,
                tb_delta_rms=0.1,
            )
        else:
            row.update(
                zeta_mean=0.0,
                zeta_std=0.2,
                zeta_variance=0.04,
                variance_loss=0.2 * node_count,
                centered_zeta_rms=0.2,
                unique_terminal_count=15,
                unique_terminal_fraction=15 / 16,
            )
        return row

    def _record(self, index: int) -> dict:
        dates = [f"2018-03-{day:02d}" for day in range(1, 32)] + [f"2018-04-{day:02d}" for day in range(1, 31)]
        values = np.linspace(-0.03, 0.04, 61) * (index + 1)
        if index == 2:
            dates = dates[2:]
            values = values[2:]
        if index == 3:
            values = np.ones(len(dates))
        node_count = 3
        prefix = [get_action_id("add"), get_action_id("close"), get_action_id("open")]
        if index == 0:
            node_count = 2
            prefix = [get_action_id("ts_mean", 5), get_action_id("volume")]
        absolute_ic = 0.5 if index in (0, 1) else 0.4 - index * 0.005
        train_ic = absolute_ic if index % 2 == 0 else -absolute_ic
        return {
            "schema": TRAIN_CANDIDATE_RECORD_SCHEMA,
            "structural_hash": f"h{index:03d}",
            "formula": f"factor_{index:03d}",
            "prefix_token_ids": prefix,
            "node_count": node_count,
            "depth": 2,
            "train_evaluation_contract_fingerprint": "contract-fp",
            "train_ic": train_ic,
            "train_ic_valid_periods": 80,
            "train_direction": 1 if train_ic >= 0 else -1,
            "train_long_ir": 0.6 - index * 0.01,
            "train_long_valid_periods": 80,
            "train_long_excess_dates": dates,
            "train_long_excess_values": [float(value) for value in values],
            "train_barra_ts_corr": 0.2 + index * 0.01,
            "train_barra_correlations": {},
            "train_barra_valid_periods_by_style": {},
            "neutralization_diagnostics": {
                "industry_neutralized": True,
                "skipped_dates": [],
                "skipped_rate": 0.0,
                "details": [],
            },
            "first_seen": {
                "optimizer_step": index % 15 + 1,
                "cycle_index": index // 16,
                "condition_position_in_cycle": index % 15,
                "condition_N": node_count,
            },
            "last_seen": {
                "optimizer_step": 15,
                "cycle_index": 0,
                "condition_position_in_cycle": 14,
                "condition_N": node_count,
            },
            "visit_count": 100 if index == 0 else 1,
        }

    def _write_fixture(
        self,
        *,
        complete: bool = True,
        step: int = 15,
        pending=None,
        checkpoint_step: int | None = None,
        checkpoint_fingerprint: str | None = None,
    ) -> Path:
        run_dir = self.root / f"run_{len(list(self.root.iterdir()))}"
        run_dir.mkdir()
        fingerprint = self.config.fingerprint()
        run_config = {
            "schema": HYBRID_VARIANCE_RUNNER_SCHEMA,
            "created_at_utc": "2026-08-17T00:00:00+00:00",
            "checkpoint_schema": HYBRID_CHECKPOINT_SCHEMA,
            "objective_mode": "hybrid_variance",
            "config_fingerprint": fingerprint,
            "reward_provider_fingerprint": "reward-fp",
            "train_candidate_artifact": {
                "enabled": True,
                "schema": TRAIN_CANDIDATE_ARTIFACT_SCHEMA,
                "filename": "train_candidate_artifact.json",
            },
        }
        run_config_path = run_dir / "hybrid_run_config.json"
        run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
        state = {
            "schema": HYBRID_VARIANCE_RUNNER_SCHEMA,
            "updated_at_utc": "2026-08-17T01:00:00+00:00",
            "global_optimizer_step": step,
            "total_trajectories_seen": step * 16,
            "pending_assignment": pending,
            "complete": complete,
            "latest_checkpoint": str(run_dir / "checkpoint_latest.pt"),
        }
        (run_dir / "runner_state.json").write_text(json.dumps(state), encoding="utf-8")
        diagnostics = [self._diagnostic(value, value) for value in range(1, step + 1)]
        (run_dir / "hybrid_diagnostics.jsonl").write_text(
            "\n".join(json.dumps(row) for row in diagnostics) + "\n",
            encoding="utf-8",
        )
        records = sorted((self._record(index) for index in range(32)), key=lambda row: row["structural_hash"])
        artifact = {
            "schema": TRAIN_CANDIDATE_ARTIFACT_SCHEMA,
            "source_run": {
                "run_directory_name": run_dir.name,
                "hybrid_run_config_sha256": _sha256(run_config_path),
            },
            "train_evaluation_contract": {
                "schema": TRAIN_EVALUATION_CONTRACT_SCHEMA,
                "provider_fingerprint": "reward-fp",
            },
            "train_evaluation_contract_fingerprint": "contract-fp",
            "committed_optimizer_step": step,
            "candidate_count": len(records),
            "records": records,
        }
        (run_dir / "train_candidate_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
        torch.save(
            {
                "schema": HYBRID_CHECKPOINT_SCHEMA,
                "saved_at_utc": "2026-08-17T01:00:00+00:00",
                "objective_mode": "hybrid_variance",
                "config_fingerprint": checkpoint_fingerprint or fingerprint,
                "reward_provider_fingerprint": "reward-fp",
                "global_optimizer_step": checkpoint_step if checkpoint_step is not None else step,
                "total_trajectories_seen": step * 16,
            },
            run_dir / "checkpoint_latest.pt",
        )
        return run_dir

    def test_snapshot_gate_complete_and_incomplete_modes(self):
        complete = self._write_fixture()
        bundle = load_stage5_report_data(complete, expected_config=self.config)
        self.assertEqual(bundle.snapshot_manifest["snapshot_consistency_status"], "consistent")
        self.assertTrue(bundle.snapshot_manifest["complete"])

        incomplete = self._write_fixture(
            complete=False,
            step=4,
            pending={"cycle_index": 0, "position": 4, "condition_N": 5},
        )
        with self.assertRaisesRegex(ValueError, "complete=true"):
            load_stage5_report_data(incomplete, expected_config=self.config)
        preview = load_stage5_report_data(
            incomplete,
            expected_config=self.config,
            allow_incomplete=True,
        )
        self.assertFalse(preview.snapshot_manifest["complete"])

    def test_snapshot_gate_rejects_step_fingerprint_and_formal_pending(self):
        mismatch = self._write_fixture(checkpoint_step=14)
        with self.assertRaisesRegex(ValueError, "snapshot step mismatch"):
            load_stage5_report_data(mismatch, expected_config=self.config)

        fingerprint = self._write_fixture(checkpoint_fingerprint="wrong")
        with self.assertRaisesRegex(ValueError, "config fingerprint mismatch"):
            load_stage5_report_data(fingerprint, expected_config=self.config)

        pending = self._write_fixture(pending={"condition_N": 1})
        with self.assertRaisesRegex(ValueError, "pending_assignment=null"):
            load_stage5_report_data(pending, expected_config=self.config)

    def test_hybrid_candidate_usage_and_exploration_semantics(self):
        bundle = load_stage5_report_data(self._write_fixture(), expected_config=self.config)
        summary = bundle.training_summary_by_n
        self.assertEqual(set(summary[summary["N"].isin((1, 2))]["metric"]), {"tb_loss", "policy_grad_norm", "reward_mean"})
        self.assertEqual(set(summary[summary["N"] >= 3]["metric"]), {"variance_loss", "policy_grad_norm", "reward_mean"})
        self.assertNotIn("loss", set(summary["metric"]))
        self.assertEqual(len(bundle.candidate_summary), 32)
        self.assertEqual(int(bundle.quality_reference_counts.iloc[0]["candidate_count"]), 32)
        self.assertEqual(int(bundle.exploration_by_n["unique_candidate_count"].sum()), 32)
        self.assertEqual(int(bundle.exploration_by_cycle.iloc[-1]["cumulative_unique_candidates"]), 32)
        self.assertEqual(set(bundle.field_usage["field"]), {"open", "high", "low", "close", "vwap", "volume"})
        self.assertEqual(set(bundle.window_usage["window"]), {5, 10, 20, 40, 60})
        self.assertIn("TS_UNARY_OP", set(bundle.operator_usage["operator_family"]))
        node_summary = bundle.complexity_summary.set_index("metric")
        self.assertEqual(float(node_summary.loc["node_count", "max"]), 3.0)

    def test_correlation_subset_order_alignment_and_invalid_pairs(self):
        bundle = load_stage5_report_data(self._write_fixture(), expected_config=self.config)
        selected = bundle.selected_long_excess_series.drop_duplicates("structural_hash")
        self.assertEqual(len(selected), 30)
        self.assertEqual(list(selected["structural_hash"].head(2)), ["h000", "h001"])
        self.assertNotIn("h031", set(selected["structural_hash"]))
        matrix = bundle.long_excess_correlation_matrix
        self.assertAlmostEqual(float(matrix.loc["h000", "h001"]), 1.0)
        self.assertTrue(np.isnan(matrix.loc["h000", "h002"]))
        summary = bundle.long_excess_correlation_summary.iloc[0]
        self.assertGreater(int(summary["valid_pair_count"]), 0)
        self.assertGreater(int(summary["invalid_pair_count"]), 0)

    def test_renderer_outputs_png_csv_and_manifest(self):
        bundle = load_stage5_report_data(self._write_fixture(), expected_config=self.config)
        output_dir = self.root / "report"
        rendered = Stage5ReportRenderer(bundle, output_dir).render_all()
        self.assertEqual(len(rendered["figures"]), 15)
        self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in rendered["figures"]))
        self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in rendered["tables"]))
        manifest = json.loads(Path(rendered["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_run_identity"], bundle.snapshot_manifest["source_run_id"])
        self.assertEqual(manifest["snapshot_steps"]["runner"], 15)
        self.assertFalse(manifest["incomplete_preview"])
        self.assertEqual(len(manifest["output_inventory"]["figures"]), 15)

    def test_reporting_notebook_contract_and_empty_outputs(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "stage5_reporting.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        headings = [
            "".join(cell.source)
            for cell in notebook.cells
            if cell.cell_type == "markdown" and "## " in "".join(cell.source)
        ]
        expected = [
            "00 Parameters and Snapshot Validation",
            "01 Run Summary",
            "02 Training Diagnostics",
            "03 Search Exploration",
            "04 Candidate Quality",
            "05 Candidate Relationships",
            "06 Expression Structure",
            "07 Candidate Examples",
            "08 Export",
        ]
        self.assertEqual(
            [next(name for name in expected if name in heading) for heading in headings],
            expected,
        )
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        self.assertTrue(all(cell.execution_count is None for cell in code_cells))
        self.assertTrue(all(cell.outputs == [] for cell in code_cells))
        source = "\n".join("".join(cell.source) for cell in code_cells)
        self.assertIn("ALLOW_INCOMPLETE = False", source)
        self.assertIn("REPLACE_WITH_COMPLETED_STAGE5_RUN_DIRECTORY", source)
        self.assertNotIn("runs/stage5_hybrid_variance_real", source.replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
