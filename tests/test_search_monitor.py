import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from factor_gfn.gfn.search_monitor import (
    export_tensorboard_history,
    format_search_status,
    freeze_search_baseline,
    load_search_status,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SearchMonitorTests(unittest.TestCase):
    def _make_run(self, root: Path) -> Path:
        run_dir = root / "run_1"
        run_dir.mkdir()
        _write_json(
            run_dir / "run_state.json",
            {
                "run_id": "run_1",
                "status": "ready",
                "current_step": 20,
                "optimizer_step": 20,
                "active_step": None,
                "evaluation_records": 190,
                "step_metric_records": 20,
                "latest_checkpoint": str(run_dir / "checkpoint_latest.pt"),
                "complete": False,
            },
        )
        _write_json(
            run_dir / "training_stats.json",
            {
                "history": [
                    {
                        "step": 20,
                        "optimizer_step": 20,
                        "loss": 1.5,
                        "reward_mean": 0.02,
                        "batch_rejection_rate": 0.1,
                        "policy_entropy_normalized_mean": 0.98,
                        "leaf_group_probability_mean": 0.4,
                        "unary_group_probability_mean": 0.35,
                        "binary_group_probability_mean": 0.25,
                        "leaf_action_rate": 0.45,
                        "unary_action_rate": 0.30,
                        "binary_action_rate": 0.25,
                        "feature_category_action_rate": 0.20,
                        "unary_category_action_rate": 0.15,
                        "ts_unary_category_action_rate": 0.15,
                        "binary_category_action_rate": 0.20,
                        "ts_binary_category_action_rate": 0.15,
                        "cross_sectional_category_action_rate": 0.15,
                        "window_5_action_rate": 0.10,
                        "window_10_action_rate": 0.20,
                        "window_20_action_rate": 0.30,
                        "window_40_action_rate": 0.25,
                        "window_60_action_rate": 0.15,
                        "terminal_node_count_p50": 7.0,
                        "terminal_node_count_p90": 12.0,
                        "max_node_terminal_rate": 0.125,
                        "illegal_action_rate": 0.0,
                    }
                ],
                "total_wall_seconds": 100.0,
                "unique_expressions": 186,
                "valid_reward_requests": 160,
                "current_step": 20,
            },
        )
        (run_dir / "step_metrics.jsonl").write_text(
            json.dumps(
                {
                    "step": 20,
                    "wall_seconds": 5.0,
                    "factor_seconds": 3.0,
                    "reward_seconds": 1.0,
                    "sampling_seconds": 0.5,
                    "training_update_seconds": 0.25,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for relative in (
            "search_run_config.json",
            "run_metadata.json",
            "evaluations.jsonl",
            "best_candidate.json",
            "experiment_manifest.json",
        ):
            (run_dir / relative).write_text("{}\n", encoding="utf-8")
        (run_dir / "checkpoint_latest.pt").write_bytes(b"latest")
        archive = run_dir / "checkpoints" / "checkpoint_step_00000020.pt"
        archive.parent.mkdir()
        archive.write_bytes(b"archive")
        return run_dir

    def test_status_is_read_only_and_formats_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._make_run(Path(temporary))
            state_path = run_dir / "run_state.json"
            before = hashlib.sha256(state_path.read_bytes()).hexdigest()
            snapshot = load_search_status(run_dir)
            after = hashlib.sha256(state_path.read_bytes()).hexdigest()
            text = format_search_status(snapshot)
        self.assertEqual(before, after)
        self.assertEqual(snapshot["current_step"], 20)
        self.assertEqual(snapshot["optimizer_step"], 20)
        self.assertEqual(snapshot["alerts"], [])
        self.assertIn("current_step=20", text)
        self.assertIn("group_p=0.400/0.350/0.250", text)
        self.assertIn("action_rate=0.450/0.300/0.250", text)
        self.assertIn("grammar_rate=0.200/0.150/0.150/0.200/0.150/0.150", text)
        self.assertIn("window_rate=0.100/0.200/0.300/0.250/0.150", text)
        self.assertIn("nodes_p50/p90=7.0/12.0", text)
        self.assertIn("sample=0.5s", text)
        self.assertIn("update=0.2s", text)

    def test_freeze_is_idempotent_and_refuses_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._make_run(Path(temporary))
            target = freeze_search_baseline(run_dir, expected_step=20)
            self.assertEqual(target, freeze_search_baseline(run_dir, expected_step=20))
            baseline = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(baseline["current_step"], 20)
            (run_dir / "evaluations.jsonl").write_text(
                '{"changed": true}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "拒绝覆盖"):
                freeze_search_baseline(run_dir, expected_step=20)

    def test_installed_tensorboard_round_trips_event_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = SummaryWriter(log_dir=temporary)
            writer.add_scalar("progress/current_step", 20, 20)
            writer.flush()
            writer.close()
            accumulator = EventAccumulator(temporary)
            accumulator.Reload()
            events = accumulator.Scalars("progress/current_step")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].step, 20)
        self.assertEqual(events[0].value, 20.0)

    def test_existing_history_exports_once_to_tensorboard(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._make_run(Path(temporary))
            (run_dir / "evaluations.jsonl").write_text(
                json.dumps(
                    {
                        "logical_step": 20,
                        "valid": True,
                        "reward": 0.02,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tensorboard_dir = export_tensorboard_history(run_dir)
            event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
            self.assertEqual(tensorboard_dir, export_tensorboard_history(run_dir))
            self.assertEqual(
                event_files, list(tensorboard_dir.glob("events.out.tfevents.*"))
            )
            accumulator = EventAccumulator(str(tensorboard_dir))
            accumulator.Reload()
            events = accumulator.Scalars("progress/current_step")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].step, 20)


if __name__ == "__main__":
    unittest.main()
