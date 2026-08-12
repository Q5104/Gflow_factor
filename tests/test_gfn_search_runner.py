import json
import io
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from factor_gfn.evaluator import IndustryNeutralizationWarning
from factor_gfn.gfn import (
    RealSearchRunner,
    RealSearchSettings,
    RecordingRewardProvider,
    SyntheticRewardProvider,
    TrainingStats,
    build_stage5_real_training_config,
    require_cuda_device,
)
from factor_gfn.gfn.search_runner import (
    SearchStepMetrics,
    _archive_orphans,
    _validate_or_upgrade_cuda_environment,
)
from factor_gfn.grammar import Expression


class _FakeCudaTrainer:
    def __init__(self, config, provider):
        self.config = config
        self.reward_provider = provider
        self.device = torch.device("cuda:0")
        self.step = 0
        self.optimizer_step = 0
        self.history = []
        self.run_id = "fake_run"
        self.last_step_timings = {
            "sampling_seconds": 0.25,
            "reward_provider_seconds": 0.50,
            "training_update_seconds": 0.125,
            "tb_loss_forward_cuda_seconds": 0.01,
            "backward_cuda_seconds": 0.02,
            "optimizer_cuda_seconds": 0.03,
        }

    def train_step(self):
        assignment = self.reward_provider.evaluate(Expression.from_prefix([3]))
        self.step += 1
        self.optimizer_step += 1
        stats = TrainingStats(
            step=self.step,
            optimizer_step=self.optimizer_step,
            loss=1.0,
            log_z=0.1,
            reward_mean=assignment.reward,
            reward_median=assignment.reward,
            effective_batch_size=1,
        )
        self.history.append(stats)
        return stats

    def save_checkpoint(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake checkpoint")


class _FailingCudaTrainer(_FakeCudaTrainer):
    def train_step(self):
        raise RuntimeError("synthetic failure")


class _WarningCudaTrainer(_FakeCudaTrainer):
    def train_step(self):
        warnings.warn("expected exclusion", IndustryNeutralizationWarning)
        return super().train_step()


class _FakeSummaryWriter:
    def __init__(self):
        self.scalars = []
        self.histograms = []
        self.flushed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, value, step):
        self.histograms.append((tag, value, step))

    def flush(self):
        self.flushed = True


class RealSearchSettingsTests(unittest.TestCase):
    def test_only_pristine_legacy_run_can_add_cublas_contract(self):
        current = {
            "device": "cuda:0",
            "cublas_workspace_config": ":4096:8",
        }
        upgraded, changed = _validate_or_upgrade_cuda_environment(
            {"device": "cuda:0"},
            current,
            {
                "current_step": 0,
                "optimizer_step": 0,
                "evaluation_records": 0,
                "step_metric_records": 0,
            },
        )
        self.assertTrue(changed)
        self.assertEqual(upgraded, current)
        with self.assertRaisesRegex(ValueError, "尚未开始训练"):
            _validate_or_upgrade_cuda_environment(
                {"device": "cuda:0"},
                current,
                {
                    "current_step": 1,
                    "optimizer_step": 1,
                    "evaluation_records": 8,
                    "step_metric_records": 1,
                },
            )

    def test_settings_are_cuda_only_and_round_trip(self):
        self.assertEqual(RealSearchSettings(max_steps=1).subexpression_cache_max_bytes, 0)
        with tempfile.TemporaryDirectory() as temporary:
            settings = RealSearchSettings(
                max_steps=100,
                seed=7,
                checkpoint_interval=5,
                device="cuda:1",
                subexpression_cache_max_bytes=123_456,
                run_root=Path(temporary),
            )
            restored = RealSearchSettings.from_manifest(settings.manifest())
        self.assertEqual(restored, settings)
        with self.assertRaisesRegex(ValueError, "只允许"):
            RealSearchSettings(max_steps=1, device="cpu")

    def test_cuda_unavailable_fails_without_fallback(self):
        with patch("factor_gfn.gfn.search_runner.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "禁止静默回退"):
                require_cuda_device("cuda:0")


class RecordingRewardProviderTests(unittest.TestCase):
    def test_every_request_is_recorded_with_step_and_structure(self):
        provider = RecordingRewardProvider(SyntheticRewardProvider())
        provider.begin_step(3)
        assignment = provider.evaluate(Expression.from_prefix([3]))
        records = provider.finish_step()
        self.assertTrue(assignment.valid)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["request_index"], 1)
        self.assertEqual(records[0]["logical_step"], 3)
        self.assertEqual(records[0]["phase"], "train_step_3")
        self.assertEqual(records[0]["formula"], "close")
        self.assertEqual(len(records[0]["structural_hash"]), 64)


class RealSearchPersistenceTests(unittest.TestCase):
    @patch("factor_gfn.gfn.search_runner.torch.cuda.max_memory_reserved", return_value=0)
    @patch("factor_gfn.gfn.search_runner.torch.cuda.max_memory_allocated", return_value=0)
    @patch("factor_gfn.gfn.search_runner.torch.cuda.memory_reserved", return_value=0)
    @patch("factor_gfn.gfn.search_runner.torch.cuda.memory_allocated", return_value=0)
    @patch("factor_gfn.gfn.search_runner.torch.cuda.reset_peak_memory_stats")
    @patch("factor_gfn.gfn.search_runner.torch.cuda.synchronize")
    def test_expected_neutralization_warnings_are_hidden_from_training_console(
        self,
        _synchronize,
        _reset_peak,
        _allocated,
        _reserved,
        _max_allocated,
        _max_reserved,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = RealSearchSettings(
                max_steps=1,
                tensorboard_enabled=False,
                console_progress=False,
                run_root=root,
            )
            config = build_stage5_real_training_config(max_steps=1)
            provider = RecordingRewardProvider(SyntheticRewardProvider())
            trainer = _WarningCudaTrainer(config, provider)
            run_dir = root / trainer.run_id
            run_dir.mkdir()
            runner = RealSearchRunner(
                settings=settings,
                config=config,
                trainer=trainer,
                recording_provider=provider,
                run_dir=run_dir,
                cuda_environment={"device": "cuda:0"},
            )

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                runner.run_until(1)

        self.assertFalse(
            any(
                issubclass(item.category, IndustryNeutralizationWarning)
                for item in captured
            )
        )

    @patch("factor_gfn.gfn.search_runner.torch.cuda.max_memory_reserved", return_value=400)
    @patch("factor_gfn.gfn.search_runner.torch.cuda.max_memory_allocated", return_value=300)
    @patch("factor_gfn.gfn.search_runner.torch.cuda.memory_reserved", return_value=200)
    @patch("factor_gfn.gfn.search_runner.torch.cuda.memory_allocated", return_value=100)
    @patch("factor_gfn.gfn.search_runner.torch.cuda.reset_peak_memory_stats")
    @patch("factor_gfn.gfn.search_runner.torch.cuda.synchronize")
    def test_steps_append_candidates_metrics_and_checkpoints(
        self,
        _synchronize,
        _reset_peak,
        _allocated,
        _reserved,
        _max_allocated,
        _max_reserved,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = RealSearchSettings(
                max_steps=2,
                checkpoint_interval=2,
                tensorboard_enabled=False,
                console_progress=False,
                run_root=root,
            )
            config = build_stage5_real_training_config(max_steps=2)
            provider = RecordingRewardProvider(SyntheticRewardProvider())
            trainer = _FakeCudaTrainer(config, provider)
            run_dir = root / trainer.run_id
            run_dir.mkdir()
            runner = RealSearchRunner(
                settings=settings,
                config=config,
                trainer=trainer,
                recording_provider=provider,
                run_dir=run_dir,
                cuda_environment={"device": "cuda:0"},
            )

            stats = runner.run_until(2)

            self.assertEqual(len(stats), 2)
            evaluations = [
                json.loads(line)
                for line in runner.evaluations_path.read_text(encoding="utf-8").splitlines()
            ]
            metrics = [
                json.loads(line)
                for line in runner.metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([item["logical_step"] for item in evaluations], [1, 2])
            self.assertEqual([item["step"] for item in metrics], [1, 2])
            self.assertEqual(metrics[-1]["sampling_seconds"], 0.25)
            self.assertEqual(metrics[-1]["reward_provider_seconds"], 0.50)
            self.assertEqual(metrics[-1]["training_update_seconds"], 0.125)
            self.assertEqual(metrics[-1]["backward_cuda_seconds"], 0.02)
            self.assertTrue(runner.latest_checkpoint_path.is_file())
            self.assertTrue(runner._archive_checkpoint_path(2).is_file())
            state = json.loads(
                (run_dir / "run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "completed")
            self.assertIsNone(state["active_step"])
            self.assertEqual(state["current_step"], 2)
            self.assertEqual(state["optimizer_step"], 2)
            manifest = json.loads(
                (run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn(str(run_dir / "experiment_manifest.json"), manifest["artifacts"])
            self.assertNotIn(str(runner.tensorboard_dir), manifest["artifacts"])

    @patch("factor_gfn.gfn.search_runner.torch.cuda.reset_peak_memory_stats")
    @patch("factor_gfn.gfn.search_runner.torch.cuda.synchronize")
    def test_failed_step_is_visible_to_independent_monitor(
        self, _synchronize, _reset_peak
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = RealSearchSettings(
                max_steps=1,
                tensorboard_enabled=False,
                console_progress=False,
                run_root=root,
            )
            config = build_stage5_real_training_config(max_steps=1)
            provider = RecordingRewardProvider(SyntheticRewardProvider())
            trainer = _FailingCudaTrainer(config, provider)
            run_dir = root / trainer.run_id
            run_dir.mkdir()
            runner = RealSearchRunner(
                settings=settings,
                config=config,
                trainer=trainer,
                recording_provider=provider,
                run_dir=run_dir,
                cuda_environment={"device": "cuda:0"},
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                runner.run_until(1)
            state = json.loads(
                (run_dir / "run_state.json").read_text(encoding="utf-8")
            )
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["active_step"], 1)
        self.assertIn("synthetic failure", state["last_error"])

    def test_tensorboard_and_console_include_progress_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = RealSearchSettings(
                max_steps=1,
                tensorboard_enabled=False,
                run_root=root,
            )
            config = build_stage5_real_training_config(max_steps=1)
            provider = RecordingRewardProvider(SyntheticRewardProvider())
            trainer = _FakeCudaTrainer(config, provider)
            run_dir = root / trainer.run_id
            run_dir.mkdir()
            runner = RealSearchRunner(
                settings=settings,
                config=config,
                trainer=trainer,
                recording_provider=provider,
                run_dir=run_dir,
                cuda_environment={"device": "cuda:0"},
            )
            stats = TrainingStats(
                step=1,
                optimizer_step=1,
                loss=1.0,
                reward_mean=0.2,
                reward_median=0.2,
                log_z=0.1,
                tb_delta_mean=-1.0,
                tb_delta_std=0.5,
                tb_delta_rms=1.11803398875,
                tb_delta_mean_square_ratio=0.8,
                tb_delta_std_square_ratio=0.2,
                mean_log_pf=-10.0,
                mean_log_pb=-2.0,
                model_gradient_norm_before_clip=9.0,
                log_z_gradient_before_clip=-2.0,
                model_gradient_clip_coefficient=0.1,
                log_z_gradient_clip_coefficient=0.5,
                model_parameter_update_norm=0.02,
                model_relative_update_norm=0.0001,
                log_z_update=0.001,
                batch_rejection_rate=0.25,
                expression_unique_rate=1.0,
                policy_entropy_normalized_mean=0.9,
                terminal_node_count_p50=7.0,
                terminal_node_count_p90=12.0,
                max_node_terminal_rate=0.125,
                group_entropy_mean=0.8,
                group_entropy_normalized_mean=0.75,
                leaf_group_probability_mean=0.4,
                unary_group_probability_mean=0.35,
                binary_group_probability_mean=0.25,
                leaf_action_rate=0.45,
                unary_action_rate=0.30,
                binary_action_rate=0.25,
                grammar_category_entropy_mean=1.2,
                grammar_category_entropy_normalized_mean=0.8,
                operator_entropy_mean=2.0,
                operator_entropy_normalized_mean=0.7,
                window_entropy_mean=1.5,
                window_entropy_normalized_mean=0.9,
                feature_category_action_rate=0.20,
                unary_category_action_rate=0.15,
                ts_unary_category_action_rate=0.15,
                binary_category_action_rate=0.20,
                ts_binary_category_action_rate=0.15,
                cross_sectional_category_action_rate=0.15,
                window_5_action_rate=0.10,
                window_10_action_rate=0.20,
                window_20_action_rate=0.30,
                window_40_action_rate=0.25,
                window_60_action_rate=0.15,
                temporal_operator_action_rate=0.30,
            )
            metrics = SearchStepMetrics(
                step=1,
                optimizer_step=1,
                wall_seconds=4.0,
                reward_requests=2,
                valid_reward_requests=1,
                unique_expressions=2,
                factor_seconds=2.0,
                reward_seconds=1.0,
                provider_cache_hits=0,
                subexpression_cache_hits=3,
                subexpression_cache_misses=1,
                subexpression_cache_evictions=0,
                gpu_memory_allocated_bytes=1024,
                gpu_memory_reserved_bytes=2048,
                gpu_peak_memory_allocated_bytes=4096,
                gpu_peak_memory_reserved_bytes=8192,
            )
            writer = _FakeSummaryWriter()
            runner._write_tensorboard_step(
                writer,
                stats=stats,
                metrics=metrics,
                records=[{"valid": True, "reward": 0.2}],
            )
            output = io.StringIO()
            with redirect_stdout(output):
                runner._print_step_progress(
                    target_step=1, stats=stats, metrics=metrics
                )
        tags = {item[0] for item in writer.scalars}
        self.assertIn("progress/current_step", tags)
        self.assertIn("progress/optimizer_step", tags)
        self.assertIn("performance/subexpression_cache_hit_rate", tags)
        self.assertIn("diagnostics/tb_delta_mean", tags)
        self.assertIn("diagnostics/model_gradient_clip_coefficient", tags)
        self.assertIn("diagnostics/log_z_gradient_clip_coefficient", tags)
        self.assertIn("diagnostics/model_relative_update_norm", tags)
        self.assertIn("diagnostics/log_z_update", tags)
        self.assertIn("policy/group_entropy", tags)
        self.assertIn("policy/leaf_group_probability", tags)
        self.assertIn("policy/leaf_action_rate", tags)
        self.assertIn("policy/grammar_category_entropy", tags)
        self.assertIn("policy/grammar_action_rate/ts_unary", tags)
        self.assertIn("policy/window_action_rate/20", tags)
        self.assertIn("train/terminal_node_count_p90", tags)
        self.assertIn("train/max_node_terminal_rate", tags)
        self.assertEqual(len(writer.histograms), 1)
        self.assertTrue(writer.flushed)
        self.assertIn("current_step=1/1", output.getvalue())
        self.assertIn("optimizer_step=1", output.getvalue())
        self.assertIn("subexpr_hits=3/4", output.getvalue())
        self.assertIn("delta_mean=-1.000000", output.getvalue())
        self.assertIn("model_clip=0.100000", output.getvalue())
        self.assertIn("log_z_clip=0.500000", output.getvalue())
        self.assertIn("model_update=0.00010000", output.getvalue())
        self.assertIn("group_p=0.400/0.350/0.250", output.getvalue())
        self.assertIn("action_rate=0.450/0.300/0.250", output.getvalue())
        self.assertIn("grammar_rate=0.200/0.150/0.150/0.200/0.150/0.150", output.getvalue())
        self.assertIn("window_rate=0.100/0.200/0.300/0.250/0.150", output.getvalue())
        self.assertIn("nodes_p50/p90=7.0/12.0", output.getvalue())
        self.assertIn("max_node_rate=0.125", output.getvalue())

    def test_resume_archives_records_ahead_of_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            evaluations = [
                {"logical_step": 1, "request_index": 1},
                {"logical_step": 2, "request_index": 2},
            ]
            metrics = [{"step": 1}, {"step": 2}]
            kept_evaluations, kept_metrics = _archive_orphans(
                run_dir,
                checkpoint_step=1,
                evaluations=evaluations,
                metrics=metrics,
            )
            self.assertEqual(kept_evaluations, evaluations[:1])
            self.assertEqual(kept_metrics, metrics[:1])
            archives = list((run_dir / "recovery_archives").glob("*.json"))
            self.assertEqual(len(archives), 1)
            archived = json.loads(archives[0].read_text(encoding="utf-8"))
            self.assertEqual(archived["evaluations"], evaluations[1:])
            self.assertEqual(archived["metrics"], metrics[1:])


if __name__ == "__main__":
    unittest.main()
