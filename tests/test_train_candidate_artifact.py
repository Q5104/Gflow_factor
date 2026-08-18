from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from factor_gfn.barra import STYLE_NAMES, BarraPenaltyResult
from factor_gfn.gfn import (
    TRAIN_CANDIDATE_ARTIFACT_FILENAME,
    TRAIN_CANDIDATE_ARTIFACT_SCHEMA,
    TRAIN_CANDIDATE_RECORD_SCHEMA,
    RealRewardProvider,
    RewardConfig,
    HybridVarianceTrainer,
    SingleConditionBatchCollection,
    TrainCandidateArtifactWriter,
    build_stage5_hybrid_variance_5_15_config,
    combine_reward_components,
    create_hybrid_variance_runner,
)
from factor_gfn.grammar import Expression, get_action_id

from tests.test_gfn_real_reward import _context
from tests.test_hybrid_variance_trainer import _trajectory


def _expression(*, reversed_children: bool = False) -> Expression:
    children = (
        (get_action_id("open"), get_action_id("close"))
        if reversed_children
        else (get_action_id("close"), get_action_id("open"))
    )
    return Expression.from_prefix((get_action_id("add"), *children))


def _provider() -> RealRewardProvider:
    return RealRewardProvider(
        _context(),
        RewardConfig(
            barra_min_common_periods=5,
            candidate_industry_neutralization=True,
        ),
    )


def _output(expressions, optimizer_step: int):
    trajectories = tuple(
        SimpleNamespace(terminal_expression=expression) for expression in expressions
    )
    return SimpleNamespace(
        updated=True,
        global_optimizer_step=optimizer_step,
        diagnostics=SimpleNamespace(
            cycle_index=0,
            condition_position_in_cycle=optimizer_step - 1,
        ),
        collection=SimpleNamespace(trajectories=trajectories),
    )


def _trainer():
    return SimpleNamespace(
        config=SimpleNamespace(
            objective=SimpleNamespace(exact_tb_node_counts=(1, 2)),
        ),
        exhaustive_reward_lookups_by_N={},
    )


def _write_run_config(run_dir: Path) -> None:
    (run_dir / "hybrid_run_config.json").write_text(
        json.dumps({"schema": "test-hybrid-run", "seed": 20260816}),
        encoding="utf-8",
    )


def _all_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key).lower()
            yield from _all_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_keys(nested)


class TrainCandidateArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "hybrid_run"
        self.run_dir.mkdir()
        _write_run_config(self.run_dir)

    def tearDown(self):
        self.temporary.cleanup()

    def _writer(
        self,
        provider: RealRewardProvider,
        *,
        step: int,
        create: bool,
    ) -> TrainCandidateArtifactWriter:
        return TrainCandidateArtifactWriter(
            run_dir=self.run_dir,
            provider=provider,
            expected_optimizer_step=step,
            create=create,
        )

    def test_artifact_preserves_frozen_train_fields_without_recalculation(self):
        provider = _provider()
        expression = _expression()
        assignment = provider.evaluate(expression)
        reward_before = assignment.reward
        result_before = dict(assignment.metadata["reward_result"])
        interpreter_calls_before = provider.interpreter_evaluation_count

        writer = self._writer(provider, step=0, create=True)
        writer.commit_update(_output((expression,), 1), _trainer())

        payload = json.loads(
            (self.run_dir / TRAIN_CANDIDATE_ARTIFACT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["schema"], TRAIN_CANDIDATE_ARTIFACT_SCHEMA)
        self.assertEqual(payload["committed_optimizer_step"], 1)
        self.assertEqual(payload["candidate_count"], 1)
        record = payload["records"][0]
        self.assertEqual(record["schema"], TRAIN_CANDIDATE_RECORD_SCHEMA)
        self.assertEqual(record["structural_hash"], expression.structural_hash())
        self.assertEqual(record["train_ic"], result_before["train_ic"])
        self.assertEqual(
            record["train_ic_valid_periods"], result_before["ic_valid_periods"]
        )
        self.assertEqual(record["train_direction"], result_before["long_direction"])
        self.assertEqual(record["train_long_ir"], result_before["train_long_ir"])
        self.assertEqual(
            record["train_long_valid_periods"],
            result_before["long_ir_valid_periods"],
        )
        self.assertEqual(
            record["train_long_excess_dates"],
            list(result_before["train_long_excess_dates"]),
        )
        self.assertEqual(
            record["train_long_excess_values"],
            list(result_before["train_long_excess_values"]),
        )
        self.assertEqual(
            record["train_barra_ts_corr"], result_before["barra_ts_corr"]
        )
        self.assertEqual(
            record["train_barra_correlations"],
            result_before["barra_correlations"],
        )
        self.assertEqual(
            record["train_barra_valid_periods_by_style"],
            result_before["barra_valid_periods"],
        )
        self.assertEqual(
            record["neutralization_diagnostics"],
            {
                "industry_neutralized": result_before["industry_neutralized"],
                "skipped_dates": list(
                    result_before["neutralization_skipped_dates"]
                ),
                "skipped_rate": result_before["neutralization_skipped_rate"],
                "details": list(
                    result_before["neutralization_skipped_details"]
                ),
            },
        )
        self.assertEqual(provider.interpreter_evaluation_count, interpreter_calls_before)
        self.assertEqual(assignment.reward, reward_before)
        self.assertFalse(
            (self.run_dir / (TRAIN_CANDIDATE_ARTIFACT_FILENAME + ".tmp")).exists()
        )
        self.assertFalse(any("validation" in key for key in _all_keys(payload)))
        self.assertNotIn("reward_config", _all_keys(payload))
        self.assertNotIn("reward_floor", _all_keys(payload))
        self.assertNotIn("reward", record)
        self.assertNotIn("valid", record)
        self.assertNotIn("floor_applied", record)

    def test_canonical_duplicate_is_one_record_with_visit_provenance(self):
        provider = _provider()
        first = _expression()
        second = _expression(reversed_children=True)
        self.assertEqual(first.structural_hash(), second.structural_hash())
        provider.evaluate(first)

        writer = self._writer(provider, step=0, create=True)
        writer.commit_update(_output((first, second), 1), _trainer())

        self.assertEqual(len(writer.records), 1)
        record = writer.records[0]
        self.assertEqual(record["visit_count"], 2)
        self.assertEqual(record["first_seen"]["optimizer_step"], 1)
        self.assertEqual(record["last_seen"]["optimizer_step"], 1)
        self.assertEqual(record["formula"], first.canonicalize().to_formula())

    def test_resume_is_idempotent_and_step_divergence_fails_closed(self):
        provider = _provider()
        expression = _expression()
        provider.evaluate(expression)
        writer = self._writer(provider, step=0, create=True)
        writer.commit_update(_output((expression,), 1), _trainer())

        with self.assertRaisesRegex(ValueError, "diverges from checkpoint"):
            self._writer(provider, step=2, create=False)

        resumed = self._writer(provider, step=1, create=False)
        resumed.commit_update(_output((expression,), 2), _trainer())
        self.assertEqual(len(resumed.records), 1)
        self.assertEqual(resumed.records[0]["visit_count"], 2)
        self.assertEqual(resumed.records[0]["last_seen"]["optimizer_step"], 2)
        reloaded = self._writer(provider, step=2, create=False)
        self.assertEqual(reloaded.records, resumed.records)

    def test_hybrid_runner_persists_artifact_without_changing_trainer_objective(self):
        context = replace(
            _context(),
            rebalance_indices=np.arange(10, 75, dtype=np.int64),
        )
        provider = RealRewardProvider(context)
        config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=1,
            trajectories_per_batch=2,
        )
        trainer = HybridVarianceTrainer(config, provider, device="cpu")
        expression = _expression()
        assignment = provider.evaluate(expression)
        state = trainer.complexity_scheduler.state_dict()
        state["current_permutation"] = (3,) + tuple(
            value for value in config.resolved_condition_node_counts if value != 3
        )
        state["position"] = 0
        trainer.complexity_scheduler.load_state_dict(state)
        trajectories = tuple(
            _trajectory(
                trainer,
                index=index,
                condition_N=3,
                reward=float(assignment.reward),
            )
            for index in (1, 3)
        )
        collection = SingleConditionBatchCollection(
            condition_N=3,
            requested_count=2,
            trajectories=trajectories,
            sampled_count=2,
            invalid_count=0,
            retry_count=0,
            retry_exhausted_count=0,
            sampling_rounds=1,
            sampling_seconds=0.0,
            reward_provider_seconds=0.0,
        )
        run_dir = Path(self.temporary.name) / "runner_integration"

        with patch(
            "factor_gfn.gfn.hybrid_search_runner.save_hybrid_checkpoint"
        ), patch.object(
            trainer,
            "collect_single_condition_batch",
            return_value=collection,
        ):
            runner = create_hybrid_variance_runner(trainer, run_dir)
            output = runner.run_attempts(1)[0]

        self.assertTrue(output.updated)
        self.assertEqual(output.objective.objective_kind, "log_partition_variance")
        self.assertEqual(trainer.optimizer_step, 1)
        artifact = json.loads(
            runner.train_candidate_artifact_path.read_text(encoding="utf-8")
        )
        self.assertEqual(artifact["committed_optimizer_step"], 1)
        self.assertEqual(artifact["candidate_count"], 1)
        self.assertEqual(artifact["records"][0]["visit_count"], 2)
        self.assertEqual(provider.interpreter_evaluation_count, 1)

    def test_long_excess_payload_does_not_change_reward_formula(self):
        penalty = BarraPenaltyResult(
            barra_ts_corr=0.2,
            correlations={name: 0.1 for name in STYLE_NAMES},
            valid_periods={name: 80 for name in STYLE_NAMES},
        )
        baseline = combine_reward_components(
            "a" * 64,
            0.03,
            0.8,
            penalty,
            long_direction=1,
            ic_valid_periods=80,
            long_ir_valid_periods=80,
        )
        with_series = combine_reward_components(
            "a" * 64,
            0.03,
            0.8,
            penalty,
            long_direction=1,
            ic_valid_periods=80,
            long_ir_valid_periods=80,
            train_long_excess_dates=np.array(
                ["2018-01-05", "2018-01-12"], dtype="datetime64[D]"
            ),
            train_long_excess_values=np.array([0.01, np.nan]),
        )
        baseline_payload = asdict(baseline)
        series_payload = asdict(with_series)
        baseline_payload.pop("train_long_excess_dates")
        baseline_payload.pop("train_long_excess_values")
        series_payload.pop("train_long_excess_dates")
        series_payload.pop("train_long_excess_values")
        self.assertEqual(series_payload, baseline_payload)
        self.assertEqual(
            with_series.train_long_excess_dates,
            ("2018-01-05", "2018-01-12"),
        )
        self.assertEqual(with_series.train_long_excess_values, (0.01, None))


if __name__ == "__main__":
    unittest.main()
