import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from factor_gfn.gfn import (
    HYBRID_CHECKPOINT_SCHEMA,
    HYBRID_OBJECTIVE_MODE,
    HYBRID_VARIANCE_RUNNER_SCHEMA,
    ExhaustivePlanningConfig,
    ExhaustiveRegistry,
    HybridVarianceTrainer,
    SingleConditionBatchCollection,
    SyntheticRewardProvider,
    build_stage5_hybrid_variance_5_15_config,
    create_hybrid_variance_runner,
    load_checkpoint,
    load_hybrid_checkpoint,
    resolve_exhaustive_plan,
    resume_hybrid_variance_runner,
    save_hybrid_checkpoint,
)
from factor_gfn.grammar import Expression, SearchSpaceConfig

from tests.test_hybrid_variance_trainer import (
    _CountingSyntheticRewardProvider,
    _trajectory,
)


def _assert_nested_equal(
    case: unittest.TestCase,
    left,
    right,
) -> None:
    if isinstance(left, torch.Tensor):
        case.assertIsInstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        return
    if isinstance(left, dict):
        case.assertIsInstance(right, dict)
        case.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(case, left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        case.assertIsInstance(right, type(left))
        case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(case, left_item, right_item)
        return
    case.assertEqual(left, right)


class HybridCheckpointRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.registry_path = Path(cls.temporary.name) / "legacy_registry.sqlite3"
        provider = _CountingSyntheticRewardProvider()
        config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=1,
            trajectories_per_batch=2,
        )
        source_space = SearchSpaceConfig(max_depth=6, max_nodes=20)
        plan = resolve_exhaustive_plan(
            source_space,
            ExhaustivePlanningConfig(
                explicit_include_node_counts=(1, 2),
                explicit_exclude_node_counts=tuple(range(3, 21)),
            ),
        )
        cls.registry = ExhaustiveRegistry(cls.registry_path)
        cls.registry.register_plan(
            plan,
            provider_fingerprint=provider.fingerprint(),
            context_fingerprint=provider.manifest()["context_fingerprint"],
        )
        for node_count in (1, 2):
            for candidate in cls.registry.pending_candidates(node_count):
                expression = Expression.from_prefix(candidate.prefix_token_ids)
                assignment = provider.evaluate(expression)
                cls.registry.record_evaluation(
                    candidate.structural_hash,
                    valid=True,
                    reward_details={
                        "reward_result": {
                            "raw_reward": assignment.reward,
                            "reward": assignment.reward,
                            "log_reward": assignment.log_reward,
                        }
                    },
                    target_mass=assignment.reward,
                )
            cls.registry.compute_exact_masses(
                node_count,
                reward_floor=config.reward.reward_floor,
            )

    @classmethod
    def tearDownClass(cls):
        cls.registry.close()
        cls.temporary.cleanup()

    def setUp(self):
        self.config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=1,
            trajectories_per_batch=2,
        )
        self.trainer = self._new_trainer()

    def _new_trainer(self, *, max_cycles: int = 1) -> HybridVarianceTrainer:
        config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=max_cycles,
            trajectories_per_batch=2,
        )
        provider = SyntheticRewardProvider()
        trainer = HybridVarianceTrainer(config, provider, device="cpu")
        semantics = trainer.target_exhaustive_reuse_semantics()
        trainer.configure_hybrid_exhaustive_registry(
            self.registry,
            source_semantics_by_N={1: semantics, 2: semantics},
        )
        return trainer

    @staticmethod
    def _force_pending_condition(
        trainer: HybridVarianceTrainer,
        condition_N: int,
    ) -> None:
        state = trainer.complexity_scheduler.state_dict()
        state["current_permutation"] = (condition_N,) + tuple(
            value
            for value in trainer.config.resolved_condition_node_counts
            if value != condition_N
        )
        state["position"] = 0
        trainer.complexity_scheduler.load_state_dict(state)

    @staticmethod
    def _complete_collection(
        trainer: HybridVarianceTrainer,
        condition_N: int,
    ) -> SingleConditionBatchCollection:
        trajectories = tuple(
            _trajectory(trainer, index=index, condition_N=condition_N)
            for index in range(1, trainer.config.training.trajectories_per_batch + 1)
        )
        return SingleConditionBatchCollection(
            condition_N=condition_N,
            requested_count=len(trajectories),
            trajectories=trajectories,
            sampled_count=len(trajectories),
            invalid_count=0,
            retry_count=0,
            retry_exhausted_count=0,
            sampling_rounds=1,
            sampling_seconds=0.0,
            reward_provider_seconds=0.0,
        )

    def _one_successful_update(self, trainer: HybridVarianceTrainer) -> None:
        self._force_pending_condition(trainer, 3)
        collection = self._complete_collection(trainer, 3)
        with patch.object(
            trainer,
            "collect_single_condition_batch",
            return_value=collection,
        ):
            output = trainer.train_step()
        self.assertTrue(output.updated)

    def test_checkpoint_contains_frozen_hybrid_state_without_lpv_normalizer(self):
        self._one_successful_update(self.trainer)
        path = Path(self.temporary.name) / "hybrid_contract.pt"
        save_hybrid_checkpoint(path, self.trainer)
        payload = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(payload["schema"], HYBRID_CHECKPOINT_SCHEMA)
        self.assertEqual(payload["objective_mode"], HYBRID_OBJECTIVE_MODE)
        self.assertEqual(payload["config_fingerprint"], self.config.fingerprint())
        self.assertEqual(
            [group["name"] for group in payload["optimizer_state"]["param_groups"]],
            ["policy"],
        )
        self.assertFalse(payload["optimizer_contract"]["learned_log_z"])
        self.assertIsNone(payload["optimizer_contract"]["normalizer_optimizer"])
        self.assertEqual(set(payload["exhaustive_registry_equivalence"]), {1, 2})
        self.assertEqual(set(payload["exact_mass_manifest"]), {1, 2})
        self.assertEqual(payload["global_optimizer_step"], 1)
        self.assertEqual(payload["total_trajectories_seen"], 2)
        self.assertEqual(len(payload["diagnostic_history"]), 1)
        self.assertEqual(payload["scheduler_state"]["position"], 1)
        self.assertEqual(
            set(payload["rng_state"]),
            {"python", "numpy", "torch_cpu", "torch_cuda"},
        )

        def all_keys(value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield str(key).lower()
                    yield from all_keys(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    yield from all_keys(nested)

        serialized_keys = set(all_keys(payload))
        for forbidden in (
            "ema_log_z",
            "rolling_log_z",
            "lpv_normalizer",
            "learned_log_z_optimizer_state",
            "normalizer_optimizer_state",
        ):
            self.assertNotIn(forbidden, serialized_keys)
        self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())

    def test_mid_cycle_round_trip_restores_policy_scheduler_history_and_rng(self):
        self._one_successful_update(self.trainer)
        path = Path(self.temporary.name) / "hybrid_mid_cycle.pt"
        self.trainer.save_checkpoint(path)
        expected_python = random.random()
        expected_numpy = float(np.random.random())
        expected_torch = torch.rand(4)

        resumed = self._new_trainer()
        metadata = resumed.load_checkpoint(path)
        self.assertEqual(metadata["global_optimizer_step"], 1)
        self.assertEqual(
            resumed.complexity_scheduler.state_dict(),
            self.trainer.complexity_scheduler.state_dict(),
        )
        self.assertEqual(
            resumed.complexity_scheduler.peek(),
            self.trainer.complexity_scheduler.peek(),
        )
        self.assertEqual(resumed.optimizer_step, self.trainer.optimizer_step)
        self.assertEqual(
            resumed.total_trajectories_seen,
            self.trainer.total_trajectories_seen,
        )
        self.assertEqual(resumed.diagnostic_history, self.trainer.diagnostic_history)
        _assert_nested_equal(
            self,
            resumed.model.state_dict(),
            self.trainer.model.state_dict(),
        )
        _assert_nested_equal(
            self,
            resumed.optimizer.state_dict(),
            self.trainer.optimizer.state_dict(),
        )
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.random()), expected_numpy)
        torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0.0, atol=0.0)

        for _ in range(20):
            original_assignment = self.trainer.complexity_scheduler.peek()
            resumed_assignment = resumed.complexity_scheduler.peek()
            self.assertEqual(original_assignment, resumed_assignment)
            self.trainer.complexity_scheduler.commit(original_assignment)
            resumed.complexity_scheduler.commit(resumed_assignment)

    def test_hybrid_and_legacy_checkpoint_schemas_reject_cross_loading(self):
        hybrid_path = Path(self.temporary.name) / "hybrid_cross_load.pt"
        save_hybrid_checkpoint(hybrid_path, self.trainer)

        class _LegacyStub:
            no_anchor_mode = False

        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_checkpoint(hybrid_path, _LegacyStub())

        legacy_path = Path(self.temporary.name) / "legacy_cross_load.pt"
        torch.save(
            {"schema": "factor_gfn.checkpoint.no_anchor.v1"},
            legacy_path,
        )
        with self.assertRaisesRegex(ValueError, "rejects legacy"):
            load_hybrid_checkpoint(legacy_path, self.trainer)

    def test_config_mismatch_is_rejected_before_trainer_mutation(self):
        path = Path(self.temporary.name) / "hybrid_config_mismatch.pt"
        save_hybrid_checkpoint(path, self.trainer)
        mismatched = self._new_trainer(max_cycles=2)
        model_before = {
            key: value.detach().clone()
            for key, value in mismatched.model.state_dict().items()
        }

        with self.assertRaisesRegex(ValueError, "config fingerprint mismatch"):
            load_hybrid_checkpoint(path, mismatched)

        _assert_nested_equal(self, mismatched.model.state_dict(), model_before)
        self.assertEqual(mismatched.optimizer_step, 0)
        self.assertEqual(mismatched.total_trajectories_seen, 0)

    def test_runner_checkpoints_after_committed_counters_and_resumes(self):
        run_dir = Path(self.temporary.name) / "hybrid_runner"
        runner = create_hybrid_variance_runner(self.trainer, run_dir)
        self._force_pending_condition(self.trainer, 3)
        collection = self._complete_collection(self.trainer, 3)

        with patch.object(
            self.trainer,
            "collect_single_condition_batch",
            return_value=collection,
        ):
            outputs = runner.run_attempts(1)

        self.assertTrue(outputs[0].updated)
        payload = torch.load(
            runner.latest_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(payload["global_optimizer_step"], 1)
        self.assertEqual(payload["total_trajectories_seen"], 2)
        self.assertEqual(payload["scheduler_state"]["position"], 1)
        diagnostics = [
            json.loads(line)
            for line in runner.diagnostics_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(diagnostics, [outputs[0].diagnostics.to_dict()])
        state = json.loads(runner.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["global_optimizer_step"], 1)
        self.assertEqual(state["total_trajectories_seen"], 2)

        resumed_trainer = self._new_trainer()
        resumed_runner = resume_hybrid_variance_runner(run_dir, resumed_trainer)
        self.assertEqual(resumed_runner.trainer.optimizer_step, 1)
        self.assertEqual(resumed_runner.trainer.total_trajectories_seen, 2)
        self.assertEqual(
            resumed_runner.trainer.complexity_scheduler.peek(),
            self.trainer.complexity_scheduler.peek(),
        )
        manifest = json.loads(
            (run_dir / "hybrid_run_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema"], HYBRID_VARIANCE_RUNNER_SCHEMA)

    def test_incomplete_runner_attempt_does_not_replace_checkpoint(self):
        run_dir = Path(self.temporary.name) / "hybrid_runner_incomplete"
        runner = create_hybrid_variance_runner(self.trainer, run_dir)
        checkpoint_before = runner.latest_checkpoint_path.read_bytes()
        state_before = runner.state_path.read_bytes()
        collection = SingleConditionBatchCollection(
            condition_N=self.trainer.complexity_scheduler.peek().condition_N,
            requested_count=2,
            trajectories=(),
            sampled_count=4,
            invalid_count=4,
            retry_count=2,
            retry_exhausted_count=2,
            sampling_rounds=2,
            sampling_seconds=0.0,
            reward_provider_seconds=0.0,
        )

        with patch.object(
            self.trainer,
            "collect_single_condition_batch",
            return_value=collection,
        ):
            outputs = runner.run_attempts(1)

        self.assertFalse(outputs[0].updated)
        self.assertEqual(runner.latest_checkpoint_path.read_bytes(), checkpoint_before)
        self.assertEqual(runner.state_path.read_bytes(), state_before)
        self.assertEqual(self.trainer.optimizer_step, 0)
        self.assertEqual(self.trainer.total_trajectories_seen, 0)


if __name__ == "__main__":
    unittest.main()
