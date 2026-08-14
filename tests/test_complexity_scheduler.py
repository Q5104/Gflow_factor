import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path

from factor_gfn.gfn import (
    BalancedNodeCountScheduler,
    ComplexitySchedulerConfig,
    GFNConfig,
    GFNTrainer,
    ModelConfig,
    RewardAssignment,
    SamplingConfig,
    SearchSpaceConfig,
    SyntheticRewardProvider,
    TrainingConfig,
)


def conditioned_config(
    *,
    batch_size: int = 4,
    retry_budget: int = 1,
    warning_threshold: float | None = None,
    max_steps: int = 10,
) -> GFNConfig:
    return GFNConfig(
        search_space=SearchSpaceConfig(max_depth=2, max_nodes=5),
        model=ModelConfig(
            d_model=16,
            num_heads=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            token_policy_mode="grammar_hierarchical",
        ),
        sampling=SamplingConfig(),
        complexity_scheduler=ComplexitySchedulerConfig(
            enabled=True,
            exhaustive_node_counts=(1,),
            exact_node_retry_budget=retry_budget,
            low_effective_update_rate_warning_threshold=warning_threshold,
        ),
        training=TrainingConfig(
            batch_size=batch_size,
            learning_rate=1e-3,
            log_z_learning_rate=1e-2,
            max_steps=max_steps,
            seed=20260812,
        ),
    )


class _RejectFirstAttemptPerNodeProvider(SyntheticRewardProvider):
    def __init__(self) -> None:
        super().__init__()
        self.attempts: list[int] = []
        self.counts: Counter[int] = Counter()

    def evaluate(self, expression):
        node_count = expression.stats.node_count
        self.attempts.append(node_count)
        self.counts[node_count] += 1
        if self.counts[node_count] == 1:
            return RewardAssignment(valid=False, rejection_reason="first attempt")
        return super().evaluate(expression)


class _RejectOneNodeProvider(SyntheticRewardProvider):
    def __init__(self, rejected_node_count: int) -> None:
        super().__init__()
        self.rejected_node_count = rejected_node_count
        self.attempts: list[int] = []

    def evaluate(self, expression):
        node_count = expression.stats.node_count
        self.attempts.append(node_count)
        if node_count == self.rejected_node_count:
            return RewardAssignment(valid=False, rejection_reason="forced exhaustion")
        return super().evaluate(expression)


class _RejectAllProvider(SyntheticRewardProvider):
    def evaluate(self, expression):
        return RewardAssignment(valid=False, rejection_reason="forced rejection")


class BalancedNodeCountSchedulerTests(unittest.TestCase):
    def test_requested_counts_are_strictly_balanced_after_complete_cycles(self):
        strata = (2, 4, 7, 9)
        scheduler = BalancedNodeCountScheduler(strata, seed=17)
        values = [scheduler.next_node_count() for _ in range(len(strata) * 8)]
        self.assertEqual(Counter(values), Counter({value: 8 for value in strata}))
        for offset in range(0, len(values), len(strata)):
            self.assertEqual(set(values[offset : offset + len(strata)]), set(strata))

    def test_state_round_trip_preserves_all_future_node_counts(self):
        source = BalancedNodeCountScheduler((2, 3, 5, 8), seed=42)
        source.next_batch(7)
        state = source.state_dict()
        expected = source.next_batch(40)
        resumed = BalancedNodeCountScheduler((2, 3, 5, 8), seed=999)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.next_batch(40), expected)


class ConditionedTrainerSchedulingTests(unittest.TestCase):
    def test_config_manifest_persists_resolved_f_e_s_and_retry_contract(self):
        config = conditioned_config(retry_budget=3)
        manifest = config.manifest()
        self.assertEqual(
            manifest["resolved_complexity_strata"],
            {
                "resolved_feasible_node_counts": (1, 2, 3, 4, 5),
                "resolved_exhaustive_node_counts": (1,),
                "resolved_discovery_node_counts": (2, 3, 4, 5),
            },
        )
        self.assertEqual(
            manifest["config"]["complexity_scheduler"]["exact_node_retry_budget"],
            3,
        )

    def test_trainer_requested_counts_balance_over_complete_cycles(self):
        trainer = GFNTrainer(
            conditioned_config(batch_size=2, max_steps=4),
            SyntheticRewardProvider(),
            device="cpu",
        )
        trainer.train(4)
        self.assertEqual(trainer.requested_count_by_N, {2: 2, 3: 2, 4: 2, 5: 2})
        self.assertEqual(
            trainer.successful_update_count_by_N,
            trainer.requested_count_by_N,
        )

    def test_same_n_retry_keeps_slot_assignment_and_counts_attempts(self):
        provider = _RejectFirstAttemptPerNodeProvider()
        trainer = GFNTrainer(conditioned_config(), provider, device="cpu")
        stats = trainer.train_step()
        self.assertFalse(stats.skipped_update)
        self.assertEqual(stats.requested_count_by_N, {2: 1, 3: 1, 4: 1, 5: 1})
        self.assertEqual(stats.sampled_attempt_count_by_N, {2: 2, 3: 2, 4: 2, 5: 2})
        self.assertEqual(stats.valid_count_by_N, {2: 1, 3: 1, 4: 1, 5: 1})
        self.assertEqual(
            stats.successful_update_count_by_N,
            {2: 1, 3: 1, 4: 1, 5: 1},
        )
        self.assertEqual(stats.retry_exhausted_count_by_N, {2: 0, 3: 0, 4: 0, 5: 0})
        self.assertEqual(stats.effective_update_rate_by_N, {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
        first_round = provider.attempts[:4]
        retry_round = provider.attempts[4:]
        self.assertEqual(retry_round, first_round)

    def test_retry_exhaustion_skips_whole_mixed_batch(self):
        provider = _RejectOneNodeProvider(rejected_node_count=3)
        trainer = GFNTrainer(conditioned_config(retry_budget=2), provider, device="cpu")
        stats = trainer.train_step()
        self.assertTrue(stats.skipped_update)
        self.assertEqual(trainer.optimizer_step, 0)
        self.assertEqual(stats.requested_count_by_N, {2: 1, 3: 1, 4: 1, 5: 1})
        self.assertEqual(stats.sampled_attempt_count_by_N[3], 3)
        self.assertEqual(stats.retry_exhausted_count_by_N[3], 1)
        self.assertEqual(stats.valid_count_by_N[3], 0)
        self.assertEqual(
            stats.successful_update_count_by_N,
            {2: 0, 3: 0, 4: 0, 5: 0},
        )
        self.assertEqual(provider.attempts.count(3), 3)
        self.assertTrue(all(value == 3 for value in provider.attempts[4:]))

    def test_warning_does_not_change_scheduler_or_reallocate_quota(self):
        config = conditioned_config(
            retry_budget=0,
            warning_threshold=0.5,
        )
        trainer = GFNTrainer(config, _RejectAllProvider(), device="cpu")
        before = trainer.complexity_scheduler.state_dict()
        expected_scheduler = BalancedNodeCountScheduler(
            trainer.resolved_discovery_node_counts,
            seed=0,
        )
        expected_scheduler.load_state_dict(before)
        expected_scheduler.next_batch(config.training.batch_size)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            stats = trainer.train_step()
        self.assertTrue(stats.skipped_update)
        self.assertTrue(captured)
        self.assertEqual(
            trainer.complexity_scheduler.state_dict(),
            expected_scheduler.state_dict(),
        )
        self.assertEqual(stats.low_effective_update_rate_node_counts, (2, 3, 4, 5))
        self.assertEqual(stats.requested_count_by_N, {2: 1, 3: 1, 4: 1, 5: 1})


if __name__ == "__main__":
    unittest.main()
