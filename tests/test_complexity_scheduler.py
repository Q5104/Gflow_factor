import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path

from factor_gfn.gfn import (
    BalancedNodeCountScheduler,
    ConditionAssignment,
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
    def test_peek_is_stable_and_failure_without_commit_does_not_advance(self):
        scheduler = BalancedNodeCountScheduler((1, 2, 3, 4), seed=17)
        before = scheduler.state_dict()
        pending = scheduler.peek()

        self.assertEqual(scheduler.peek(), pending)
        self.assertEqual(scheduler.state_dict(), before)
        self.assertEqual(pending.cycle_index, 0)
        self.assertEqual(pending.condition_position_in_cycle, 0)

        scheduler.commit(pending)
        self.assertEqual(scheduler.position, 1)
        self.assertNotEqual(scheduler.peek(), pending)

    def test_commit_rejects_stale_double_and_foreign_assignments(self):
        scheduler = BalancedNodeCountScheduler((1, 2, 3), seed=5)
        pending = scheduler.peek()
        scheduler.commit(pending)

        with self.assertRaisesRegex(ValueError, "stale"):
            scheduler.commit(pending)
        with self.assertRaisesRegex(ValueError, "stale"):
            scheduler.commit(
                ConditionAssignment(
                    cycle_index=scheduler.cycle_index,
                    condition_position_in_cycle=scheduler.position,
                    condition_N=999,
                )
            )
        with self.assertRaisesRegex(TypeError, "ConditionAssignment"):
            scheduler.commit((0, 0, 1))  # type: ignore[arg-type]

    def test_transactional_cycles_cover_each_hybrid_condition_once(self):
        conditions = tuple(range(1, 16))
        scheduler = BalancedNodeCountScheduler(conditions, seed=42)
        assignments = []
        for _ in range(2 * len(conditions)):
            assignment = scheduler.peek()
            assignments.append(assignment)
            scheduler.commit(assignment)

        for cycle_index in range(2):
            cycle = [
                assignment
                for assignment in assignments
                if assignment.cycle_index == cycle_index
            ]
            self.assertEqual(
                [assignment.condition_position_in_cycle for assignment in cycle],
                list(range(15)),
            )
            self.assertEqual(
                {assignment.condition_N for assignment in cycle},
                set(conditions),
            )

    def test_transactional_shuffle_is_seed_reproducible(self):
        left = BalancedNodeCountScheduler(tuple(range(1, 16)), seed=20260816)
        right = BalancedNodeCountScheduler(tuple(range(1, 16)), seed=20260816)
        left_assignments = []
        right_assignments = []
        for _ in range(40):
            left_pending = left.peek()
            right_pending = right.peek()
            left_assignments.append(left_pending)
            right_assignments.append(right_pending)
            left.commit(left_pending)
            right.commit(right_pending)
        self.assertEqual(left_assignments, right_assignments)

    def test_state_round_trip_preserves_pending_assignment_and_future(self):
        source = BalancedNodeCountScheduler((1, 2, 3, 4, 5), seed=31)
        for _ in range(5):
            pending = source.peek()
            source.commit(pending)
        state_at_cycle_boundary = source.state_dict()
        pending = source.peek()
        self.assertEqual(source.state_dict(), state_at_cycle_boundary)

        resumed = BalancedNodeCountScheduler((1, 2, 3, 4, 5), seed=999)
        resumed.load_state_dict(state_at_cycle_boundary)
        self.assertEqual(resumed.peek(), pending)

        source_future = []
        resumed_future = []
        for _ in range(20):
            source_pending = source.peek()
            resumed_pending = resumed.peek()
            source_future.append(source_pending)
            resumed_future.append(resumed_pending)
            source.commit(source_pending)
            resumed.commit(resumed_pending)
        self.assertEqual(resumed_future, source_future)

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
