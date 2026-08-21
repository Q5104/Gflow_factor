import tempfile
import unittest
import json
import sqlite3
from pathlib import Path

from factor_gfn.gfn import (
    EXHAUSTIVE_SOURCE,
    ExhaustivePlanningConfig,
    ExhaustiveRegistry,
    SyntheticRewardProvider,
    count_canonical_terminals,
    resolve_exhaustive_plan,
)
from factor_gfn.gfn.diagnostic_support import build_or_resume_n1_n2_registry
from factor_gfn.grammar import (
    DAILY_DERIVED_ACTION_REGISTRY,
    RAW_ACTION_REGISTRY,
    Expression,
    SearchSpaceConfig,
)


class BoundedCanonicalCountingTests(unittest.TestCase):
    def test_known_n1_n2_counts_depths_and_unique_structural_hashes(self):
        search_space = SearchSpaceConfig(max_depth=2, max_nodes=2)
        n1 = count_canonical_terminals(
            search_space=search_space,
            target_node_count=1,
        )
        n2 = count_canonical_terminals(
            search_space=search_space,
            target_node_count=2,
        )
        self.assertEqual(n1.canonical_terminal_count, 6)
        self.assertEqual(n1.depth_distribution, {0: 6})
        self.assertEqual(n2.canonical_terminal_count, 636)
        self.assertEqual(n2.depth_distribution, {1: 636})
        for result in (n1, n2):
            self.assertTrue(result.canonical_count_exact)
            self.assertFalse(result.count_cap_reached)
            self.assertEqual(result.count_relation, "=")
            hashes = [expression.structural_hash() for expression in result.expressions]
            self.assertEqual(len(hashes), len(set(hashes)))
            for expression in result.expressions:
                rebuilt = Expression.from_prefix(expression.to_prefix())
                self.assertEqual(rebuilt.structural_hash(), expression.structural_hash())

    def test_derived_n1_n2_counts_and_prefix_roundtrip_use_derived_registry(self):
        search_space = SearchSpaceConfig(max_depth=2, max_nodes=2)
        results = tuple(
            count_canonical_terminals(
                search_space=search_space,
                target_node_count=node_count,
                canonical_count_cap=None,
                action_registry=DAILY_DERIVED_ACTION_REGISTRY,
            )
            for node_count in (1, 2)
        )
        self.assertEqual(
            tuple(result.canonical_terminal_count for result in results),
            (16, 1_696),
        )
        self.assertEqual(
            tuple(result.depth_distribution for result in results),
            ({0: 16}, {1: 1_696}),
        )
        for result in results:
            hashes = [expression.structural_hash() for expression in result.expressions]
            self.assertEqual(len(hashes), len(set(hashes)))
            for expression in result.expressions:
                rebuilt = Expression.from_prefix(
                    expression.to_prefix(),
                    action_registry=DAILY_DERIVED_ACTION_REGISTRY,
                )
                self.assertEqual(rebuilt.structural_hash(), expression.structural_hash())
                self.assertEqual(
                    rebuilt.action_registry.fingerprint(),
                    DAILY_DERIVED_ACTION_REGISTRY.fingerprint(),
                )

    def test_raw_plan_manifest_and_fingerprint_remain_historical(self):
        search_space = SearchSpaceConfig(max_depth=2, max_nodes=2)
        default = resolve_exhaustive_plan(search_space)
        explicit_raw = resolve_exhaustive_plan(
            search_space,
            action_registry=RAW_ACTION_REGISTRY,
        )
        self.assertEqual(default.manifest(), explicit_raw.manifest())
        self.assertNotIn("vocabulary", default.manifest())
        self.assertEqual(
            default.fingerprint(),
            "f3a7f13de5464d6ef7b30fdf75a868f0c52305a39db4a6f0b4f8b6607ca56dc9",
        )

    def test_cap_stops_at_cap_plus_one_and_reports_strict_lower_bound(self):
        result = count_canonical_terminals(
            search_space=SearchSpaceConfig(max_depth=2, max_nodes=2),
            target_node_count=2,
            canonical_count_cap=100,
            estimated_seconds_per_candidate=0.75,
        )
        self.assertEqual(result.canonical_terminal_count, 101)
        self.assertTrue(result.count_cap_reached)
        self.assertFalse(result.canonical_count_exact)
        self.assertEqual(result.count_relation, ">")
        self.assertFalse(result.depth_distribution_exact)
        self.assertEqual(result.estimated_evaluation_seconds, 75.75)
        self.assertTrue(result.estimated_evaluation_seconds_is_lower_bound)


class ExhaustiveResolutionTests(unittest.TestCase):
    def test_default_cumulative_budget_resolves_n1_n2_but_not_capped_n3(self):
        plan = resolve_exhaustive_plan(
            SearchSpaceConfig(max_depth=2, max_nodes=3),
            ExhaustivePlanningConfig(),
        )
        self.assertEqual(plan.exhaustive_budget_seconds, 720.0)
        self.assertEqual(plan.resolved_exhaustive_node_counts, (1, 2))
        self.assertEqual(plan.automatic_exhaustive_node_counts, (1, 2))
        self.assertEqual(plan.resolved_discovery_node_counts, (3,))
        self.assertEqual(plan.resolved_estimated_evaluation_seconds, 481.5)
        n3 = plan.count_result(3)
        self.assertEqual(n3.canonical_terminal_count, 10_001)
        self.assertEqual(n3.count_relation, ">")
        self.assertTrue(n3.count_cap_reached)

    def test_budget_is_cumulative_not_reapplied_per_stratum(self):
        plan = resolve_exhaustive_plan(
            SearchSpaceConfig(max_depth=2, max_nodes=2),
            ExhaustivePlanningConfig(
                planned_real_reward_budget_seconds=480.0,
                max_budget_fraction=1.0,
            ),
        )
        self.assertEqual(plan.resolved_exhaustive_node_counts, (1,))
        self.assertEqual(plan.resolved_discovery_node_counts, (2,))
        self.assertEqual(plan.resolved_estimated_evaluation_seconds, 4.5)

    def test_explicit_exclude_wins_and_overlap_is_rejected(self):
        plan = resolve_exhaustive_plan(
            SearchSpaceConfig(max_depth=2, max_nodes=2),
            ExhaustivePlanningConfig(explicit_exclude_node_counts=(1,)),
        )
        self.assertEqual(plan.resolved_exhaustive_node_counts, (2,))
        self.assertEqual(plan.resolved_discovery_node_counts, (1,))
        self.assertNotIn(1, {item.node_count for item in plan.count_results})
        with self.assertRaisesRegex(ValueError, "overlap"):
            ExhaustivePlanningConfig(
                explicit_include_node_counts=(2,),
                explicit_exclude_node_counts=(2,),
            )

    def test_explicit_include_over_budget_requires_secondary_approval(self):
        search_space = SearchSpaceConfig(max_depth=2, max_nodes=2)
        config = ExhaustivePlanningConfig(
            planned_real_reward_budget_seconds=100.0,
            max_budget_fraction=1.0,
            explicit_include_node_counts=(2,),
        )
        with self.assertRaisesRegex(ValueError, "secondary approval"):
            resolve_exhaustive_plan(search_space, config)
        approved = resolve_exhaustive_plan(
            search_space,
            ExhaustivePlanningConfig(
                planned_real_reward_budget_seconds=100.0,
                max_budget_fraction=1.0,
                explicit_include_node_counts=(2,),
                approve_explicit_include_over_budget=True,
            ),
        )
        self.assertEqual(approved.resolved_exhaustive_node_counts, (2,))
        self.assertEqual(approved.resolved_discovery_node_counts, (1,))
        self.assertTrue(approved.explicit_over_budget_approval_used)
        self.assertEqual(approved.resolved_estimated_evaluation_seconds, 477.0)


class ExhaustiveRegistryTests(unittest.TestCase):
    def _plan(self):
        return resolve_exhaustive_plan(
            SearchSpaceConfig(max_depth=2, max_nodes=2),
            ExhaustivePlanningConfig(),
        )

    def test_registry_is_resumable_auditable_and_keeps_invalid_zero_mass(self):
        plan = self._plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exhaustive_registry.sqlite3"
            with ExhaustiveRegistry(path) as registry:
                registry.register_plan(
                    plan,
                    provider_fingerprint="provider-v1",
                    context_fingerprint="context-v1",
                )
                # Idempotent registration must not duplicate canonical candidates.
                registry.register_plan(
                    plan,
                    provider_fingerprint="provider-v1",
                    context_fingerprint="context-v1",
                )
                self.assertEqual(registry.coverage(1)["registered_count"], 6)
                self.assertEqual(registry.coverage(2)["registered_count"], 636)
                self.assertTrue(registry.coverage(2)["enumeration_complete"])
                self.assertFalse(registry.coverage(2)["evaluation_complete"])
                self.assertEqual(len(registry.pending_candidates()), 642)

                first, second = registry.pending_candidates()[:2]
                registry.record_evaluation(
                    first.structural_hash,
                    valid=False,
                    rejection_reason="synthetic invalid",
                    reward_details={"reason": "non-finite"},
                )
                registry.record_evaluation(
                    second.structural_hash,
                    valid=True,
                    reward_details={"reward": 2.5, "log_reward": 0.916},
                    target_mass=2.5,
                )
                invalid = registry.candidate(first.structural_hash)
                self.assertEqual(invalid.source, EXHAUSTIVE_SOURCE)
                self.assertFalse(invalid.valid)
                self.assertEqual(invalid.target_mass, 0.0)
                self.assertEqual(invalid.provider_fingerprint, "provider-v1")
                self.assertEqual(invalid.context_fingerprint, "context-v1")
                self.assertEqual(
                    Expression.from_prefix(invalid.prefix_token_ids).structural_hash(),
                    invalid.structural_hash,
                )
                with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                    registry.record_evaluation(
                        first.structural_hash,
                        valid=False,
                        rejection_reason="different reason",
                        reward_details={"reason": "different"},
                    )

            with ExhaustiveRegistry(path) as resumed:
                pending_hashes = {
                    candidate.structural_hash
                    for candidate in resumed.pending_candidates()
                }
                self.assertNotIn(first.structural_hash, pending_hashes)
                self.assertNotIn(second.structural_hash, pending_hashes)
                self.assertEqual(len(pending_hashes), 640)

            with ExhaustiveRegistry(path, read_only=True) as read_only:
                self.assertTrue(read_only.read_only)
                self.assertEqual(read_only.coverage(1)["registered_count"], 6)
                self.assertEqual(read_only.candidate(first.structural_hash), invalid)
                with self.assertRaisesRegex(RuntimeError, "read-only"):
                    read_only.register_plan(
                        plan,
                        provider_fingerprint="provider-v1",
                        context_fingerprint="context-v1",
                    )

    def test_coverage_completion_and_discovery_separation(self):
        plan = resolve_exhaustive_plan(
            SearchSpaceConfig(max_depth=2, max_nodes=3),
            ExhaustivePlanningConfig(canonical_count_cap=100),
        )
        self.assertEqual(plan.resolved_exhaustive_node_counts, (1,))
        self.assertEqual(plan.resolved_discovery_node_counts, (2, 3))
        with tempfile.TemporaryDirectory() as directory:
            with ExhaustiveRegistry(Path(directory) / "pool.sqlite3") as registry:
                registry.register_plan(
                    plan,
                    provider_fingerprint="synthetic-provider",
                    context_fingerprint="synthetic-context",
                )
                candidates = registry.pending_candidates()
                self.assertEqual({candidate.node_count for candidate in candidates}, {1})
                for index, candidate in enumerate(candidates):
                    if index == 0:
                        registry.record_evaluation(
                            candidate.structural_hash,
                            valid=False,
                            rejection_reason="synthetic rejection",
                            reward_details={"index": index},
                        )
                    else:
                        registry.record_evaluation(
                            candidate.structural_hash,
                            valid=True,
                            reward_details={"index": index, "reward": 1.0},
                            target_mass=1.0,
                        )
                coverage = registry.coverage(1)
                self.assertTrue(coverage["coverage_complete"])
                self.assertEqual(coverage["evaluated_count"], 6)
                self.assertEqual(coverage["invalid_count"], 1)
                self.assertEqual(coverage["valid_count"], 5)

    def test_derived_registry_binds_vocabulary_and_rejects_cross_open(self):
        plan = resolve_exhaustive_plan(
            SearchSpaceConfig(max_depth=2, max_nodes=2),
            ExhaustivePlanningConfig(
                explicit_include_node_counts=(1, 2),
                approve_explicit_include_over_budget=True,
            ),
            action_registry=DAILY_DERIVED_ACTION_REGISTRY,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "derived.sqlite3"
            with ExhaustiveRegistry(
                path,
                action_registry=DAILY_DERIVED_ACTION_REGISTRY,
            ) as registry:
                registry.register_plan(
                    plan,
                    provider_fingerprint="derived-provider",
                    context_fingerprint="derived-context",
                )
                self.assertEqual(registry.coverage(1)["registered_count"], 16)
                self.assertEqual(registry.coverage(2)["registered_count"], 1_696)

            connection = sqlite3.connect(path)
            try:
                metadata = {
                    key: json.loads(value)
                    for key, value in connection.execute(
                        "SELECT key, value_json FROM metadata"
                    )
                }
            finally:
                connection.close()
            self.assertEqual(metadata["feature_space_id"], "daily_derived_v1")
            self.assertEqual(metadata["action_count"], 152)
            self.assertEqual(
                metadata["action_space_fingerprint"],
                DAILY_DERIVED_ACTION_REGISTRY.fingerprint(),
            )
            with self.assertRaisesRegex(ValueError, "vocabulary identity mismatch"):
                ExhaustiveRegistry(path, read_only=True)
            with ExhaustiveRegistry(
                path,
                read_only=True,
                action_registry=DAILY_DERIVED_ACTION_REGISTRY,
            ) as read_only:
                self.assertEqual(read_only.coverage(2)["registered_count"], 1_696)

    def test_legacy_raw_registry_rejects_derived_before_metadata_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.sqlite3"
            with ExhaustiveRegistry(path) as registry:
                registry.register_plan(
                    self._plan(),
                    provider_fingerprint="raw-provider",
                    context_fingerprint="raw-context",
                )
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Raw-only"):
                ExhaustiveRegistry(
                    path,
                    action_registry=DAILY_DERIVED_ACTION_REGISTRY,
                )
            self.assertEqual(path.read_bytes(), before)

    def test_derived_builder_is_complete_resumable_and_reports_dynamic_progress(self):
        provider = SyntheticRewardProvider()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "derived.sqlite3"
            first_progress: list[str] = []
            registry = build_or_resume_n1_n2_registry(
                path,
                provider,
                reward_floor=1e-8,
                progress_every=2_000,
                progress=first_progress.append,
                action_registry=DAILY_DERIVED_ACTION_REGISTRY,
                approve_explicit_include_over_budget=True,
            )
            try:
                self.assertTrue(registry.coverage(1)["coverage_complete"])
                self.assertTrue(registry.coverage(2)["coverage_complete"])
                self.assertEqual(len(registry.pending_candidates()), 0)
                self.assertEqual(
                    set(registry.exact_mass_result(n).node_count for n in (1, 2)),
                    {1, 2},
                )
            finally:
                registry.close()

            resumed_progress: list[str] = []
            resumed = build_or_resume_n1_n2_registry(
                path,
                provider,
                reward_floor=1e-8,
                progress=resumed_progress.append,
                action_registry=DAILY_DERIVED_ACTION_REGISTRY,
                approve_explicit_include_over_budget=True,
            )
            resumed.close()
            self.assertIn(
                "[exhaustive] pending=0, completed_before=1712",
                resumed_progress,
            )


if __name__ == "__main__":
    unittest.main()
