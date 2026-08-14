import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from factor_gfn.gfn.calibration_stability import (
    CalibrationStabilityConfig,
    assess_calibration_stability,
)
from factor_gfn.gfn.exhaustive import (
    ExhaustivePlanningConfig,
    ExhaustiveRegistry,
    count_canonical_terminals,
    resolve_exhaustive_plan,
)
from factor_gfn.gfn.exhaustive_registry_reuse import (
    ExhaustiveReuseSemantics,
    ProvenExhaustiveRewardLookup,
    prove_exhaustive_stratum_reuse,
)
from factor_gfn.gfn.no_anchor_contract import NoAnchorStrataContract
from factor_gfn.grammar import SearchSpaceConfig


class NoAnchorStrataContractTests(unittest.TestCase):
    def test_discovery_is_all_feasible_and_normalizer_types_partition_it(self):
        contract = NoAnchorStrataContract.resolve(range(1, 21), (1, 2))
        self.assertEqual(contract.discovery_node_counts, tuple(range(1, 21)))
        self.assertEqual(contract.exact_normalizer_node_counts, (1, 2))
        self.assertEqual(contract.learned_normalizer_node_counts, tuple(range(3, 21)))
        manifest = contract.manifest()
        self.assertTrue(manifest["normal_discovery_equals_feasible"])
        self.assertFalse(any("anchor" in key.lower() for key in manifest))

    def test_old_e_not_discovery_semantics_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "D=F"):
            NoAnchorStrataContract(
                feasible_node_counts=(1, 2, 3),
                discovery_node_counts=(3,),
                exact_normalizer_node_counts=(1, 2),
                learned_normalizer_node_counts=(3,),
            )
        with self.assertRaisesRegex(ValueError, "L must equal F-E"):
            NoAnchorStrataContract(
                feasible_node_counts=(1, 2, 3),
                discovery_node_counts=(1, 2, 3),
                exact_normalizer_node_counts=(1,),
                learned_normalizer_node_counts=(3,),
            )


class CalibrationStabilityTests(unittest.TestCase):
    @staticmethod
    def config() -> CalibrationStabilityConfig:
        return CalibrationStabilityConfig()

    def test_collects_until_64_and_fails_closed_if_valid_count_never_reaches_it(self):
        collecting = assess_calibration_stability(
            [0.0] * 63, requested_slots=100, config=self.config()
        )
        self.assertEqual(collecting.status, "collecting")
        exhausted = assess_calibration_stability(
            [0.0] * 63, requested_slots=128, config=self.config()
        )
        self.assertEqual(exhausted.status, "fail_closed")

    def test_stable_at_64_or_continues_then_fails_closed_at_128(self):
        stable = assess_calibration_stability(
            [1.0] * 64, requested_slots=64, config=self.config()
        )
        self.assertEqual(stable.status, "stable")
        unstable_values = [0.0] * 48 + [10.0] * 16
        continuing = assess_calibration_stability(
            unstable_values, requested_slots=64, config=self.config()
        )
        self.assertEqual(continuing.status, "continue_to_hard_limit")
        exhausted = assess_calibration_stability(
            [0.0] * 112 + [10.0] * 16,
            requested_slots=128,
            config=self.config(),
        )
        self.assertEqual(exhausted.status, "fail_closed")

    def test_thresholds_are_explicit_and_64_128_contract_is_fixed(self):
        with self.assertRaisesRegex(ValueError, "must be 64"):
            CalibrationStabilityConfig(minimum_valid_samples=32)
        with self.assertRaisesRegex(ValueError, "must be 128"):
            CalibrationStabilityConfig(maximum_requested_slots=96)
        self.assertFalse(
            any("relative" in field for field in asdict(self.config()))
        )

    def test_stability_is_only_rechecked_each_16_additional_valid_samples(self):
        values = [0.0] * 48 + [10.0] * 17
        waiting = assess_calibration_stability(
            values,
            requested_slots=65,
            config=self.config(),
        )
        self.assertEqual(waiting.status, "continue_to_hard_limit")
        self.assertIn("next 16-valid-sample", waiting.reason)
        self.assertIsNone(waiting.median_shift)


class ExhaustiveRegistryReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.registry = ExhaustiveRegistry(
            Path(self.temporary.name) / "registry.sqlite3"
        )
        self.plan = resolve_exhaustive_plan(
            SearchSpaceConfig(max_depth=2, max_nodes=2),
            ExhaustivePlanningConfig(),
        )
        self.registry.register_plan(
            self.plan,
            provider_fingerprint="provider-v1",
            context_fingerprint="context-v1",
        )
        for index, candidate in enumerate(self.registry.pending_candidates(1)):
            if index == 0:
                self.registry.record_evaluation(
                    candidate.structural_hash,
                    valid=False,
                    rejection_reason="synthetic invalid",
                    reward_details={"audit": "kept"},
                )
                continue
            raw_reward = float(index) / 10.0
            reward = max(raw_reward, 1e-8)
            self.registry.record_evaluation(
                candidate.structural_hash,
                valid=True,
                reward_details={
                    "reward_result": {
                        "raw_reward": raw_reward,
                        "reward": reward,
                        "log_reward": math.log(reward),
                    },
                    "audit": "kept",
                },
                target_mass=reward,
            )
        self.registry.compute_exact_masses(1, reward_floor=1e-8)
        self.semantics = ExhaustiveReuseSemantics(
            grammar_semantics_fingerprint="grammar-v1",
            operator_semantics_fingerprint="operator-v1",
            interpreter_semantics_fingerprint="interpreter-v1",
            provider_fingerprint="provider-v1",
            data_context_fingerprint="context-v1",
            reward_config_fingerprint="reward-config-v1",
            reward_floor=1e-8,
        )

    def tearDown(self):
        self.registry.close()
        self.temporary.cleanup()

    def _expressions(self):
        return self.plan.count_result(1).expressions

    def test_per_stratum_proof_and_valid_invalid_reward_lookup(self):
        # The target enumeration deliberately uses a different global search
        # boundary.  N=1 identity, not the global search fingerprint, governs
        # reuse.
        target_expressions = count_canonical_terminals(
            search_space=SearchSpaceConfig(max_depth=7, max_nodes=20),
            target_node_count=1,
        ).expressions
        proof = prove_exhaustive_stratum_reuse(
            self.registry,
            node_count=1,
            target_expressions=target_expressions,
            source_semantics=self.semantics,
            target_semantics=self.semantics,
        )
        self.assertEqual(len(proof.canonical_structural_hashes), 6)
        lookup = ProvenExhaustiveRewardLookup(self.registry, proof)
        results = [lookup.lookup(expression) for expression in self._expressions()]
        self.assertEqual(sum(result.valid for result in results), 5)
        invalid = next(result for result in results if not result.valid)
        self.assertEqual(invalid.rejection_reason, "synthetic invalid")
        self.assertEqual(invalid.metadata["audit"], "kept")
        valid = next(result for result in results if result.valid)
        self.assertGreater(valid.reward, 0.0)
        self.assertEqual(valid.reward_source, "exhaustive_registry_cache")
        self.assertTrue(valid.metadata["provider_cache_hit"])

    def test_semantics_or_canonical_hash_mismatch_fails_closed(self):
        changed = ExhaustiveReuseSemantics(
            **{
                **asdict(self.semantics),
                "interpreter_semantics_fingerprint": "interpreter-v2",
            }
        )
        with self.assertRaisesRegex(ValueError, "semantics mismatch"):
            prove_exhaustive_stratum_reuse(
                self.registry,
                node_count=1,
                target_expressions=self._expressions(),
                source_semantics=self.semantics,
                target_semantics=changed,
            )
        with self.assertRaisesRegex(ValueError, "hash set mismatch"):
            prove_exhaustive_stratum_reuse(
                self.registry,
                node_count=1,
                target_expressions=self._expressions()[:-1],
                source_semantics=self.semantics,
                target_semantics=self.semantics,
            )


if __name__ == "__main__":
    unittest.main()
