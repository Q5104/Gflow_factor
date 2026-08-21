import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from factor_gfn.gfn import (
    ExhaustivePlanningConfig,
    ExhaustiveRegistry,
    HybridVarianceTrainer,
    SyntheticRewardProvider,
    TrajectoryBalanceLoss,
    build_stage5_hybrid_variance_5_15_config,
    count_canonical_terminals,
    resolve_exhaustive_plan,
)
from factor_gfn.gfn.exhaustive_registry_reuse import (
    ExhaustiveReuseSemantics,
    ProvenExhaustiveRewardLookup,
    prove_exhaustive_stratum_reuse,
)
from factor_gfn.grammar import (
    DAILY_DERIVED_ACTION_REGISTRY,
    DAILY_DERIVED_V1_FEATURE_SPACE,
    Expression,
    SearchSpaceConfig,
    action_space_fingerprint,
    state_space_fingerprint,
    transition_space_fingerprint,
)


class _CountingSyntheticRewardProvider(SyntheticRewardProvider):
    def __init__(self) -> None:
        super().__init__()
        self.evaluate_count = 0

    def evaluate(self, expression):
        self.evaluate_count += 1
        return super().evaluate(expression)


def _fingerprint(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class HybridExactZCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.provider = _CountingSyntheticRewardProvider()
        self.hybrid_config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=1
        )

        source_search_space = SearchSpaceConfig(max_depth=6, max_nodes=20)
        planning = ExhaustivePlanningConfig(
            explicit_include_node_counts=(1, 2),
            explicit_exclude_node_counts=tuple(range(3, 21)),
        )
        self.source_plan = resolve_exhaustive_plan(
            source_search_space,
            planning,
        )
        self.registry = ExhaustiveRegistry(
            Path(self.temporary.name) / "legacy_6_20_registry.sqlite3"
        )
        self.registry.register_plan(
            self.source_plan,
            provider_fingerprint=self.provider.fingerprint(),
            context_fingerprint=self.provider.manifest()["context_fingerprint"],
        )
        for node_count in (1, 2):
            for candidate in self.registry.pending_candidates(node_count):
                expression = Expression.from_prefix(candidate.prefix_token_ids)
                assignment = self.provider.evaluate(expression)
                self.registry.record_evaluation(
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
            self.registry.compute_exact_masses(
                node_count,
                reward_floor=self.hybrid_config.reward.reward_floor,
            )
        self.provider.evaluate_count = 0

        provider_manifest = self.provider.manifest()
        interpreter_payload = {
            "provider_schema": provider_manifest.get("schema"),
            "declared_interpreter": False,
        }
        self.semantics = ExhaustiveReuseSemantics(
            grammar_semantics_fingerprint=_fingerprint(
                {
                    "state_space": state_space_fingerprint(),
                    "transition_space": transition_space_fingerprint(),
                }
            ),
            operator_semantics_fingerprint=action_space_fingerprint(),
            interpreter_semantics_fingerprint=_fingerprint(interpreter_payload),
            provider_fingerprint=self.provider.fingerprint(),
            data_context_fingerprint=provider_manifest["context_fingerprint"],
            reward_config_fingerprint=_fingerprint(
                asdict(self.hybrid_config.reward)
            ),
            reward_floor=self.hybrid_config.reward.reward_floor,
        )

    def tearDown(self):
        self.registry.close()
        self.temporary.cleanup()

    def test_legacy_6_20_N1_N2_registry_reuses_under_hybrid_5_15(self):
        self.assertNotEqual(
            self.source_plan.search_space_fingerprint,
            self.hybrid_config.search_space.fingerprint(),
        )
        self.assertEqual(
            self.source_plan.resolved_exhaustive_node_counts,
            (1, 2),
        )

        proofs = {}
        lookups = {}
        for node_count in self.hybrid_config.objective.exact_tb_node_counts:
            target = count_canonical_terminals(
                search_space=self.hybrid_config.search_space,
                target_node_count=node_count,
                canonical_count_cap=None,
            )
            source = self.source_plan.count_result(node_count)
            self.assertEqual(
                tuple(
                    sorted(expression.structural_hash() for expression in target.expressions)
                ),
                tuple(
                    sorted(expression.structural_hash() for expression in source.expressions)
                ),
            )
            proof = prove_exhaustive_stratum_reuse(
                self.registry,
                node_count=node_count,
                target_expressions=target.expressions,
                source_semantics=self.semantics,
                target_semantics=self.semantics,
            )
            proofs[node_count] = proof
            lookups[node_count] = ProvenExhaustiveRewardLookup(
                self.registry,
                proof,
            )

        self.assertEqual(set(proofs), {1, 2})
        self.assertTrue(
            all(proof.exact_aggregation_fingerprint for proof in proofs.values())
        )
        for node_count, lookup in lookups.items():
            expression = self.source_plan.count_result(node_count).expressions[0]
            result = lookup.lookup(expression)
            self.assertTrue(result.valid)
            self.assertEqual(result.reward_source, "exhaustive_registry_cache")
            self.assertTrue(result.metadata["provider_cache_hit"])
        self.assertEqual(self.provider.evaluate_count, 0)

    def test_existing_exact_values_load_as_fixed_float64_buffers(self):
        objective = TrajectoryBalanceLoss(
            max_nodes=self.hybrid_config.search_space.max_nodes,
            exact_node_counts=self.hybrid_config.objective.exact_tb_node_counts,
        )
        exact_values = {}
        for node_count in self.hybrid_config.objective.exact_tb_node_counts:
            result = self.registry.exact_mass_result(node_count)
            exact_values[node_count] = result.exact_tb_log_z
            objective.set_exact_log_z(node_count, result.exact_tb_log_z)

        named_buffers = dict(objective.named_buffers())
        named_parameters = dict(objective.named_parameters())
        exact_buffer = named_buffers["exact_tb_log_z_by_node_count"]
        exact_mask = named_buffers["exact_log_z_mask"]
        self.assertEqual(str(exact_buffer.dtype), "torch.float64")
        self.assertFalse(exact_buffer.requires_grad)
        self.assertNotIn("exact_tb_log_z_by_node_count", named_parameters)
        self.assertEqual(exact_mask[:2].tolist(), [True, True])
        self.assertFalse(any(exact_mask[2:].tolist()))
        for node_count, expected in exact_values.items():
            self.assertEqual(float(exact_buffer[node_count - 1]), expected)

    def test_semantics_mismatch_fails_before_fixed_buffer_mutation(self):
        objective = TrajectoryBalanceLoss(
            max_nodes=self.hybrid_config.search_space.max_nodes,
            exact_node_counts=self.hybrid_config.objective.exact_tb_node_counts,
        )
        before = {
            name: value.detach().clone()
            for name, value in objective.named_buffers()
        }
        changed = replace(
            self.semantics,
            reward_config_fingerprint="different-reward-config",
        )
        target = count_canonical_terminals(
            search_space=self.hybrid_config.search_space,
            target_node_count=1,
            canonical_count_cap=None,
        )

        with self.assertRaisesRegex(ValueError, "semantics mismatch"):
            prove_exhaustive_stratum_reuse(
                self.registry,
                node_count=1,
                target_expressions=target.expressions,
                source_semantics=self.semantics,
                target_semantics=changed,
            )

        for name, value in objective.named_buffers():
            self.assertTrue(value.equal(before[name]), name)

    def test_derived_registry_proves_and_loads_without_raw_fallback(self):
        provider = _CountingSyntheticRewardProvider()
        config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=1,
            feature_space=DAILY_DERIVED_V1_FEATURE_SPACE,
        )
        plan = resolve_exhaustive_plan(
            SearchSpaceConfig(max_depth=2, max_nodes=2),
            ExhaustivePlanningConfig(
                explicit_include_node_counts=(1, 2),
                approve_explicit_include_over_budget=True,
            ),
            action_registry=DAILY_DERIVED_ACTION_REGISTRY,
        )
        path = Path(self.temporary.name) / "derived_registry.sqlite3"
        with ExhaustiveRegistry(
            path,
            action_registry=DAILY_DERIVED_ACTION_REGISTRY,
        ) as registry:
            registry.register_plan(
                plan,
                provider_fingerprint=provider.fingerprint(),
                context_fingerprint=provider.manifest()["context_fingerprint"],
            )
            for candidate in registry.pending_candidates():
                expression = Expression.from_prefix(
                    candidate.prefix_token_ids,
                    action_registry=DAILY_DERIVED_ACTION_REGISTRY,
                )
                assignment = provider.evaluate(expression)
                registry.record_evaluation(
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
            for node_count in (1, 2):
                registry.compute_exact_masses(
                    node_count,
                    reward_floor=config.reward.reward_floor,
                )

        trainer = HybridVarianceTrainer(config, provider, device="cpu")
        semantics = trainer.target_exhaustive_reuse_semantics()
        with ExhaustiveRegistry(
            path,
            read_only=True,
            action_registry=DAILY_DERIVED_ACTION_REGISTRY,
        ) as registry:
            proofs = trainer.configure_hybrid_exhaustive_registry(
                registry,
                source_semantics_by_N={1: semantics, 2: semantics},
            )
        self.assertEqual(set(proofs), {1, 2})
        self.assertEqual(set(trainer.registered_exact_masses_by_N), {1, 2})
        self.assertEqual(
            trainer.action_registry.fingerprint(),
            DAILY_DERIVED_ACTION_REGISTRY.fingerprint(),
        )


if __name__ == "__main__":
    unittest.main()
