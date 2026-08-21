import unittest

import numpy as np
import torch
from torch import nn

from factor_gfn.gfn import (
    HybridVarianceTrainer,
    ModelConfig,
    PolicyOutput,
    RewardAssignment,
    SamplingConfig,
    sample_trajectory,
    state_hash,
)
from factor_gfn.gfn.hybrid_config import build_stage5_hybrid_variance_5_15_config
from factor_gfn.gfn.model import ForwardPolicyNetwork
from factor_gfn.gfn.state_adapter import StateAdapter
from factor_gfn.grammar import (
    DAILY_DERIVED_ACTION_REGISTRY,
    DAILY_DERIVED_FEATURE_NAMES,
    DAILY_DERIVED_V1_FEATURE_SPACE,
    NON_LEAF_OPERATORS,
    RAW_ACTION_REGISTRY,
    RAW_DAILY_FEATURE_NAMES,
    RAW_DAILY_FEATURE_SPACE,
    DAGAction,
    Expression,
    ExpressionNode,
    ExactNodeGrammarState,
    GrammarState,
    build_action_registry,
)


RAW_ACTION_FINGERPRINT = (
    "5689dbceb1bb42716773bcaf4cb5845041e578a3bb11fe67445ede6cde7938cc"
)
DERIVED_ACTION_FINGERPRINT = (
    "57b836ec59632df44517ccc14486e57bf5ab9d8b9184ac153c899f59b4a61193"
)
RAW_TS_MEAN_CLOSE_5_HASH = (
    "d6a8bd4626831fbca32543721bc72e4e3e7a0615cee5066c898cc534017fecf6"
)
DERIVED_TS_MEAN_RET_CC1_5_HASH = (
    "a3105c118393efa3ebb960b83a98c4d7b931897ce80b55910a7068b958cd8aad"
)
RAW_HYBRID_CONFIG_FINGERPRINT = (
    "c18bc89438b6570ef4f528790da9a58cb3004f588e063ee63d774149d13ab9ec"
)
RAW_SOURCE_STATE_HASH = (
    "c0200f02f624b04c75e1159189aea602937199a1e1dda8b022b6080993d56323"
)
RAW_CLOSE_STATE_HASH = (
    "a401a62c74f686979055b6ea9908b0b6f8db59e9003581a77b1a6fb618051eb8"
)


class _DerivedHighTokenPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_registry = DAILY_DERIVED_ACTION_REGISTRY
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch, *, temperature: float = 1.0) -> PolicyOutput:
        batch_size, slot_count = batch.slot_mask.shape
        slot_log_probs = torch.full(
            (batch_size, slot_count), -torch.inf, device=self.anchor.device
        )
        token_log_probs = torch.full(
            (batch_size, slot_count, self.action_registry.action_count),
            -torch.inf,
            device=self.anchor.device,
        )
        root_id = self.action_registry.get_action_id("ts_beta", 5)
        leaf_id = self.action_registry.get_action_id("ret_cc1")
        for row in range(batch_size):
            slot = int(torch.nonzero(batch.slot_mask[row], as_tuple=False)[0, 0])
            token_id = (
                root_id
                if bool(batch.legal_token_mask[row, slot, root_id])
                else leaf_id
            )
            slot_log_probs[row, slot] = self.anchor * 0.0
            token_log_probs[row, slot, token_id] = self.anchor * 0.0
        return PolicyOutput(
            slot_logits=slot_log_probs,
            token_logits=token_log_probs,
            slot_log_probs=slot_log_probs,
            token_log_probs=token_log_probs,
            slot_mask=batch.slot_mask,
            legal_token_mask=batch.legal_token_mask,
        )


class _AlwaysValidProvider:
    def __init__(self, feature_space_id: str | None = None) -> None:
        if feature_space_id is not None:
            self.context = type(
                "Context",
                (),
                {"expression_feature_space_id": feature_space_id},
            )()

    def evaluate(self, expression: Expression) -> RewardAssignment:
        return RewardAssignment(valid=True, reward=1.0, log_reward=0.0)

    def fingerprint(self) -> str:
        return "f" * 64

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "test.always_valid.v1",
            "data_scope": "training_only",
            "context_fingerprint": "test",
        }


def _two_node_expression(registry, leaf_name: str) -> Expression:
    return Expression.from_prefix(
        (
            registry.get_action_id("ts_mean", 5),
            registry.get_action_id(leaf_name),
        ),
        action_registry=registry,
    )


def _source(registry) -> GrammarState:
    return GrammarState(max_depth=2, max_nodes=3, action_registry=registry)


class ActionRegistryTests(unittest.TestCase):
    def test_raw_contract_is_byte_compatible(self) -> None:
        self.assertEqual(RAW_ACTION_REGISTRY.leaf_names, RAW_DAILY_FEATURE_NAMES)
        self.assertEqual(RAW_ACTION_REGISTRY.action_count, 142)
        self.assertEqual(
            [RAW_ACTION_REGISTRY.get_action_id(name) for name in RAW_DAILY_FEATURE_NAMES],
            list(range(6)),
        )
        self.assertEqual(RAW_ACTION_REGISTRY.get_action_id("ts_mean", 5), 37)
        self.assertEqual(RAW_ACTION_REGISTRY.fingerprint(), RAW_ACTION_FINGERPRINT)
        self.assertEqual(
            _two_node_expression(RAW_ACTION_REGISTRY, "close").structural_hash(),
            RAW_TS_MEAN_CLOSE_5_HASH,
        )

    def test_derived_contract_and_shared_non_leaf_semantics(self) -> None:
        self.assertEqual(
            DAILY_DERIVED_ACTION_REGISTRY.leaf_names,
            DAILY_DERIVED_FEATURE_NAMES,
        )
        self.assertEqual(DAILY_DERIVED_ACTION_REGISTRY.action_count, 152)
        self.assertEqual(
            DAILY_DERIVED_ACTION_REGISTRY.fingerprint(),
            DERIVED_ACTION_FINGERPRINT,
        )
        self.assertNotEqual(
            DAILY_DERIVED_ACTION_REGISTRY.feature_space_fingerprint,
            RAW_ACTION_REGISTRY.feature_space_fingerprint,
        )
        for operator in NON_LEAF_OPERATORS:
            with self.subTest(operator=operator.name):
                derived = DAILY_DERIVED_ACTION_REGISTRY.get_operator(operator.name)
                self.assertEqual(derived, operator)

    def test_registries_are_order_independent_and_isolated(self) -> None:
        derived_first = build_action_registry(DAILY_DERIVED_V1_FEATURE_SPACE)
        raw_second = build_action_registry(RAW_DAILY_FEATURE_SPACE)
        raw_first = build_action_registry(RAW_DAILY_FEATURE_SPACE)
        derived_second = build_action_registry(DAILY_DERIVED_V1_FEATURE_SPACE)
        self.assertEqual(raw_first.fingerprint(), raw_second.fingerprint())
        self.assertEqual(derived_first.fingerprint(), derived_second.fingerprint())
        with self.assertRaises(KeyError):
            RAW_ACTION_REGISTRY.get_action_id("ret_cc1")
        with self.assertRaises(KeyError):
            DAILY_DERIVED_ACTION_REGISTRY.get_action_id("close")


class RegistryBoundGrammarTests(unittest.TestCase):
    def test_raw_and_derived_expressions_construct_formally(self) -> None:
        raw = _two_node_expression(RAW_ACTION_REGISTRY, "close")
        derived = _two_node_expression(DAILY_DERIVED_ACTION_REGISTRY, "ret_cc1")
        self.assertEqual(str(raw), "ts_mean(close, 5)")
        self.assertEqual(str(derived), "ts_mean(ret_cc1, 5)")
        self.assertEqual(derived.structural_hash(), DERIVED_TS_MEAN_RET_CC1_5_HASH)

    def test_simple_trajectories_complete_under_each_registry(self) -> None:
        for registry, leaf in (
            (RAW_ACTION_REGISTRY, "close"),
            (DAILY_DERIVED_ACTION_REGISTRY, "ret_cc1"),
        ):
            with self.subTest(feature_space=registry.feature_space.feature_space_id):
                state = _source(registry)
                self.assertEqual(
                    state.get_legal_token_mask(()).shape,
                    (registry.action_count,),
                )
                state = state.step(
                    DAGAction((), registry.get_action_id("ts_mean", 5), registry)
                )
                state = state.step(
                    DAGAction((0,), registry.get_action_id(leaf), registry)
                )
                self.assertTrue(state.done)
                self.assertEqual(state.to_expression().action_registry, registry)

    def test_exact_node_state_uses_derived_registry_without_changing_condition(self) -> None:
        registry = DAILY_DERIVED_ACTION_REGISTRY
        state = ExactNodeGrammarState.source(
            target_node_count=2,
            search_space=_source(registry).search_space,
            action_registry=registry,
        )
        self.assertEqual(state.get_legal_token_mask(()).shape, (152,))
        state = state.step(
            DAGAction((), registry.get_action_id("ts_mean", 5), registry)
        )
        state = state.step(
            DAGAction((0,), registry.get_action_id("ret_cc1"), registry)
        )
        self.assertTrue(state.done)
        self.assertEqual(state.node_count, 2)

    def test_cross_registry_nodes_and_actions_fail_closed(self) -> None:
        raw_leaf = ExpressionNode(
            RAW_ACTION_REGISTRY.get_action_id("close"),
            action_registry=RAW_ACTION_REGISTRY,
        )
        with self.assertRaisesRegex(ValueError, "跨 ActionRegistry"):
            ExpressionNode(
                DAILY_DERIVED_ACTION_REGISTRY.get_action_id("ts_mean", 5),
                (raw_leaf,),
                DAILY_DERIVED_ACTION_REGISTRY,
            )
        with self.assertRaisesRegex(ValueError, "跨 ActionRegistry"):
            _source(RAW_ACTION_REGISTRY).step(
                DAGAction(
                    (),
                    DAILY_DERIVED_ACTION_REGISTRY.get_action_id("ret_cc1"),
                    DAILY_DERIVED_ACTION_REGISTRY,
                )
            )


class RegistryBoundPolicyTests(unittest.TestCase):
    def _model_config(self) -> ModelConfig:
        return ModelConfig(
            d_model=32,
            num_heads=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            token_policy_mode="grammar_hierarchical",
        )

    def test_state_adapter_and_model_smoke_for_both_registries(self) -> None:
        for registry in (RAW_ACTION_REGISTRY, DAILY_DERIVED_ACTION_REGISTRY):
            with self.subTest(feature_space=registry.feature_space.feature_space_id):
                state = _source(registry)
                adapter = StateAdapter(state.search_space, registry)
                batch = adapter.batch((state,))
                self.assertEqual(adapter.hole_token_id, registry.action_count)
                self.assertEqual(adapter.pad_token_id, registry.action_count + 1)
                self.assertEqual(adapter.model_token_count, registry.action_count + 2)
                self.assertEqual(
                    batch.legal_token_mask.shape[-1], registry.action_count
                )
                self.assertLess(int(batch.token_ids.max()), adapter.model_token_count)

                model = ForwardPolicyNetwork(
                    self._model_config(),
                    state.search_space,
                    registry,
                )
                output = model(batch)
                legal = batch.legal_token_mask
                self.assertTrue(torch.isfinite(output.token_log_probs[legal]).all())
                self.assertTrue(torch.isneginf(output.token_log_probs[~legal]).all())
                totals = torch.where(
                    legal,
                    output.token_log_probs.exp(),
                    torch.zeros_like(output.token_log_probs),
                ).sum(dim=-1)
                torch.testing.assert_close(totals[batch.slot_mask], torch.ones(1))

    def test_adapter_and_model_reject_cross_registry_batch(self) -> None:
        raw_state = _source(RAW_ACTION_REGISTRY)
        raw_batch = StateAdapter(raw_state.search_space, RAW_ACTION_REGISTRY).batch(
            (raw_state,)
        )
        with self.assertRaisesRegex(ValueError, "跨 ActionRegistry"):
            StateAdapter(
                raw_state.search_space,
                DAILY_DERIVED_ACTION_REGISTRY,
            ).batch((raw_state,))
        with self.assertRaisesRegex(ValueError, "跨 ActionRegistry"):
            ForwardPolicyNetwork(
                self._model_config(),
                raw_state.search_space,
                DAILY_DERIVED_ACTION_REGISTRY,
            )(raw_batch)

    def test_hybrid_config_identity_preserves_raw_and_isolates_derived(self) -> None:
        raw = build_stage5_hybrid_variance_5_15_config(max_cycles=1)
        derived = build_stage5_hybrid_variance_5_15_config(
            max_cycles=1,
            feature_space=DAILY_DERIVED_V1_FEATURE_SPACE,
        )
        self.assertEqual(raw.fingerprint(), RAW_HYBRID_CONFIG_FINGERPRINT)
        self.assertNotIn("feature_space_fingerprint", raw.manifest())
        self.assertEqual(
            derived.manifest()["feature_space_fingerprint"],
            DAILY_DERIVED_V1_FEATURE_SPACE.fingerprint(),
        )
        self.assertEqual(
            derived.manifest()["token_space_fingerprint"],
            DERIVED_ACTION_FINGERPRINT,
        )
        self.assertNotEqual(raw.fingerprint(), derived.fingerprint())

    def test_raw_state_hash_keeps_historical_payload_and_values(self) -> None:
        from factor_gfn.grammar import SearchSpaceConfig

        search_space = SearchSpaceConfig(max_depth=0, max_nodes=1)
        source = GrammarState(
            search_space=search_space,
            action_registry=RAW_ACTION_REGISTRY,
        )
        child = source.step(
            DAGAction(
                (),
                RAW_ACTION_REGISTRY.get_action_id("close"),
                RAW_ACTION_REGISTRY,
            )
        )
        self.assertEqual(state_hash(source), RAW_SOURCE_STATE_HASH)
        self.assertEqual(state_hash(child), RAW_CLOSE_STATE_HASH)

    def test_derived_sampler_uses_high_token_and_replays_with_derived_registry(self) -> None:
        registry = DAILY_DERIVED_ACTION_REGISTRY
        search_space = _source(registry).search_space
        adapter = StateAdapter(search_space, registry)
        trajectory = sample_trajectory(
            _DerivedHighTokenPolicy(),
            adapter,
            sampling_config=SamplingConfig(greedy=True),
        )
        self.assertGreaterEqual(trajectory.steps[0].selected_token_id, 142)
        self.assertEqual(trajectory.terminal_expression.action_registry, registry)
        self.assertEqual(
            trajectory.terminal_expression.to_formula(),
            "ts_beta(ret_cc1, ret_cc1, 5)",
        )
        source = GrammarState(search_space=search_space, action_registry=registry)
        trajectory.replay(source)
        with self.assertRaisesRegex(ValueError, "跨 ActionRegistry"):
            trajectory.replay(GrammarState(search_space=search_space))

    def test_derived_hybrid_trainer_assembles_and_collects_without_exact_tb(self) -> None:
        config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=1,
            trajectories_per_batch=2,
            feature_space=DAILY_DERIVED_V1_FEATURE_SPACE,
        )
        trainer = HybridVarianceTrainer(config, _AlwaysValidProvider())
        self.assertEqual(trainer.action_registry, DAILY_DERIVED_ACTION_REGISTRY)
        self.assertEqual(trainer.adapter.action_registry, trainer.action_registry)
        self.assertEqual(trainer.model.action_registry, trainer.action_registry)
        collection = trainer.collect_single_condition_batch(
            condition_N=3,
            trajectories_per_batch=2,
        )
        self.assertTrue(collection.complete)
        self.assertEqual(trainer.optimizer_step, 0)
        for trajectory in collection.trajectories:
            self.assertEqual(
                trajectory.terminal_expression.action_registry,
                DAILY_DERIVED_ACTION_REGISTRY,
            )
        with self.assertRaisesRegex(TypeError, "ExhaustiveRegistry"):
            trainer.configure_hybrid_exhaustive_registry(
                object(), source_semantics_by_N={}
            )

    def test_hybrid_trainer_rejects_feature_space_context_mismatch(self) -> None:
        config = build_stage5_hybrid_variance_5_15_config(
            max_cycles=1,
            trajectories_per_batch=2,
            feature_space=DAILY_DERIVED_V1_FEATURE_SPACE,
        )
        with self.assertRaisesRegex(ValueError, "Feature Space"):
            HybridVarianceTrainer(config, _AlwaysValidProvider("raw_daily"))


if __name__ == "__main__":
    unittest.main()
