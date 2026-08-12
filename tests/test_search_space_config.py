import unittest

from factor_gfn.gfn import GFNConfig, ModelConfig, build_stage5_real_training_config
from factor_gfn.grammar import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NODES,
    DAGAction,
    GrammarState,
    SearchSpaceConfig,
    action_space_fingerprint,
    get_action_id,
    state_space_fingerprint,
    transition_space_fingerprint,
)


class UnifiedSearchSpaceConfigTests(unittest.TestCase):
    def test_stage_two_and_stage_four_accept_equivalent_config_values(self):
        stage_two_config = SearchSpaceConfig(max_depth=4, max_nodes=9)
        stage_four_config = SearchSpaceConfig(max_depth=4, max_nodes=9)
        self.assertIsNot(stage_two_config, stage_four_config)
        state = GrammarState(search_space=stage_two_config)
        gfn = GFNConfig(search_space=stage_four_config)
        self.assertEqual(state.search_space, gfn.search_space)
        self.assertEqual(
            state.search_space.fingerprint(), gfn.search_space.fingerprint()
        )

    def test_child_and_parent_states_preserve_config_value(self):
        search_space = SearchSpaceConfig(max_depth=3, max_nodes=7)
        source = GrammarState(search_space=search_space)
        child = source.step(DAGAction((), get_action_id("add")))
        self.assertEqual(child.search_space, search_space)
        for parent_transition in child.enumerate_parents():
            self.assertEqual(parent_transition.parent.search_space, search_space)

    def test_legacy_limits_are_converted_to_unified_config(self):
        state = GrammarState(max_depth=2, max_nodes=5)
        self.assertEqual(state.search_space, SearchSpaceConfig(max_depth=2, max_nodes=5))
        with self.assertRaisesRegex(ValueError, "不能与"):
            GrammarState(
                search_space=SearchSpaceConfig(max_depth=2, max_nodes=5),
                max_depth=2,
            )

    def test_defaults_and_fingerprint_are_stable(self):
        default = SearchSpaceConfig()
        self.assertEqual(default.max_depth, DEFAULT_MAX_DEPTH)
        self.assertEqual(default.max_nodes, DEFAULT_MAX_NODES)
        self.assertEqual(len(default.fingerprint()), 64)
        self.assertNotEqual(
            default.fingerprint(),
            SearchSpaceConfig(max_depth=DEFAULT_MAX_DEPTH - 1).fingerprint(),
        )

    def test_gfn_manifest_contains_runtime_and_grammar_fingerprints(self):
        manifest = GFNConfig().manifest()
        self.assertEqual(manifest["config"]["search_space"], {"max_depth": 10, "max_nodes": 30})
        self.assertEqual(manifest["token_space_fingerprint"], action_space_fingerprint())
        self.assertEqual(manifest["state_space_fingerprint"], state_space_fingerprint())
        self.assertEqual(
            manifest["transition_space_fingerprint"], transition_space_fingerprint()
        )

    def test_stage5_real_training_preset_is_frozen_and_fingerprinted(self):
        config = build_stage5_real_training_config(max_steps=250, seed=7)
        self.assertEqual(config.search_space, SearchSpaceConfig(max_depth=6, max_nodes=15))
        self.assertEqual(config.model.d_model, 128)
        self.assertEqual(config.model.num_heads, 4)
        self.assertEqual(config.model.num_layers, 4)
        self.assertEqual(config.model.dim_feedforward, 512)
        self.assertEqual(config.model.dropout, 0.0)
        self.assertEqual(config.model.token_policy_mode, "grammar_hierarchical")
        self.assertEqual(config.training.batch_size, 8)
        self.assertEqual(config.training.log_z_learning_rate, 1e-2)
        self.assertEqual(config.training.initial_log_z, 39.0)
        self.assertEqual(config.training.model_gradient_clip_norm, 5.0)
        self.assertEqual(config.training.log_z_gradient_clip_norm, 5.0)
        self.assertEqual(config.training.max_steps, 250)
        self.assertEqual(config.training.seed, 7)
        self.assertTrue(config.reward.candidate_industry_neutralization)
        self.assertEqual(config.sampling.temperature, 1.0)
        self.assertFalse(config.sampling.greedy)
        self.assertEqual(len(config.fingerprint()), 64)
        self.assertEqual(
            config.fingerprint(),
            build_stage5_real_training_config(max_steps=250, seed=7).fingerprint(),
        )
        self.assertNotEqual(
            config.fingerprint(),
            build_stage5_real_training_config(max_steps=251, seed=7).fingerprint(),
        )
        flat_model = ModelConfig(
            d_model=128,
            num_heads=4,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.0,
            token_policy_mode="flat",
        )
        self.assertNotEqual(
            config.fingerprint(),
            GFNConfig(
                search_space=config.search_space,
                model=flat_model,
                sampling=config.sampling,
                reward=config.reward,
                training=config.training,
            ).fingerprint(),
        )
        arity_model = ModelConfig(
            d_model=128,
            num_heads=4,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.0,
            token_policy_mode="arity_hierarchical",
        )
        self.assertNotEqual(
            config.fingerprint(),
            GFNConfig(
                search_space=config.search_space,
                model=arity_model,
                sampling=config.sampling,
                reward=config.reward,
                training=config.training,
            ).fingerprint(),
        )


if __name__ == "__main__":
    unittest.main()
