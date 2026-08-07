import unittest

from factor_gfn.gfn import GFNConfig
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


if __name__ == "__main__":
    unittest.main()
