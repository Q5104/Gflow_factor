import math
import unittest

import numpy as np

from factor_gfn.grammar import (
    DAGAction,
    ExactNodeGrammarState,
    ExactNodeReachability,
    GrammarState,
    SearchSpaceConfig,
    get_action,
    get_action_id,
    resolve_exact_node_strata,
)


def complete_with_first_legal(state: ExactNodeGrammarState) -> ExactNodeGrammarState:
    while not state.done:
        action = state.legal_transitions()[0]
        state = state.step(action)
    return state


class ExactNodeReachabilityTests(unittest.TestCase):
    def test_resolves_feasible_and_infeasible_strata_for_multiple_limits(self):
        cases = (
            (SearchSpaceConfig(max_depth=0, max_nodes=5), (1,), (2, 3, 4, 5)),
            (SearchSpaceConfig(max_depth=1, max_nodes=5), (1, 2, 3), (4, 5)),
            (SearchSpaceConfig(max_depth=2, max_nodes=9), tuple(range(1, 8)), (8, 9)),
            (SearchSpaceConfig(max_depth=3, max_nodes=6), tuple(range(1, 7)), ()),
        )
        for search_space, feasible, infeasible in cases:
            with self.subTest(search_space=search_space):
                strata = resolve_exact_node_strata(search_space)
                self.assertEqual(strata.resolved_feasible_node_counts, feasible)
                self.assertEqual(strata.resolved_infeasible_node_counts, infeasible)
                self.assertEqual(
                    sorted(feasible + infeasible),
                    list(range(1, search_space.max_nodes + 1)),
                )

    def test_all_feasible_targets_finish_exactly_without_overshoot(self):
        for search_space in (
            SearchSpaceConfig(max_depth=0, max_nodes=4),
            SearchSpaceConfig(max_depth=2, max_nodes=7),
            SearchSpaceConfig(max_depth=3, max_nodes=8),
        ):
            strata = resolve_exact_node_strata(search_space)
            for target in strata.resolved_feasible_node_counts:
                state = ExactNodeGrammarState.source(
                    target_node_count=target,
                    search_space=search_space,
                )
                while not state.done:
                    self.assertLess(state.node_count, target)
                    state = state.step(state.legal_transitions()[0])
                    self.assertLessEqual(state.node_count, target)
                    self.assertEqual(state.done, state.node_count == target)
                with self.subTest(search_space=search_space, target=target):
                    self.assertEqual(state.node_count, target)
                    self.assertEqual(state.pending_slots, 0)
                    self.assertLessEqual(state.max_depth_seen, search_space.max_depth)

    def test_every_legal_successor_has_an_exact_completion(self):
        search_space = SearchSpaceConfig(max_depth=3, max_nodes=6)
        engine = ExactNodeReachability(search_space)
        for target in resolve_exact_node_strata(
            search_space
        ).resolved_feasible_node_counts:
            source = ExactNodeGrammarState.source(
                target_node_count=target,
                search_space=search_space,
            )
            for action in source.legal_transitions():
                successor = source.step(action)
                with self.subTest(target=target, action=action):
                    self.assertTrue(
                        engine.can_complete_exactly(successor.state, target)
                    )
                    terminal = complete_with_first_legal(successor)
                    self.assertEqual(terminal.node_count, target)

    def test_exact_mask_accounts_for_canonical_slots_and_arity(self):
        search_space = SearchSpaceConfig(max_depth=3, max_nodes=7)
        source = ExactNodeGrammarState.source(
            target_node_count=3,
            search_space=search_space,
        )
        state = source.step(DAGAction((), get_action_id("add")))
        self.assertEqual(state.pending_slots, 2)
        self.assertEqual(len(state.open_slots()), 1)
        legal_arities = {
            get_action(int(token_id)).arity
            for token_id in state.legal_token_ids(state.open_slots()[0])
        }
        self.assertEqual(legal_arities, {0})

    def test_infeasible_target_is_rejected_at_source(self):
        search_space = SearchSpaceConfig(max_depth=0, max_nodes=3)
        with self.assertRaisesRegex(ValueError, "no legal completion"):
            ExactNodeGrammarState.source(
                target_node_count=2,
                search_space=search_space,
            )


class ExactNodeIdentityAndParentTests(unittest.TestCase):
    def test_structural_and_conditioned_identities_are_separate(self):
        search_space = SearchSpaceConfig(max_depth=2, max_nodes=5)
        n1 = ExactNodeGrammarState.source(
            target_node_count=1, search_space=search_space
        )
        n3 = ExactNodeGrammarState.source(
            target_node_count=3, search_space=search_space
        )
        self.assertEqual(n1.state_key, n3.state_key)
        self.assertNotEqual(n1.conditioned_key, n3.conditioned_key)
        self.assertEqual(
            n3.conditioned_key,
            (n3.state_key, 3, search_space.fingerprint()),
        )
        self.assertEqual(n3.conditioned_cache_key, n3.conditioned_key)
        other_space = ExactNodeGrammarState.source(
            target_node_count=3,
            search_space=SearchSpaceConfig(max_depth=3, max_nodes=5),
        )
        self.assertEqual(n3.state_key, other_space.state_key)
        self.assertNotEqual(n3.conditioned_cache_key, other_space.conditioned_cache_key)

        terminal = n3.step(DAGAction((), get_action_id("add")))
        terminal = terminal.step(
            DAGAction(terminal.open_slots()[0].path, get_action_id("open"))
        )
        terminal = terminal.step(
            DAGAction(terminal.open_slots()[0].path, get_action_id("close"))
        )
        expression = terminal.to_expression()
        structural = GrammarState(search_space=search_space)
        structural = structural.step(DAGAction((), get_action_id("add")))
        structural = structural.step(
            DAGAction(structural.open_slots()[0].path, get_action_id("open"))
        )
        structural = structural.step(
            DAGAction(structural.open_slots()[0].path, get_action_id("close"))
        )
        self.assertEqual(
            expression.structural_hash(),
            structural.to_expression().structural_hash(),
        )

    def test_conditioned_parent_enumeration_matches_forward_dag(self):
        allowed = {
            get_action_id("open"),
            get_action_id("close"),
            get_action_id("neg"),
            get_action_id("add"),
            get_action_id("sub"),
        }
        source = ExactNodeGrammarState.source(
            target_node_count=3,
            search_space=SearchSpaceConfig(max_depth=2, max_nodes=3),
        )
        states = {source.state_key: source}
        incoming: dict[str, set[str]] = {source.state_key: set()}
        queue = [source]
        while queue:
            parent = queue.pop()
            if parent.done:
                continue
            for action in parent.legal_transitions():
                if action.token_id not in allowed:
                    continue
                child = parent.step(action)
                incoming.setdefault(child.state_key, set()).add(parent.state_key)
                if child.state_key not in states:
                    states[child.state_key] = child
                    queue.append(child)

        for key, child in states.items():
            actual = {
                transition.parent.state_key
                for transition in child.enumerate_parents()
            }
            self.assertEqual(actual, incoming.get(key, set()), msg=key)
            for transition in child.enumerate_parents():
                self.assertEqual(
                    transition.parent.target_node_count,
                    child.target_node_count,
                )
                self.assertEqual(
                    transition.parent.step(transition.forward_action).state_key,
                    child.state_key,
                )
            if actual:
                self.assertAlmostEqual(
                    child.log_backward_probability(),
                    -math.log(len(actual)),
                )

    def test_seeded_conditioned_trajectories_never_reach_a_dead_state(self):
        rng = np.random.default_rng(20260812)
        search_space = SearchSpaceConfig(max_depth=4, max_nodes=9)
        feasible = resolve_exact_node_strata(
            search_space
        ).resolved_feasible_node_counts
        for trajectory_number in range(200):
            target = feasible[trajectory_number % len(feasible)]
            state = ExactNodeGrammarState.source(
                target_node_count=target,
                search_space=search_space,
            )
            while not state.done:
                transitions = state.legal_transitions()
                self.assertTrue(transitions)
                state = state.step(
                    transitions[int(rng.integers(len(transitions)))]
                )
            with self.subTest(trajectory=trajectory_number, target=target):
                self.assertEqual(state.node_count, target)
                self.assertEqual(state.pending_slots, 0)


if __name__ == "__main__":
    unittest.main()
