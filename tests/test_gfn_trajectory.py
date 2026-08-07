import math
import unittest
from dataclasses import replace

import torch

from factor_gfn.gfn import Trajectory, TrajectoryStep, state_hash
from factor_gfn.grammar import (
    DAGAction,
    GrammarState,
    SearchSpaceConfig,
    get_action_id,
)


def known_leaf_trajectory() -> tuple[GrammarState, Trajectory]:
    source = GrammarState(search_space=SearchSpaceConfig(max_depth=0, max_nodes=1))
    slot = source.open_slots()[0]
    child = source.step(DAGAction(slot.path, get_action_id("close")))
    log_p_slot = torch.tensor(0.0, requires_grad=True)
    log_p_token = torch.tensor(-math.log(6.0), requires_grad=True)
    step = TrajectoryStep(
        state_hash=state_hash(source),
        selected_slot_index=0,
        selected_slot_path=slot.path,
        selected_slot_orbit_key=slot.orbit_key,
        selected_token_id=get_action_id("close"),
        log_p_slot=log_p_slot,
        log_p_token=log_p_token,
        log_pf=log_p_slot + log_p_token,
        child_state_hash=state_hash(child),
        n_parents=1,
        log_pb=0.0,
    )
    return source, Trajectory(
        steps=[step],
        terminal_state_hash=state_hash(child),
        terminal_expression=child.to_expression(),
        sampling_mode="stochastic",
    )


class TrajectoryContractTests(unittest.TestCase):
    def test_known_trajectory_validates_replays_and_keeps_gradients(self):
        source, trajectory = known_leaf_trajectory()
        trajectory.validate()
        replayed = trajectory.replay(source)
        self.assertTrue(replayed.done)
        trajectory.sum_log_pf.backward()
        self.assertIsNotNone(trajectory.steps[0].log_p_slot.grad)
        self.assertIsNotNone(trajectory.steps[0].log_p_token.grad)

    def test_replay_recomputes_true_parent_count(self):
        source, trajectory = known_leaf_trajectory()
        tampered_step = replace(
            trajectory.steps[0],
            n_parents=2,
            log_pb=-math.log(2.0),
        )
        tampered = replace(trajectory, steps=[tampered_step])
        tampered.validate()
        with self.assertRaisesRegex(ValueError, "真实父状态数"):
            tampered.replay(source)

    def test_state_hash_includes_search_space_fingerprint(self):
        first = GrammarState(search_space=SearchSpaceConfig(max_depth=1, max_nodes=3))
        second = GrammarState(search_space=SearchSpaceConfig(max_depth=2, max_nodes=3))
        self.assertEqual(first.state_key, second.state_key)
        self.assertNotEqual(state_hash(first), state_hash(second))

    def test_step_rejects_invalid_metadata_and_nonfinite_probabilities(self):
        _, trajectory = known_leaf_trajectory()
        step = trajectory.steps[0]
        invalid_steps = (
            replace(step, selected_slot_index=-1),
            replace(step, selected_slot_path=(-1,)),
            replace(step, selected_token_id=999),
            replace(step, log_p_slot=torch.tensor(float("nan"))),
            replace(step, log_p_token=torch.tensor(0.1)),
            replace(step, n_parents=0),
        )
        for invalid in invalid_steps:
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, IndexError)):
                    invalid.validate()

    def test_greedy_trajectory_is_explicitly_training_ineligible(self):
        _, trajectory = known_leaf_trajectory()
        greedy = replace(trajectory, sampling_mode="greedy")
        greedy.validate()
        self.assertFalse(greedy.training_eligible)
        with self.assertRaisesRegex(ValueError, "禁止"):
            greedy.require_training_eligible()
        trajectory.require_training_eligible()

    def test_unknown_sampling_mode_is_rejected(self):
        _, trajectory = known_leaf_trajectory()
        invalid = replace(trajectory, sampling_mode="unknown")
        with self.assertRaisesRegex(ValueError, "sampling_mode"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
