import unittest

import torch

from factor_gfn.gfn.config import TrainingStats
from factor_gfn.gfn.diagnostic_support import run_training_with_progress
from factor_gfn.gfn.training_health import (
    LogZInitializationHealthConfig,
    build_log_z_initialization_health,
)


def _trajectory_rows(node_count: int, deltas: list[float]):
    return [
        {
            "logical_batch": index,
            "target_node_count": node_count,
            "selected_log_z": 10.0,
            "tb_delta": delta,
            "successful_gradient_exposure": True,
            "structural_hash": f"N{node_count}-{index}",
        }
        for index, delta in enumerate(deltas, start=1)
    ]


def _training_rows(values_by_batch: list[float]):
    return [
        {
            "logical_batch": index,
            "learned_log_z_by_N": {2: value},
        }
        for index, value in enumerate(values_by_batch, start=1)
    ]


class LogZInitializationHealthTests(unittest.TestCase):
    def test_training_audit_captures_absolute_batch_and_pre_update_log_z(self):
        class Loss:
            log_z_by_node_count = torch.tensor([0.0, 10.0])

        class Trainer:
            step = 4
            optimizer_step = 3
            resolved_learned_node_counts = (2,)
            tb_loss = Loss()
            device = torch.device("cpu")
            last_step_timings = {}
            last_discovery_trajectory_diagnostics = []

            def train_step(self):
                pre = float(self.tb_loss.log_z_by_node_count[1])
                self.last_discovery_trajectory_diagnostics = [{
                    "source": "discovery",
                    "target_node_count": 2,
                    "terminal_node_count": 2,
                    "terminal_depth": 1,
                    "structural_hash": "candidate",
                    "reward": 1.0,
                    "log_reward": 0.0,
                    "sum_log_pf": -1.0,
                    "sum_log_pb": 0.0,
                    "selected_log_z": pre,
                    "tb_delta": pre - 1.0,
                }]
                self.tb_loss.log_z_by_node_count[1] -= 0.1
                self.step += 1
                self.optimizer_step += 1
                return TrainingStats(
                    step=self.step,
                    optimizer_step=self.optimizer_step,
                    loss=1.0,
                    tb_delta_rms=1.0,
                    skipped_update=False,
                    requested_count_by_N={2: 1},
                    retry_exhausted_count_by_N={2: 0},
                )

        rows, trajectories = run_training_with_progress(
            Trainer(),
            logical_batches=1,
            progress=lambda message: None,
        )
        self.assertEqual(rows[0]["logical_batch"], 5)
        self.assertEqual(rows[0]["optimizer_step_before"], 3)
        self.assertEqual(rows[0]["learned_log_z_pre_update_by_N"], {2: 10.0})
        self.assertAlmostEqual(rows[0]["learned_log_z_by_N"][2], 9.9, places=5)
        self.assertEqual(trajectories[0]["logical_batch"], 5)
        self.assertEqual(trajectories[0]["optimizer_step_before"], 3)
        self.assertEqual(trajectories[0]["optimizer_step_after"], 4)
        self.assertTrue(trajectories[0]["successful_gradient_exposure"])

    def test_phases_are_disjoint_and_initial_log_z_is_not_overwritten_by_current(self):
        rows = _trajectory_rows(2, [8.0, 5.0, 4.0, 3.0, 0.5, 0.4, 0.3])
        result = build_log_z_initialization_health(
            rows,
            _training_rows([9.8, 9.6, 9.4, 9.2, 9.1, 9.0, 9.0]),
            node_counts=(2,),
            learned_node_counts=(2,),
            initial_log_z_by_N={2: 10.0},
            current_log_z_by_N={2: 9.0},
        )
        health = result.per_N[2]
        self.assertEqual(
            health["tb_delta_by_phase"]["initialization_pre_update"]["mean"],
            8.0,
        )
        self.assertEqual(health["tb_delta_by_phase"]["early"]["count"], 3)
        self.assertEqual(health["tb_delta_by_phase"]["late"]["count"], 3)
        phases = [row["initialization_health_phase"] for row in result.enriched_trajectory_rows]
        self.assertEqual(phases, ["initialization_pre_update"] + ["early"] * 3 + ["late"] * 3)
        self.assertEqual(health["initial_log_z"], 10.0)
        self.assertEqual(health["current_log_z"], 9.0)
        self.assertEqual(health["net_change_log_z"], -1.0)
        # A large initial offset plus persistent one-direction correction remains
        # visible even though the late delta itself has become small.
        self.assertEqual(health["status"], "review_targeted_recalibration")

    def test_32_batches_do_not_override_per_N_evidence_insufficiency(self):
        training = [
            {"logical_batch": index, "learned_log_z_by_N": {2: 10.0}}
            for index in range(1, 33)
        ]
        result = build_log_z_initialization_health(
            _trajectory_rows(2, [1.0] * 6),
            training,
            node_counts=(2,),
            learned_node_counts=(2,),
            initial_log_z_by_N={2: 10.0},
            current_log_z_by_N={2: 10.0},
        )
        health = result.per_N[2]
        self.assertEqual(health["valid_trajectory_count"], 6)
        self.assertEqual(health["successful_gradient_exposure_count"], 6)
        self.assertEqual(health["status"], "insufficient_evidence")
        self.assertEqual(result.insufficient_evidence_node_counts, (2,))
        self.assertFalse(result.all_learned_initializations_usable)

    def test_sufficient_stable_learned_N_is_usable_and_exact_N_is_diagnostic(self):
        rows = _trajectory_rows(1, [0.2] * 7) + _trajectory_rows(
            2, [1.0, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2]
        )
        result = build_log_z_initialization_health(
            rows,
            _training_rows([10.0] * 7),
            node_counts=(1, 2),
            learned_node_counts=(2,),
            initial_log_z_by_N={1: 2.0, 2: 10.0},
            current_log_z_by_N={1: 2.0, 2: 10.0},
        )
        self.assertEqual(result.per_N[1]["status"], "fixed_exact_diagnostic")
        self.assertEqual(result.per_N[2]["status"], "usable")
        self.assertTrue(result.all_learned_initializations_usable)
        self.assertEqual(result.targeted_recalibration_node_counts, ())

    def test_skipped_batch_trajectory_does_not_count_as_gradient_exposure(self):
        rows = _trajectory_rows(2, [0.5] * 7)
        rows[-1]["successful_gradient_exposure"] = False
        result = build_log_z_initialization_health(
            rows,
            _training_rows([10.0] * 7),
            node_counts=(2,),
            learned_node_counts=(2,),
            initial_log_z_by_N={2: 10.0},
            current_log_z_by_N={2: 10.0},
        )
        self.assertEqual(result.per_N[2]["valid_trajectory_count"], 7)
        self.assertEqual(result.per_N[2]["successful_gradient_exposure_count"], 6)
        self.assertEqual(result.per_N[2]["status"], "insufficient_evidence")

    def test_thresholds_and_disjoint_window_minimums_are_configurable(self):
        config = LogZInitializationHealthConfig(
            minimum_valid_trajectories_per_N=5,
            minimum_successful_gradient_exposures_per_N=5,
            early_exposure_count=2,
            late_exposure_count=2,
            initial_abs_delta_mean_review_threshold=4.0,
            late_abs_delta_mean_review_threshold=2.0,
            log_z_net_change_review_threshold=0.5,
            directional_drift_fraction_threshold=0.75,
            minimum_log_z_step_for_direction=1e-5,
        )
        self.assertEqual(config.early_exposure_count, 2)
        with self.assertRaisesRegex(ValueError, "keep initialization"):
            LogZInitializationHealthConfig(
                minimum_successful_gradient_exposures_per_N=6,
                early_exposure_count=3,
                late_exposure_count=3,
            )


if __name__ == "__main__":
    unittest.main()
