from pathlib import Path
import json
import math
import tempfile
import unittest

import numpy as np

from factor_gfn.backtest.stage6_evaluation_store import VerifiedEvaluationRun
from factor_gfn.backtest.stage6_selection import (
    HARD_CONDITION_CODES,
    Stage6SelectionConfig,
    Stage6SelectionIntegrityError,
    run_stage6_selection,
)


def _hash(index: int) -> str:
    return f"{index:064x}"


def _dates(count: int = 60) -> list[str]:
    return [f"2018-{index // 28 + 1:02d}-{index % 28 + 1:02d}" for index in range(count)]


def _orthogonal_series(count: int = 60) -> tuple[list[float], list[float]]:
    left = np.arange(count, dtype=np.float64)
    left -= left.mean()
    right = np.sin(np.arange(count, dtype=np.float64) * 0.73)
    right -= right.mean()
    right -= left * float(np.dot(left, right) / np.dot(left, left))
    left /= math.sqrt(float(np.dot(left, left)))
    right /= math.sqrt(float(np.dot(right, right)))
    return left.tolist(), right.tolist()


def _record(
    index: int,
    *,
    train_ic: float = 0.02,
    validation_ic: float = 0.02,
    train_ir: float = 0.3,
    validation_ir: float = 0.3,
    barra: float = 0.2,
    train_values: list[float | None] | None = None,
    validation_values: list[float | None] | None = None,
    status: str = "completed",
) -> dict:
    structural_hash = _hash(index)
    dates = _dates()
    train_values = train_values if train_values is not None else list(range(60))
    validation_values = (
        validation_values if validation_values is not None else list(range(60))
    )
    result = {
        "status": status,
        "invalid_reasons": [] if status == "completed" else ["synthetic_invalid"],
        "expression": {
            "structural_hash": structural_hash,
            "formula": f"candidate_{index}",
            "prefix_token_ids": [0],
            "node_count": 1,
            "depth": 0,
        },
        "source_identity": {"origin_ids": [f"origin-{index}"]},
        "context_fingerprint": "a" * 64,
        "evaluation_contract_fingerprint": "b" * 64,
        "train_direction": 1 if status == "completed" else None,
        "train": {
            "ic": {"mean": train_ic},
            "long": {
                "annualized_ir": train_ir,
                "excess_series": {"dates": dates, "values": train_values},
            },
            "barra": {"max_abs_correlation": barra},
        },
        "validation": {
            "ic": {"mean": validation_ic},
            "long": {
                "annualized_ir": validation_ir,
                "excess_series": {"dates": dates, "values": validation_values},
            },
            "barra": {"max_abs_correlation": barra},
        },
    }
    return {
        "ordinal": index,
        "structural_hash": structural_hash,
        "cache_key": f"cache-{index}",
        "result_fingerprint": f"result-{index}",
        "result": result,
    }


class StubStore:
    def __init__(self, records, *, include_oos=False):
        manifest = {
            "context_fingerprint": "a" * 64,
            "evaluation_contract_fingerprint": "b" * 64,
            "oos": "not_loaded",
        }
        self.verified = VerifiedEvaluationRun(
            run_id="evaluation-run",
            manifest=manifest,
            records=tuple(records),
            ordered_result_set_fingerprint="c" * 64,
        )
        if include_oos:
            self.verified.records[0]["result"]["oos"] = {}

    def load_verified_run_results(self, run_id):
        if run_id != self.verified.run_id:
            raise KeyError(run_id)
        return self.verified


class Stage6SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temporary.name) / "selection"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, records):
        return run_stage6_selection(
            StubStore(records), "evaluation-run", self.output_root
        )

    def test_all_six_strict_boundary_failures_are_recorded_in_fixed_order(self):
        record = _record(
            1,
            train_ic=0.01,
            validation_ic=-0.01,
            train_ir=0.25,
            validation_ir=0.25,
            barra=0.7,
        )
        result = self._run([record])
        row = json.loads(
            (result.output_directory / "hard_filter_results.jsonl").read_text()
        )
        self.assertFalse(row["hard_filter_pass"])
        self.assertEqual(row["failed_conditions"], list(HARD_CONDITION_CODES))
        self.assertFalse(row["evaluation_ineligible"])

    def test_completed_invalid_is_ineligible_not_a_seventh_hard_condition(self):
        record = _record(1, status="completed_invalid")
        result = self._run([record])
        row = json.loads(
            (result.output_directory / "hard_filter_results.jsonl").read_text()
        )
        self.assertTrue(row["evaluation_ineligible"])
        self.assertFalse(row["hard_filter_pass"])
        self.assertEqual(row["failed_conditions"], [])
        self.assertNotIn("evaluation_ineligible", row["condition_results"])
        self.assertEqual(result.evaluation_ineligible_count, 1)

    def test_train_prefilter_failure_does_not_invent_validation_failures(self):
        record = _record(1, train_ir=0.25, status="completed_invalid")
        record["result"]["invalid_reasons"] = [
            "train_prefilter_failed",
            HARD_CONDITION_CODES[3],
        ]
        record["result"]["train"]["train_prefilter"] = {
            "status": "train_prefilter_failed",
            "failed_conditions": [HARD_CONDITION_CODES[3]],
        }
        record["result"]["validation"]["ic"]["mean"] = None
        record["result"]["validation"]["long"]["annualized_ir"] = None
        result = self._run([record])
        row = json.loads(
            (result.output_directory / "hard_filter_results.jsonl").read_text()
        )
        self.assertEqual(row["failed_conditions"], [HARD_CONDITION_CODES[3]])
        self.assertIsNone(row["condition_results"][HARD_CONDITION_CODES[1]])
        self.assertIsNone(row["condition_results"][HARD_CONDITION_CODES[2]])
        self.assertIsNone(row["condition_results"][HARD_CONDITION_CODES[4]])
        self.assertFalse(row["validation_evaluated"])
        self.assertEqual(row["train_prefilter_status"], "train_prefilter_failed")

    def test_sort_uses_abs_train_ic_then_structural_hash(self):
        left, right = _orthogonal_series()
        first = _record(
            2, train_ic=-0.03, validation_ic=-0.02, train_values=right
        )
        second = _record(1, train_ic=0.03, train_values=left)
        result = self._run([first, second])
        rows = [
            json.loads(line)
            for line in (
                result.output_directory / "greedy_decorrelation_results.jsonl"
            ).read_text().splitlines()
        ]
        self.assertEqual([row["structural_hash"] for row in rows], [_hash(1), _hash(2)])
        self.assertEqual(result.retained_count, 2)

    def test_rejection_and_invalid_are_distinct_and_neither_is_retained(self):
        left, _ = _orthogonal_series()
        retained = _record(1, train_ic=0.04, train_values=left)
        blocked = _record(2, train_ic=0.03, train_values=left)
        insufficient = _record(
            3,
            train_ic=0.02,
            train_values=[*left[:-1], None],
        )
        result = self._run([insufficient, blocked, retained])
        rows = [
            json.loads(line)
            for line in (
                result.output_directory / "greedy_decorrelation_results.jsonl"
            ).read_text().splitlines()
        ]
        by_hash = {row["structural_hash"]: row for row in rows}
        self.assertEqual(
            by_hash[_hash(2)]["decorrelation_status"], "rejected_by_correlation"
        )
        self.assertEqual(by_hash[_hash(2)]["blocked_by_structural_hash"], _hash(1))
        self.assertGreaterEqual(abs(by_hash[_hash(2)]["blocking_corr"]), 0.7)
        self.assertEqual(
            by_hash[_hash(3)]["decorrelation_status"], "decorrelation_invalid"
        )
        self.assertEqual(by_hash[_hash(3)]["common_valid_periods"], 59)
        self.assertEqual(
            by_hash[_hash(3)]["decorrelation_failure_reason"],
            "common_valid_periods_below_minimum",
        )
        self.assertEqual(result.retained_count, 1)
        self.assertEqual(result.rejected_by_correlation_count, 1)
        self.assertEqual(result.decorrelation_invalid_count, 1)

    def test_validation_correlation_is_diagnostic_only(self):
        left, right = _orthogonal_series()
        first = _record(1, train_ic=0.04, train_values=left, validation_values=left)
        second = _record(2, train_ic=0.03, train_values=right, validation_values=left)
        result = self._run([first, second])
        rows = [
            json.loads(line)
            for line in (
                result.output_directory / "greedy_decorrelation_results.jsonl"
            ).read_text().splitlines()
        ]
        self.assertTrue(rows[1]["greedy_retained"])
        diagnostic = rows[1]["comparison_trace"][0]["validation"]
        self.assertAlmostEqual(diagnostic["correlation"], 1.0)
        self.assertEqual(result.retained_count, 2)

    def test_unavailable_training_correlation_is_decorrelation_invalid(self):
        left, _ = _orthogonal_series()
        first = _record(1, train_ic=0.04, train_values=left)
        constant = _record(2, train_ic=0.03, train_values=[1.0] * 60)
        result = self._run([first, constant])
        rows = [
            json.loads(line)
            for line in (
                result.output_directory / "greedy_decorrelation_results.jsonl"
            ).read_text().splitlines()
        ]
        self.assertEqual(rows[1]["decorrelation_status"], "decorrelation_invalid")
        self.assertEqual(rows[1]["common_valid_periods"], 60)
        self.assertEqual(
            rows[1]["decorrelation_failure_reason"],
            "zero_variance_or_nonfinite_denominator",
        )
        self.assertFalse(rows[1]["greedy_retained"])

    def test_long_excess_series_are_aligned_by_date_not_array_position(self):
        left, _ = _orthogonal_series()
        first = _record(1, train_ic=0.04, train_values=left)
        second = _record(2, train_ic=0.03, train_values=list(reversed(left)))
        second_dates = second["result"]["train"]["long"]["excess_series"]["dates"]
        second["result"]["train"]["long"]["excess_series"]["dates"] = list(
            reversed(second_dates)
        )
        result = self._run([first, second])
        rows = [
            json.loads(line)
            for line in (
                result.output_directory / "greedy_decorrelation_results.jsonl"
            ).read_text().splitlines()
        ]
        self.assertEqual(rows[1]["decorrelation_status"], "rejected_by_correlation")
        self.assertAlmostEqual(rows[1]["blocking_corr"], 1.0)

    def test_exact_threshold_is_rejected(self):
        left, orthogonal = _orthogonal_series()
        x = np.asarray(left)
        z = np.asarray(orthogonal)
        boundary_corr = float(np.nextafter(0.7, 1.0))
        boundary = (
            boundary_corr * x + math.sqrt(1.0 - boundary_corr**2) * z
        ).tolist()
        first = _record(1, train_ic=0.04, train_values=left)
        second = _record(2, train_ic=0.03, train_values=boundary)
        result = self._run([first, second])
        rows = [
            json.loads(line)
            for line in (
                result.output_directory / "greedy_decorrelation_results.jsonl"
            ).read_text().splitlines()
        ]
        self.assertGreaterEqual(abs(rows[1]["blocking_corr"]), 0.7)
        self.assertEqual(rows[1]["decorrelation_status"], "rejected_by_correlation")

    def test_outputs_are_repeatable_and_existing_artifacts_are_immutable(self):
        record = _record(1)
        first = self._run([record])
        second = self._run([record])
        self.assertEqual(first.selection_fingerprint, second.selection_fingerprint)
        self.assertEqual(
            first.selection_manifest_fingerprint,
            second.selection_manifest_fingerprint,
        )
        target = first.output_directory / "alpha_pool.jsonl"
        target.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(Stage6SelectionIntegrityError):
            self._run([record])

    def test_oos_payload_is_rejected(self):
        store = StubStore([_record(1)])
        store.verified.records[0]["result"]["oos"] = {}
        with self.assertRaisesRegex(Stage6SelectionIntegrityError, "OOS"):
            run_stage6_selection(store, "evaluation-run", self.output_root)

    def test_only_provisional_mode_is_permitted_in_this_batch(self):
        with self.assertRaisesRegex(ValueError, "provisional"):
            Stage6SelectionConfig(mode="final")

    def test_frozen_thresholds_cannot_be_adjusted(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            Stage6SelectionConfig(decorrelation_abs_corr_max=0.8)


if __name__ == "__main__":
    unittest.main()
