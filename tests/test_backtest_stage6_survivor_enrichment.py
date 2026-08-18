import json
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import tempfile
import unittest

import numpy as np

from factor_gfn.backtest.stage6_evaluation import _stable_hash
from factor_gfn.backtest.stage6_evaluation_store import VerifiedEvaluationRun
from factor_gfn.backtest.stage6_survivor_enrichment import (
    Stage6LongExcessEnrichmentResult,
    Stage6TrainLongExcessEnricher,
    run_stage6_survivor_enrichment_selection,
    select_stage6_9c_engineering_smoke_candidates,
)


def _hash(index: int) -> str:
    return f"{index:064x}"


def _dates() -> list[str]:
    return [f"2018-{index // 28 + 1:02d}-{index % 28 + 1:02d}" for index in range(60)]


def _orthogonal() -> tuple[list[float], list[float]]:
    left = np.arange(60, dtype=np.float64)
    left -= left.mean()
    right = np.sin(np.arange(60, dtype=np.float64) * 0.73)
    right -= right.mean()
    right -= left * float(np.dot(left, right) / np.dot(left, left))
    return left.tolist(), right.tolist()


def _stored(
    index: int,
    *,
    train_ic: float = 0.03,
    validation_ic: float = 0.02,
    train_ir: float = 0.4,
    validation_ir: float = 0.4,
    train_values=None,
    validation_values=None,
    origin="stage5_verified_reuse",
) -> dict:
    structural_hash = _hash(index)
    result = {
        "status": "completed",
        "invalid_reasons": [],
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
        "train_direction": 1,
        "train": {
            "metric_origin": origin,
            "ic": {"mean": train_ic},
            "long": {
                "annualized_ir": train_ir,
                "valid_periods": 60,
                "excess_series": {
                    "dates": _dates() if train_values is not None else None,
                    "values": train_values,
                },
            },
            "barra": {"max_abs_correlation": 0.2},
        },
        "validation": {
            "metric_origin": "stage6_fresh_evaluation",
            "ic": {"mean": validation_ic},
            "long": {
                "annualized_ir": validation_ir,
                "excess_series": {
                    "dates": _dates(),
                    "values": validation_values or list(range(60)),
                },
            },
            "barra": {"max_abs_correlation": 0.2},
        },
    }
    return {
        "ordinal": index,
        "structural_hash": structural_hash,
        "cache_key": f"cache-{index}",
        "result_fingerprint": f"result-{index}",
        "result": result,
    }


def _candidate(index: int) -> dict:
    return {
        "current_structural_hash": _hash(index),
        "formula": f"candidate_{index}",
        "prefix_token_ids": [0],
        "node_count": 1,
        "depth": 0,
    }


class StubStore:
    def __init__(self, records):
        self.verified = VerifiedEvaluationRun(
            run_id="mixed-run",
            manifest={
                "context_fingerprint": "a" * 64,
                "evaluation_contract_fingerprint": "b" * 64,
                "oos": "not_loaded",
            },
            records=tuple(records),
            ordered_result_set_fingerprint="c" * 64,
        )

    def load_verified_run_results(self, run_id):
        if run_id != "mixed-run":
            raise KeyError(run_id)
        return self.verified


class FakeEnricher:
    def __init__(self, values, *, invalid=False):
        self.context = SimpleNamespace(fingerprint="a" * 64)
        self.contract_fingerprint = "d" * 64
        self.values = values
        self.invalid = invalid
        self.calls = []
        self.timing = 0.1

    def evaluate(
        self,
        candidate,
        *,
        preserved_train_ic,
        preserved_train_direction,
        preserved_train_long_valid_periods,
    ):
        self.calls.append(candidate["current_structural_hash"])
        structural_hash = candidate["current_structural_hash"]
        status = "enrichment_invalid" if self.invalid else "completed"
        reason = "synthetic_failure" if self.invalid else None
        long_excess = None if self.invalid else {
            "dates": _dates(),
            "values": list(self.values),
            "direction": preserved_train_direction,
            "valid_periods": 60,
            "total_periods": 60,
            "origin": "stage6_fresh_long_excess",
        }
        deterministic = {
            "schema": "factor_gfn.stage6_train_long_excess_enrichment.v1",
            "structural_hash": structural_hash,
            "status": status,
            "failure_reason": reason,
            "expected_direction": preserved_train_direction,
            "derived_direction": preserved_train_direction,
            "direction_match": not self.invalid,
            "context_fingerprint": "a" * 64,
            "enrichment_contract_fingerprint": "d" * 64,
            "long_excess": long_excess,
            "diagnostics": {},
        }
        self.timing += 0.1
        return Stage6LongExcessEnrichmentResult(
            structural_hash=structural_hash,
            status=status,
            failure_reason=reason,
            expected_direction=preserved_train_direction,
            derived_direction=preserved_train_direction,
            direction_match=not self.invalid,
            context_fingerprint="a" * 64,
            enrichment_contract_fingerprint="d" * 64,
            long_excess=MappingProxyType(long_excess) if long_excess else None,
            diagnostics=MappingProxyType({}),
            factor_seconds=self.timing,
            train_long_excess_seconds=self.timing,
            total_seconds=self.timing,
            result_fingerprint=_stable_hash(deterministic),
        )


class Stage6SurvivorEnrichmentTests(unittest.TestCase):
    def test_engineering_sample_uses_only_train_gate_and_hash_order(self):
        candidates = [_candidate(index) for index in range(30)]
        records = {}
        for index in range(30):
            records[_hash(index)] = {
                "train_metrics": {
                    "train_ic": 0.02 if index != 1 else 0.001,
                    "train_long_ir": 0.3,
                    "train_barra_ts_corr": 0.2,
                },
                "validation_ic": 999.0 - index,
            }
        overlay = SimpleNamespace(records=records)
        selected = select_stage6_9c_engineering_smoke_candidates(
            list(reversed(candidates)), overlay, count=24
        )
        self.assertEqual(
            [row["current_structural_hash"] for row in selected],
            [_hash(index) for index in range(30) if index != 1][:24],
        )

    def test_direction_mismatch_fails_before_interpretation(self):
        class Fresh:
            evaluation_contract_fingerprint = "b" * 64
            context = SimpleNamespace(fingerprint="a" * 64)

            def _expression(self, candidate):
                return object(), {"structural_hash": candidate["current_structural_hash"]}

        enricher = Stage6TrainLongExcessEnricher(Fresh())
        result = enricher.evaluate(
            _candidate(1),
            preserved_train_ic=-0.02,
            preserved_train_direction=1,
            preserved_train_long_valid_periods=60,
        )
        self.assertEqual(result.status, "enrichment_invalid")
        self.assertEqual(result.failure_reason, "preserved_train_direction_mismatch")

    def test_only_missing_survivor_is_enriched_and_metrics_are_preserved(self):
        left, right = _orthogonal()
        missing = _stored(1, train_ic=0.04, train_values=None)
        existing = _stored(
            2,
            train_ic=0.03,
            train_values=right,
            origin="stage6_fresh_evaluation",
        )
        failed = _stored(3, train_ic=0.001, train_values=None)
        enricher = FakeEnricher(left)
        progress_events = []
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = run_stage6_survivor_enrichment_selection(
                store=StubStore([failed, existing, missing]),
                evaluation_run_id="mixed-run",
                accepted_candidates=[_candidate(1), _candidate(2), _candidate(3)],
                enricher=enricher,
                output_root=Path(directory),
                provisional_universe={
                    "fingerprint": "u" * 64,
                    "original_accepted_candidate_count": 5,
                    "evaluation_eligible_count": 3,
                    "deferred_candidate_count": 2,
                    "deferred_reason_counts": {
                        "historical_train_contract_not_equivalent": 2
                    },
                    "pool_basis": "resource_limited_evaluation_eligible_universe_only",
                    "final_stage_may_reevaluate_deferred": True,
                },
                progress_callback=progress_events.append,
            )
            manifest = json.loads(manifest_path.read_text())
            hard_rows = [
                json.loads(line)
                for line in (manifest_path.parent / "hard_filter_results.jsonl")
                .read_text()
                .splitlines()
            ]
            first = manifest_path
            second = run_stage6_survivor_enrichment_selection(
                store=StubStore([failed, existing, missing]),
                evaluation_run_id="mixed-run",
                accepted_candidates=[_candidate(1), _candidate(2), _candidate(3)],
                enricher=enricher,
                output_root=Path(directory),
                provisional_universe={
                    "fingerprint": "u" * 64,
                    "original_accepted_candidate_count": 5,
                    "evaluation_eligible_count": 3,
                    "deferred_candidate_count": 2,
                    "deferred_reason_counts": {
                        "historical_train_contract_not_equivalent": 2
                    },
                    "pool_basis": "resource_limited_evaluation_eligible_universe_only",
                    "final_stage_may_reevaluate_deferred": True,
                },
            )
        self.assertEqual(first, second)
        self.assertEqual(progress_events[0]["event_type"], "selection_started")
        self.assertEqual(progress_events[-1]["event_type"], "selection_completed")
        self.assertIn(
            "hard_filter_complete",
            [event["event_type"] for event in progress_events],
        )
        self.assertIn(
            "greedy_complete",
            [event["event_type"] for event in progress_events],
        )
        self.assertEqual(enricher.calls, [_hash(1), _hash(1)])
        self.assertEqual(manifest["counts"]["hard_filter_pass"], 2)
        self.assertEqual(manifest["counts"]["survivors_enriched"], 1)
        self.assertEqual(
            manifest["counts"]["survivors_already_have_train_long_excess"], 1
        )
        self.assertEqual(manifest["counts"]["retained"], 2)
        self.assertEqual(manifest["scope"], "resource_limited_provisional_selection")
        self.assertEqual(manifest["counts"]["original_accepted_candidate_count"], 5)
        self.assertEqual(manifest["counts"]["evaluation_eligible_count"], 3)
        self.assertEqual(manifest["counts"]["deferred_candidate_count"], 2)
        self.assertEqual(
            {row["structural_hash"]: row["metrics"]["train_ic"] for row in hard_rows},
            {_hash(1): 0.04, _hash(2): 0.03, _hash(3): 0.001},
        )

    def test_hybrid_available_long_excess_skips_enricher_and_keeps_origin(self):
        existing = _stored(1, train_values=list(range(60)))
        series = existing["result"]["train"]["long"]["excess_series"]
        series.update(
            {
                "availability": "available_hybrid_artifact_reuse",
                "origin": "stage5_hybrid_train_artifact_reuse",
                "train_evaluation_contract_fingerprint": "e" * 64,
                "overlay_record_fingerprint": "f" * 64,
            }
        )
        enricher = FakeEnricher(list(range(60)))
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = run_stage6_survivor_enrichment_selection(
                store=StubStore([existing]),
                evaluation_run_id="mixed-run",
                accepted_candidates=[_candidate(1)],
                enricher=enricher,
                output_root=Path(directory),
                engineering_smoke=True,
            )
            enrichment = json.loads(
                (manifest_path.parent / "survivor_long_excess_enrichment.jsonl")
                .read_text()
                .strip()
            )
            manifest = json.loads(manifest_path.read_text())
        self.assertEqual(enricher.calls, [])
        self.assertEqual(enrichment["status"], "already_available")
        self.assertEqual(
            enrichment["origin"], "stage5_hybrid_train_artifact_reuse"
        )
        self.assertEqual(
            enrichment["source_provenance"][
                "train_evaluation_contract_fingerprint"
            ],
            "e" * 64,
        )
        self.assertEqual(
            manifest["counts"]["survivors_already_have_train_long_excess"], 1
        )
        self.assertEqual(manifest["counts"]["survivors_enriched"], 0)

    def test_enrichment_failure_is_decorrelation_invalid_even_at_first_rank(self):
        missing = _stored(1, train_ic=0.04, train_values=None)
        enricher = FakeEnricher(list(range(60)), invalid=True)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = run_stage6_survivor_enrichment_selection(
                store=StubStore([missing]),
                evaluation_run_id="mixed-run",
                accepted_candidates=[_candidate(1)],
                enricher=enricher,
                output_root=Path(directory),
                engineering_smoke=True,
            )
            row = json.loads(
                (manifest_path.parent / "greedy_decorrelation_results.jsonl")
                .read_text()
                .strip()
            )
            manifest = json.loads(manifest_path.read_text())
        self.assertEqual(row["decorrelation_status"], "decorrelation_invalid")
        self.assertEqual(row["decorrelation_failure_reason"], "enrichment:synthetic_failure")
        self.assertEqual(manifest["counts"]["retained"], 0)

    def test_existing_enriched_artifact_tamper_fails_closed(self):
        missing = _stored(1, train_ic=0.04, train_values=None)
        enricher = FakeEnricher(list(range(60)))
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = run_stage6_survivor_enrichment_selection(
                store=StubStore([missing]),
                evaluation_run_id="mixed-run",
                accepted_candidates=[_candidate(1)],
                enricher=enricher,
                output_root=Path(directory),
                engineering_smoke=True,
            )
            (manifest_path.parent / "alpha_pool.jsonl").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                run_stage6_survivor_enrichment_selection(
                    store=StubStore([missing]),
                    evaluation_run_id="mixed-run",
                    accepted_candidates=[_candidate(1)],
                    enricher=enricher,
                    output_root=Path(directory),
                    engineering_smoke=True,
                )


if __name__ == "__main__":
    unittest.main()
