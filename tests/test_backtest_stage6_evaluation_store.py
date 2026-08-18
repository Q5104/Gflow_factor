from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import json
import tempfile
import unittest

from factor_gfn.backtest.stage6_evaluation import (
    STAGE6_EVALUATION_RESULT_SCHEMA,
    Stage6CandidateEvaluationResult,
    _stable_hash,
)
from factor_gfn.backtest.stage6_evaluation_store import (
    DeterminismConflictError,
    EvaluationStore,
    EvaluationStoreIntegrityError,
    ProcessRssSampler,
    Stage6EvaluationRunner,
    evaluation_cache_identity,
    select_stage6_smoke_candidates,
)
from factor_gfn.backtest.stage6_two_phase_pipeline import (
    STAGE6_RESOURCE_LIMITED_TRAIN_SCOPE,
    _CacheOnlyTrainPreparationEvaluator,
)
from factor_gfn.grammar import get_action_id


def _candidate(index: int, *, invalid: bool = False) -> dict:
    structural_hash = f"{index + 1:064x}"
    return {
        "current_structural_hash": structural_hash,
        "source_claimed_structural_hash": structural_hash,
        "formula": f"candidate_{index}",
        "prefix_token_ids": [get_action_id("open")],
        "node_count": 1,
        "depth": 0,
        "origin_ids": [f"origin-{index}"],
        "source_ids": ["source"],
        "compatibility_record_fingerprint": "c" * 64,
        "historical_metric_reuse": "forbidden",
        "stage6_metric_recompute_required": True,
        "test_invalid": invalid,
    }


class FakeEvaluator:
    def __init__(self, *, context="a", contract="b") -> None:
        self.context = SimpleNamespace(fingerprint=context * 64)
        self.evaluation_contract_fingerprint = contract * 64
        self.compatibility_audit_fingerprint = "c" * 64
        self.accepted_registry_fingerprint = "d" * 64
        self.calls: list[str] = []
        self.fail_hashes: set[str] = set()
        self.variant = 0

    def resolve_candidate_identity(self, candidate):
        return {
            "structural_hash": candidate["current_structural_hash"],
            "formula": candidate["formula"],
            "prefix_token_ids": candidate["prefix_token_ids"],
            "node_count": candidate["node_count"],
            "depth": candidate["depth"],
        }

    def evaluate(self, candidate):
        structural_hash = candidate["current_structural_hash"]
        self.calls.append(structural_hash)
        if structural_hash in self.fail_hashes:
            raise RuntimeError(f"synthetic failure for {structural_hash}")
        expression = dict(self.resolve_candidate_identity(candidate))
        invalid = bool(candidate.get("test_invalid"))
        status = "completed_invalid" if invalid else "completed"
        train = {"ic": {"mean": None if invalid else 0.01 + self.variant}}
        validation = {"ic": {"mean": 0.02}}
        coverage = {"train": {"rate": 1.0}, "validation": {"rate": 1.0}}
        deterministic = {
            "schema": STAGE6_EVALUATION_RESULT_SCHEMA,
            "status": status,
            "invalid_reasons": ["train_direction_unavailable"] if invalid else [],
            "expression": expression,
            "context_fingerprint": self.context.fingerprint,
            "evaluation_contract_fingerprint": self.evaluation_contract_fingerprint,
            "train_direction": None if invalid else 1,
            "train": train,
            "validation": validation,
            "factor_finite_coverage": coverage,
        }
        fingerprint = _stable_hash(deterministic)
        return Stage6CandidateEvaluationResult(
            schema=STAGE6_EVALUATION_RESULT_SCHEMA,
            status=status,
            invalid_reasons=("train_direction_unavailable",) if invalid else (),
            expression=MappingProxyType(expression),
            source_identity=MappingProxyType({"origin_ids": candidate["origin_ids"]}),
            context_fingerprint=self.context.fingerprint,
            evaluation_contract_fingerprint=self.evaluation_contract_fingerprint,
            train_direction=None if invalid else 1,
            train=MappingProxyType(train),
            validation=MappingProxyType(validation),
            factor_finite_coverage=MappingProxyType(coverage),
            factor_seconds=0.01,
            train_evaluation_seconds=0.02,
            validation_evaluation_seconds=0.01,
            total_seconds=0.04,
            result_fingerprint=fingerprint,
        )


class EvaluationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.database = root / "store" / "stage6_evaluations.sqlite"
        self.artifacts = root / "runs"
        self.store = EvaluationStore(self.database, self.artifacts)
        self.evaluator = FakeEvaluator()
        self.candidates = [_candidate(index, invalid=index == 4) for index in range(12)]

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_cache_key_binds_expression_context_contract_not_provenance(self):
        candidate = self.candidates[0]
        first = evaluation_cache_identity(self.evaluator, candidate)
        changed_source = dict(candidate)
        changed_source["origin_ids"] = ["different"]
        self.assertEqual(
            first.cache_key,
            evaluation_cache_identity(self.evaluator, changed_source).cache_key,
        )
        self.assertNotEqual(
            first.cache_key,
            evaluation_cache_identity(FakeEvaluator(context="e"), candidate).cache_key,
        )
        self.assertNotEqual(
            first.cache_key,
            evaluation_cache_identity(FakeEvaluator(contract="f"), candidate).cache_key,
        )
        changed_expression = dict(candidate)
        changed_expression["current_structural_hash"] = "9" * 64
        self.assertNotEqual(
            first.cache_key,
            evaluation_cache_identity(self.evaluator, changed_expression).cache_key,
        )

    def test_run_is_frozen_with_full_order_before_bounded_execution(self):
        frozen = self.store.create_run(self.candidates, self.evaluator)
        self.assertEqual(frozen.candidate_count, 12)
        self.assertEqual(frozen.manifest["candidate_count"], 12)
        self.assertEqual(
            frozen.manifest["ordered_candidate_hashes"],
            [candidate["current_structural_hash"] for candidate in self.candidates],
        )
        manifest_path = self.artifacts / frozen.run_id / "run_manifest.json"
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(json.loads(manifest_path.read_text())["candidate_count"], 12)

    def test_five_then_resume_uses_resume_skipped_not_cache_hit(self):
        frozen = self.store.create_run(self.candidates, self.evaluator)
        runner = Stage6EvaluationRunner(
            self.store, self.evaluator, rss_sampling_interval_seconds=0.01
        )
        first = runner.run(frozen.run_id, max_new_evaluations=5)
        self.assertEqual(first.run_status, "paused")
        self.assertEqual(first.newly_evaluated, 5)
        self.assertEqual(first.resume_skipped, 0)
        self.assertEqual(first.cache_hits, 0)
        self.assertEqual(first.pending_after, 7)
        second = runner.run(frozen.run_id)
        self.assertEqual(second.run_status, "complete")
        self.assertEqual(second.resume_skipped, 5)
        self.assertEqual(second.cache_hits, 0)
        self.assertEqual(second.newly_evaluated, 7)
        self.assertEqual(len(self.evaluator.calls), 12)
        events = self.store.connection.execute(
            "SELECT event_type,COUNT(*) FROM run_events GROUP BY event_type"
        ).fetchall()
        self.assertEqual(dict(events)["resume_skipped"], 5)

    def test_different_run_can_use_ordinary_cache_hits_without_evaluation(self):
        first = self.store.create_run(self.candidates, self.evaluator)
        runner = Stage6EvaluationRunner(self.store, self.evaluator)
        runner.run(first.run_id)
        calls = len(self.evaluator.calls)
        reordered = list(reversed(self.candidates))
        second = self.store.create_run(reordered, self.evaluator)
        self.assertNotEqual(first.run_id, second.run_id)
        summary = runner.run(second.run_id)
        self.assertEqual(summary.cache_hits, 12)
        self.assertEqual(summary.resume_skipped, 0)
        self.assertEqual(summary.newly_evaluated, 0)
        self.assertEqual(len(self.evaluator.calls), calls)

    def test_completed_invalid_is_cached_and_resume_eligible(self):
        frozen = self.store.create_run(self.candidates, self.evaluator)
        runner = Stage6EvaluationRunner(self.store, self.evaluator)
        runner.run(frozen.run_id, max_new_evaluations=5)
        item = self.store.run_candidates(frozen.run_id)[4]
        self.assertEqual(item["state"], "completed_invalid")
        identity = evaluation_cache_identity(self.evaluator, item["candidate"])
        self.assertEqual(self.store.lookup_verified(identity)["status"], "completed_invalid")

    def test_selection_loader_revalidates_a_complete_run_without_evaluator_calls(self):
        frozen = self.store.create_run(self.candidates, self.evaluator)
        Stage6EvaluationRunner(self.store, self.evaluator).run(frozen.run_id)
        calls = len(self.evaluator.calls)
        verified = self.store.load_verified_run_results(frozen.run_id)
        self.assertEqual(len(verified.records), 12)
        self.assertEqual(verified.run_id, frozen.run_id)
        self.assertEqual(len(verified.ordered_result_set_fingerprint), 64)
        self.assertEqual(len(self.evaluator.calls), calls)
        self.assertEqual(
            [record["structural_hash"] for record in verified.records],
            frozen.manifest["ordered_candidate_hashes"],
        )

    def test_selection_loader_rejects_an_incomplete_run(self):
        frozen = self.store.create_run(self.candidates, self.evaluator)
        Stage6EvaluationRunner(self.store, self.evaluator).run(
            frozen.run_id, max_new_evaluations=5
        )
        with self.assertRaisesRegex(EvaluationStoreIntegrityError, "complete"):
            self.store.load_verified_run_results(frozen.run_id)

    def test_partial_loader_revalidates_only_completed_rows_without_mutation(self):
        frozen = self.store.create_run(self.candidates, self.evaluator)
        Stage6EvaluationRunner(self.store, self.evaluator).run(
            frozen.run_id, max_new_evaluations=5
        )
        states_before = [
            item["state"] for item in self.store.run_candidates(frozen.run_id)
        ]
        calls_before = len(self.evaluator.calls)
        verified = self.store.load_verified_completed_results(frozen.run_id)
        states_after = [
            item["state"] for item in self.store.run_candidates(frozen.run_id)
        ]
        self.assertEqual(verified.completed_count, 5)
        self.assertEqual(verified.candidate_count, 12)
        self.assertEqual(len(verified.records), 5)
        self.assertEqual(states_after, states_before)
        self.assertEqual(len(self.evaluator.calls), calls_before)

    def test_resource_limited_run_resolves_only_existing_cache_rows(self):
        source = self.store.create_run(self.candidates[:3], self.evaluator)
        Stage6EvaluationRunner(self.store, self.evaluator).run(source.run_id)
        cache_only = _CacheOnlyTrainPreparationEvaluator(
            context_fingerprint=self.evaluator.context.fingerprint,
            evaluation_contract_fingerprint=(
                self.evaluator.evaluation_contract_fingerprint
            ),
            compatibility_audit_fingerprint=(
                self.evaluator.compatibility_audit_fingerprint
            ),
            accepted_registry_fingerprint=(
                self.evaluator.accepted_registry_fingerprint
            ),
            reusable_hashes=set(),
        )
        frozen = self.store.create_run(
            self.candidates[:3],
            cache_only,
            scope=STAGE6_RESOURCE_LIMITED_TRAIN_SCOPE,
        )
        summary = Stage6EvaluationRunner(self.store, cache_only).run(frozen.run_id)
        self.assertEqual(summary.run_status, "complete")
        self.assertEqual(summary.cache_hits, 3)
        self.assertEqual(summary.newly_evaluated, 0)

    def test_stale_running_is_recovered_and_recomputed(self):
        frozen = self.store.create_run(self.candidates, self.evaluator)
        first = self.store.run_candidates(frozen.run_id)[0]
        self.store.begin_attempt(frozen.run_id, 0, first["cache_key"])
        runner = Stage6EvaluationRunner(self.store, self.evaluator)
        summary = runner.run(frozen.run_id, max_new_evaluations=1)
        self.assertEqual(summary.recovered_interrupted, 1)
        attempt_outcomes = [
            row[0]
            for row in self.store.connection.execute(
                "SELECT outcome FROM attempts WHERE run_id=? ORDER BY started_at",
                (frozen.run_id,),
            )
        ]
        self.assertIn("interrupted", attempt_outcomes)
        self.assertIn("completed", attempt_outcomes)

    def test_failed_is_not_cached_and_requires_explicit_retry(self):
        frozen = self.store.create_run(self.candidates[:1], self.evaluator)
        target = self.candidates[0]["current_structural_hash"]
        self.evaluator.fail_hashes.add(target)
        runner = Stage6EvaluationRunner(self.store, self.evaluator)
        first = runner.run(frozen.run_id)
        self.assertEqual(first.run_status, "incomplete")
        identity = evaluation_cache_identity(self.evaluator, self.candidates[0])
        self.assertIsNone(self.store.lookup_verified(identity))
        calls = len(self.evaluator.calls)
        second = runner.run(frozen.run_id)
        self.assertEqual(second.newly_evaluated, 0)
        self.assertEqual(len(self.evaluator.calls), calls)
        self.evaluator.fail_hashes.clear()
        third = runner.run(frozen.run_id, retry_failed=True)
        self.assertEqual(third.run_status, "complete")
        self.assertEqual(third.newly_evaluated, 1)

    def test_corrupt_result_json_and_nonfinite_json_fail_closed(self):
        frozen = self.store.create_run(self.candidates[:1], self.evaluator)
        Stage6EvaluationRunner(self.store, self.evaluator).run(frozen.run_id)
        identity = evaluation_cache_identity(self.evaluator, self.candidates[0])
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE evaluations SET result_json='{}' WHERE cache_key=?",
                (identity.cache_key,),
            )
        with self.assertRaises(EvaluationStoreIntegrityError):
            self.store.lookup_verified(identity)
        with self.store.connection:
            self.store.connection.execute(
                "UPDATE evaluations SET result_json='NaN' WHERE cache_key=?",
                (identity.cache_key,),
            )
        with self.assertRaises(EvaluationStoreIntegrityError):
            self.store.lookup_verified(identity)

    def test_run_refuses_different_context_or_contract(self):
        frozen = self.store.create_run(self.candidates, self.evaluator)
        with self.assertRaisesRegex(EvaluationStoreIntegrityError, "different evaluator"):
            self.store.validate_run_evaluator(frozen.run_id, FakeEvaluator(context="e"))

    def test_determinism_conflict_preserves_cached_result_and_records_ledger(self):
        frozen = self.store.create_run(self.candidates[:1], self.evaluator)
        runner = Stage6EvaluationRunner(self.store, self.evaluator)
        runner.run(frozen.run_id)
        identity = evaluation_cache_identity(self.evaluator, self.candidates[0])
        original = self.store.lookup_verified(identity)["result_fingerprint"]
        self.evaluator.variant = 1
        with self.assertRaises(DeterminismConflictError):
            runner.verify_determinism(frozen.run_id, [0])
        self.assertEqual(
            self.store.lookup_verified(identity)["result_fingerprint"], original
        )
        self.assertEqual(self.store.database_counts()["determinism_conflicts"], 1)

    def test_determinism_bypass_matches_without_overwriting_cache(self):
        frozen = self.store.create_run(self.candidates[:1], self.evaluator)
        runner = Stage6EvaluationRunner(self.store, self.evaluator)
        runner.run(frozen.run_id)
        checks = runner.verify_determinism(frozen.run_id, [0])
        self.assertTrue(checks[0]["match"])
        self.assertEqual(self.store.database_counts()["evaluations"], 1)
        outcome = self.store.connection.execute(
            "SELECT outcome FROM attempts WHERE mode='determinism_bypass'"
        ).fetchone()[0]
        self.assertEqual(outcome, "determinism_verified")

    def test_smoke_selection_is_deterministic_unique_and_ignores_metrics(self):
        rows = []
        token_sets = (
            [get_action_id("open")],
            [get_action_id("neg"), get_action_id("open")],
            [get_action_id("add"), get_action_id("open"), get_action_id("close")],
            [get_action_id("ts_mean", 5), get_action_id("close")],
            [get_action_id("cs_rank"), get_action_id("volume")],
            [
                get_action_id("cs_rank"),
                get_action_id("ts_mean", 5),
                get_action_id("close"),
            ],
        )
        for index in range(24):
            row = _candidate(index)
            tokens = token_sets[index % len(token_sets)]
            row["prefix_token_ids"] = tokens
            row["node_count"] = len(tokens) + index // 6
            row["depth"] = min(6, row["node_count"] - 1)
            row["origin_ids"] = ["a", "b"] if index % 7 == 0 else ["a"]
            row["reward"] = 1000 - index
            rows.append(row)
        first = select_stage6_smoke_candidates(rows, 12)
        second = select_stage6_smoke_candidates(list(reversed(rows)), 12)
        hashes = [row["current_structural_hash"] for row in first]
        self.assertEqual(hashes, [row["current_structural_hash"] for row in second])
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(len(hashes), 12)

    def test_rss_sampler_records_process_memory(self):
        with ProcessRssSampler(0.01) as sampler:
            payload = bytearray(1024 * 1024)
            self.assertEqual(len(payload), 1024 * 1024)
        measurement = sampler.measurement()
        self.assertGreater(measurement.rss_before_bytes, 0)
        self.assertGreaterEqual(measurement.peak_rss_bytes, measurement.rss_before_bytes)
        self.assertGreaterEqual(measurement.peak_delta_bytes, 0)


if __name__ == "__main__":
    unittest.main()
