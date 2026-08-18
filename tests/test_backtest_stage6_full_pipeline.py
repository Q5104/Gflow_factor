from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import json
import tempfile
import unittest

import nbformat

from factor_gfn.backtest.stage6_evaluation import (
    STAGE6_EVALUATION_RESULT_SCHEMA,
    Stage6CandidateEvaluationResult,
    _stable_hash,
)
from factor_gfn.backtest.stage6_evaluation_store import (
    EvaluationStore,
    Stage6EvaluationRunner,
)
from factor_gfn.backtest.stage6_full_pipeline import (
    FULL_EVALUATION_SCOPE,
    STAGE6_FULL_ENTRY_SCHEMA,
    STAGE6_FULL_PIPELINE_VERSION,
    load_stage6_mean_ic_distribution,
    stage6_evaluation_run_snapshot,
)
from factor_gfn.grammar import get_action_id


def _candidate(index: int) -> dict:
    structural_hash = f"{index + 1:064x}"
    return {
        "current_structural_hash": structural_hash,
        "formula": f"candidate_{index}",
        "prefix_token_ids": [get_action_id("open")],
        "node_count": 1,
        "depth": 0,
        "origin_ids": [f"origin-{index}"],
    }


class _Evaluator:
    def __init__(self) -> None:
        self.context = SimpleNamespace(fingerprint="a" * 64)
        self.evaluation_contract_fingerprint = "b" * 64
        self.compatibility_audit_fingerprint = "c" * 64
        self.accepted_registry_fingerprint = "d" * 64
        self.calls: list[str] = []
        self.fail_hashes: set[str] = set()
        self.include_train_prefilter = False

    def resolve_candidate_identity(self, candidate):
        return {
            "structural_hash": candidate["current_structural_hash"],
            "formula": candidate["formula"],
            "prefix_token_ids": candidate["prefix_token_ids"],
            "node_count": candidate["node_count"],
            "depth": candidate["depth"],
        }

    def has_reusable_train(self, candidate):
        return int(candidate["current_structural_hash"], 16) % 2 == 1

    def planned_path(self, candidate):
        return (
            "verified_stage5_train_preparation_reuse"
            if self.has_reusable_train(candidate)
            else "stage6_fresh_train_preparation"
        )

    def evaluate(self, candidate):
        structural_hash = candidate["current_structural_hash"]
        self.calls.append(structural_hash)
        if structural_hash in self.fail_hashes:
            raise RuntimeError("synthetic failure")
        index = int(structural_hash, 16) - 1
        expression = dict(self.resolve_candidate_identity(candidate))
        train = {"ic": {"mean": -0.02 + index * 0.01}}
        if self.include_train_prefilter:
            train["train_prefilter"] = {
                "status": (
                    "train_prefilter_passed"
                    if index % 2 == 0
                    else "train_prefilter_failed"
                )
            }
        validation = {"ic": {"mean": -0.01 + index * 0.005}}
        coverage = {"train": {"rate": 1.0}, "validation": {"rate": 1.0}}
        deterministic = {
            "schema": STAGE6_EVALUATION_RESULT_SCHEMA,
            "status": "completed",
            "invalid_reasons": [],
            "expression": expression,
            "context_fingerprint": self.context.fingerprint,
            "evaluation_contract_fingerprint": self.evaluation_contract_fingerprint,
            "train_direction": 1,
            "train": train,
            "validation": validation,
            "factor_finite_coverage": coverage,
        }
        return Stage6CandidateEvaluationResult(
            schema=STAGE6_EVALUATION_RESULT_SCHEMA,
            status="completed",
            invalid_reasons=(),
            expression=MappingProxyType(expression),
            source_identity=MappingProxyType({"origin_ids": candidate["origin_ids"]}),
            context_fingerprint=self.context.fingerprint,
            evaluation_contract_fingerprint=self.evaluation_contract_fingerprint,
            train_direction=1,
            train=MappingProxyType(train),
            validation=MappingProxyType(validation),
            factor_finite_coverage=MappingProxyType(coverage),
            factor_seconds=0.01,
            train_evaluation_seconds=0.01,
            validation_evaluation_seconds=0.01,
            total_seconds=0.03,
            result_fingerprint=_stable_hash(deterministic),
        )


class Stage6FullPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "store" / "stage6_evaluations.sqlite"
        self.artifacts = self.root / "runs"
        self.store = EvaluationStore(self.database, self.artifacts)
        self.evaluator = _Evaluator()
        self.candidates = [_candidate(index) for index in range(4)]

    def tearDown(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass
        self.temporary.cleanup()

    def test_full_scope_freezes_all_candidates_and_reports_progress(self):
        frozen = self.store.create_run(
            self.candidates,
            self.evaluator,
            scope=FULL_EVALUATION_SCOPE,
        )
        self.evaluator.fail_hashes.add(self.candidates[0]["current_structural_hash"])
        events = []
        summary = Stage6EvaluationRunner(self.store, self.evaluator).run(
            frozen.run_id,
            max_new_evaluations=2,
            progress_callback=events.append,
        )
        self.assertEqual(frozen.candidate_count, 4)
        self.assertEqual(frozen.manifest["scope"], FULL_EVALUATION_SCOPE)
        self.assertEqual(len(self.evaluator.calls), 2)
        self.assertEqual(summary.newly_failed, 1)
        self.assertEqual(summary.newly_evaluated, 1)
        self.assertEqual(events[0]["event_type"], "invocation_started")
        self.assertEqual(events[-1]["event_type"], "invocation_completed")
        started = [row for row in events if row["event_type"] == "candidate_started"]
        self.assertEqual([row["reusable_train"] for row in started], [True, False])
        self.assertEqual(events[-1]["reused_train_resolved"], 1)
        self.assertEqual(events[-1]["fresh_train_resolved"], 1)
        self.assertIn("train_prefilter_status_counts", events[-1])
        self.assertEqual(events[-1]["train_prefilter_status_by_path"], {})
        self.assertIn("validation_available", events[-1])

    def test_progress_separates_prefilter_status_by_train_path(self):
        self.evaluator.include_train_prefilter = True
        frozen = self.store.create_run(
            self.candidates[:2], self.evaluator, scope=FULL_EVALUATION_SCOPE
        )
        events = []
        Stage6EvaluationRunner(self.store, self.evaluator).run(
            frozen.run_id, progress_callback=events.append
        )
        self.assertEqual(
            events[-1]["train_prefilter_status_by_path"],
            {
                "stage6_fresh_train_preparation": {"train_prefilter_failed": 1},
                "verified_stage5_train_preparation_reuse": {
                    "train_prefilter_passed": 1
                },
            },
        )

    def test_snapshot_separates_planned_reuse_fresh_and_cache_events(self):
        frozen = self.store.create_run(
            self.candidates,
            self.evaluator,
            scope=FULL_EVALUATION_SCOPE,
        )
        runner = Stage6EvaluationRunner(self.store, self.evaluator)
        runner.run(frozen.run_id)
        runner.run(frozen.run_id)
        overlay = SimpleNamespace(
            records={
                self.candidates[0]["current_structural_hash"]: {},
                self.candidates[2]["current_structural_hash"]: {},
            }
        )
        snapshot = stage6_evaluation_run_snapshot(
            self.store, frozen.run_id, overlay=overlay
        )
        self.assertEqual(snapshot["validation_completed"], 4)
        self.assertEqual(snapshot["planned_reused_train"], 2)
        self.assertEqual(snapshot["completed_reused_train"], 2)
        self.assertEqual(snapshot["resume_skipped_events"], 4)
        self.assertEqual(snapshot["cache_hit_events"], 0)
        self.assertEqual(snapshot["oos"], "not_loaded")

    def test_distribution_requires_complete_run_and_reports_ddof0_stats(self):
        frozen = self.store.create_run(
            self.candidates,
            self.evaluator,
            scope=FULL_EVALUATION_SCOPE,
        )
        Stage6EvaluationRunner(self.store, self.evaluator).run(frozen.run_id)
        self.store.close()
        stable = {
            "schema": STAGE6_FULL_ENTRY_SCHEMA,
            "pipeline_version": STAGE6_FULL_PIPELINE_VERSION,
            "evaluation_run_id": frozen.run_id,
            "evaluation_run_scope": FULL_EVALUATION_SCOPE,
            "candidate_count": 4,
            "database_path": str(self.database.resolve()),
            "run_artifact_root": str(self.artifacts.resolve()),
            "oos": "not_loaded_not_evaluated",
        }
        entry = {**stable, "entry_manifest_fingerprint": _stable_hash(stable)}
        entry_path = self.root / "full_evaluation_entry_manifest.json"
        entry_path.write_text(json.dumps(entry), encoding="utf-8")
        result = load_stage6_mean_ic_distribution(
            entry_manifest_path=entry_path,
            include_validation=True,
        )
        self.assertEqual(result["train_ic"]["statistics"]["count"], 4)
        self.assertAlmostEqual(result["train_ic"]["statistics"]["mean"], -0.005)
        self.assertAlmostEqual(
            result["train_ic"]["statistics"]["std_ddof0"],
            0.011180339887498949,
        )
        self.assertIn("validation_ic", result)

    def test_distribution_can_limit_to_train_prefilter_pass_candidates(self):
        class _PrefilterEvaluator(_Evaluator):
            def evaluate(self, candidate):
                result = super().evaluate(candidate)
                payload = result.to_dict()
                index = int(candidate["current_structural_hash"], 16) - 1
                payload["train"]["train_prefilter"] = {
                    "status": (
                        "train_prefilter_passed"
                        if index in {1, 3}
                        else "train_prefilter_failed"
                    )
                }
                deterministic = {
                    key: payload[key]
                    for key in (
                        "schema",
                        "status",
                        "invalid_reasons",
                        "expression",
                        "context_fingerprint",
                        "evaluation_contract_fingerprint",
                        "train_direction",
                        "train",
                        "validation",
                        "factor_finite_coverage",
                    )
                }
                return replace(
                    result,
                    train=MappingProxyType(payload["train"]),
                    result_fingerprint=_stable_hash(deterministic),
                )

        evaluator = _PrefilterEvaluator()
        frozen = self.store.create_run(
            self.candidates, evaluator, scope=FULL_EVALUATION_SCOPE
        )
        Stage6EvaluationRunner(self.store, evaluator).run(frozen.run_id)
        self.store.close()
        stable = {
            "schema": STAGE6_FULL_ENTRY_SCHEMA,
            "pipeline_version": STAGE6_FULL_PIPELINE_VERSION,
            "evaluation_run_id": frozen.run_id,
            "evaluation_run_scope": FULL_EVALUATION_SCOPE,
            "candidate_count": 4,
            "database_path": str(self.database.resolve()),
            "run_artifact_root": str(self.artifacts.resolve()),
            "oos": "not_loaded_not_evaluated",
        }
        entry = {**stable, "entry_manifest_fingerprint": _stable_hash(stable)}
        entry_path = self.root / "prefilter_entry.json"
        entry_path.write_text(json.dumps(entry), encoding="utf-8")
        result = load_stage6_mean_ic_distribution(
            entry_manifest_path=entry_path,
            train_prefilter_pass_only=True,
        )
        self.assertEqual(result["scope_candidate_count"], 2)
        self.assertEqual(
            result["candidate_scope"],
            "train_prefilter_pass_with_complete_stage6_evaluation",
        )

    def test_formal_hybrid_notebook_is_the_only_stage6_execution_entry(self):
        project_root = Path(__file__).resolve().parents[1]
        full_path = project_root / "notebooks/run_stage6_full_train_validation_evaluation.ipynb"
        selection_path = project_root / "notebooks/run_stage6_provisional_selection.ipynb"
        old_unified_path = project_root / "notebooks/run_stage6_provisional_pipeline.ipynb"
        unified_path = project_root / "notebooks/run_stage6_hybrid_formal_selection.ipynb"
        self.assertFalse(full_path.exists())
        self.assertFalse(selection_path.exists())
        self.assertFalse(old_unified_path.exists())
        unified = nbformat.read(unified_path, as_version=4)
        unified_source = "\n".join(cell.source for cell in unified.cells)
        for index, cell in enumerate(unified.cells):
            if cell.cell_type == "code":
                self.assertGreater(index, 0)
                self.assertEqual(unified.cells[index - 1].cell_type, "markdown")
                self.assertTrue(unified.cells[index - 1].source.strip())
        self.assertIn("HYBRID_RUN_ID = 'hybrid_5_15_k16_seed42_20260816T025559Z'", unified_source)
        self.assertIn("materialize_source_set(", unified_source)
        self.assertIn("import_candidate_source_set(", unified_source)
        self.assertIn("run_current_stage6_train_preparation(", unified_source)
        self.assertIn("run_current_stage6_validation_evaluation(", unified_source)
        self.assertIn("run_current_stage6_two_phase_provisional_selection(", unified_source)
        self.assertLess(
            unified_source.index("run_current_stage6_train_preparation("),
            unified_source.index("run_current_stage6_validation_evaluation("),
        )
        self.assertLess(
            unified_source.index("run_current_stage6_validation_evaluation("),
            unified_source.index("run_current_stage6_two_phase_provisional_selection("),
        )

    def test_two_phase_pipeline_passes_real_data_paths_with_supported_keyword(self):
        project_root = Path(__file__).resolve().parents[1]
        pipeline_source = (
            project_root / "factor_gfn/backtest/stage6_two_phase_pipeline.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            "build_stage6_evaluation_context(data_paths=", pipeline_source
        )
        self.assertEqual(
            pipeline_source.count("build_stage6_evaluation_context(paths=data_paths)"),
            3,
        )


if __name__ == "__main__":
    unittest.main()
