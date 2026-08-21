import json
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factor_gfn.backtest.stage6_selection import HARD_CONDITION_CODES
from factor_gfn.grammar import get_action_id
from factor_gfn.reporting.stage6_data import (
    _correlation_matrix,
    _greedy_audit,
    _rank_hashes,
    _token_prevalence,
    build_stage6_report_data,
    load_stage6_report_data,
    pair_correlation,
)
from factor_gfn.backtest.stage6_evaluation import _stable_hash
from factor_gfn.backtest.stage6_selection import Stage6SelectionConfig
from factor_gfn.reporting.stage6_renderer import Stage6ReportRenderer


def _hash(index: int) -> str:
    return f"{index:064x}"


def _dates(count: int = 60) -> list[str]:
    return [f"2020-{index // 28 + 1:02d}-{index % 28 + 1:02d}" for index in range(count)]


def _source(index: int) -> dict:
    return {
        "current_structural_hash": _hash(index),
        "formula": "open",
        "prefix_token_ids": [get_action_id("open")],
        "node_count": 1,
        "depth": 0,
    }


def _stored(index: int, *, train_ic: float, validation_ic: float, train_ir: float, validation_ir: float, values) -> dict:
    expression = {
        "structural_hash": _hash(index),
        "formula": "open",
        "prefix_token_ids": [get_action_id("open")],
        "node_count": 1,
        "depth": 0,
    }
    result = {
        "status": "completed",
        "expression": expression,
        "source_identity": {"origin_ids": [f"origin-{index}"]},
        "train_direction": 1 if train_ic > 0 else -1,
        "train": {
            "ic": {"mean": train_ic},
            "long": {"annualized_ir": train_ir, "excess_series": {"dates": _dates(), "values": list(values)}},
            "barra": {"max_abs_correlation": .2},
        },
        "validation": {
            "ic": {"mean": validation_ic},
            "long": {"annualized_ir": validation_ir},
        },
    }
    return {"ordinal": index, "structural_hash": _hash(index), "result_fingerprint": f"result-{index}", "result": result}


def _hard(index: int, *, passed: bool, validation_long_pass: bool = True) -> dict:
    checks = {code: True for code in HARD_CONDITION_CODES}
    failed = []
    if not validation_long_pass:
        checks["validation_long_ir_gt_0_25"] = False
        failed = ["validation_long_ir_gt_0_25"]
    return {
        "structural_hash": _hash(index),
        "hard_filter_pass": passed,
        "condition_results": checks,
        "failed_conditions": failed,
        "metrics": {
            "train_ic": {1: .03, 2: .02, 3: .015}[index],
            "validation_ic": {1: .02, 2: .018, 3: .012}[index],
            "train_long_ir": .4,
            "validation_long_ir": .4 if validation_long_pass else .1,
            "train_barra_ts_corr": .2,
        },
    }


def _bundle():
    base = np.arange(60, dtype=float)
    orthogonal = np.sin(np.arange(60, dtype=float) * .73)
    validation = [
        _stored(1, train_ic=.03, validation_ic=.02, train_ir=.4, validation_ir=.4, values=base),
        _stored(2, train_ic=.02, validation_ic=.018, train_ir=.35, validation_ir=.38, values=base * 2),
        _stored(3, train_ic=.015, validation_ic=.012, train_ir=.3, validation_ir=.1, values=orthogonal),
    ]
    train = []
    for index in range(4):
        passed = index != 0
        checks = {
            "train_abs_ic_gt_0_01": passed,
            "train_long_ir_gt_0_25": True,
            "train_barra_ts_corr_lt_0_7": True,
        }
        train.append({"structural_hash": _hash(index), "status": "train_prefilter_passed" if passed else "train_prefilter_failed", "condition_results": checks, "failed_conditions": [] if passed else ["train_abs_ic_gt_0_01"]})
    hard = [_hard(1, passed=True), _hard(2, passed=True), _hard(3, passed=False, validation_long_pass=False)]
    greedy = [
        {"sorted_rank": 1, "structural_hash": _hash(1), "decorrelation_status": "retained", "blocking_corr": None, "blocked_by_structural_hash": None},
        {"sorted_rank": 2, "structural_hash": _hash(2), "decorrelation_status": "rejected_by_correlation", "blocking_corr": 1.0, "blocked_by_structural_hash": _hash(1)},
    ]
    pool = [{"sorted_rank": 1, "structural_hash": _hash(1), "expression": validation[0]["result"]["expression"], "source_identity": {}, "result_fingerprint": "result-1"}]
    snapshot = {"schema": "factor_gfn.reporting.stage6_data.v1", "source_snapshot_fingerprint": "s" * 64, "source_set_fingerprint": "t" * 64, "accepted_registry_fingerprint": "a" * 64, "train_entry_fingerprint": "b" * 64, "train_pass_fingerprint": "c" * 64, "validation_run_id": "validation", "validation_context_fingerprint": "d" * 64, "validation_evaluation_fingerprint": "e" * 64, "selection_fingerprint": "f" * 64, "selection_contract_fingerprint": "1" * 64, "enrichment_fingerprint": "2" * 64, "selection_scope": "provisional_selection", "oos": "not_loaded_not_evaluated", "top_k_correlation": 30, "minimum_common_periods": 60, "pair_audit_version": "stage6-reporting-pair-audit-v1"}
    return build_stage6_report_data(snapshot_manifest=snapshot, source_candidates=[_source(index) for index in range(4)], train_prefilter_results=train, validation_records=validation, hard_filter_results=hard, greedy_results=greedy, alpha_pool=pool, enrichment_results=[])


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    return {"size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _formal_fixture(root: Path) -> dict[str, Path]:
    structural_hash = _hash(1)
    representation = {"formula": "open", "prefix_token_ids": [get_action_id("open")], "node_count": 1, "depth": 0, "representation_digest": "r", "origin_ids": ["origin-1"]}
    snapshot = {
        "schema": "factor_gfn.stage6_source_snapshot.v1", "source_id": "hybrid", "source_type": "hybrid_train_artifact", "source_role": "formal_discovery", "inclusion_status": "approved", "approval_note": "test", "candidate_record_policy": {}, "source_semantics": {}, "snapshot_kind": "completed_hybrid_train_artifact", "cutoff": {"complete": True, "pending_assignment": None, "candidate_count": 1}, "record_counts": {"records": 1}, "artifacts": [], "logical_content_fingerprint": "l" * 64,
    }
    snapshot_core = dict(snapshot)
    snapshot["snapshot_fingerprint"] = _stable_hash(snapshot_core)
    snapshot_path = root / "source" / "source_snapshot.json"; _write_json(snapshot_path, snapshot)
    source_core = {"schema": "factor_gfn.stage6_source_set.v1", "mode": "provisional", "sources": [{"source_id": "hybrid", "source_type": "hybrid_train_artifact", "source_role": "formal_discovery", "snapshot_fingerprint": snapshot["snapshot_fingerprint"]}]}
    source_set = {**source_core, "source_set_fingerprint": _stable_hash(source_core), "source_manifests": [{**source_core["sources"][0], "snapshot_manifest": str(snapshot_path)}]}
    source_set_path = root / "source_set_manifest.json"; _write_json(source_set_path, source_set)

    group = {"source_claimed_structural_hash": structural_hash, "downstream_eligible": True, "representations": [representation]}
    import_dir = root / "candidate_import"; registry_meta = _write_jsonl(import_dir / "candidate_registry.jsonl", [group])
    import_payload = {"fixture": "candidate-import"}
    candidate_import = {"schema": "factor_gfn.stage6_candidate_import_manifest.v1", "source_set_fingerprint": source_set["source_set_fingerprint"], "downstream_eligible": True, "fingerprint_payload": import_payload, "registry_fingerprint": _stable_hash(import_payload), "artifacts": {"candidate_registry.jsonl": registry_meta}}
    candidate_path = import_dir / "candidate_import_manifest.json"; _write_json(candidate_path, candidate_import)

    accepted = [{"current_structural_hash": structural_hash, **{key: representation[key] for key in ("formula", "prefix_token_ids", "node_count", "depth")}}]
    compatibility_dir = root / "compatibility"; accepted_meta = _write_jsonl(compatibility_dir / "auto_accepted_candidate_registry.jsonl", accepted)
    compatibility_payload = {"fixture": "compatibility"}
    compatibility = {"schema": "factor_gfn.stage6_expression_compatibility_manifest.v1", "downstream_eligible": True, "fingerprint_payload": compatibility_payload, "audit_fingerprint": _stable_hash(compatibility_payload), "accepted_registry_fingerprint": _stable_hash(accepted), "artifacts": {"auto_accepted_candidate_registry.jsonl": accepted_meta}}
    compatibility_path = compatibility_dir / "expression_compatibility_manifest.json"; _write_json(compatibility_path, compatibility)

    train_entry = {"schema": "factor_gfn.stage6_train_preparation_entry.v1", "evaluation_run_scope": "train_preparation_full_accepted_registry", "candidate_count": 1, "accepted_registry_fingerprint": compatibility["accepted_registry_fingerprint"], "oos": "not_loaded_not_evaluated"}
    train_entry["entry_manifest_fingerprint"] = _stable_hash(train_entry)
    train_entry_path = root / "train_entry.json"; _write_json(train_entry_path, train_entry)
    prefilter = [{"structural_hash": structural_hash, "status": "train_prefilter_passed", "condition_results": {"train_abs_ic_gt_0_01": True, "train_long_ir_gt_0_25": True, "train_barra_ts_corr_lt_0_7": True}, "failed_conditions": []}]
    pass_dir = root / "train_pass"; prefilter_meta = _write_jsonl(pass_dir / "train_prefilter_results.jsonl", prefilter)
    train_pass = {"schema": "factor_gfn.stage6_train_pass_manifest.v1", "train_entry_manifest_fingerprint": train_entry["entry_manifest_fingerprint"], "selection_config_fingerprint": Stage6SelectionConfig().fingerprint, "train_pass_count": 1, "oos": "not_loaded_not_evaluated", "artifacts": {"train_prefilter_results.jsonl": prefilter_meta}}
    train_pass["train_pass_manifest_fingerprint"] = _stable_hash(train_pass)
    train_pass_path = pass_dir / "train_pass_manifest.json"; _write_json(train_pass_path, train_pass)

    values = np.sin(np.arange(60, dtype=float)).tolist()
    stored = _stored(1, train_ic=.03, validation_ic=.02, train_ir=.4, validation_ir=.4, values=values)
    stored["cache_key"] = "cache-1"
    stored["result"].update({"schema": "factor_gfn.stage6_evaluation_result.v1", "invalid_reasons": [], "context_fingerprint": "c" * 64, "evaluation_contract_fingerprint": "e" * 64, "factor_finite_coverage": {}})
    deterministic_keys = ("schema", "status", "invalid_reasons", "expression", "context_fingerprint", "evaluation_contract_fingerprint", "train_direction", "train", "validation", "factor_finite_coverage")
    result_fingerprint = _stable_hash({key: stored["result"][key] for key in deterministic_keys})
    stored["result"]["result_fingerprint"] = result_fingerprint
    database = root / "validation.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript("CREATE TABLE runs(run_id TEXT PRIMARY KEY,manifest_json TEXT,status TEXT); CREATE TABLE run_candidates(run_id TEXT,ordinal INTEGER,structural_hash TEXT,cache_key TEXT,state TEXT,result_fingerprint TEXT); CREATE TABLE evaluations(cache_key TEXT PRIMARY KEY,result_json TEXT,result_fingerprint TEXT); CREATE TABLE determinism_conflicts(cache_key TEXT);")
    run_manifest = {"candidate_count": 1, "context_fingerprint": "c" * 64, "evaluation_contract_fingerprint": "e" * 64, "oos": "not_loaded"}
    connection.execute("INSERT INTO runs VALUES(?,?,?)", ("validation-run", json.dumps(run_manifest), "complete"))
    connection.execute("INSERT INTO run_candidates VALUES(?,?,?,?,?,?)", ("validation-run", 0, structural_hash, "cache-1", "completed", result_fingerprint))
    connection.execute("INSERT INTO evaluations VALUES(?,?,?)", ("cache-1", json.dumps(stored["result"]), result_fingerprint)); connection.commit(); connection.close()
    identity = [{"ordinal": 0, "structural_hash": structural_hash, "cache_key": "cache-1", "result_fingerprint": result_fingerprint}]
    validation_entry = {"schema": "factor_gfn.stage6_validation_entry.v1", "evaluation_run_scope": "validation_from_frozen_train_pass_manifest", "evaluation_run_id": "validation-run", "candidate_count": 1, "context_fingerprint": "c" * 64, "evaluation_contract_fingerprint": "e" * 64, "database_path": str(database), "train_pass_manifest_fingerprint": train_pass["train_pass_manifest_fingerprint"], "accepted_registry_fingerprint": compatibility["accepted_registry_fingerprint"], "oos": "not_loaded_not_evaluated"}
    validation_entry["entry_manifest_fingerprint"] = _stable_hash(validation_entry)
    validation_path = root / "validation_entry.json"; _write_json(validation_path, validation_entry)

    hard = [_hard(1, passed=True)]
    greedy = [{"sorted_rank": 1, "structural_hash": structural_hash, "decorrelation_status": "retained", "blocking_corr": None, "blocked_by_structural_hash": None}]
    pool = [{"sorted_rank": 1, "structural_hash": structural_hash, "expression": stored["result"]["expression"], "source_identity": {}, "result_fingerprint": result_fingerprint}]
    enrichment = [{"structural_hash": structural_hash, "status": "already_available"}]
    selection_dir = root / "selection"
    artifacts = {name: _write_jsonl(selection_dir / name, rows) for name, rows in (("hard_filter_results.jsonl", hard), ("greedy_decorrelation_results.jsonl", greedy), ("alpha_pool.jsonl", pool), ("survivor_long_excess_enrichment.jsonl", enrichment))}
    selection = {"schema": "factor_gfn.stage6_enriched_selection_manifest.v1", "version": "test", "engineering_smoke": False, "evaluation_run_id": "validation-run", "evaluation_ordered_result_set_fingerprint": _stable_hash(identity), "evaluation_contract_fingerprint": "e" * 64, "context_fingerprint": "c" * 64, "selection_contract_fingerprint": Stage6SelectionConfig().fingerprint, "enrichment_contract_fingerprint": "n" * 64, "hard_filter_digest": _stable_hash(hard), "enrichment_digest": _stable_hash(enrichment), "greedy_digest": _stable_hash(greedy), "alpha_pool_digest": _stable_hash(pool), "oos": "not_loaded_not_evaluated"}
    selection["enriched_selection_fingerprint"] = _stable_hash(selection)
    selection["counts"] = {"input_candidates": 1, "retained": 1}; selection["artifacts"] = artifacts
    selection["created_at_utc"] = "2026-08-18T00:00:00+00:00"
    selection["created_at_excluded_from_fingerprint"] = True
    selection["scope"] = "provisional_selection"
    selection_path = selection_dir / "enriched_selection_manifest.json"; _write_json(selection_path, selection)
    return {"source_set_manifest_path": source_set_path, "candidate_import_manifest_path": candidate_path, "compatibility_manifest_path": compatibility_path, "train_entry_manifest_path": train_entry_path, "train_pass_manifest_path": train_pass_path, "validation_entry_manifest_path": validation_path, "selection_manifest_path": selection_path}


class Stage6ReportingDataTests(unittest.TestCase):
    def test_universe_contract_and_validation_missing_are_preserved(self):
        bundle = _bundle()
        self.assertEqual(bundle.funnel_summary["remaining_count"].tolist(), [4, 3, 3, 2, 2, 1, 1])
        validation_condition = bundle.hard_filter_condition_summary.set_index("condition").loc["validation_abs_ic_gt_0_01"]
        self.assertEqual(int(validation_condition["observed_count"]), 3)
        self.assertEqual(int(validation_condition["not_evaluated_count"]), 1)
        self.assertEqual(set(bundle.failure_combinations["observation_scope"]), {"train_prefilter_exit", "six_condition_observed"})

    def test_signed_mean_ic_and_exact_universe_mapping(self):
        bundle = _bundle()
        metrics = bundle.validation_candidate_metrics.set_index("structural_hash")
        self.assertEqual(metrics.loc[_hash(1), "train_ic"], .03)
        self.assertEqual(metrics.loc[_hash(1), "abs_train_ic"], abs(.03))
        self.assertEqual(set(bundle.decorrelation_input["structural_hash"]), {_hash(1), _hash(2)})
        self.assertEqual(set(bundle.provisional_factor_pool["structural_hash"]), {_hash(1)})
        self.assertEqual(bundle.top100_candidate_metrics["provisional_rank"].tolist(), [1])
        self.assertEqual(bundle.top_candidate_examples["structural_hash"].tolist(), [_hash(1)])
        self.assertEqual(
            bundle.complexity_summary["metric"].tolist(),
            ["node_count", "depth", "operator_count", "leaf_count"],
        )
        self.assertEqual(set(bundle.complexity_summary["universe"]), {"frozen_order_top100"})
        self.assertIn("operator_count", bundle.top100_candidate_metrics.columns)
        self.assertIn("leaf_count", bundle.top100_candidate_metrics.columns)
        self.assertEqual(bundle.after_top20_correlation.shape, (1, 1))

    def test_universe_mismatch_fails_closed(self):
        bundle = _bundle()
        validation = []
        for row in bundle.validation_candidate_metrics.to_dict("records"):
            validation.append(row)
        with self.assertRaisesRegex(ValueError, "Train-pass set"):
            build_stage6_report_data(snapshot_manifest=bundle.snapshot_manifest, source_candidates=[_source(i) for i in range(4)], train_prefilter_results=bundle.train_prefilter_results.to_dict("records"), validation_records=[], hard_filter_results=[], greedy_results=[], alpha_pool=[], enrichment_results=[])

    def test_pair_contract_intersection_finite_minimum_and_zero_variance(self):
        dates = _dates(61)
        left = {date: float(index) for index, date in enumerate(dates)}
        right = {date: float(index * 2) for index, date in enumerate(dates[1:], start=1)}
        valid = pair_correlation(left, right)
        self.assertEqual(valid["status"], "valid")
        self.assertEqual(valid["common_valid_periods"], 60)
        short = pair_correlation(dict(list(left.items())[:59]), right)
        self.assertEqual(short["status"], "insufficient_common_periods")
        constant = pair_correlation({date: 1.0 for date in dates[:60]}, {date: float(i) for i, date in enumerate(dates[:60])})
        self.assertEqual(constant["status"], "correlation_unavailable")

    def test_pair_audit_preserves_persisted_status_and_nan_heatmap(self):
        bundle = _bundle(); audit = bundle.greedy_pair_audit.set_index("structural_hash")
        self.assertEqual(audit.loc[_hash(2), "persisted_decorrelation_status"], "rejected_by_correlation")
        self.assertEqual(int(audit.loc[_hash(2), "valid_pair_count"]), 1)
        self.assertAlmostEqual(float(audit.loc[_hash(2), "max_abs_valid_corr_to_previous_retained"]), 1.0)
        dates = _dates()
        matrix = _correlation_matrix(
            [_hash(1), _hash(3)],
            {
                _hash(1): {date: float(index) for index, date in enumerate(dates)},
                _hash(3): {date: 1.0 for date in dates},
            },
        )
        self.assertTrue(np.isnan(matrix.loc[_hash(3), _hash(1)]))

    def test_replay_audits_all_previous_retained_without_redeciding_status(self):
        dates = _dates()
        series = {
            _hash(1): {date: float(index) for index, date in enumerate(dates)},
            _hash(2): {date: float(np.sin(index)) for index, date in enumerate(dates)},
            _hash(3): {date: float(index * 2) for index, date in enumerate(dates)},
        }
        greedy = [
            {"sorted_rank": 1, "structural_hash": _hash(1), "decorrelation_status": "retained"},
            {"sorted_rank": 2, "structural_hash": _hash(2), "decorrelation_status": "retained"},
            {"sorted_rank": 3, "structural_hash": _hash(3), "decorrelation_status": "rejected_by_correlation", "blocked_by_structural_hash": _hash(1), "blocking_corr": 1.0},
        ]
        row = _greedy_audit(greedy, series).iloc[2]
        self.assertEqual(int(row["previous_retained_count"]), 2)
        self.assertEqual(int(row["valid_pair_count"]), 2)
        self.assertEqual(row["persisted_decorrelation_status"], "rejected_by_correlation")

    def test_structure_prevalence_is_unique_candidate_weighted(self):
        frame = pd.DataFrame([
            {"structural_hash": _hash(1), "prefix_token_ids": [get_action_id("ts_mean", 5), get_action_id("open")], "node_count": 2, "depth": 1},
            {"structural_hash": _hash(2), "prefix_token_ids": [get_action_id("ts_mean", 5), get_action_id("ts_mean", 5), get_action_id("close")], "node_count": 3, "depth": 2},
        ])
        _, operators, fields, windows = _token_prevalence(frame, "validation_evaluated")
        ts_mean = operators.set_index("operator").loc["ts_mean"]
        self.assertEqual(int(ts_mean["occurrence_count"]), 3)
        self.assertEqual(int(ts_mean["candidate_prevalence"]), 2)
        self.assertEqual(float(ts_mean["prevalence_ratio"]), 1.0)
        self.assertEqual(int(fields.set_index("field").loc["open", "candidate_prevalence"]), 1)
        self.assertEqual(int(windows.set_index("window").loc[5, "candidate_prevalence"]), 2)

    def test_top30_ranking_uses_abs_train_ic_then_hash(self):
        rows = {
            _hash(3): {"metrics": {"train_ic": -.04}},
            _hash(2): {"metrics": {"train_ic": .04}},
            _hash(1): {"metrics": {"train_ic": .03}},
        }
        self.assertEqual(_rank_hashes(reversed(list(rows)), rows), [_hash(2), _hash(3), _hash(1)])


class Stage6ReportingFormalGateTests(unittest.TestCase):
    def test_completed_single_hybrid_snapshot_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _formal_fixture(Path(temporary))
            bundle = load_stage6_report_data(**paths)
            self.assertEqual(bundle.funnel_summary["remaining_count"].tolist(), [1, 1, 1, 1, 1, 1, 1])

    def test_selection_oos_smoke_and_scope_fail_closed(self):
        for field, value, message in (
            ("oos", "loaded", "OOS lock"),
            ("engineering_smoke", True, "engineering smoke"),
            ("provisional_evaluation_universe", {"scope": "limited"}, "provisional_selection scope"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                paths = _formal_fixture(Path(temporary)); path = paths["selection_manifest_path"]
                manifest = json.loads(path.read_text(encoding="utf-8")); manifest[field] = value; _write_json(path, manifest)
                with self.assertRaisesRegex(ValueError, message):
                    load_stage6_report_data(**paths)

    def test_legacy_mixed_incomplete_and_snapshot_fingerprint_fail(self):
        cases = ("legacy", "mixed", "incomplete", "fingerprint")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                paths = _formal_fixture(Path(temporary)); source_path = paths["source_set_manifest_path"]
                source_set = json.loads(source_path.read_text(encoding="utf-8")); snapshot_path = Path(source_set["source_manifests"][0]["snapshot_manifest"]); snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if case == "legacy":
                    source_set["sources"][0]["source_type"] = "discovery_run"
                elif case == "mixed":
                    source_set["sources"].append({"source_id": "legacy", "source_type": "discovery_run", "source_role": "historical_discovery", "snapshot_fingerprint": "z" * 64})
                elif case == "incomplete":
                    snapshot["cutoff"]["complete"] = False
                else:
                    snapshot["snapshot_fingerprint"] = "0" * 64
                if case in {"legacy", "mixed"}:
                    core = {"schema": source_set["schema"], "mode": source_set["mode"], "sources": [{key: item.get(key) for key in ("source_id", "source_type", "source_role", "snapshot_fingerprint")} for item in source_set["sources"]]}; source_set["source_set_fingerprint"] = _stable_hash(core); _write_json(source_path, source_set)
                else:
                    _write_json(snapshot_path, snapshot)
                with self.assertRaises(ValueError):
                    load_stage6_report_data(**paths)

    def test_compatibility_loss_and_artifact_sha_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _formal_fixture(Path(temporary)); manifest_path = paths["candidate_import_manifest_path"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")); registry_path = manifest_path.parent / "candidate_registry.jsonl"
            rows = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()]
            second = dict(rows[0]); second["source_claimed_structural_hash"] = _hash(2); rows.append(second)
            manifest["artifacts"]["candidate_registry.jsonl"] = _write_jsonl(registry_path, rows); _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "compatibility universe"):
                load_stage6_report_data(**paths)
        with tempfile.TemporaryDirectory() as temporary:
            paths = _formal_fixture(Path(temporary)); registry = paths["candidate_import_manifest_path"].parent / "candidate_registry.jsonl"
            registry.write_text(registry.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                load_stage6_report_data(**paths)


class Stage6ReportingRendererTests(unittest.TestCase):
    def test_synthetic_bundle_exports_expected_png_csv_and_manifest(self):
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as temporary:
            renderer = Stage6ReportRenderer(bundle, Path(temporary) / "report")
            outputs = renderer.render_all()
            self.assertEqual(len(outputs["figures"]), 21)
            self.assertEqual(len(outputs["tables"]), 19)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs["figures"] + outputs["tables"]))
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["report_schema"], "factor_gfn.reporting.stage6_report.v2")
            self.assertEqual(manifest["report_version"], 2)
            self.assertEqual(manifest["candidate_universe_counts"]["provisional_pool"], 1)
            self.assertEqual(manifest["candidate_universe_counts"]["frozen_order_top100"], 1)
            self.assertIn("05_train_long_excess_corr_after_top20.png", [path.name for path in outputs["figures"]])
            self.assertNotIn("04_candidate_quality_summary.png", [path.name for path in outputs["figures"]])
            self.assertIn("06_top100_quality_summary.png", [path.name for path in outputs["figures"]])
            self.assertIn("06_top100_ic_distribution.png", [path.name for path in outputs["figures"]])
            self.assertIn("06_top100_long_ir_distribution.png", [path.name for path in outputs["figures"]])
            self.assertIn("06_top100_barra_corr_distribution.png", [path.name for path in outputs["figures"]])
            self.assertIn("06_complexity_summary.png", [path.name for path in outputs["figures"]])
            self.assertIn("07_top_candidate_examples.png", [path.name for path in outputs["figures"]])
            self.assertIn("05_after_top20_correlation_matrix.csv", [path.name for path in outputs["tables"]])
            self.assertIn("06_top100_candidate_metrics.csv", [path.name for path in outputs["tables"]])
            funnel = renderer.figure_candidate_screening_funnel()
            self.assertGreaterEqual(len(funnel.axes[0].patches), 7)
            plt.close(funnel)

    def test_notebook_is_unexecuted_and_has_required_sections(self):
        import nbformat
        path = Path(__file__).resolve().parents[1] / "notebooks" / "stage6_reporting.ipynb"
        notebook = nbformat.read(path, as_version=4)
        headings = ["".join(cell.source) for cell in notebook.cells if cell.cell_type == "markdown"]
        required = ["00 Parameters and Source Validation", "01 Screening Funnel", "02 Hard Filter Diagnostics", "03 Train → Validation Stability", "04 Quality Before / After", "05 Decorrelation", "06 Provisional Factor Pool and Frozen-order Top100", "07 Ranked Candidate Examples and Expression Structure Shift", "08 Export"]
        positions = [next(index for index, text in enumerate(headings) if heading in text) for heading in required]
        self.assertEqual(positions, sorted(positions))
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
        code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        self.assertNotIn("EvaluationStore", code)
        self.assertNotIn("pair_correlation", code)
        self.assertIn("stage6_reporting_v2", code)
        self.assertIn("figure_train_long_excess_corr_after_top20", code)
        self.assertNotIn("figure_candidate_quality_summary", code)
        self.assertIn("figure_top100_quality_summary", code)
        self.assertIn("figure_top100_ic_distribution", code)
        self.assertIn("figure_top100_long_ir_distribution", code)
        self.assertIn("figure_top100_barra_corr_distribution", code)
        self.assertIn("figure_top_candidate_examples", code)


if __name__ == "__main__":
    unittest.main()
