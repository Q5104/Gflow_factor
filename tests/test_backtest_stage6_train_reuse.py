from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from factor_gfn.backtest import stage6_train_reuse as reuse
from factor_gfn.backtest.candidate_import import CANDIDATE_IMPORT_MANIFEST_SCHEMA
from factor_gfn.backtest.expression_compatibility import (
    ACCEPTED_REGISTRY_SCHEMA,
    EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA,
)
from factor_gfn.backtest.sources import SOURCE_SET_SCHEMA, SOURCE_SNAPSHOT_SCHEMA


def _provider_manifest(schema: str = "factor_gfn.real_reward_provider.v8") -> dict:
    return {
        "schema": schema,
        "context_fingerprint": "context-fingerprint",
        "reward_evaluator_context_fingerprint": "reward-context-fingerprint",
        "evaluation_config": {
            "horizon": 5,
            "entry_lag": 1,
            "rebalance_interval": 5,
            "rebalance_offset": 0,
            "annualization": 50.4,
            "long_quantile": 0.1,
            "min_cross_section_count": 20,
            "performance_ddof": 1,
        },
        "reward_config": {
            "candidate_industry_neutralization": True,
            "barra_min_common_periods": 60,
        },
        "calendar": {
            "sha256": "calendar-fingerprint",
            "periods": 386,
            "first_date": "2011-01-18",
            "last_date": "2018-12-17",
        },
        "reward_panel": {
            "mode": "fixed_rebalance_compact",
            "history_rows_interpreted": 2187,
            "evaluation_rows": 386,
            "candidate_cleaning_calls_per_evaluation": 1,
        },
        "interpreter": {
            "numeric_kernel_schema": "factor_gfn.numba_cpu_loops.v2",
            "numba_kernel_schema": "factor_gfn.numba_cpu_loops.v2",
        },
        "industry_neutralization": {
            "enabled": True,
            "policy_schema": "factor_gfn.strict_industry_neutralization.v1",
            "encoding_schema": "factor_gfn.point_in_time_industry_codes.v1",
            "projection": "group_mean_equivalent_to_intercept_plus_full_dummies",
            "failed_date_action": "exclude_entire_candidate_cross_section",
            "unknown_industry_stock_action": "exclude_stock",
            "calendar_action": "keep_global_phase_without_backfill",
        },
    }


def _metrics(seed: float) -> dict:
    return {
        "train_ic": seed,
        "train_long_ir": 0.5 + seed,
        "barra_ts_corr": 0.2,
        "barra_correlations": {
            "market_beta": 0.1,
            "size": -0.2,
            "momentum": 0.03,
            "volatility": -0.04,
            "liquidity": 0.05,
        },
        "barra_valid_periods": {
            "market_beta": 386,
            "size": 386,
            "momentum": 386,
            "volatility": 386,
            "liquidity": 386,
        },
        "long_direction": 1,
        "ic_valid_periods": 386,
        "long_ir_valid_periods": 386,
        "industry_neutralized": True,
        "neutralization_skipped_dates": [],
        "neutralization_skipped_rate": 0.0,
        "neutralization_skipped_details": [],
    }


class _FakeEvaluator:
    def __init__(self, metrics_by_hash: dict[str, dict], accepted: str, audit: str) -> None:
        self.metrics_by_hash = metrics_by_hash
        self.accepted_registry_fingerprint = accepted
        self.compatibility_audit_fingerprint = audit
        self.evaluation_contract_fingerprint = "evaluation-contract"
        self.context = SimpleNamespace(fingerprint="stage6-context")

    def evaluate(self, candidate: dict) -> SimpleNamespace:
        metrics = self.metrics_by_hash[candidate["current_structural_hash"]]
        return SimpleNamespace(
            train_direction=metrics["train_direction"],
            train={
                "ic": {
                    "mean": metrics["train_ic"],
                    "valid_periods": metrics["train_ic_valid_periods"],
                },
                "long": {
                    "annualized_ir": metrics["train_long_ir"],
                    "valid_periods": metrics["train_long_valid_periods"],
                },
                "barra": {
                    "max_abs_correlation": metrics["train_barra_ts_corr"],
                    "correlations": metrics["train_barra_correlations"],
                    "common_valid_periods": metrics[
                        "train_barra_valid_periods_by_style"
                    ],
                },
                "neutralization": {
                    "skipped_dates": metrics["neutralization"]["skipped_dates"],
                    "skipped_rate": metrics["neutralization"]["skipped_rate"],
                    "details": metrics["neutralization"]["details"],
                },
            },
            result_fingerprint="result-" + candidate["current_structural_hash"],
            total_seconds=0.01,
        )


class Stage6TrainReuseTest(unittest.TestCase):
    def test_static_gate_accepts_v7_and_rejects_kernel_mismatch(self) -> None:
        target = _provider_manifest()
        target_fp = reuse._stable_hash(target)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = _provider_manifest("factor_gfn.real_reward_provider.v7")
            bad = _provider_manifest("factor_gfn.real_reward_provider.v6")
            bad["interpreter"]["numeric_kernel_schema"] = "old-kernel"
            bad["interpreter"]["numba_kernel_schema"] = "old-kernel"
            snapshots = []
            for index, provider in enumerate((good, bad)):
                source = root / f"source-{index}"
                source.mkdir()
                fingerprint = reuse._stable_hash(provider)
                (source / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "reward_provider": provider,
                            "reward_provider_fingerprint": fingerprint,
                        }
                    ),
                    encoding="utf-8",
                )
                snapshots.append(
                    {
                        "source_id": f"source-{index}",
                        "source_semantics": {
                            "provider_fingerprint": fingerprint,
                            "context_fingerprint": "context-fingerprint",
                        },
                        "_directory": source,
                    }
                )
            audits = reuse._static_audit_batches(
                snapshots,
                target_provider_manifest=target,
                target_provider_fingerprint=target_fp,
            )
        statuses = {row["source_ids"][0]: row for row in audits}
        self.assertEqual(
            statuses["source-0"]["status"],
            reuse.TRAIN_REUSE_NUMERIC_VERIFICATION_REQUIRED,
        )
        self.assertEqual(
            statuses["source-1"]["status"], reuse.TRAIN_REUSE_NOT_ALLOWED
        )
        self.assertIn(
            "provider_schema_not_proven_compatible",
            statuses["source-1"]["reason_codes"],
        )

    def test_representative_selection_is_deterministic_and_structural(self) -> None:
        candidates = [
            {
                "current_structural_hash": f"{index:064x}",
                "prefix_token_ids": prefix,
                "node_count": len(prefix),
                "depth": min(index, 6),
            }
            for index, prefix in enumerate(
                ([2], [13, 2], [18, 1, 2], [52, 3], [36, 0], [122, 0, 1]),
                start=1,
            )
        ]
        first = reuse.select_representative_candidates(candidates, limit=4)
        second = reuse.select_representative_candidates(list(reversed(candidates)), limit=4)
        self.assertEqual(
            [row["current_structural_hash"] for row in first],
            [row["current_structural_hash"] for row in second],
        )
        tags = set().union(*(reuse._candidate_tags(row) for row in first))
        self.assertIn("arity:unary", tags)
        self.assertIn("arity:binary", tags)

    def test_legacy_boolean_industry_identity_fails_closed(self) -> None:
        target = _provider_manifest()
        legacy = deepcopy(target)
        legacy["schema"] = "factor_gfn.real_reward_provider.v1"
        legacy["industry_neutralization"] = True
        projection = reuse._provider_projection(legacy)
        self.assertIsNone(projection["industry_policy"]["policy_schema"])
        self.assertNotEqual(projection, reuse._provider_projection(target))

    def test_end_to_end_overlay_is_immutable_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = _provider_manifest()
            provider_fp = reuse._stable_hash(provider)
            source_dir = root / "sources" / "source-a"
            source_dir.mkdir(parents=True)
            candidates = [
                {
                    "schema": ACCEPTED_REGISTRY_SCHEMA,
                    "current_structural_hash": f"{index:064x}",
                    "source_claimed_structural_hash": f"{index:064x}",
                    "formula": "low" if index == 1 else "relu(low)",
                    "prefix_token_ids": [2] if index == 1 else [13, 2],
                    "node_count": index,
                    "depth": index - 1,
                    "origin_ids": [f"origin-{index}"],
                    "source_ids": ["source-a"],
                    "compatibility_record_fingerprint": f"compat-{index}",
                    "historical_metric_reuse": "forbidden",
                    "stage6_metric_recompute_required": True,
                }
                for index in (1, 2)
            ]
            old_metrics = {candidate["current_structural_hash"]: _metrics(0.01 * index) for index, candidate in enumerate(candidates, start=1)}
            evaluations = source_dir / "evaluations.jsonl"
            with evaluations.open("w", encoding="utf-8") as stream:
                for candidate in candidates:
                    stream.write(
                        json.dumps(
                            {
                                "structural_hash": candidate["current_structural_hash"],
                                "metadata": {
                                    "reward_result": old_metrics[
                                        candidate["current_structural_hash"]
                                    ]
                                },
                            }
                        )
                        + "\n"
                    )
            metadata = source_dir / "run_metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "reward_provider": provider,
                        "reward_provider_fingerprint": provider_fp,
                    }
                ),
                encoding="utf-8",
            )
            artifacts = [
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": reuse._sha256_file(path),
                }
                for path in (evaluations, metadata)
            ]
            snapshot = {
                "schema": SOURCE_SNAPSHOT_SCHEMA,
                "source_id": "source-a",
                "source_type": "discovery_run",
                "source_role": "historical_discovery",
                "inclusion_status": "approved",
                "approval_note": "test",
                "candidate_record_policy": {},
                "source_semantics": {
                    "provider_fingerprint": provider_fp,
                    "context_fingerprint": "context-fingerprint",
                },
                "snapshot_kind": "stable_full_jsonl",
                "cutoff": {},
                "record_counts": {"records": 2},
                "artifacts": artifacts,
                "logical_content_fingerprint": "logical",
            }
            snapshot_fp = reuse._stable_hash(reuse._snapshot_fingerprint_payload(snapshot))
            final_source_dir = root / "snapshot" / snapshot_fp
            final_source_dir.parent.mkdir()
            source_dir.rename(final_source_dir)
            snapshot["snapshot_fingerprint"] = snapshot_fp
            snapshot_path = final_source_dir / "source_snapshot.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

            source_set = {
                "schema": SOURCE_SET_SCHEMA,
                "mode": "provisional",
                "sources": [
                    {
                        "source_id": "source-a",
                        "source_type": "discovery_run",
                        "source_role": "historical_discovery",
                        "snapshot_fingerprint": snapshot_fp,
                    }
                ],
            }
            source_set_fp = reuse._stable_hash(reuse._source_set_payload(source_set))
            source_set["source_set_fingerprint"] = source_set_fp
            source_set["source_manifests"] = [
                {
                    "source_id": "source-a",
                    "snapshot_fingerprint": snapshot_fp,
                    "snapshot_manifest": str(snapshot_path),
                }
            ]
            source_set_dir = root / "source_sets" / source_set_fp
            source_set_dir.mkdir(parents=True)
            source_set_path = source_set_dir / "source_set_manifest.json"
            source_set_path.write_text(json.dumps(source_set), encoding="utf-8")

            candidate_dir = root / "candidate" / "registry-fingerprint"
            candidate_dir.mkdir(parents=True)
            candidate_manifest = {
                "schema": CANDIDATE_IMPORT_MANIFEST_SCHEMA,
                "registry_fingerprint": "registry-fingerprint",
                "source_set_fingerprint": source_set_fp,
            }
            candidate_path = candidate_dir / "candidate_import_manifest.json"
            candidate_path.write_text(json.dumps(candidate_manifest), encoding="utf-8")

            audit_fp = "a" * 64
            compatibility_dir = root / "compatibility" / audit_fp
            compatibility_dir.mkdir(parents=True)
            accepted_path = compatibility_dir / "auto_accepted_candidate_registry.jsonl"
            with accepted_path.open("w", encoding="utf-8") as stream:
                for candidate in candidates:
                    stream.write(json.dumps(candidate) + "\n")
            accepted_digest = reuse._stable_hash(candidates)
            compatibility_manifest = {
                "schema": EXPRESSION_COMPATIBILITY_MANIFEST_SCHEMA,
                "candidate_registry_fingerprint": "registry-fingerprint",
                "source_set_fingerprint": source_set_fp,
                "counts": {"AUTO_ACCEPT": 2},
                "digests": {"accepted_registry": accepted_digest},
                "artifacts": {
                    "auto_accepted_candidate_registry.jsonl": {
                        "size_bytes": accepted_path.stat().st_size,
                        "sha256": reuse._sha256_file(accepted_path),
                    }
                },
            }
            compatibility_path = compatibility_dir / "expression_compatibility_manifest.json"
            compatibility_path.write_text(
                json.dumps(compatibility_manifest), encoding="utf-8"
            )
            accepted_sha_before = reuse._sha256_file(accepted_path)
            extracted = {
                structural_hash: reuse._extract_metrics(metrics)
                for structural_hash, metrics in old_metrics.items()
            }
            evaluator = _FakeEvaluator(extracted, accepted_digest, audit_fp)
            first = reuse.run_stage6_train_reuse_audit(
                source_set_manifest_path=source_set_path,
                candidate_import_manifest_path=candidate_path,
                compatibility_manifest_path=compatibility_path,
                evaluator=evaluator,
                target_provider_manifest=provider,
                target_provider_fingerprint=provider_fp,
                output_root=root / "overlays",
            )
            second = reuse.run_stage6_train_reuse_audit(
                source_set_manifest_path=source_set_path,
                candidate_import_manifest_path=candidate_path,
                compatibility_manifest_path=compatibility_path,
                evaluator=evaluator,
                target_provider_manifest=provider,
                target_provider_fingerprint=provider_fp,
                output_root=root / "overlays",
            )
            manifest = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(first, second)
            self.assertEqual(manifest["counts"]["overlay_candidates"], 2)
            self.assertEqual(manifest["counts"]["numeric_verified_batches"], 1)
            verified_manifest, overlay_hashes = reuse._verify_existing_overlay(first)
            self.assertEqual(verified_manifest, manifest)
            self.assertEqual(overlay_hashes, set(old_metrics))
            self.assertEqual(reuse._sha256_file(accepted_path), accepted_sha_before)

    def test_v6_projection_evidence_allows_only_declared_implementation_fields(self) -> None:
        target = _provider_manifest()
        source = deepcopy(target)
        source["schema"] = "factor_gfn.real_reward_provider.v6"
        source["interpreter"]["numeric_kernel_schema"] = "factor_gfn.rolling_moments.v1"
        source["interpreter"]["numba_kernel_schema"] = "factor_gfn.numba_ts_loops.v1"
        source["industry_neutralization"].pop("encoding_schema")
        source["industry_neutralization"].pop("projection")

        evidence = reuse._projection_difference_evidence(
            reuse._provider_projection(source), reuse._provider_projection(target)
        )
        self.assertTrue(evidence["all_differences_are_declared_implementation_fields"])
        self.assertEqual(
            set(evidence["differences"]), reuse._V6_EXPECTED_PROJECTION_DIFFERENCES
        )

        source["industry_neutralization"]["failed_date_action"] = "keep_failed_date"
        evidence = reuse._projection_difference_evidence(
            reuse._provider_projection(source), reuse._provider_projection(target)
        )
        self.assertFalse(evidence["all_differences_are_declared_implementation_fields"])
        self.assertEqual(
            evidence["unexpected_difference_fields"],
            ["industry_policy.failed_date_action"],
        )

    def test_v6_equivalence_comparison_includes_neutralization_details(self) -> None:
        stage5 = reuse._extract_metrics(_metrics(0.01))
        self.assertIsNotNone(stage5)
        stage6 = deepcopy(stage5)
        stage5["neutralization"]["details"] = [
            {
                "date": "2011-07-04",
                "row_index": 362,
                "factor_valid_count": 8,
                "known_industry_count": 8,
                "industry_count": 8,
                "required_regression_count": 9,
                "reason": "insufficient_industry_regression_samples",
            }
        ]
        stage6["neutralization"]["details"] = [
            {
                "compact_row": 22,
                "date": "2011-07-04",
                "global_row": 362,
                "factor_valid_count": 8,
                "known_industry_count": 8,
                "industry_count": 8,
                "required_regression_count": 9,
                "reason": "insufficient_industry_regression_samples",
            }
        ]
        passed, comparisons, details = reuse._compare_v6_equivalence_metrics(
            stage5, stage6
        )
        self.assertTrue(passed)
        self.assertTrue(all(row["pass"] for row in comparisons.values()))
        self.assertTrue(details["pass"])
        self.assertFalse(details["raw_exact_pass"])

        stage6["neutralization"]["details"][0]["reason"] = "different"
        passed, comparisons, details = reuse._compare_v6_equivalence_metrics(
            stage5, stage6
        )
        self.assertFalse(passed)
        self.assertTrue(all(row["pass"] for row in comparisons.values()))
        self.assertFalse(details["pass"])


if __name__ == "__main__":
    unittest.main()
