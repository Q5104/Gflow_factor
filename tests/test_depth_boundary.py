import csv
import json
import tempfile
import unittest
from pathlib import Path

from factor_gfn.gfn.depth_boundary import (
    DEPTH_BOUNDARY_DIAGNOSTIC_SCHEMA,
    DepthBoundaryDiagnosticConfig,
    build_depth_boundary_diagnostic,
    write_depth_boundary_outputs,
)


def _record(
    key: str,
    *,
    node_count: int = 12,
    depth: int,
    valid: bool = True,
    reward: float | None = 1.0,
    train_ic: float = 0.1,
    factor_seconds: float = 2.0,
    reward_seconds: float = 1.0,
    cache_hit: bool = False,
    source: str = "discovery",
) -> dict:
    return {
        "source": source,
        "structural_hash": key,
        "node_count": node_count,
        "depth": depth,
        "valid": valid,
        "reward": reward if valid else None,
        "rejection_reason": None if valid else "invalid",
        "metadata": {
            "provider_cache_hit": cache_hit,
            "factor_seconds": factor_seconds,
            "reward_seconds": reward_seconds,
            "reward_result": {"train_ic": train_ic},
        },
    }


def _small_config(**overrides) -> DepthBoundaryDiagnosticConfig:
    values = {
        "minimum_total_unique_candidates": 6,
        "minimum_unique_candidates_per_focus_depth": 2,
        "minimum_valid_quality_samples_per_focus_depth": 2,
        "consider_expansion_min_boundary_share": 0.20,
        "no_expansion_max_boundary_share": 0.05,
        "valid_rate_non_degradation_tolerance": 0.05,
        "valid_rate_clear_decline": 0.10,
        "quality_relative_non_degradation_tolerance": 0.10,
        "quality_relative_clear_decline": 0.15,
    }
    values.update(overrides)
    return DepthBoundaryDiagnosticConfig(**values)


def _provenance() -> dict:
    return {
        "config_fingerprint": "config-fingerprint",
        "provider_fingerprint": "provider-fingerprint",
        "context_fingerprint": "context-fingerprint",
        "provider_manifest": {
            "data_scope": "training_only",
            "validation_oos_loaded": False,
            "context_fingerprint": "context-fingerprint",
        },
    }


class DepthBoundaryConfigTests(unittest.TestCase):
    def test_conservative_defaults_and_threshold_ordering(self):
        config = DepthBoundaryDiagnosticConfig()
        self.assertEqual(config.minimum_total_unique_candidates, 500)
        self.assertEqual(config.minimum_unique_candidates_per_focus_depth, 100)
        self.assertEqual(config.minimum_valid_quality_samples_per_focus_depth, 100)
        self.assertEqual(config.consider_expansion_min_boundary_share, 0.10)
        self.assertEqual(config.no_expansion_max_boundary_share, 0.03)
        with self.assertRaisesRegex(ValueError, "boundary share"):
            DepthBoundaryDiagnosticConfig(
                consider_expansion_min_boundary_share=0.1,
                no_expansion_max_boundary_share=0.2,
            )
        with self.assertRaisesRegex(ValueError, "valid-rate"):
            DepthBoundaryDiagnosticConfig(
                valid_rate_non_degradation_tolerance=0.2,
                valid_rate_clear_decline=0.1,
            )


class DepthBoundaryAggregationTests(unittest.TestCase):
    def test_all_depths_deduplicate_and_keep_invalid_timing(self):
        records = [
            _record("a", node_count=10, depth=6, reward=1.0, train_ic=-0.10),
            _record("a", node_count=10, depth=6, reward=1.0, train_ic=-0.10, cache_hit=True),
            _record("b", node_count=10, depth=6, valid=False, reward=None, factor_seconds=4.0),
            _record("c", node_count=12, depth=8, reward=3.0, train_ic=-0.30),
            _record("ignored", depth=8, source="anchor"),
        ]
        result = build_depth_boundary_diagnostic(
            records,
            max_depth=8,
            max_nodes=24,
            discovery_node_counts=(10, 12),
            **_provenance(),
            config=_small_config(minimum_total_unique_candidates=10),
        )
        self.assertEqual(len(result.candidate_audit), 3)
        self.assertEqual(result.summary["focus_depths"], (6, 7, 8))
        self.assertEqual(result.summary["max_nodes_boundary_diagnostic"], "disabled_by_design")
        self.assertEqual(len(result.depth_metrics), (2 + 1) * 9)
        row = next(
            item
            for item in result.depth_metrics
            if item["scope"] == "by_node_count"
            and item["node_count"] == 10
            and item["depth"] == 6
        )
        self.assertEqual(row["unique_candidate_count"], 2)
        self.assertEqual(row["valid_count"], 1)
        self.assertEqual(row["finite_reward_count"], 1)
        self.assertEqual(row["finite_abs_ic_count"], 1)
        self.assertEqual(row["abs_ic_p90"], 0.10)
        self.assertEqual(row["evaluation_seconds_mean"], 4.0)
        empty = next(
            item
            for item in result.depth_metrics
            if item["scope"] == "overall" and item["depth"] == 0
        )
        self.assertEqual(empty["unique_candidate_count"], 0)
        self.assertIsNone(empty["valid_rate"])
        self.assertIsNone(empty["reward_p90"])

    def test_conflicting_duplicate_fails_closed(self):
        records = [
            _record("same", depth=6, reward=1.0),
            _record("same", depth=7, reward=1.0),
        ]
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            build_depth_boundary_diagnostic(
                records,
                max_depth=8,
                max_nodes=24,
                discovery_node_counts=(12,),
                **_provenance(),
                config=_small_config(),
            )

    def test_every_unique_candidate_requires_evaluation_timing(self):
        record = _record("missing", depth=6)
        del record["metadata"]["factor_seconds"]
        with self.assertRaisesRegex(ValueError, "timings"):
            build_depth_boundary_diagnostic(
                [record],
                max_depth=8,
                max_nodes=24,
                discovery_node_counts=(12,),
                **_provenance(),
                config=_small_config(),
            )

    def test_source_and_training_only_provenance_fail_closed(self):
        missing_source = _record("missing-source", depth=6)
        del missing_source["source"]
        with self.assertRaisesRegex(ValueError, "explicit source"):
            build_depth_boundary_diagnostic(
                [missing_source],
                max_depth=8,
                max_nodes=24,
                discovery_node_counts=(12,),
                **_provenance(),
                config=_small_config(),
            )
        provenance = _provenance()
        provenance["provider_manifest"]["validation_oos_loaded"] = True
        with self.assertRaisesRegex(ValueError, "validation_oos_loaded=False"):
            build_depth_boundary_diagnostic(
                [_record("oos", depth=6)],
                max_depth=8,
                max_nodes=24,
                discovery_node_counts=(12,),
                **provenance,
                config=_small_config(),
            )


class DepthBoundaryRecommendationTests(unittest.TestCase):
    @staticmethod
    def _focus_records(boundary_reward: float = 12.0, boundary_ic: float = 0.12):
        records = []
        for depth, reward, ic in ((6, 10.0, 0.10), (7, 11.0, 0.11), (8, boundary_reward, boundary_ic)):
            for index in range(2):
                records.append(
                    _record(
                        f"{depth}-{index}",
                        depth=depth,
                        reward=reward + index,
                        train_ic=ic + index * 0.001,
                    )
                )
        return records

    def test_consider_expansion_when_boundary_is_common_and_not_degraded(self):
        result = build_depth_boundary_diagnostic(
            self._focus_records(),
            max_depth=8,
            max_nodes=24,
            discovery_node_counts=(12,),
            **_provenance(),
            config=_small_config(),
        )
        self.assertEqual(result.summary["recommendation"], "consider_expansion")
        self.assertTrue(result.summary["sample_sufficiency"]["sufficient"])
        self.assertFalse(result.summary["automatic_boundary_change"])
        self.assertFalse(result.summary["automatic_run_creation"])

    def test_no_expansion_when_boundary_quality_clearly_declines(self):
        result = build_depth_boundary_diagnostic(
            self._focus_records(boundary_reward=5.0, boundary_ic=0.05),
            max_depth=8,
            max_nodes=24,
            discovery_node_counts=(12,),
            **_provenance(),
            config=_small_config(),
        )
        self.assertEqual(result.summary["recommendation"], "no_expansion_evidence")
        self.assertIn("both show clear decline", result.summary["reasons"][0])

    def test_rare_boundary_is_no_expansion_evidence_after_total_sample_gate(self):
        records = [
            _record(f"d6-{index}", depth=6, reward=1.0 + index, train_ic=0.1)
            for index in range(6)
        ]
        result = build_depth_boundary_diagnostic(
            records,
            max_depth=8,
            max_nodes=24,
            discovery_node_counts=(12,),
            **_provenance(),
            config=_small_config(),
        )
        self.assertEqual(result.summary["recommendation"], "no_expansion_evidence")
        self.assertEqual(result.summary["overall_max_depth_share"], 0.0)

    def test_sample_shortfall_is_insufficient_evidence(self):
        result = build_depth_boundary_diagnostic(
            [_record("one", depth=8)],
            max_depth=8,
            max_nodes=24,
            discovery_node_counts=(12,),
            **_provenance(),
            config=_small_config(),
        )
        self.assertEqual(result.summary["recommendation"], "insufficient_evidence")
        self.assertEqual(
            result.summary["depth_boundary_status"],
            result.summary["recommendation"],
        )
        self.assertEqual(
            result.summary["unique_discovery_candidate_count"],
            result.summary["total_unique_discovery_candidates"],
        )
        self.assertTrue(any("total_unique_candidates" in item for item in result.summary["reasons"]))


class DepthBoundaryOutputTests(unittest.TestCase):
    def test_writer_emits_exactly_three_auditable_files(self):
        result = build_depth_boundary_diagnostic(
            DepthBoundaryRecommendationTests._focus_records(),
            max_depth=8,
            max_nodes=24,
            discovery_node_counts=(12,),
            **_provenance(),
            config=_small_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_depth_boundary_outputs(result, directory)
            self.assertEqual(
                {path.name for path in paths},
                {
                    "depth_boundary_summary.json",
                    "depth_metrics.csv",
                    "depth_candidate_audit.csv",
                },
            )
            self.assertEqual(set(Path(directory).iterdir()), set(paths))
            summary = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["schema"], DEPTH_BOUNDARY_DIAGNOSTIC_SCHEMA)
            with paths[1].open(encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))
            with paths[2].open(encoding="utf-8") as handle:
                audit = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 18)
            self.assertEqual(len(audit), 6)

    def test_writer_keeps_headers_for_empty_candidate_audit(self):
        result = build_depth_boundary_diagnostic(
            [_record("not-discovery", depth=8, source="anchor")],
            max_depth=8,
            max_nodes=24,
            discovery_node_counts=(12,),
            **_provenance(),
            config=_small_config(),
        )
        self.assertEqual(result.summary["recommendation"], "insufficient_evidence")
        with tempfile.TemporaryDirectory() as directory:
            _, _, audit_path = write_depth_boundary_outputs(result, directory)
            with audit_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
