import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from factor_gfn.backtest.candidate_import import import_candidate_source_set
from factor_gfn.backtest.expression_compatibility import (
    ACCEPTED_REGISTRY_SCHEMA,
    audit_expression_compatibility,
)
from factor_gfn.backtest.sources import (
    CandidateSourceSpec,
    materialize_source_set,
)
from factor_gfn.backtest.stage6_evaluation import (
    Stage6CandidateEvaluator,
    Stage6EvaluationConfig,
    _sha256_file,
    _stable_hash,
    build_stage6_evaluation_context_from_arrays,
)
from factor_gfn.backtest.stage6_evaluation_store import (
    EvaluationStore,
    Stage6EvaluationRunner,
    evaluation_cache_identity,
)
from factor_gfn.backtest.stage6_mixed_evaluation import (
    TRAIN_METRIC_ORIGIN_FRESH,
    TRAIN_METRIC_ORIGIN_REUSE,
    Stage6MixedCandidateEvaluator,
    Stage6TrainReuseOverlay,
    TrainReuseOverlayIntegrityError,
    select_stage6_mixed_smoke_candidates,
)
from factor_gfn.backtest.stage6_prefilter_evaluation import (
    PRIOR_FULL_RESULT_REUSE,
    TRAIN_PREFILTER_FAILED,
    TRAIN_PREFILTER_PASSED,
    Stage6TrainPrefilterEvaluator,
    train_prefilter_decision,
)
from factor_gfn.backtest.stage6_train_reuse import (
    HYBRID_TRAIN_REUSE_ADAPTER_VERSION,
    HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA,
    HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA,
    TRAIN_REUSE_AUDITOR_VERSION,
    TRAIN_REUSE_MANIFEST_SCHEMA,
    TRAIN_REUSE_OVERLAY_SCHEMA,
    run_stage6_hybrid_train_reuse_overlay,
)
from factor_gfn.backtest.stage6_survivor_enrichment import (
    Stage6TrainLongExcessEnricher,
    run_stage6_survivor_enrichment_selection,
)
from factor_gfn.backtest.stage6_two_phase_pipeline import (
    TRAIN_NOT_EVALUATED_VALIDATION,
    Stage6TrainPreparationEvaluator,
    Stage6ValidationFromFrozenTrainEvaluator,
    _freeze_train_pass_manifest,
)
from factor_gfn.barra import STYLE_NAMES, BarraConfig
from factor_gfn.evaluator import EvaluationConfig
from factor_gfn.grammar import Expression, get_action_id


class Stage6MixedEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-03-02"),
            dtype="datetime64[D]",
        )
        stocks = np.asarray([f"S{index:02d}" for index in range(8)])
        day = np.arange(dates.size, dtype=np.float64)[:, None]
        stock = np.arange(stocks.size, dtype=np.float64)[None, :]
        tensor = np.empty((dates.size, 6, stocks.size), dtype=np.float64)
        tensor[:, 0, :] = 10.0 + stock + day * (0.05 + 0.004 * stock**2)
        for feature in range(1, 6):
            tensor[:, feature, :] = 20.0 + feature + 0.1 * day + (feature + 1) * stock
        universe = np.ones((dates.size, stocks.size), dtype=bool)
        industry = np.tile(np.repeat(np.arange(4, dtype=np.int32), 2), (dates.size, 1))
        exposures = {
            name: np.broadcast_to(
                stock + 0.01 * day * (index + 1), (dates.size, stocks.size)
            ).copy()
            for index, name in enumerate(STYLE_NAMES)
        }
        evaluation = EvaluationConfig(min_cross_section_count=4, long_quantile=0.25)
        barra = BarraConfig(
            beta_window=3,
            beta_min_periods=2,
            momentum_lookback=3,
            momentum_skip=1,
            volatility_window=3,
            volatility_min_periods=2,
            liquidity_window=2,
            long_short_quantile=0.25,
            min_cross_section_count=4,
            min_common_periods=2,
        )
        self.context = build_stage6_evaluation_context_from_arrays(
            dates=dates,
            stocks=stocks,
            factor_tensor=tensor,
            universe_mask=universe,
            industry_labels=industry,
            barra_exposures=exposures,
            config=Stage6EvaluationConfig(
                train_start="2020-01-01",
                train_end="2020-01-31",
                validation_start="2020-02-01",
                validation_end="2020-02-29",
                evaluation=evaluation,
                barra=barra,
            ),
            source_manifest={"fixture": "mixed-v1"},
        )
        self.fresh = Stage6CandidateEvaluator(
            self.context,
            compatibility_audit_fingerprint="a" * 64,
            accepted_registry_fingerprint="b" * 64,
        )
        expression = Expression.from_prefix([get_action_id("close")])
        self.candidate = {
            "schema": ACCEPTED_REGISTRY_SCHEMA,
            "current_structural_hash": expression.structural_hash(),
            "source_claimed_structural_hash": expression.structural_hash(),
            "formula": expression.to_formula(),
            "prefix_token_ids": list(expression.to_prefix()),
            "node_count": expression.stats.node_count,
            "depth": expression.stats.depth,
            "origin_ids": ["origin-1"],
            "source_ids": ["source-1"],
            "compatibility_record_fingerprint": "c" * 64,
            "historical_metric_reuse": "forbidden",
            "stage6_metric_recompute_required": True,
        }
        factor = np.zeros((self.context.dates.size, self.context.stocks.size))
        for name in ("train", "validation"):
            split = self.context.get_split_data(name)
            factor[split.global_rebalance_rows] = split.forward_returns
        self.factor = factor
        with patch.object(self.fresh._interpreter, "evaluate", return_value=factor):
            self.fresh_result = self.fresh.evaluate(self.candidate)

    def _metrics(self) -> dict:
        train = self.fresh_result.train
        return {
            "train_ic": train["ic"]["mean"],
            "train_ic_valid_periods": train["ic"]["valid_periods"],
            "train_direction": self.fresh_result.train_direction,
            "train_long_ir": train["long"]["annualized_ir"],
            "train_long_valid_periods": train["long"]["valid_periods"],
            "train_barra_ts_corr": train["barra"]["max_abs_correlation"],
            "train_barra_correlations": dict(train["barra"]["correlations"]),
            "train_barra_valid_periods_by_style": dict(
                train["barra"]["common_valid_periods"]
            ),
            "neutralization": dict(train["neutralization"]),
        }

    def _write_overlay(
        self,
        root: Path,
        *,
        include_candidate: bool = True,
        metric_overrides: dict | None = None,
    ) -> Path:
        rows = []
        if include_candidate:
            metrics = {**self._metrics(), **dict(metric_overrides or {})}
            payload = {
                "schema": TRAIN_REUSE_OVERLAY_SCHEMA,
                "structural_hash": self.candidate["current_structural_hash"],
                "train_metric_origin": TRAIN_METRIC_ORIGIN_REUSE,
                "train_metrics": metrics,
                "origins": [
                    {
                        "batch_id": "batch-1",
                        "source_id": "source-1",
                        "snapshot_fingerprint": "d" * 64,
                        "locator": {"artifact": "evaluations.jsonl", "line_number": 1},
                    }
                ],
            }
            rows.append({**payload, "record_fingerprint": _stable_hash(payload)})
        payload = {
            "schema": TRAIN_REUSE_MANIFEST_SCHEMA,
            "auditor_version": TRAIN_REUSE_AUDITOR_VERSION,
            "source_set_fingerprint": "1" * 64,
            "candidate_registry_fingerprint": "2" * 64,
            "compatibility_audit_fingerprint": "a" * 64,
            "accepted_registry_fingerprint": "b" * 64,
            "stage6_context_fingerprint": self.context.fingerprint,
            "stage6_evaluation_contract_fingerprint": (
                self.fresh.evaluation_contract_fingerprint
            ),
            "target_provider_fingerprint": "3" * 64,
            "target_train_contract_projection_fingerprint": "4" * 64,
            "source_audit_digest": "5" * 64,
            "numeric_verification_digest": "6" * 64,
            "overlay_digest": _stable_hash(rows),
        }
        fingerprint = _stable_hash(payload)
        directory = root / fingerprint
        directory.mkdir(parents=True)
        overlay_path = directory / "train_reuse_overlay.jsonl"
        overlay_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = {
            **payload,
            "train_reuse_overlay_fingerprint": fingerprint,
            "counts": {"overlay_candidates": len(rows), "accepted_candidates": 1},
            "artifacts": {
                "train_reuse_overlay.jsonl": {
                    "size_bytes": overlay_path.stat().st_size,
                    "sha256": _sha256_file(overlay_path),
                }
            },
        }
        manifest_path = directory / "train_reuse_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def _write_hybrid_overlay(
        self, root: Path, *, include_long_excess: bool = True
    ) -> Path:
        contract_fingerprint = "7" * 64
        fresh_series = self.fresh_result.train["long"]["excess_series"]
        long_excess = (
            {
                "dates": list(fresh_series["dates"]),
                "values": list(fresh_series["values"]),
                "direction": self.fresh_result.train_direction,
                "valid_periods": self.fresh_result.train["long"]["valid_periods"],
                "finite_periods": sum(
                    value is not None for value in fresh_series["values"]
                ),
                "total_periods": len(fresh_series["dates"]),
                "origin": "stage5_hybrid_train_artifact_reuse",
            }
            if include_long_excess
            else None
        )
        row_payload = {
            "schema": HYBRID_TRAIN_REUSE_OVERLAY_SCHEMA,
            "structural_hash": self.candidate["current_structural_hash"],
            "node_count": self.candidate["node_count"],
            "train_metric_origin": TRAIN_METRIC_ORIGIN_REUSE,
            "verification_mode": "hybrid_exact_contract",
            "train_evaluation_contract_fingerprint": contract_fingerprint,
            "train_metrics": self._metrics(),
            "train_long_excess": long_excess,
            "origins": [
                {
                    "verification_mode": "hybrid_exact_contract",
                    "source_id": "hybrid-source",
                    "snapshot_fingerprint": "8" * 64,
                    "locator": {
                        "artifact": "train_candidate_artifact.json",
                        "record_index": 0,
                    },
                    "source_record_fingerprint": "9" * 64,
                }
            ],
        }
        rows = [{**row_payload, "record_fingerprint": _stable_hash(row_payload)}]
        verification = {
            "verification_mode": "hybrid_exact_contract",
            "numeric_verification": "not_required_by_approved_hybrid_contract",
            "source_id": "hybrid-source",
            "old_train_metrics_allowed": True,
            "fresh_train_fallback_reason": None,
            "contract_difference_paths": [],
            "result": "TRAIN_METRICS_REUSABLE",
        }
        manifest_payload = {
            "schema": HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA,
            "adapter_version": HYBRID_TRAIN_REUSE_ADAPTER_VERSION,
            "verification_mode": "hybrid_exact_contract",
            "fresh_train_fallback_reason": None,
            "source_set_fingerprint": "1" * 64,
            "source_snapshot_fingerprint": "8" * 64,
            "candidate_registry_fingerprint": "2" * 64,
            "compatibility_audit_fingerprint": "a" * 64,
            "accepted_registry_fingerprint": "b" * 64,
            "stage6_context_fingerprint": self.context.fingerprint,
            "stage6_evaluation_contract_fingerprint": (
                self.fresh.evaluation_contract_fingerprint
            ),
            "target_provider_fingerprint": "3" * 64,
            "artifact_train_contract_fingerprint": contract_fingerprint,
            "current_train_contract_fingerprint": contract_fingerprint,
            "train_contract_verification_digest": _stable_hash(verification),
            "overlay_digest": _stable_hash(rows),
        }
        fingerprint = _stable_hash(manifest_payload)
        directory = root / fingerprint
        directory.mkdir(parents=True)
        overlay_path = directory / "train_reuse_overlay.jsonl"
        overlay_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        verification_path = directory / "train_reuse_contract_verification.json"
        verification_path.write_text(json.dumps(verification), encoding="utf-8")
        manifest = {
            **manifest_payload,
            "train_reuse_overlay_fingerprint": fingerprint,
            "counts": {
                "overlay_candidates": 1,
                "accepted_candidates": 1,
                "fresh_train_fallback_candidates": 0,
            },
            "coverage_ratio": 1.0,
            "artifacts": {
                path.name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in (overlay_path, verification_path)
            },
        }
        manifest_path = directory / "train_reuse_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def _write_hybrid_full_fresh_fallback(self, root: Path) -> Path:
        verification = {
            "verification_mode": "hybrid_full_fresh_train_fallback",
            "numeric_verification": "forbidden_because_train_contract_mismatch",
            "old_train_metrics_allowed": False,
            "fresh_train_fallback_reason": "current_train_contract_mismatch",
            "contract_difference_paths": [
                "train_scope_projection.evaluation_config.entry_lag"
            ],
            "result": "FULL_FRESH_TRAIN_FALLBACK",
        }
        rows = []
        manifest_payload = {
            "schema": HYBRID_TRAIN_REUSE_MANIFEST_SCHEMA,
            "adapter_version": HYBRID_TRAIN_REUSE_ADAPTER_VERSION,
            "verification_mode": "hybrid_full_fresh_train_fallback",
            "fresh_train_fallback_reason": "current_train_contract_mismatch",
            "source_set_fingerprint": "1" * 64,
            "source_snapshot_fingerprint": "8" * 64,
            "candidate_registry_fingerprint": "2" * 64,
            "compatibility_audit_fingerprint": "a" * 64,
            "accepted_registry_fingerprint": "b" * 64,
            "stage6_context_fingerprint": self.context.fingerprint,
            "stage6_evaluation_contract_fingerprint": (
                self.fresh.evaluation_contract_fingerprint
            ),
            "target_provider_fingerprint": "3" * 64,
            "artifact_train_contract_fingerprint": "7" * 64,
            "current_train_contract_fingerprint": "6" * 64,
            "train_contract_verification_digest": _stable_hash(verification),
            "overlay_digest": _stable_hash(rows),
        }
        fingerprint = _stable_hash(manifest_payload)
        directory = root / fingerprint
        directory.mkdir(parents=True)
        overlay_path = directory / "train_reuse_overlay.jsonl"
        overlay_path.write_text("", encoding="utf-8")
        verification_path = directory / "train_reuse_contract_verification.json"
        verification_path.write_text(json.dumps(verification), encoding="utf-8")
        manifest = {
            **manifest_payload,
            "train_reuse_overlay_fingerprint": fingerprint,
            "counts": {
                "overlay_candidates": 0,
                "accepted_candidates": 1,
                "fresh_train_fallback_candidates": 1,
            },
            "coverage_ratio": 0.0,
            "artifacts": {
                path.name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in (overlay_path, verification_path)
            },
        }
        manifest_path = directory / "train_reuse_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def _write_completed_hybrid_source(self, root: Path) -> tuple[Path, dict]:
        run = root / "completed_hybrid_run"
        run.mkdir()
        train = self.fresh_result.train
        train_series = train["long"]["excess_series"]
        provider = {
            "schema": "factor_gfn.real_reward_provider.v8",
            "data_scope": "training_only",
            "context_fingerprint": self.context.fingerprint,
            "reward_evaluator_context_fingerprint": "d" * 64,
            "evaluation_config": {"rebalance_interval": 5, "entry_lag": 1},
            "reward_config": {
                "barra_min_common_periods": 2,
                "candidate_industry_neutralization": True,
            },
            "calendar": {
                "periods": len(train_series["dates"]),
                "first_date": train_series["dates"][0],
                "last_date": train_series["dates"][-1],
                "sha256": "e" * 64,
            },
            "reward_panel": {"mode": "synthetic_fixed_rebalance"},
            "interpreter": {"numeric_kernel_schema": "synthetic-test-kernel"},
            "industry_neutralization": {"enabled": True},
        }
        provider_fingerprint = _stable_hash(provider)
        config = {
            "schema": "factor_gfn.hybrid_variance_runner.v1",
            "checkpoint_schema": "factor_gfn.checkpoint.hybrid_variance.v1",
            "objective_mode": "hybrid_variance",
            "config_fingerprint": "f" * 64,
            "reward_provider_fingerprint": provider_fingerprint,
            "train_candidate_artifact": {
                "enabled": True,
                "schema": "factor_gfn.stage5_train_candidate_artifact.v1",
                "filename": "train_candidate_artifact.json",
            },
        }
        config_bytes = (json.dumps(config, indent=2) + "\n").encode("utf-8")
        (run / "hybrid_run_config.json").write_bytes(config_bytes)
        gfn_root = Path(__file__).resolve().parents[1] / "factor_gfn" / "gfn"
        train_scope_projection = {
            "provider_schema": provider["schema"],
            "data_scope": provider["data_scope"],
            "context_fingerprint": provider["context_fingerprint"],
            "reward_evaluator_context_fingerprint": provider[
                "reward_evaluator_context_fingerprint"
            ],
            "evaluation_config": provider["evaluation_config"],
            "barra_metric_config": {
                "min_common_periods": 2,
                "candidate_industry_neutralization": True,
            },
            "calendar": provider["calendar"],
            "reward_panel": provider["reward_panel"],
            "interpreter": provider["interpreter"],
            "industry_neutralization": provider["industry_neutralization"],
        }
        contract = {
            "schema": "factor_gfn.train_evaluation_contract.v1",
            "provider_fingerprint": provider_fingerprint,
            "train_scope_projection": train_scope_projection,
            "implementation": {
                "artifact_module_sha256": _sha256_file(
                    gfn_root / "train_candidate_artifact.py"
                ),
                "reward_module_sha256": _sha256_file(gfn_root / "reward.py"),
                "real_reward_module_sha256": _sha256_file(
                    gfn_root / "real_reward.py"
                ),
            },
        }
        contract_fingerprint = _stable_hash(contract)
        expression = self.fresh_result.expression
        artifact = {
            "schema": "factor_gfn.stage5_train_candidate_artifact.v1",
            "source_run": {
                "run_directory_name": run.name,
                "hybrid_run_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            "train_evaluation_contract": contract,
            "train_evaluation_contract_fingerprint": contract_fingerprint,
            "committed_optimizer_step": 3,
            "candidate_count": 1,
            "records": [
                {
                    "schema": "factor_gfn.stage5_train_candidate_record.v1",
                    "structural_hash": expression["structural_hash"],
                    "formula": expression["formula"],
                    "prefix_token_ids": expression["prefix_token_ids"],
                    "node_count": expression["node_count"],
                    "depth": expression["depth"],
                    "train_evaluation_contract_fingerprint": contract_fingerprint,
                    "first_seen": {"optimizer_step": 1, "condition_N": 1},
                    "last_seen": {"optimizer_step": 3, "condition_N": 1},
                    "visit_count": 2,
                    "train_ic": train["ic"]["mean"],
                    "train_ic_valid_periods": train["ic"]["valid_periods"],
                    "train_direction": self.fresh_result.train_direction,
                    "train_long_ir": train["long"]["annualized_ir"],
                    "train_long_valid_periods": train["long"]["valid_periods"],
                    "train_long_excess_dates": train_series["dates"],
                    "train_long_excess_values": train_series["values"],
                    "train_barra_ts_corr": 0.2,
                    "train_barra_correlations": {
                        name: 0.1 for name in STYLE_NAMES
                    },
                    "train_barra_valid_periods_by_style": {
                        name: train["ic"]["valid_periods"] for name in STYLE_NAMES
                    },
                    "neutralization_diagnostics": {
                        "industry_neutralized": True,
                        "skipped_dates": [],
                        "skipped_rate": 0.0,
                        "details": [],
                    },
                }
            ],
        }
        (run / "train_candidate_artifact.json").write_text(
            json.dumps(artifact), encoding="utf-8"
        )
        checkpoint = {
            "schema": "factor_gfn.checkpoint.hybrid_variance.v1",
            "objective_mode": "hybrid_variance",
            "config_fingerprint": "f" * 64,
            "reward_provider_fingerprint": provider_fingerprint,
            "global_optimizer_step": 3,
            "total_trajectories_seen": 48,
        }
        torch.save(checkpoint, run / "checkpoint_latest.pt")
        (run / "runner_state.json").write_text(
            json.dumps(
                {
                    "schema": "factor_gfn.hybrid_variance_runner.v1",
                    "global_optimizer_step": 3,
                    "total_trajectories_seen": 48,
                    "pending_assignment": None,
                    "complete": True,
                    "latest_checkpoint": str(run / "checkpoint_latest.pt"),
                }
            ),
            encoding="utf-8",
        )
        return run, provider

    def test_reused_train_and_fresh_validation_match_full_fresh_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(Path(directory))
            )
            mixed = Stage6MixedCandidateEvaluator(self.fresh, overlay)
            with patch.object(self.fresh._interpreter, "evaluate", return_value=self.factor):
                result = mixed.evaluate(self.candidate)
        self.assertEqual(result.train["metric_origin"], TRAIN_METRIC_ORIGIN_REUSE)
        self.assertEqual(result.validation["metric_origin"], "stage6_fresh_evaluation")
        self.assertEqual(result.train["ic"]["mean"], self.fresh_result.train["ic"]["mean"])
        self.assertEqual(
            result.train["long"]["annualized_ir"],
            self.fresh_result.train["long"]["annualized_ir"],
        )
        self.assertEqual(result.train["long"]["excess_series"]["values"], None)
        self.assertEqual(result.validation["ic"], self.fresh_result.validation["ic"])
        self.assertEqual(result.validation["long"], self.fresh_result.validation["long"])
        self.assertNotEqual(
            result.evaluation_contract_fingerprint,
            self.fresh_result.evaluation_contract_fingerprint,
        )

    def test_candidate_absent_from_overlay_falls_back_to_full_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(Path(directory), include_candidate=False)
            )
            mixed = Stage6MixedCandidateEvaluator(self.fresh, overlay)
            with patch.object(self.fresh._interpreter, "evaluate", return_value=self.factor):
                result = mixed.evaluate(self.candidate)
        self.assertEqual(result.train["metric_origin"], TRAIN_METRIC_ORIGIN_FRESH)
        self.assertIsNotNone(result.train["long"]["excess_series"]["values"])
        self.assertEqual(result.validation["ic"], self.fresh_result.validation["ic"])

    def test_overlay_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_overlay(Path(directory))
            overlay_path = path.parent / "train_reuse_overlay.jsonl"
            overlay_path.write_text(overlay_path.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaises(TrainReuseOverlayIntegrityError):
                Stage6TrainReuseOverlay.load(path)

    def test_mixed_contract_uses_evaluation_store_without_fresh_cache_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = Stage6TrainReuseOverlay.load(self._write_overlay(root / "overlay"))
            mixed = Stage6MixedCandidateEvaluator(self.fresh, overlay)
            self.assertNotEqual(
                evaluation_cache_identity(self.fresh, self.candidate).cache_key,
                evaluation_cache_identity(mixed, self.candidate).cache_key,
            )
            with EvaluationStore(root / "store.sqlite", root / "runs") as store:
                run = store.create_run([self.candidate], mixed)
                runner = Stage6EvaluationRunner(store, mixed, rss_sampling_interval_seconds=0.01)
                with patch.object(
                    self.fresh._interpreter, "evaluate", return_value=self.factor
                ):
                    first = runner.run(run.run_id)
                    second = runner.run(run.run_id)
                self.assertEqual(first.newly_evaluated, 1)
                self.assertEqual(second.resume_skipped, 1)

    def test_mixed_smoke_selection_is_deterministic_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(
                    Path(directory),
                    metric_overrides={
                        "train_ic": 0.02,
                        "train_direction": 1,
                        "train_long_ir": 1.0,
                        "train_barra_ts_corr": 0.1,
                    },
                )
            )
            rows = []
            reusable_hash = self.candidate["current_structural_hash"]
            for index in range(12):
                row = dict(self.candidate)
                row["current_structural_hash"] = reusable_hash if index == 0 else f"{index:064x}"
                rows.append(row)
            first = select_stage6_mixed_smoke_candidates(
                rows, overlay, reused_count=1, fresh_count=3
            )
            second = select_stage6_mixed_smoke_candidates(
                list(reversed(rows)), overlay, reused_count=1, fresh_count=3
            )
        self.assertEqual(
            [row["current_structural_hash"] for row in first],
            [row["current_structural_hash"] for row in second],
        )
        self.assertEqual(len(first), 4)

    def test_train_prefilter_strict_boundaries_fail(self):
        train = dict(self.fresh_result.train)
        train["ic"] = {**train["ic"], "mean": 0.01}
        train["long"] = {**train["long"], "annualized_ir": 0.25}
        train["barra"] = {**train["barra"], "max_abs_correlation": 0.7}
        decision = train_prefilter_decision(train)
        self.assertEqual(decision["status"], TRAIN_PREFILTER_FAILED)
        self.assertEqual(len(decision["failed_conditions"]), 3)

    def test_overlay_prefilter_failure_skips_factor_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(
                    Path(directory), metric_overrides={"train_long_ir": 0.25}
                )
            )
            evaluator = Stage6TrainPrefilterEvaluator(self.fresh, overlay)
            with patch.object(
                self.fresh._interpreter,
                "evaluate",
                side_effect=AssertionError("FactorInterpreter must not run"),
            ):
                result = evaluator.evaluate(self.candidate)
        self.assertEqual(result.status, "completed_invalid")
        self.assertEqual(
            result.train["train_prefilter"]["status"], TRAIN_PREFILTER_FAILED
        )
        self.assertEqual(
            result.validation["availability"],
            "not_evaluated_train_prefilter_failed",
        )
        self.assertEqual(result.validation_evaluation_seconds, 0.0)

    def test_overlay_prefilter_pass_computes_fresh_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(
                    Path(directory),
                    metric_overrides={
                        "train_ic": 0.02,
                        "train_direction": 1,
                        "train_long_ir": 1.0,
                        "train_barra_ts_corr": 0.1,
                    },
                )
            )
            evaluator = Stage6TrainPrefilterEvaluator(self.fresh, overlay)
            with patch.object(
                self.fresh._interpreter, "evaluate", return_value=self.factor
            ) as interpreted:
                result = evaluator.evaluate(self.candidate)
        self.assertEqual(interpreted.call_count, 1)
        self.assertEqual(
            result.train["train_prefilter"]["status"], TRAIN_PREFILTER_PASSED
        )
        self.assertEqual(result.validation["ic"], self.fresh_result.validation["ic"])
        self.assertEqual(
            result.source_identity["evaluation_path"],
            "verified_stage5_train_then_fresh_validation",
        )

    def test_fresh_train_failure_interprets_once_and_skips_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(Path(directory), include_candidate=False)
            )
            evaluator = Stage6TrainPrefilterEvaluator(self.fresh, overlay)
            zero_factor = np.zeros_like(self.factor)
            original_prepare = self.fresh._prepare_split
            prepared_splits = []

            def track_prepare(factor, split_name):
                prepared_splits.append(split_name)
                return original_prepare(factor, split_name)

            with patch.object(
                self.fresh._interpreter, "evaluate", return_value=zero_factor
            ) as interpreted, patch.object(
                self.fresh, "_prepare_split", side_effect=track_prepare
            ):
                result = evaluator.evaluate(self.candidate)
        self.assertEqual(interpreted.call_count, 1)
        self.assertEqual(prepared_splits, ["train"])
        self.assertEqual(
            result.train["train_prefilter"]["status"], TRAIN_PREFILTER_FAILED
        )

    def test_prior_full_result_is_repackaged_without_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(Path(directory), include_candidate=False)
            )
            evaluator = Stage6TrainPrefilterEvaluator(
                self.fresh,
                overlay,
                prior_full_results={
                    self.candidate["current_structural_hash"]: self.fresh_result.to_dict()
                },
            )
            with patch.object(
                self.fresh._interpreter,
                "evaluate",
                side_effect=AssertionError("prior full result must be reused"),
            ):
                result = evaluator.evaluate(self.candidate)
        self.assertEqual(
            result.source_identity["evaluation_path"], PRIOR_FULL_RESULT_REUSE
        )
        self.assertEqual(result.validation["ic"], self.fresh_result.validation["ic"])
        self.assertEqual(result.total_seconds, 0.0)

    def test_global_train_preparation_reuses_only_stage5_and_never_evaluates_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(
                    Path(directory), metric_overrides={"train_long_ir": 0.25}
                )
            )
            evaluator = Stage6TrainPreparationEvaluator(self.fresh, overlay)
            with patch.object(
                self.fresh._interpreter,
                "evaluate",
                side_effect=AssertionError("verified Stage 5 Train must not interpret"),
            ):
                result = evaluator.evaluate(self.candidate)
        self.assertEqual(
            result.source_identity["evaluation_path"],
            "verified_stage5_train_preparation_reuse",
        )
        self.assertEqual(
            result.validation["availability"], TRAIN_NOT_EVALUATED_VALIDATION
        )
        self.assertEqual(result.validation_evaluation_seconds, 0.0)
        self.assertFalse(result.source_identity["old_stage6_train_reuse"])
        self.assertEqual(
            result.train["train_prefilter"]["status"], TRAIN_PREFILTER_FAILED
        )
        self.assertIn(
            "train_long_ir_gt_0_25",
            result.train["train_prefilter"]["failed_conditions"],
        )

    def test_hybrid_v2_train_preparation_never_interprets(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_hybrid_overlay(Path(directory))
            )
            evaluator = Stage6TrainPreparationEvaluator(self.fresh, overlay)
            with patch.object(
                self.fresh._interpreter,
                "evaluate",
                side_effect=AssertionError("Hybrid v2 Train reuse must not interpret"),
            ) as interpreter:
                result = evaluator.evaluate(self.candidate)
        interpreter.assert_not_called()
        self.assertEqual(result.train["metric_origin"], TRAIN_METRIC_ORIGIN_REUSE)
        self.assertEqual(
            result.train["train_evaluation_contract_fingerprint"], "7" * 64
        )
        self.assertEqual(
            result.validation["availability"], TRAIN_NOT_EVALUATED_VALIDATION
        )
        self.assertEqual(result.train["ic"]["mean"], self.fresh_result.train["ic"]["mean"])
        self.assertEqual(
            result.train["ic"]["valid_periods"],
            self.fresh_result.train["ic"]["valid_periods"],
        )
        self.assertEqual(result.train_direction, self.fresh_result.train_direction)
        self.assertEqual(
            result.train["long"]["annualized_ir"],
            self.fresh_result.train["long"]["annualized_ir"],
        )
        self.assertEqual(result.train["barra"], self.fresh_result.train["barra"])
        self.assertEqual(
            result.train["neutralization"],
            self.fresh_result.train["neutralization"],
        )
        self.assertEqual(
            result.train["long"]["excess_series"]["dates"],
            self.fresh_result.train["long"]["excess_series"]["dates"],
        )
        self.assertEqual(
            result.train["long"]["excess_series"]["values"],
            self.fresh_result.train["long"]["excess_series"]["values"],
        )
        self.assertEqual(
            result.train["long"]["excess_series"]["origin"],
            "stage5_hybrid_train_artifact_reuse",
        )

    def test_hybrid_contract_mismatch_forces_fresh_train_with_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_hybrid_full_fresh_fallback(Path(directory))
            )
            evaluator = Stage6TrainPreparationEvaluator(self.fresh, overlay)
            prepared_splits = []
            original_prepare = self.fresh._prepare_split

            def track_prepare(factor, split_name):
                prepared_splits.append(split_name)
                return original_prepare(factor, split_name)

            with patch.object(
                self.fresh._interpreter, "evaluate", return_value=self.factor
            ) as interpreter, patch.object(
                self.fresh, "_prepare_split", side_effect=track_prepare
            ):
                result = evaluator.evaluate(self.candidate)
        interpreter.assert_called_once()
        self.assertEqual(prepared_splits, ["train"])
        self.assertEqual(result.train["metric_origin"], TRAIN_METRIC_ORIGIN_FRESH)
        self.assertEqual(
            result.source_identity["train_reuse_fallback_reason"],
            "current_train_contract_mismatch",
        )
        self.assertEqual(
            result.source_identity["evaluation_path"],
            "stage6_fresh_train_preparation",
        )
        self.assertEqual(
            result.validation["availability"], TRAIN_NOT_EVALUATED_VALIDATION
        )

    def test_completed_hybrid_source_reaches_provisional_alpha_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, provider = self._write_completed_hybrid_source(root)
            source_set = materialize_source_set(
                [
                    CandidateSourceSpec(
                        source_id="hybrid-e2e",
                        source_type="hybrid_train_artifact",
                        source_role="formal_discovery",
                        source_path=run,
                        approval_note="synthetic B5 completed Hybrid source",
                    )
                ],
                root / "snapshots",
            )
            candidate_import = import_candidate_source_set(
                source_set, root / "registries"
            )
            compatibility = audit_expression_compatibility(
                candidate_import, source_set, root / "compatibility"
            )
            compatibility_manifest = json.loads(
                compatibility.read_text(encoding="utf-8")
            )
            accepted = json.loads(
                (compatibility.parent / "auto_accepted_candidate_registry.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            fresh = Stage6CandidateEvaluator(
                self.context,
                compatibility_audit_fingerprint=compatibility_manifest[
                    "audit_fingerprint"
                ],
                accepted_registry_fingerprint=compatibility_manifest[
                    "accepted_registry_fingerprint"
                ],
            )
            overlay_manifest = run_stage6_hybrid_train_reuse_overlay(
                source_set_manifest_path=source_set,
                candidate_import_manifest_path=candidate_import,
                compatibility_manifest_path=compatibility,
                evaluator=fresh,
                target_provider_manifest=provider,
                target_provider_fingerprint=_stable_hash(provider),
                output_root=root / "overlays",
            )
            overlay = Stage6TrainReuseOverlay.load(overlay_manifest)
            train_evaluator = Stage6TrainPreparationEvaluator(fresh, overlay)
            with patch.object(
                fresh._interpreter,
                "evaluate",
                side_effect=AssertionError("exact-contract Train must reuse"),
            ) as train_interpreter:
                train_result = train_evaluator.evaluate(accepted)
            train_interpreter.assert_not_called()
            self.assertEqual(
                train_result.train["train_prefilter"]["status"],
                TRAIN_PREFILTER_PASSED,
            )
            validation_evaluator = Stage6ValidationFromFrozenTrainEvaluator(
                fresh,
                train_records={
                    accepted["current_structural_hash"]: train_result.to_dict()
                },
                train_pass_manifest={
                    "train_pass_manifest_fingerprint": "1" * 64,
                    "train_ordered_result_set_fingerprint": "2" * 64,
                },
                prior_validation_results={},
                prior_validation_seed_set_fingerprint="3" * 64,
            )
            with EvaluationStore(root / "store.sqlite", root / "runs") as store:
                frozen = store.create_run(
                    [accepted],
                    validation_evaluator,
                    scope="validation_from_frozen_train_pass_manifest",
                )
                with patch.object(
                    fresh._interpreter, "evaluate", return_value=self.factor
                ) as validation_interpreter:
                    summary = Stage6EvaluationRunner(
                        store, validation_evaluator
                    ).run(frozen.run_id)
                validation_interpreter.assert_called_once()
                self.assertEqual(summary.run_status, "complete")
                enricher = Stage6TrainLongExcessEnricher(fresh)
                with patch.object(
                    fresh._interpreter,
                    "evaluate",
                    side_effect=AssertionError(
                        "persisted Hybrid long-excess must skip enrichment"
                    ),
                ) as enrichment_interpreter:
                    selection_manifest_path = (
                        run_stage6_survivor_enrichment_selection(
                            store=store,
                            evaluation_run_id=frozen.run_id,
                            accepted_candidates=[accepted],
                            enricher=enricher,
                            output_root=root / "hybrid_provisional",
                        )
                    )
                enrichment_interpreter.assert_not_called()
            selection_manifest = json.loads(
                selection_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(selection_manifest["counts"]["hard_filter_pass"], 1)
            self.assertEqual(
                selection_manifest["counts"][
                    "survivors_already_have_train_long_excess"
                ],
                1,
            )
            self.assertEqual(selection_manifest["counts"]["retained"], 1)
            self.assertEqual(selection_manifest["oos"], "not_loaded_not_evaluated")
            alpha_pool = json.loads(
                (selection_manifest_path.parent / "alpha_pool.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                alpha_pool["structural_hash"], accepted["current_structural_hash"]
            )

    def test_hybrid_exact_n_missing_long_excess_stays_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_hybrid_overlay(
                    Path(directory), include_long_excess=False
                )
            )
            evaluator = Stage6TrainPreparationEvaluator(self.fresh, overlay)
            with patch.object(
                self.fresh._interpreter,
                "evaluate",
                side_effect=AssertionError("exact-N Train summary must still reuse"),
            ):
                result = evaluator.evaluate(self.candidate)
        self.assertIsNone(result.train["long"]["excess_series"]["values"])
        self.assertEqual(
            result.train["long"]["excess_series"]["availability"],
            "missing_allowed_exact_n_1_2",
        )

    def test_hybrid_v2_contract_verification_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_hybrid_overlay(Path(directory))
            verification = path.parent / "train_reuse_contract_verification.json"
            verification.write_text("{}", encoding="utf-8")
            with self.assertRaises(TrainReuseOverlayIntegrityError):
                Stage6TrainReuseOverlay.load(path)

    def test_global_train_preparation_fresh_path_calls_train_split_only(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(Path(directory), include_candidate=False)
            )
            evaluator = Stage6TrainPreparationEvaluator(self.fresh, overlay)
            original_prepare = self.fresh._prepare_split
            prepared_splits = []

            def track_prepare(factor, split_name):
                prepared_splits.append(split_name)
                return original_prepare(factor, split_name)

            with patch.object(
                self.fresh._interpreter, "evaluate", return_value=self.factor
            ), patch.object(self.fresh, "_prepare_split", side_effect=track_prepare):
                result = evaluator.evaluate(self.candidate)
        self.assertEqual(prepared_splits, ["train"])
        self.assertEqual(result.validation_evaluation_seconds, 0.0)
        self.assertEqual(result.train["metric_origin"], TRAIN_METRIC_ORIGIN_FRESH)
        self.assertEqual(
            result.train["train_prefilter"], train_prefilter_decision(result.train)
        )
        self.assertEqual(
            evaluator.evaluation_contract["train_prefilter_timing"],
            "immediate_per_candidate",
        )

    def test_train_pass_manifest_freezes_persisted_candidate_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = Stage6TrainReuseOverlay.load(self._write_overlay(root))
            evaluator = Stage6TrainPreparationEvaluator(self.fresh, overlay)
            with EvaluationStore(root / "store.sqlite", root / "runs") as store:
                frozen = store.create_run(
                    [self.candidate], evaluator, scope="train_preparation_full_accepted_registry"
                )
                with patch.object(
                    self.fresh._interpreter,
                    "evaluate",
                    side_effect=AssertionError("verified Stage 5 Train must not interpret"),
                ):
                    summary = Stage6EvaluationRunner(store, evaluator).run(frozen.run_id)
                self.assertEqual(summary.run_status, "complete")
                manifest_path = _freeze_train_pass_manifest(
                    store=store,
                    run_id=frozen.run_id,
                    candidates=[self.candidate],
                    entry={"entry_manifest_fingerprint": "e" * 64},
                    output_root=root / "output",
                    provisional_universe={
                        "provisional_evaluation_universe_fingerprint": "u" * 64,
                        "original_accepted_candidate_count": 2,
                        "evaluation_eligible_count": 1,
                        "deferred_candidate_count": 1,
                        "deferred_reason_counts": {
                            "historical_train_contract_not_equivalent": 1
                        },
                    },
                )
                stored = store.load_verified_run_results(frozen.run_id).records[0]["result"]
            decision = json.loads(
                (manifest_path.parent / "train_prefilter_results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                decision["status"], stored["train"]["train_prefilter"]["status"]
            )
            self.assertEqual(
                decision["condition_results"],
                stored["train"]["train_prefilter"]["condition_results"],
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["provisional_evaluation_universe"][
                    "evaluation_eligible_count"
                ],
                1,
            )
            self.assertEqual(
                manifest["provisional_evaluation_universe"][
                    "deferred_candidate_count"
                ],
                1,
            )

    def test_validation_phase_consumes_frozen_train_and_only_prepares_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(Path(directory), include_candidate=False)
            )
            train_evaluator = Stage6TrainPreparationEvaluator(self.fresh, overlay)
            with patch.object(
                self.fresh._interpreter, "evaluate", return_value=self.factor
            ):
                train_result = train_evaluator.evaluate(self.candidate)
            train_record = train_result.to_dict()
            evaluator = Stage6ValidationFromFrozenTrainEvaluator(
                self.fresh,
                train_records={
                    self.candidate["current_structural_hash"]: train_record
                },
                train_pass_manifest={
                    "train_pass_manifest_fingerprint": "d" * 64,
                    "train_ordered_result_set_fingerprint": "e" * 64,
                },
                prior_validation_results={},
                prior_validation_seed_set_fingerprint="f" * 64,
            )
            original_prepare = self.fresh._prepare_split
            prepared_splits = []

            def track_prepare(factor, split_name):
                prepared_splits.append(split_name)
                return original_prepare(factor, split_name)

            with patch.object(
                self.fresh._interpreter, "evaluate", return_value=self.factor
            ), patch.object(self.fresh, "_prepare_split", side_effect=track_prepare):
                result = evaluator.evaluate(self.candidate)
        self.assertEqual(prepared_splits, ["validation"])
        self.assertEqual(result.train, train_result.train)
        self.assertEqual(result.train_evaluation_seconds, 0.0)
        self.assertEqual(
            result.source_identity["evaluation_path"],
            "stage6_fresh_validation_from_frozen_train",
        )

    def test_validation_phase_reuses_prior_validation_only_when_direction_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = Stage6TrainReuseOverlay.load(
                self._write_overlay(Path(directory), include_candidate=False)
            )
            train_evaluator = Stage6TrainPreparationEvaluator(self.fresh, overlay)
            with patch.object(
                self.fresh._interpreter, "evaluate", return_value=self.factor
            ):
                train_result = train_evaluator.evaluate(self.candidate)
            evaluator = Stage6ValidationFromFrozenTrainEvaluator(
                self.fresh,
                train_records={
                    self.candidate["current_structural_hash"]: train_result.to_dict()
                },
                train_pass_manifest={
                    "train_pass_manifest_fingerprint": "d" * 64,
                    "train_ordered_result_set_fingerprint": "e" * 64,
                },
                prior_validation_results={
                    self.candidate["current_structural_hash"]: self.fresh_result.to_dict()
                },
                prior_validation_seed_set_fingerprint="f" * 64,
            )
            with patch.object(
                self.fresh._interpreter,
                "evaluate",
                side_effect=AssertionError("trusted Validation must not interpret"),
            ):
                result = evaluator.evaluate(self.candidate)
        self.assertEqual(
            result.source_identity["evaluation_path"],
            "verified_prior_stage6_validation_reuse",
        )
        self.assertEqual(result.validation_evaluation_seconds, 0.0)
        self.assertEqual(result.train, train_result.train)


if __name__ == "__main__":
    unittest.main()
