"""Read-only reporting adapter for a verified OOS baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from factor_gfn.backtest.oos_evaluation import VerifiedOOSBaselineEvaluation
from factor_gfn.backtest.static_strategy_bundle import STRATEGY_IDS


OOS_REPORT_DATA_SCHEMA = "factor_gfn.reporting.oos_baseline_data.v2"
DISPLAY_NAMES = {
    "equal_weight": "Equal Weight",
    "fixed_icir": "Rolling ICIR",
    "lightgbm": "LightGBM",
}


@dataclass(frozen=True)
class OOSReportDataBundle:
    evaluation_fingerprint: str
    coverage_by_date: pd.DataFrame
    strategy_scores_test: pd.DataFrame
    strategy_rank_ic_by_date: pd.DataFrame
    decile_returns: pd.DataFrame
    portfolio_returns: pd.DataFrame
    turnover_by_date: pd.DataFrame
    nav_series: pd.DataFrame
    rolling_icir_weights_by_update: pd.DataFrame
    rolling_icir_diagnostics_by_update: pd.DataFrame
    strategy_score_correlation: pd.DataFrame
    main_strategy_performance_summary: pd.DataFrame
    decile_return_table: pd.DataFrame
    turnover_summary: pd.DataFrame
    coverage_summary: pd.DataFrame
    lightgbm_split_effectiveness: pd.DataFrame
    strategy_freeze_summary: pd.DataFrame


def build_oos_report_data(
    evaluation: VerifiedOOSBaselineEvaluation,
) -> OOSReportDataBundle:
    """Adapt persisted verified payloads; no scores or statistics are recomputed."""

    if not isinstance(evaluation, VerifiedOOSBaselineEvaluation):
        raise TypeError("evaluation must be VerifiedOOSBaselineEvaluation")
    payloads = evaluation.payloads
    performance = payloads["performance_summary"]
    main_rows = []
    for strategy_id in STRATEGY_IDS:
        row = performance[strategy_id]
        main_rows.append(
            {
                "Strategy": DISPLAY_NAMES[strategy_id],
                "Mean RankIC": row["mean_rank_ic"],
                "ICIR": row["icir"],
                "G10 Geometric Annualized Return": row["g10_long"]["geometric_annualized_return"],
                "Geometric Annualized Excess Return": row["excess"]["geometric_annualized_excess_return"],
                "Excess IR": row["excess"]["excess_ir"],
                "G10-G1 Geometric Annualized Return": row["long_short"]["geometric_annualized_return"],
                "G10 Sharpe": row["g10_long"]["sharpe"],
                "G10 Max Drawdown": row["g10_long"]["max_drawdown"],
                "Excess Win Rate": row["excess"]["excess_win_rate"],
                "Mean One-Way Turnover": row["turnover"]["mean_one_way_turnover"],
            }
        )
    deciles = payloads["decile_returns"]
    reporting_summary = performance["reporting_summary"]
    decile_rows = []
    for strategy_id in STRATEGY_IDS:
        stored = reporting_summary["mean_decile_returns"][strategy_id]
        decile_rows.append(
            {
                "Strategy": DISPLAY_NAMES[strategy_id],
                **{f"G{index}": stored[f"G{index}"] for index in range(1, 11)},
                "G10-G1": stored["G10_G1"],
            }
        )
    turnover_rows = []
    for strategy_id in STRATEGY_IDS:
        row = performance[strategy_id]["turnover"]
        turnover_rows.append(
            {
                "Strategy": DISPLAY_NAMES[strategy_id],
                "Mean One-Way Turnover": row["mean_one_way_turnover"],
                "Median": row["median_one_way_turnover"],
                "P75": row["p75_one_way_turnover"],
                "Max": row["max_one_way_turnover"],
                "Turnover Observations": row["turnover_observation_count"],
                "Mean Constituent Replacement Rate": row["mean_constituent_replacement_rate"],
            }
        )
    coverage = payloads["coverage_by_date"]
    stored_coverage = reporting_summary["coverage"]
    coverage_summary = pd.DataFrame(
        [
            {
                "Sample": "common_sample",
                "Mean Raw Universe Count": stored_coverage["mean_raw_universe_count"],
                "Mean Complete-Case Eligible Count": stored_coverage["mean_complete_case_eligible_count"],
                "Mean Label-Eligible Count": stored_coverage["mean_label_eligible_count"],
                "Min Eligible Count": stored_coverage["min_eligible_count"],
                "Mean Coverage Ratio": stored_coverage["mean_coverage_ratio"],
                "Min Coverage Ratio": stored_coverage["min_coverage_ratio"],
                "Invalid Rebalance Periods": stored_coverage["invalid_rebalance_periods"],
            }
        ]
    )
    split_rows = []
    for split in ("Train", "Validation", "Test"):
        row = payloads["lightgbm_split_effectiveness"]["rows"][split]
        split_rows.append(
            {
                "Split": split,
                "Mean RankIC": row["mean_rank_ic"],
                "ICIR": row["icir"],
                "IC Positive Ratio": row["ic_positive_ratio"],
                "Valid Periods": row["valid_ic_periods"],
                "Mean 5D G10-G1": row["mean_5d_g10_g1"],
            }
        )
    freeze = evaluation.manifest["strategy_freeze_summary"]
    freeze_rows = [
        {"Field": "Factor Pool fingerprint", "Value": freeze["factor_pool_fingerprint"]},
        {"Field": "Strategy Bundle fingerprint", "Value": freeze["strategy_bundle_fingerprint"]},
        {"Field": "Test Score Artifact fingerprint", "Value": freeze["test_score_artifact_fingerprint"]},
        {"Field": "Equal K", "Value": freeze["equal_weight"]["K"]},
        {"Field": "Equal weights identity", "Value": freeze["equal_weight"]["weights_identity"]},
        {"Field": "Rolling ICIR initial weights fingerprint", "Value": freeze["fixed_icir"]["initial_weights_fingerprint"]},
        {"Field": "Rolling ICIR weight path fingerprint", "Value": freeze["fixed_icir"]["weight_path_fingerprint"]},
        {"Field": "Rolling ICIR update count", "Value": freeze["fixed_icir"]["update_count"]},
        {"Field": "Rolling ICIR config", "Value": json.dumps(freeze["fixed_icir"]["rolling_config"], ensure_ascii=True, sort_keys=True)},
        {"Field": "Rolling ICIR development IC contract", "Value": json.dumps(freeze["fixed_icir"]["development_ic_contract"], ensure_ascii=True, sort_keys=True)},
        {"Field": "LightGBM parameter fingerprint", "Value": freeze["lightgbm"]["parameter_fingerprint"]},
        {"Field": "LightGBM best_iteration", "Value": freeze["lightgbm"]["best_iteration"]},
        {"Field": "LightGBM selection model fingerprint", "Value": freeze["lightgbm"]["selection_model_fingerprint"]},
        {"Field": "LightGBM final model fingerprint", "Value": freeze["lightgbm"]["final_model_fingerprint"]},
        {"Field": "LightGBM version", "Value": freeze["lightgbm"]["lightgbm_version"]},
        {"Field": "Strategy Bundle freeze timestamp", "Value": freeze["strategy_bundle_freeze_timestamp"]},
        {"Field": "Freeze-time OOS status", "Value": freeze["freeze_time_oos_status"]},
        {"Field": "First Test access evidence", "Value": freeze["first_test_access_evidence"]["event"]},
    ]
    correlation = payloads["strategy_score_correlation"]
    correlation_frame = pd.DataFrame(
        correlation["matrix"],
        index=[DISPLAY_NAMES[name] for name in correlation["strategy_ids"]],
        columns=[DISPLAY_NAMES[name] for name in correlation["strategy_ids"]],
    )
    return OOSReportDataBundle(
        evaluation_fingerprint=evaluation.fingerprint,
        coverage_by_date=coverage.copy(),
        strategy_scores_test=payloads["strategy_scores_test"].copy(),
        strategy_rank_ic_by_date=payloads["strategy_rank_ic_by_date"].copy(),
        decile_returns=deciles.copy(),
        portfolio_returns=payloads["portfolio_returns"].copy(),
        turnover_by_date=payloads["turnover_by_date"].copy(),
        nav_series=payloads["nav_series"].copy(),
        rolling_icir_weights_by_update=payloads[
            "rolling_icir_weights_by_update"
        ].copy(),
        rolling_icir_diagnostics_by_update=payloads[
            "rolling_icir_diagnostics_by_update"
        ].copy(),
        strategy_score_correlation=correlation_frame,
        main_strategy_performance_summary=pd.DataFrame(main_rows),
        decile_return_table=pd.DataFrame(decile_rows),
        turnover_summary=pd.DataFrame(turnover_rows),
        coverage_summary=coverage_summary,
        lightgbm_split_effectiveness=pd.DataFrame(split_rows),
        strategy_freeze_summary=pd.DataFrame(freeze_rows),
    )


__all__ = [
    "DISPLAY_NAMES",
    "OOS_REPORT_DATA_SCHEMA",
    "OOSReportDataBundle",
    "build_oos_report_data",
]
