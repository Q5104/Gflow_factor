"""Fixed ten-figure/six-table renderer for verified OOS report data."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from factor_gfn.backtest.static_strategy_bundle import STRATEGY_IDS

from .oos_data import DISPLAY_NAMES, OOSReportDataBundle


OOS_REPORT_SCHEMA = "factor_gfn.reporting.oos_baseline_report.v1"


class OOSReportRenderer:
    FIGURES = {
        "decile_return": "01_g1_g10_decile_return.png",
        "g10_nav": "02_g10_long_only_nav.png",
        "excess_nav": "03_g10_excess_nav.png",
        "long_short_nav": "04_g10_g1_nav.png",
        "rank_ic": "05_test_strategy_score_rankic.png",
        "lightgbm_splits": "06_lightgbm_effectiveness_across_splits.png",
        "average_turnover": "07_average_one_way_turnover.png",
        "turnover_series": "08_one_way_turnover_time_series.png",
        "coverage": "09_eligible_stock_coverage.png",
        "score_correlation": "10_strategy_score_correlation_heatmap.png",
    }
    TABLES = {
        "main_strategy_performance_summary": "01_main_strategy_performance_summary.csv",
        "decile_return_table": "02_decile_return_table.csv",
        "turnover_summary": "03_turnover_summary.csv",
        "coverage_summary": "04_coverage_summary.csv",
        "lightgbm_split_effectiveness": "05_lightgbm_split_effectiveness.csv",
        "strategy_freeze_summary": "06_strategy_freeze_summary.csv",
    }

    def __init__(self, bundle: OOSReportDataBundle, output_dir: str | Path) -> None:
        if not isinstance(bundle, OOSReportDataBundle):
            raise TypeError("bundle must be OOSReportDataBundle")
        self.bundle = bundle
        self.output_dir = Path(output_dir).resolve()
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"

    def _ensure(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def _finish(self, figure: Figure, name: str, save: bool) -> Figure:
        figure.tight_layout()
        if save:
            self._ensure()
            target = self.figures_dir / self.FIGURES[name]
            temporary = target.with_suffix(".png.tmp")
            figure.savefig(temporary, format="png", dpi=170, bbox_inches="tight")
            os.replace(temporary, target)
        return figure

    def figure_decile_return(self, *, save: bool = False) -> Figure:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        deciles = self.bundle.decile_returns.copy()
        portfolio = self.bundle.portfolio_returns
        benchmark_counts = portfolio.groupby("date")["benchmark_return"].nunique(
            dropna=False
        )
        if not benchmark_counts.eq(1).all():
            raise ValueError("same-date benchmark differs across OOS strategies")
        benchmark_by_date = portfolio[["date", "benchmark_return"]].drop_duplicates(
            subset=["date"]
        )
        deciles = deciles.merge(
            benchmark_by_date,
            on="date",
            how="left",
            validate="many_to_one",
        )
        for axis, strategy_id in zip(axes, STRATEGY_IDS):
            subset = deciles.loc[deciles["strategy_id"] == strategy_id]
            values = [
                float((subset[f"G{i}"] - subset["benchmark_return"]).mean())
                for i in range(1, 11)
            ]
            axis.bar(range(1, 11), values, color="#4778a8")
            axis.axhline(0.0, color="black", linewidth=1.0)
            axis.set_title(DISPLAY_NAMES[strategy_id])
            axis.set_xticks(range(1, 11), [f"G{i}" for i in range(1, 11)])
            axis.text(
                0.02,
                0.95,
                f"Mean G10-G1: {values[9] - values[0]:.4f}",
                transform=axis.transAxes,
                va="top",
            )
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Mean 5-day excess return")
        figure.suptitle(
            "G1-G10 Mean Excess Return vs Same-Date Equal-Weight Benchmark\n"
            "Benchmark: same-date evaluation-eligible universe equal-weight return"
        )
        return self._finish(figure, "decile_return", save)

    def _nav_figure(self, column: str, name: str, title: str, *, benchmark: bool = False, save: bool = False) -> Figure:
        figure, axis = plt.subplots(figsize=(10, 5.5))
        frame = self.bundle.nav_series
        for strategy_id in STRATEGY_IDS:
            subset = frame.loc[frame["strategy_id"] == strategy_id].sort_values("date")
            axis.plot(subset["date"], subset[column], label=DISPLAY_NAMES[strategy_id])
        if benchmark:
            subset = frame.loc[frame["strategy_id"] == STRATEGY_IDS[0]].sort_values("date")
            axis.plot(subset["date"], subset["benchmark_nav"], label="Benchmark", color="black", linestyle="--")
        axis.set_title(title); axis.set_ylabel("Gross NAV"); axis.legend(); axis.grid(alpha=.2)
        return self._finish(figure, name, save)

    def figure_g10_nav(self, *, save: bool = False) -> Figure:
        return self._nav_figure("g10_nav", "g10_nav", "G10 Long-Only NAV", benchmark=True, save=save)

    def figure_excess_nav(self, *, save: bool = False) -> Figure:
        return self._nav_figure("excess_nav", "excess_nav", "G10 Excess NAV (G10 - Benchmark)", save=save)

    def figure_long_short_nav(self, *, save: bool = False) -> Figure:
        return self._nav_figure("long_short_nav", "long_short_nav", "G10-G1 NAV", save=save)

    def figure_rank_ic(self, *, save: bool = False) -> Figure:
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        frame = self.bundle.strategy_rank_ic_by_date
        for strategy_id in STRATEGY_IDS:
            subset = frame.loc[frame["strategy_id"] == strategy_id].sort_values("date")
            axes[0].plot(subset["date"], subset["rank_ic"], label=DISPLAY_NAMES[strategy_id])
            axes[1].plot(subset["date"], subset["cumulative_ic"], label=DISPLAY_NAMES[strategy_id])
        axes[0].set_title("Periodic Test RankIC"); axes[1].set_title("Cumulative IC (not NAV)")
        for axis in axes: axis.axhline(0, color="black", linewidth=.7); axis.grid(alpha=.2); axis.legend()
        return self._finish(figure, "rank_ic", save)

    def figure_lightgbm_splits(self, *, save: bool = False) -> Figure:
        frame = self.bundle.lightgbm_split_effectiveness
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].bar(frame["Split"], frame["Mean RankIC"], color="#4778a8")
        axes[1].bar(frame["Split"], frame["ICIR"], color="#e17c45")
        axes[0].set_title("Mean RankIC"); axes[1].set_title("ICIR")
        figure.suptitle("Validation = development / early-stopping holdout; Test = True OOS")
        return self._finish(figure, "lightgbm_splits", save)

    def figure_average_turnover(self, *, save: bool = False) -> Figure:
        frame = self.bundle.turnover_summary
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(frame["Strategy"], frame["Mean One-Way Turnover"], color="#4778a8")
        axis.set_ylabel("Mean drift-adjusted one-way turnover"); axis.grid(axis="y", alpha=.2)
        return self._finish(figure, "average_turnover", save)

    def figure_turnover_series(self, *, save: bool = False) -> Figure:
        figure, axis = plt.subplots(figsize=(10, 5.5))
        frame = self.bundle.turnover_by_date
        for strategy_id in STRATEGY_IDS:
            subset = frame.loc[frame["strategy_id"] == strategy_id].sort_values("date")
            axis.plot(subset["date"], subset["one_way_turnover"], label=DISPLAY_NAMES[strategy_id])
        axis.set_title("One-Way Turnover (NaN marks first/reset periods)"); axis.legend(); axis.grid(alpha=.2)
        return self._finish(figure, "turnover_series", save)

    def figure_coverage(self, *, save: bool = False) -> Figure:
        frame = self.bundle.coverage_by_date.sort_values("date")
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(frame["date"], frame["raw_universe_count"], label="Raw universe")
        axes[0].plot(frame["date"], frame["eligible_stock_count"], label="Complete-case eligible")
        axes[0].legend(); axes[0].set_ylabel("Stocks")
        axes[1].plot(frame["date"], frame["coverage_ratio"], color="#4778a8")
        axes[1].set_ylabel("Coverage ratio")
        for axis in axes: axis.grid(alpha=.2)
        return self._finish(figure, "coverage", save)

    def figure_score_correlation(self, *, save: bool = False) -> Figure:
        frame = self.bundle.strategy_score_correlation
        figure, axis = plt.subplots(figsize=(6, 5.5))
        image = axis.imshow(frame.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(3), frame.columns, rotation=25, ha="right")
        axis.set_yticks(range(3), frame.index)
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{frame.iloc[row, column]:.3f}", ha="center", va="center")
        axis.set_title("Mean Periodic Cross-Sectional Spearman")
        figure.colorbar(image, ax=axis)
        return self._finish(figure, "score_correlation", save)

    def export_table(self, name: str) -> Path:
        if name not in self.TABLES:
            raise KeyError(f"unknown OOS report table: {name}")
        self._ensure()
        target = self.tables_dir / self.TABLES[name]
        temporary = target.with_suffix(".csv.tmp")
        getattr(self.bundle, name).to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, target)
        return target

    def render_all(self) -> Path:
        self._ensure()
        methods = {
            "decile_return": self.figure_decile_return,
            "g10_nav": self.figure_g10_nav,
            "excess_nav": self.figure_excess_nav,
            "long_short_nav": self.figure_long_short_nav,
            "rank_ic": self.figure_rank_ic,
            "lightgbm_splits": self.figure_lightgbm_splits,
            "average_turnover": self.figure_average_turnover,
            "turnover_series": self.figure_turnover_series,
            "coverage": self.figure_coverage,
            "score_correlation": self.figure_score_correlation,
        }
        for method in methods.values():
            figure = method(save=True); plt.close(figure)
        for name in self.TABLES:
            self.export_table(name)
        files = sorted([*self.figures_dir.iterdir(), *self.tables_dir.iterdir()])
        manifest = {
            "schema": OOS_REPORT_SCHEMA,
            "evaluation_fingerprint": self.bundle.evaluation_fingerprint,
            "figures": list(self.FIGURES.values()),
            "tables": list(self.TABLES.values()),
            "file_hashes": {
                str(path.relative_to(self.output_dir)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": "verified_oos_evaluation_only",
        }
        target = self.output_dir / "report_manifest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)
        return target


__all__ = ["OOS_REPORT_SCHEMA", "OOSReportRenderer"]
