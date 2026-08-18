"""Renderer for the standardized Stage 6 screening-report bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from .stage6_data import Stage6ReportDataBundle


STAGE6_REPORT_SCHEMA = "factor_gfn.reporting.stage6_report.v1"


class Stage6ReportRenderer:
    """Render figures/tables without reading Stage 6 core artifacts."""

    TABLES = {
        "funnel_summary": "01_stage6_funnel_summary.csv",
        "hard_filter_condition_summary": "02_hard_filter_condition_summary.csv",
        "failure_combinations": "02_common_failure_combinations.csv",
        "stability_summary": "03_train_validation_stability_summary.csv",
        "validation_candidate_metrics": "03_validation_candidate_metrics.csv",
        "before_after_quality_summary": "04_quality_before_after_summary.csv",
        "decorrelation_pair_summary": "05_decorrelation_summary.csv",
        "greedy_pair_audit": "05_greedy_pair_audit.csv",
        "before_top30_correlation": "05_before_top30_correlation_matrix.csv",
        "after_top30_correlation": "05_after_top30_correlation_matrix.csv",
        "provisional_factor_pool": "06_provisional_factor_pool.csv",
        "structure_shift_summary": "07_structure_shift_summary.csv",
        "operator_prevalence_shift": "07_operator_prevalence_shift.csv",
        "field_prevalence_shift": "07_field_prevalence_shift.csv",
        "window_prevalence_shift": "07_window_prevalence_shift.csv",
    }
    FIGURES = {
        "candidate_screening_funnel": "01_candidate_screening_funnel.png",
        "hard_filter_failure_counts": "02_hard_filter_failure_counts.png",
        "train_validation_ic_scatter": "03_train_validation_ic_scatter.png",
        "abs_train_validation_ic_scatter": "03_abs_train_validation_ic_scatter.png",
        "train_validation_long_ir_scatter": "03_train_validation_long_ir_scatter.png",
        "ic_distribution_before_after": "04_ic_distribution_before_after.png",
        "long_ir_distribution_before_after": "04_long_ir_distribution_before_after.png",
        "barra_corr_before_after": "04_barra_corr_before_after.png",
        "train_long_excess_corr_before_top30": "05_train_long_excess_corr_before_top30.png",
        "train_long_excess_corr_after_top30": "05_train_long_excess_corr_after_top30.png",
        "greedy_decorrelation_decisions": "05_greedy_decorrelation_decisions.png",
        "provisional_pool_train_validation_ic": "06_provisional_pool_train_validation_ic.png",
        "provisional_pool_quality_summary": "06_provisional_pool_quality_summary.png",
        "complexity_shift": "07_complexity_shift.png",
        "operator_field_preference_shift": "07_operator_field_preference_shift.png",
    }

    def __init__(self, bundle: Stage6ReportDataBundle, output_dir: str | Path) -> None:
        if not isinstance(bundle, Stage6ReportDataBundle):
            raise TypeError("bundle must be Stage6ReportDataBundle")
        self.bundle = bundle
        self.output_dir = Path(output_dir).resolve()
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"

    def _ensure(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_json(value: dict, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def export_table(self, name: str) -> Path:
        if name not in self.TABLES:
            raise KeyError(f"unknown Stage 6 report table: {name}")
        self._ensure(); target = self.tables_dir / self.TABLES[name]
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            frame = getattr(self.bundle, name)
            frame.to_csv(temporary, index=name in {"before_top30_correlation", "after_top30_correlation"}, encoding="utf-8-sig")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def _finish(self, figure: Figure, key: str, save: bool) -> Figure:
        figure.tight_layout()
        if save:
            self._ensure(); target = self.figures_dir / self.FIGURES[key]
            temporary = target.with_suffix(target.suffix + ".tmp")
            try:
                figure.savefig(temporary, format="png", dpi=170, bbox_inches="tight")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return figure

    @staticmethod
    def _scatter_reference(axis, x: pd.Series, y: pd.Series, *, threshold_x: float, threshold_y: float, signed: bool) -> None:
        axis.scatter(x, y, s=18, alpha=.35, color="#356a9a", edgecolors="none")
        finite = pd.concat([x, y]).dropna()
        if not finite.empty:
            low, high = float(finite.min()), float(finite.max()); axis.plot([low, high], [low, high], "--", color="0.4", linewidth=1)
        for value in ((-threshold_x, threshold_x) if signed else (threshold_x,)):
            axis.axvline(value, color="crimson", linestyle=":", linewidth=1)
        for value in ((-threshold_y, threshold_y) if signed else (threshold_y,)):
            axis.axhline(value, color="crimson", linestyle=":", linewidth=1)
        if signed:
            axis.axvline(0, color="0.75", linewidth=.8); axis.axhline(0, color="0.75", linewidth=.8)
        axis.grid(alpha=.2)

    @staticmethod
    def _ecdf(axis, values: pd.Series, label: str) -> None:
        data = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
        if data.size:
            axis.step(data, np.arange(1, data.size + 1) / data.size, where="post", label=f"{label} (n={data.size})")

    def figure_candidate_screening_funnel(self, *, save: bool = False) -> Figure:
        data = self.bundle.funnel_summary.reset_index(drop=True)
        figure, axis = plt.subplots(figsize=(10, 7)); maximum = max(int(data["remaining_count"].max()), 1)
        height = .72
        for index, row in data.iterrows():
            width = .9 * max(float(row["remaining_count"]) / maximum, .08)
            next_width = width if index == len(data) - 1 else .9 * max(float(data.iloc[index + 1]["remaining_count"]) / maximum, .08)
            y_top, y_bottom = -index, -index - height
            polygon = Polygon([(-width, y_top), (width, y_top), (next_width, y_bottom), (-next_width, y_bottom)], closed=True, facecolor=plt.cm.Blues(.38 + .08 * index), edgecolor="white")
            axis.add_patch(polygon)
            pct = float(row["retention_from_source"]) * 100
            axis.text(0, (y_top + y_bottom) / 2, f"{row['stage']}\n{int(row['remaining_count']):,}  ({pct:.1f}%)", ha="center", va="center", fontsize=10)
        axis.set_xlim(-1, 1); axis.set_ylim(-len(data), .15); axis.axis("off"); axis.set_title("Stage 6 Candidate Screening Funnel\nFormal Hybrid candidate universes", pad=18)
        return self._finish(figure, "candidate_screening_funnel", save)

    def figure_hard_filter_failure_counts(self, *, save: bool = False) -> Figure:
        data = self.bundle.hard_filter_condition_summary
        figure, axis = plt.subplots(figsize=(11, 5)); bars = axis.barh(data["condition"], data["fail_count"], color="#d95f5f")
        for bar, (_, row) in zip(bars, data.iterrows()):
            rate = row["fail_count"] / row["observed_count"] if row["observed_count"] else math.nan
            axis.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f" {rate:.1%}" if math.isfinite(rate) else " unavailable", va="center")
        axis.set_title("Hard-filter Failure Counts (conditions are non-exclusive)"); axis.set_xlabel("Fail count"); axis.grid(axis="x", alpha=.2)
        return self._finish(figure, "hard_filter_failure_counts", save)

    def _metric_scatter(self, x_name: str, y_name: str, title: str, key: str, threshold: float, signed: bool, frame: pd.DataFrame | None = None, universe: str = "validation_evaluated", save: bool = False) -> Figure:
        data = self.bundle.validation_candidate_metrics if frame is None else frame
        figure, axis = plt.subplots(figsize=(7, 6)); self._scatter_reference(axis, data[x_name], data[y_name], threshold_x=threshold, threshold_y=threshold, signed=signed)
        axis.set(xlabel=x_name, ylabel=y_name, title=f"{title}\nUniverse: {universe} (n={len(data)})")
        return self._finish(figure, key, save)

    def figure_train_validation_ic_scatter(self, *, save: bool = False) -> Figure:
        return self._metric_scatter("train_ic", "validation_ic", "Train IC vs Validation IC (signed mean RankIC)", "train_validation_ic_scatter", .01, True, save=save)

    def figure_abs_train_validation_ic_scatter(self, *, save: bool = False) -> Figure:
        return self._metric_scatter("abs_train_ic", "abs_validation_ic", "|Train IC| vs |Validation IC| (abs of stored mean)", "abs_train_validation_ic_scatter", .01, False, save=save)

    def figure_train_validation_long_ir_scatter(self, *, save: bool = False) -> Figure:
        return self._metric_scatter("train_long_ir", "validation_long_ir", "Train Long IR vs Validation Long IR", "train_validation_long_ir_scatter", .25, False, save=save)

    def _before_after_ecdf(self, metrics: tuple[str, ...], title: str, key: str, threshold: float | None, save: bool) -> Figure:
        before = self.bundle.validation_candidate_metrics
        hashes = set(self.bundle.decorrelation_input["structural_hash"])
        after = before[before["structural_hash"].isin(hashes)]
        figure, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)
        for axis, metric in zip(axes.flat, metrics):
            self._ecdf(axis, before[metric], "Before: validation_evaluated"); self._ecdf(axis, after[metric], "After: hard_filter_pass")
            if threshold is not None: axis.axvline(threshold, color="crimson", linestyle="--", linewidth=1)
            axis.set(title=metric, xlabel=metric, ylabel="ECDF"); axis.grid(alpha=.2); axis.legend(fontsize=8)
        figure.suptitle(title)
        return self._finish(figure, key, save)

    def figure_ic_distribution_before_after(self, *, save: bool = False) -> Figure:
        return self._before_after_ecdf(("abs_train_ic", "abs_validation_ic"), "IC Distribution Before vs After Hard Filter", "ic_distribution_before_after", .01, save)

    def figure_long_ir_distribution_before_after(self, *, save: bool = False) -> Figure:
        return self._before_after_ecdf(("train_long_ir", "validation_long_ir"), "Long IR Distribution Before vs After Hard Filter", "long_ir_distribution_before_after", .25, save)

    def figure_barra_corr_before_after(self, *, save: bool = False) -> Figure:
        return self._before_after_ecdf(("train_barra_ts_corr",), "Train Barra Correlation Before vs After Hard Filter", "barra_corr_before_after", .7, save)

    def _heatmap(self, matrix: pd.DataFrame, title: str, key: str, save: bool) -> Figure:
        size = max(6, min(13, 4 + .28 * len(matrix))); figure, axis = plt.subplots(figsize=(size, size))
        masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float)); image = axis.imshow(masked, vmin=-1, vmax=1, cmap="coolwarm")
        labels = [str(value)[:8] for value in matrix.index]; axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7); axis.set_yticks(range(len(labels)), labels, fontsize=7)
        axis.set_title(f"{title}\nNaN = invalid/unavailable pair"); figure.colorbar(image, ax=axis, shrink=.75, label="Pearson correlation")
        return self._finish(figure, key, save)

    def figure_train_long_excess_corr_before_top30(self, *, save: bool = False) -> Figure:
        return self._heatmap(self.bundle.before_top30_correlation, "Top-30 Train Long-Excess Correlation\nBefore Decorrelation: hard_filter_pass", "train_long_excess_corr_before_top30", save)

    def figure_train_long_excess_corr_after_top30(self, *, save: bool = False) -> Figure:
        return self._heatmap(self.bundle.after_top30_correlation, "Top-30 Train Long-Excess Correlation\nAfter Decorrelation: provisional_pool", "train_long_excess_corr_after_top30", save)

    def figure_greedy_decorrelation_decisions(self, *, save: bool = False) -> Figure:
        data = self.bundle.greedy_pair_audit; figure, axis = plt.subplots(figsize=(11, 5))
        colors = {"retained": "#2a9d8f", "rejected_by_correlation": "#e76f51", "decorrelation_invalid": "#777777"}
        for status, group in data.groupby("persisted_decorrelation_status"):
            available = group["max_abs_valid_corr_to_previous_retained"].notna()
            axis.scatter(group.loc[available, "sorted_rank"], group.loc[available, "max_abs_valid_corr_to_previous_retained"], label=status, color=colors.get(status), s=28)
            axis.scatter(group.loc[~available, "sorted_rank"], np.full((~available).sum(), -.04), marker="x", color=colors.get(status), s=35)
        axis.axhline(.7, color="crimson", linestyle="--", label="threshold = 0.7"); axis.set(xlabel="Persisted sorted rank / processing order", ylabel="Max |valid corr| to all previously retained", title="Greedy Decorrelation Pair Audit (persisted status remains authoritative)"); axis.grid(alpha=.2); axis.legend()
        return self._finish(figure, "greedy_decorrelation_decisions", save)

    def figure_provisional_pool_train_validation_ic(self, *, save: bool = False) -> Figure:
        frame = self.bundle.provisional_factor_pool.copy(); frame["abs_train_ic"] = frame["train_ic"].abs(); frame["abs_validation_ic"] = frame["validation_ic"].abs()
        return self._metric_scatter("train_ic", "validation_ic", "Provisional Pool Train vs Validation IC", "provisional_pool_train_validation_ic", .01, True, frame=frame, universe="provisional_pool", save=save)

    def figure_provisional_pool_quality_summary(self, *, save: bool = False) -> Figure:
        data = self.bundle.provisional_factor_pool.copy(); data["abs_train_ic"] = data["train_ic"].abs(); data["abs_validation_ic"] = data["validation_ic"].abs()
        metrics = ("abs_train_ic", "abs_validation_ic", "train_long_ir", "validation_long_ir", "train_barra_ts_corr")
        figure, axes = plt.subplots(1, 5, figsize=(17, 4))
        for axis, metric in zip(axes, metrics):
            axis.boxplot(pd.to_numeric(data[metric], errors="coerce").dropna(), widths=.5, showfliers=True); axis.set_title(metric); axis.set_xticks([]); axis.grid(axis="y", alpha=.2)
        figure.suptitle(f"Provisional Factor Pool Quality Summary (n={len(data)})")
        return self._finish(figure, "provisional_pool_quality_summary", save)

    def figure_complexity_shift(self, *, save: bool = False) -> Figure:
        data = self.bundle.structure_shift_summary; data = data[data["category"] == "complexity"]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, metric in zip(axes, ("node_count", "depth")):
            rows = data[data["item"] == metric]; positions = np.arange(len(rows)); axis.bar(positions, rows["median"], color=["#4c78a8", "#f58518"][:len(rows)]); axis.set_xticks(positions, rows["universe"], rotation=15); axis.set(title=f"Median {metric}", ylabel=metric); axis.grid(axis="y", alpha=.2)
        figure.suptitle("Expression Complexity Shift: validation_evaluated → provisional_pool")
        return self._finish(figure, "complexity_shift", save)

    def figure_operator_field_preference_shift(self, *, save: bool = False) -> Figure:
        operators = self.bundle.operator_prevalence_shift.sort_values("prevalence_ratio_before", ascending=False).head(20)
        fields = self.bundle.field_prevalence_shift
        figure, axes = plt.subplots(1, 2, figsize=(17, 7))
        for axis, data, key, title in ((axes[0], operators, "operator", "Top-20 operator prevalence"), (axes[1], fields, "field", "Field prevalence")):
            y = np.arange(len(data)); axis.barh(y + .18, data["prevalence_ratio_before"], height=.34, label="validation_evaluated"); axis.barh(y - .18, data["prevalence_ratio_after"], height=.34, label="provisional_pool"); axis.set_yticks(y, data[key]); axis.invert_yaxis(); axis.set(title=title, xlabel="Unique-candidate prevalence"); axis.grid(axis="x", alpha=.2); axis.legend(fontsize=8)
        figure.suptitle("Operator / Field Preference Shift")
        return self._finish(figure, "operator_field_preference_shift", save)

    def render_figure(self, name: str, *, save: bool = False) -> Figure:
        if name not in self.FIGURES:
            raise KeyError(f"unknown Stage 6 report figure: {name}")
        method: Callable[..., Figure] = getattr(self, f"figure_{name}")
        return method(save=save)

    def render_all(self) -> dict[str, object]:
        self._ensure(); tables = [self.export_table(name) for name in self.TABLES]
        figures = []
        for name in self.FIGURES:
            figure = self.render_figure(name, save=True); plt.close(figure); figures.append(self.figures_dir / self.FIGURES[name])
        universes = {str(row["stage"]): int(row["remaining_count"]) for _, row in self.bundle.funnel_summary.iterrows()}
        manifest = {**self.bundle.snapshot_manifest, "report_schema": STAGE6_REPORT_SCHEMA, "report_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "candidate_universe_counts": universes, "output_inventory": {"figures": [str(path.relative_to(self.output_dir)) for path in figures], "tables": [str(path.relative_to(self.output_dir)) for path in tables]}}
        path = self.output_dir / "report_manifest.json"; self._atomic_json(manifest, path)
        return {"manifest": path, "figures": figures, "tables": tables}


__all__ = ["STAGE6_REPORT_SCHEMA", "Stage6ReportRenderer"]
