"""Thin renderer for standardized Stage 5 report data."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import textwrap
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .stage5_data import Stage5ReportDataBundle


STAGE5_REPORT_SCHEMA = "factor_gfn.reporting.stage5_report.v2"


class Stage5ReportRenderer:
    """Render compact report tables and figures without reading run artifacts."""

    TABLES = {
        "run_summary": "01_run_summary.csv",
        "training_summary_by_n": "02_training_diagnostics_summary.csv",
        "training_updates": "02_training_updates.csv",
        "exploration_summary": "03_search_exploration_summary.csv",
        "exploration_by_cycle": "03_exploration_by_cycle.csv",
        "exploration_by_n": "03_per_n_exploration_summary.csv",
        "candidate_quality_summary": "04_candidate_quality_summary.csv",
        "candidate_summary": "04_candidate_metrics.csv",
        "quality_reference_counts": "04_quality_reference_counts.csv",
        "long_excess_correlation_summary": "05_long_excess_correlation_summary.csv",
        "selected_long_excess_series": "05_selected_long_excess_series.csv",
        "long_excess_correlation_matrix": "05_train_long_excess_correlation_matrix.csv",
        "complexity_summary": "06_complexity_summary.csv",
        "operator_usage": "06_operator_usage.csv",
        "field_usage": "06_field_usage.csv",
        "window_usage": "06_window_usage.csv",
        "top_candidate_examples": "07_top_candidate_examples.csv",
        "availability_and_warnings": "08_availability_and_warnings.csv",
    }
    FIGURES = {
        "exact_tb_loss": "02_exact_tb_loss_by_cycle.png",
        "lpv_loss": "02_lpv_loss_by_cycle_per_n.png",
        "gradient_norm": "02_gradient_norm_by_step.png",
        "reward_mean": "02_reward_mean_by_cycle_per_n.png",
        "cumulative_unique": "03_cumulative_unique_candidates.png",
        "candidate_count_by_n": "03_candidate_count_by_n.png",
        "abs_train_ic_distribution": "04_abs_train_ic_distribution.png",
        "train_long_ir_distribution": "04_train_long_ir_distribution.png",
        "train_barra_corr_distribution": "04_train_barra_corr_distribution.png",
        "ic_long_ir_scatter": "04_ic_long_ir_scatter.png",
        "candidate_quality_table": "04_candidate_quality_summary.png",
        "long_excess_corr": "05_train_long_excess_corr_top20.png",
        "depth_distribution": "06_depth_distribution.png",
        "complexity_table": "06_complexity_summary.png",
        "operator_usage": "06_operator_usage.png",
        "field_usage": "06_field_usage.png",
        "window_usage": "06_window_usage.png",
        "top_candidate_examples_table": "07_top_candidate_examples.png",
    }

    def __init__(self, bundle: Stage5ReportDataBundle, output_dir: str | Path) -> None:
        if not isinstance(bundle, Stage5ReportDataBundle):
            raise TypeError("bundle must be Stage5ReportDataBundle")
        self.bundle = bundle
        self.output_dir = Path(output_dir).resolve()
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"

    def _ensure_directories(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_csv(frame: pd.DataFrame, path: Path, *, include_index: bool) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            frame.to_csv(temporary, index=include_index, encoding="utf-8-sig")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json(payload: dict, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def export_table(self, name: str) -> Path:
        if name not in self.TABLES:
            raise KeyError(f"unknown Stage 5 report table: {name}")
        frame = getattr(self.bundle, name)
        self._ensure_directories()
        target = self.tables_dir / self.TABLES[name]
        self._atomic_csv(
            frame,
            target,
            include_index=name == "long_excess_correlation_matrix",
        )
        return target

    def export_all_tables(self) -> list[Path]:
        return [self.export_table(name) for name in self.TABLES]

    def _finish(self, figure: Figure, key: str, save: bool) -> Figure:
        figure.tight_layout()
        if save:
            self._ensure_directories()
            target = self.figures_dir / self.FIGURES[key]
            temporary = target.with_suffix(target.suffix + ".tmp")
            try:
                figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return figure

    @staticmethod
    def _decorate(axis, *, xlabel: str, ylabel: str) -> None:
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linewidth=0.7)

    def figure_exact_tb_loss(self, *, save: bool = False) -> Figure:
        data = self.bundle.training_updates
        figure, axis = plt.subplots(figsize=(9, 5))
        for node_count in (1, 2):
            group = data[data["condition_N"] == node_count].sort_values("cycle_index")
            axis.plot(group["cycle_index"], group["tb_loss"], marker="o", markersize=3, label=f"N={node_count} raw")
            if len(group) >= 5:
                rolling = group["tb_loss"].rolling(5, min_periods=3).median()
                axis.plot(group["cycle_index"], rolling, linewidth=2, label=f"N={node_count} rolling median (5)")
        axis.set_title("Exact-TB Loss by Cycle (N=1/2)")
        self._decorate(axis, xlabel="Cycle index", ylabel="TB loss")
        axis.legend()
        return self._finish(figure, "exact_tb_loss", save)

    def figure_lpv_loss(self, *, save: bool = False) -> Figure:
        data = self.bundle.training_updates
        figure, axes = plt.subplots(4, 4, figsize=(15, 12), sharex=True)
        for axis, node_count in zip(axes.flat, range(3, 16)):
            group = data[data["condition_N"] == node_count].sort_values("cycle_index")
            axis.plot(
                group["cycle_index"],
                group["variance_loss"],
                color="#4c78a8",
                alpha=0.30,
                linewidth=1.0,
                marker="o",
                markersize=2.0,
            )
            if len(group) >= 5:
                rolling = group["variance_loss"].rolling(5, min_periods=3).median()
                axis.plot(
                    group["cycle_index"],
                    rolling,
                    color="#1f4e79",
                    linewidth=2.2,
                )
            axis.set_title(f"N={node_count}")
            axis.grid(alpha=0.25)
        for axis in axes.flat[13:]:
            axis.set_visible(False)
        figure.suptitle(
            "LPV Loss by Cycle and N (N=3..15)\n"
            "Faint line: raw batch loss; bold line: rolling median (5 cycles)"
        )
        figure.supxlabel("Cycle index")
        figure.supylabel("Variance loss")
        return self._finish(figure, "lpv_loss", save)

    def figure_gradient_norm(self, *, save: bool = False) -> Figure:
        data = self.bundle.training_updates
        figure, axis = plt.subplots(figsize=(10, 5))
        for objective, label, color in (("exact_tb", "Exact-TB", "#1f77b4"), ("log_partition_variance", "LPV", "#ff7f0e")):
            group = data[data["objective_kind"] == objective]
            axis.scatter(group["global_optimizer_step"], group["policy_grad_norm"], s=14, alpha=0.65, label=label, color=color)
        axis.axhline(5.0, color="crimson", linestyle="--", linewidth=1.2, label="clip threshold = 5")
        finite_norms = data["policy_grad_norm"].dropna()
        clipping_rate = float((finite_norms > 5.0).mean()) if not finite_norms.empty else float("nan")
        axis.set_yscale("log")
        title = "Policy Gradient Norm by Optimizer Step (Pre-clip, Log Scale)"
        if np.isfinite(clipping_rate):
            title += f"\nOverall clipping trigger rate: {clipping_rate:.1%}"
        axis.set_title(title)
        self._decorate(axis, xlabel="Optimizer step", ylabel="Pre-clip gradient norm")
        axis.legend()
        return self._finish(figure, "gradient_norm", save)

    def figure_reward_mean(self, *, save: bool = False) -> Figure:
        data = self.bundle.training_updates
        figure, axes = plt.subplots(5, 3, figsize=(14, 14), sharex=True)
        for axis, node_count in zip(axes.flat, range(1, 16)):
            group = data[data["condition_N"] == node_count].sort_values("cycle_index")
            axis.plot(
                group["cycle_index"],
                group["reward_mean"],
                color="#4c78a8",
                alpha=0.30,
                linewidth=1.0,
                marker="o",
                markersize=2.0,
            )
            if len(group) >= 5:
                rolling = group["reward_mean"].rolling(5, min_periods=3).median()
                axis.plot(
                    group["cycle_index"],
                    rolling,
                    color="#1f4e79",
                    linewidth=2.2,
                )
            axis.set_title(f"N={node_count}")
            axis.grid(alpha=0.25)
        figure.suptitle(
            "Batch Mean Reward by Cycle and N\n"
            "Faint line: raw batch mean; bold line: rolling median (5 cycles)"
        )
        figure.supxlabel("Cycle index")
        figure.supylabel("Batch reward_mean")
        return self._finish(figure, "reward_mean", save)

    def figure_cumulative_unique(self, *, save: bool = False) -> Figure:
        data = self.bundle.exploration_by_cycle
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(data["cycle_index"], data["cumulative_unique_candidates"], color="#1f77b4")
        axes[0].set_title("Cumulative Unique Candidates (one structural hash = one vote)")
        axes[0].set_ylabel("Cumulative unique")
        axes[1].bar(data["cycle_index"], data["new_unique_candidates"], color="#7aa6c2")
        axes[1].set_ylabel("New unique")
        axes[1].set_xlabel("First-seen cycle index")
        for axis in axes:
            axis.grid(alpha=0.25)
        return self._finish(figure, "cumulative_unique", save)

    def figure_candidate_count_by_n(self, *, save: bool = False) -> Figure:
        data = self.bundle.exploration_by_n
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(data["N"], data["unique_candidate_count"], color="#4c78a8", label="Unique structural hashes")
        accepted_counts = data["accepted_trajectories"].dropna().unique()
        if len(accepted_counts) == 1:
            accepted_per_n = int(accepted_counts[0])
            axis.axhline(
                accepted_per_n,
                color="0.35",
                linestyle="--",
                linewidth=1.2,
                label=f"Fixed training budget: {accepted_per_n:,} trajectories per N",
            )
        axis.set_title("Unique Search Outcomes by Conditional N")
        self._decorate(axis, xlabel="N / node count", ylabel="Unique candidate count")
        secondary = axis.twinx()
        secondary.plot(data["N"], data["discovery_efficiency"], color="#f58518", marker="o", label="Unique / accepted trajectories")
        secondary.set_ylabel("Unique structural hashes / accepted trajectories")
        handles = axis.get_legend_handles_labels()[0] + secondary.get_legend_handles_labels()[0]
        labels = axis.get_legend_handles_labels()[1] + secondary.get_legend_handles_labels()[1]
        axis.legend(handles, labels, loc="lower right", fontsize=9)
        return self._finish(figure, "candidate_count_by_n", save)

    def _histogram(self, column: str, title: str, xlabel: str, threshold: float, key: str) -> Figure:
        values = self.bundle.candidate_summary[column].dropna()
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.hist(values, bins=min(50, max(10, int(np.sqrt(max(len(values), 1))))), color="#4c78a8", alpha=0.8)
        axis.axvline(threshold, color="crimson", linestyle="--", label=f"reference = {threshold:g}")
        axis.set_title(f"{title} (Unique Candidate Weighted)")
        self._decorate(axis, xlabel=xlabel, ylabel="Candidate count")
        axis.legend()
        return self._finish(figure, key, False)

    def figure_abs_train_ic_distribution(self, *, save: bool = False) -> Figure:
        figure = self._histogram("abs_train_ic", "Train |IC| Distribution", "abs(train_ic)", 0.01, "abs_train_ic_distribution")
        return self._finish(figure, "abs_train_ic_distribution", save)

    def figure_train_long_ir_distribution(self, *, save: bool = False) -> Figure:
        figure = self._histogram("train_long_ir", "Train Long IR Distribution", "train_long_ir", 0.25, "train_long_ir_distribution")
        return self._finish(figure, "train_long_ir_distribution", save)

    def figure_train_barra_corr_distribution(self, *, save: bool = False) -> Figure:
        figure = self._histogram("train_barra_ts_corr", "Train Barra Correlation Distribution", "train_barra_ts_corr", 0.7, "train_barra_corr_distribution")
        return self._finish(figure, "train_barra_corr_distribution", save)

    def figure_ic_long_ir_scatter(self, *, save: bool = False) -> Figure:
        data = self.bundle.candidate_summary
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(data["abs_train_ic"], data["train_long_ir"], s=12, alpha=0.35, color="#4c78a8")
        axis.axvline(0.01, color="crimson", linestyle="--", linewidth=1)
        axis.axhline(0.25, color="crimson", linestyle="--", linewidth=1)
        axis.set_title("Train |IC| vs Long IR (Unique Candidate Weighted)")
        self._decorate(axis, xlabel="abs(train_ic)", ylabel="train_long_ir")
        return self._finish(figure, "ic_long_ir_scatter", save)

    @staticmethod
    def _compact_formula(formula: str, *, width: int = 42) -> str:
        compact = formula.replace(" ", "")
        if len(compact) <= width:
            return compact
        left = max(1, width - 14)
        return f"{compact[:left]}…{compact[-13:]}"

    @staticmethod
    def _style_table(
        axis,
        frame: pd.DataFrame,
        *,
        column_widths: list[float],
        font_size: float,
        row_scale: float,
        highlighted_columns: set[int] | None = None,
    ):
        table = axis.table(
            cellText=frame.to_numpy(),
            colLabels=list(frame.columns),
            cellLoc="center",
            colLoc="center",
            colWidths=column_widths,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.0, row_scale)
        highlighted = highlighted_columns or set()
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#d7dee8")
            cell.set_linewidth(0.7)
            if row == 0:
                cell.set_facecolor("#244a73")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#f4f7fb" if row % 2 == 0 else "white")
                if column in highlighted:
                    cell.set_facecolor("#dceaf7")
                    cell.set_text_props(weight="bold", color="#173a5e")
        return table

    def figure_candidate_quality_table(self, *, save: bool = False) -> Figure:
        source = self.bundle.candidate_quality_summary.copy()
        metric_labels = {
            "abs_train_ic": "Train |IC|",
            "train_long_ir": "Train Long IR",
            "train_barra_ts_corr": "Train Barra Corr",
        }
        rows: list[dict[str, str]] = []
        for _, record in source.iterrows():
            metric = str(record["metric"])
            formatter = (lambda value: f"{float(value):.2%}") if metric == "abs_train_ic" else (lambda value: f"{float(value):.3f}")
            rows.append(
                {
                    "Metric": metric_labels[metric],
                    "Min": formatter(record["min"]),
                    "P25": formatter(record["25%"]),
                    "Median": formatter(record["50%"]),
                    "P75": formatter(record["75%"]),
                    "Max": formatter(record["max"]),
                    "Mean": formatter(record["mean"]),
                }
            )
        display_frame = pd.DataFrame(rows)
        figure, axis = plt.subplots(figsize=(12.5, 4.2))
        axis.set_axis_off()
        axis.set_title("Train Candidate Quality Summary", fontsize=17, weight="bold", pad=18)
        self._style_table(
            axis,
            display_frame,
            column_widths=[0.22, 0.12, 0.12, 0.13, 0.12, 0.12, 0.12],
            font_size=11,
            row_scale=1.9,
            highlighted_columns={3, 4},
        )
        figure.text(
            0.5,
            0.04,
            f"{len(self.bundle.candidate_summary):,} unique candidates; one structural hash = one vote",
            ha="center",
            color="#4b5563",
            fontsize=10,
        )
        return self._finish(figure, "candidate_quality_table", save)

    def figure_long_excess_correlation(self, *, save: bool = False) -> Figure:
        matrix = self.bundle.long_excess_correlation_matrix
        factor_codes = [f"F{rank:02d}" for rank in range(1, len(matrix) + 1)]
        size = max(9.0, min(14.0, 0.35 * max(len(matrix), 1) + 6.0))
        figure, axis = plt.subplots(figsize=(size, size))
        if matrix.empty:
            axis.text(0.5, 0.5, "No eligible candidates", ha="center", va="center")
            axis.set_axis_off()
        else:
            image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
            axis.set_xticks(range(len(factor_codes)), labels=factor_codes, rotation=90, fontsize=8)
            axis.set_yticks(range(len(factor_codes)), labels=factor_codes, fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pearson correlation")
        axis.set_title("Train Directional Long-Excess Correlation — Stage 5 diagnostic only")
        figure.text(
            0.5,
            0.01,
            "Top-20 by Train |IC|; F01-F20 follow matrix order; >=60 finite common dates; no Stage 6 filtering",
            ha="center",
            fontsize=9,
        )
        return self._finish(figure, "long_excess_corr", save)

    def figure_depth_distribution(self, *, save: bool = False) -> Figure:
        counts = self.bundle.candidate_summary["depth"].value_counts().sort_index()
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(counts.index.astype(str), counts.values, color="#4c78a8")
        axis.set_title("Expression Depth Distribution")
        self._decorate(axis, xlabel="Depth", ylabel="Unique candidate count")
        return self._finish(figure, "depth_distribution", save)

    def figure_complexity_table(self, *, save: bool = False) -> Figure:
        source = self.bundle.complexity_summary.copy()
        metric_labels = {
            "node_count": "Node count",
            "depth": "Depth",
            "operator_count": "Operator count",
            "leaf_count": "Leaf count",
        }
        rows = []
        for _, record in source.iterrows():
            rows.append(
                {
                    "Metric": metric_labels[str(record["metric"])],
                    "Min": f"{float(record['min']):.1f}",
                    "Median": f"{float(record['median']):.1f}",
                    "Mean": f"{float(record['mean']):.2f}",
                    "P95": f"{float(record['p95']):.1f}",
                    "Max": f"{float(record['max']):.1f}",
                }
            )
        display_frame = pd.DataFrame(rows)
        figure, axis = plt.subplots(figsize=(10.5, 5.0))
        axis.set_axis_off()
        axis.set_title("Expression Complexity Summary", fontsize=17, weight="bold", pad=18)
        self._style_table(
            axis,
            display_frame,
            column_widths=[0.27, 0.13, 0.15, 0.14, 0.13, 0.13],
            font_size=11,
            row_scale=1.8,
            highlighted_columns={2, 4},
        )
        figure.text(
            0.5,
            0.04,
            "Search contract: max_nodes = 15; max_depth = 5. Boundary use is descriptive, not a failure flag.",
            ha="center",
            color="#4b5563",
            fontsize=10,
        )
        return self._finish(figure, "complexity_table", save)

    def figure_operator_usage(self, *, save: bool = False) -> Figure:
        data = self.bundle.operator_usage
        families = data[data["usage_level"] == "family"].sort_values("prevalence_ratio")
        individuals = data[data["usage_level"] == "operator"].nlargest(20, "prevalence_ratio").sort_values("prevalence_ratio")
        figure, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes[0].barh(families["operator"], families["prevalence_ratio"], color="#72b7b2")
        axes[0].set_title("Operator Family Prevalence")
        axes[1].barh(individuals["operator"], individuals["prevalence_ratio"], color="#4c78a8")
        axes[1].set_title("Top 20 Individual Operators")
        for axis in axes:
            self._decorate(axis, xlabel="Share of unique candidates", ylabel="")
        return self._finish(figure, "operator_usage", save)

    def figure_field_usage(self, *, save: bool = False) -> Figure:
        data = self.bundle.field_usage.sort_values("prevalence_ratio")
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.barh(data["field"], data["prevalence_ratio"], color="#4c78a8")
        axis.set_title("Underlying Field Usage")
        self._decorate(axis, xlabel="Share of unique candidates", ylabel="Field")
        return self._finish(figure, "field_usage", save)

    def figure_window_usage(self, *, save: bool = False) -> Figure:
        data = self.bundle.window_usage
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(data["window"].astype(str), data["prevalence_ratio"], color="#4c78a8")
        axis.set_title("Temporal Window Usage")
        self._decorate(axis, xlabel="Window", ylabel="Share of unique candidates")
        return self._finish(figure, "window_usage", save)

    def figure_top_candidate_examples_table(self, *, save: bool = False) -> Figure:
        source = self.bundle.top_candidate_examples
        figure, axes = plt.subplots(2, 1, figsize=(15.5, 10.0))
        figure.suptitle(
            "Representative Train Candidate Examples",
            fontsize=21,
            weight="bold",
            y=0.975,
        )
        figure.text(
            0.5,
            0.935,
            "Two complementary rankings; the highlighted column defines each panel. Nodes / Depth = expression size / tree depth.",
            ha="center",
            color="#4b5563",
            fontsize=11.5,
        )

        panels = (
            (axes[0], source.iloc[:5], "Top 5 by Train |IC|", "#244a73", 3),
            (axes[1], source.iloc[5:10], "Top 5 by Train Long IR", "#28766e", 4),
        )
        for axis, panel_source, title, header_color, highlighted_column in panels:
            axis.set_axis_off()
            axis.set_title(title, loc="left", fontsize=15, weight="bold", pad=9, color=header_color)
            rows = []
            for rank, (_, record) in enumerate(panel_source.iterrows(), start=1):
                formula = "\n".join(textwrap.wrap(str(record["formula"]), width=72))
                rows.append(
                    {
                        "#": str(rank),
                        "Formula": formula,
                        "Nodes / Depth": f"{int(record['node_count'])} / {int(record['depth'])}",
                        "Train IC": f"{float(record['train_ic']):+.2%}",
                        "Long IR": f"{float(record['train_long_ir']):.3f}",
                        "Barra Corr": f"{float(record['train_barra_ts_corr']):.3f}",
                    }
                )
            display_frame = pd.DataFrame(rows)
            if display_frame.empty:
                axis.text(
                    0.5,
                    0.42,
                    "No additional eligible candidates",
                    ha="center",
                    va="center",
                    color="#6b7280",
                    fontsize=12,
                )
                continue
            table = axis.table(
                cellText=display_frame.to_numpy(),
                colLabels=list(display_frame.columns),
                cellLoc="center",
                colLoc="center",
                colWidths=[0.055, 0.545, 0.115, 0.10, 0.085, 0.10],
                bbox=[0.0, 0.0, 1.0, 0.88],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11.5)
            for (row, column), cell in table.get_celld().items():
                cell.set_edgecolor("#d7dee8")
                cell.set_linewidth(0.8)
                if row == 0:
                    cell.set_facecolor(header_color)
                    cell.set_text_props(color="white", weight="bold")
                else:
                    cell.set_facecolor("#f4f7fb" if row % 2 == 0 else "white")
                    if column == 1:
                        cell.set_text_props(ha="left", family="monospace", fontsize=11.0)
                    if column == highlighted_column:
                        cell.set_facecolor("#dceaf7" if highlighted_column == 3 else "#d9eee9")
                        cell.set_text_props(weight="bold", color="#173a5e")
        figure.subplots_adjust(left=0.035, right=0.965, top=0.90, bottom=0.035, hspace=0.22)
        return self._finish(figure, "top_candidate_examples_table", save)

    def render_all_figures(self) -> list[Path]:
        methods: tuple[tuple[str, Callable[..., Figure]], ...] = (
            ("exact_tb_loss", self.figure_exact_tb_loss),
            ("lpv_loss", self.figure_lpv_loss),
            ("gradient_norm", self.figure_gradient_norm),
            ("reward_mean", self.figure_reward_mean),
            ("cumulative_unique", self.figure_cumulative_unique),
            ("candidate_count_by_n", self.figure_candidate_count_by_n),
            ("abs_train_ic_distribution", self.figure_abs_train_ic_distribution),
            ("train_long_ir_distribution", self.figure_train_long_ir_distribution),
            ("train_barra_corr_distribution", self.figure_train_barra_corr_distribution),
            ("ic_long_ir_scatter", self.figure_ic_long_ir_scatter),
            ("candidate_quality_table", self.figure_candidate_quality_table),
            ("long_excess_corr", self.figure_long_excess_correlation),
            ("depth_distribution", self.figure_depth_distribution),
            ("complexity_table", self.figure_complexity_table),
            ("operator_usage", self.figure_operator_usage),
            ("field_usage", self.figure_field_usage),
            ("window_usage", self.figure_window_usage),
            ("top_candidate_examples_table", self.figure_top_candidate_examples_table),
        )
        paths = []
        for key, method in methods:
            figure = method(save=True)
            plt.close(figure)
            paths.append(self.figures_dir / self.FIGURES[key])
        return paths

    def write_manifest(self) -> Path:
        self._ensure_directories()
        inventory = {
            "figures": sorted(path.name for path in self.figures_dir.glob("*.png")),
            "tables": sorted(path.name for path in self.tables_dir.glob("*.csv")),
        }
        snapshot = self.bundle.snapshot_manifest
        payload = {
            "report_schema": STAGE5_REPORT_SCHEMA,
            "report_version": 2,
            "version_label": "Stage 5 Reporting v2",
            "version_status": "frozen",
            "baseline_name": "Raw Daily Baseline",
            "source_description": "completed 100-cycle Raw Daily Baseline run",
            "source_run_identity": snapshot["source_run_id"],
            "snapshot_steps": snapshot["snapshot_steps"],
            "fingerprints": snapshot["fingerprints"],
            "complete": snapshot["complete"],
            "incomplete_preview": not snapshot["complete"],
            "generation_time_utc": datetime.now(timezone.utc).isoformat(),
            "output_inventory": inventory,
        }
        target = self.output_dir / "report_manifest.json"
        self._atomic_json(payload, target)
        return target

    def render_all(self) -> dict[str, list[Path] | Path]:
        tables = self.export_all_tables()
        figures = self.render_all_figures()
        manifest = self.write_manifest()
        return {"tables": tables, "figures": figures, "manifest": manifest}


__all__ = ["STAGE5_REPORT_SCHEMA", "Stage5ReportRenderer"]
