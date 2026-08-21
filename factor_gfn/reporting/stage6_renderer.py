"""Renderer for the standardized Stage 6 screening-report bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import textwrap
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from .stage6_data import Stage6ReportDataBundle


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


STAGE6_REPORT_SCHEMA = "factor_gfn.reporting.stage6_report.v2"


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
        "after_top20_correlation": "05_after_top20_correlation_matrix.csv",
        "provisional_factor_pool": "06_provisional_factor_pool.csv",
        "top100_candidate_metrics": "06_top100_candidate_metrics.csv",
        "top100_quality_summary": "06_top100_quality_summary.csv",
        "complexity_summary": "06_complexity_summary.csv",
        "top_candidate_examples": "07_top_candidate_examples.csv",
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
        "train_long_excess_corr_after_top20": "05_train_long_excess_corr_after_top20.png",
        "greedy_decorrelation_decisions": "05_greedy_decorrelation_decisions.png",
        "provisional_pool_train_validation_ic": "06_provisional_pool_train_validation_ic.png",
        "provisional_pool_quality_summary": "06_provisional_pool_quality_summary.png",
        "top100_quality_summary": "06_top100_quality_summary.png",
        "top100_ic_distribution": "06_top100_ic_distribution.png",
        "top100_long_ir_distribution": "06_top100_long_ir_distribution.png",
        "top100_barra_corr_distribution": "06_top100_barra_corr_distribution.png",
        "complexity_summary": "06_complexity_summary.png",
        "top_candidate_examples": "07_top_candidate_examples.png",
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
            frame.to_csv(temporary, index=name in {"before_top30_correlation", "after_top20_correlation"}, encoding="utf-8-sig")
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

    @staticmethod
    def _compact_formula(formula: str, *, width: int = 46) -> str:
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
        bbox: list[float],
        highlighted_columns: set[int] | None = None,
        header_color: str = "#244a73",
    ):
        table = axis.table(
            cellText=frame.to_numpy(),
            colLabels=list(frame.columns),
            cellLoc="center",
            colLoc="center",
            colWidths=column_widths,
            bbox=bbox,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        highlighted = highlighted_columns or set()
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#d7dee8")
            cell.set_linewidth(0.8)
            if row == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#f4f7fb" if row % 2 == 0 else "white")
                if column in highlighted:
                    cell.set_facecolor("#dceaf7")
                    cell.set_text_props(weight="bold", color="#173a5e")
        return table

    def figure_candidate_screening_funnel(self, *, save: bool = False) -> Figure:
        data = self.bundle.funnel_summary.reset_index(drop=True)
        stage_labels = {
            "stage5_source": "Stage 5 来源候选",
            "train_prefilter_pass": "Train 预筛选通过",
            "validation_evaluated": "Validation 已评估",
            "hard_filter_pass": "硬筛选通过",
            "decorrelation_input": "去相关输入",
            "provisional_pool": "Provisional pool",
            "frozen_order_top100": "最终 Top100",
        }
        figure, axis = plt.subplots(figsize=(10, 8)); maximum = max(int(data["remaining_count"].max()), 1)
        height = .72
        def scaled_width(count: float) -> float:
            return .9 * max(math.sqrt(max(count, 0.0) / maximum), .045)
        for index, row in data.iterrows():
            width = scaled_width(float(row["remaining_count"]))
            next_width = width if index == len(data) - 1 else scaled_width(float(data.iloc[index + 1]["remaining_count"]))
            y_top, y_bottom = -index, -index - height
            polygon = Polygon([(-width, y_top), (width, y_top), (next_width, y_bottom), (-next_width, y_bottom)], closed=True, facecolor=plt.cm.Blues(.38 + .08 * index), edgecolor="white")
            axis.add_patch(polygon)
            pct = float(row["retention_from_source"]) * 100
            stage = stage_labels.get(str(row["stage"]), str(row["stage"]))
            axis.text(0, (y_top + y_bottom) / 2, f"{stage}\n{int(row['remaining_count']):,}  ({pct:.1f}%)", ha="center", va="center", fontsize=10)
        axis.set_xlim(-1, 1); axis.set_ylim(-len(data), .15); axis.axis("off"); axis.set_title("Stage 6 候选筛选漏斗\n正式 Hybrid 候选集合", pad=18)
        return self._finish(figure, "candidate_screening_funnel", save)

    def figure_hard_filter_failure_counts(self, *, save: bool = False) -> Figure:
        data = self.bundle.hard_filter_condition_summary
        figure, axis = plt.subplots(figsize=(11, 5)); bars = axis.barh(data["condition"], data["fail_count"], color="#d95f5f")
        for bar, (_, row) in zip(bars, data.iterrows()):
            rate = row["fail_count"] / row["observed_count"] if row["observed_count"] else math.nan
            axis.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f" {rate:.1%}" if math.isfinite(rate) else " unavailable", va="center")
        axis.set_title("硬筛选失败数量（各条件互不排斥）"); axis.set_xlabel("失败候选数量"); axis.grid(axis="x", alpha=.2)
        return self._finish(figure, "hard_filter_failure_counts", save)

    def _metric_scatter(self, x_name: str, y_name: str, title: str, key: str, threshold: float, signed: bool, frame: pd.DataFrame | None = None, universe: str = "validation_evaluated", save: bool = False) -> Figure:
        data = self.bundle.validation_candidate_metrics if frame is None else frame
        figure, axis = plt.subplots(figsize=(7, 6)); self._scatter_reference(axis, data[x_name], data[y_name], threshold_x=threshold, threshold_y=threshold, signed=signed)
        universe_label = {"validation_evaluated": "Validation 已评估集合", "provisional_pool": "临时候选池"}.get(universe, universe)
        axis.set(xlabel=x_name, ylabel=y_name, title=f"{title}\n候选集合：{universe_label}（n={len(data)}）")
        return self._finish(figure, key, save)

    def figure_train_validation_ic_scatter(self, *, save: bool = False) -> Figure:
        return self._metric_scatter("train_ic", "validation_ic", "Train IC 与 Validation IC 稳定性（有符号 mean RankIC）", "train_validation_ic_scatter", .01, True, save=save)

    def figure_abs_train_validation_ic_scatter(self, *, save: bool = False) -> Figure:
        return self._metric_scatter("abs_train_ic", "abs_validation_ic", "Train |IC| 与 Validation |IC| 强度延续", "abs_train_validation_ic_scatter", .01, False, save=save)

    def figure_train_validation_long_ir_scatter(self, *, save: bool = False) -> Figure:
        return self._metric_scatter("train_long_ir", "validation_long_ir", "Train Long IR 与 Validation Long IR 稳定性", "train_validation_long_ir_scatter", .25, False, save=save)

    def _before_after_ecdf(self, metrics: tuple[str, ...], title: str, key: str, threshold: float | None, save: bool) -> Figure:
        before = self.bundle.validation_candidate_metrics
        hashes = set(self.bundle.decorrelation_input["structural_hash"])
        after = before[before["structural_hash"].isin(hashes)]
        figure, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)
        for axis, metric in zip(axes.flat, metrics):
            self._ecdf(axis, before[metric], "筛选前：validation_evaluated"); self._ecdf(axis, after[metric], "筛选后：hard_filter_pass")
            if threshold is not None: axis.axvline(threshold, color="crimson", linestyle="--", linewidth=1)
            axis.set(title=metric, xlabel=metric, ylabel="累计比例（ECDF）"); axis.grid(alpha=.2); axis.legend(fontsize=8)
        figure.suptitle(title)
        return self._finish(figure, key, save)

    def figure_ic_distribution_before_after(self, *, save: bool = False) -> Figure:
        return self._before_after_ecdf(("abs_train_ic", "abs_validation_ic"), "硬筛选前后 IC 分布", "ic_distribution_before_after", .01, save)

    def figure_long_ir_distribution_before_after(self, *, save: bool = False) -> Figure:
        return self._before_after_ecdf(("train_long_ir", "validation_long_ir"), "硬筛选前后 Long IR 分布", "long_ir_distribution_before_after", .25, save)

    def figure_barra_corr_before_after(self, *, save: bool = False) -> Figure:
        return self._before_after_ecdf(("train_barra_ts_corr",), "硬筛选前后 Train Barra 相关性分布", "barra_corr_before_after", .7, save)

    def _heatmap(
        self,
        matrix: pd.DataFrame,
        title: str,
        key: str,
        save: bool,
        *,
        note: str | None = None,
    ) -> Figure:
        size = max(6, min(13, 4 + .28 * len(matrix))); figure, axis = plt.subplots(figsize=(size, size))
        masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float)); image = axis.imshow(masked, vmin=-1, vmax=1, cmap="coolwarm")
        factor_codes = [f"F{rank:02d}" for rank in range(1, len(matrix) + 1)]
        axis.set_xticks(range(len(factor_codes)), factor_codes, rotation=90, fontsize=8)
        axis.set_yticks(range(len(factor_codes)), factor_codes, fontsize=8)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=.75, label="Pearson correlation")
        if note:
            figure.text(0.5, 0.01, note, ha="center", fontsize=9)
        return self._finish(figure, key, save)

    def figure_train_long_excess_corr_before_top30(self, *, save: bool = False) -> Figure:
        return self._heatmap(
            self.bundle.before_top30_correlation,
            "Train Directional Long-Excess Correlation — Before Stage 6 decorrelation",
            "train_long_excess_corr_before_top30",
            save,
            note="Top-30 hard-filter-pass candidates; F01-F30 follow matrix order; >=60 finite common dates; NaN = invalid pair",
        )

    def figure_train_long_excess_corr_after_top20(self, *, save: bool = False) -> Figure:
        return self._heatmap(
            self.bundle.after_top20_correlation,
            "Train Directional Long-Excess Correlation — Stage 6 provisional pool",
            "train_long_excess_corr_after_top20",
            save,
            note="Top-20 provisional-pool candidates after decorrelation; F01-F20 follow matrix order; >=60 finite common dates; NaN = invalid pair",
        )

    def figure_greedy_decorrelation_decisions(self, *, save: bool = False) -> Figure:
        data = self.bundle.greedy_pair_audit; figure, axis = plt.subplots(figsize=(11, 5))
        colors = {"retained": "#2a9d8f", "rejected_by_correlation": "#e76f51", "decorrelation_invalid": "#777777"}
        for status, group in data.groupby("persisted_decorrelation_status"):
            available = group["max_abs_valid_corr_to_previous_retained"].notna()
            axis.scatter(group.loc[available, "sorted_rank"], group.loc[available, "max_abs_valid_corr_to_previous_retained"], label=status, color=colors.get(status), s=28)
            axis.scatter(group.loc[~available, "sorted_rank"], np.full((~available).sum(), -.04), marker="x", color=colors.get(status), s=35)
        axis.axhline(.7, color="crimson", linestyle="--", label="门槛 = 0.7"); axis.set(xlabel="持久化 sorted rank / 处理顺序", ylabel="相对全部已保留候选的最大有效 |corr|", title="贪心去相关配对审计（持久化状态为权威结果）"); axis.grid(alpha=.2); axis.legend()
        return self._finish(figure, "greedy_decorrelation_decisions", save)

    def figure_provisional_pool_train_validation_ic(self, *, save: bool = False) -> Figure:
        frame = self.bundle.provisional_factor_pool.copy(); frame["abs_train_ic"] = frame["train_ic"].abs(); frame["abs_validation_ic"] = frame["validation_ic"].abs()
        return self._metric_scatter("train_ic", "validation_ic", "临时候选池 Train IC 与 Validation IC", "provisional_pool_train_validation_ic", .01, True, frame=frame, universe="provisional_pool", save=save)

    def figure_provisional_pool_quality_summary(self, *, save: bool = False) -> Figure:
        data = self.bundle.provisional_factor_pool.copy()
        data["abs_train_ic"] = pd.to_numeric(data["train_ic"], errors="coerce").abs()
        data["abs_validation_ic"] = pd.to_numeric(data["validation_ic"], errors="coerce").abs()
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
        panels = (
            (
                axes[0],
                [data["abs_train_ic"].dropna() * 100, data["abs_validation_ic"].dropna() * 100],
                "IC 强度",
                "绝对 mean RankIC（%）",
                1.0,
                ["Train", "Validation"],
            ),
            (
                axes[1],
                [data["train_long_ir"].dropna(), data["validation_long_ir"].dropna()],
                "Long-only IR",
                "年化 IR",
                0.25,
                ["Train", "Validation"],
            ),
            (
                axes[2],
                [data["train_barra_ts_corr"].dropna()],
                "Barra 暴露相关性",
                "最大绝对相关系数（越低越好）",
                0.70,
                ["Train"],
            ),
        )
        for axis, values, title, xlabel, threshold, labels in panels:
            medians = [float(pd.Series(value).median()) if len(value) else math.nan for value in values]
            annotated_labels = [
                f"{label}\n中位数 {median:.2f}{'%' if title == 'IC 强度' else ''}"
                if math.isfinite(median) else label
                for label, median in zip(labels, medians, strict=True)
            ]
            boxplot = axis.boxplot(
                values,
                orientation="horizontal",
                tick_labels=annotated_labels,
                widths=0.52,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#173a5e", "linewidth": 2.2},
                whiskerprops={"color": "#64748b"},
                capprops={"color": "#64748b"},
            )
            for patch, color in zip(boxplot["boxes"], ("#b9d5ee", "#bfe3d9"), strict=False):
                patch.set_facecolor(color)
                patch.set_edgecolor("#5f7895")
            axis.axvline(threshold, color="#c23b3b", linestyle="--", linewidth=1.5, label=f"筛选门槛 = {threshold:g}")
            axis.set_title(title, fontsize=13, weight="bold")
            axis.set_xlabel(xlabel)
            axis.grid(axis="x", alpha=.2)
            axis.legend(loc="lower right", fontsize=8, frameon=False)
        figure.suptitle(f"临时候选池质量概览（n={len(data):,}）", fontsize=17, weight="bold")
        return self._finish(figure, "provisional_pool_quality_summary", save)

    def figure_top100_quality_summary(self, *, save: bool = False) -> Figure:
        labels = {
            "abs_train_ic": "Train |IC|",
            "abs_validation_ic": "Validation |IC|",
            "train_long_ir": "Train Long IR",
            "validation_long_ir": "Validation Long IR",
            "train_barra_ts_corr": "Train Barra Corr",
        }
        rows = []
        for _, record in self.bundle.top100_quality_summary.iterrows():
            metric = str(record["metric"])
            formatter = (lambda value: f"{float(value):.2%}") if metric in {"abs_train_ic", "abs_validation_ic"} else (lambda value: f"{float(value):.3f}")
            rows.append(
                {
                    "Metric": labels[metric],
                    "Min": formatter(record["min"]),
                    "P25": formatter(record["q25"]),
                    "Median": formatter(record["median"]),
                    "P75": formatter(record["q75"]),
                    "Max": formatter(record["max"]),
                    "Mean": formatter(record["mean"]),
                }
            )
        display_frame = pd.DataFrame(rows)
        figure, axis = plt.subplots(figsize=(12.5, 5.2))
        axis.set_axis_off()
        axis.set_title("最终 Top100 候选质量汇总", fontsize=17, weight="bold", pad=18)
        self._style_table(
            axis,
            display_frame,
            column_widths=[0.22, 0.12, 0.12, 0.13, 0.12, 0.12, 0.12],
            font_size=11,
            bbox=[0.02, 0.10, 0.96, 0.72],
            highlighted_columns={3, 4},
        )
        figure.text(
            0.5,
            0.04,
            "最终 Top100；按 Stage 6 权威候选池顺序截取，每个结构哈希计一次",
            ha="center",
            color="#4b5563",
            fontsize=10,
        )
        return self._finish(figure, "top100_quality_summary", save)

    def _top100_histogram(
        self,
        columns: tuple[tuple[str, str, str], ...],
        *,
        title: str,
        xlabel: str,
        threshold: float,
        threshold_label: str,
        key: str,
        scale: float = 1.0,
        save: bool = False,
    ) -> Figure:
        data = self.bundle.top100_candidate_metrics
        series = [pd.to_numeric(data[column], errors="coerce").dropna() * scale for column, _, _ in columns]
        combined = pd.concat(series, ignore_index=True)
        bin_count = min(30, max(10, int(np.sqrt(max(len(data), 1)))))
        bins = np.histogram_bin_edges(combined, bins=bin_count) if not combined.empty else bin_count
        figure, axis = plt.subplots(figsize=(9, 5))
        for values, (_, label, color) in zip(series, columns, strict=True):
            axis.hist(values, bins=bins, color=color, alpha=.62, label=label, edgecolor="white", linewidth=.6)
        axis.axvline(threshold * scale, color="crimson", linestyle="--", linewidth=1.4, label=threshold_label)
        axis.set_title(f"{title}（最终 Top100）")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("候选数量")
        axis.grid(axis="y", alpha=.25, linewidth=.7)
        axis.legend(frameon=False)
        return self._finish(figure, key, save)

    def figure_top100_ic_distribution(self, *, save: bool = False) -> Figure:
        return self._top100_histogram(
            (("abs_train_ic", "Train |IC|", "#4c78a8"), ("abs_validation_ic", "Validation |IC|", "#72b7b2")),
            title="IC 绝对值分布",
            xlabel="绝对 mean RankIC（%）",
            threshold=.01,
            threshold_label="筛选门槛 = 1%",
            key="top100_ic_distribution",
            scale=100.0,
            save=save,
        )

    def figure_top100_long_ir_distribution(self, *, save: bool = False) -> Figure:
        return self._top100_histogram(
            (("train_long_ir", "Train Long IR", "#4c78a8"), ("validation_long_ir", "Validation Long IR", "#f2a65a")),
            title="Long IR 分布",
            xlabel="年化 IR",
            threshold=.25,
            threshold_label="筛选门槛 = 0.25",
            key="top100_long_ir_distribution",
            save=save,
        )

    def figure_top100_barra_corr_distribution(self, *, save: bool = False) -> Figure:
        return self._top100_histogram(
            (("train_barra_ts_corr", "Train Barra Corr", "#72b7b2"),),
            title="Barra 暴露相关性分布",
            xlabel="最大绝对 Barra 相关系数（越低越好）",
            threshold=.70,
            threshold_label="上限 = 0.70",
            key="top100_barra_corr_distribution",
            save=save,
        )

    def figure_complexity_summary(self, *, save: bool = False) -> Figure:
        metric_labels = {
            "node_count": "节点数",
            "depth": "深度",
            "operator_count": "算子数",
            "leaf_count": "叶节点数",
        }
        rows = []
        for _, record in self.bundle.complexity_summary.iterrows():
            rows.append(
                {
                    "指标": metric_labels[str(record["metric"])],
                    "最小值": f"{float(record['min']):.1f}",
                    "中位数": f"{float(record['median']):.1f}",
                    "均值": f"{float(record['mean']):.2f}",
                    "P95": f"{float(record['p95']):.1f}",
                    "最大值": f"{float(record['max']):.1f}",
                }
            )
        display_frame = pd.DataFrame(rows)
        figure, axis = plt.subplots(figsize=(10.5, 5.0))
        axis.set_axis_off()
        axis.set_title("最终 Top100 表达式复杂度汇总", fontsize=17, weight="bold", pad=18)
        self._style_table(
            axis,
            display_frame,
            column_widths=[0.27, 0.13, 0.15, 0.14, 0.13, 0.13],
            font_size=11,
            bbox=[0.02, 0.14, 0.96, 0.64],
            highlighted_columns={2, 4},
        )
        figure.text(
            0.5,
            0.04,
            "搜索约束：max_nodes = 15；max_depth = 5。触及边界仅作描述，不代表失败。",
            ha="center",
            color="#4b5563",
            fontsize=10,
        )
        return self._finish(figure, "complexity_summary", save)

    def figure_top_candidate_examples(self, *, save: bool = False) -> Figure:
        source = self.bundle.top_candidate_examples
        rows = []
        for _, record in source.iterrows():
            rows.append(
                {
                    "排名": str(int(record["provisional_rank"])),
                    "公式": "\n".join(textwrap.wrap(str(record["formula"]), width=62)),
                    "节点数 / 深度": f"{int(record['node_count'])} / {int(record['depth'])}",
                    "Train IC": f"{float(record['train_ic']):+.2%}",
                    "Validation IC": f"{float(record['validation_ic']):+.2%}",
                    "Train IR": f"{float(record['train_long_ir']):.3f}",
                    "Validation IR": f"{float(record['validation_long_ir']):.3f}",
                    "Train Barra Corr": f"{float(record['train_barra_ts_corr']):.3f}",
                }
            )
        display_frame = pd.DataFrame(rows)
        figure, axis = plt.subplots(figsize=(20, 11.5))
        axis.set_axis_off()
        axis.set_title(
            "Stage 6 最终排名前 10 候选",
            fontsize=23,
            weight="bold",
            color="#173a5e",
            pad=20,
        )
        axis.text(
            0.5,
            0.945,
            "按权威 Stage 6 Rank 展示；节点数 / 深度表示表达式规模 / 树深度",
            transform=axis.transAxes,
            ha="center",
            color="#4b5563",
            fontsize=13,
        )
        table = self._style_table(
            axis,
            display_frame,
            column_widths=[0.055, 0.415, 0.105, 0.085, 0.10, 0.075, 0.09, 0.085],
            font_size=12.5,
            bbox=[0.01, 0.025, 0.98, 0.845],
            highlighted_columns={0, 3},
        )
        for (row, column), cell in table.get_celld().items():
            cell.PAD = 0.06
            cell.get_text().set_verticalalignment("center")
            if row == 0:
                cell.set_text_props(
                    family="Microsoft YaHei",
                    fontsize=12.5,
                    weight="bold",
                )
            elif column != 1:
                cell.set_text_props(family="Microsoft YaHei", fontsize=12.0)
            if row > 0 and column == 1:
                cell.set_text_props(
                    ha="left",
                    family="Consolas",
                    fontsize=11.5,
                    color="#1f2937",
                )
        return self._finish(figure, "top_candidate_examples", save)

    def figure_complexity_shift(self, *, save: bool = False) -> Figure:
        data = self.bundle.structure_shift_summary; data = data[data["category"] == "complexity"]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, metric in zip(axes, ("node_count", "depth")):
            rows = data[data["item"] == metric]; positions = np.arange(len(rows)); axis.bar(positions, rows["median"], color=["#4c78a8", "#f58518"][:len(rows)]); axis.set_xticks(positions, rows["universe"], rotation=15); axis.set(title=f"{metric} 中位数", ylabel=metric); axis.grid(axis="y", alpha=.2)
        figure.suptitle("表达式复杂度变化：validation_evaluated → provisional_pool")
        return self._finish(figure, "complexity_shift", save)

    def figure_operator_field_preference_shift(self, *, save: bool = False) -> Figure:
        operators = self.bundle.operator_prevalence_shift.sort_values("prevalence_ratio_before", ascending=False).head(20)
        fields = self.bundle.field_prevalence_shift
        figure, axes = plt.subplots(1, 2, figsize=(17, 7))
        for axis, data, key, title in ((axes[0], operators, "operator", "Top20 算子覆盖率"), (axes[1], fields, "field", "字段覆盖率")):
            y = np.arange(len(data)); axis.barh(y + .18, data["prevalence_ratio_before"], height=.34, label="validation_evaluated"); axis.barh(y - .18, data["prevalence_ratio_after"], height=.34, label="provisional_pool"); axis.set_yticks(y, data[key]); axis.invert_yaxis(); axis.set(title=title, xlabel="唯一候选覆盖率"); axis.grid(axis="x", alpha=.2); axis.legend(fontsize=8)
        figure.suptitle("算子 / 字段偏好变化")
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
        manifest = {
            **self.bundle.snapshot_manifest,
            "report_schema": STAGE6_REPORT_SCHEMA,
            "report_version": 2,
            "version_label": "Stage 6 Reporting v2",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_universe_counts": universes,
            "top100_policy": "first 100 records in authoritative provisional/frozen pool order; no reselection",
            "after_correlation_top_k": 20,
            "output_inventory": {
                "figures": [str(path.relative_to(self.output_dir)) for path in figures],
                "tables": [str(path.relative_to(self.output_dir)) for path in tables],
            },
        }
        path = self.output_dir / "report_manifest.json"; self._atomic_json(manifest, path)
        return {"manifest": path, "figures": figures, "tables": tables}


__all__ = ["STAGE6_REPORT_SCHEMA", "Stage6ReportRenderer"]
