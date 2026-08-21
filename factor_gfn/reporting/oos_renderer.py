"""Verified OOS renderer, including rolling-ICIR diagnostics."""

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from factor_gfn.backtest.oos_evaluation import geometric_annualized_return
from factor_gfn.backtest.static_strategy_bundle import STRATEGY_IDS

from .oos_data import DISPLAY_NAMES, OOSReportDataBundle


OOS_REPORT_SCHEMA = "factor_gfn.reporting.oos_baseline_report.v4"


STRATEGY_COLORS = {
    "equal_weight": "#4C78A8",
    "fixed_icir": "#F2A65A",
    "lightgbm": "#59A14F",
}

FIGURE_DISPLAY_NAMES = {
    "equal_weight": "等权",
    "fixed_icir": "滚动 ICIR",
    "lightgbm": "LightGBM",
}

PERFORMANCE_COLUMNS = (
    ("Mean RankIC", "平均 RankIC", False),
    ("ICIR", "ICIR", False),
    ("Geometric Annualized Excess Return", "年化超额", True),
    ("Excess IR", "超额 IR", False),
    ("G10 Max Drawdown", "最大回撤", True),
    ("Excess Win Rate", "超额胜率", True),
    ("G10-G1 Geometric Annualized Return", "年化多空", True),
    ("Mean One-Way Turnover", "换手率", True),
)


class OOSReportRenderer:
    FIGURES = {
        "decile_return": "01_g1_g10_decile_return.png",
        "g10_nav": "02_g10_long_only_nav.png",
        "excess_nav": "03_g10_excess_nav.png",
        "long_short_nav": "04_g10_g1_nav.png",
        "rank_ic_equal_weight": "05_equal_weight_test_strategy_score_rankic.png",
        "rank_ic_fixed_icir": "05_rolling_icir_test_strategy_score_rankic.png",
        "rank_ic_lightgbm": "05_lightgbm_test_strategy_score_rankic.png",
        "lightgbm_splits": "06_lightgbm_effectiveness_across_splits.png",
        "average_turnover": "07_average_one_way_turnover.png",
        "turnover_series": "08_one_way_turnover_time_series.png",
        "coverage": "09_eligible_stock_coverage.png",
        "score_correlation": "10_strategy_score_correlation_heatmap.png",
        "rolling_icir": "11_rolling_icir_weight_diagnostics.png",
        "performance_comparison": "01_oos_strategy_performance_comparison.png",
        "annual_return_equal_weight": "12_equal_weight_annual_returns.png",
        "annual_return_fixed_icir": "13_rolling_icir_annual_returns.png",
        "annual_return_lightgbm": "14_lightgbm_annual_returns.png",
        "excess_drawdown_equal_weight": "15_equal_weight_excess_nav_drawdown.png",
        "excess_drawdown_fixed_icir": "16_rolling_icir_excess_nav_drawdown.png",
        "excess_drawdown_lightgbm": "17_lightgbm_excess_nav_drawdown.png",
    }
    TABLES = {
        "main_strategy_performance_summary": "01_main_strategy_performance_summary.csv",
        "decile_return_table": "02_decile_return_table.csv",
        "turnover_summary": "03_turnover_summary.csv",
        "coverage_summary": "04_coverage_summary.csv",
        "lightgbm_split_effectiveness": "05_lightgbm_split_effectiveness.csv",
        "strategy_freeze_summary": "06_strategy_freeze_summary.csv",
        "rolling_icir_diagnostics_by_update": "07_rolling_icir_diagnostics_by_update.csv",
        "rolling_icir_weights_by_update": "08_rolling_icir_weights_by_update.csv",
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
            subset = deciles.loc[
                deciles["strategy_id"] == strategy_id
            ].sort_values("date")
            values = [
                geometric_annualized_return(
                    subset[f"G{i}"] - subset["benchmark_return"]
                )
                for i in range(1, 11)
            ]
            bars = axis.bar(
                range(1, 11), values, color=STRATEGY_COLORS[strategy_id]
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.1%}" for value in values],
                padding=3,
                fontsize=7.5,
            )
            axis.axhline(0.0, color="black", linewidth=1.0)
            axis.set_title(FIGURE_DISPLAY_NAMES[strategy_id])
            axis.set_xticks(range(1, 11), [f"G{i}" for i in range(1, 11)])
            annualized_spread = geometric_annualized_return(
                subset["G10"] - subset["G1"]
            )
            axis.text(
                0.02,
                0.95,
                f"年化 G10-G1：{annualized_spread:.1%}",
                transform=axis.transAxes,
                va="top",
            )
            axis.grid(axis="y", alpha=0.2)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[0].set_ylabel("几何年化超额收益")
        figure.suptitle(
            "G1-G10 相对同期等权基准的几何年化超额收益\n"
            "5 日调仓频率按每年 50.4 期年化；基准为同期评价合格股票池等权收益"
        )
        return self._finish(figure, "decile_return", save)

    def _nav_figure(self, column: str, name: str, title: str, *, benchmark: bool = False, save: bool = False) -> Figure:
        figure, axis = plt.subplots(figsize=(10, 5.5))
        frame = self.bundle.nav_series
        for strategy_id in STRATEGY_IDS:
            subset = frame.loc[frame["strategy_id"] == strategy_id].sort_values("date")
            axis.plot(
                subset["date"], subset[column], label=FIGURE_DISPLAY_NAMES[strategy_id],
                color=STRATEGY_COLORS[strategy_id], linewidth=2.0,
            )
        if benchmark:
            subset = frame.loc[frame["strategy_id"] == STRATEGY_IDS[0]].sort_values("date")
            axis.plot(subset["date"], subset["benchmark_nav"], label="Benchmark", color="#555555", linestyle="--")
        axis.set_title(title); axis.set_ylabel("净值"); axis.legend(frameon=False); axis.grid(alpha=.18)
        return self._finish(figure, name, save)

    def figure_g10_nav(self, *, save: bool = False) -> Figure:
        return self._nav_figure("g10_nav", "g10_nav", "G10 多头净值", benchmark=True, save=save)

    def figure_excess_nav(self, *, save: bool = False) -> Figure:
        return self._nav_figure("excess_nav", "excess_nav", "G10 超额净值（G10 - Benchmark）", save=save)

    def figure_long_short_nav(self, *, save: bool = False) -> Figure:
        return self._nav_figure("long_short_nav", "long_short_nav", "G10-G1 多空净值", save=save)

    def figure_rank_ic(
        self, strategy_id: str, *, save: bool = False
    ) -> Figure:
        if strategy_id not in STRATEGY_IDS:
            raise KeyError(f"unknown strategy_id: {strategy_id}")
        subset = self.bundle.strategy_rank_ic_by_date.loc[
            self.bundle.strategy_rank_ic_by_date["strategy_id"] == strategy_id
        ].sort_values("date")
        rank_ic = subset["rank_ic"].to_numpy(dtype=float)
        finite = rank_ic[np.isfinite(rank_ic)]
        mean_rank_ic = float(np.mean(finite)) if finite.size else float("nan")
        ic_std = float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
        icir = mean_rank_ic / ic_std if np.isfinite(ic_std) and ic_std > 0 else float("nan")

        figure, primary = plt.subplots(figsize=(13.5, 6.6))
        secondary = primary.twinx()
        bar_colors = np.where(rank_ic >= 0.0, "#6B9CC6", "#DC7C7C")
        primary.bar(
            subset["date"], rank_ic, width=3.2, color=bar_colors,
            alpha=.78, edgecolor="white", linewidth=.35,
        )
        secondary.plot(
            subset["date"], subset["cumulative_ic"],
            color="#FF8500", linewidth=2.4, label="累计 IC",
        )
        primary.axhline(0.0, color="#222222", linewidth=1.0)
        primary.set_ylabel("当期 RankIC")
        secondary.set_ylabel("累计 IC")
        primary.set_xlabel("调仓日期")
        primary.set_title(
            f"{FIGURE_DISPLAY_NAMES[strategy_id]} Test RankIC 与累计 IC "
            f"（平均 RankIC={mean_rank_ic:.4f}，ICIR={icir:.4f}）"
        )
        secondary.legend(loc="upper left", frameon=True)
        primary.grid(True, color="#C8C8C8", linewidth=.8, alpha=.72)
        primary.set_axisbelow(True)
        date_locator = AutoDateLocator(minticks=5, maxticks=9)
        primary.xaxis.set_major_locator(date_locator)
        primary.xaxis.set_major_formatter(ConciseDateFormatter(date_locator))
        return self._finish(figure, f"rank_ic_{strategy_id}", save)

    def figure_lightgbm_splits(self, *, save: bool = False) -> Figure:
        frame = self.bundle.lightgbm_split_effectiveness
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].bar(frame["Split"], frame["Mean RankIC"], color="#4778a8")
        axes[1].bar(frame["Split"], frame["ICIR"], color="#e17c45")
        axes[0].set_title("平均 RankIC"); axes[1].set_title("ICIR")
        figure.suptitle("Validation = 开发期 / early-stopping 留出集；Test = 真实 OOS")
        return self._finish(figure, "lightgbm_splits", save)

    def figure_average_turnover(self, *, save: bool = False) -> Figure:
        frame = self.bundle.turnover_summary
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(frame["Strategy"], frame["Mean One-Way Turnover"], color="#4778a8")
        axis.set_title("平均单边换手率")
        axis.set_ylabel("平均漂移调整单边换手率"); axis.grid(axis="y", alpha=.2)
        return self._finish(figure, "average_turnover", save)

    def figure_turnover_series(self, *, save: bool = False) -> Figure:
        figure, axis = plt.subplots(figsize=(10, 5.5))
        frame = self.bundle.turnover_by_date
        for strategy_id in STRATEGY_IDS:
            subset = frame.loc[frame["strategy_id"] == strategy_id].sort_values("date")
            axis.plot(subset["date"], subset["one_way_turnover"], label=FIGURE_DISPLAY_NAMES[strategy_id])
        axis.set_title("单边换手率（NaN 表示首期或重置期）"); axis.legend(); axis.grid(alpha=.2)
        return self._finish(figure, "turnover_series", save)

    def figure_coverage(self, *, save: bool = False) -> Figure:
        frame = self.bundle.coverage_by_date.sort_values("date")
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(frame["date"], frame["raw_universe_count"], label="原始股票池")
        axes[0].plot(frame["date"], frame["eligible_stock_count"], label="完整数据合格股票")
        axes[0].legend(); axes[0].set_ylabel("股票数量")
        axes[1].plot(frame["date"], frame["coverage_ratio"], color="#4778a8")
        axes[1].set_ylabel("覆盖率")
        figure.suptitle("Test 股票池覆盖情况")
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
        axis.set_title("平均分期横截面 Spearman 相关系数")
        figure.colorbar(image, ax=axis)
        return self._finish(figure, "score_correlation", save)

    def figure_rolling_icir(self, *, save: bool = False) -> Figure:
        frame = self.bundle.rolling_icir_diagnostics_by_update.sort_values("update_date")
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        if frame.empty or "max_weight" not in frame:
            axes[0].text(0.5, 0.5, "该产物不包含滚动 ICIR 诊断", ha="center", va="center")
            axes[1].axis("off")
            return self._finish(figure, "rolling_icir", save)
        axes[0].plot(frame["update_date"], frame["max_weight"], label="最大单因子权重")
        axes[0].plot(frame["update_date"], frame["weight_hhi"], label="权重 HHI")
        axes[0].axhline(0.03, color="#c44e52", linestyle="--", linewidth=1, label="3% 上限")
        axes[0].set_title("滚动 ICIR 权重集中度")
        axes[0].legend()
        axes[1].plot(
            frame["update_date"],
            frame["positive_dynamic_factor_count"],
            color="#4778a8",
        )
        axes[1].set_title("ICIR 为正且参与动态加权的因子数量")
        axes[1].set_ylabel("因子数量")
        for axis in axes:
            axis.grid(alpha=0.2)
        return self._finish(figure, "rolling_icir", save)

    @staticmethod
    def _compound_return(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if not numeric.size:
            return float("nan")
        return float(np.prod(1.0 + numeric) - 1.0)

    def figure_performance_comparison(self, *, save: bool = False) -> Figure:
        frame = self.bundle.main_strategy_performance_summary.set_index("Strategy")
        row_order = [DISPLAY_NAMES[strategy_id] for strategy_id in reversed(STRATEGY_IDS)]
        frame = frame.loc[row_order, [column for column, _, _ in PERFORMANCE_COLUMNS]]
        frame.index = [FIGURE_DISPLAY_NAMES[strategy_id] for strategy_id in reversed(STRATEGY_IDS)]
        figure, axis = plt.subplots(figsize=(14.5, 4.8))
        axis.axis("off")
        axis.set_title("三种冻结策略样本外绩效对比", fontsize=16, pad=18)

        def format_value(value: float | None, percentage: bool) -> str:
            if value is None or not np.isfinite(float(value)):
                return "—"
            return f"{value:.2%}" if percentage else f"{value:.4f}"

        table = axis.table(
            cellText=[
                [format_value(row[column], percentage) for column, _, percentage in PERFORMANCE_COLUMNS]
                for _, row in frame.iterrows()
            ],
            rowLabels=frame.index,
            colLabels=[label for _, label, _ in PERFORMANCE_COLUMNS],
            cellLoc="center",
            rowLoc="center",
            bbox=[0.0, 0.10, 1.0, 0.76],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        color_map = LinearSegmentedColormap.from_list(
            "soft_red_yellow_green", ("#E6A1AA", "#F4E3A1", "#9BC8AF")
        )
        for (row_index, column_index), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            if row_index == 0:
                cell.set_facecolor("#294E68")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif column_index == -1:
                cell.set_facecolor("#E6EDF2")
                cell.get_text().set_weight("bold")
            else:
                column = PERFORMANCE_COLUMNS[column_index][0]
                values = frame[column].to_numpy(dtype=float)
                current_value = frame.iloc[row_index - 1][column]
                current = float(current_value) if current_value is not None else float("nan")
                finite_values = values[np.isfinite(values)]
                if not np.isfinite(current) or not finite_values.size:
                    cell.set_facecolor("#ECEFF1")
                    continue
                lower, upper = np.min(finite_values), np.max(finite_values)
                scaled = 0.5 if np.isclose(lower, upper) else (
                    current - lower
                ) / (upper - lower)
                if column == "Mean One-Way Turnover":
                    scaled = 1.0 - scaled
                # 压缩两端，避免仅有三种策略时把尚可的相对弱项渲染成强烈红色。
                scaled = 0.18 + 0.72 * float(np.clip(scaled, 0.0, 1.0))
                cell.set_facecolor(color_map(scaled))
        axis.text(
            0.5, 0.015,
            "三种策略使用完全相同的 2021–2025 Test 股票与调仓日样本；G10 为最高得分组；结果未扣交易成本。",
            ha="center", va="bottom", fontsize=9, color="#555555", transform=axis.transAxes,
        )
        return self._finish(figure, "performance_comparison", save)

    def figure_annual_returns(
        self, strategy_id: str, *, save: bool = False
    ) -> Figure:
        if strategy_id not in STRATEGY_IDS:
            raise KeyError(f"unknown strategy_id: {strategy_id}")
        frame = self.bundle.portfolio_returns.loc[
            self.bundle.portfolio_returns["strategy_id"] == strategy_id
        ].copy()
        frame["year"] = pd.to_datetime(frame["date"]).dt.year
        annual = frame.groupby("year", sort=True)["excess_return"].agg(
            self._compound_return
        )
        figure, axis = plt.subplots(figsize=(11.5, 6.2))
        x = np.arange(len(annual), dtype=float)
        colors = np.where(annual.to_numpy(dtype=float) >= 0.0, "#4682B4", "#D95F5F")
        bars = axis.bar(x, annual.to_numpy(dtype=float), width=.78, color=colors)
        axis.bar_label(
            bars,
            labels=[f"{value:.1%}" for value in annual],
            padding=4,
            fontsize=10,
        )
        axis.axhline(0.0, color="#333333", linewidth=1.1)
        axis.set_xticks(x, annual.index.astype(str), rotation=45, ha="right")
        axis.set_xlabel("年份")
        axis.set_ylabel("年度复利超额收益")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_title(f"{FIGURE_DISPLAY_NAMES[strategy_id]}策略年度超额收益")
        axis.grid(True, color="#C8C8C8", linewidth=.8, alpha=.7)
        axis.set_axisbelow(True)
        return self._finish(figure, f"annual_return_{strategy_id}", save)

    def figure_excess_nav_drawdown(
        self, strategy_id: str, *, save: bool = False
    ) -> Figure:
        if strategy_id not in STRATEGY_IDS:
            raise KeyError(f"unknown strategy_id: {strategy_id}")
        frame = self.bundle.nav_series.loc[
            self.bundle.nav_series["strategy_id"] == strategy_id
        ].sort_values("date")
        nav = frame["excess_nav"].to_numpy(dtype=float)
        drawdown = nav / np.maximum.accumulate(nav) - 1.0
        figure, axes = plt.subplots(
            2, 1, figsize=(10.5, 6.4), sharex=True,
            gridspec_kw={"height_ratios": (3.2, 1.0), "hspace": .08},
        )
        line_label = {
            "equal_weight": "等权多头超额净值",
            "fixed_icir": "滚动 ICIR 加权多头超额净值",
            "lightgbm": "LightGBM 多头超额净值",
        }[strategy_id]
        axes[0].plot(
            frame["date"], nav, color="#3E7CB1", linewidth=2.0,
            label=line_label,
        )
        axes[0].axhline(1.0, color="#888888", linewidth=.8, linestyle="--")
        axes[0].set_ylabel("超额净值")
        axes[0].set_title(f"{FIGURE_DISPLAY_NAMES[strategy_id]}多头超额净值")
        axes[0].legend(frameon=False, loc="upper left")
        axes[1].fill_between(
            frame["date"], drawdown, 0.0, color="#F3B6B8", alpha=.42
        )
        axes[1].plot(frame["date"], drawdown, color="#E75B5B", linewidth=1.05)
        axes[1].axhline(0.0, color="#B0B0B0", linewidth=.7)
        axes[1].set_ylabel("回撤")
        axes[1].set_xlabel("调仓日期")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
        date_locator = AutoDateLocator(minticks=5, maxticks=9)
        axes[1].xaxis.set_major_locator(date_locator)
        axes[1].xaxis.set_major_formatter(ConciseDateFormatter(date_locator))
        for axis in axes:
            axis.grid(alpha=.14)
        return self._finish(figure, f"excess_drawdown_{strategy_id}", save)

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
            "lightgbm_splits": self.figure_lightgbm_splits,
            "average_turnover": self.figure_average_turnover,
            "turnover_series": self.figure_turnover_series,
            "coverage": self.figure_coverage,
            "score_correlation": self.figure_score_correlation,
            "rolling_icir": self.figure_rolling_icir,
            "performance_comparison": self.figure_performance_comparison,
        }
        for method in methods.values():
            figure = method(save=True); plt.close(figure)
        for strategy_id in STRATEGY_IDS:
            figure = self.figure_rank_ic(strategy_id, save=True)
            plt.close(figure)
            figure = self.figure_annual_returns(strategy_id, save=True)
            plt.close(figure)
            figure = self.figure_excess_nav_drawdown(strategy_id, save=True)
            plt.close(figure)
        for name in self.TABLES:
            self.export_table(name)
        obsolete_rank_ic = self.figures_dir / "05_test_strategy_score_rankic.png"
        if obsolete_rank_ic.is_file():
            obsolete_rank_ic.unlink()
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
