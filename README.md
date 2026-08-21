# GFlowNet 日频表达式因子挖掘

> **English summary.** This research repository studies symbolic alpha discovery on China A-share daily data with a grammar-hierarchical Conditional Hybrid GFlowNet. It contains a completed six-leaf Raw Daily baseline and an in-progress sixteen-leaf Daily-Derived experiment. The Raw baseline has completed Train/Validation selection and one-time Test/OOS evaluation; the Derived experiment has not completed Stage 5 and has no formal Stage 6 or OOS result. See [Reproducibility](docs/REPRODUCIBILITY.md) for the exact data and Notebook workflow.

本项目使用 A 股日频数据，通过文法约束的 Conditional Hybrid GFlowNet 生成可解释的表达式因子，再依次完成 Train/Validation 筛选、因子池与策略冻结，以及一次性 Test/OOS 评价。当前研究包含两个严格隔离的 Feature Space：

- **Raw Daily Baseline v1**：以 `open/high/low/close/vwap/volume` 六个原始日频字段为叶子，完整实验已经冻结；
- **Daily-Derived v1**：以 16 个工程重建的日频衍生字段替换六个 Raw 叶子，当前仍在开发与训练中。

详细研究和工程决策以 [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md) 为准；Raw Baseline 的完成证据见 [BASELINE_DEVELOPMENT_LOG.md](BASELINE_DEVELOPMENT_LOG.md)；Derived 的公式合同和当前开发记录分别见 [设计文档](docs/daily_derived/DAILY_DERIVED_FEATURE_DESIGN.md)与[开发日志](docs/daily_derived/DAILY_DERIVED_FEATURE_DEVELOPMENT_LOG.md)。全部公开文档入口见 [docs/README.md](docs/README.md)。

## 使用与授权边界

本仓库当前仅用于公开研究展示，**未提供开源许可证，也未授权第三方复制、修改、再分发或商业使用**。公开可见不等于获得使用许可。如需复用代码或研究产物，请先联系仓库所有者取得书面授权。

仓库不包含原始行情、申万逐日行业数据、处理后数组、run、checkpoint、SQLite registry、模型或生成报告。这些文件由 `.gitignore` 排除。

问题反馈和外部协作边界见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 研究流程

```text
原始行情 + PIT 行业/股本
        │
        ├── Raw Daily：6 leaves
        │
        └── Daily-Derived v1：16 leaves
                     │
                     ▼
        Grammar-hierarchical Conditional GFlowNet
                     │
          Exact-TB (N=1/2) + LPV (N=3..15)
                     │
                     ▼
        Stage 5 候选发现（Train only）
                     │
                     ▼
        Stage 6 Train/Validation 筛选与去相关
                     │
                     ▼
        Frozen Factor Pool → Top100 → 三种策略
                     │
                     ▼
             一次性 Test/OOS 评价
```

时间划分固定为：

- Train：2010–2018；
- Validation：2019–2020；
- Test/OOS：2021–2025。

Test/OOS 只在候选方向、筛选阈值、Factor Pool、Top100 和策略全部冻结后解锁，不用于反向选特征或调参。

## 当前状态

状态快照日期：**2026-08-21**。运行进度以本地 artifact 为准，README 中的数字不是动态状态页。

| 工作包 | Raw Daily Baseline v1 | Daily-Derived v1 |
|---|---|---|
| Feature contract / tensor | 完成并冻结：6 leaves | 完成并冻结：16 leaves，真实 tensor 已构建 |
| Action space | 142 actions | 152 actions，独立 fingerprint |
| N=1/2 Exact-TB | 完成 | 完成，独立 registry |
| Stage 5 | 完成：100 cycles / 1500 steps / 24000 trajectories | **暂停且未完成**：98 cycles / 1470 steps / 23520 trajectories |
| Stage 6 | 完成：21261 → 6011 → 2815 → 1610 | 仅兼容性与 synthetic smoke；无正式执行 |
| Factor Pool / Top100 | 1610 个完整冻结；frozen-order Top100 | 未生成 |
| Strategy / Test/OOS | 三策略与一次性 OOS 已冻结 | 仅 synthetic plumbing；未读取正式 Test labels |
| 可作出的结论 | 可报告冻结 Baseline 结果 | 不得声称优于 Raw，也不得报告最终性能 |

Derived 正式 run 为：

```text
runs/daily_derived_v1/stage5_hybrid_variance_real_5_15/
  derived_hybrid_5_15_k16_seed42_20260818T154007Z/
```

停止后的 `runner_state.json` 为 `complete=false`、`global_optimizer_step=1470`、`total_trajectories_seen=23520`。完成目标仍是 1500 steps / 24000 trajectories；只有 state、checkpoint 和 candidate artifact 共同通过一致性核验后才能标记 Stage 5 完成。

## Raw Baseline v1

Raw Baseline 的冻结配置包括：Grammar hierarchical Conditional `N=1..15`、Hybrid Exact-TB/LPV、`K=16`、`lr=1e-4`、global grad clip `5`、seed `42` 和 100 cycles。

正式 Stage 5 run `hybrid_5_15_k16_seed42_20260816T025559Z` 完成 1500 optimizer steps、24000 trajectories，得到 21261 个唯一候选。Stage 6 经过兼容性审计、Train/Validation 联合硬筛选及 Train long-excess 去相关，冻结 1610 个 Provisional Factors；三种策略严格共用 frozen ordering 的 Top100：

- Equal Weight；
- Fixed rolling ICIR；
- Static LightGBM。

冻结的 OOS 摘要为：Equal Weight 年化超额收益 11.64%、IR 1.3084；Fixed ICIR 11.96%、IR 1.3860；LightGBM 17.30%、IR 1.9730。以上是本仓库特定数据、时间段和冻结合同下的历史研究结果，不构成投资建议，也不保证可迁移到其他数据源或时期。

已知 caveat：Stage 5 梯度裁剪触发率较高；Fixed ICIR 与 Equal Weight 的 OOS 评分高度相关但不完全相同；LightGBM 换手率较高。权威事实和 artifact fingerprint 见 [Baseline 开发日志](BASELINE_DEVELOPMENT_LOG.md)。

## Daily-Derived v1

Derived v1 的 16 个叶子依次为：

```text
ret_gap, ret_cc1, ret_co, ret_hl,
ret_range, ret_body, ret_upper_shadow, ret_lower_shadow,
ret_close_vwap, ret_open_vwap,
ret_vol_chg1, ret_vol_chg5, turnover, illiq,
ret_amt_chg5, clv
```

这些公式是**工程重建**，不是所参考研报公开披露的精确公式。完整公式、复权、PIT/lag、NaN、dtype、mask 和 fingerprint 合同见 [Daily-Derived v1 Feature Contract](docs/daily_derived/DAILY_DERIVED_FEATURE_DESIGN.md)。

Derived 相对 Raw 的主要实验变量只允许是叶子 Feature Space。Grammar/operator/window、Reward、训练预算、Stage 6 数学、Top100 规则、三策略和 OOS 定义保持一致。Raw 与 Derived 的 tensor、vocabulary、Exact-TB registry、checkpoint、run 和所有下游 artifact 均隔离，禁止交叉 resume 或覆盖。

## 快速开始

推荐环境为 Windows、PowerShell、Python 3.12 和支持 CUDA 的 NVIDIA GPU。正式 Stage 5 将 `cuda:0` 作为训练设备；无 CUDA 时只允许执行 CPU-safe preflight。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m jupyter lab
```

本开发工作区的解释器位于 `.\.venv\python.exe`；标准 Windows venv 通常位于 `.\.venv\Scripts\python.exe`。后续命令请使用实际存在的那个路径。Jupyter 必须从仓库根目录启动，数据准备 Notebook 依赖该工作目录。

依赖版本范围记录在 [requirements.txt](requirements.txt)。当前尚未提交逐包锁定文件，因此可重建兼容环境，但不能保证未来安装得到与本次实验逐项相同的 transitive dependencies。

## Notebook 入口

完整数据要求、闸门、new/resume 操作和 artifact 验收见 [复现指南](docs/REPRODUCIBILITY.md)。不要对训练 Notebook 直接执行 “Run All”。

Raw Baseline 的正式顺序：

1. `notebooks/download_data.ipynb`
2. `notebooks/prepare_daily_data.ipynb`
3. `notebooks/prepare_industry_data.ipynb`
4. 准备 `data/processed/barra/` 所需 artifact（当前没有独立的正式 Notebook，见复现限制）
5. `notebooks/run_stage5_hybrid_variance_real_5_15.ipynb`
6. `notebooks/stage5_reporting.ipynb`
7. `notebooks/run_stage6_hybrid_formal_selection.ipynb`
8. `notebooks/stage6_reporting.ipynb`
9. `notebooks/run_baseline_freeze_and_oos.ipynb`
10. `notebooks/oos_baseline_evaluation.ipynb`

Derived 当前可执行到 Stage 5：

1. 先完成 Raw 的数据、行业及 Barra 前置 artifact；
2. `notebooks/prepare_daily_derived_data.ipynb`
3. `notebooks/build_daily_derived_v1_exact_tb_n1_n2.ipynb`
4. `notebooks/run_stage5_daily_derived_v1_hybrid_variance_real_5_15.ipynb`

Derived 尚无正式 Stage 6 / Factor Pool / Strategy / OOS Notebook，不能用 Raw Stage 6 Notebook 替代或把中间 candidates 当成正式输入。

## 外部数据

Adata 行情和股本下载由 `notebooks/download_data.ipynb` 编排。申万逐日 PIT 行业数据没有稳定的公开获取入口，需研究者自行提供到：

```text
参考文件/swind/swind_YYYYMMDD.csv
```

每个 UTF-8 CSV 必须包含以下严格表头：

```text
TradingDay, StockCode, StockName,
SWCode1, SWName1, SWCode2, SWName2, SWCode3, SWName3
```

行业文件格式、代码校验和输出 schema 详见[复现指南](docs/REPRODUCIBILITY.md)。因上游数据不可随仓库分发，第三方若使用不同数据供应商或数据库 vintage，无法保证逐项复现本仓库的 fingerprint 和数值结果。

## 代码结构

```text
factor_gfn/
├── data/        # 下载、预处理、股票池、PIT 行业和 Derived builder
├── grammar/     # Token、算子、partial AST 与 exact-N 文法
├── evaluator/   # 表达式解释、截面清洗与指标
├── barra/       # 五类 Barra-style 暴露与收益序列
├── gfn/         # Conditional Hybrid 策略、TB/LPV、Trainer 与 Stage 5 runner
├── backtest/    # Stage 6、Factor Pool、StrategyInput、策略与 OOS authority
└── reporting/   # Stage 5 / Stage 6 / OOS 报告
docs/            # 设计合同、开发日志、交接与复现说明
notebooks/       # 用户手动执行的正式入口和保留的 Legacy 入口
tests/           # 可执行合同与 synthetic/focused 验证
data/            # 本地原始/处理数据，不提交 Git
runs/            # 本地 registry、run、checkpoint 与冻结 artifact，不提交 Git
outputs/         # 本地报告，不提交 Git
```

## 验证

```powershell
.\.venv\python.exe -m pip check
.\.venv\python.exe -m unittest discover -s tests -v
```

自动测试主要验证代码合同和 synthetic fixture。测试通过不等于真实数据已下载、Derived 训练已完成、Stage 6 已运行或 OOS 结果已生成。长时间下载、数据处理、Exact-TB、训练、Stage 6 和 OOS 均由用户通过显式 Notebook 闸门手动运行。

## 研究完整性

- 不使用 Test/OOS 反向修改特征、Reward、筛选阈值、方向、Factor Pool 或策略；
- 不把 synthetic smoke、partial run 或历史 Notebook 输出描述为正式结果；
- 配置 fingerprint 改变时创建新 run，不强制恢复旧 checkpoint；
- 不覆盖 Raw Baseline 或其他实验的 registry、checkpoint 和 artifact；
- 表达式因子与策略结果仅用于研究，不构成投资建议。
