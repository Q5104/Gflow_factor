# GFlowNet 日频 K 线因子挖掘

本项目使用 A 股日频 `open/high/low/close/vwap/volume` 构造表达式因子，通过 Conditional Hybrid GFlowNet 完成候选发现，再经 Train/Validation 筛选、因子池与策略冻结，最终进行一次性 Test/OOS 评价。

研究口径与工程决策以 [DEVELOPMENT_SPEC.md](DEVELOPMENT_SPEC.md) 为准；当前完成状态与权威 artifact 见 [BASELINE_DEVELOPMENT_LOG.md](BASELINE_DEVELOPMENT_LOG.md)。

## 当前 Baseline

- 数据与评价：2010–2018 Train，2019–2020 Validation，2021–2025 Test/OOS；申万一级行业标签使用 point-in-time 数据。
- Stage 5：Grammar hierarchical Conditional `N=1..15`，Hybrid Exact-TB / LPV，`K=16`，`lr=1e-4`，global grad clip `5`，100 cycles。
- 正式 Stage 5 run：`hybrid_5_15_k16_seed42_20260816T025559Z`，已完成 1500 optimizer steps、24000 trajectories，得到 21261 个唯一候选。
- Stage 5 Reporting：`Raw Daily Baseline / Stage 5 Reporting v1`，15 figures + 18 tables。
- Stage 6：单一正式 Hybrid source；21261 → 6011 → 2815 → 1610 Provisional Factors。
- D1：完整冻结 1610 个 Baseline Factors，不重排、不重筛、不重新确定方向。
- StrategyInput：严格按 frozen ordering 取 Top100 固定前缀，Equal Weight、Fixed ICIR、LightGBM 共用同一输入。
- Development Matrix：Train + Validation，100 factors；因子特定 missing 在截面 cleaning 后填 0，不采用 Top100 complete-case 交集。
- OOS：三策略、Test scores 与 evaluation artifact 均已冻结并完成验证；共 241 个调仓期，0 个无效期。
- 正式 OOS 报告：10 figures + 6 tables，输出绑定 evaluation fingerprint。

已知但不改变 Baseline 的 caveat：Stage 5 有较高梯度裁剪触发率；Fixed ICIR 与 Equal Weight 的 OOS 评分高度相关但并非完全相同；LightGBM 换手率较高。具体事实见 Baseline 开发日志和正式报告。

## 权威入口

按真实执行顺序：

1. `notebooks/download_data.ipynb`
2. `notebooks/prepare_daily_data.ipynb`
3. `notebooks/prepare_industry_data.ipynb`
4. `notebooks/run_stage5_hybrid_variance_real_5_15.ipynb`
5. `notebooks/stage5_reporting.ipynb`
6. `notebooks/run_stage6_hybrid_formal_selection.ipynb`
7. `notebooks/stage6_reporting.ipynb`
8. `notebooks/run_baseline_freeze_and_oos.ipynb`
9. `notebooks/oos_baseline_evaluation.ipynb`

长时间下载、数据处理、训练、Stage 6 与 OOS 执行均由用户手动运行。源 Notebook 默认清空输出；正式报告保存在本地 `outputs/`，不提交 Git。

## Legacy 边界

Conditional 改造前只保留两类证据：

- Primary：`d521789d86de425794a9e871b42db586`，110-step grammar-hierarchical；同时保留其 grammar-only 历史训练入口 `notebooks/run_real_candidate_search.ipynb`。
- Secondary：`8778d49870c244a6996e31aa49f40e45`，768-step flat-policy；只作证据，不保留训练 checkpoint/入口。
- 双 run 解释入口：`notebooks/legacy_gflownet_conditional_motivation.ipynb` 与 `outputs/legacy_conditional_motivation/`。

旧 arity、no-anchor、AB、resource-limited 与失败 Hybrid 尝试不属于当前 Baseline，也不再作为正式 Notebook 入口。

## 项目结构

```text
factor_gfn/
├── data/        # 下载、预处理、股票池与 PIT 行业
├── grammar/     # Token、算子、partial AST 与 exact-N 文法
├── evaluator/   # 表达式计算、截面清洗与指标
├── barra/       # Barra 风格暴露与收益序列
├── gfn/         # Conditional Hybrid 策略、TB/LPV、Trainer 与 Stage 5 runner
├── backtest/    # Stage 6、Factor Pool、StrategyInput、策略与 OOS authority
└── reporting/   # Stage 5 / Stage 6 / OOS reporting
docs/            # 专项设计合同与实施记录
notebooks/       # 用户手动执行的正式入口与保留的 Legacy 入口
tests/           # 可执行合同与合成验证
data/            # 本地数据，不提交 Git
runs/            # 本地运行与冻结 artifact，不提交 Git
outputs/         # 本地报告，不提交 Git
```

## 环境与验证

项目使用 Python 3.12 与仓库内 `.venv`：

```powershell
.\.venv\python.exe -m pip check
.\.venv\python.exe -m unittest discover -s tests -v
.\.venv\python.exe -m jupyter lab
```

本地长任务与真实 artifact 不由自动测试替代。提交前先检查 `git status --short`，不要提交数据、run、checkpoint 或生成报告。

## 研究边界

当前 Baseline 已完成后才允许以独立实验加入日频衍生特征、调整策略参数或简化冻结流程。任何新实验必须使用新 run/artifact，不覆盖本次 Baseline，也不得利用 Test/OOS 反向修改方向、筛选阈值、因子池或策略。
