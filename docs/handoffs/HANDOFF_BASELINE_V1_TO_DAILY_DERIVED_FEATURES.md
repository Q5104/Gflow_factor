# GFlowNet 项目交接｜Baseline v1 完成 → 日频衍生特征实验

> 交接日期：2026-08-18
>
> 当前仓库：`<project-root>`
>
> 当前分支：`main`
>
> Baseline v1 commit：`1cb12de feat: freeze conditional hybrid baseline pipeline`
>
> 当前状态：Raw Daily Conditional Hybrid Baseline 已完整跑通、验证并提交；下一工作包是日频衍生特征实验设计与实现。

## 1. 新窗口先做什么

按以下顺序阅读，不要先扫描整个仓库：

1. `AGENTS.md`：协作、安全、测试和长任务边界；
2. `README.md`：项目入口、当前 Baseline 摘要和正式 Notebook 顺序；
3. `BASELINE_DEVELOPMENT_LOG.md`：Baseline v1 的真实 artifact、fingerprint、结果与 caveat；
4. `DEVELOPMENT_SPEC.md`：研究口径、数据合同和历史决策；
5. `docs/stage5_hybrid/STAGE5_HYBRID_VARIANCE_DESIGN.md`：Conditional Hybrid Stage 5 冻结合同；
6. 本交接文档：下一实验的边界和用户规划。

随后只读确认：

```powershell
git status --short
git log -1 --oneline --decorate
```

若当前状态与本交接记录不一致，以实时文件、Git 状态和真实 manifest 为准；不要把本文件中的时间快照当作永远不变的运行状态。

## 2. 项目目标与当前架构

项目使用 A 股日频数据生成表达式因子，通过 Conditional Hybrid GFlowNet 搜索候选，再完成 Train/Validation 筛选、因子池与策略冻结，最后进行一次性 Test/OOS 评价。

当前 Baseline 链路：

```text
Raw Daily open/high/low/close/vwap/volume
→ grammar-hierarchical Conditional Hybrid GFlowNet
→ Stage 5 candidate discovery
→ Stage 6 Train/Validation selection
→ full Baseline Factor Pool
→ frozen-order Top100 StrategyInput
→ Development Matrix
→ Equal Weight / Fixed ICIR / LightGBM
→ frozen Test scores
→ frozen OOS evaluation and reporting
```

主要代码目录：

```text
factor_gfn/data/        下载、预处理、股票池和 PIT 行业
factor_gfn/grammar/     Token、算子、partial AST 与 exact-N 文法
factor_gfn/evaluator/   表达式计算、截面 cleaning 与指标
factor_gfn/barra/       Barra 风格暴露与收益序列
factor_gfn/gfn/         Conditional Hybrid 策略、TB/LPV、Trainer、runner
factor_gfn/backtest/    Stage 6、Factor Pool、StrategyInput、策略、OOS authority
factor_gfn/reporting/   Stage 5、Stage 6、OOS reporting
```

长时间下载、真实数据处理、训练、Stage 6 和 OOS 均由用户手动执行。Codex 默认只实现入口并做静态、合成或 focused checks，除非用户明确要求，不启动真实长任务。

## 3. Baseline v1 已完成事实

### 3.1 数据分段

- Train：2010-01-01 至 2018-12-31；
- Validation：2019-01-01 至 2020-12-31；
- Test/OOS：2021-01-01 至 2025-12-31；
- 行业：point-in-time 申万一级行业。

### 3.2 Stage 5

正式合同：

- grammar hierarchical；
- external condition `N=1..15`；
- `N=1/2` fixed Exact-TB；
- `N=3..15` direct LPV；
- `K=16`；
- learning rate `1e-4`；
- global gradient clip `5`；
- 100 cycles = 1500 optimizer steps = 24000 trajectories。

权威 run：

```text
runs/stage5_hybrid_variance_real_5_15/
  hybrid_5_15_k16_seed42_20260816T025559Z
```

完成结果：21261 个唯一 candidates / structural hashes。已知 caveat 是梯度裁剪触发率高；该事实不授权修改 Baseline 的 LR、clip、Reward 或 Conditional-N 合同。

### 3.3 Stage 6

权威目录：

```text
runs/stage6/hybrid_provisional/
  hybrid_5_15_k16_seed42_20260816T025559Z
```

真实漏斗：

```text
Stage 5 accepted candidates          21261
Train prefilter / Validation entry    6011
Six-item hard-filter pass             2815
Train long-excess decorrelation       1610
```

`2815` 是 hard-filter pass / decorrelation input；`1610` 才是 decorrelation 后的 Provisional Factor Pool。两者不可混用。

### 3.4 Factor Pool、StrategyInput 与策略

```text
Full Baseline Factor Pool
runs/baseline_factor_pools/
  f9a3945945ee04eb357896b7b8e20d63db4a8a9a8db5c3a2a10820a70ab211d4
factor count = 1610

Top100 StrategyInput
runs/baseline_strategy_inputs/
  a666a8b1a7db2d7e4f6020362f41510eae7b489ae91594dbe1d7a9d9086a6bc8

Development Matrix
runs/development_factor_matrices/
  98a5a626ed30130bc52b2610ed3a6243293012c9e02709fdd6a5ae44b3037de4

Static Strategy Bundle
runs/baseline_strategy_bundles/
  5e058b0ad182ef329584ee060d22d3f8d2d070c561dcf3e86168c100804263d3
```

D1 冻结完整 1610 因子；Top100 只是 Equal Weight、Fixed ICIR 和 LightGBM 的统一策略输入，不改变完整 Factor Pool。

### 3.5 OOS

```text
Test Score Artifact
runs/oos_test_scores/
  63d77fbd3bf23aaccbd8a25c38cc27b79ddc895f1d5514679ddc63b2586433c0

OOS Evaluation
runs/oos_baseline_evaluations/
  9d271368528d002a8af0807c807042a0e93f6e6b21e5190d65a378deacdc7951
```

OOS evaluation 状态为 `complete_verified_oos`，241 个调仓期、0 个无效期。主要结果：

| Strategy | Annual excess return | Excess IR |
|---|---:|---:|
| Equal Weight | 11.64% | 1.3084 |
| Fixed ICIR | 11.96% | 1.3860 |
| LightGBM | 17.30% | 1.9730 |

Equal Weight 与 Fixed ICIR 的 OOS score correlation 为 0.9971637，并非严格等于 1。LightGBM 平均单边换手率约 81.97%，只作为后续策略研究 caveat，不允许反向调 Baseline。

## 4. 当前真实 cleaning 与 missing 合同

Strategy Matrix 顺序：

```text
Raw expression
→ cross-sectional 1%/99% winsorization
→ point-in-time SW level-1 industry neutralization
→ cross-sectional z-score
→ factor-specific nonfinite to zero on base-eligible stocks
→ frozen Train direction
```

Train、Validation 与 Test 复用同一合同。当前不存在 size / market-cap neutralization。OOS eligibility 在 cleaning 和因子特定 nonfinite 填 0 后形成，不要求 Top100 原始值同时非缺失。

任何日频衍生特征实验都必须明确它是在“表达式输入字段层”增加新 raw fields，还是改变其他层。默认不允许借增加字段之名改变上述截面 cleaning、Reward、Stage 6、方向或 OOS 定义。

## 5. 正式 Notebook 入口

Baseline 的权威执行顺序：

1. `notebooks/download_data.ipynb`
2. `notebooks/prepare_daily_data.ipynb`
3. `notebooks/prepare_industry_data.ipynb`
4. `notebooks/run_stage5_hybrid_variance_real_5_15.ipynb`
5. `notebooks/stage5_reporting.ipynb`
6. `notebooks/run_stage6_hybrid_formal_selection.ipynb`
7. `notebooks/stage6_reporting.ipynb`
8. `notebooks/run_baseline_freeze_and_oos.ipynb`
9. `notebooks/oos_baseline_evaluation.ipynb`

这些入口属于 Raw Daily Baseline v1。下一实验应创建独立配置/入口或明确参数化入口，不能覆盖其中已绑定的正式 Baseline 路径和 artifact。

## 6. Legacy 与受保护依赖

Conditional 改造前只保留：

- grammar Primary：`runs/real_search/d521789d86de425794a9e871b42db586`；
- flat Secondary：`runs/real_search/8778d49870c244a6996e31aa49f40e45`；
- `notebooks/legacy_gflownet_conditional_motivation.ipynb`；
- grammar-only 历史入口 `notebooks/run_real_candidate_search.ipynb`。

当前 Hybrid Notebook 仍只读使用：

```text
runs/complexity_diagnostic_6_20/
  manual_diagnostic_6_20_seed42/exhaustive_registry.sqlite3
```

它是 `N=1/2` Exact-TB registry。父目录名称虽然带旧 diagnostic，但当前仍有真实依赖；不得按名称删除或迁移。迁移必须作为独立工作包完成内容/哈希校验、引用更新、focused tests 和 step-zero preflight。

## 7. 下一工作包｜日频衍生特征实验

这是 Baseline v1 之后的独立实验版本，不是 continuation run，也不得 resume 当前 Baseline checkpoint。

目标顺序：

```text
确认衍生特征研究定义
→ 冻结新的 data / feature contract
→ 最小实现与测试
→ 新训练配置、新 run ID、新输出目录
→ 轻量 preflight
→ 用户手动启动正式训练
```

### 7.1 实现前必须确认的研究选择

当前尚未冻结日频衍生特征的具体清单。新窗口不能自行猜测。至少先与用户确认：

1. 新字段名称与逐项数学公式；
2. 每个字段使用的原始输入和复权口径；
3. 是否使用 lag，明确 `t` 日可见信息边界；
4. rolling window、`min_periods`、warm-up 和边界行为；
5. 分母为零、停牌、涨跌停、缺失值和无穷值语义；
6. 新字段是在预处理阶段持久化，还是表达式计算时派生；
7. 是否作为 grammar leaf/token 暴露，以及这会如何改变动作空间；
8. 数据 schema、feature ordering、dtype、mask 和 fingerprint；
9. 新实验如何命名，以及与 Raw Daily Baseline 的比较变量。

上述选择会改变数据语义、模型输入和 provenance；未得到用户确认前，只做只读审计和方案，不实现。

### 7.2 必须保持不变的对照条件

若实验目标是识别“加入日频衍生特征”的增量价值，除明确冻结的 feature-contract 差异外，优先保持：

- Train / Validation / Test 时间分段；
- Conditional `N=1..15`；
- grammar-hierarchical policy；
- Exact-TB / LPV routing；
- `K=16`、LR、grad clip、cycles 和 seed；
- Reward、evaluation、cleaning、Stage 6、Factor Pool 和策略定义。

如果实际实现证明某项无法保持，应停止并报告，不得静默扩大实验变量。

### 7.3 输出隔离

新实验必须：

- 使用新 branch；
- 使用新 run ID；
- 使用独立 Stage 5 output root；
- 不写入 Baseline 的 run、checkpoint、candidate artifact 或 reporting 目录；
- 在 config / manifest 中记录 feature schema/fingerprint 和启动代码 commit；
- 不通过复制 Baseline runner state 冒充 resume。

真实训练启动前只做比例适当的检查：字段公式/边界单测、无未来数据检查、schema/fingerprint 检查、grammar leaf 合法性、单 batch 或小型 synthetic smoke、Notebook 路径和安全 gate 检查。不要由 Codex 自动运行正式长训练。

## 8. Git branch / worktree 规划

Baseline v1 已提交，因此可以从 `1cb12de` 建立隔离工作线。当前规划是：

### Worktree A｜日频衍生特征（当前优先）

建议分支：

```text
codex/daily-derived-features
```

建议独立目录示例：

```text
<separate-derived-worktree>
```

用于新增特征、测试、独立配置和正式训练。训练启动后，该 worktree 的核心代码保持不变，以便 checkpoint、resume 和 provenance 对齐。

### Worktree B｜Baseline 展示优化（训练启动后再创建）

建议分支：

```text
codex/post-baseline-presentation
```

建议独立目录示例：

```text
<separate-presentation-worktree>
```

本工作线当前只优先考虑 supplementary / presentation，例如：

- Factor Pool 与 Top100 结构说明；
- Development Matrix missing 与 zero-fill 摘要；
- Equal Weight、Fixed ICIR、LightGBM 差异；
- Baseline OOS 补充诊断图。

新增展示只能读取同一 frozen Baseline artifact，标注为 supplementary，不覆盖 v1，不改变策略、权重、指标、筛选或 OOS 结果。

### 暂缓｜流程轻量化

减少 freeze/reload/hash、增加 fast path 等工程简化暂时不是重点。用户担心这类修改难以充分验证并可能破坏 authority chain，因此放到更远的独立优化任务：

- 当前不改 freeze、reload、hash 或 fingerprint 语义；
- 当前不加入绕过完整校验的快速路径；
- 不把流程轻量化与日频衍生特征混入同一个 commit；
- 将来若开展，先写最小方案、列出不变合同和回归范围，再等待用户批准。

### 数据共享边界

`git worktree` 共享 Git 对象，但不会复制被忽略的 `data/`、`runs/` 和 `outputs/`。创建 worktree 前必须先检查真实数据路径和代码的相对路径假设。

- 原始和 processed 数据优先只读共享；
- 不擅自创建 junction / symlink；
- 不复制大型数据；
- 两个 worktree 不共享可写 checkpoint/run 目录；
- 不覆盖 Raw Daily Baseline artifact。

只有用户明确授权后才实际创建 worktree、junction 或新数据路径。

## 9. 不可反向修改的 Baseline v1 边界

以下内容已冻结，不得因新特征、补充图表或后续结果而覆盖：

- 正式 100-cycle Raw Daily Stage 5 run；
- Stage 6 selection 和 1610 retained ordering；
- frozen Train directions；
- full Baseline Factor Pool fingerprint；
- Top100 StrategyInput fingerprint；
- Static Strategy Bundle；
- Test Score Artifact；
- OOS Evaluation Artifact；
- 已发布的 Baseline 指标和报告口径。

不得利用 Validation/Test/OOS 结果反向选择日频衍生字段、调 Reward、调筛选阈值或修改 Baseline。新实验必须与 Baseline 并列比较，而不是改写 Baseline。

## 10. 当前验证与 Git 状态

Baseline 里程碑提交：

```text
1cb12de feat: freeze conditional hybrid baseline pipeline
```

该提交包括 Conditional Hybrid Stage 5、正式 Stage 6、Factor Pool / StrategyInput、三策略、OOS、reporting、测试、Notebook 和整理后的文档。数据、runs、outputs、checkpoint、SQLite 和模型文件未提交。

提交前完整测试结果：

```text
597 tests
OK
```

之后仅整理文档路径和删除旧下载兼容入口，没有启动真实训练、Stage 6 或 OOS。新窗口仍应检查实时 `git status`，不要只依赖此处记录。

## 11. 给新窗口的明确任务边界

用户当前优先级是：

```text
第一优先：设计并加入日频衍生特征，形成独立实验并启动新训练
第二优先：训练运行期间，在另一 worktree 补充 Baseline 解释性图表
暂缓：freeze/reload/hash 等流程轻量化
```

新窗口的第一轮工作应是轻量、只读的日频衍生特征接入审计，回答：

1. 当前 raw fields、processed schema、feature ordering 和 mask 在哪里定义；
2. grammar leaf/token、policy 输入和 fingerprint 会受哪些文件影响；
3. 怎样用最小改动建立新的 feature contract 和独立 run identity；
4. 哪些具体衍生特征定义仍需用户确认。

完成审计后先给出 3–4 步实现方案和每步验证结果，等待用户确认。不要在第一轮直接修改核心数据/模型合同，不创建 worktree，不启动训练，也不要进入流程轻量化。
