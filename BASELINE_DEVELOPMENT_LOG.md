# Raw Daily Conditional Hybrid Baseline Development Log

> 状态：Baseline 全链路已完成并冻结  
> 最近同步：2026-08-18  
> 用途：记录当前 Baseline 的权威来源、真实完成结果、已知 caveat 与后续不可反向修改的边界。

## 1. Baseline 定义

```text
Raw Daily data
→ Conditional Hybrid Stage 5
→ Formal Stage 6 Train/Validation selection
→ Full Baseline Factor Pool freeze
→ Frozen Top100 StrategyInput
→ Development Matrix + three static strategies
→ Frozen Test scores
→ Frozen OOS evaluation + reporting
```

数据分段固定为：

- Train：2010-01-01 至 2018-12-31；
- Validation：2019-01-01 至 2020-12-31；
- Test/OOS：2021-01-01 至 2025-12-31。

## 2. Stage 5｜Conditional Hybrid 候选发现

正式配置：

- grammar hierarchical；
- external condition `N=1..15`；
- `N=1/2` fixed Exact-TB，`N=3..15` direct LPV；
- `K=16`，policy learning rate `1e-4`，global gradient clip `5`；
- 100 cycles = 1500 optimizer steps = 24000 trajectories；
- Validation 与 Test/OOS 未进入训练或 Reward。

权威 run：

```text
runs/stage5_hybrid_variance_real_5_15/
  hybrid_5_15_k16_seed42_20260816T025559Z
```

完成结果：

- runner complete：true；
- pending assignment：none；
- checkpoint、diagnostics、candidate artifact 与 runner state 均对齐 step 1500；
- unique candidates / structural hashes：21261；
- config fingerprint：`0fe6af55b7b6ab0078df051f76ef6d342478ce35850f541af078033907f16243`。

正式报告：

```text
outputs/stage5_reporting/report_manifest.json
version = Raw Daily Baseline / Stage 5 Reporting v1
15 figures + 18 tables
```

已知 caveat：90.1% updates 的 pre-clip norm 大于 5，`N=4..15` clipping trigger 接近 100%。这是 Baseline 结果事实，不在本日志中改动 LR、clip、Reward 或训练合同。

## 3. Stage 6｜正式 Train/Validation 筛选

权威目录：

```text
runs/stage6/hybrid_provisional/
  hybrid_5_15_k16_seed42_20260816T025559Z
```

Stage 6 只接受上述单一 completed Hybrid source，实际漏斗为：

```text
Stage 5 accepted candidates          21261
Train prefilter / Validation entry    6011
Six-item hard-filter pass             2815
Train long-excess decorrelation       1610
```

`2815` 是 hard-filter pass 与 decorrelation input；`1610` 是按冻结顺序完成 Train long-excess greedy decorrelation 后的 Provisional Factor Pool。二者语义不同，不可互换。

Selection fingerprint：

```text
b96a163e797ec7d803fc423f6625f8614bd55ddfdf827a4bd11a157bd80484fb
```

正式 Stage 6 报告：

```text
outputs/stage6_reporting/report_manifest.json
15 figures + 15 tables
```

## 4. D1 / E3｜Factor Pool、StrategyInput 与策略冻结

完整 Baseline Factor Pool：

```text
runs/baseline_factor_pools/
  f9a3945945ee04eb357896b7b8e20d63db4a8a9a8db5c3a2a10820a70ab211d4
factor count = 1610
```

D1 对 Stage 6 retained records 原样冻结：不重新筛选、不重排、不改 direction、不重新 decorrelate。

固定 StrategyInput：

```text
runs/baseline_strategy_inputs/
  a666a8b1a7db2d7e4f6020362f41510eae7b489ae91594dbe1d7a9d9086a6bc8
input = first 100 factors from full frozen ordering
```

Top100 只是三策略的统一输入，不改变完整 1610-factor Pool。Development Matrix、Strategy Bundle 与 OOS authority 同时绑定 full pool fingerprint 与 StrategyInput fingerprint。

Development Matrix：

```text
runs/development_factor_matrices/
  98a5a626ed30130bc52b2610ed3a6243293012c9e02709fdd6a5ae44b3037de4
Train rows = 792883
Train finite labels = 778473
Validation rows = 303575
Validation finite labels = 302374
factor count = 100
```

Static Strategy Bundle：

```text
runs/baseline_strategy_bundles/
  5e058b0ad182ef329584ee060d22d3f8d2d070c561dcf3e86168c100804263d3
strategies = Equal Weight / Fixed ICIR / LightGBM
```

## 5. OOS｜一次性 Test 评价

Test Score Artifact：

```text
runs/oos_test_scores/
  63d77fbd3bf23aaccbd8a25c38cc27b79ddc895f1d5514679ddc63b2586433c0
```

OOS Evaluation：

```text
runs/oos_baseline_evaluations/
  9d271368528d002a8af0807c807042a0e93f6e6b21e5190d65a378deacdc7951
status = complete_verified_oos
rebalance periods = 241
invalid periods = 0
```

正式报告：

```text
outputs/oos_baseline/
  9d271368528d002a8af0807c807042a0e93f6e6b21e5190d65a378deacdc7951/
10 figures + 6 tables
```

主要真实结果：

| Strategy | Annual excess return | Excess IR |
|---|---:|---:|
| Equal Weight | 11.64% | 1.3084 |
| Fixed ICIR | 11.96% | 1.3860 |
| LightGBM | 17.30% | 1.9730 |

平均 coverage 99.789%，最低 coverage 98.635%，最低 eligible stock count 3360。Equal Weight 与 Fixed ICIR 的 OOS score correlation 为 0.9971637，不是严格等于 1。LightGBM 平均单边换手率约 81.97%，作为后续策略优化 caveat 保留。

OOS 十分位主图使用：

```text
decile excess return
= decile 5-day return
- same-date evaluation-eligible universe equal-weight benchmark return
```

三个策略共用同一 benchmark；原始绝对收益仍保存在 frozen artifact 中。

## 6. Cleaning 与 missing 合同

当前 Strategy Matrix 的真实顺序：

```text
Raw expression
→ cross-sectional 1%/99% winsorization
→ point-in-time SW level-1 industry neutralization
→ cross-sectional z-score
→ factor-specific nonfinite to zero on base-eligible stocks
→ frozen Train direction
```

Train、Validation 与 Test 复用同一 cleaning contract。不存在 size / market-cap neutralization。当前 `complete_case` 字段名是兼容性命名残留；实际 OOS eligibility 在 cleaning 后、因子特定 nonfinite 填 0 后形成，不要求 Top100 原始值同时非缺失。

## 7. Legacy 保留边界

Conditional 改造前仅保留：

- grammar Primary：`runs/real_search/d521789d86de425794a9e871b42db586`；
- flat Secondary：`runs/real_search/8778d49870c244a6996e31aa49f40e45`；
- 双 run motivation Notebook 与输出；
- grammar-only 历史训练入口 `notebooks/run_real_candidate_search.ipynb`。

flat run 仅保留日志、candidate/evaluation 与 provenance 证据，不保留 checkpoint。旧 arity/no-anchor/AB/失败 Hybrid/resource-limited 结果不属于当前 Baseline。

## 8. 依赖保护

`notebooks/run_stage5_hybrid_variance_real_5_15.ipynb` 当前仍只读使用：

```text
runs/complexity_diagnostic_6_20/
  manual_diagnostic_6_20_seed42/exhaustive_registry.sqlite3
```

作为 `N=1/2` Exact-TB registry。即使父目录名称带有旧 diagnostic，也不得按名称删除。本次清理完整保留该目录。若以后迁移，必须复制到新权威资源路径、校验 SQLite 内容与哈希、更新引用并完成 focused tests + step-zero preflight 后，才能删除旧目录。

## 9. 冻结边界与下一阶段

本 Baseline 的 Stage 5 数据、Stage 6 selection、完整 Factor Pool、Top100 StrategyInput、Strategy Bundle、Test scores 与 OOS evaluation 均不再因展示或后续实验反向修改。

后续允许：

- 基于同一 frozen artifact 增加 supplementary visualization；
- 在新分支/新 run 中加入日频衍生特征；
- 在完整跑通一次 Baseline 后简化执行层工程；
- 独立研究 LightGBM 参数、换手约束或策略输入，但不得覆盖当前 Baseline。
