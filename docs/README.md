# 文档导航

本目录保存公开复现说明、冻结设计合同、开发日志和阶段性交接。实际代码与测试是可执行合同；文档发生冲突时，先停止操作并核对最新代码、artifact 和相应的权威设计文件。

## 新读者阅读顺序

1. [`../README.md`](../README.md)：项目目标、Raw/Derived 状态、结果边界和 Notebook 总入口；
2. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)：环境、外部数据 schema、Notebook 闸门、new/resume 与 artifact 验收；
3. [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md)：公开提交前的已通过项、阻塞项和 commit 分组；
4. [`../DEVELOPMENT_SPEC.md`](../DEVELOPMENT_SPEC.md)：项目级研究和工程决策；
5. [`../BASELINE_DEVELOPMENT_LOG.md`](../BASELINE_DEVELOPMENT_LOG.md)：已冻结 Raw Baseline 的完成证据；
6. [`daily_derived/DAILY_DERIVED_FEATURE_DESIGN.md`](daily_derived/DAILY_DERIVED_FEATURE_DESIGN.md)：Daily-Derived v1 公式和数据合同；
7. [`daily_derived/DAILY_DERIVED_FEATURE_DEVELOPMENT_LOG.md`](daily_derived/DAILY_DERIVED_FEATURE_DEVELOPMENT_LOG.md)：Derived 当前实现和运行边界。

## 权威性

| 文档 | 用途 | 边界 |
|---|---|---|
| `DEVELOPMENT_SPEC.md` | 项目级决策记录 | 历史章节可能描述已被后续设计替代的路线 |
| `BASELINE_DEVELOPMENT_LOG.md` | Raw Baseline v1 完成状态 | Baseline 已冻结，不由 Derived 结果改写 |
| `DAILY_DERIVED_FEATURE_DESIGN.md` | Derived v1 公式、PIT、lag、NaN、schema | 公式为工程重建，不是研报披露的精确公式 |
| `DAILY_DERIVED_FEATURE_DEVELOPMENT_LOG.md` | Derived 实施与验证记录 | partial training 不等于正式完成 |
| `REPRODUCIBILITY.md` | 面向第三方的实际执行说明 | 明确外部数据和 Barra 入口的当前缺口 |

## 专项设计与交接

```text
stage5_hybrid/
  STAGE5_HYBRID_VARIANCE_DESIGN.md
  STAGE5_HYBRID_VARIANCE_DEVELOPMENT_LOG.md

handoffs/
  HANDOFF_BASELINE_V1_TO_DAILY_DERIVED_FEATURES.md
```

`handoffs/` 中的文件用于阶段切换，可能包含创建当时的历史快照。接管工作必须重新读取 `git status`、runner state、checkpoint 和 artifact，不能把旧交接数字当成实时状态。

## 授权与受限材料

本仓库当前没有开源许可证，不授权第三方复用。原始行情、申万 PIT 行业数据、本地 run/checkpoint、模型、registry、报告输出和未审查的演示文稿不属于公开 Git 内容。
