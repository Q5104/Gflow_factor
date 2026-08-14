# 历史参数诊断 Notebook

本目录只保存已经完成的 Stage 5 参数与架构诊断证据，不是正式训练入口。

- 不得从这些 Notebook 恢复或创建正式训练 run。
- 不得把其中的短期 Reward/IC 结果用于重新选择已冻结参数。
- 对应 `runs/` 产物继续只读保留，用于 exact Z、历史 logZ、targeted calibration 和决策审计。
- 当前唯一正式训练入口位于 `notebooks/run_stage5_no_anchor_formal_6_20.ipynb`。

归档内容包括 6/20 complexity/depth diagnostic、Step 12 training-health、N=17/18 targeted calibration、policy clipping 对照及最终 health confirmation。
