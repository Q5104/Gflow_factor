# GFlowNet 日频 K 线因子挖掘

本项目以 A 股日频 K 线为输入，复现并扩展基于 GFlowNet 的量价因子表达式搜索。第一版使用
`open/high/low/close/vwap/volume` 六个特征；研究口径、工程假设和变更记录以
[`DEVELOPMENT_SPEC.md`](DEVELOPMENT_SPEC.md) 为准。

## 当前进度

- 数据层：支持 adata 股票主表、后复权行情、不复权收盘价和历史股本的断点下载与 QA；申万行业改用 `参考文件/swind/` 逐交易日一、二、三级点时 CSV。
- 预处理：已实现后复权 VWAP、六特征矩阵、有效值 mask 和股票池 mask。
- 阶段 2：已实现 52 个算子、142 个 Token、规范化部分 AST、多路径 DAG 和固定均匀后向概率。
- 阶段 3：已实现 NumPy 因子解释器、截面清洗、行业中性化、5 日指标和五类 Barra 风格因子。
- 阶段 4：已实现路径条件化 Transformer、真实数据上下文、真实 Reward Provider、TB Loss、Trainer 与确定性检查点恢复；合成闭环、真实 Reward 预检和 CPU 五步最小真实训练均已通过。
- 点时行业：已生成与 `daily_clean.parquet` 键完全一致的三级行业长表，并完成一级行业中性化接入验证。正式 Reward 采用 fail-closed 口径：行业回归无法执行时排除整个候选日期，缺失行业股票不进入清洗后截面，并持久化逐日期样本数和失败原因；全局调仓相位不移动、不补位。
- 正式 Reward 性能优化批次一已完成：表达式仍在完整 2010–2018 历史上解释以保留时序 warmup，但 IC、Long IR、候选 LS 与 Barra 相关只保留冻结的 386 个调仓截面；候选截面仅清洗一次并由全部 Reward 分量复用。审计日期和行号仍指向原始评价日期轴。
- 因子解释性能优化批次二已完成：正式只读 `float64` mmap 不再在解释器初始化时复制；叶子在表达式栈中使用只读视图，数值算子不再无条件复制每个输入或二次复制自身新建的输出。可写输入、非 `float64` 输入和含 Inf 输入仍复制并规范化，解释器对外返回的因子矩阵保持独立可写。
- 滚动统计 CPU 优化已完成：`ts_sum/mean/std/wma/slope/residual/zscore/cov/corr/beta/orth` 在保持完整窗口、`ddof=0`、NaN 重置、回归方向和因果边界的前提下，进一步迁入预热后的串行 Numba 状态核；不启用并行。
- Rank 与循环算子 CPU 优化已完成：`ts_rank/max/min/argmax/argmin/position/range/ema` 以及 `cs_rank/cs_quantile/cs_rank_gauss` 使用预热后的 Numba 内核，继续保持平均并列排名、最近极值位置和缺失重置合同。
- Reward CPU 优化已完成：386个点时行业截面只编码一次；完整行业哑变量 OLS 以数学等价的行业组均值投影实现；RankIC 使用不落地完整排名面板的串行内核，多头与多空组合复用稳定排序。fail-closed 日期、审计字段、Reward 公式和五项 Barra 相关口径不变。
- 子表达式 LRU 实现仍保留且可配置，但最新50步实测仅命中 `3/6425` 并淘汰 `6417` 次，因此正式搜索默认上限改为0；Provider 完整表达式 Reward 缓存继续启用。该调整只消除哈希和大矩阵淘汰开销，不删除任何旧候选。
- 正式搜索控制台不再逐条打印已持久化审计的 `IndustryNeutralizationWarning`，只显示每步训练摘要；完整剔除日期、原因和样本数仍保存在候选评价记录中。
- 平坦 Token 策略300步和 `arity_hierarchical` 元数分组策略100步均已冻结为历史对照。元数分组相对平坦策略明显提高Reward/IC尾部和训练侧通过密度，但唯一结构率下降、唯一候选节点P50升至14且15节点占比升至48.2%，表现为重复短叶子与长表达式两极化。两类旧策略及其run只保留用于审计、历史检查点读取和阶段6候选导入，不再由正式Stage 5入口创建或恢复。
- 正式 Stage 5 已升级为 exact-N complexity-conditioned no-anchor GFlowNet：`max_depth=6/max_nodes=20`，全部可达 N=1...20 通过同一个 balanced scheduler 参加 normal discovery；N=1/2 使用 exhaustive registry 与固定 exact TB logZ，N=3...20 使用独立 learned `logZ_N`。旧 anchor、scalar-logZ 和旧 Stage 5 checkpoint 不得进入新 Trainer。
- 正式配置固定 batch 8、policy/logZ 学习率 `1e-4/1e-2`、两组 `max_norm=5`、same-N retry 3、seed 42 和 1000 logical-step 上限；配置指纹及初始化来源写入 no-anchor checkpoint。validation/OOS 在正式搜索中保持未加载。
- 正式搜索继续记录文法层级、窗口概率、每 N 请求/有效/成功暴露、retry、TB delta、梯度裁剪、实际参数更新、吞吐、显存和候选审计。所有候选表达式及旧 run 保留用于阶段 6 导入，不因训练口径更新而删除。
- Trainer采样热区已将每个动作的策略、文法和Window诊断改为GPU批量累计、每轮补采统一回传审计；采样RNG顺序、候选表达式、Reward与训练目标不变。逐步性能新增采样、完整Provider、训练更新及TB前向/backward/optimizer耗时，可从控制台、`step_metrics.jsonl`、TensorBoard和只读监控定位剩余瓶颈；真实CUDA收益仍需手工续跑实测。
- 阶段 6 第一版筛选合同已冻结：2010-2018 训练指标与 2019-2020 验证指标联合执行研报硬筛选；训练期确定方向、按 `abs(train_ic)` 排序，并计算 `barra_ts_corr < 0.7` 与多头超额收益相关性 `< 0.7` 的贪心池结构。验证期对应相关性只作稳定性诊断，2021-2025 最终样本外在 Alpha 池冻结前不参与任何选择。
- 清理后的完整自动化回归共 344 项，全部通过。
- 当前正式入口为 `run_stage5_no_anchor_formal_6_20.ipynb`；真实长任务仍由使用者手动运行。参数诊断 Notebook 已移入 `notebooks/archive/diagnostics/`，只作历史证据，不得恢复为正式训练入口。

## 项目结构

```text
data/                       # 本地原始数据、断点与处理结果（不提交 Git）
factor_gfn/
├── data/                   # 下载、预处理、行业与股票池
├── grammar/                # Token、算子注册表、部分 AST 与 DAG 文法
├── evaluator/              # 数值算子、解释器、截面清洗与指标
├── barra/                  # 五个 Barra 风格因子及独立多空收益序列
└── gfn/                    # Transformer、采样、Reward、TB Loss 与 Trainer
notebooks/                  # 手工下载、数据准备与唯一正式训练入口
tests/                      # 不依赖真实行情的单元与集成测试
tmp/                        # 临时图表、检查点和调试输出（不提交 Git）
参考文件/                   # 本地研报、流程资料与 swind 点时 CSV（不提交 Git）
```

## 环境与测试

项目当前使用 Python 3.12。进入根目录后：

```powershell
conda activate .\.venv
python -m pip install -r .\requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

也可以不激活环境，直接使用 `.\.venv\python.exe` 对应的项目内解释器。

## Notebook 工作流

```powershell
.\.venv\python.exe -m jupyter lab
```

当前入口：

1. `download_data.ipynb`：下载与续传 adata 原始数据（不再下载行业）；
2. `prepare_daily_data.ipynb`：构造 VWAP、清洗、mask 和六特征矩阵；
3. `prepare_industry_data.ipynb`：将逐日申万三级 CSV 对齐为点时行业长表；
4. `run_stage5_no_anchor_formal_6_20.ipynb`：唯一正式 Stage 5 入口，只允许新 no-anchor schema；每个 step 用一行输出关键训练、TB、retry、吞吐和显存指标。

`run_real_candidate_search.ipynb` 仅保留为旧输出格式参考，不得创建或恢复当前正式 run。已经完成的参数诊断位于 `notebooks/archive/diagnostics/`，其代码和输出均只读归档。

长时间下载、真实数据处理和正式训练均由使用者手动启动。续传时保持 `force_update=False`，不要删除尚未完成的数据断点。

阶段 5 训练可在另一个 PowerShell 中进行独立只读监控，不会加载或修改检查点：

```powershell
.\.venv\python.exe -m factor_gfn.gfn.search_monitor watch --run-dir "runs\real_search\<run_id>" --interval 10
.\.venv\python.exe -m factor_gfn.gfn.search_monitor export-tensorboard --run-dir "runs\real_search\<run_id>"
.\.venv\python.exe -m tensorboard.main --logdir "runs\real_search\<run_id>\tensorboard" --host 127.0.0.1 --port 6006
```

浏览器访问 `http://127.0.0.1:6006/`。若只需查询一次，把 `watch` 改为 `status` 并删除 `--interval 10`。`export-tensorboard` 仅用于把监控功能接入前已经完成的历史步骤一次性导出；新步骤由训练入口实时写入。

## 研究口径

研报明确披露的内容与本项目工程假设必须分开记录。当前部分 AST、多路径 DAG、收益标签精确索引、Barra 构造及部分数值边界属于第一版复现规范，不应表述为研报内部实现。提交代码前请同步更新测试与 `DEVELOPMENT_SPEC.md`。
