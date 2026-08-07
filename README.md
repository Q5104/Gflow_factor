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
- 点时行业：已生成与 `daily_clean.parquet` 键完全一致的三级行业长表，并完成一级行业中性化接入验证。
- 当前测试：178 项单元与集成测试通过；真实长任务仍由使用者在 Notebook 中手动运行。
- 待推进：阶段 5 因子筛选与回测验证；当前五步训练仅为工程验收，不代表研报规模的正式训练参数或结果。

## 项目结构

```text
data/                       # 本地原始数据、断点与处理结果（不提交 Git）
factor_gfn/
├── data/                   # 下载、预处理、行业与股票池
├── grammar/                # Token、算子注册表、部分 AST 与 DAG 文法
├── evaluator/              # 数值算子、解释器、截面清洗与指标
├── barra/                  # 五个 Barra 风格因子及独立多空收益序列
└── gfn/                    # Transformer、采样、Reward、TB Loss 与 Trainer
notebooks/                  # 手工下载、处理和分阶段验证入口
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

建议按需运行：

1. `download_data.ipynb`：下载与续传 adata 原始数据（不再下载行业）；
2. `prepare_daily_data.ipynb`：构造 VWAP、清洗、mask 和六特征矩阵；
3. `prepare_industry_data.ipynb`：将逐日申万三级 CSV 对齐为点时行业长表；
4. `validate_stage2_grammar.ipynb`：验证文法、DAG、动作空间和表达式转换；
5. `validate_stage3_evaluator.ipynb`：验证解释器、算子和 5 日指标；
6. `barra_long_short_analysis.ipynb`：检查五个 Barra 风格多空收益序列；
7. `validate_stage4_synthetic_training.ipynb`：不依赖真实数据的 GFlowNet 合成训练闭环。
8. `validate_stage4_real_reward.ipynb`：在人工表达式上检查真实 Reward、性能、内存、行业中性化和缓存。
9. `validate_stage4_real_training.ipynb`：执行 CPU 五步最小真实训练，并验证第 4 步检查点恢复及第 5 步确定性续跑。

长时间下载、真实数据处理和正式训练均由使用者手动启动。续传时保持 `force_update=False`，不要删除尚未完成的数据断点。

## 研究口径

研报明确披露的内容与本项目工程假设必须分开记录。当前部分 AST、多路径 DAG、收益标签精确索引、Barra 构造及部分数值边界属于第一版复现规范，不应表述为研报内部实现。提交代码前请同步更新测试与 `DEVELOPMENT_SPEC.md`。
