# GFlowNet 日频 K 线因子挖掘

本项目以 A 股日频 K 线为输入，复现并扩展基于 GFlowNet 的量价因子表达式搜索。第一版使用
`open/high/low/close/vwap/volume` 六个特征；研究口径、工程假设和变更记录以
[`DEVELOPMENT_SPEC.md`](DEVELOPMENT_SPEC.md) 为准。

## 当前进度

- 数据层：支持 adata 股票主表、后复权行情、不复权收盘价、历史股本和申万行业的断点下载与 QA。
- 预处理：已实现后复权 VWAP、六特征矩阵、有效值 mask 和股票池 mask。
- 阶段 2：已实现 52 个算子、142 个 Token、规范化部分 AST、多路径 DAG 和固定均匀后向概率。
- 阶段 3：已实现 NumPy 因子解释器、截面清洗、行业中性化、5 日指标和五类 Barra 风格因子。
- 阶段 4：已实现路径条件化 Transformer、可微轨迹采样、Reward、TB Loss、可学习 `logZ`、Trainer 与检查点，并通过合成训练闭环验证。
- 待推进：真实数据 Reward 接入、正式训练、候选因子归档、阶段 5 筛选与回测。行业数据不完整时可运行合成训练，但不能完成正式行业中性化评价。

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
参考文件/                   # 本地研报与流程资料（不提交 Git）
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

1. `download_data.ipynb`：下载与续传原始数据；
2. `prepare_daily_data.ipynb`：构造 VWAP、清洗、mask 和六特征矩阵；
3. `validate_stage2_grammar.ipynb`：验证文法、DAG、动作空间和表达式转换；
4. `validate_stage3_evaluator.ipynb`：验证解释器、算子和 5 日指标；
5. `barra_long_short_analysis.ipynb`：检查五个 Barra 风格多空收益序列；
6. `validate_stage4_synthetic_training.ipynb`：不依赖真实数据的 GFlowNet 合成训练闭环。

长时间下载、真实数据处理和正式训练均由使用者手动启动。续传时保持 `force_update=False`，不要删除尚未完成的数据断点。

## 研究口径

研报明确披露的内容与本项目工程假设必须分开记录。当前部分 AST、多路径 DAG、收益标签精确索引、Barra 构造及部分数值边界属于第一版复现规范，不应表述为研报内部实现。提交代码前请同步更新测试与 `DEVELOPMENT_SPEC.md`。
