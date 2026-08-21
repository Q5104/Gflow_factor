# 复现指南｜数据、Notebook、训练与验收

本文说明如何从代码环境走到 Raw Daily Baseline 或 Daily-Derived v1 的训练入口，并明确哪些环节当前能够公开复现、哪些依赖本仓库不分发的外部数据或本地 artifact。

状态基准日期：2026-08-21。

## 1. 复现范围

本仓库区分三种“复现”：

1. **代码合同复现**：安装依赖并运行单元测试；不需要真实数据。
2. **训练流程复现**：自行准备全部数据和前置 artifact，通过 Notebook 新建独立 run；不要求得到与作者相同的随机轨迹或最终候选集合。
3. **历史结果逐项复现**：要求相同的数据供应商、数据库 vintage、外部申万逐日文件、依赖环境、fingerprint、seed 和代码提交。由于行情、PIT 行业数据及本地 run/artifact 不随仓库分发，第三方目前无法仅凭 Git checkout 保证做到这一层。

当前正式状态：Raw Baseline 已完整完成；Daily-Derived 已完成数据、vocabulary、N=1/2 Exact-TB 与 Stage 5 入口，但正式 Stage 5 停在 98/100 cycles，正式 Stage 6、Factor Pool、策略和 OOS 尚未开始。

## 2. 环境

已验证的开发环境：

- Windows + PowerShell；
- Python 3.12；
- PyTorch 2.6–2.x；
- NVIDIA CUDA GPU；正式训练使用 `cuda:0`；
- Jupyter Lab。

从仓库根目录安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m jupyter lab
```

如果工作区使用仓库自带的定制环境，解释器可能是 `.\.venv\python.exe`。以下项目维护命令采用该路径：

```powershell
.\.venv\python.exe -m pip check
.\.venv\python.exe -m unittest discover -s tests -v
```

必须从仓库根目录启动 Jupyter。部分新 Notebook 会向上搜索项目根目录，但早期数据 Notebook 使用 `Path.cwd()`；从其他目录启动可能把输入或输出解析到错误位置。

`requirements.txt` 记录兼容范围而非完整 lock。若要逐项重现实验环境，应在开始正式 run 前自行保存 `pip freeze`、Python/PyTorch/CUDA 版本和 Git commit，并把这些信息与本地 run 一同归档；不要把机器路径或大型环境文件提交到仓库。

## 3. 数据目录与保护规则

```text
data/raw/                         # adata 原始缓存
data/download_parts/              # 股票级断点分片
data/processed/                   # Raw axes、tensor、mask、行业和 Barra
data/processed/daily_derived_v1/  # Derived tensor 与 metadata
参考文件/swind/                   # 用户自备申万逐日 PIT CSV
runs/                             # registry、checkpoint、runner state、冻结 artifact
outputs/                          # 报告图表和表格
```

这些路径中的大型产物默认不进入 Git。不要使用 `git clean`、批量删除或覆盖来“重新开始”。不同 Feature Space 必须使用不同 registry、run 和 output root；配置或 fingerprint 改变时必须新建 run。

## 4. 行情与股本数据

打开：

```text
notebooks/download_data.ipynb
```

该 Notebook 使用 `adata` 下载四份必需缓存：

```text
data/raw/adata_listing_dates.parquet
data/raw/market_data.parquet       # k_type=1, adjust_type=2
data/raw/raw_close.parquet         # k_type=1, adjust_type=0
data/raw/stock_shares_history.parquet  # 历史 total/list_a_shares
```

下载任务可中断恢复。保持 `force_update=False`，重新运行对应单元会读取 `data/download_parts/` 中已经成功的股票分片。不要删除分片来处理普通的限流或网络中断。

Notebook 尾部的核心财务指标和分红是附录研究输入，不进入当前五类 Barra-style Reward，也不是重现 Stage 5 的必需项。

不同日期下载得到的数据可能因供应商修订、退市股票覆盖或复权序列变化而不同，因此可能产生不同 metadata hash 和候选结果。

## 5. Raw Daily 数据准备

按顺序打开并人工运行：

```text
notebooks/prepare_daily_data.ipynb
notebooks/prepare_industry_data.ipynb
```

`prepare_daily_data.ipynb` 生成：

```text
data/processed/daily_clean.parquet
data/processed/data_tensor.npy       # (date, 6, stock)
data/processed/valid_mask.npy
data/processed/universe_mask.npy
data/processed/date_list.npy
data/processed/stock_list.npy
data/processed/metadata.json
```

`data_tensor.npy` 的固定 feature order 为：

```text
open, high, low, close, vwap, volume
```

首次执行时先检查路径和下载摘要，再显式运行预处理单元。代码不会修改 `data/raw/`。

## 6. 用户自备申万逐日 PIT 行业数据

本仓库没有稳定、可公开分发的申万逐日数据源。研究者必须自行取得有合法使用权限的数据，并按下列目录和 schema 提供：

```text
参考文件/swind/swind_YYYYMMDD.csv
```

文件要求：

- UTF-8 或 UTF-8 with BOM；
- 一天一个文件，文件名日期唯一；
- 严格表头顺序：

```text
TradingDay,StockCode,StockName,SWCode1,SWName1,SWCode2,SWName2,SWCode3,SWName3
```

字段约束：

| 字段 | 要求 |
|---|---|
| `TradingDay` | `YYYY-MM-DD`，并与文件名日期一致 |
| `StockCode` | `000001.SZ`、`600000.SH`、`xxxxxx.BJ` 形式 |
| `SWCode1/2/3` | 六位数字加 `.SI`；允许缺失，但非法格式会 fail closed |
| `SWName1/2/3` | 对应层级名称；空字符串转为缺失 |

构建输出：

```text
data/processed/industry_sw_daily.parquet
data/processed/industry_sw_daily_metadata.json
```

输出列为：

```text
trade_date, stock_code, stock_name,
sw_code_1, sw_name_1, sw_code_2, sw_name_2, sw_code_3, sw_name_3
```

Stage 5 行业中性化使用当日 `sw_code_1`。`prepare_industry_data.ipynb` 的只读全量检查与正式构建会校验表头、日期、代码格式、重复键以及与 `daily_clean.parquet` 的完整键对齐。源文件不随仓库分发；不同供应商映射或历史修订会改变研究结果。

## 7. Barra 前置 artifact

真实 Reward context 还要求：

```text
data/processed/barra/metadata.json
data/processed/barra/market_return.npy
data/processed/barra/list_a_shares.npy
data/processed/barra/<style>.npy
```

生产 API 位于 `factor_gfn.barra`：

- `build_barra_auxiliary_arrays(...)`；
- `run_barra_factor_pipeline(...)`；
- `load_barra_factor_set(...)`。

当前仓库**没有独立的正式 Barra 构建 Notebook**。因此，从全新 checkout 只按 Notebook 顺序尚不能完全自包含地生成所有真实训练前置文件。公开复现时必须使用与 `data/processed/` axes 对齐的既有 Barra artifact，或先基于上述生产 API 编写并审查单独的人工入口。不能跳过这些文件、伪造 fingerprint，或把 synthetic fixture 当成真实前置数据。

这是当前复现流程的已知缺口，不应在补齐正式入口前声称仓库可以从零一键复现训练。

## 8. Raw Baseline Stage 5

入口：

```text
notebooks/run_stage5_hybrid_variance_real_5_15.ipynb
```

不要直接 Run All。按 Cell 顺序操作：

1. 环境与 CUDA 检查；
2. 冻结配置；
3. CPU-safe / step-zero preflight；
4. 明确选择 `new` 或 `resume`；
5. 新 run 先执行一个完整 cycle；
6. 检查按 N 诊断、Reward 和耗时；
7. 检查 candidate artifact；
8. 只读验证 checkpoint/resume；
9. 设置绝对累计目标 cycle，分段续训。

新建独立 run：

```python
RUN_REAL_ONE_CYCLE = True
MODE = "new"
RESUME_RUN_DIR = None
```

恢复已有 run：

```python
RUN_REAL_ONE_CYCLE = True
MODE = "resume"
RESUME_RUN_DIR = r"<完整的已有 run 目录>"
```

恢复必须使用原始 config、代码和数据 fingerprint；不得用新配置强制载入旧 checkpoint。Cell 9 的 `TARGET_CYCLE` 是绝对累计 cycle，不是“再跑多少轮”。执行前显式设置 `RUN_TO_TARGET_CYCLE=True`，完成或暂停后恢复为 `False`。

正式 Raw Baseline 历史 run 已完成，公开复现者应创建自己的新 run，不应期待本仓库包含该 checkpoint。

## 9. Daily-Derived v1 数据与 Exact-TB

先完整准备 Raw axes、Raw market context、PIT 行业和 Barra artifact，再运行：

```text
notebooks/prepare_daily_derived_data.ipynb
```

输出：

```text
data/processed/daily_derived_v1/data_tensor.npy
data/processed/daily_derived_v1/metadata.json
```

正式 tensor contract 为 `(date, 16, stock)`、artifact dtype `float32`，date/stock axes 必须与 Raw 完全一致。当前作者 artifact 的 shape 为 `(4027, 16, 5424)`，文件大小约 1.40 GB；第三方数据覆盖不同并不一定得到同一 shape。

完整公式和无未来数据语义见 `docs/daily_derived/DAILY_DERIVED_FEATURE_DESIGN.md`。构建器不覆盖已有正式输出；不要为重跑而删除原 artifact。

随后运行：

```text
notebooks/build_daily_derived_v1_exact_tb_n1_n2.ipynb
```

先执行只读 count、vocabulary、数据和路径预检，再人工设置：

```python
RUN_REAL_EXHAUSTIVE = True
```

该长任务只构建 Derived 独立 registry 和 exact TB logZ，不创建 Stage 5 run。目标路径：

```text
runs/daily_derived_v1/exact_tb_n1_n2/exhaustive_registry.sqlite3
```

Raw 与 Derived registry 的 Reward、feature/action fingerprint 和 logZ 不可交叉复用。

## 10. Daily-Derived v1 Stage 5

入口：

```text
notebooks/run_stage5_daily_derived_v1_hybrid_variance_real_5_15.ipynb
```

操作原则与 Raw 相同，但该 Notebook 显式绑定：

- `daily_derived_v1` 16-leaf tensor；
- 152-action vocabulary；
- Derived N=1/2 Exact-TB registry；
- `runs/daily_derived_v1/stage5_hybrid_variance_real_5_15/` 独立根目录。

公开提交态的 Notebook 已清空输出，并恢复为 `RUN_REAL_ONE_CYCLE=False`、`MODE='new'`、`RESUME_RUN_DIR=None`、`RUN_TO_TARGET_CYCLE=False`。因此它可以安全地执行环境检查和只读 preflight，但不会在 Run All 时自动创建或恢复正式 run。

正式执行时仍须人工逐格操作：新实验显式启用 one-cycle gate；恢复已有 run 时把 `MODE` 改为 `resume` 并填写真实目录；累计训练前再单独启用 continuation gate。不得复用 README 中作者的本地 run 路径。运行完成或暂停后，应重新关闭所有执行闸门并清空输出再提交。

作者当前正式 run 停止在：

```text
complete=false
global_optimizer_step=1470
total_trajectories_seen=23520
cycle_index=98
```

因此仍是 partial training。不能基于中间 Reward 或 candidate 数量宣称 Derived 优于 Raw。

## 11. 完成验收与恢复检查

不要用 Notebook 内存、最后一行打印或进度条判断完成。至少同时检查：

```text
runner_state.json
checkpoint_latest.pt
train_candidate_artifact.json
hybrid_diagnostics.jsonl
hybrid_run_config.json
```

Stage 5 正式完成目标：

```text
complete=true
global_optimizer_step=1500
total_trajectories_seen=24000
pending assignment 与 complete 状态一致
candidate artifact committed step=1500
checkpoint/config/provider/action fingerprints 一致
```

若 Notebook 重启后从 cycle 0 开始，先检查是否误用了 `MODE='new'` 或 `RESUME_RUN_DIR=None`。恢复时必须明确指向正确 run 目录；不要通过复制 runner state、改 fingerprint 或新建同名目录冒充 resume。

## 12. Stage 6、策略和 OOS

Raw Baseline 的后续正式入口依次为：

```text
notebooks/stage5_reporting.ipynb
notebooks/run_stage6_hybrid_formal_selection.ipynb
notebooks/stage6_reporting.ipynb
notebooks/run_baseline_freeze_and_oos.ipynb
notebooks/oos_baseline_evaluation.ipynb
```

`run_baseline_freeze_and_oos.ipynb` 在 OOS gate 前默认停止；只有所有 Development 决策和 artifact 冻结后才能显式解锁 Test。

这些 Notebook 当前绑定 Raw Baseline 的 run 和 artifact，**不能用于 Derived 正式执行**。Derived 目前只有 compatibility/synthetic plumbing：

- 没有正式 Derived Stage 6 Notebook；
- 没有正式 Derived Factor Pool 或 Top100；
- 没有正式 Derived Strategy Bundle、Test scores 或 OOS evaluation；
- synthetic Test fixture 不等于读取真实 Test labels。

Derived Stage 5 完成前不得进入正式 Stage 6。未来 Derived 流程必须创建独立输出根并复用同一筛选和策略数学，不能覆盖或重新解释 Raw Baseline。

## 13. 验证命令

环境检查：

```powershell
.\.venv\python.exe -m pip check
```

完整代码回归：

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
```

Derived 关键 focused tests：

```powershell
.\.venv\python.exe -m unittest `
  tests.test_daily_derived `
  tests.test_daily_derived_artifact `
  tests.test_dual_feature_space `
  tests.test_daily_derived_stage5_training_notebook `
  tests.test_derived_strategy_plumbing `
  -v
```

自动测试和 syntax check 只能证明代码合同或 synthetic fixture；它们不能证明真实下载、完整 artifact、训练、Stage 6 或 OOS 已完成。

## 14. 常见问题

### `factor_gfn` 无法 import

确认 Jupyter 从仓库根目录启动。若代码刚更新过，先重启 kernel，避免旧模块留在内存。

### CUDA 不可用

可以执行 CPU-safe preflight，但正式 Stage 5 会被门禁阻止。检查 PyTorch/CUDA 匹配、驱动和 Jupyter kernel 使用的解释器。

### runner 又从 cycle 0 开始

停止执行，检查 `MODE`、`RESUME_RUN_DIR` 和真实 `runner_state.json`。不要继续写入意外新 run，也不要删除它。

### 数据 hash 与作者不同

检查下载日期、adata 版本、复权口径、股票覆盖、申万源文件、数据库 vintage、Raw axes 和 Derived metadata。不同外部数据不能通过修改 metadata 或跳过 fingerprint 校验来伪装一致。

### 想缩短训练或改变参数

这会形成新实验。使用新 run、独立输出和新的 config fingerprint，不得 resume 正式 Baseline 或 Derived v1 checkpoint，也不得与冻结结果直接混为同一实验。
