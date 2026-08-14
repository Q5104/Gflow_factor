# GFlowNet 日频 K 线因子挖掘开发规范

> 状态：初版、持续更新  
> 建立日期：2026-08-04  
> 最近同步：2026-08-13
> 当前阶段：阶段 1–4 工程验收及阶段 6 前置防泄露合同已完成；阶段 5 已冻结 `6/20` no-anchor 正式配置并进入 1000-step training-only 正式 run
> 重要说明：本文用于记录开发路线、已确认决策、待确认问题和验收标准，不是不可修改的冻结规格。

## 1. 项目目标

以 A 股原始日频 K 线数据为输入，使用 GFlowNet 在表达式空间中搜索一组具有预测能力、低相关性和结构多样性的量价因子。

当前第一版只处理以下六个输入特征：

- `open`
- `high`
- `low`
- `close`
- `vwap`
- `volume`

其中 OHLC 与 VWAP 使用后复权价格口径，`volume` 保持成交量口径。分钟频数据、日频人工衍生特征、AlphaEval、DPP 和 LGBM 合成不属于当前第一阶段。

第一版目标是完成方法上的最小可运行复现，不要求与研报数值完全一致。

## 2. 依据与决策优先级

### 2.1 主要参考

1. `参考文件/GFlowNet_日频K线因子挖掘完整流程.md`
   - 当前项目的主要开发路线。
   - 包含数据、文法、奖励、模型、训练、筛选和回测的初版流程。
   - 其中标注为“我的推断”或“推荐初始参数”的内容均需通过实际数据和实验确认。
2. `参考文件/国金证券_gflownet挖因子.pdf`
   - 核对研报明确披露内容的首要来源。
3. `参考文件/国金证券_gflownet挖因子.md`
   - 用于全文检索和快速定位，但可能因 PDF 转换遗漏正文、图表或公式。
4. 当前项目的真实代码、数据字段、运行结果和测试结果。

### 2.2 决策原则

- 用户最新确认的项目要求优先于初版流程文档。
- 研报明确披露的内容与工程推断必须分开记录。
- 目录、接口、标签、算子子集、奖励参数和存储格式均可随开发结果调整。
- 每项关键口径应标注为“已确认”“暂定”“待确认”或“暂缓”。
- 发现数据质量、性能、未来数据泄露或实现不可行问题时，应先记录问题和备选方案，再调整实现。

## 3. 当前项目状态

### 3.1 已存在内容

- 独立 Python 3.12 Conda 环境：项目内 `.venv`；
- 下载与预处理：`factor_gfn/data/`，根目录 `data_loader.py` 仅保留兼容入口；
- 表达式文法与部分 AST DAG：`factor_gfn/grammar/`；
- 因子算子、解释器、截面清洗与指标：`factor_gfn/evaluator/`；
- 五个手工 Barra 风格因子及独立多空收益：`factor_gfn/barra/`；
- Transformer 前向策略、轨迹、Reward、TB Loss、Trainer 与检查点：`factor_gfn/gfn/`；
- 手工下载、处理及阶段 2–4 验证入口：`notebooks/`；
- 单元与集成测试：`tests/`，当前完整测试为 237 项；
- 原始数据、断点和处理结果分别位于 `data/raw/`、`data/download_parts/`、`data/processed/`，均不进入 Git；
- Git 仓库已初始化并已建立项目基线提交；后续修改仍需分阶段审查后提交。

### 3.2 当前数据层状态

`factor_gfn/data/downloader.py` 当前负责：

- 获取股票主表；
- 下载 `adjust_type=2` 后复权日频行情；
- 下载 `adjust_type=0` 不复权收盘价；
- 下载历史股本；
- 断点续传、分片合并和轻量 QA。

`factor_gfn/data/preprocess.py` 已实现后复权 VWAP、六特征清洗、有效值 mask、股票池 mask、索引与元数据输出。股票主表、两套行情、历史股本及处理结果已经完成下载或验证。申万行业不再依赖受限流影响的 adata 静态接口，改用 `参考文件/swind/` 的逐交易日一、二、三级 CSV；`factor_gfn/data/industry.py` 负责严格校验、流式合并和行情键对齐，全量转换由用户通过 `prepare_industry_data.ipynb` 手动启动。

### 3.3 当前阶段实现状态

- 阶段 1：核心清洗、VWAP、矩阵化及验证已经完成；
- 阶段 2：52 个算子、142 个 Token、规范化部分 AST、多路径 DAG、父状态枚举和表达式转换已经完成；
- 阶段 3：全部 52 个数值算子、解释器、候选截面清洗、行业中性化、5 日指标及 Barra 风格计算已经完成；
- 阶段 4：路径条件化 Transformer、可微采样、Reward 合同、固定均匀 `P_B`、TB Loss、可学习 `logZ`、Trainer、检查点、合成训练闭环、真实数据上下文、真实 Reward 预检及五步最小真实训练已经完成；
- 点时行业长表已完成全量构建与 QA，并已通过阶段三行业中性化实跑验证；阶段 1–4 的工程验收已完成，阶段 6 前置输入与防泄露合同已提前冻结，当前进入阶段 5 GPU 真实候选训练。五步最小实验只证明真实训练链路与确定性续跑可用，不等同于正式训练结果。

### 3.4 当前执行边界

- 根目录 `data_loader.py` 继续作为兼容入口，正式实现以 `factor_gfn/data/downloader.py` 为准；
- `industry_sw_daily.parquet` 已生成并通过 QA；正式 Reward 必须启用点时申万一级行业中性化，不再允许静默降级；
- 长时间下载、真实数据预处理和正式训练由用户手动启动；
- 不删除未完成的行业断点，不重写原始 Parquet，不将数据、研报、检查点或本地运行结果提交 Git；
- 真实训练前必须再次核对数据指纹、行业覆盖、Reward 配置、训练/验证日期和运行元数据。

## 4. 原始数据合同

### 4.1 当前文件

| 文件 | 口径 | 关键字段 |
|---|---|---|
| `data/raw/adata_listing_dates.parquet` | 股票主表 | `stock_code`, `short_name`, `exchange`, `list_date` |
| `data/raw/market_data.parquet` | `k_type=1`, `adjust_type=2` | `trade_date`, `stock_code`, `open`, `high`, `low`, `close`, `volume`, `amount` |
| `data/raw/raw_close.parquet` | `k_type=1`, `adjust_type=0` | `trade_date`, `stock_code`, `close` |
| `data/raw/stock_shares_history.parquet` | `get_stock_shares(is_history=True)`，下载后生成 | `stock_code`, `change_date`, `total_shares`, `limit_shares`, `list_a_shares`, `change_reason` |
| `参考文件/swind/swind_YYYYMMDD.csv` | 外部逐交易日申万一、二、三级行业源数据，不进入 Git | `TradingDay`, `StockCode`, `StockName`, `SWCode1/SWName1`, `SWCode2/SWName2`, `SWCode3/SWName3` |
| `data/processed/industry_sw_daily.parquet` | 以 `daily_clean.parquet` 行情键左连接生成的点时行业长表 | `trade_date`, `stock_code`, `stock_name`, `sw_code_1/sw_name_1`, `sw_code_2/sw_name_2`, `sw_code_3/sw_name_3` |

### 4.2 已确认 VWAP 口径

```text
adj_factor = close_adj / close_raw
amount_adj = amount_raw * adj_factor
vwap_adj = amount_adj / volume
```

边界条件：

- `close_raw` 缺失、非有限或不大于 0 时，`adj_factor` 无效；
- `volume` 缺失、非有限或不大于 0 时，VWAP 无效；
- 必须核对 `amount` 与 `volume` 的单位，使原始 `amount / volume` 与不复权价格处于一致量纲；
- VWAP 只在预处理阶段生成，不写回原始下载文件。

### 4.3 原始数据保护

- `data/raw/` 视为不可逆修改的原始层；
- 清洗结果写入 `data/processed/`；
- 所有输出使用临时文件加同盘原子替换；
- 原始文件和断点分片不进入 Git；
- 清洗程序必须输出输入路径、行数、股票数、日期范围和参数摘要。

## 5. 研报已核实的方法内容

### 5.1 表达式文法

```text
Expr ::= Feature
       | UnaryOp(Expr)
       | BinaryOp(Expr, Expr)
       | TsUnaryOp(Expr, Window)
       | TsBinaryOp(Expr, Expr, Window)
       | CsOp(Expr)
```

研报窗口集合为：`[5, 10, 20, 40, 60]`。

研报给出一元变换、一元时序、二元组合、二元时序和截面变换五类算子。第一版是否一次实现全部算子仍待工程评估，允许先用可验证的算子子集建立最小闭环。

### 5.2 GFlowNet 与状态表示

- 使用 Trajectory Balance 目标；
- `Z` 为可学习总流量标量；
- `P_F` 为需要学习的前向策略；
- 研报说明 `P_B` 可学习或固定均匀；本项目第一版已确认使用固定均匀后向策略；
- 研报正文将动作历史描述为由 `op/window/feature` 构成并使用 Transformer encoding，但未披露开放槽位选择、部分状态相等判定、父状态枚举或交换等价状态合并算法；
- 本项目将生成环境实现为“规范化部分有序 AST 的多路径 DAG”。这是与研报通用 DAG/TB 原理一致的工程复现口径，不标记为研报披露的内部实现；
- 辅助状态包含当前深度、已用算子比例和已用节点比例。

项目状态图与表达式树属于两个层次。完整表达式仍是有语义参数位置的 AST；GFlowNet 状态则是包含显式空槽位的部分 AST。每个前向转移选择一个开放槽位并填入一个 Token：

```text
ForwardAction = (open_slot, token_id)
```

非交换算子的子节点顺序始终保留。仅 `add`、`mul`、`max2`、`min2` 对完整和部分子树执行交换规范化；`ts_corr`、`ts_cov`、`ts_beta`、`ts_orth` 均保留参数方向。不同槽位填充顺序在规范化后允许汇合到同一状态，从而形成真实的多父状态 DAG。

后向策略对当前状态所有不同父状态固定均匀：

```text
P_B(parent | child) = 1 / n_parents(child)
log P_B = -log(n_parents(child))
```

`n_parents` 按可撤销的前沿节点枚举、撤销后规范化并按父状态键去重计算，不依赖或保存“最后一步”历史。第一版状态图不允许平行边；多个原始槽位操作若产生同一规范后继状态，只保留一个规范转移。

### 5.3 研报奖励函数

```text
reward = abs(train_ic)
       * (1 + LONG_IR_LAMBDA
              * clip(train_long_ir, 0, LONG_IR_CAP))
       * (1 - BARRA_TS_PENALTY_MU
              * clip(barra_ts_corr, 0, 1))
```

以上 reward 结构来自研报；其内部参数数值和 Barra 具体构造方法未在研报中披露。当前项目将以下内容固定为第一版工程复现规范，不标记为研报原文：

- `LONG_IR_LAMBDA` 和 `BARRA_TS_PENALTY_MU` 均为可配置参数；
- 默认 `LONG_IR_LAMBDA=0.3`，用于适度提高多头收益质量；
- 默认 `BARRA_TS_PENALTY_MU=0.2`，用于适度降低传统风险因子暴露；
- 两个参数必须保留网格调整接口，不允许散落硬编码在训练循环中；
- 默认 `LONG_IR_CAP=2` 并保持可配置；当 `train_long_ir>=2` 时不再增加该奖励，配合 `LONG_IR_LAMBDA=0.3`，Long IR 奖励项的最大乘数为 `1.6`。

所有 reward、模型和训练超参数都必须进入统一配置与实验记录。每次正式运行至少记录：参数名、默认值、实际值、参数来源（研报披露或工程假设）、选择理由、候选搜索范围、数据版本、随机种子和最终采用结果。不得只在代码中保留一个无法追溯的数值。

#### 5.3.1 第一版 Barra 风险因子集合

手工构造以下 5 个基础风格因子，用于判断候选表达式是否只是传统风险因子的变形：

1. Market Beta；
2. Size；
3. Momentum；
4. Volatility；
5. Liquidity。

第一版 Barra 惩罚集合仅包含上述 5 个因子。Value、Earnings Yield、Growth、Leverage、Profitability、Dividend Yield 等扩展风格因子全部暂缓，不下载其附加财务或分红数据，也不进入第一版 reward；其候选公式和数据源仅记录在“附录A：暂缓的 Barra 风格因子参考实现”，供后续版本评估。

#### 5.3.2 `barra_ts_corr` 定义

`barra_ts_corr` 不使用候选因子与 Barra 暴露的逐日截面相关性。它使用候选因子及 5 个 Barra 因子的多空收益序列进行时间序列相关分析。

五个 Barra 风格因子分别构造五条独立的 Long-Short 收益序列，不进行因子等权、暴露平均或收益序列合成。这里的“等权”只表示每条风格序列内部的 Top 10% 和 Bottom 10% 股票各自在组合内等权。

在每个 5 日调仓评价期：

- 候选因子只在当期 `universe_mask=True` 且因子有限的截面内，依次执行 1%/99% 缩尾、申万一级行业哑变量 OLS 回归取残差、`ddof=0` z-score；
- 候选因子的行业分类使用 `industry_sw_daily.parquet` 当日 `sw_code_1`。每个调仓日基于当前有效股票池实际出现的申万一级代码构造哑变量，第一版 OLS 含截距，并通过稳定最小二乘求解；`sw_name_1` 仅用于展示和 QA；
- 候选股票缺失行业信息时不参与行业回归，也不进入清洗后的候选截面；任何进入 IC、Long IR 或 Barra 相关性计算的候选股票都必须实际完成行业残差化；
- 若某行业只有一只有效股票，保留该行业哑变量，该股票在可识别的行业回归中允许得到 0 残差；若无已知行业、截面行业回归样本数少于行业数加 1，或 OLS 求解失败，则该候选当日整行保持 NaN。不得退回到未经中性化的缩尾值，不得移动全局调仓相位或用相邻日期补位；
- 每个 Barra 风格暴露仍只执行 1%/99% 缩尾与 `ddof=0` z-score，禁止进行行业中性化或市值中性化；NaN 保持 NaN，常数截面清洗后记为 NaN；
- 候选因子按清洗后暴露排序，Top 10% 等权做多、Bottom 10% 等权做空；
- 每个 Barra 风格暴露使用相同股票池、评价日期和 5 日收益标签，但只执行其自身的“缩尾 → z-score”清洗，随后按 Top 10% 等权做多、Bottom 10% 等权做空；
- 多空收益定义为 `R_LS = R_top10% - R_bottom10%`；
- 相关性只在候选序列与对应 Barra 序列共同有效的调仓期上计算；
- `candidate_corr_k = Corr(R_candidate_LS, R_barra_k_LS)`，第一版使用 Pearson 时间序列相关性；
- 最终惩罚值为：

```text
barra_ts_corr = max_k abs(candidate_corr_k)
```

绝对值使该惩罚对候选因子和 Barra 因子的正负方向不敏感。第一版最低共同有效调仓期数固定为 60，约对应 300 个交易日或 14 个月；该设置用于提高相关性估计稳定性及跨市场状态覆盖概率，但不保证样本必然包含完整牛熊周期。任一候选因子与所有 Barra 序列均不足 60 个共同有效调仓期时，`barra_ts_corr=NaN`；硬性条件 `barra_ts_corr < 0.7` 对 NaN 返回不通过。分位数边界并列值沿用稳定股票代码顺序处理。

点时行业源数据按交易日提供历史分类。使用时以候选因子日期 `t` 的同日 `sw_code_1` 作为行业标签，并继续在实验元数据中记录源文件范围和数据指纹。源文件中出现的上市前回填、退市股票和行情体系外代码不直接进入面板；最终长表必须以 `daily_clean.parquet` 的 `(trade_date, stock_code)` 为左表对齐，避免行业源数据扩张股票池或制造上市前样本。

#### 5.3.3 Barra 数据需求与当前缺口

历史股本下载完成前，当前原始数据只有股票主表、后复权 OHLCVA 和不复权收盘价。五个风格因子的第一版数据需求如下：

| 风格因子 | 候选构造 | 当前数据是否足够 | 仍需确认或补充 |
|---|---|---:|---|
| Market Beta | 252 日滚动 `Cov(stock_return, market_return) / Var(market_return)`，`min_periods=120` | 下载股本后足够 | 市场收益使用前一日流通市值加权全 A 收益 |
| Size | `log(流通市值)` | 下载股本后足够 | `list_a_shares` 与不复权收盘价按变更日向后对齐后计算流通市值 |
| Momentum | `close_adj[t-21] / close_adj[t-252] - 1` | 足够 | 只要求两个端点有效 |
| Volatility | 252 日日收益总体标准差，`min_periods=120` | 足够 | 不年化，不使用市场模型残差 |
| Liquidity | 60 日平均 `volume / list_a_shares` | 下载股本后足够 | 不使用接口 `turnover_ratio`，不做对数变换 |

因此，新增历史股本下载即可覆盖 Market Beta 与 Size 的市值需求；若 Liquidity 选择换手率，也使用 `volume / list_a_shares` 自行计算。第一版不再下载或使用 PB/Value 数据。

Market Beta 第一版口径固定如下：

```text
stock_return_i,t = close_adj_i,t / close_adj_i,t-1 - 1

weight_i,t-1 = market_cap_i,t-1
               / sum_j(market_cap_j,t-1)

R_market,t = sum_i(weight_i,t-1 * stock_return_i,t)

Beta_i,t = Cov_252(stock_return_i, market_return)
           / Var_252(market_return)
```

- `close_adj` 使用后复权收盘价，日收益为相邻交易日收盘到收盘收益；
- 市场组合覆盖当日股票池内具有有效收益和有效滞后市值的全 A 股票；权重在该有效截面重新归一化；
- 权重只能使用前一交易日 `t-1` 已知的市值，禁止使用 `t` 日市值，避免未来信息；
- `market_cap_type` 保持显式配置，第一版默认使用流通市值；若 adata 无法稳定提供可靠的历史流通市值，则允许切换为总市值，但一次实验只能使用一种口径，并必须写入运行元数据；
- “流通市值/总市值归一化”表示选定流通市值或总市值后，以个股市值除以当日有效股票市值合计，不表示以个股流通市值除以个股总市值；
- Beta 使用过去最多 252 个交易日（含评价日 `t` 的当日收益）的滚动总体协方差与总体方差，即 `ddof=0`；股票和市场收益共同有效样本数至少为 120，不足返回 `NaN`；市场收益方差接近 0 时返回 `NaN`；
- 第一版直接对股票收益和市场收益计算 Beta，不减无风险利率；若后续引入超额收益 Beta，必须作为新实验口径另行记录。

### 5.4 研报训练与筛选口径

研报明确披露：

- 训练集：2010-01-01 至 2018-12-31；
- 验证集：2019-01-01 至 2020-12-31；
- 因子评估调仓周期：5 日；
- 评估范围：全 A；
- 硬性筛选条件：
  - `abs(train_ic) > 0.01`
  - `abs(test_ic) > 0.01`
  - `train_ic * test_ic > 0`
  - `train_long_ir > 0.25`
  - `test_long_ir > 0.25`
  - `barra_ts_corr < 0.7`
- 通过硬性条件后，按 `abs(train_ic)` 排序；
- 根据因子多头超额收益率序列相关性小于 `0.7` 进行贪心保留。

注意：研报正文将 2019-2020 称为验证集，但筛选公式使用 `test_ic`、`test_long_ir` 命名。项目实现时必须统一术语，避免把验证集和最终样本外测试集混用。

研报未为 `barra_ts_corr` 和“因子多头超额收益率序列相关性”增加 `train/test` 前缀，也未在当前披露文本中明确两者的估计区间。项目第一版复现将这两项固定在训练期计算；这是为保持信息集一致、降低短验证期相关估计噪声并避免额外使用验证集优化因子池结构而作出的工程选择，不标记为研报原文。验证期对应相关性仍须计算和报告，但不参与第一版硬筛选、排序或贪心保留。

### 5.5 研报未披露、项目已确认的标签假设

研报明确了 5 日评估周期，但没有在当前 PDF 中披露精确的信号日、买入日和卖出日索引。当前项目采用以下第一版复现假设，不视为研报原文：

```text
forward_return_5d[t] = open[t+6] / open[t+1] - 1
```

该假设表示因子使用交易日 `t` 的完整日频信息，最早在 `t+1` 开盘成交，并持有 5 个交易日。以下实际可交易性问题仍待真实数据阶段处理：

- 因子可用时间；
- 下单时间；
- 成交价格；
- 持有区间；
- 停牌、涨跌停和缺失收益处理；
- 训练、验证与最终测试之间的隔离和必要 purge 区间。

## 6. 分阶段开发路线

### 阶段 0：原始数据下载与 QA

目标：获得完整、可续传、字段明确的四份原始 Parquet。

验收条件：

- 股票主表、后复权行情、不复权收盘价和历史股本均存在；
- `(trade_date, stock_code)` 无重复键；
- 历史股本 `(stock_code, change_date)` 无重复键；核心字段 `total_shares`、`list_a_shares` 为正且满足 `total_shares >= list_a_shares`；早期记录的 `limit_shares` 和 `change_reason` 允许缺失并单独统计；
- 字段类型和日期范围正确；
- 股票覆盖、缺失代码和失败代码有摘要；
- 后复权行情与不复权收盘价的键覆盖差异已统计；
- 下载程序已经停止，最终文件不再被写入。

当前状态（2026-08-07）：股票主表、后复权行情、不复权收盘价和历史股本已完成。原 adata 行业接口已退出并移除；正式行业输入改为 `参考文件/swind/` 点时 CSV，已经独立完成全量转换、QA 和阶段三中性化接入验证。

### 阶段 1：数据清洗、VWAP 与矩阵化

目标：生成六特征的日频面板，并建立明确的有效值规则和数据版本记录。

主要工作：

- 核对两套行情键覆盖；
- 构造复权因子和后复权 VWAP；
- 执行价格、成交量、日期和上市信息 QA；
- 将数据质量清洗与股票池资格过滤区分开；
- 决定 ST、上市时长、停牌和退市股票的点时处理；
- 决定缺失行、部分缺失和有效值 mask 的处理方式；
- 根据规模选择普通 NumPy、MemMap、Zarr 或其他矩阵存储；
- 输出股票索引、交易日索引、字段顺序和元数据。

当前状态（2026-08-07）：核心清洗、VWAP、六特征张量、有效值/股票池 mask、日期/股票索引和元数据已经生成并通过用户运行验证；后续若变更数据口径，必须重新版本化而不是覆盖原始数据。

### 阶段 2：表达式文法与合法状态机

目标：在不读取行情数据的情况下，生成、序列化和验证合法表达式。

主要工作：

- 带显式 Hole 的规范化部分有序 AST；
- 特征、算子和窗口注册表，以及自动终止规则与模型特殊符号边界；
- 开放槽位轨道、联合动作 `(slot, token_id)` 和两级合法动作 mask；
- 深度、节点数和表达式长度限制；
- 结构哈希、规范化和等价表达式去重；
- 表达式序列化和反序列化；
- 父状态枚举、固定均匀后向概率和小规模状态图穷举；
- 对每类合法与非法路径建立单元测试。

实现状态（2026-08-05）：原前序固定槽位生成状态机废止，第二阶段按规范化部分 AST DAG 全量重构。前序/后序方法只保留为完整 `Expression` 的序列化与解释器交换格式，不再定义生成状态或生成路径。

- 完整覆盖 6 个叶子、52 个非叶子算子和 `[5, 10, 20, 40, 60]` 五个窗口；
- 142 个表达式 Token 不包含 `BOS`、`EOS`、`PAD`，这些符号如后续需要，仅属于模型序列层；
- 待填槽位为空即自动形成终止状态，不额外把 `STOP` 作为表达式树节点；
- `max_depth=10`、`max_nodes=30` 是可覆盖的工程默认值，不视为研报固定参数；
- 142 个动作只表示稳定 Token 词表；完整状态转移动作为动态联合动作 `(slot, token_id)`，因此不得将 Token 数量误称为某一状态的完整出边数量；
- 槽位统一期待一个 `Expr`，不按 FEATURE/UNARY/BINARY 等 Token 类别拆分槽位类型；Token 元数决定填充后产生 0、1 或 2 个新 Hole；
- 结构去重当前仅对 `add`、`mul`、`max2`、`min2` 使用交换规范化；
- `ts_corr`、`ts_cov` 虽数值对称，但不纳入交换去重，以保持与 `ts_beta`、`ts_orth` 的方向性一致；结构规范化仍只覆盖 `add`、`mul`、`max2`、`min2`；
- Token 空间协议为 `factor_gfn.action_space.v1`，指纹继续固定为 `5689dbceb1bb42716773bcaf4cb5845041e578a3bb11fe67445ede6cde7938cc`；
- 部分状态协议为 `factor_gfn.dag_state.v1`，当前指纹为 `5301fb14e197376dfc2a5aaf9c398aa47642bcc889f6eef87278839142b824d9`；
- 联合转移协议为 `factor_gfn.dag_transition.v1`，当前指纹为 `bb9c95a05c0bccb375c47828fb6d95821db1f625f4a2513ba55e7151bac55a4b`；三类指纹相互独立，避免把 Token 稳定性与状态图语义混为一谈。

### 阶段 3：因子解释器与指标计算

目标：可靠、快速地将表达式计算为 `(date, stock)` 因子矩阵。

实现进度（更新至 2026-08-07）：四个步骤均已完成，以下保留各步骤的冻结口径与验收记录。

- 已建立 12 个一元变换、10 个二元组合和 9 个截面变换的 NumPy `float64` 基准实现；
- 输入和输出均为 `(date, stock)`，非有限值统一转为 NaN，算子不修改输入；
- `log(x)=log(abs(x)+eps)`、`sqrt(x)=sqrt(abs(x))`，除法和逆运算在分母接近零时返回 NaN；
- `signed_ratio(x,y)=(x-y)/(abs(x)+abs(y)+eps)`；
- `log_ratio(x,y)=log((x+eps)/(y+eps))`，比例非正或分母为零时返回 NaN；
- `cs_normalize` 使用逐日 min-max 口径；
- `cs_winsorize` 将逐日 5%/95% 分位点外数值截断到边界；
- `cs_truncate` 将逐日 5%/95% 分位点外数值设为 NaN，不改变矩阵形状；
- `cs_quantile` 使用逐日平均并列零基排名除以 `N-1`；
- 截面最少有效股票数暂定为 2，并列值使用平均排名。

以上保护运算和截面参数属于用户确认的工程口径，不标记为研报原文。

第二步“21 个时序算子”也已完成 NumPy `float64` 基准实现：

- 普通 rolling 算子严格使用 `[t-w+1, t]` 完整窗口，窗口内任意 NaN 均返回 NaN；
- `ts_delay(x,w)=x[t-w]`、`ts_delta(x,w)=x[t]-x[t-w]`，前 `w` 行为 NaN且只检查端点；
- `ts_std`、`ts_cov` 使用 `ddof=0`；零方差相关系数、z-score、beta 和 orth 返回 NaN；
- WMA 权重为 `[1,2,...,w]`；EMA 使用 `alpha=2/(w+1)`、`adjust=False`，遇 NaN 后重置并重新等待连续 `w` 个有效值；
- `ts_rank` 使用当前值的平均并列零基排名，`ts_position` 使用窗口 min-max 位置；
- `ts_argmax`、`ts_argmin` 返回从 0 开始的窗口位置，并列取最近一次；
- `ts_slope` 为含截距时间趋势回归斜率，`ts_residual` 返回当前日趋势残差；
- `ts_beta(x,y)` 将 `x` 作为因变量、`y` 作为解释变量，`ts_orth(x,y)` 返回该回归当前日残差；
- `ts_corr`、`ts_cov` 使用对称完整窗口，但表达式结构仍保留参数顺序，不进行交换去重；四个二元时序算子统一视为有序参数。

以上时序定义同样属于第一版工程复现规范，不代表研报披露的内部实现。

第三步“表达式解释器”已完成：

- 输入固定为 `(date, feature, stock)`，特征顺序为 `open/high/low/close/vwap/volume`；
- 递归遍历表达式树并统一调用全部 52 个算子；
- 输出为 `(date, stock)`，并校验输入维度、特征数、动作元数、窗口签名和算子输出；
- 解释器不修改原始六特征张量；正式只读 `float64` mmap 采用借用模式，循环算子使用预热后的 Numba 内核，非叶且非根的子表达式按结构哈希进入有字节上限的 LRU 缓存。

第四步“5日指标与性能基线”已完成：

- 将 `open[t+6]/open[t+1]-1` 作为匹配5日调仓的第一版复现标签假设；研报只确认5日评估调仓周期，未披露该精确标签；
- 同时保留逐日滚动5日 RankIC 和每5日非重叠 RankIC，reward 默认使用后者；
- `rebalance_offset=0` 从 warmup 完成且具有足够完整标签的首个评价日期开始；
- IC 和绩效统计采用 `ddof=1`，与时序算子 `ts_std(ddof=0)` 相互独立；
- 训练集 IC 方向仅用于多头组合排序，不改写原始因子值；验证集必须复用训练集方向；
- 多头组合为前10%等权，基准为同期联合有效股票池等权，超额为两者之差；
- 年化期数为 `252/5`，超额 IR 使用 `sqrt(252/5)` 年化；
- 已实现逐日有效样本数、覆盖率、因子截面 Spearman 相关性及多头超额收益序列 Pearson 相关性；
- 阶段三统一评价入口的候选因子清洗顺序固定为：1%/99% 缩尾 → 申万一级行业哑变量 OLS 残差化 → `ddof=0` z-score。该流程用于 RankIC、因子截面相关、Top 10% 多头和 Long-Short 收益；Barra 暴露仍仅做缩尾与 z-score；解释器输出及保存的原始因子值不被改写；
- 候选因子市值中性化暂缓，待阶段六回测评估实际市值暴露后再决定是否加入；第一版先由 reward 中的 Barra Size 收益序列相关性惩罚把关；
- 1000日×2000股合成矩阵的标签、IC与多头评价总耗时约0.48秒，当前无需引入 Numba、缓存或 MemMap。

主要工作：

- NumPy 基准实现；
- 算子数值边界、NaN 传播和最小有效样本规则；
- 时序算子只使用当前及历史信息；
- 截面算子只使用同一交易日信息；
- 与简单 Pandas 参考实现进行小样本对照；
- 再根据性能瓶颈决定是否引入 Numba、MemMap 或缓存；
- 实现 IC、IR、多头超额和相关性指标。

### 阶段 4：GFlowNet 最小训练闭环

目标：完成合法轨迹采样、奖励计算、TB Loss 和参数更新。

实现状态（2026-08-07）：六个子步骤均已完成，164 项合成单元与集成测试通过，`validate_stage4_synthetic_training.ipynb` 已验证参数和 `logZ` 更新、监控指标及确定性检查点恢复。该结果仅证明训练链路闭合，不代表真实 Reward、真实数据吞吐或最终因子质量已经验证。

主要工作：

- 前向策略网络；
- 固定均匀后向策略 `P_B=1/n_parents`；
- 轨迹采样与终止状态；
- reward 的正值化、数值稳定与缓存，以及 `LONG_IR_LAMBDA=0.3`、`BARRA_TS_PENALTY_MU=0.2` 的集中可配置实现；
- 分别构造 5 个手工 Barra 风格因子的独立多空收益序列，并以候选多空收益与各序列相关性的最大绝对值形成 `barra_ts_corr`；不合成 Barra 因子，也不使用逐日截面相关；
- Trajectory Balance Loss；
- 可学习 `logZ`；
- 训练检查点、随机种子和运行元数据；
- 监控奖励、表达式唯一率、batch 内相关性和非法动作率。

Transformer 层数、维度、head 数、batch size、学习率和训练步数均从小规模实验开始，不预先固定。

前向策略必须覆盖联合动作概率，而不是只输出 Token 概率：

```text
P_F(slot, token | state)
  = P_F(slot | state) * P_F(token | state, slot)
```

TB 轨迹必须同时记录 `log P_F(slot)`、`log P_F(token|slot)`、后继状态父状态数及 `-log(n_parents)`。同一规范状态无论由哪条历史到达，都必须产生完全相同的模型状态表示；原始动作历史不得成为破坏马尔可夫状态等价性的隐藏输入。

#### 6.4.1 第一步：训练配置与状态接口

- `factor_gfn/grammar/config.py` 中的不可变 `SearchSpaceConfig(max_depth, max_nodes)` 是阶段二与阶段四共享的结构约束类型和唯一字段定义；`GrammarState`、`StateAdapter` 与策略模型要求配置值及指纹等价，但不要求引用同一个 Python 对象。标准装配入口优先从 `GFNConfig.search_space` 向下传递，分别构造的等值实例同样合法。
- `factor_gfn/gfn/config.py` 集中定义状态、模型、采样、reward 和训练配置；所有小规模默认值均为实验起点，不视为最终超参数。
- 配置清单同时记录 Token、DAG 状态和 DAG 转移三个指纹；序列化配置可生成稳定指纹，供检查点和运行元数据核对。
- `RewardConfig` 预留 `reward_clip_min`、`reward_clip_max`，第一版默认均为 `None`，不执行截断；截断逻辑实现前不得静默接受非空配置。
- `TrainingStats` 预留 `batch_corr_mean`、`batch_corr_median`，第一版默认均为 `None`，不强制计算 batch 内两两 Spearman 相关。
- Reward 结果必须保留 `barra_correlations` 字典中的五个带符号 Pearson 相关系数；`barra_ts_corr` 仅作为 `max_k abs(corr_k)` 聚合惩罚值。同时记录绝对相关性最大的风格名称及其带符号相关系数，供后续分析暴露方向。
- `reward_floor=1e-8` 是可配置的 TB 数值稳定下界，只对指标均有效但原始 reward 过小的样本生效；原始 reward、稳定化 reward、`log_reward` 和是否触发 floor 必须分别记录。任何 IC、Long IR 或 Barra 相关性无效的候选均保持无效，禁止用 floor 伪装成有效低奖励样本。
- Reward 缓存第一版采用有容量上限的进程内 LRU，以“数据上下文指纹 + Reward/Evaluation 配置 + 表达式结构哈希”隔离结果。候选行业中性化开关及行业数据指纹必须进入上下文；不得在开启和暂时跳过行业中性化的运行间共享缓存。
- 点时行业长表尚未生成或未通过 QA 时，阶段四联调允许通过显式配置 `candidate_industry_neutralization=False` 暂时跳过候选行业中性化，并在每个 Reward 结果中记录该状态；阶段三默认口径仍是启用行业中性化，正式训练切换为启用时必须同时提供当日一级行业标签与行业数据指纹。
- 研报图表中的 batch 内相关性与 Barra 风格相关性是两个独立诊断：前者后续按固定间隔统计候选因子两两相关性的 `mean(abs(corr))` 与 `median(abs(corr))`，只用于模式坍塌分析，不进入第一版 reward。
- TB 目标严格定义为 `mean((logZ + sum_log_pf - log_reward - sum_log_pb)^2)`；`logZ` 是全局单一可学习标量，第一版初始化为 `0.0`（即 `Z=1`），研报未披露该初始化值。
- TB 的 Delta 使用 `float64` 累加以提高极小 Reward 和长轨迹的数值稳定性，梯度仍回传至原 dtype 的 Transformer 与 `logZ`。Loss 不加入轨迹长度归一化、Delta 截断、Reward 重加权、流守恒项或其他正则。
- `sum_log_pb` 累加每一步子状态的 `-log(n_parents(child))`；父状态数来自规范 DAG 的不同父状态枚举。后向概率固定，不设置可学习参数或梯度。
- TB Loss 拒绝空 batch、greedy 轨迹、未挂载 Reward、非正/非有限 Reward 及不一致的 `reward/log_reward`。后续 Trainer 负责补采有效轨迹；补采失败时显式跳过更新并记录原因，Loss 本身不得把空 batch 解释为零损失。
- 文法层继续去重等价状态和规范平行边，Reward 层继续按表达式结构哈希缓存；TB batch 不按终态表达式去重。到达同一终态的不同采样路径共享 Reward，但分别以各自的前向与后向路径概率进入 TB Loss。
- 第一版 Trainer 使用 Adam：Transformer `learning_rate=1e-4`、`logZ learning_rate=1e-3`、`betas=(0.9,0.999)`、`optimizer_eps=1e-8`、`weight_decay=0`。Transformer 与 `logZ` 联合执行 `clip_grad_norm_=1.0` 并记录裁剪前范数；这些均为工程实验初值，不标记为研报参数。
- `TrainingConfig.optimizer_eps` 是 Adam 分母数值稳定参数，`RewardConfig.reward_floor` 是 TB 对数所需的正值 Reward 下界；二者默认值虽同为 `1e-8`，但禁止共享变量，并在运行元数据中分别记录语义。
- Trainer 目标是补满配置的 `batch_size`，每次更新最多采样 `max_sampling_multiplier × batch_size` 个候选，第一版倍数为 10。未补满则整次更新跳过，不使用不完整 batch；记录 `batch_rejection_rate`、`effective_batch_size`、拒绝数、补采轮数和跳过次数，只提供监控，不自动放宽 Barra 共同期数或改变采样倍数。
- 策略熵采用联合动作的精确定义 `H(slot)+E_slot[H(token|slot)]`，在采样时计算并以普通浮点数记录，不用已选动作 surprisal 代替，也不进入梯度。同时记录 `H/log(K_legal_joint)` 归一化熵，其中 `K_legal_joint` 是当前状态合法的 `(slot, token)` 联合动作数；仅有一个合法动作的强制状态不纳入归一化熵均值。成功采样的非法动作率必须为 0；任何 mask 外动作仍立即终止运行。
- 检查点保存 Transformer、`logZ`、optimizer、训练步数、配置/状态/转移/Reward Provider 指纹、Python/NumPy/PyTorch CPU及CUDA随机状态、统计历史与运行元数据，并通过同目录临时文件和 `os.replace` 原子替换。确定性续跑首先保证相同软件和CPU环境；不同CUDA版本或设备不承诺逐位一致。
- 合成闭环 Reward 固定为 `log_reward=0.75*contains(close)+0.50*contains(ts_mean)-0.10*node_count`，仅用于验证训练链路，不代表研报或真实因子奖励。验证 Notebook 为 `notebooks/validate_stage4_synthetic_training.ipynb`，输出写入已忽略的 `tmp/synthetic_runs/`。
- `factor_gfn/gfn/state_adapter.py` 将规范部分 AST 转为确定性模型输入，包含节点 Token、Hole、深度、父子角色、规范路径、开放槽位、联合合法动作 mask 和剩余结构预算。
- 研报中的动作序列由算子、窗口和特征构成；在多路径 DAG 下，本项目编码规范部分 AST 的确定性序列而非实际采样历史顺序，并将每个节点拆为“类别、算子/叶子特征、窗口”三组 embedding。这样保留研报的表达式内容编码，同时不破坏状态汇合后的马尔可夫等价性。
- 研报所述 3 个手工状态特征固定为：`max(所有开放槽位深度, 已填节点最大深度) / max_depth`、`operator_count / max_nodes`、`node_count / max_nodes`；直接来自阶段二 `GrammarState.auxiliary_features()`，经投影后与 Transformer 全局状态表示融合。被选槽位的具体深度另外作为槽位条件特征，不占用这三个全局特征。
- 状态适配器以 `GrammarState.legal_transitions()` 为联合边真值来源，不直接拼接各槽位局部 mask，防止重新引入已由状态图去除的平行边。
- 相同规范状态无论由何种动作历史到达，编码、槽位顺序和 mask 必须完全一致。

#### 6.4.2 第二步：路径条件化 Transformer 前向策略

第一版不为动态槽位建立互不共享参数的独立 Token 头。采用路径条件化共享 Token 头：共享参数接收状态表示及当前槽位表示，为每个槽位产生独立的 142 维 Token logits。

```text
P_F(slot, token | state)
  = P_F(slot | state) * P_F(token | state, slot)
```

槽位表示至少包含：上下文化 Hole 表示、规范路径表示、父算子、参数角色、深度和剩余节点/深度预算。非交换算子使用不同的 `ARG_0`、`ARG_1` 角色；`add`、`mul`、`max2`、`min2` 的子槽位统一使用 `COMMUTATIVE_CHILD`，并继续服从阶段二的槽位轨道去重。不得将交换律节点规范化后不稳定的原始左右下标当作语义角色。

前向模型分别应用槽位 mask 和条件 Token mask；非法联合动作概率必须为零。表达式填满所有 Hole 后自动终止，不增加表达式 EOS 动作。

策略采样时关闭 dropout，使 `P_F` 成为规范状态和当前模型参数的确定函数，避免未记录的 dropout mask 变成隐藏状态；该模式不切断梯度，采样结束后恢复模型原训练/评估模式。

#### 6.4.3 第三步：可微轨迹与策略采样器

阶段二已实现确定性的 DAG 状态机、合法转移、父状态枚举和随机文法测试；阶段四策略采样器只负责根据神经网络概率选择联合动作并记录 TB 所需概率，不重复实现 DAG。

每个轨迹步骤至少记录：当前状态哈希、规范槽位索引、槽位路径、槽位轨道键、Token ID、`log P_F(slot)`、`log P_F(token|slot)`、二者之和、子状态哈希、`n_parents(child)` 和 `-log(n_parents(child))`。训练中的前向 log 概率必须保留为 PyTorch Tensor，不得通过 `.item()` 提前切断计算图；固定后向概率可以保存为普通浮点数。

`sum_log_pf` 和 `sum_log_pb` 由步骤列表实时派生，不保存可失配的冗余副本。验收使用容差比较，并逐步验证：

```text
log_pf = log_p_slot + log_p_token
log_pb = -log(n_parents)
sum_log_pf = sum(step.log_pf)
sum_log_pb = sum(step.log_pb)
```

策略采样器支持随机采样和用于诊断的贪心采样。训练默认使用随机采样；无论何种方式，都必须调用阶段二不可变状态转移，并保证非法动作率为零。

第一版采样超参数固定为实验默认值而非研报参数：正式训练和随机验证使用 `temperature=1.0, greedy=False`，不额外扭曲模型原始策略分布；确定性诊断使用 `temperature=1.0, greedy=True`，只用于查看当前最高概率表达式、检查 mask 和检查点复现，不将贪心轨迹用于 TB 参数更新。第一版不启用温度退火；只有观察到探索不足后，才另行实验如 `temperature=1.1` 的配置。

轨迹数据结构必须显式保存 `sampling_mode="stochastic"|"greedy"`。只有随机策略轨迹的 `training_eligible=True`；贪心轨迹必须由轨迹合同、后续 TB Loss 和 Trainer 三层共同拒绝参与参数更新，不能只依赖调用方约定。

#### 6.4.4 第四步：Reward 适配、正值化与缓存

`factor_gfn/gfn/reward.py` 将阶段三指标和五条独立 Barra Long-Short 收益序列接入统一奖励合同。真实奖励严格采用：

```text
reward_raw = abs(train_ic)
           * (1 + LONG_IR_LAMBDA * clip(train_long_ir, 0, LONG_IR_CAP))
           * (1 - BARRA_TS_PENALTY_MU * clip(barra_ts_corr, 0, 1))
```

默认工程参数为 `LONG_IR_LAMBDA=0.3`、`LONG_IR_CAP=2`、`BARRA_TS_PENALTY_MU=0.2`。`barra_ts_corr=max_k abs(corr_k)`，同时保留五个带符号 `barra_correlations`、共同有效期数、最大暴露风格及其带符号相关系数。共同有效调仓期不足 60、IC/Long IR 非有限或其他指标无效时，候选保持无效，不用 Reward floor 兜底。

有效但过小的奖励使用 `reward=max(reward_raw, reward_floor)`，默认 `reward_floor=1e-8`，并分别保存原始 Reward、稳定化 Reward、`log_reward` 和 floor 是否触发。进程内 LRU 缓存按数据上下文、评价/奖励配置和表达式结构哈希隔离。行业数据未完整时只允许通过显式 `candidate_industry_neutralization=False` 暂时跳过；正式启用必须提供行业标签和行业数据指纹。

验收覆盖公式精确值、Long IR 截顶、带符号相关性、无效样本拒绝、Reward floor、缓存命中和行业开关隔离。

#### 6.4.5 第五步：Trajectory Balance Loss 与可学习 logZ

`factor_gfn/gfn/loss.py` 定义全局单一可学习标量 `log_z`，默认 `initial_log_z=0.0`，并严格实现：

```text
delta_i = logZ + sum_log_pf_i - log_reward_i - sum_log_pb_i
TB_loss = mean(delta_i^2)
```

Delta 使用 `float64` 累加，梯度回传至原 dtype 的 Transformer 和 `logZ`。`sum_log_pb` 使用每一步子状态的 `-log(n_parents(child))`，固定后向概率不参与求导。轨迹通过 `attach_reward()` 挂载有限正 Reward，并校验 `log_reward=log(reward)`；不同 Reward 不得静默覆盖。

Loss 拒绝空 batch、greedy 轨迹、缺失/非正/非有限 Reward 和 Reward/logReward 不一致。不同路径到达同一终态时共享 Reward，但分别作为 TB 样本，不按终态去重。第一版不加入长度归一化、Delta 截断、Reward 重加权、流守恒项或额外正则。

验收覆盖人工平衡轨迹零 Loss、Transformer 与 `logZ` 有限梯度、`1e-8` Reward、长轨迹、拒绝路径及 `state_dict` 恢复。

#### 6.4.6 第六步：Trainer、检查点与合成训练闭环

`factor_gfn/gfn/trainer.py` 通过统一 `RewardProvider` 执行“状态编码 → Transformer → 随机 DAG 采样 → Reward → TB Loss → backward → 梯度裁剪 → Adam step”。第一版 `SyntheticRewardProvider` 使用：

```text
log_reward = 0.75*contains(close)
           + 0.50*contains(ts_mean)
           - 0.10*node_count
```

该公式只验证训练链路，不代表研报或真实因子奖励。优化器包含 Transformer 与 `logZ` 两个参数组，学习率分别为 `1e-4` 和 `1e-3`；Adam 默认 `betas=(0.9,0.999)`、`optimizer_eps=1e-8`、`weight_decay=0`。所有参数联合执行 `clip_grad_norm_=1.0`。

Trainer 最多采样 `10*batch_size` 个候选以补满有效 batch，未补满则整次跳过更新。监控至少包括 Loss、Reward、`logZ`、表达式唯一率、轨迹长度、联合动作精确熵及按合法联合动作数归一化的熵、梯度范数、非法动作率、`batch_rejection_rate`、`effective_batch_size`、拒绝数和补采轮数。监控不自动修改 Barra 期限、Reward 或过采样倍数。

`factor_gfn/gfn/checkpoint.py` 原子保存模型、`logZ`、optimizer、步数、统计历史、完整配置及三类空间指纹、Reward Provider 指纹、Python/NumPy/PyTorch CPU和CUDA随机状态与运行元数据。恢复前必须核对 schema、配置、Reward Provider 和设备类型。相同软件及 CPU 环境要求连续训练与保存恢复续跑逐项一致；跨 CUDA 环境不承诺逐位一致。

手动验收入口为 `notebooks/validate_stage4_synthetic_training.ipynb`，不读取真实数据，输出只写入 `tmp/synthetic_runs/`。

#### 6.4.7 第七步：真实 Reward 接入与最小实验

目标：在不改变已冻结 Reward 公式、行业中性化和 Barra 惩罚口径的前提下，将真实日频数据接入现有 `RewardProvider → TB Loss → Trainer` 合同；先建立可复核的性能基线，再决定是否优化或扩大训练。

实现状态（2026-08-07）：四部分均已完成工程验收。`factor_gfn/gfn/real_data.py` 以只读 mmap 组装训练上下文并排除验证期，建立统一五日调仓日历、五条 Barra LS、数据指纹和 QA 清单。真实训练期为 2010-01-04 至 2018-12-28；全局调仓日为 2011-01-18 至 2018-12-17，共 386 期，五条 Barra LS 均有 386 个有效期。训练股票池内共有 4,318,753 个有效股票—日期样本，申万一级缺失为 0；稠密矩阵中的 `-1` 主要对应尚未上市或不在当日股票池的占位格。上下文不暴露 2019 年后的特征行，末端不完整未来收益标签也不进入调仓日历。`factor_gfn/gfn/real_reward.py` 已实现解释器接入、强制行业中性化、全局日历 Reward、解释前 LRU、完整诊断记录和 Provider 指纹；通用阶段三指标保留未显式传入日历时的原分析行为。真实 Reward 人工表达式预检已得到多个有效候选，并确认五条 Barra LS 可参与相关性计算、缓存命中和标签边界合同。CPU 五步最小真实训练完成 5 次参数更新，无跳过更新、非法动作或 NaN/Inf Loss；第 4 步检查点完整恢复后，第 5 步候选序列、统计、模型、`logZ`、优化器及随机状态与连续运行逐项一致。当前完整测试为 178 项。

**第一部分：真实 Reward 数据上下文**

- 以只读 mmap 加载六特征张量、`universe_mask`、日期、股票和五个 Barra 暴露矩阵；加载与行情键完全一致的申万一级点时行业矩阵。
- 未来五日收益继续采用 `open[t+6] / open[t+1] - 1`。训练集固定为 2010-01-01 至 2018-12-31；任何退出日在 2018-12-31 之后的标签不得进入训练 Reward，2019-2020 验证集保持隔离。
- 在统一训练评价日历上分别构造 Market Beta、Size、Momentum、Volatility、Liquidity 五条 Long-Short 收益序列，不合成 Barra 因子。
- 数据上下文指纹至少覆盖日期、股票、预处理元数据、行业元数据、Barra 元数据、评价配置和 Reward 配置；检查点不得跨不同上下文恢复。
- 验收包括所有矩阵轴严格一致、行业缺失值为 `-1`、五条 Barra LS 非全 NaN/非常数、有效期充足、评价日期不越过训练边界，以及每条序列的有效期数、均值、波动和累计收益摘要。

**第二部分：`RealRewardProvider`**

- 新增真实 Provider，将 `Expression → FactorInterpreter → 原始因子矩阵 → RewardEvaluator → RewardAssignment` 接入 Trainer；正式模式强制 `candidate_industry_neutralization=True`。
- `RewardEvaluator` 必须显式接收第一部分生成的全局 `rebalance_indices`，RankIC、Long IR、候选 LS 与五条 Barra LS 全部使用同一日历；禁止再次调用“按候选首次有效日”生成日历的旧分支。通用阶段三接口仍可保留未传入固定日历时的原行为，以维持分析兼容性。
- Reward 严格保持 `abs(train_ic) × (1 + 0.3 × clip(train_long_ir, 0, 2)) × (1 - 0.2 × clip(barra_ts_corr, 0, 1))`；`barra_ts_corr=max_k abs(corr_k)`，最低共同有效期为 60。
- 保存五个带符号 Barra 相关系数、最大暴露风格、共同有效期数、IC/Long IR 有效期数、原始/稳定化/log Reward 和行业中性化状态；每个候选还必须持久化固定调仓日历上去重后的 `neutralization_skipped_dates`、其占调仓期数的 `neutralization_skipped_rate`，以及 `neutralization_skipped_details`。逐日期明细至少包含日期、矩阵行号、候选有效股票数、已知行业股票数、行业数、最低所需回归样本数和稳定失败原因，包括最终无效的候选。
- 在解释表达式前按“数据上下文指纹 + 表达式结构哈希”检查 Provider 级有界缓存，避免重复表达式再次执行整张因子矩阵；不缓存完整因子矩阵。
- Provider 记录每次真正执行的表达式公式、结构哈希、节点数、深度、因子有限覆盖率、解释器耗时、指标耗时、有效性、拒绝原因及完整 Reward 拆解；缓存命中另行计数，不伪装成一次新的因子计算。最小真实训练 Notebook 将完整候选清单持久化为 `evaluations.jsonl`。
- Provider 指纹必须包含真实数据上下文指纹、评价配置、Reward 配置、固定日历摘要、行业中性化状态和缓存合同；相同结构表达式只在同一 Provider 上下文中共享结果。
- 指标无效的候选返回明确拒绝原因且不得用 Reward floor 兜底；解释器、形状或数据合同错误必须立即抛出，不得伪装为普通无效候选。

**第三部分：真实 Reward 预检与性能基线**

- 先评价少量人工表达式，不进行参数更新；至少覆盖叶子、时序、二元组合和截面算子。
- 记录因子计算与 Reward 计算耗时、峰值内存、有效覆盖率、RankIC、Long IR、五个带符号 Barra 相关、最终 Reward、拒绝原因和缓存命中。
- 只有确认行业中性化实际运行、五条 Barra LS 可参与相关性、标签无越界且时间/内存可接受后，才进入训练。
- 若该基线已经形成瓶颈，先根据实测定位解释器、滚动算子、截面回归或组合评价的占比，再决定是否引入 Numba、额外 MemMap、并行或分层缓存，不预先优化。

**第四部分：最小真实训练实验**

- 第一轮仅采用工程试验配置：`max_depth=5`、`max_nodes=12`、`d_model=64`、4 heads、2 layers、feedforward 128、dropout 0、batch size 2、最多 5 步、`temperature=1.0`、随机采样、Transformer 学习率 `1e-4`、logZ 学习率 `1e-3`、补采倍数 10、seed 42、CPU。它们不代表研报超参数或正式训练配置。
- 监控 Loss、Reward、logZ、梯度范数、有效 batch、拒绝率、跳过更新、结构唯一率、轨迹长度、联合策略熵和非法动作率。
- 保存整个实验期间所有被评价表达式的公式、结构哈希、有效性、Reward 拆解和 Barra 暴露方向，不只保留最后一个 batch；结果、运行元数据和检查点写入已忽略的 `runs/real_minimal/<run_id>/`。
- 验收要求至少完成 3 次真实参数更新，Transformer 与 logZ 均有限变化，非法动作率为 0，Loss 无 NaN/Inf，且检查点可以继续运行一步。
- 若拒绝率持续超过 80%，先分析无效原因；不得自动降低 60 期门槛、关闭行业中性化或直接扩大正式训练规模。

**已确认：全局统一调仓日历**

所有候选与五个 Barra 风格共用一组训练调仓日期：在训练区间内寻找五个 Barra 风格均达到最小截面样本数且未来收益标签完整的首个日期，从该日期起每 5 个交易日评价，并截断在最后一个全局可评价日以内。候选表达式在自身尚未形成有效值的调仓日保留 NaN，相关性及指标只使用共同有效期。不得再根据各候选首次有效日平移其调仓相位，也不得为每个候选动态重建另一套 Barra 日历。

### 阶段 5：GPU 真实训练与候选生成

目标：在冻结的训练数据与 Reward 口径上，以可恢复的 GPU 正式训练生成足量、来源可追踪的候选表达式；本阶段不使用验证集或最终样本外结果调整策略。

#### 5.1 旧 scalar-logZ 基线的模型、搜索空间与训练动态（2026-08-11 历史合同）

- 本节记录2026-08-11至2026-08-12已经完成并提交的旧 `grammar_hierarchical + scalar logZ` 可回退基线，不再定义5.3新架构的最终配置。该基线搜索空间固定为 `max_depth=6`、`max_nodes=15`；历史 run、候选和检查点继续只读保留，禁止跨 schema 恢复到新架构。
- 旧基线策略网络结构固定为 `d_model=128`、`num_heads=4`、`num_layers=4`、`dim_feedforward=512`、`dropout=0`。5.3重构继续保留 `grammar_hierarchical` 主体，但搜索边界改为配置化；6/15兼容性smoke已经完成，当前独立6/20 diagnostic只判断`max_depth=6`是否可能过紧。`max_nodes=20`是当前人为固定上限，不做自动扩边；最终`max_depth`与训练动态参数须按5.3验收，不继承本节“正式冻结”表述。
- 下一轮延长诊断配置使用有效 `batch_size=8`、`temperature=1.0`、`greedy=False`、Transformer 学习率 `1e-4`、`initial_log_z=39.0`、`logZ` 学习率 `1e-2`，模型与 `logZ` 继续采用各自 `max_norm=5.0` 的独立梯度裁剪；Adam `betas=(0.9,0.999)`、`eps=1e-8`、无 weight decay、补采倍数 10、确定性算法和强制候选行业中性化保持不变。`initial_log_z=39.0` 与 `logZ` 学习率 `1e-2` 来自旧 run 的训练期 TB 诊断，不使用验证期或 OOS。
- 上述学习率、`initial_log_z`、独立梯度裁剪和 batch 等训练动态参数仍处于阶段 5 诊断优化期，当前值只是具有独立配置指纹的工程基线，尚未最终冻结为后续全部 seed 的正式配置。平坦 Token 策略的 300 步、元数分组策略的 100 步和完整文法分层策略的首个 100 步均已完成并分别保留；当前只允许从完整文法分层 run 的同策略检查点续跑，不丢弃前 100 步候选，不跨策略配置续接或改写历史 run。
- 代码通过 `build_stage5_real_training_config(max_steps=..., seed=...)` 生成完整 `GFNConfig` 和配置指纹。总步数属于每次运行预算，seed 属于独立 run 身份，二者必须显式记录，不在本步固定为单一全局值。
- 超参数网格探索暂缓为后续优化内容；当前不设置小、中、大模型对比，不用验证集或 OOS 选择模型规模，也不在正式训练前额外开展搜索空间收益调参。
- `device=cuda`、禁止静默回落 CPU、周期检查点及同 GPU 恢复已经作为正式训练入口合同实现；训练仍由用户在 Notebook 中手动启动和恢复。

##### 5.1.1 当前完整文法分层策略合同

正式 Stage 5 的 `grammar_hierarchical` 不把 142 个 Token 直接放进一个平坦 softmax，也不把文法选择实现为额外的环境动作。环境每一步仍然只产生一条原有的 `(canonical_slot, token_id)` DAG 转移；策略网络在内部将该 Token 的条件概率按文法结构分解：

1. 先选择元数组 `Feature / UnaryFamily / BinaryFamily`；
2. 一元族内选择 `UnaryOp / TsUnaryOp / CsOp`，二元族内选择 `BinaryOp / TsBinaryOp`；
3. 在已选类别内选择具体 Feature 或 Operator；
4. 仅当类别为 `TsUnaryOp` 或 `TsBinaryOp` 时，再条件选择 Window `5 / 10 / 20 / 40 / 60`。

普通非时序动作和时序动作的内部联合概率分别为：

`P(token | s, slot) = P(family | s, slot) × P(category | family, s, slot) × P(operator_or_feature | category, s, slot)`；

`P(token | s, slot) = P(family | s, slot) × P(category | family, s, slot) × P(operator | category, s, slot) × P(window | operator, category, s, slot)`。

最终前向边概率仍为 `P_F(action | s) = P(slot | s) × P(token | s, slot)`，并精确合成为原 142 维合法 Token 分布。所有层都使用当前状态与槽位的合法 mask 后重新归一化；禁止先从非法类别采样再拒绝。三个元数组 head 以零权重、零偏置初始化，因此三组都合法时初始质量各为 `1/3`；一元三个子类初始各 `1/3`，二元两个子类初始各 `1/2`，类别内 Feature/Operator 与五个窗口初始等概率。以上只规定中性初值，不是训练期间的固定概率、下限或配额。

该分解不改变部分 AST DAG、状态汇合、父状态枚举、固定均匀 `P_B`、TB 方程、`logZ`、Reward、节点/深度边界或轨迹长度定义。`flat` 与 `arity_hierarchical` 具有独立配置指纹，只保留历史审计、回归兼容和阶段 6 候选来源；正式 Stage 5 入口不得创建或恢复这些旧策略。

这套分层只消除了“某一类别因 Token 个数多而在平坦 softmax 中机械占据更大初始总质量”的局部动作偏置。它没有改变各节点长度下终态表达式数量随组合深度增长的事实，也不保证训练后短表达式占优；相关底层问题、当前实证与待研究方案统一记录在附录 B 首项。

#### 5.2 旧 scalar-logZ 基线的 GPU 搜索入口（2026-08-11 已实现，历史运行只读保留）

- `factor_gfn/gfn/search_runner.py` 提供新 run 与恢复入口；只接受 `cuda`/`cuda:<index>`，CUDA 不可用时立即失败，不允许回落 CPU。策略网络、TB Loss 与优化器位于 GPU，真实因子解释、截面清洗和 Reward 继续使用既有 NumPy CPU 路径。
- 新 run 写入 `runs/real_search/<run_id>/`。不同 seed 通过分别创建新 run 实现；每个 run 固定完整配置指纹、真实数据上下文指纹、Reward Provider 指纹和 CUDA 型号/能力/设备数/运行时摘要。
- 恢复时必须同时匹配配置、数据、Reward、run ID、CUDA 设备环境和检查点设备类型；训练上下文强制为 2010-01-01 至 2018-12-31，Provider 不暴露验证期或 OOS 特征与收益。
- CUDA 强确定性训练固定要求 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并在首次 CUDA 矩阵运算前配置。该值同时进入 CUDA 环境摘要、运行元数据与恢复检查；冲突值立即报错，不允许静默改写。2026-08-10 首次 GPU 启动在第 1 步采样前暴露了这一缺口，失败 run 保持 `step=0`、`optimizer_step=0` 且无评价记录；仅这种尚未开始训练的旧 run 可以在恢复时一次性补录该合同，已产生训练状态的旧 run 禁止迁移。
- 检查点文件统一先暂存加载到 CPU，再由模型与优化器的 `load_state_dict()` 恢复到目标 GPU；Python、NumPy、PyTorch CPU 与各 CUDA RNG 状态在恢复前必须显式验证并规范化为 CPU `torch.uint8` Tensor。禁止用 `map_location=cuda` 将 RNG 字节状态一并搬到 GPU，否则 `torch.cuda.set_rng_state_all()` 无法恢复。
- 每一步原子更新 `evaluations.jsonl`、`step_metrics.jsonl`、训练汇总、最佳候选、latest 检查点和自包含 manifest；另按配置间隔及每次手工目标步保存归档检查点。若崩溃发生在日志落盘但 latest 检查点尚未更新之间，恢复时将超前记录移入 `recovery_archives/` 后再续跑，不静默删除。
- 监控覆盖 Trainer 全部统计、每步墙钟耗时、Reward 请求/有效/唯一数量、真实因子与 Reward 累计耗时、Provider 缓存命中以及当前/峰值 GPU allocated/reserved 显存。缓存命中不重复计入实际因子和 Reward 耗时。每步完成后立即向控制台打印 `current_step`、`optimizer_step`、Loss、Reward、拒绝率、归一化熵及耗时，并刷新该 run 的 TensorBoard 标量与 Reward 直方图。TB 优化诊断同时记录 `delta_mean/std`、前向/后向/Reward 对数均值、Transformer 与 `logZ` 裁剪前梯度、实际裁剪系数、Transformer 绝对/相对参数更新量及 `logZ` 单步更新量；历史检查点缺少这些字段时按空值恢复，不改写旧统计。
- `run_state.json` 使用原子写入记录 `ready/running/completed/failed/interrupted`、正在计算的 `active_step`、步骤开始时间、最后完成时间和最近异常。`factor_gfn.gfn.search_monitor status/watch` 只读取状态、训练统计及逐步性能文件，不加载模型或修改训练产物，可在独立终端持续查看进度与基础告警。
- 正式扩大训练前，允许通过 `factor_gfn.gfn.search_monitor freeze` 为已停止的 run 生成带 SHA-256、文件大小和核心计数的只增不改基线快照；若同名快照后的原始产物发生变化，工具拒绝覆盖。
- `notebooks/run_real_candidate_search.ipynb` 是旧基线的手动入口：旧 run 只使用 `grammar_hierarchical` 完整文法分层策略，恢复模式从原 run 读取冻结配置，只允许提高 `TARGET_STEP`，不得改写原 seed、配置或指纹。历史 `flat`、`arity_hierarchical` 与 scalar-logZ `grammar_hierarchical` run 均只读参与工程对照和后续候选导入；5.3新架构须升级入口、schema和指纹后创建独立 run，不得恢复这些旧检查点。Notebook 由用户手动运行，代码验收不启动真实训练。

后续工作：

- Complexity-conditioned无-anchor正式架构完成极短integration smoke后，只做训练动态健康检查：检查same-N retry、policy与learned-logZ学习率、两组独立梯度裁剪和batch是否出现结构性异常；不把短期Reward/IC高低作为optimizer选择依据，不进行六维网格搜索。无明确异常时优先沿用`policy_lr=1e-4`、`learned_logz_lr=1e-2`、两组`max_norm=5`和`batch_size=8`工程baseline；在训练动态与策略口径最终冻结前，不启动其他seed的批量正式搜索。
- 参数冻结后，按完全相同的模型、搜索空间、Reward 与训练配置执行一个或多个独立 seed，并持续记录吞吐、拒绝率、唯一率、策略熵、耗时、显存、检查点和完整候选清单。
- 若拒绝率持续超过 80%、出现 NaN/Inf、非法动作、显存不足或候选多样性坍缩，先分析原因，不自动改变 Reward、行业中性化、搜索空间或模型规模。

#### 5.3 Complexity-conditioned GFlowNet 重构合同（2026-08-12 已确认，分阶段实现中）

##### 5.3.1 问题定义、目标分布与不变合同

旧 `grammar_hierarchical` 已消除平坦 Token 数量造成的局部动作初始概率偏置，但仍保留不同终态节点数层之间的组合数量优势。标准未条件化 GFlowNet 的节点数层总质量满足：

`P(node_count = n) ∝ Σ_{x: node_count(x)=n} R(x)`。

因此，本轮将 Stage 5 的正式候选生成方法改为外部先选择目标节点数 `N`，再学习该复杂度层内部的 Reward 比例分布：

```text
N ~ q(N)
P_F(x | N) ∝ R_TB(x)
```

第一版 complexity condition 只使用终态 `node_count`，不同时条件化 terminal depth。`depth` 继续作为硬结构约束并按 `(node_count, terminal_depth)` 进行二维诊断；只有在固定 N 后仍观察到无质量增益的 `max_depth` 机械堆积，才另立实验研究 depth conditioning。本轮不加入复杂度 Reward、长度归一化、复杂度惩罚、课程学习、learned `q(N)`、高 Reward 重放或 depth 配额。

以下合同保持不变：

- 六个原始 Feature、52 个非叶子算子、142 个 Token、规范部分 AST、多路径 DAG、canonicalization、父状态枚举和原 `(slot, token_id)` 环境动作；
- `grammar_hierarchical` 的 Feature/UnaryFamily/BinaryFamily、六类文法、Operator 与条件 Window 概率分解；
- 固定均匀后向概率 `P_B=1/n_parents`；
- 原始 Reward 公式、Reward floor、严格行业中性化、Barra 定义、训练期数据边界和固定五日调仓日历；
- 阶段 6 的训练/验证/OOS 切分、方向、联合硬筛选、排序和贪心去相关合同。

`N` 是整条 trajectory 的外部 condition，不是 Grammar Token、环境动作或 AST 节点，不增加 trajectory length。Conditional TB 对每条轨迹 `i` 定义为：

```text
delta_i
  = logZ(N_i)
  + sum_log_pf(a_t | s_t, N_i)
  - log_reward_TB_i
  - sum_log_pb_i

TB_loss = mean(delta_i^2)
```

外部调度分布 `q(N)` 只负责训练任务覆盖，不进入 TB 方程；禁止加入 `log q(N)` 或 `-log q(N)`。固定均匀 `P_B` 的数值定义不变，但父状态重建、父边合法性复核和 trajectory replay 必须携带同一个 `target_node_count`，确保前向与后向工作在同一个 conditioned DAG 上。

##### 5.3.2 Exact-N 可达性、状态身份与缓存合同

对每条 trajectory 先确定 `target_node_count=N`，其 terminal 必须同时满足：

```text
node_count == N
holes == 0
terminal_depth <= max_depth
N <= max_nodes
```

每个 conditioned legal action 不仅要满足旧的 Token、槽位、`max_depth/max_nodes` 约束，还必须保证 action 执行后至少存在一种合法 completion 能够恰好终止于 N。Exact completion feasibility 必须考虑当前节点数、Hole 数量与深度、unary/binary arity 和剩余 depth budget；禁止只用 `max_nodes - node_count` 粗略判断，也禁止“先进入 dead state、最后再拒绝 trajectory”。

代码不得假定 `1..max_nodes` 全部可达。必须在当前 Grammar、`max_depth` 和 `max_nodes` 下解析，并将“是否参加normal discovery”与“normalizer是否exact”彻底解耦：

```text
F = resolved feasible node-count strata
D = normal discovery strata = F
E = exact-normalizer strata = exhaustive evaluated strata
L = learned-normalizer strata = F - E
```

例如当前若解析为`F=(1,...,20)`、`E=(1,2)`，则`D=(1,...,20)`、`L=(3,...,20)`。`exhaustive`只描述完整评价与normalizer类型，不再决定该层是否进入discovery。配置的 `max_depth/max_nodes`、解析后的 `F/D/E/L` 及其生成依据全部写入配置 manifest、运行元数据和指纹。所有底层结构必须依赖配置动态创建，禁止为 6/15、6/20、7/20 或具体 N 写死数组长度和分支。

现有结构身份继续分层：

- `GrammarState.state_key` 只代表 canonical partial AST；
- terminal structural hash 只代表 canonical expression；
- Stage 6 仍按结构哈希跨来源去重；
- 凡是合法动作、模型输入、采样或缓存结果依赖目标 N 的组件，完整 key 必须至少包含 `(state_key, target_node_count, search_space_fingerprint)`。

禁止将 N 写入表达式 structural hash，亦禁止只用旧 `GrammarState.state_key` 缓存 conditioned legal mask。

##### 5.3.3 Balanced discovery scheduler、same-N retry 与公平性诊断

Discovery scheduler 遍历`D=F`，包括exact-normalizer strata。理论覆盖目标为 `q(N)=Uniform(D)`，工程实现使用可确定性恢复的 balanced shuffled cycle，而不是逐轨迹 IID `random.choice`。Scheduler 至少持久化：

```text
resolved normal discovery strata D
current shuffled permutation
cycle index
position within cycle
scheduler RNG state
```

Batch size 与 strata 数量、`max_nodes` 解耦；`batch_size=8` 只作为第一轮诊断起点，不是最终冻结值。每个 batch slot 一旦由 scheduler 分配 N，Reward 无效后的补采必须继续使用相同 N，不得消费下一个 N 顶替。达到配置化 exact-node retry budget 后，fail-closed 跳过整个 optimizer update、记录导致耗尽的 N 并继续 scheduler；禁止换 N、放宽 Reward、关闭行业中性化、降低 Barra 共同期数或自动修改 `q(N)`。

必须按 N 持久化：

```text
requested_count_by_N
valid_count_by_N
successful_update_count_by_N
retry_exhausted_count_by_N
effective_update_rate_by_N = successful_update_count_by_N / requested_count_by_N
```

如果 mixed-N batch 整体跳过，该 batch 内所有 N 均不得计为 successful update。`effective_update_rate_by_N` 低于配置化告警阈值时只提示“均匀请求未转化为均匀梯度暴露”，不自动重新分配配额。

##### 5.3.4 动态 per-N conditional normalizer

旧全局 scalar `logZ` 在新模式下改为按 `max_nodes` 动态创建的独立 Parameter vector：

```python
log_z_by_node_count = nn.Parameter(torch.empty(max_nodes))
```

索引固定为 `N -> N-1`，不得硬编码 15/30 个变量。Normalizer 同时持久化动态长度的：

```text
exact_tb_log_z_by_node_count
exact_log_z_mask
```

两类 strata 使用同一 conditional normalizer 接口：

- `N in E`：该轨迹仍来自normal discovery，forward使用buffer中固定的`exact_tb_log_z[N]`，不使用对应Parameter值；TB只训练conditional policy，exact Z不接收梯度且不得被optimizer momentum、weight decay或checkpoint恢复路径改变；
- `N in L`：forward 使用 `log_z_by_node_count[N-1]`，由 TB 正常学习；未出现在当前 batch 的 N 不应更新。

禁止仅依赖 gradient hook 或 step 前后手动归零冻结 exhaustive index。Exact buffer 必须是 forward 的权威值；Parameter vector 仍保持动态统一结构，便于 mixed-N batch、状态保存和 legacy/conditional 模式隔离。

Policy 参数与 non-exhaustive normalizer 参数使用独立 optimizer group、学习率、梯度裁剪和监控。具体学习率与 max norm 仍待 synthetic 与真实短测确定，旧 scalar `logZ` 参数不直接视为新 vector 的最终配置。

##### 5.3.5 Exhaustive strata、Reward mass 与固定 exact Z

是否 exhaustive 不按 `N<=k` 硬编码。先按 canonical terminal expression 进行 bounded counting：统计 `canonical_terminal_count`、depth distribution、exact reachability 和估计 RealReward 成本；达到配置化 count cap 时立即停止并判为非 exhaustive，无需为了证明空间过大而继续枚举。Exhaustive 判定同时考虑 canonical terminal 数量和完整 RealReward 预算，阈值、预算比例及显式 include/exclude 均保持配置化。

当前文法下已确认：

- `N=1` 有 6 个 canonical terminal，直接 exhaustive；
- `N=2` 有 `106 × 6 = 636` 个 canonical terminal；现有真实 run 中 63 个唯一 N=2 表达式平均完整评价约 0.53 秒，据此全量约 5.6 分钟，P90 粗估仍低于 8 分钟，因此当前已确认的 resolved 结果将 N=2 直接 exhaustive；
- 以上是当前 Grammar 和实测预算得到的解析结果，不得实现为 `if N <= 2`。

Exhaustive evaluation 必须可恢复并逐条保存结构、formula、prefix Token、Reward 拆解、valid/invalid、拒绝原因、context/provider/reward 指纹和 `source=exhaustive_full_evaluation`。只有在完整 canonical coverage 可证明、结构不重复且 Provider/Reward 上下文一致时，才能宣称该层 exact Z 已知。

当前 TB 对“指标有效但 raw Reward 低于 floor”的候选实际使用：

```text
R_TB(x) = max(R_raw(x), reward_floor)
```

无效 candidate 仍是零 target mass，不应用 floor、不进入 partition sum，但必须保留审计。因此 exhaustive 层同时保存：

```text
exact_raw_reward_log_mass
  = log(sum_{valid x} R_raw(x))

exact_tb_log_z
  = log(sum_{valid x} R_TB(x))
  = logsumexp(valid terminal log_reward)
```

`exact_raw_reward_log_mass` 只用于经济口径与审计；固定 normalizer 必须使用与 TB 实际 `log_reward` 完全一致的 `exact_tb_log_z`。若某个 exhaustive N 没有任何有效 terminal，则 `Z_N=0`、不存在有限 `logZ_N`；正式run在启动normal discovery前整体fail-closed并要求人工审查，不得静默将该层改成learned Z或从`D`中删除。

##### 5.3.6 Training-only calibration

Learned-normalizer strata `L` 在正式 joint training 前使用训练期 conditioned trajectory 计算：

```text
logZ_implied
  = log_reward_TB
  + sum_log_pb
  - sum_log_pf
```

每个 N 同时记录：

```text
calibration_requested
calibration_valid
median(logZ_implied)
logmeanexp(logZ_implied)
P10 / P25 / P75 / P90
IQR
```

第一版使用 `median(logZ_implied)` 作为learned scalar的工程初始化，`logmeanexp`及分布统计只作高方差重要性估计诊断，不增加auxiliary loss。当前6/20 no-anchor正式入口优先复用已完成的旧6/20 training-only diagnostic中N=3...20的median，作为新Parameter的**初始化常数**；复用前必须严格核对Grammar/operator/interpreter、provider/data context、Reward config、reward floor、`max_depth=6`、`max_nodes=20`及逐N来源统计。只复制通过验证的数值和审计来源，不恢复旧Parameter、model、optimizer、scheduler、anchor或checkpoint state。

Exact-normalizer strata `E` 直接使用固定`exact_tb_log_z`，不参加implied-logZ calibration。64/128 calibration不再是所有L层进入Step 12前的强制全量步骤，而是问题驱动的targeted fallback：只有Step 12显示某个N的历史median明显失配，才在**全新的、optimizer step为0的no-anchor training state**中只重校准该N；其他N继续使用已验证的历史median。Targeted calibration固定`minimum_valid=64`、`maximum_requested=128`、`comparison_window=16`、`median_abs_tolerance=0.25 logZ`、`IQR_abs_tolerance=0.50 logZ`作为工程预检合同，不进入第12步调参，也不声称最优；达到64 valid后比较最近两个不重叠16-valid窗口，若不稳定则每增加16 valid复检，到128 requested仍不足或不稳定即fail-closed。Calibration使用独立、可恢复scheduler/RNG，不消费normal discovery scheduler，冻结policy且不执行optimizer update，只读取training-only，禁止validation/OOS；禁止在已经训练过的Trainer中途重置任何logZ。

##### 5.3.7 Unified normal discovery 与 exhaustive registry cache

正式Stage 5不再为exhaustive strata创建独立anchor训练路径。所有`N in D=F`都由同一个balanced shuffled-cycle scheduler分配normal discovery slot，执行同一条流程：

1. scheduler分配固定target N；
2. conditional policy在exact-N合法mask下生成trajectory；
3. Reward无效时只做same-N retry；
4. 有效trajectory进入统一mixed-N TB batch；
5. `N in E`使用fixed exact Z且只产生policy gradient，`N in L`同时更新policy与对应learned Z。

不允许因某层已完成exhaustive evaluation而跳过其normal discovery、用其他N替换其slot，或另设anchor frequency/batch/scheduler/RNG/optimizer step。设计原则固定为：**能用exhaustive信息提高exact Z和缓存效率，就到这里为止；不要仅因为某层可exhaustive，就再人为创造一套独立训练机制。**

当normal discovery再次生成`N in E`的terminal时，必须优先按canonical structural hash从已完成的exhaustive registry读取RewardAssignment，避免重复调用RealReward。Registry命中必须同时核对terminal身份、valid/invalid、stored Reward/log_reward、provider、data context、Reward config和reward floor，并明确审计`source=normal_discovery`及`reward_source=exhaustive_registry_cache`；invalid记录仍返回原拒绝原因并触发same-N retry。任何缺失、指纹不一致或数值合同不一致都fail-closed或显式回退fresh evaluation并记录原因，禁止伪装成cache hit。

N=1/2允许跨等价search space复用exhaustive registry，但不得只依赖全局search-space fingerprint。每个目标run只在初始化阶段执行一次严格equivalence verification：在目标Grammar下重新枚举该N的全部canonical terminal并逐层核对structural-hash全集，同时核对Grammar/operator/interpreter semantics、provider/data context、Reward config与reward floor。全部一致才允许复用stored Reward和exact Z，否则必须fresh evaluation；验证完成后，每个normal discovery candidate只按structural hash查询只读registry，不得重复枚举全集。该逐层证明只放宽与N=1/2终态集合无关的全局边界差异，不放宽任何计算语义或数据指纹。

##### 5.3.8 配置、检查点、旧 run 与实施边界

新的正式模式使用独立的no-anchor architecture schema（首版命名为`factor_gfn.complexity_conditioned_no_anchor.v1`），并升级config、model、trainer、checkpoint和运行manifest schema。正式active path的配置、schema、指纹、metadata及checkpoint不得再包含anchor frequency、batch size、scheduler、cycle/position、RNG、optimizer step、loss或resume state。历史anchor实现及其旧run可只读保留，但不得由formal conditional Stage 5 runner调用。

新正式模式将以下内容纳入指纹与兼容检查：

```text
max_depth / max_nodes
resolved F / D / E / L
complexity scheduler config and state
exact-node retry config
conditional policy features
normalizer mode and vector length
exact_tb_log_z values and mask
calibration config and completion state
exhaustive counting/evaluation contract
per-stratum exhaustive registry reuse proof
```

旧`flat`、`arity_hierarchical`、`grammar_hierarchical + scalar logZ` run全部保持只读，仍可作为Stage 6候选来源；旧checkpoint不得恢复到complexity-conditioned模型。所有带anchor training state的旧conditional checkpoint，包括6/15 smoke和当前6/20 diagnostic checkpoint，都必须被新的no-anchor formal schema拒绝。当前6/20 run继续按创建时的原Trainer、配置、checkpoint context和Notebook运行到既定停止点，不得中途修改；它只提供depth统计结论，模型、optimizer、scheduler、anchor state、learned scalar和checkpoint均不得进入后续正式训练。

所有正式实现继续支持配置化`max_depth/max_nodes`，但当前`max_nodes=20`是人为固定complexity upper bound，不根据`node_count==max_nodes`占比、Reward或IC自动扩大。当前真实运行与迁移顺序为：

1. 用 6/15 做很短的兼容性 smoke，只回答新代码是否破坏已验证环境；
2. 兼容性smoke通过后建立独立`max_depth=6/max_nodes=20` diagnostic run；`max_nodes=20`为本轮人工固定上限，不做自动扩边判断；
3. 当前6/20沿旧anchor架构运行，只使用training-only normal discovery候选诊断depth分布、depth=6饱和、按depth的Reward/IC和评价成本；只对max_depth输出人工建议，其checkpoint不进入正式训练；
4. 只有6/20输出`consider_expansion`并经人工确认，才新建独立7/20；仍有证据时再新建8/20，禁止在原run中修改边界；
5. 最终depth确认后，创建极短no-anchor integration smoke，验证`D=F`、E/L normalizer选择、registry cache、same-N retry、mixed-N梯度、无anchor state和确定性恢复；不判断Reward/IC质量；
6. smoke通过后执行精简的第12步训练动态健康检查并冻结配置，才创建多个独立seed的正式Stage 5 run；
7. 所有长时间RealReward、真实smoke、diagnostic和正式训练均由用户手动启动。

分阶段验收顺序更新为：保留第0–10步和当前第11步作为历史实现/诊断证据；当前6/20完成后确认最终depth；实现no-anchor formal schema与unified normal discovery；执行极短no-anchor integration smoke；完成精简后的第12步训练动态健康检查与冻结；再进入多seed正式训练。每一阶段完成测试和结果记录后暂停，未经人工确认不得自动进入下一阶段。

##### 5.3.9 第12步：最终训练动态健康检查与冻结

第12步不再全量重做non-exhaustive calibration，也不调整anchor frequency/batch size、exhaustive count/budget threshold或搜索边界。N=3...20先使用5.3.6所述、经语义核验的历史median初始化；E resolution属于搜索空间预检；当前搜索边界已冻结为`max_depth=6`、`max_nodes=20`，不建立7/20。

优先沿用以下工程baseline，不声称数学最优：

```text
policy_lr = 1e-4
learned_logz_lr = 1e-2
policy_max_norm = 5
logz_max_norm = 5
batch_size = 8
```

待检查参数只包括same-N retry budget及上述两组学习率、两组梯度裁剪和batch size，但禁止把它们展开成六维超参数搜索。无明确异常时不重新搜索；retry budget优先使用当前depth diagnostic的真实`retry_exhausted_count_by_N`与effective update rate判断，若budget=2已表现稳定则直接冻结2，只有某些N系统性耗尽时才调整。

成功标准只看训练健康：per-N delta mean/std与finite TB loss；learned `logZ_N`轨迹、更新幅度和振荡；policy gradient norm、clipping rate、实际参数更新、entropy和non-finite rate；逐N requested/valid/retry-exhausted/successful/effective-update统计；吞吐、factor/reward耗时、GPU显存与wall time；以及checkpoint恢复后的deterministic continuation。32个logical batches只是初始总预算，不能自动代表每个L层证据充分；每个N必须同时报告valid trajectory count与successful gradient exposure，任一不足都只能判`insufficient_evidence`。每层分别保存initialization/pre-update、early、late TB delta，以及initial/current/net-change logZ，防止短run内learned logZ自行修正后掩盖初值失配。只有证据充分且显示异常的N才进入targeted recalibration候选；重校准完成后必须从全新no-anchor training state启动，禁止原Trainer中途重置。短期Reward/IC P90/P99和高质量Alpha数量不用于选择optimizer参数。

### 阶段 6：因子筛选与回测验证

目标：从阶段 5 搜索结果中得到稳定、低相关的 Alpha 池。

#### 6.1 冻结输入、时间切分与防泄露合同（2026-08-10 已提前完成）

- 本步只建立阶段 6 数据上下文和候选注册表，不重新计算候选指标、不执行筛选或回测。训练、验证、最终样本外请求区间分别固定为 `2010-01-01` 至 `2018-12-31`、`2019-01-01` 至 `2020-12-31`、`2021-01-01` 至 `2025-12-31`，并同时记录各段在交易日轴上的实际首末日期。
- 三段共用一条从训练期首个五类 Barra 均可评价日期锚定、此后每 5 个交易日延伸的全局调仓日历。计划调仓日若标签越界或 Barra 截面不足，只排除并记录原因，禁止平移补位或在分段边界重新定相位。
- 收益标签继续使用 `open[t+6] / open[t+1] - 1`；信号日 `t`、入场日 `t+1`、退出日 `t+6` 必须全部位于同一分段。验证和样本外可使用此前历史作为因果时序算子 warmup，但不得使用未来特征或跨边界标签。最终样本外候选评价默认锁定，第一步必须保持评价次数为 0。
- 多头方向不继承旧 Reward，在后续步骤只按阶段 6 训练期 RankIC 确定并冻结；验证和最终样本外只能复用冻结方向。
- 候选以运行目录为来源单元，强制配套读取 `evaluations.jsonl` 和 `run_metadata.json`；逐条以 prefix Token 重建表达式并核对公式、结构哈希、节点数、深度及 Provider 指纹。默认只登记主分支搜索候选，确定性重放记录只参与审计。按结构哈希去重并保留全部来源；旧 valid、Reward 和拆解仅作来源信息，不参与阶段 6 排序。
- 不同 context/provider/reward 口径默认禁止混合；确需导入时必须显式划分兼容组。`experiment_manifest.json`、`best_candidate.json`、`training_stats.json` 和 `determinism_report.json` 如存在则用于辅助完整性审计，不作为候选导入的强制输入。
- 本步仅创建 `factor_gfn/backtest/__init__.py`、`context.py` 和 `selection.py` 及自动化测试；不创建空壳回测模块，不新增人工验证 Notebook，不写入或修改历史运行产物。

#### 6.2 联合硬筛选与贪心去相关合同（2026-08-11 已确认）

- 阶段 6 不是“训练期先筛完，再由验证期另筛一次”。候选必须先在统一上下文中分别计算训练期与验证期指标，再一次性应用研报披露的联合硬条件。`test_ic`、`test_long_ir` 在项目字段和文档中统一解释为 2019-2020 验证期指标，不得与 2021-2025 最终样本外混用。
- 联合硬筛选固定为：训练期 `abs(train_ic) > 0.01`、验证期 `abs(validation_ic) > 0.01`、`train_ic * validation_ic > 0`、训练期 `train_long_ir > 0.25`、验证期 `validation_long_ir > 0.25`，以及训练期 `barra_ts_corr < 0.7`。阶段 6 必须重新计算这些指标，阶段 5 保存的 valid、Reward 或拆解只作来源审计。
- 因子多头方向只由训练期 RankIC 确定并冻结；验证期 Long IR、验证期多头超额收益和最终样本外指标必须沿用该方向，禁止根据验证期或样本外结果再次翻转。
- 通过联合硬条件的候选按 `abs(train_ic)` 降序进入贪心去相关。候选之间的多头超额收益率序列相关性使用训练期序列估计；与已保留任一因子的相关性达到或超过 `0.7` 时不保留，全部低于 `0.7` 时才加入 Alpha 池。
- 训练期同时承担排序、Barra 去暴露和候选池相关结构确定，保持同一信息集。训练期约 386 个有效调仓期，显著多于验证期约 97 期；后者在扣除 warmup、缺失和严格行业中性化失败日期后更容易接近 60 期最低共同样本门槛，不作为第一版相关结构选择样本。
- 验证期仍须计算候选的 Barra 相关及候选两两多头超额收益相关，作为风格漂移、同质化和跨阶段稳定性诊断并持久化；这些诊断不得在本轮筛选中改变阈值、排序或贪心结果。若未来采用 `max(train_corr, validation_corr)` 或训练与验证拼接相关性，必须登记为新的筛选实验口径，不得覆盖第一版结果。
- 2021-2025 最终样本外在联合硬筛选、方向、排序、阈值和 Alpha 池全部冻结前保持锁定；样本外只做最终评价，不参与淘汰、补选、翻转方向或调整相关阈值。

上述安排保留研报明确授权的验证用途，即通过 `validation_ic`、方向一致性和 `validation_long_ir` 检验预测能力延续，同时避免在数千候选上反复利用同一短验证期构造相关矩阵并优化池结构。它不宣称训练期相关性优于其他方案，而是第一版复现中可审计、较少额外假设的选择。

#### 6.3 后续工作

- 样本内、验证集和最终样本外指标；
- 研报硬性筛选条件的可配置实现；
- 表达式结构去重；
- 截面相关性和多头收益序列相关性分析；
- 贪心去相关；
- 分组、多头、多空、换手和交易成本分析；
- 按年份和市场阶段检查稳定性。

### 暂缓阶段

- 正式训练模型规模、batch size 与搜索空间的超参数网格探索；
- 日频人工衍生特征；
- 分钟聚合特征；
- 原始分钟数据与 MemMap block cache；
- AlphaEval；
- DPP；
- LGBM 多因子合成；
- 指数增强组合。

只有原始日频 K 线最小闭环稳定后，才评估是否进入这些阶段。

## 7. 候选代码结构

以下结构反映当前已经创建的核心模块；`backtest/` 当前只包含阶段 6 第一步提前冻结所需的上下文与候选注册表：

```text
factor_gfn/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── downloader.py
│   ├── industry.py
│   ├── masks.py
│   └── preprocess.py
├── grammar/
│   ├── __init__.py
│   ├── config.py
│   ├── operators.py
│   ├── tokens.py
│   ├── partial_ast.py
│   ├── grammar_state.py
│   └── expression.py
├── evaluator/
│   ├── __init__.py
│   ├── ops_impl.py
│   ├── interpreter.py
│   ├── cross_section.py
│   └── metrics.py
├── barra/
│   ├── __init__.py
│   ├── config.py
│   ├── factors.py
│   ├── pipeline.py
│   └── portfolio.py
├── gfn/
│   ├── __init__.py
│   ├── config.py
│   ├── state_adapter.py
│   ├── model.py
│   ├── trajectory.py
│   ├── policy_sampler.py
│   ├── reward.py
│   ├── real_data.py
│   ├── real_reward.py
│   ├── search_runner.py
│   ├── loss.py
│   ├── trainer.py
│   └── checkpoint.py
└── backtest/
    ├── __init__.py
    ├── context.py
    └── selection.py
```

原则：

- 只在对应阶段开始时创建实际需要的模块；
- 避免只有空壳、没有职责的文件；
- 文法定义与数值实现分离，但必须由统一算子注册信息校验一致性；
- 下载、预处理、训练和回测均不依赖硬编码绝对路径；
- Windows 默认入口使用 Python 或 PowerShell，不默认创建 `.sh` 脚本。

## 8. 测试与验收原则

- 优先使用小型合成数据验证边界和未来数据泄露；
- 大规模下载、完整清洗和训练由用户在 Notebook 或明确命令中手工启动；
- 不以“代码可运行”替代数据合同验证；
- 每个阶段至少检查：输入合同、输出合同、重复键、NaN/Inf、边界样本和可重复性；
- 优化实现必须与参考实现进行数值对照；
- 不修改原始缓存来修复下游问题；
- 所有长任务应支持检查点或可恢复运行。

## 9. 当前待确认问题

### 数据阶段

- 后复权行情和不复权收盘价的最终键覆盖是否足够一致？
- adata 的 `amount` 与 `volume` 单位如何换算为每股 VWAP？
- ST 和风险警示状态是否有历史点时数据？
- 当前股票主表是否包含已退市股票？
- 当前静态 ST 股票池口径造成的历史偏差应如何在最终报告中披露？
- 成交量是否需要缩尾，以及零成交量是否排除在分位数计算之外？
- 清洗后是否保留完整股票—日期网格和有效值 mask？
- 矩阵存储格式和数据版本标识如何设计？

### 标签与评估

- 停牌和涨跌停导致无法成交时如何处理？

### GFlowNet

- Complexity-conditioned模型主体、exact-N、`D=F` balanced normal discovery、per-N exact/learned normalizer和exhaustive registry合同已确认；正式架构不再使用anchor。N=3...20先复用通过语义核验的旧6/20 training-only median作为初始化常数，64/128 median/IQR稳定性calibration只作为异常N的targeted fallback；
- 6/15只作为已完成的历史兼容性smoke；legacy 6/20 depth diagnostic已经完成并支持冻结`max_depth=6/max_nodes=20`，不建立7/20。旧模型及checkpoint不进入no-anchor正式训练，但其depth结论、N=1/2 registry/exact Z和N=3...20 implied-logZ诊断可按各自严格等价性合同复用；
- 第12步只检查same-N retry、两组学习率、两组梯度裁剪和batch的训练健康；无明确异常时不做网格搜索，Reward/IC短期高低不作为optimizer调参目标；
- 正式训练总步数、运行数量和 GPU 硬件预算仍待按新入口与人工运行预算确认；
- 训练期间候选表达式持久化归档、验证检查点选择和训练后补采规模；
- 数值指纹、退化因子和高度相关因子的阶段 6 去重与筛选细节。

## 10. 变更与决策日志

### 2026-08-04

- 建立本开发规范。
- 确认以《GFlowNet_日频K线因子挖掘完整流程》作为当前主要路线，以原始 PDF 核实研报事实。
- 确认第一版只做原始日频 K 线六特征。
- 确认 VWAP 在预处理阶段使用后复权乘数构造，不混入原始下载文件。
- 确认下载运行期间不迁移 `data_loader.py`。
- 目录结构采用渐进式创建，不提前生成所有空模块。
- 当前简称以 `ST`、`*ST`、`S*ST`、`SST` 开头的股票作为静态股票池排除项，不删除基础行情；该口径不代表历史点时 ST 状态。
- 上市不足 180 个自然日作为逐日股票池资格 mask，不删除上市初期基础行情，以保留时序算子的历史窗口。
- 数据质量无效行保留股票—日期键，并将六个特征统一设为 NaN；股票池不合格行保留基础特征，仅在资格 mask 中排除。
- 第一版不做年度长期停牌过滤，也不对原始 `volume` 预先缩尾；`cs_winsorize` 保留为表达式算子候选。
- 完成第二阶段文法初版：6 个叶子、52 个算子、五个窗口和 142 个表达式 Token。
- 初版曾采用前序固定槽位生成；该生成口径已于 2026-08-05 被规范化部分 AST DAG 完全替代，不再保留为可执行状态机。模型特殊符号继续与表达式 Token 空间分离。
- 深度和节点限制保留为可配置工程参数，当前默认值为 10 和 30。
- 表达式去重使用不依赖 action ID 的规范结构哈希；第一版仅规范化四个明确交换算子。
- 建立动作空间版本与 SHA-256 指纹，防止 Token ID 顺序在后续开发中静默漂移。
- 确认第三阶段非时序算子的保护运算、比例算子和截面变换口径，并完成 31 个 NumPy 基准实现。
- 确认第三阶段21个时序算子的窗口、缺失值、排名、EMA和回归口径，并完成 NumPy 基准实现。
- 完成第三阶段表达式解释器：固定输入为 `(date, feature, stock)`，六特征顺序为 `open/high/low/close/vwap/volume`；通过后序栈统一分派 52 个算子，输出 `(date, stock)`，暂不引入缓存、Numba 或 MemMap。
- 完成第三阶段5日指标与性能基线：采用已确认的次日开盘至第六日开盘复现标签，区分逐日分析 IC 和5日非重叠 reward IC，并实现多头超额、年化 IR、覆盖率和两类相关性；中等规模基线暂未发现需要提前优化的瓶颈。
- 整理项目目录：原始数据和全部下载断点保持原位；下载实现迁入 `factor_gfn/data/downloader.py`，Notebook 改用包内导入，根目录暂留 `data_loader.py` 兼容入口；空的训练、回测和脚本目录延后到对应阶段创建。
- 新增阶段2和阶段3验证 Notebook：阶段2执行纯文法随机集成检查；阶段3先以合成矩阵覆盖算子、解释器和5日指标，再在下载完成后对限量真实行情执行只读贯通，不生成或修改数据文件。
- 将数据准备与阶段3验证彻底分离：`prepare_daily_data.ipynb` 调用 `factor_gfn/data/preprocess.py` 生成清洗长表、六特征张量、有效性/股票池 mask 和索引；阶段3 Notebook 只消费 `data/processed/`，不再合并原始行情或计算 VWAP。

### 2026-08-05

- 将第一版手工 Barra 风格集合调整并固定为 Market Beta、Size、Momentum、Volatility、Liquidity 五类；Value 因依赖财报或估值数据且超出原始日频 K 线挖掘范围而移除。
- 将 `barra_ts_corr` 固定为候选因子与五个 Barra 因子的5日 Top 10%减Bottom 10%多空收益序列 Pearson 相关性的最大绝对值，不使用逐日截面相关。
- 将 `barra_ts_corr` 的最低共同有效调仓期数由 20 提高为 60；不足 60 期返回 NaN，并在 `< 0.7` 硬筛选中直接判为不通过。
- reward 参数保持集中可配置，第一版默认 `LONG_IR_LAMBDA=0.3`、`LONG_IR_CAP=2`、`BARRA_TS_PENALTY_MU=0.2`；Long IR 奖励项最大乘数为 `1.6`。
- 所有超参数必须记录默认值、实际值、来源、理由、搜索范围、数据版本、随机种子和最终采用结果，禁止无法追溯的散落硬编码。
- Market Beta 固定使用全 A 市值加权市场收益：股票收益取后复权收盘价相邻日收益，市场权重使用前一交易日市值并在有效截面归一化；第一版默认流通市值、可靠性不足时可配置切换总市值，Beta 为过去 252 个交易日的 `Cov/Var`，不减无风险利率。
- Market Beta 与 Size 所需的历史市值将由历史股本和不复权收盘价自行构造；不再需要历史 PB。
- 确认 adata 历史股本接口字段：`total_shares` 为总股本，`list_a_shares` 为流通 A 股股本；新增可续传的 `download_stock_shares()` 和 Notebook 运行单元，完整保留六个接口字段及下载断点。失败股票不写成已完成，原样重跑会继续重试。
- Liquidity 若采用换手率，使用已保存的股单位 `volume / list_a_shares` 自行计算，不为 `turnover_ratio` 重下全部行情。
- 明确 Barra 风险惩罚不将五个风格因子等权合成：五个风格分别生成 Long-Short 收益序列，候选序列逐一计算 Pearson 时间序列相关，最终取 `max_k |corr_k|`。组合内部股票仍按 Top 10% 与 Bottom 10% 各自等权。
- 经真实接口探针确认，部分早期历史股本记录正常缺失 `limit_shares`。下载校验调整为：该字段允许为空；核心字段无效时只过滤对应记录并统计，整只股票过滤后为空或同日记录冲突时才判定失败。
- 更新阶段三截面清洗口径：候选因子在每个评价/调仓截面执行 1%/99% 缩尾、申万一级行业哑变量 OLS 残差化、`ddof=0` z-score；缺失行业股票不参与回归但保留缩尾值进入 z-score。五个 Barra 风格仍只做缩尾与 z-score，禁止行业或市值中性化；原始解释器输出保持不变。
- 候选因子市值中性化暂缓，待阶段六回测评估市值暴露后决定；当前由 reward 中的 Barra Size 惩罚把关。
- 第一版 Barra 集合继续仅保留 Market Beta、Size、Momentum、Volatility、Liquidity；Value、Earnings Yield、Growth、Leverage、Profitability、Dividend Yield 全部暂缓，仅在附录记录参考公式和数据源。
- Barra 参数调整为：Beta 252 日且 `min_periods=120`；Volatility 252 日且 `min_periods=120`；Liquidity 默认 60 日平均换手率；Size 固定 `log(流通市值)`；Momentum 保持 252/21。
- 早期曾接入 `get_industry_sw(stock_code=code)` 静态行业下载；该方案因限流且无法提供历史点时行业，已于 2026-08-07 被逐日 `swind` 数据替代，相关下载器、断点和静态加载器均已删除。
- 阶段三候选评价接口将行业标签设为必需输入，并实现“1%/99% 缩尾 → 申万一级行业 OLS 残差化 → `ddof=0` z-score”；支持静态 `(stock,)` 和未来点时 `(date, stock)` 行业标签。Barra 清洗入口保持不变。
- 移除 `ts_corr`、`ts_cov` 的交换律标记：二者虽数值对称，但不纳入交换去重，以保持与 `ts_beta`、`ts_orth` 的方向性一致；交换规范化仅用于 `add`、`mul`、`max2`、`min2`，动作空间内容与指纹不变。
- 废止前序固定槽位生成状态机，改用带显式 Hole 的规范化部分有序 AST DAG；前序和后序仅保留为完整表达式序列化格式。
- 固定后向策略为 `P_B(parent|child)=1/n_parents(child)`；父状态通过撤销所有可撤销前沿节点、重新规范化并按不同父状态键去重获得，状态图不允许平行边。
- 完整前向转移动作改为 `(open_slot, token_id)`；142 个 Token 内容、ID 和原指纹保持不变，另行建立状态模式和转移模式指纹。
- 完成阶段四 Reward 适配：严格实现 `abs(train_ic) × (1 + lambda × clip(train_long_ir, 0, cap)) × (1 - mu × clip(barra_ts_corr, 0, 1))`，分别保存五个带符号 Barra 相关系数、最大暴露风格、原始/稳定化/log reward，并加入上下文隔离的进程内 LRU 缓存。行业数据未完整时仅允许通过显式配置暂时跳过候选行业中性化。
- 完成阶段四 TB Loss：新增全局可学习 `logZ`、经过校验的轨迹 Reward 挂载合同及 float64 Delta；固定均匀后向概率继续使用每一步子状态的不同父状态数，TB batch 保留同终态的不同采样路径。
- 完成阶段四合成 Trainer 闭环：实现批量采样、Reward Provider、有效轨迹补采、TB 反传、两组 Adam 学习率、联合梯度裁剪、精确联合策略熵、拒绝率/有效 batch 监控、原子检查点和随机状态恢复，并提供无需真实数据的手动验证 Notebook。
- 新状态指纹为 `5301fb14e197376dfc2a5aaf9c398aa47642bcc889f6eef87278839142b824d9`，新转移指纹为 `bb9c95a05c0bccb375c47828fb6d95821db1f625f4a2513ba55e7151bac55a4b`。
- 部分状态与终态仅对 `add`、`mul`、`max2`、`min2` 合并交换等价子树；所有其他二元算子保留语义参数位置。

### 2026-08-06

- 保留原始联合策略熵，并新增 `H/log(K_legal_joint)` 归一化熵监控；强制状态因 `K_legal_joint=1` 而不纳入归一化熵均值。
- 明确 `expression_unique_rate` 只基于规范结构哈希，无法识别更广泛的数值等价或高相关因子；将数值指纹和 batch 因子值相关性列为真实训练监控的待优化项，不在文法层进行激进代数化简。

### 2026-08-07

- 同步项目状态：阶段 1–3 已完成，阶段 4 六个子步骤及合成训练闭环已完成，当前转入真实 Reward 接入与正式训练准备。
- 明确行业数据未完整时只允许跳过行业中性化进行合成或显式降级联调，不得据此完成正式训练结论。
- 更新实际代码结构、当前执行边界和待确认问题；原始数据、断点、处理结果、研报及模型检查点继续排除在 Git 之外。
- 申万行业正式来源由 adata 当前静态接口替换为 `参考文件/swind/` 逐交易日点时 CSV；源数据覆盖 2008-12-01 至 2026-08-06，并同时保留申万一、二、三级代码和名称。
- 新增长表合同 `factor_gfn.industry_sw_daily.v1`：股票及三级行业代码去除市场后缀但保持六位字符串，以 `daily_clean.parquet` 行情键左连接，输出键必须与行情长表完全一致；候选因子行业中性化使用当日 `sw_code_1`。
- 新增严格源数据检查、DuckDB 流式原子构建、输出覆盖 QA、`int32` 点时面板加载及 `prepare_industry_data.ipynb`；`-1` 表示行业缺失，旧 adata 静态加载器及下载入口已删除。
- 完成点时行业全量构建：输出 13,070,721 行、4,027 个交易日、5,424 只股票，与 `daily_clean.parquet` 双向键差异均为 0；三级行业缺失率均为 0.1923%，重复键及同日代码—名称冲突均为 0。
- 一级行业 `int32` 点时矩阵已实际接入阶段三候选因子中性化并产生有限结果。由此删除旧 adata 行业下载器、静态加载器、旧 13,954 字节 Parquet 和 51 个未完成分片；4,296 个源 CSV 同盘迁移至 `参考文件/swind/`，不进入 Git。
- 完成真实 Reward 人工表达式预检与性能基线；确认存在多个有效 Reward 候选、五条 Barra LS 均可计算相关性、Provider 缓存命中且训练标签不越过训练边界。小样本截面跳过行业回归的警告属于可追踪的数据完整度诊断，不改变 60 期门槛或行业中性化总口径。
- 完成 CPU 五步最小真实训练：5 步均发生有效参数更新，拒绝率最高为 1/3，非法动作率为 0，无跳过更新及 NaN/Inf Loss；Transformer 与 `logZ` 均发生有限变化。
- 完成确定性续跑验收：第 4 步检查点的模型、`logZ`、优化器、步数和 Python/NumPy/PyTorch 随机状态完整恢复，恢复后的第 5 步与连续五步运行逐项一致。
- 为每个候选 Reward 增加 `neutralization_skipped_dates` 与 `neutralization_skipped_rate` 持久化诊断，并修复实验 manifest 的自包含文件清单。上述两项是在五步实验后发现的非阻塞产物完整性修复；历史 run 保留原样，新运行使用修复后的合同。
- 完成阶段 1–4 收口检查：旧 adata 静态行业下载代码、旧行业 Parquet 与断点均已移除，原始数据、处理产物、运行目录和检查点继续由 `.gitignore` 隔离；项目转入阶段 5 前置状态。

### 2026-08-10

- 原阶段 5 筛选与回测顺延为阶段 6；已完成的数据上下文与候选注册表合同改记为阶段 6.1 前置合同。最终样本外截止日固定为 2025-12-31；三段共用单一五交易日相位，区间末端标签越界只排除不补位；OOS 候选评价默认锁定。
- 真实数据自动检查得到 727 个计划调仓日，其中训练、验证、OOS 分别计划 387、98、242 期，各因边界标签排除 1 期，最终有效 386、97、241 期；全局锚点仍为 2011-01-18，OOS 候选评价数为 0。
- 当前最小真实训练的 `evaluations.jsonl` 共 15 行：13 行主分支候选全部通过公式、Token、结构哈希和来源指纹审计并形成 13 个唯一结构，2 行确定性重放仅审计不登记。历史 run 缺少中性化跳过字段且 manifest 未列出自身，均按历史产物缺口记录，不改写旧文件。
- 本步未新增人工验证 Notebook；使用合成自动化测试和真实只读装配验证，后续候选指标仍须在统一阶段 6 上下文中重新计算。
- 当时的新阶段 5 单纯负责 GPU 真实训练与候选生成，并将旧 scalar-logZ 基线配置冻结为 `max_depth=6`、`max_nodes=15`、`d_model=128`、4 heads、4 layers、feedforward 512、dropout 0、batch size 8，由 `build_stage5_real_training_config()` 统一生成可指纹化配置。该条是历史决策记录；2026-08-12起由5.3 complexity-conditioned合同取代其“当前正式配置”地位，旧run及候选继续只读保留。
- 完成阶段 5.2 GPU 正式搜索入口的静态与合成验证：强制 CUDA、训练区间隔离、配置/数据/Reward/CUDA 环境恢复校验、逐步候选持久化、latest 与周期归档检查点、孤儿日志保留恢复、耗时和显存监控均已接入。首次人工启动发现 CUDA 确定性模式还要求在 CuBLAS 运算前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；补齐后恢复又暴露 `map_location=cuda` 会错误搬移 RNG ByteTensor，现已改为 CPU 暂存加载并显式规范化全部 RNG 状态，同时为零步失败 run 保留受限兼容恢复。完整自动化回归 195 项通过，真实 GPU 训练仍由用户通过 Notebook 手动启动。
- 阶段 5 正式配置首个 GPU run 已完成 20 步启动验收：`current_step=optimizer_step=20`，190 次评价中 160 次有效、186 个唯一结构，无跳过更新、非法动作或非有限 Loss；累计耗时约 5585.53 秒，其中因子解释约占 83.2%。该 run 已冻结为 step 20 工程性能基线，不作为行业中性化口径修订后的正式候选池续跑依据。
- 为后续长运行加入逐步控制台输出、TensorBoard、状态机式 `run_state.json` 和独立只读监控入口；基线冻结文件保存关键产物 SHA-256，旧 run、检查点和评价记录均不改写或删除。历史 20 步已一次性导出为可回读的 TensorBoard 事件。
- 正式 Reward 行业中性化改为 fail-closed：回归无法执行的候选日期整行排除，缺失行业股票排除，调仓日历不移相且不补位；新增逐日期完整失败审计。Reward 上下文与真实 Provider schema、指纹同步升级，因此旧 20 步检查点只保留为工程基线，禁止续接到新口径正式候选池。完整自动化回归现为 204 项通过。
- 完成性能优化批次一：解释器继续计算完整训练历史以满足时序算子 warmup；进入 Reward 后立即按冻结全局日历压缩为 386 个调仓截面，候选只执行一次“缩尾 → 行业中性化 → z-score”，RankIC、Long IR 与候选 LS 复用同一已清洗矩阵，五条 Barra LS 同步压缩到相同日期。行业中性化失败明细从紧凑行号映射回原始评价日期轴，日期相位、有效掩码和 Reward 数值已用旧严格路径作合成对照；完整自动化回归为 205 项通过。Reward 上下文与真实 Provider 指纹再次升级，禁止旧检查点跨实现恢复；解释器复制、累计量滚动核、Numba 和子表达式缓存仍留待后续批次。
- 完成性能优化批次二：`FactorInterpreter` 对不含 Inf 的只读 `float64` mmap 采用借用模式，避免初始化时复制完整历史张量；可写、非 `float64` 或含 Inf 输入继续创建规范化隔离副本。叶子节点在栈内使用只读特征视图，仅当叶子本身是最终结果时复制；52 个算子输入校验不再无条件复制，已由算子新建且可写的输出就地规范化非有限值，仍保证公开算子不修改输入且返回独立 `float64` 矩阵。Provider manifest 持久化解释器存储模式，便于正式 run 确认 mmap 零拷贝已生效；完整自动化回归为 207 项通过。累计量滚动核、Numba 和子表达式缓存仍未实施。
- 完成性能优化批次三：`ts_sum/mean/std/wma/slope/residual/zscore/cov/corr/beta/orth` 共 11 个算子改用股票轴 O(stock) 状态的滑动累计量，避免逐日期复制 `(window, stock)` 样本。方差和联合矩先按每只股票首个有限值平移再累计，以降低大数值水平下的消减误差；完整窗口、NaN 传播、`ddof=0`、零方差阈值、回归方向、当前残差和因果边界保持不变。随机缺失值、五种窗口和大数值水平均与旧逐窗口参考实现对照通过；600×1000、窗口 60 的合成单次基准中 `ts_mean/ts_std/ts_corr` 分别约为旧实现的 3.03/4.25/13.58 倍，完整自动化回归为 209 项通过。Provider 升级为 v4 并记录 `factor_gfn.rolling_moments.v1`，阻止跨数值内核恢复检查点；Numba 与子表达式缓存仍未实施，真实性能以新的 20 步 run 为准。
- 完成性能优化批次四：新增 `factor_gfn.numba_ts_loops.v1`。`ts_rank` 以完整窗口计数保持平均并列排名；`ts_max/min/argmax/argmin/position/range` 使用同时维护最大、最小值及最近并列位置的单调队列；`ts_ema` 使用缺失后重置的状态循环。真实 Provider 初始化时预热两类 JIT 签名，首次编译不计入第一个候选因子耗时，且不启用 Numba 并行。随机缺失、并列值、五种窗口、Pandas 参考和因果性对照均通过。600×1000、窗口 60 的预热后合成基准中，代表性的 `ts_max/ts_argmax/ts_position/ts_rank/ts_ema` 加速约为 2.08/3.91/3.53/203/110 倍；完整自动化回归为 210 项通过。Provider 升级为 v5，禁止旧数值内核检查点恢复。子表达式缓存仍未实施，真实性能以新的 20 步 run 为准。
- 完成性能优化批次五：新增 `factor_gfn.subexpression_lru.v1`。解释器递归计算表达式并按规范结构哈希复用跨候选的非叶、非根子树结果；叶子视图和完整候选不进入该缓存，继续分别由只读 mmap 借用与 Provider 完整表达式 LRU 负责。正式配置冻结为 512 MiB 可配置字节上限，按最近最少使用顺序逐项淘汰，缓存矩阵设为只读；每个候选在 `evaluations.jsonl` 中记录命中、未命中、淘汰、条目数、当前字节数与上限，逐步控制台与 TensorBoard 汇总命中数和命中率。子树缓存不写入检查点，恢复时为空，只影响计算耗时，不恢复或消耗随机状态。Provider 升级为 v6、正式搜索配置与统计 schema 升级为 v2，搜索配置将字节上限纳入不可变 run 清单；Notebook 已切换为全新 run、首段 20 步并清除历史输出。完整自动化回归为 214 项通过，实际命中率与端到端收益以新的 20 步正式 run 为准。
- 正式搜索的 `IndustryNeutralizationWarning` 属于已写入 `evaluations.jsonl` 的逐日期 fail-closed 审计，Runner 控制台默认不再重复逐条打印，只保留每步训练摘要；该显示层降噪不改变清洗结果、Reward、Provider 指纹或检查点兼容性。完整自动化回归为 215 项通过。

### 2026-08-11

- 冻结阶段 6 第一版联合筛选口径：训练期与验证期指标分别重算后一次性应用研报硬条件；研报字段 `test_ic/test_long_ir` 在项目中统一解释为 2019-2020 验证指标。训练期决定因子方向、`abs(train_ic)` 排序、`barra_ts_corr < 0.7` 和多头超额收益序列相关性 `< 0.7` 的贪心池结构；验证期对应相关性只作持久化稳定性诊断。该选择用于减少未披露假设、利用训练期 386 个而非验证期约 97 个调仓期提高相关估计稳定性，并避免在数千候选上额外利用验证集优化池结构。最终样本外继续保持锁定，只在 Alpha 池完全冻结后评价。
- 阶段 5 首个正式 run 在第 748 步完整落盘后由用户主动中断，以检查长期梯度裁剪与 `logZ` 校准。Trainer 新增最小充分诊断：利用 `loss = delta_mean² + delta_std²` 区分整体偏移与轨迹间方差，记录 `mean_log_pf/pb/reward` 定位 TB 项尺度，并同时记录分组裁剪前梯度、全局裁剪系数、Transformer 实际参数更新及 `logZ` 更新。正式搜索 Notebook 保持原 `max_steps=1000` 和配置指纹，从完整检查点自动续跑 20 步收集诊断；本次不改变 `max_norm=1`、`logZ` 学习率、Reward、搜索空间或数据口径。旧检查点兼容、确定性续跑和监控输出已纳入回归，完整自动化测试为 216 项通过；真实 20 步仍由用户手动执行。
- 第 749–768 步诊断确认 `delta_mean²` 平均占 Loss 的 87.80%，隐含最优 `logZ` 中位数为 39.32，按实际约 `0.001019/step` 的更新速度预计需约 37,800 步完成校准；`delta_std²` 仅占 12.20%。虽然联合裁剪系数中位数约 `0.000905`，Transformer 实际相对更新量均值仍为 `1.32e-4/step`，未显示参数更新停滞。因此新配置只将 `initial_log_z` 固定为 39.0、`logZ` 学习率提高为 `1e-2`，保留 `max_norm=1`、Transformer 学习率 `1e-4` 和 batch 8。旧 768 步 run 的候选继续保留作阶段 6 来源，但模型不跨配置续接；优化效果使用同 seed 的新 run 首20步独立验收。`initial_log_z` 已进入配置 schema、指纹和运行元数据，完整自动化测试为 218 项通过。
- 旧诊断 run `8778d49870c244a6996e31aa49f40e45` 不再续训，但其 5,905 个有效唯一结构中有 826 个已满足训练侧 `abs(train_ic) > 0.01`、`train_long_ir > 0.25`、`barra_ts_corr < 0.7` 的近似条件，因此候选及完整审计不可删除。为释放存储，仅删除 `checkpoints/` 下 77 个周期归档检查点，共 903,346,312 字节（861.50 MiB）；继续保留 `checkpoint_latest.pt`、评价、统计、运行元数据、状态、manifest 和 TensorBoard。该 run 的 manifest 已移除失效归档路径并写入检查点清理记录。
- 同 seed 的新参数 run `6c45bb08b68c4c5aacc036d54c7be7b3` 已完成 100 个逻辑步骤和 100 次优化器更新，共请求 944 个候选、获得 800 个有效训练样本，非法动作率为 0，未出现跳过更新或非有限 Loss。首 20 步平均 Loss 从旧配置同期约 1710.34 降至 245.65，说明 `initial_log_z=39.0` 与 `logZ` 学习率 `1e-2` 已消除主要的初始整体尺度错配；第 21–100 步 `delta_mean` 均值约 0.88，最后 20 步约 0.37，而 `delta_std` 分别约 14.85 和 14.22，TB Loss 当前主要由轨迹间方差而非 `logZ` 整体偏移构成。与此同时，第 21–100 步联合裁剪系数均值仍约 `0.00252`，平均 Loss 约 260.90，Reward 尚无足够长、稳定的上升证据。故本轮优化方向初步有效，但 100 步不足以证明收敛或确定最终超参数；当前仍处于训练参数诊断优化阶段，`initial_log_z`、两组学习率、`max_norm` 与 batch 等尚未最终冻结，也尚未据此启动多 seed 正式搜索。
- 基于上述 100 步结果启动第二轮同 seed、同有效样本量的独立参数诊断：保留 `initial_log_z=39.0`、`logZ` 学习率 `1e-2` 与模型学习率 `1e-4`，将旧的联合 `max_norm=1.0` 拆为模型和 `logZ` 各自 `max_norm=5.0`，并把有效 batch 从 8 提高到 16。新 run 只执行 50 步，因此与旧 run 的 `8×100=800` 个有效训练样本对齐；比较项固定为 `delta_std`、Loss、模型相对更新量、策略熵、Reward 分位数和拒绝率。配置 schema 升级为 `factor_gfn.gfn_config.v3`，两个裁剪阈值进入配置指纹；旧 run 只读保留且禁止跨配置续接。Trainer、控制台、TensorBoard、只读监控和历史导出均分别持久化两组裁剪系数，旧联合裁剪历史仍可兼容读取；完整自动化回归为 220 项通过。该第二轮参数仍属于诊断候选，不代表最终冻结。
- 完成继续训练前的纯 CPU 热点优化。冻结基线为 run `14af43fa57ff449a9e81cb17d3419e13` 的50步：平均每步99.08秒，其中因子解释48.58秒、Reward 25.41秒；子表达式缓存仅命中 `3/6425` 且淘汰6417次。因此保留 LRU 实现但把正式默认上限改为0，完整表达式 Reward LRU 不变。点时行业标签在386个固定截面上一次性编码；“截距＋完整行业哑变量”秩亏 OLS 改为数学等价的行业组均值投影，fail-closed、缺失行业排除、日期审计和调仓相位不变。`cs_rank/cs_quantile/cs_rank_gauss`、11个滑动累计统计及紧凑 Reward 的 RankIC/组合排序改用预热后的串行 Numba 内核；不启用并行、不使用 float32或近似排名。600×1000合成基准中 `cs_rank` 约为旧实现12.0倍，`ts_mean/ts_corr/ts_wma` 约为6.7/3.8/2.5倍；2000股票×386截面的 Reward 指标部分约从0.23秒降到0.13秒。旧 OLS、Pandas/SciPy、缺失值、并列值和稳定排序对照均作为硬门槛；公式、Token、结构哈希、Reward 公式、Barra口径、训练配置和筛选目标均未改变。Provider 升级为 v7并记录 `factor_gfn.numba_cpu_loops.v2`，旧 run 全部保留为阶段6来源，但其检查点不得跨 Provider 指纹续接；优化后必须新建 run。完整自动化回归为222项通过；正式搜索 Notebook 已清空旧输出并改为新 run 的10步端到端验收，不会自动续跑到1000步。
- CPU 优化后的同 seed 10 步验收 run `97182448b9024089878bc3aff91b3fcf` 正常完成：共请求175个候选并获得160个有效样本，平均每步墙钟约52.13秒、因子解释约24.11秒、Reward约3.56秒；相对优化前 batch 16 基线的平均99.08/48.58/25.41秒均明显下降。相同 seed 的前10步结构哈希序列一致，说明采样路径未被 CPU 优化改变；个别 Reward 存在约千分之一量级差异，已由旧实现对照测试覆盖，未改变定义或筛选目标。第二轮 batch 16 的50步没有降低 `delta_std`（约14.86，对照 batch 8 的100步约14.75），因此下一轮恢复 batch 8，以相同候选评价预算获得更多优化器更新；其他训练动态参数保持不变。Notebook 新建独立 run 并先运行300步，在训练期内比较第1–100步与第201–300步的 Reward、IC、训练侧近似硬筛选通过率和多样性；验证期与最终样本外继续保持不可见。该300步是延长诊断与首个正式候选积累窗口，不宣称已经收敛，也不以复现研报候选数量为验收条件。
- batch 8 延长诊断 run `378fd0dd73da4ac489510b1b551c2839` 已完成300步和300次真实更新，共获得2400个有效训练样本、2347个唯一有效结构。第1–100步对比第201–300步时，平均 TB Loss 从237.29降至177.50、`delta_std` 从14.20降至10.01，策略归一化熵只从0.971缓慢降至0.959且非法动作率保持0，说明训练动态在改善且未发生模式坍塌；但 Reward P90仅从0.01771升至0.01856、P99从0.04717降至0.04042，训练侧近似硬筛选通过率从15.33%变为14.82%，高分尾部没有明确富集。同期平均节点数从10.77升至13.21，15节点终止表达式占比从33.7%升至66.1%；后100步的1–3节点候选虽样本很少，但平均Reward约0.023，明显高于15节点候选约0.0078，因此节点饱和不能解释为长表达式已经获得更高质量。
- 上述长度偏移暴露出平坦142维 Token softmax 的动作数量偏置：动作空间包含6个叶子、106个一元和30个二元 Token，近似同 logit 时叶子组只能获得约`6/142=4.2%`总概率。阶段5下一轮新增显式配置 `token_policy_mode="arity_hierarchical"`：仍先选择规范槽位，再以可学习三分类 head 选择叶子/一元/二元组，最后在合法组内选择具体 Token；完整前向概率保持 `log P_F=log P(slot)+log P(group)+log P(token|group)`，实现对外仍合成为142维联合 Token 概率。三组 head 采用零权重、零偏置初始化，在三组都合法时初始组质量各为1/3；该比例只作为中性初值，之后按状态、槽位和预算由TB梯度学习，不作固定或下限约束。此改动只消除类别中 Token 数量对组总质量的机械挤压，不承诺最终短表达式占优，也不引入复杂度 Reward。
- 分组策略不改变部分AST DAG、142个 Token、`max_depth=6`、`max_nodes=15`、固定均匀 `P_B`、TB方程、logZ、原始Reward、行业中性化、训练区间或防泄露合同；组选择不是环境中的额外状态转移，不增加轨迹长度。配置 schema 和指纹必须区分旧 `flat` 与新 `arity_hierarchical`，旧300步 run 及候选永久保留为阶段6来源，但其平坦策略检查点不得恢复到分组策略模型。新 run 记录三组概率/实际动作率、组熵、终止节点P50/P90和15节点占比，先以相同batch 8运行100步，与旧run第1–100步的800个有效样本公平对照；若无NaN/Inf、非法动作或熵坍塌，15节点占比明显下降且Reward P90/P99及训练侧通过率未明显恶化，再沿同一分组run续至300步。验证集与最终样本外不参与本次结构对照。
- `arity_hierarchical` run `3bb6e7af03b648a1a1f4ba51123554df` 已完成100步和800个有效训练样本。相对同seed平坦策略前100步，其Reward P90/P99由0.01771/0.04717提高至0.03065/0.06942，`abs(train_ic)` P90由1.73%提高至2.90%，训练侧近似通过数由120提高至158；但唯一有效结构由783降至593，唯一结构节点P50由11升至14，15节点占比由33.7%升至48.2%。因此该策略提高了候选质量密度，却没有解决长表达式和重复短叶子的两极化；run停在100步并保留为阶段6来源，不自动续至300步。
- 下一轮只将策略概率分解升级为 `grammar_hierarchical`，其余训练、搜索空间、Reward、数据和筛选参数全部保持不变。联合策略严格分为：先选`Feature/UnaryFamily/BinaryFamily`；再在一元族条件选择`UnaryOp/TsUnaryOp/CsOp`、在二元族条件选择`BinaryOp/TsBinaryOp`；再选择具体Feature或Operator；仅对时序类别条件选择窗口`5/10/20/40/60`。初始化时三个元数组各1/3，一元三子类各1/3、二元两子类各1/2，各类别内算子及五个窗口等概率；这些仅是初始值，之后均由模型学习。该分解仍映射为原142维合法动作联合概率，不增加环境动作、AST节点或轨迹长度。
- 本轮明确不加入课程学习、复杂度Reward先验或高Reward重放。新模式必须使用独立配置指纹并拒绝加载`flat`或`arity_hierarchical`检查点；新增六类文法的条件概率/实际动作率、五个窗口条件概率/使用率、分层熵及时序算子占比诊断。新run仍以seed42、batch8运行100步，与`arity_hierarchical`同期800个有效样本比较Reward/IC尾部、高质量候选产出、唯一结构率、文法与窗口覆盖及节点分布；100步后先停下分析，验证集与最终样本外继续不可见。
- 完整文法分层工程实现已完成：配置升级为`factor_gfn.gfn_config.v5`，正式Stage 5预设唯一使用`grammar_hierarchical`；`flat`与`arity_hierarchical`退役为历史审计和通用回归兼容模式，真实搜索恢复入口明确拒绝旧策略run。模型将元数、六类文法、类别内Operator及条件Window概率精确合成为原142维合法动作，所有层级初始质量符合冻结合同，联合概率归一化、严格掩码、各概率头有限梯度及同策略确定性检查点续跑均有回归覆盖。六类与五窗口概率/动作率、各层熵及时序占比已写入训练统计、step metrics、TensorBoard、控制台和只读监控；Notebook已清空历史输出并默认新建100步完整文法run，对照旧元数分组run。完整自动化回归为236项通过；真实训练未由代码验收启动。
- `grammar_hierarchical` run `d521789d86de425794a9e871b42db586` 已由用户完成100步和800个有效训练样本。相对同seed的 `arity_hierarchical` 100步run，Reward P90/P99由0.03132/0.06942提高至0.04303/0.08137，最大Reward由0.08074提高至0.10094，`abs(train_ic)` P90由2.95%提高至3.56%，训练侧近似通过数由158提高至263，唯一有效结构由593提高至623；但平均每步wall由29.59秒升至34.55秒，后50步节点数和拒绝率继续上升，Reward尾部反而低于前50步。该结果支持保留完整文法分层，但不支持“继续收敛必然自动消除长表达式”的假设；长结构的终态组合数量优势列为附录B首要待优化问题。
- 完成完整文法策略的低风险采样热区优化：Trainer专用路径不再为每个AST动作分别执行策略熵、组概率、六类文法概率、Operator熵、Window概率及三项log概率的GPU到CPU同步；同一补采轮次的全部诊断改在GPU保持为紧凑向量，统一回传后在CPU逐项执行原有有限性、范围与归一化审计。`torch.multinomial`调用次数、顺序、输入联合概率、Reward、TB Loss和模型配置均未改变；相同seed下逐动作旧路径与批量路径的动作序列、终态表达式及全部诊断值已有回归对照。新增`sampling_seconds`、`reward_provider_seconds`、`training_update_seconds`以及TB前向、backward、optimizer的CUDA Event耗时，写入`step_metrics.jsonl`、TensorBoard、控制台和只读监控，但不写入确定性checkpoint历史。完整237项测试通过；真实CUDA加速幅度仍须由用户从现有100步检查点续跑少量步骤实测，代码验收未启动真实训练。
- 分组策略工程实现已完成：配置升级为`factor_gfn.gfn_config.v4`，通用`ModelConfig`默认保留`flat`以维持既有合成与旧接口语义，阶段5正式预设显式使用`arity_hierarchical`。策略网络新增零初始化三分类head，动态屏蔽无合法Token的组，并把组概率与组内概率合成为严格归一化的142维联合分布；greedy与随机采样仍直接依据该联合分布。Trainer、控制台、TensorBoard、历史导出和独立只读监控已接入组概率、实际动作率、组熵及终止节点诊断；分组检查点已验证模型、head、logZ、优化器及RNG的确定性续跑。旧v3平坦策略的通用检查点指纹仍可兼容读取，但其状态不能加载到分组模型；真实旧run仍按配置指纹拒绝迁移。正式搜索Notebook已清空旧输出，默认新建分组run、先运行100步，并只读比较`378fd0dd73da4ac489510b1b551c2839`的前100步训练期结果。Notebook静态验证及完整230项自动化测试通过；真实CUDA训练未由代理启动。

### 2026-08-12

- 完整文法分层 run 在100步后继续完成10步真实CUDA计时诊断，最终停在`current_step=110`、`optimizer_step=110`且状态`ready`，检查点、评价和逐步统计完整，无非法动作、跳过更新或非有限Loss。新增GPU批量诊断未产生可确认的整体提速：第101–110步平均wall约58.99秒，中位约57.44秒，采样/Reward Provider/训练更新分别约占66.4%/32.4%/1.2%；实现无严重冲突，作为旧scalar-logZ历史基线保留，不再围绕该微优化继续修改。
- （历史方案，已于2026-08-13被no-anchor正式合同取代）终态组合数量优势的解决方案确定为complexity-conditioned GFlowNet：外部按可达节点数层进行balanced discovery，层内继续严格按原`R_TB`学习；不修改原始Reward或Stage6。该日最初方案让exhaustive层退出normal discovery并只通过少量anchor训练policy；这一双路径设计后来被确认没有必要，相关实现与6/15、6/20结果只保留为历史工程证据，不再定义正式Stage 5。
- 完成第1步 exact-N Grammar 与可达性引擎：现有无条件`GrammarState`、AST `state_key`和终态结构哈希保持不变；新增外部`ExactNodeGrammarState(state, target_node_count)`分层，并将conditioned cache identity固定为`(state_key, target_node_count, search_space_fingerprint)`。可达性按实际Token arity、每个Hole绝对深度、剩余depth/node预算做精确节点数集合合成，从根状态独立解析`resolved_feasible_node_counts`与`resolved_infeasible_node_counts`，不假设层连续。conditioned mask保证每条legal successor至少存在严格`node_count=N`的completion；父状态重建、规范边复核和固定均匀`P_B`均携带同一个N。新增8项exact-N回归，并连同原无条件Grammar在完整245项自动化测试中通过；本步未接Transformer、Reward、Trajectory或Trainer，也未启动真实训练。
- 完成第2步 conditioned Trajectory、StateAdapter 与 Policy 接口：conditioned trajectory持久化`target_node_count`、`terminal_node_count`及由N和search-space生成的condition fingerprint，验证并重放时全程使用同一个`ExactNodeGrammarState` conditioned DAG，强制终态节点数等于N。StateAdapter新增`target_node_count/max_nodes`与`(target_node_count-current_node_count)/max_nodes`两个归一化标量；legacy `GrammarState`对应特征固定为零。Policy保留原`grammar_hierarchical`全部层级，只通过无bias的`2 -> d_model`小型线性投影使`P_F(a|s,N)`感知condition，不增加Token、环境动作、轨迹步骤或node-count embedding。StateAdapter schema升级为v2、GFN config schema升级为v6以阻止旧scalar-logZ run误恢复；TB normalizer、Reward、scheduler和Trainer训练逻辑均未修改。新增6项conditioned policy/trajectory回归，完整251项自动化测试通过；本步未启动真实训练。
- 完成第3步 balanced scheduler、same-N retry与per-N累计统计：对动态解析的`F`和配置给出的`E`生成`S=F-E`，使用独立RNG驱动的shuffled cycle逐slot分配N；完整保存当前permutation、cycle index、position与scheduler RNG state。`exact_node_retry_budget=k`严格定义为首次失败后最多额外k次same-N尝试，每个slot总尝试上限为`1+k`；retry不消费新scheduler N。任一slot耗尽即fail-closed跳过整个mixed-N optimizer update，batch中已有效slot仍计入`valid_count_by_N`但所有slot均不增加`successful_update_count_by_N`。累计持久化`requested_count_by_N`、`sampled_attempt_count_by_N`、`valid_count_by_N`、`successful_update_count_by_N`、`retry_exhausted_count_by_N`和`effective_update_rate_by_N`；低有效率阈值默认关闭，启用后只告警而不改变strata、permutation、RNG或配额。解析后的F/E/S进入配置manifest与指纹，scheduler及计数进入checkpoint；GFN config/checkpoint/Trainer schema分别升级为v7/v2/v2。新增8项调度与统计回归，完整259项自动化测试通过；per-N normalizer、calibration、exhaustive counting/anchor尚未实现，本步未启动真实训练。

- 完成第4步动态 conditional normalizer：scheduler关闭时继续使用旧全局scalar `log_z`；开启时按`max_nodes`动态创建`log_z_by_node_count` Parameter vector，并固定`N -> N-1`。同时创建同长度`exact_tb_log_z_by_node_count`与`exact_log_z_mask` buffers，mask初始全False，不预填N=1/N=2；严格`set_exact_log_z(N, value)`只允许为已声明E注册有限exact值，相同值重复注册幂等、冲突覆盖则拒绝。Mixed-N TB逐轨迹选择normalizer：E内层必须mask=True并使用fixed exact buffer，否则fail-closed；S内层使用对应learned scalar。Policy与normalizer使用独立Adam group、学习率和梯度裁剪；normalizer group禁用weight decay，并保护当前batch未激活scalar不受既有momentum推动。Checkpoint schema升级为v3，持久化normalizer模式、动态长度、exact values/mask并显式拒绝legacy scalar恢复到conditional vector。新增7项conditional normalizer回归，完整266项自动化测试通过；本步未实现exhaustive counting、exact Reward求和、calibration或anchor，也未启动真实训练。

- 完成第5步基础 conditional TB 合成闭环：在真实Grammar的N=1六个叶子终态上进行完整分布训练，constant Reward从故意打破均匀的policy初始化收敛回终态均匀分布；不等toy Reward的精确`Z_1=sum_x R(x)`验证learned `logZ_1`与`P(x|N=1)=R(x)/Z_1`共同收敛。Synthetic exact模式将同一精确值写入fixed buffer，只训练policy，exact buffer与对应Parameter均不移动。Mixed-N batch同时包含N=1单步和N=2两步trajectory，逐步从实际conditioned DAG重算`log_pf=log_p_slot+log_p_token`、`log_pb=-log(n_parents(child))`，再独立核对多步`sum_log_pf`、`sum_log_pb`、per-N normalizer选择及TB delta。Checkpoint恢复后下一批target N、动作/槽位序列、terminal state/structure、loss、selected logZ、delta及scheduler位置逐值一致。新增5项闭环回归；N=2不做636终态整体收敛，本步未接exhaustive registry、anchor、真实Reward或真实训练。

- 完成第 6 步 bounded canonical counting 与 exhaustive pool：新增显式预解析层，避免 `GFNConfig.manifest()` 或配置指纹计算隐式触发枚举。默认 `canonical_count_cap=10_000`，非人工 include 层在发现第 `cap+1` 个唯一 canonical terminal 时立即停止，并以 `count_relation=">"`、`canonical_count_exact=False` 和成本下界记录，禁止伪装成精确等于 cap；计数对象为 canonical expression 的结构哈希，不是 trajectory 或 action sequence。当前完整计数确认 N=1 为 6 个、深度分布 `{0: 6}`，N=2 为 636 个、深度分布 `{1: 636}`。RealReward 单候选保守估算基线为 0.75 秒且可配置；`planned_real_reward_budget_seconds=3600` 与 `max_budget_fraction=0.20` 共同给出全部 resolved exhaustive strata 的累计 720 秒上限，默认 N=1+N=2 预计 481.5 秒，因此自动 E=(1,2)。`explicit_exclude` 优先级最高，排除层直接保持 discovery-only，不为证明排除而执行无用 canonical counting；include/exclude 重叠直接报错。`explicit_include` 可绕过自动 count/cost 规则，但未获二次批准时只允许计数到累计剩余预算可容纳的候选数，发现下一唯一终态即可证明超预算并 fail-closed；只有显式 `approve_explicit_include_over_budget=True` 后才允许超预算 include 绕过 cap 完整计数。新增独立 SQLite authoritative registry，与 discovery `evaluations.jsonl` 隔离；每个候选持久化固定 source、N、depth、formula、prefix token、structural hash、provider/context fingerprints、Reward details、valid/invalid、rejection reason 与 target mass。结构哈希为主键，已完成评价幂等恢复且禁止冲突覆盖；invalid 候选保留审计并强制 target mass=0；枚举覆盖与 Reward 评价覆盖分别记录，只有二者均完整才报告 coverage complete。本步只用 synthetic 评价验证断点续跑和覆盖状态，未执行真实 RealReward 全量评价、未求 exact Z、未接 anchor 或训练。

- 完成第 7 步 Exact Z 与 training-only calibration：exhaustive registry schema 升级为 v2，在全量评价覆盖完成后逐候选核对 `raw_reward`、`reward=max(raw_reward,reward_floor)`、`log_reward=log(reward)` 与 registry target mass，再分别以稳定求和计算 `exact_raw_reward_log_mass` 和 `exact_tb_log_z`。raw 总质量为零时持久化 `exact_raw_reward_log_mass=None` 与 `raw_reward_mass_status=zero_mass`，TB floor 后质量仍可形成 finite exact Z；valid candidate 数为零时持久化 failed 状态并禁止 exact Z/anchor。Exact aggregate 绑定 provider/context/reward-floor 与候选级聚合指纹，完成后禁止冲突覆盖。TB fixed buffer 改为 float64，learned per-N Parameter 保持 float32；exact strata 只接受 audited `ExactMassResult`，buffer 与审计值必须逐值一致。新增配置化 `minimum_valid_calibration_samples=64` 与 `maximum_requested_calibration_slots_per_N=128` 诊断默认值，短 smoke 可降低；calibration 使用覆盖全部 feasible N 的独立 balanced scheduler/RNG，不消费 discovery scheduler，不执行 optimizer update，只在 `optimizer_step==0` 且 optimizer state 为空时运行。每个 slot 使用同 N 并允许既有 retry budget，持久化 requested/valid/sampled-attempts、全部 implied-logZ 观测、median、logmeanexp、P10/P25/P75/P90、IQR；non-exhaustive scalar 只用 median 初始化，exhaustive strata 不初始化 learned scalar而记录 median/logmeanexp 相对 exact TB logZ 的差值。任一 N 达到最大请求预算仍不足最小有效样本即 fail-closed；calibration 未完成时 Trainer 禁止训练。Provider 必须显式声明 `data_scope=training_only`、`validation_oos_loaded=False` 与 context fingerprint，缺失即拒绝；真实 Reward context 本身不暴露 validation/OOS。GFN config、Trainer、checkpoint、RealRewardProvider schema 分别升级为 v8/v4/v4/v8，并保留 pre-calibration scalar 配置与 provider 指纹的只读 checkpoint 兼容。本步仅执行 synthetic exact/calibration/恢复测试，未运行真实 N=1/N=2 exhaustive RealReward、真实 calibration、anchor 或真实训练。

- （历史实现，已退出正式active path）第8步曾实现Exhaustive TB anchor并验证N=1/2 unique-trajectory抽样、固定exact Z、独立scheduler/RNG、共享Adam计数和checkpoint恢复合同。该机制用于当时“E不参加normal discovery”的方案；2026-08-13改为`D=F`后不再需要。旧模块、测试、run和checkpoint仅作历史审计；这些anchor字段在新的no-anchor formal config/schema/fingerprint/metadata/checkpoint中必须删除。
- （历史实现）第9步曾将exact Z、calibration、discovery、anchor和deterministic checkpoint串成完整synthetic integration；除anchor专属部分外，exact/learned mixed-N TB、same-N retry、非连续feasible strata、动态`max_nodes/max_depth`和schema拒绝等测试资产继续作为no-anchor重构的回归基础。
- 第 10 步新增独立手动入口 `notebooks/run_complexity_conditioned_smoke_6_15.ipynb`，默认 `RUN_SMOKE=False` 并强制CUDA，不复用或改写旧scalar-logZ正式搜索Notebook。该入口使用真实training-only 6/15上下文，显式E=(1,2)、S=(3..15)，以可恢复SQLite逐条完成N=1/2共642个canonical terminal的RealReward评价和exact TB logZ；随后用smoke专用1/2 calibration门槛完成全部feasible N初始化，只执行两次batch-8 discovery并按frequency=1插入两次batch-8 anchor。第1次更新后保存checkpoint，第2次连续运行与恢复运行在完整TrainingStats、policy参数、discovery/anchor scheduler及RNG上严格对照。Notebook输出并落盘resolved F/E/S、exact coverage/logZ、calibration、逐N requested/valid/successful/effective/retry、逐trajectory delta/Reward/depth、anchor loss、分阶段wall time、GPU显存与checkpoint acceptance；默认结果目录为`runs/complexity_smoke_6_15/manual_smoke_6_15_seed42/`。代码验收只完成Notebook语法、静态装配与自动化回归，真实评价和CUDA更新必须由用户人工打开安全锁运行；通过后立即停止6/15，不增加训练步数。
- 第10步首次人工预检在真实评价前发现Notebook沿用了旧版`provider_manifest['industry_neutralization'] is True`布尔断言，而当前Provider manifest已将该字段升级为包含`enabled/policy_schema/...`的结构化合同；生产Provider与`RewardConfig.candidate_industry_neutralization=True`均未回退。Smoke入口与历史`validate_stage4_real_reward.ipynb`已统一改为同时检查嵌套`enabled`和权威RewardConfig；新增跨全部活跃Notebook的静态扫描，禁止旧布尔访问模式再次进入入口。该问题未开始exhaustive、calibration或optimizer update，无需清理运行产物。
- 第10步真实6/15 smoke已由用户完成并通过：N=1的6个与N=2的636个canonical terminal均100%覆盖，分别有6/629个有效候选，固定`exact_tb_log_z=-1.7669613347/2.8494631164`；全部15个可达N完成smoke calibration，两次discovery与两次anchor更新均成功，所有discovery N最终有效更新率为1且无retry exhausted，连续与checkpoint恢复的第二步严格一致。行业中性化开启、Provider为training-only且未加载validation/OOS；总wall约611秒、N=1/2 exhaustive RealReward约345秒、CUDA峰值分配显存约110 MiB。该结果只确认真实主链兼容，不冻结单样本calibration质量或正式训练动态，6/15不再增加训练步数。
- 第11步建立独立`max_depth=6/max_nodes=20`真实conditional diagnostic；此前暂定8/24方案作废。`max_nodes=20`仅是本轮人为配置化complexity upper bound，明确禁止用`node_count==max_nodes`占比、Reward或IC自动判断扩到24/30；`max_depth=6`是可检验起点，后续只有本run输出`consider_expansion`且经人工确认后，才能新建独立7/20 run，再视证据新建8/20，禁止原run运行中改边界。depth统计动态覆盖`0..max_depth`及每个resolved discovery N，重点比较`max_depth-2/-1/max_depth`。候选主口径为显式`source=discovery`且按structural hash去重，invalid进入valid-rate与耗时总体，Reward/abs(train_ic)只使用valid finite样本，耗时为首次实际评价元数据中的`factor_seconds+reward_seconds`；calibration、exhaustive、anchor及非training-only来源均排除。保守默认门槛为总unique 500、三个重点depth各100个unique且各100个finite Reward/IC；max-depth占比10%为consider门槛、3%为低碰撞门槛，valid-rate非恶化/明确下降阈值为5/10个百分点，质量相对非恶化/明确下降阈值为10%/15%。模块只输出`consider_expansion/no_expansion_evidence/insufficient_evidence`建议，强制advisory-only；固定输出为summary JSON、depth metrics CSV与candidate audit CSV。
- 第11步工程入口新增通用`ConditionalDiagnosticRunner`与`DiagnosticRewardProvider`审计包装，不改变Reward返回值、TB方程、scheduler或共享Adam。每次RealReward调用必须显式声明`exhaustive_full_evaluation/calibration/discovery`来源并耐中断写入JSONL；depth统计只读取normal discovery记录。Trainer新增只读的最近discovery逐trajectory TB诊断，记录per-N selected logZ、`sum_log_pf/sum_log_pb`与delta，不进入优化状态。独立run context同时绑定search-space、完整config、model、training、provider和data-context指纹，拒绝6/15及legacy scalar-logZ恢复；checkpoint可恢复discovery/anchor scheduler与RNG。本次Notebook工作参数为batch8、same-N retry2、calibration 16/32、anchor frequency8、最少/最多64/128次成功discovery update和160 logical-batch硬上限，全部明确不是正式Stage 5冻结值。64次成功update只决定最早检查时点，样本充分性必须读取实际去重`unique_discovery_candidate_count`；即使update数达到下限，只要unique总数或重点depth样本不足，仍输出`insufficient_evidence`直至满足门槛或硬停止。该真实6/20 CUDA/Reward任务现已由用户启动并继续按原配置运行；2026-08-13的no-anchor决策不回写、不重启也不中断本run。
- 第11步首次人工运行在配置单元暴露出不必要的重复计数成本：通用planner会对6/20中N=3…20逐层枚举到`canonical_count_cap+1`，20分钟仍未完成，用户已安全中断且尚未开始exhaustive RealReward、exact Z、calibration或训练。基于第6步及真实6/15 smoke已确认的N=1六个、N=2六百三十六个和N=3超过自动阈值，本次6/20 Notebook改为显式复用`E=(1,2)`：只重新复核N=1/2的canonical count，将当前search space中其余可达N全部以`explicit_exclude`保持为discovery，不再重复计数N=3…20。该选择仅属于本次diagnostic配置并进入配置/run指纹，不改变通用planner、未来独立实验或“代码不得用N<=2自动推断E”的底层合同。
- 第二次人工运行已完成N=1/2共642条exhaustive评价及N=1…20 calibration，但在最终表格展示时错误地对`calibration_report()`已经返回的dict再次调用`dataclasses.asdict()`，40多分钟计算结果本身未失败。Notebook改为直接由dict构造DataFrame和summary；calibration循环每20个slot输出逐N requested/valid/sampled-attempts进度，并在calibration完成、runner context建立后立即保存checkpoint，避免展示错误导致重复长任务。当前审计已有486次calibration Reward调用、355个有效结果；若原Kernel仍存活，可先手工保存该完成状态再继续训练单元。
- 同一审计表明行业中性化跳过不是可全局预剔除的固定日期集合：1128次评价中仅122个候选出现跳过、共有60种日期组合，1006个候选完全没有跳过；失败主要来自复杂表达式自身在某截面的finite factor覆盖不足，因候选和lookback而变化。故继续按候选逐截面fail-closed排除，不改变冻结全局调仓日历；全局删除这些日期会改变所有候选Reward、N=1/2 exact Z及provider/context指纹，当前不执行。

### 2026-08-13

- Complexity-conditioned Stage 5正式设计明确简化为no-anchor单路径：`F`表示exact-reachable strata，`D=F`表示全部normal discovery strata，`E`只表示exhaustive evaluated且使用fixed exact normalizer的strata，`L=F-E`使用learned per-N normalizer。N=1/2不再退出discovery，而是与N=3...20共同接受balanced normal discovery、same-N retry和policy gradient；唯一差别是N=1/2从float64 exact buffer读取固定`exact_tb_log_z`且对应Parameter无梯度，N>=3使用learned scalar。Exhaustive知识只用于normalizer准确性与Reward缓存，不再创造anchor training pathway。
- 正式active path将删除anchor frequency/batch、scheduler/cycle/RNG、optimizer step、loss、checkpoint state和专属optimizer逻辑；新的architecture schema为`factor_gfn.complexity_conditioned_no_anchor.v1`，必须拒绝所有带anchor state的旧checkpoint。历史anchor模块、synthetic测试、6/15 smoke与当前6/20 run可只读保留用于审计，但不得被formal runner调用或恢复。
- 旧6/20 diagnostic已完成：training-only审计包含1159个unique normal-discovery candidate，depth=4/5/6的unique与finite质量样本均达到预设门槛；自动建议为`no_expansion_evidence`，直接原因是depth=6相对depth=5的valid rate下降约10.50个百分点。虽然depth=6占比约57.03%且Reward/abs(IC)上尾仍有改善，但按预先冻结的保守规则，有效率达到“明确下降”阈值，因此不把撞边界单独解释为扩深证据。用户据此确认冻结`max_depth=6/max_nodes=20`，不建立7/20。旧run的模型、optimizer、scheduler、anchor和checkpoint均不进入正式训练；其depth结论、严格等价的N=1/2 registry/exact Z和经语义核验的N=3...20 implied-logZ median可分别按当前合同复用。
- （已被后续决策取代）此前要求新no-anchor run对全部`L`重新执行64/128 calibration；最终口径改为：严格验证并复用旧6/20 training-only median作为新Parameter初始化常数，不继承任何旧训练状态。Step 12只在每N的valid trajectory与successful gradient exposure均充分时判断初始化健康；明显失配的N才在全新training state中执行64/128 targeted recalibration，其他N继续使用已验证历史median，禁止训练中途重置logZ。
- N=1/2 exhaustive registry复用不得仅依赖全局search-space fingerprint。每个目标run初始化时只执行一次逐N canonical structural-hash全集重枚举与Grammar/operator/interpreter、provider/data context、Reward config、reward floor核验；全部一致才复用stored Reward/exact Z，否则fresh evaluation。初始化证明通过后，normal discovery逐候选只做structural-hash lookup，不重复枚举全集。
- `6/20`边界确认后已执行极短no-anchor integration smoke，只验证`D=F`、N=1/2 normal discovery与registry cache、E/L normalizer选择、same-N retry、mixed-N梯度、所有N的policy exposure、无anchor state、旧checkpoint拒绝和确定性恢复，不比较Reward/IC质量。下一步第12步只做训练动态健康检查；无明确异常时优先沿用`policy_lr=1e-4`、`learned_logz_lr=1e-2`、两组`max_norm=5`、`batch_size=8`及budget=2，不做多维超参数搜索。
- 2026-08-13已完成正式no-anchor checkpoint切换：唯一可写schema为`factor_gfn.checkpoint.no_anchor.v1`，只允许`NoAnchorGFNConfig`对应Trainer写入；payload不含任何anchor字段，并持久化F/D/E/L调度统计、L-only calibration、动态exact/learned normalizer和逐E registry equivalence proof。No-anchor Trainer严格拒绝旧v1-v5、scalar normalizer及任意嵌套anchor字段，不做隐式迁移。旧Stage 4非conditioned scalar Trainer仅保留历史checkpoint只读加载；兼容缺少condition projection的旧模型/optimizer布局且可继续运行，但所有`save_checkpoint()`均fail-closed，因此active代码已不能再生成旧schema。旧anchor模块、导出和专属测试已退出active path，6/15与旧6/20 Notebook标记为不可恢复的历史诊断归档，已有run文件不删除。
- 新增`notebooks/run_no_anchor_integration_smoke_6_20.ipynb`与`notebooks/run_step12_no_anchor_training_health_6_20.ipynb`。前者已由Codex用项目解释器逐格执行通过：N=1/2分别完整覆盖6/636个canonical terminal，E/L、registry一次性证明、exact buffer冻结、learned更新、same-N retry、新schema确定性恢复和旧schema拒绝全部通过；该synthetic smoke仅验证工程闭环，不代表真实implied-logZ质量。Step 12 Notebook保持`RUN_REAL_STEP12=False`，只允许用户手动启动真实training-only CUDA 6/20新run；初始化阶段以只读registry及严格来源/语义证明导入N=1/2 exact Z和N=3...20历史median，不重复RealReward或全量calibration。训练每logical batch输出进度，并记录每N的pre-update/early/late delta、initial/current/net-change logZ、valid trajectory与successful gradient exposure、两组梯度/裁剪、retry、吞吐和显存；32 batches后逐N输出`usable/review_targeted_recalibration/insufficient_evidence/fixed_exact_diagnostic`，不自动重校准或修改配置。确定性恢复另行验证且不计入冻结的32-batch健康样本。Notebook所有代码格前均有用途与预计耗时说明，长provider加载、单batch与恢复验证每20秒heartbeat。真实Step 12仍未运行。
- 2026-08-13最终文档同步后运行完整自动化回归。首次335项运行发现旧conditional calibration收尾会误把exhaustive层的diagnostic median写入learned scalar；现已改为exact strata仍保留implied-logZ偏差诊断，但只对non-exact strata调用learned初始化接口。相关exact/calibration、no-anchor和training-health 28项回归通过，随后完整335项全部通过。该自动化结果不等价于真实Step 12已运行；真实training-only CUDA健康检查仍由用户手动启动。
- 真实Step 12已由用户完成：32个logical batch中24次成功optimizer update、8次按合同fail-closed跳过，所有N均达到至少7条valid trajectory与7次successful gradient exposure，成功更新无NaN/Inf或非法动作，N=1/2 exact Z固定，checkpoint continuation逐值确定。自动初始化健康检查将N=3...16、19、20判为`usable`，将N=17/18判为`review_targeted_recalibration`；后两层分别出现initial delta mean约-9.24/+5.88且late delta mean约-3.01/-4.60。Retry budget=2的25% logical-batch跳过率偏高，下一轮直接使用3并验证，不进行2/3/4网格。Policy在24/24成功更新中均被`max_norm=5`裁剪，median clip coefficient约0.019；这只证明当前处于强裁剪regime，不把Adam实际参数更新简单解释为缩小到2%，后续仅做5与20的极短同源初始化对照。
- Targeted recalibration实现限定为N=17/18、training-only、policy冻结、optimizer step=0，固定`minimum_valid=64`、`maximum_requested=128`、`comparison_window=16`、median/IQR absolute tolerance=0.25/0.50和same-N retry=3。新增独立calibration-only progress schema，逐slot原子保存独立scheduler、观测和RNG，不保存或恢复model/optimizer/discovery state；完成结果写入严格指纹化JSON artifact。新Trainer必须先导入除17/18外的verified historical medians，再在fresh state中核对search/model/sampling/Reward/provider/context/registry equivalence、历史provenance和初始policy state fingerprint后导入artifact；只初始化17/18，禁止中途覆盖。该targeted provenance随no-anchor checkpoint持久化。
- 新增手动入口`notebooks/run_targeted_logz_calibration_n17_n18_6_20.ipynb`，默认安全锁关闭且支持`new/resume`。真实数据加载、registry proof和每个calibration slot均有20秒heartbeat；每20个累计slot输出逐N requested/valid/sampled-attempts、稳定性、elapsed与upper-bound ETA，每个slot保存`latest_targeted_calibration.pt`。完成后输出targeted JSON、per-N CSV、summary和fresh-Trainer hybrid import verification；该Notebook不得执行任何训练，用户交回结果确认后才进入`policy_max_norm=5 vs 20`短对照。
- N=17/18 targeted calibration在已有progress checkpoint中分别保留124/126条有效training-only implied-logZ观测。全样本median分别为46.30188640483371/46.427256389816854，IQR分别为10.402814921422546/12.405252585713164；严格的两个不重叠16-valid窗口稳定性检查未通过。经人工明确批准，不再bootstrap或扩样本；结果以`initialization_status=high_variance_engineering_estimate`、`strict_stability_check=failed`登记，只把全样本median作为工程初始化常数。严格stable artifact路径仍保持fail-closed，二者不得混称。该复用不恢复任何model/optimizer/scheduler/checkpoint训练状态，并承认异常strata的大TB residual可通过共享policy梯度影响其他N。
- 下一步只做`policy_max_norm=5`与`20`的同源极短对照，same-N retry固定为3；batch=8、policy LR=1e-4、learned logZ LR=1e-2、logZ max-norm=5保持相同。每组目标16次成功optimizer update、最多32个logical batch；A/B必须具有相同初始policy、scheduler、exact/historical/targeted logZ和seed，并用各自checkpoint恢复RNG。只比较pre-clip gradient、clip coefficient、实际参数更新、TB RMS/loss、policy entropy、per-N delta、skip/吞吐/显存与非有限值，不使用短期Reward/IC选参数，不自动产生胜者。手动入口为`notebooks/run_policy_clip_comparison_5_vs_20_6_20.ipynb`，默认安全锁关闭，所有长batch具有20秒heartbeat和逐batch checkpoint。
- `policy_max_norm=5/20`真实同源对照已完成：两组均为17个logical batch、16次成功update和1次skip，采样的128条trajectory结构身份完全相同且TB delta最大绝对差约1.53e-5；median pre-clip norm均约213.97，clip coefficient分别约0.02383/0.09533，但median实际policy参数更新范数分别约0.032632/0.032634，TB、loss、entropy和per-N delta近乎逐值一致。该结果符合Adam对统一梯度尺度近似不变的预期，不能据“裁剪系数更大”声称20更优；按预先约定保留更保守的`policy_max_norm=5`，不再扩展裁剪阈值搜索。Retry=3下skip由旧Step 12的25%降至1/17约5.9%，当前候选固定为3并在最终health确认中复核。
- 进入正式Stage 5前新增唯一一次独立最终health confirmation：`max_depth=6/max_nodes=20`、batch=8、policy/logZ LR=1e-4/1e-2、policy/logZ max-norm=5/5、retry=3，使用同一verified exact/historical初始化以及N=17/18高方差工程初值，从全新model/optimizer/scheduler运行32个logical batch。逐N必须输出initialization/pre-update、early、late TB delta、initial/current/net-change logZ、valid trajectory与successful gradient exposure；同时输出skip/retry、梯度/裁剪、实际更新、entropy、吞吐、显存和非有限值，并做独立checkpoint确定性恢复。证据不足或异常仅输出`insufficient_evidence/review_required`，不得中途重置logZ、自动改参数或创建正式run。手动入口为`notebooks/run_final_no_anchor_health_confirmation_6_20.ipynb`，默认安全锁关闭，长阶段逐batch输出且每20秒heartbeat。
- 最终health confirmation已完成并获准进入正式Stage 5：32个logical batch中31次成功update、1次fail-closed skip（3.125%），所有N均有至少11条valid trajectory及11次successful gradient exposure，无证据不足层、无自动targeted-recalibration层、无NaN/Inf，checkpoint恢复逐值确定。全局delta mean均值约-0.138；TB RMS首4/末4均值约4.36/4.65，entropy稳定，median实际policy参数更新范数约0.02777且随训练下降。N=13/15/16/19/20后期delta仍有较大方差或偏移，但learned logZ净移动绝对值均小于0.07，判为尚未收敛而非结构性错误；正式1000步必须持续监控，不因这些短期尾部值重新调参。
- 正式6/20 no-anchor seed42合同现已冻结：`max_steps=1000`、batch=8、policy/logZ LR=1e-4/1e-2、policy/logZ max-norm=5/5、retry=3、Adam betas=0.9/0.999、eps=1e-8、weight decay=0、sampling multiplier=10、deterministic algorithms启用，N=17/18继续使用已登记的高方差工程初值。冻结配置指纹为`b6453816d90f89609e506e02d6c8c0a9d3eda37571ea64079ecd91c9ad341789`；运行控制的`TARGET_STEP`不进入配置，首次只到10，检查后从同一新schema checkpoint恢复至1000。所有complexity diagnostic、health、A/B、registry与checkpoint保持只读且不作为正式训练状态恢复来源。
- 正式入口为`notebooks/run_stage5_no_anchor_formal_6_20.ipynb`，新run schema为`factor_gfn.no_anchor_real_search.v1`，旧scalar/anchor/legacy real-search schema均不得进入。Notebook默认安全锁关闭、首次`MODE=new/TARGET_STEP=10`；10步通过后显式填写同一run绝对目录并设`MODE=resume/TARGET_STEP=1000`。控制台仿照既有real candidate search但压缩为每step一行，记录step/target、optimizer、skip、valid/request、retry、loss、Reward、TB、裁剪、实际更新、entropy、learned-logZ范围、wall time、动态ETA与显存；每步原子保存latest checkpoint、每10步保存归档checkpoint，validation/OOS保持未加载。
- 正式run `c3a1c2747cbb41dbbb3f8f23e6ddddcb` 的首10步检查已通过并获准从同一checkpoint续跑到1000：run状态为`ready`、current/optimizer step为10/9、无active step或last error；100次Reward请求得到71条valid、100个unique structural hash，唯一skip发生在step 2的N=17 retry耗尽，非法动作率始终为0。9次成功update的loss/TB/梯度/参数更新均有限，delta mean均值约-0.202，normalized entropy均值约0.897，实际policy相对更新从约6.73e-4下降到2.21e-4；显存峰值约161.4 MiB。`checkpoint_latest.pt`和step-10归档均存在，正式config/provider/context/初始化来源指纹一致。Notebook现已清空执行输出并固定为`MODE=resume`、`TARGET_STEP=1000`及该run绝对目录；禁止新建替代run或改配置。
- 2026-08-14完成入口清理：根目录Notebook只保留数据下载/准备、旧输出格式参考和唯一正式no-anchor训练入口；validation、synthetic/real smoke及Barra手工Notebook删除，参数诊断Notebook统一移入`notebooks/archive/diagnostics/`并标记为只读历史证据。该清理不删除任何`runs/`、checkpoint、exhaustive registry、logZ初始化来源、候选表达式或阶段6可导入记录，也不移除no-anchor依赖的legacy只读/拒绝代码路径。

## 附录A：暂缓的 Barra 风格因子参考实现

本附录只记录后续版本的候选方案，不代表当前数据需求或第一版实现任务。下列因子不进入第一版 Barra 惩罚集合，不在当前阶段下载相应财务或分红数据。若后续启用，必须首先确认财务报告披露日、可得日、除权除息日及历史成分映射，避免将事后数据回填到历史截面。

| 因子 | 构造公式 | 数据源及字段 | 备注 |
|:---|:---|:---|:---|
| Value | `net_asset_ps / close_raw` | `stock.finance.get_core_index()` → `net_asset_ps`；不复权收盘价 `close_raw` | BP 比，需注意报告期和实际披露日滞后 |
| Earnings Yield | `basic_eps / close_raw` | `stock.finance.get_core_index()` → `basic_eps`；不复权收盘价 `close_raw` | EP 比，与 Value 高度相关，需注意报告期和实际披露日滞后 |
| Growth | `total_rev_yoy_gr` 或 `net_profit_yoy_gr` | `stock.finance.get_core_index()` 直接取值 | 营收或利润同比增长率；必须按当时可得报告使用 |
| Leverage | `asset_liab_ratio` | `stock.finance.get_core_index()` 直接取值 | 资产负债率；必须按当时可得报告使用 |
| Profitability | `roe_wtd` | `stock.finance.get_core_index()` 直接取值 | 加权净资产收益率；必须按当时可得报告使用 |
| Dividend Yield | 过去 12 个月累计每股股利 / `close_raw` | `stock.market.get_dividend()`，解析 `dividend_plan` 并按 `ex_dividend_date` 筛选 | 需解析方案字符串，严格按除权除息日截断 |

## 附录B：设计取舍说明

### 首要待优化项：终态数量导致的长结构总质量优势

这是当前 GFlowNet 搜索最重要的底层问题；解决方案已收敛为5.3所述的no-anchor complexity-conditioned GFlowNet。理想收敛时，单个终态表达式 `x` 的采样概率满足 `P(x) ∝ R(x)`；然而某一长度层的总采样概率取决于该层所有终态 Reward 的总和：

`P(node_count = n) ∝ Σ_{x: node_count(x)=n} R(x)`。

因此，即使单个长表达式的平均 Reward 低于短表达式，只要长表达式的合法组合数量大得多，该长度层仍可能获得更大的总概率质量。完整文法分层消除了“某类 Token 数量更多，初始化时自然占据更多动作概率”的局部偏置，但不会自动抵消完整终态空间随节点数增长产生的组合数量优势。这不是实现错误，也不能仅凭“训练最终偏向高 Reward”推断长表达式比例必然下降。

当前100步完整文法分层run已经出现直接证据：前50步到后50步，有效候选平均节点数由6.74升至11.16，Reward P90由0.05416降至0.03520；两个半程内节点数与Reward的相关性分别约为-0.32和-0.35。该现象仍可能包含早期训练和logZ校准影响，尚不足以确定最终稳态，但已足以要求独立处理复杂度层覆盖，而不能假设继续训练会自动消失。

已确认方法先解析可达节点数层，并让全部`D=F`通过同一个balanced shuffled cycle参加normal discovery和exact-N conditional TB；小型canonical终态空间额外完成100% exhaustive coverage并固定精确TB partition，但不再创建anchor训练路径。后续必须按N记录候选数量、唯一结构数、Reward/IC尾部、训练侧通过密度、TB、有效更新率和depth分布。任何实现都不得静默修改原始Reward、阶段6筛选指标或验证/OOS隔离合同；复杂度Reward、课程学习、重放和长度归一化仍不在本轮方法中启用。

### Reward v2：待优化方向

在当前研报Reward复现完成后，考虑增加一个更对称、可解释的多目标Reward作为独立实验，不覆盖当前基线。

第一版候选采用**Weighted Sum**，暂不使用pilot数据自动标准化；各指标的质量映射$q(\cdot)$根据金融含义、风险偏好和明确阈值预先设定，并在单个run内保持固定：

\[
Q =
w_{IC}q_{IC}
+w_{IS}q_{ICStability}
+w_{RET}q_{NetLongMean}
+w_{RS}q_{LongStability},
\qquad
\sum_i w_i=1.
\]

最终保留现有Barra风险折扣：

\[
\boxed{
R = Q\left(1-\mu_B\cdot BarraCorr\right)
}
\]

四个质量维度分别表示：

- `q_IC`：平均RankIC的预测强度；
- `q_ICStability`：IC时间序列稳定性，具体指标待确认，可考虑`std(IC_t)`的反向质量、分期一致性等；
- `q_NetLongMean`：扣除交易成本后的多头超额收益均值；
- `q_LongStability`：净多头超额收益的时间稳定性，第一候选可用`std(net_long_excess_t)`的反向质量，也可后续比较downside deviation、正收益胜率等定义；
- `BarraCorr`：继续使用现有`max_k |corr_k|`，通过$1-\mu_B BarraCorr$独立惩罚传统风险因子暴露。

这里不再直接使用`ICIR`或`LongIR`作为Reward项，目的是把“均值水平”和“时间稳定性”拆开，减少同一均值信息在IR中重复进入Reward。

权重$w_i$、各$q(\cdot)$映射阈值/尺度以及$\mu_B$均作为待研究参数，优先依据经济意义和风险偏好设定少量候选配置，再仅使用训练期做敏感性分析，不通过validation/OOS精细调参。

同时保留**weighted geometric**作为后续独立对照：

\[
R_{\rm geo} =
q_{IC}^{w_{IC}}
q_{ICStability}^{w_{IS}}
q_{NetLongMean}^{w_{RET}}
q_{LongStability}^{w_{RS}}
\left(1-\mu_B BarraCorr\right).
\]

该对照用于比较Weighted Sum的“目标之间允许补偿”和geometric的“单项短板惩罚更强”哪一种更适合Alpha discovery。

暂不纳入第一版的内容包括Alpha-pool novelty、动态regime、learned weights和在线动态normalization；等基础因子池建立后再研究。

### Barra 相关性惩罚为何采用最大绝对相关性

当前版本保持以下已确认方案不变：

```text
barra_ts_corr = max_k |corr_k|
```

即分别计算候选因子 Long-Short 收益序列与五个 Barra 风格 Long-Short 收益序列的 Pearson 时间序列相关性，并取最大绝对值作为惩罚项。

选择最大绝对相关性而不是多元回归 R²，首先是为了获得更稳定的训练信号。训练早期候选表达式质量不稳定，其 Long-Short 收益序列通常噪声较大；多元回归同时估计多个风格暴露，容易受到采样误差、解释变量共线性和短样本波动的共同影响，使 R² 及 reward 剧烈震荡，进而干扰 GFlowNet 前向策略的稳定学习。

`max_k |corr_k|` 只关注当前最强、也最需要防范的单一风格暴露。与多变量联合拟合相比，其信号结构更简单、数值更鲁棒，reward 随候选因子变化通常更平滑，有利于策略网络收敛。因此，当前方案是在训练效率、稳定性和风险惩罚效果之间采取的工程平衡，并非等待被替换的临时缺陷。

多元回归 R² 可以在未来作为对照实验，用于研究候选因子对多种风格的联合暴露，但不预设其优于当前方案，也不列为当前待优化任务。若开展该实验，仍须单独规定共同有效期对齐、共线性处理、截距、正则化、样本内外估计和 R² 数值稳定规则。

### 表达式结构唯一率的边界与待优化方向

`expression_unique_rate` 只统计 batch 内不同规范结构哈希的比例。它能合并完全相同的表达式和四个已确认交换算子的参数换序，但不能识别 `sub(close, close)` 这类常数退化、数值恒等、单调变换后的等排序因子，也不能发现结构不同但因子值高度相关的候选。

真实训练阶段应将结构唯一率与以下指标联合观察：累计表达式访问频次、固定校验样本上的因子值指纹、batch 内两两 Spearman 绝对相关均值/中位数以及常数因子比例。优先在评价/监控层发现数值等价，暂不扩大文法层的代数化简规则，以免引入数值域假设或误合并。
