# GFlowNet 日频 K 线因子挖掘开发规范

> 状态：初版、持续更新  
> 建立日期：2026-08-04  
> 最近同步：2026-08-07  
> 当前阶段：阶段 1–4 工程验收已完成，准备进入阶段 5 因子筛选与回测验证
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
- 单元与集成测试：`tests/`，当前完整测试为 178 项；
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
- 点时行业长表已完成全量构建与 QA，并已通过阶段三行业中性化实跑验证；阶段 1–4 的工程验收已完成，下一步进入阶段 5。五步最小实验只证明真实训练链路与确定性续跑可用，不等同于研报规模的正式训练。

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
- 候选股票缺失行业信息时不参与行业回归，其缩尾后但未经残差替换的因子值保留并进入同一截面的后续 z-score。这里的“保留原始值”特指保留缩尾后的值，不回退到缩尾前数值；
- 若某行业只有一只有效股票，保留该行业哑变量，该股票在可识别的行业回归中允许得到 0 残差；若截面有效股票数少于行业数加 1，则整个截面跳过行业中性化、直接 z-score，并为该日期记录一次可汇总警告；
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
- 使用表达式后序序列和栈统一调用全部 52 个算子；
- 输出为 `(date, stock)`，并校验输入维度、特征数、动作元数、窗口签名、算子输出和最终栈长度；
- 解释器不修改原始六特征张量，当前不引入缓存、Numba 或 MemMap。

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
- 候选因子市值中性化暂缓，待阶段五回测评估实际市值暴露后再决定是否加入；第一版先由 reward 中的 Barra Size 收益序列相关性惩罚把关；
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
- 保存五个带符号 Barra 相关系数、最大暴露风格、共同有效期数、IC/Long IR 有效期数、原始/稳定化/log Reward 和行业中性化状态；每个候选还必须持久化固定调仓日历上去重后的 `neutralization_skipped_dates` 及其占调仓期数的 `neutralization_skipped_rate`，包括最终无效的候选。
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

### 阶段 5：因子筛选与回测验证

目标：从搜索结果中得到稳定、低相关的 Alpha 池。

主要工作：

- 样本内、验证集和最终样本外指标；
- 研报硬性筛选条件的可配置实现；
- 表达式结构去重；
- 截面相关性和多头收益序列相关性分析；
- 贪心去相关；
- 分组、多头、多空、换手和交易成本分析；
- 按年份和市场阶段检查稳定性。

### 暂缓阶段

- 日频人工衍生特征；
- 分钟聚合特征；
- 原始分钟数据与 MemMap block cache；
- AlphaEval；
- DPP；
- LGBM 多因子合成；
- 指数增强组合。

只有原始日频 K 线最小闭环稳定后，才评估是否进入这些阶段。

## 7. 候选代码结构

以下结构反映当前已经创建的核心模块；`backtest/` 将在阶段 5 开始时再创建：

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
│   ├── loss.py
│   ├── trainer.py
│   └── checkpoint.py
└── backtest/
    ├── __init__.py
    ├── selector.py
    ├── engine.py
    └── metrics.py
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
- 2019-2020 在项目中称为验证集还是测试集？
- 最终样本外测试区间是否采用 2021 年以后，以及截止日期如何冻结？

### GFlowNet

- 正式训练的 Transformer 规模、batch size、学习率、训练步数和硬件预算；
- 正式规模训练中的无效轨迹比例、有效 batch 补采上限及是否需要调整；
- 训练期间候选表达式持久化归档、验证检查点选择和训练后补采规模；
- 数值指纹、退化因子和高度相关因子的阶段 5 去重与筛选细节。

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
- 候选因子市值中性化暂缓，待阶段五回测评估市值暴露后决定；当前由 reward 中的 Barra Size 惩罚把关。
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
