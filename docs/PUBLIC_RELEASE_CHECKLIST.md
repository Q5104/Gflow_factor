# 公开发布检查清单

状态日期：2026-08-21
当前结论：**文档与 Notebook 已达到本轮本地提交条件；提交后仍不自动 push。**

## 已通过

- [x] README 明确 Raw Baseline 已完成、Derived 尚未完成及可报告结论边界。
- [x] README 顶部包含简短英文摘要，正文以中文为主。
- [x] 明确当前无 License，不授权第三方复用。
- [x] 提供环境、外部数据 schema、Notebook 顺序、new/resume 和 artifact 验收说明。
- [x] Derived 数据与训练 Notebook 已清空输出并恢复默认关闭闸门。
- [x] Raw Stage 5 Notebook 已恢复 `MODE='new'`、`RESUME_RUN_DIR=None` 的可移植发布态。
- [x] 下载 Notebook 不再写死本机项目目录或 Conda 安装路径。
- [x] 文档相对链接和引用的 Notebook 路径通过检查。
- [x] 候选文本文件未发现疑似硬编码密钥或邮箱。
- [x] `data/`、`runs/`、checkpoint 和 `outputs/` 受 `.gitignore` 保护。
- [x] Derived 与 Raw Stage 5 Notebook 的 focused publication tests 通过。
- [x] `stage5_reporting.ipynb` 已清空 45 个输出和执行计数，不再携带本机绝对路径。
- [x] 2026-08-21 前台运行本轮变更覆盖的 21 个测试模块，共 192 项测试全部通过，无失败或跳过。

## 发布前仍需处理

- [ ] **全量单元测试本轮未完成。** 2026-08-21 的一次 `unittest discover` 被人工终止，不能记录为通过或失败；本轮只声明上述 192 项变更覆盖测试通过。
- [ ] Legacy 证据 Notebook 有 16 个有意保留的输出，也包含作者机器路径；若保留，应把它明确视为历史展示快照而非可移植执行结果。
- [x] 根目录 `GFlowNet 因子挖掘.pptx` 已从本轮两个提交中人工排除。该文件未被 `.gitignore` 排除，未来仍需单独审查后才能发布。
- [x] 已审查全部 staged files，本轮提交不包含 `data/`、`runs/`、`outputs/`、checkpoint、SQLite、模型、PPT 或受限申万数据。
- [x] 已在 commit 前记录 `git status --short`、变更覆盖测试命令、192 项通过、0 失败、0 跳过，并明确全量测试未完成。

## 有意保留的限制

- [x] 不添加 License；仓库是公开研究展示，不是开源发布。
- [x] 申万 PIT 逐日数据没有稳定公开来源，只公开输入 schema，由使用者自备合法数据。
- [x] 当前没有独立正式 Barra 构建 Notebook，因此不声称从零一键复现真实训练。
- [x] Derived Stage 5 为 partial run；正式 Stage 6、Factor Pool、Strategy 和 OOS 尚未完成。
- [x] 不提交本地数据、run、checkpoint、registry、模型和生成报告。

## 本轮 commit 分组

为了不制造无法独立运行的中间历史，本轮将 Feature Space、下游兼容和相关测试合并为一个功能提交，公开文档单独提交。

### Commit A｜Daily-Derived 功能与下游兼容

```text
docs/daily_derived/
factor_gfn/data/daily_derived*.py
factor_gfn/feature_spaces.py
factor_gfn/data/__init__.py
factor_gfn/grammar/*.py
factor_gfn/evaluator/interpreter.py
factor_gfn/gfn/*.py 中的 Feature Space / registry plumbing
notebooks/prepare_daily_derived_data.ipynb
notebooks/build_daily_derived_v1_exact_tb_n1_n2.ipynb
notebooks/run_stage5_daily_derived_v1_hybrid_variance_real_5_15.ipynb
对应 data / grammar / GFN / Notebook tests
factor_gfn/backtest/*.py 的 vocabulary / matrix / authority 修改
factor_gfn/reporting/*.py
factor_gfn/backtest/rolling_icir.py
Raw reporting / OOS Notebook 的对应修改
对应 backtest / reporting / OOS tests
```

建议信息：

```text
feat: add daily-derived feature-space pipeline
```

### Commit B｜公开文档与复现说明

```text
README.md
CONTRIBUTING.md
docs/README.md
docs/REPRODUCIBILITY.md
docs/PUBLIC_RELEASE_CHECKLIST.md
docs/handoffs/
DEVELOPMENT_SPEC.md 的对应 authority 更新
```

建议信息：

```text
docs: document project status and reproducibility workflow
```

## 最终提交前命令

```powershell
.\.venv\python.exe -m pip check
.\.venv\python.exe -m unittest discover -s tests -v
git diff --check
git status --short
git diff --cached --stat
```

只有运行实际完成并得到完整输出后，才能在开发日志或提交说明中记录测试通过数量。
