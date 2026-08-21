# 公开反馈与协作说明

感谢关注本研究仓库。当前仓库用于公开展示研究过程和工程实现，尚未提供开源许可证，也未开放常规的外部代码贡献流程。

## 当前接受的反馈

欢迎通过 GitHub Issue 提交：

- README、复现指南或 Notebook 操作步骤中的错误；
- 可以最小复现的代码缺陷；
- 数据泄漏、未来信息、Train/Validation/Test 边界或指标定义问题；
- artifact、checkpoint、fingerprint 和 resume 一致性问题；
- 不涉及受限数据再分发的复现差异报告。

Issue 请尽量包含：

```text
Git commit
Python / PyTorch / CUDA 版本
操作系统
执行的命令或 Notebook + Cell
完整错误信息
最小复现步骤
实际结果与预期结果
```

请删除用户名、绝对本地路径、账号、密钥、公司内部数据和无法公开的日志内容。

## Pull Request 边界

当前不接受未经事先沟通的外部 Pull Request。仓库公开可见不代表获得复制、修改、再分发或商业使用代码的许可。如确有合作需要，请先通过 Issue 说明目标、范围和拟修改文件，等待仓库所有者明确授权。

任何讨论、Issue 或受邀贡献均不得包含：

- 原始行情、申万逐日行业文件或其他受许可约束的数据；
- `data/`、`runs/`、checkpoint、模型、SQLite registry 或生成报告；
- API key、token、账号信息、个人隐私或单位内部材料；
- 无法说明来源与授权状态的研报、图表、图片或演示文稿素材。

## 研究完整性要求

受邀修改必须保持：

- Test/OOS 不参与特征、Reward、方向、阈值、Factor Pool 或策略选择；
- Raw Baseline artifact 不被 Derived 或其他实验覆盖；
- material config 变化创建新 run，不强制恢复旧 checkpoint；
- synthetic smoke、partial run 和正式真实结果明确区分；
- 公式、时间窗口、NaN、PIT、dtype、mask 和指标语义变化有测试与文档记录。

## 本地验证

项目使用 Python 3.12。提交缺陷报告前可运行：

```powershell
.\.venv\python.exe -m pip check
.\.venv\python.exe -m unittest discover -s tests -v
```

长时间下载、数据构建、Exact-TB、真实训练、Stage 6 和 OOS 不属于普通自动验证，不应为了提交 Issue 或代码建议而自动启动。
