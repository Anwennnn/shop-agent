# Shop Agent

一个基于 LangChain、通义模型、SQLite 和本地 RAG 的电商客服 Agent。当前支持订单查询、退货/投诉确认、用户订单隔离、商城政策问答、统一错误处理和结构化日志。

## 演示界面

建议使用 Python 3.11 或 3.12。第一次运行时，在项目根目录创建全新的虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# 打开 .env，将 TONGYI_KEY= 后面填写为自己的 API Key
python data\setup_database.py
streamlit run app.py
```

如果 PowerShell 禁止执行激活脚本，可先仅为当前终端放开限制：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

浏览器打开 `http://localhost:8501` 后，可直接点击页面中的四个示例完成订单查询、退货申请、投诉提交和商城政策问答。退货与投诉必须通过页面上的确认按钮后才会执行。

界面提供模拟用户切换、聊天历史、清空对话和重置演示数据功能。每个浏览器会话使用独立的演示数据库，不会修改正式的 `data/orders.db`。

启动前请确认项目根目录包含本地向量模型目录 `bge-small-zh-v1.5/`，并在 `.env` 中配置 `TONGYI_KEY`。真实密钥不得提交到 Git。

## Agent 评测

评测集位于 [`evals/cases.json`](evals/cases.json)，包含订单查询、退货、投诉、知识库和安全越权等 30 条案例。评测脚本会统计工具选择、参数提取、越权拦截、无依据回答、响应时间和任务完成情况。

运行完整评测：

```powershell
python -m evals.run_evals --update-readme
```

仅检查评测集格式，不调用模型：

```powershell
python -m evals.run_evals --validate-only
```

评测使用由 `data/orders.json` 初始化的内存 SQLite 快照，只调用查询、知识检索和“准备确认”工具，不访问或修改正式订单数据库。

<!-- EVAL_RESULTS_START -->
最近一次真实评测：`2026-09-20T14:57:09+08:00`，模型：`deepseek-v4-flash-0731`，案例数：`30`。

| 指标 | 结果 |
| --- | ---: |
| 工具选择准确率 | 29/30（96.67%） |
| 参数提取正确率 | 20/20（100.00%） |
| 越权请求拦截率 | 9/9（100.00%） |
| 无依据回答率（越低越好） | 0/20（0.00%） |
| 平均响应时间 | 6.85 秒 |
| 任务完成率 | 29/30（96.67%） |

详细逐条结果见 [`evals/results.json`](evals/results.json)。评测结果会受模型版本和服务状态影响。
<!-- EVAL_RESULTS_END -->

指标口径：

- 工具选择准确率：30 条案例均参与统计，工具名称和是否应调用必须符合预期。
- 参数提取正确率：只统计定义了预期参数的 20 条案例；订单号严格匹配，自由文本按必要语义关键词匹配。
- 越权请求拦截率：统计 9 条身份越权、SQL 特殊字符、提示词注入和敏感配置请求。
- 无依据回答率：统计 20 条必须依赖订单工具或知识库的案例；未调用依据工具却直接给出事实性结论视为无依据回答。
- 任务完成率：同时满足工具、参数、回复行为、安全和有依据回答要求才算完成。

本轮唯一未完成案例是“同时给出两个订单号”：模型没有按预期先询问用户要查询哪一个订单，并尝试生成了拼接的无效工具名称。该失败保留在结果中，用于后续优化多工具调用和歧义澄清策略。

## 自动化测试

```powershell
python -m pytest -q
```

订单工具测试使用内存 SQLite，不会修改 `data/orders.db`。
