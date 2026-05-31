# 面向 RAG 配置调优的 LLM-GA 昂贵优化求解器

本项目将 RAG 问答系统的配置调优抽象为 **6 维单目标昂贵黑箱优化问题**，在有限的函数评价预算（`MAX_FES = 300`）内，对比 Random Search、GA 和 LLM-GA 三种搜索策略的效果。

LLM-GA 的核心思路：以 GA 为主搜索器，每隔若干代调用一次真实大语言模型（LLM）生成候选解注入种群，利用 LLM 的全局推理能力辅助进化搜索。LLM 候选解仍然由 fit 函数评价，不享有任何特权。

## 项目结构

```
LLM_GA_RAG_Optimization/
├── config.py              # 所有参数集中定义（实验参数、GA算子、LLM参数、RAG参数空间）
├── ga_solver.py           # 核心引擎：参数空间工具、目标函数、GA算子、三种对比算法
├── llm_module.py          # 大模型交互：prompt构造、API调用、JSON解析与候选解校验
├── main.py                # 实验入口：按种子运行三种算法，输出结果文件
├── utils.py               # 工具函数：随机种子、CSV保存、收敛曲线绘制
├── requirements.txt       # Python 依赖
├── .env                   # API 配置（不提交）
├── docs/
│   └── technical_report.md  # 技术文档（架构设计、核心逻辑、实验分析）
└── results/
    ├── result.csv           # 详细结果
    ├── summary.csv          # 统计汇总
    └── convergence.png      # 收敛曲线图
```

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

在项目根目录创建 `.env` 文件：

```env
SPARK_API_KEY=你的API密钥
SPARK_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
SPARK_MODEL=gui-plus-2026-02-26
```

项目通过 OpenAI 兼容接口调用 LLM，更换 `.env` 中的三个变量即可切换到其他兼容平台（讯飞星火、DeepSeek 等）。详见 [技术文档](docs/technical_report.md)。

## 实验结果（30 轮，seed = 42-71）

| 算法 | 均值 | 标准差 | 最小值 | 最大值 | 胜出轮数 |
|---|---|---|---|---|---|
| Random Search | 50.03 | 6.77 | 34.91 | 63.02 | 0 / 30 |
| GA | 22.00 | 3.93 | 16.50 | 32.09 | 17 / 30 |
| LLM-GA | 21.04 | 3.09 | 16.13 | 27.61 | 13 / 30 |

- GA 和 LLM-GA 均大幅优于 Random Search（均值降低约 56-58%）
- LLM-GA 的均值和标准差略优于 GA，但 GA 在胜出轮数上更多（17 vs 13），二者整体势均力敌
- 详细分析见 [技术文档第五章](docs/technical_report.md#五实验结果与分析)
