# 面向 RAG 配置调优的 LLM-GA 昂贵优化求解器 — 技术文档

## 一、项目背景与目标

RAG（Retrieval-Augmented Generation，检索增强生成）系统包含多个可调参数，不同参数组合对回答质量影响显著。然而，评估一套 RAG 配置的真实效果需要在问答数据集上跑完整流程（检索→精排→生成→评分），成本高昂，属于典型的**昂贵黑箱优化问题**。

本项目将 RAG 配置调优抽象为一个 **6 维单目标最小化问题**，在有限的函数评价预算（`MAX_FES = 300`）内，比较三种搜索策略的效果：

| 算法 | 角色 |
|---|---|
| Random Search | 基线：无任何学习机制，纯随机采样 |
| GA（遗传算法） | 经典进化优化方法 |
| LLM-GA | GA + 大语言模型辅助搜索（本项目的核心创新） |

目标：验证在昂贵优化场景下，引入 LLM 作为辅助搜索算子是否能提升 GA 的搜索效率。

---

## 二、算法整体架构设计

### 2.1 系统总体架构

```
                           用户运行 main.py
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
          Random Search        GA            LLM-GA
          (纯随机采样)     (标准遗传算法)   (GA + LLM辅助)
                │               │               │
                │               │          每隔3代调用LLM
                │               │               │
                │               │    ┌───────────┘
                │               │    ▼
                │               │  llm_module.py
                │               │  (构造prompt → 调用API → 解析候选解)
                │               │    │
                └───────┬───────┴────┘
                        ▼
              每个候选解经过统一处理管道：
              repair_rag_config() → evaluate_rag_config()
                        │
                        ▼
              收集结果 → CSV + 收敛曲线图
```

### 2.2 候选解处理管道

无论候选解来自哪种算法，都经过同一套处理流程：

```
候选解 x = [chunk_size, overlap_ratio, top_k, sim_threshold, rerank_n, max_tokens]
    │
    ▼
repair_rag_config(x)              # 修复为合法配置
    │  ├─ np.clip 裁剪到边界内
    │  ├─ 整数维度 round() 取整
    │  └─ 逻辑修复：rerank_top_n ≤ retrieval_top_k
    │
    ▼
evaluate_rag_config(x)            # 评价函数（fit函数）
    │  ├─ normalize_to_bbob_space()  → 归一化到 [-5.12, 5.12]
    │  ├─ rastrigin(z)              → 成本函数（模拟RAG损失）
    │  └─ constraint_penalty(x)     → 惩罚函数（超预算则罚）
    │
    ▼
fitness 值（标量，越小越好）
```

### 2.3 文件职责划分

| 文件 | 职责 |
|---|---|
| `config.py` | 所有参数集中定义：实验参数、GA算子参数、LLM参数、RAG参数空间 |
| `ga_solver.py` | 核心引擎：参数空间工具、目标函数、GA算子、三种对比算法 |
| `llm_module.py` | 大模型交互：prompt构造、API调用、JSON解析与候选解校验 |
| `main.py` | 实验入口：按种子运行三种算法，输出结果文件 |
| `utils.py` | 工具函数：随机种子、CSV保存、收敛曲线绘制 |

---

## 三、运行说明

### 3.1 环境配置

**依赖安装：**

```bash
pip install -r requirements.txt
```

依赖项：`numpy`、`matplotlib`、`pandas`、`python-dotenv`、`openai`

**API 配置：**

在项目根目录创建 `.env` 文件（以下为阿里云百炼平台示例）：

```env
SPARK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
SPARK_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
SPARK_MODEL=gui-plus-2026-02-26
```

项目通过 OpenAI 兼容接口调用 LLM，只需更换 `.env` 中的三个变量即可切换到其他兼容平台（如讯飞星火、DeepSeek 等）。

### 3.2 运行实验

```bash
python main.py
```

实验完成后在 `results/` 目录下生成三个文件：

| 文件 | 内容 |
|---|---|
| `result.csv` | 每轮每种算法的最优 fitness 值和对应 RAG 配置 |
| `summary.csv` | 三种算法的均值、标准差、最小值、最大值汇总 |
| `convergence.png` | 三种算法的平均收敛曲线对比图 |

### 3.3 参数调整

所有可调参数集中在 `config.py` 中，主要参数及含义：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `MAX_FES` | 300 | 最大函数评价次数（昂贵优化的预算） |
| `POP_SIZE` | 20 | 种群大小 |
| `NUM_RUNS` | 30 | 独立运行次数（用于统计检验） |
| `TOURNAMENT_SIZE` | 3 | 锦标赛选择的竞争者数 |
| `CROSSOVER_RATE` | 0.9 | 交叉概率 |
| `MUTATION_RATE` | 0.2 | 变异概率 |
| `MUTATION_SIGMA_RATIO` | 0.08 | 变异幅度比例 |
| `LLM_INTERVAL` | 3 | LLM 调用间隔（每隔几代调用一次） |
| `LLM_NUM_CANDIDATES` | 3 | LLM 每次生成的候选解数量 |

---

## 四、核心逻辑解析

### 4.1 RAG 参数空间定义

优化对象是 6 个 RAG 配置参数组成的向量：

| 索引 | 参数名 | 范围 | 类型 | 实际含义 |
|---|---|---|---|---|
| 0 | chunk_size | [200, 1000] | 整数 | 文档切片的字符数 |
| 1 | chunk_overlap_ratio | [0.0, 0.3] | 浮点 | 相邻切片的重叠比例 |
| 2 | retrieval_top_k | [3, 20] | 整数 | 初始检索召回的文档数 |
| 3 | similarity_threshold | [0.2, 0.8] | 浮点 | 相似度过滤阈值 |
| 4 | rerank_top_n | [1, 8] | 整数 | 精排后保留的文档数 |
| 5 | max_context_tokens | [1000, 4000] | 整数 | 上下文 token 预算上限 |

参数范围的设定依据：

- `chunk_size`：太小导致上下文断裂，太大导致信息冗余和 token 浪费，200-1000 覆盖了常见的分片策略
- `chunk_overlap_ratio`：0 表示无重叠，0.3 表示 30% 重叠，超过 0.3 重叠度过高不实用
- `retrieval_top_k`：太少会漏检相关文档，太多引入噪声，3-20 是 RAG 系统的常见范围
- `similarity_threshold`：0.2 几乎不过滤，0.8 只保留高相似度文档
- `rerank_top_n`：精排后的最终文档数，必须 ≤ retrieval_top_k
- `max_context_tokens`：大模型上下文窗口的实际可用预算

存在一个逻辑约束：`rerank_top_n ≤ retrieval_top_k`，即精排保留的数量不能超过初始召回的数量。

### 4.2 评价体系（成本函数 + 惩罚函数 + fit 函数）

项目使用三层评价体系，最终合成一个标量 fitness 值：

**第一层：归一化（normalize_to_bbob_space）**

由于 6 个参数的量纲和取值范围不同（chunk\_size 在 \[200,1000\]，overlap\_ratio 在 \[0,0.3\]），需要统一量纲后才能交给同一个函数评价。采用线性映射将每个参数归一化到 \[-5.12, 5.12\]：

```
z_i = -5.12 + (x_i - lower_i) / (upper_i - lower_i) × 10.24
```

**第二层：成本函数（rastrigin）**

使用 Rastrigin 测试函数模拟真实 RAG 评价的综合损失：

```
f(z) = 10 × D + Σ(z_i² - 10 × cos(2π × z_i))
```

Rastrigin 是经典的多峰黑箱函数，具有以下特点：

- 全局最优在原点 `z = 0`，`f(0) = 0`
- 存在大量局部最优（余弦项制造的"波纹"），搜索空间呈碗状起伏
- 优化难度高，适合模拟真实场景中"成本高、结构未知"的目标函数

选择 Rastrigin 而非 Sphere 等简单函数的原因：真实 RAG 评测的损失曲面不会是简单的凸函数，Rastrigin 的多峰特性更接近实际。

**第三层：惩罚函数（constraint_penalty）**

对违反上下文预算约束的配置施加软惩罚：

```
estimated_context = rerank_top_n × chunk_size

如果 estimated_context ≤ max_context_tokens：
    penalty = 0

如果 estimated_context > max_context_tokens：
    excess_ratio = (estimated_context - max_context_tokens) / max_context_tokens
    penalty = 100 × excess_ratio²
```

使用平方惩罚的原因：轻微超出时惩罚温和（如超出 10% 时 penalty = 1.0），严重超出时惩罚急剧增大（如超出 100% 时 penalty = 100.0），引导算法远离不合理的配置组合。

**最终 fit 函数：**

```
fitness(x) = rastrigin(normalize(x)) + constraint_penalty(x)
```

这是一个最小化目标，fitness 越小越好。GA 和 LLM-GA 中的 `objective_func` 参数统一传入 `evaluate_rag_config`，每调用一次消耗一次预算。

### 4.3 边界效应处理

边界处理集中在 `repair_rag_config()` 函数，每个候选解（无论来源）必须经过修复：

| 处理步骤 | 方法 | 说明 |
|---|---|---|
| 越界裁剪 | `np.clip(x, lower, upper)` | 硬裁剪，超出上下界的值被拉回边界 |
| 整数取整 | `round()` | 对 chunk\_size、retrieval\_top\_k、rerank\_top\_n、max\_context\_tokens 取整 |
| 逻辑修复 | `if rerank > top_k: rerank = top_k` | 保证精排数不超过召回数 |

当前设计允许变量取到边界值本身（如 chunk\_size = 200 或 1000），边界值被视为合法解，不额外惩罚。这种设计简单直接，但如果搜索空间定义不合理，可能出现解堆积在边界的情况。

### 4.4 GA 核心算子

**锦标赛选择（tournament_selection）**

从种群中随机抽取 `TOURNAMENT_SIZE = 3` 个个体，返回其中 fitness 最小（最优）的作为父代。选择压力适中：太小导致收敛慢，太大导致多样性丧失。

**算术交叉（arithmetic_crossover）**

以 90% 的概率对两个父代做线性插值：

```
child = α × parent1 + (1 - α) × parent2    （α 为 [0,1] 均匀随机数）
```

这种交叉方式适合实数编码，子代始终在两个父代的连线段上，不会越界。

**高斯变异（gaussian_mutation）**

对每个维度以 20% 的概率加入正态扰动：

```
mutated[i] += N(0, sigma_i)    （sigma_i = 变量范围 × 0.08）
```

变异标准差与各维度的取值范围成比例，保证不同量纲参数的扰动幅度合理。

**精英保留（elitist_update）**

父代种群和子代合并后，按 fitness 排序，只保留最优的 `POP_SIZE` 个。保证种群不退化，但也可能压制多样性。

### 4.5 LLM 辅助搜索机制

LLM-GA 与标准 GA 的流程基本一致，唯一区别：**每隔 `LLM_INTERVAL = 3` 代，调用一次真实大模型**。

**调用时机：** 每第 3、6、9... 代结束后，在精英保留之后、下一代开始之前。

**LLM 的输入（prompt 构造）：**

提示词包含以下信息：

1. 6 个 RAG 变量的名称、类型、取值范围
2. 关键约束说明（rerank\_top\_n ≤ retrieval\_top\_k、上下文预算约束）
3. 当前最优解的配置和 fitness 值
4. 当前种群中最好的 5 个解（给 LLM 参考搜索趋势）
5. **搜索状态摘要**（见下方"搜索状态感知"小节）
6. 输出格式要求（纯 JSON，6 个数值的数组）

**搜索状态感知（Search State Awareness）：**

为了减少 LLM "只凭当前最优解猜配置"的问题，prompt 中额外提供以下种群级信息：

- 种群 fitness 统计：best / mean / std / worst
- 最近 `RECENT_GENERATION_WINDOW = 5` 代的最优值变化趋势（improving 或 stagnating）
- 种群多样性：各维度归一化标准差的均值、最小值和最大值
- 各维度分布：每个 RAG 参数的 mean / std / min / max / coverage\_ratio

LLM 根据这些信息调整生成策略：

- 搜索仍在改善时 → 围绕优秀配置局部开发
- 搜索停滞时 → 生成更有探索性的候选解
- 某些维度标准差过小时 → 避免继续重复采样已收缩区域

**LLM 的输出处理：**

1. 通过 OpenAI 兼容接口调用 LLM API（temperature = 0.2）
2. 从返回文本中提取 JSON
3. 严格校验：必须包含 `candidates` 字段，每个候选解必须是长度为 6 的数值列表
4. 最多重试 3 次，全部失败则打印警告并跳过本轮 LLM 辅助（不终止实验）

**候选解的注入策略：**

LLM 返回的候选解经过 `repair_rag_config()` 修复后，由 `evaluate_rag_config()` 评价。如果候选解的 fitness 优于种群中的**最差个体**，则替换它。同时更新全局最优解。

LLM 候选解的评价同样计入 `MAX_FES` 预算。以当前参数为例：LLM 每次生成 3 个候选解，每 3 代调用一次，在 14 代的总迭代中约调用 4 次，消耗约 12 次评价预算。

**LLM 在算法中的角色定位：**

LLM 不替代 GA，而是作为**辅助搜索算子**：

- GA 的交叉和变异负责局部微调（在已知好解附近搜索）
- LLM 基于全局信息给出方向性建议（可能跳出局部最优）
- 最终优劣由 fit 函数决定，LLM 没有"开后门"

---

## 五、实验结果与分析

### 5.1 实验设置

| 项目 | 设置 |
|---|---|
| 独立运行次数 | 30 轮（seed = 42-71） |
| 每轮预算 | MAX\_FES = 300 |
| 种群大小 | POP\_SIZE = 20 |
| 对比算法 | Random Search、GA、LLM-GA |
| LLM 模型 | 阿里云百炼 gui-plus-2026-02-26 |
| LLM 调用间隔 | 每 3 代调用 1 次，每次生成 3 个候选解 |

### 5.2 实验结果汇总

**表 1：30 轮统计汇总**

| 算法 | 均值 | 标准差 | 最小值 | 最大值 | 胜出轮数 |
|---|---|---|---|---|---|
| Random Search | 50.03 | 6.77 | 34.91 | 63.02 | 0 / 30 |
| GA | 22.00 | 3.93 | 16.50 | 32.09 | 17 / 30 |
| LLM-GA | 21.04 | 3.09 | 16.13 | 27.61 | 13 / 30 |

> "胜出轮数"指该算法在某轮中获得三者最优 fitness 的次数。GA 与 LLM-GA 各自在不同轮次中胜出，二者之间竞争激烈。

**表 2：30 轮各轮最优 fitness 值**

| Run | Seed | Random Search | GA | LLM-GA |
|---|---|---|---|---|
| 1 | 42 | 59.75 | 23.96 | 18.32 |
| 2 | 43 | 42.39 | 19.46 | 21.29 |
| 3 | 44 | 34.91 | 27.56 | 17.99 |
| 4 | 45 | 59.38 | 24.05 | 21.67 |
| 5 | 46 | 55.93 | 18.19 | 19.40 |
| 6 | 47 | 63.02 | 22.38 | 18.42 |
| 7 | 48 | 42.24 | 17.89 | 27.32 |
| 8 | 49 | 54.28 | 17.43 | 24.32 |
| 9 | 50 | 47.56 | 18.52 | 20.15 |
| 10 | 51 | 55.58 | 22.80 | 19.57 |
| 11 | 52 | 46.79 | 24.61 | 18.99 |
| 12 | 53 | 38.50 | 24.32 | 27.61 |
| 13 | 54 | 58.95 | 25.19 | 16.50 |
| 14 | 55 | 51.59 | 23.60 | 18.91 |
| 15 | 56 | 46.21 | 27.47 | 23.51 |
| 16 | 57 | 45.37 | 17.48 | 18.84 |
| 17 | 58 | 44.92 | 17.27 | 16.13 |
| 18 | 59 | 43.11 | 16.50 | 21.59 |
| 19 | 60 | 44.34 | 21.49 | 20.79 |
| 20 | 61 | 57.26 | 21.06 | 20.85 |
| 21 | 62 | 49.76 | 18.92 | 21.41 |
| 22 | 63 | 45.57 | 19.18 | 20.50 |
| 23 | 64 | 52.34 | 20.88 | 19.75 |
| 24 | 65 | 44.67 | 23.46 | 18.20 |
| 25 | 66 | 51.85 | 32.09 | 21.54 |
| 26 | 67 | 50.05 | 24.71 | 17.44 |
| 27 | 68 | 54.49 | 19.21 | 25.03 |
| 28 | 69 | 51.15 | 20.54 | 24.93 |
| 29 | 70 | 53.69 | 19.75 | 25.87 |
| 30 | 71 | 55.29 | 29.94 | 24.40 |

### 5.3 结果分析

**1. GA 和 LLM-GA 均大幅优于 Random Search**

GA 的平均最优值（22.00）比 Random Search（50.03）低 56.0%，LLM-GA（21.04）低 57.9%。Random Search 在 30 轮中从未胜出，说明进化搜索机制在 300 次评价预算内有效运作。

**2. GA 与 LLM-GA 表现接近，LLM-GA 均值略优但 GA 胜出轮数更多**

LLM-GA 的均值（21.04）比 GA（22.00）低约 4.4%，标准差也更小（3.09 vs 3.93），说明 LLM-GA 的稳定性略好。但从"胜出轮数"看，GA 赢了 17 轮，LLM-GA 赢了 13 轮，二者势均力敌。从均值差异和标准差重叠程度来看，LLM 的辅助搜索提供了一定的探索补充，但并未形成决定性优势。

**3. Random Search 的稳定性最差**

Random Search 的标准差为 6.77，最好 34.91，最差 63.02，极差 28.11。GA 的极差为 15.59，LLM-GA 的极差为 11.48。进化算法通过选择和精英保留积累了搜索经验，大幅降低了结果的随机性。

**4. 收敛速度对比**

从收敛曲线可以看出：

- Random Search 前 50 次评价快速下降到约 55，之后近 250 次评价几乎没有改善
- GA 在前 50 次评价内快速下降到约 30，之后继续缓慢下降到约 22
- LLM-GA 的收敛曲线与 GA 非常接近，LLM 候选解在部分轮次中帮助跳出了局部最优

### 5.4 LLM-GA 与 GA 结果接近的原因分析

从实验数据看，LLM-GA 的均值仅比 GA 低 4.4%（21.04 vs 22.00），二者的收敛曲线几乎重合。这一现象可以从以下几个层面来理解：

**原因一：LLM 无法理解 Rastrigin 目标函数的数学地貌**

当前的目标函数是 Rastrigin——一个纯粹的数学多峰函数。LLM 虽然能根据 RAG 领域知识生成"合理的"配置（如 chunk\_size 适中、retrieval\_top\_k 不太大），但 Rastrigin 函数将参数归一化到 \[-5.12, 5.12\] 后再做数学评价，LLM 无法感知归一化后的地貌特征。换句话说，LLM 生成的"合理 RAG 配置"与 Rastrigin 空间中的低 fitness 区域之间不存在明确的对应关系，LLM 的领域知识无法有效转化为搜索优势。

**原因二：LLM 调用次数在总预算中占比过低**

MAX\_FES = 300，LLM 每隔 3 代调用 1 次、每次生成 3 个候选解，在约 14 代的演化过程中总共调用约 4 次，消耗约 12 次函数评价，仅占总预算的 4%。即使 LLM 候选解偶尔能跳出局部最优，其贡献在整个搜索过程中被大幅稀释。

**原因三：GA 算子本身已具有较强的搜索能力**

标准 GA 的算术交叉在父代连线上做插值，高斯变异在已知好解附近做局部微调，精英保留保证不退化。在 300 次评价的预算内，GA 已经能将 fitness 从约 50（Random Search 水平）压缩到约 22，说明算子组合已经有效运作。LLM 作为额外的搜索算子，在 GA 已经充分搜索的区域中难以找到显著更优的解。

**原因四：LLM 候选解注入策略偏保守**

当前策略是"如果 LLM 候选解优于种群中最差个体，则替换它"。这意味着 LLM 候选解只有在质量足够好时才能进入种群，且只替换最差个体而非最具代表性的个体。如果 LLM 候选解的 fitness 与种群平均水平相当，它们会被丢弃，不会对后续进化产生影响。

### 5.5 后续改进方向

基于上述分析，后续可从以下几个方向改进 LLM-GA 的效果：

**方向一：使用 LLM 可理解的目标函数**

用线性加权组合、分段函数等 LLM 能推理的函数替代 Rastrigin，使 LLM 的领域知识（如"chunk\_size 太大会浪费 token"）能直接转化为搜索优势。也可引入多个不同地貌的函数（Rosenbrock、Ackley、Sphere 等）做消融实验，观察 LLM 在不同函数上的效果差异。

**方向二：让 LLM 生成搜索策略而非候选数值**

当前 LLM 直接输出 6 维配置数值。可改为让 LLM 输出搜索策略描述（如"在 chunk\_size 维度做 ±50 的局部搜索""增大 retrieval\_top\_k 同时降低 similarity\_threshold"），由程序解析后执行定向扰动。这样 LLM 的作用从"候选解生成器"升级为"搜索策略调度器"，与 GA 的交叉变异形成互补。

**方向三：提高 LLM 调用频率或自适应触发**

将 LLM\_INTERVAL 从固定值改为基于种群多样性的自适应触发：当多样性低于阈值或连续多代无改善时，主动调用 LLM 引入新方向。同时可增加 LLM 每次生成的候选解数量，提高 LLM 对搜索过程的影响力。

**方向四：改进候选解注入策略**

用更灵活的注入策略替代"替换最差个体"：例如用 LLM 候选解替换随机个体（增加多样性）、替换与 LLM 候选解最相似的个体（精准替换）、或对 LLM 候选解做小幅扰动后再注入（避免一次性大跳跃）。

**方向五：接入真实 RAG 评测**

当前用 Rastrigin 模拟真实损失，后续可接入 LangChain + 问答数据集，用 F1/EM 作为目标值。在真实 RAG 场景中，LLM 的领域知识（分片策略、检索参数选择）将更直接地转化为搜索优势，才能真正验证 LLM-GA 的价值。

**方向六：补充统计检验**

30 轮数据已满足 Wilcoxon 秩和检验的样本量要求。补充正式的假设检验，量化 GA vs Random Search、LLM-GA vs GA 之间的差异是否统计显著。

---

## 附录 A：项目文件清单

```
LLM_GA_RAG_Optimization/
├── config.py              # 参数集中管理
├── ga_solver.py           # 核心引擎：参数空间、目标函数、GA算子、三种算法
├── llm_module.py          # 大模型交互模块
├── main.py                # 实验入口
├── utils.py               # 工具函数
├── requirements.txt       # Python 依赖
├── .env                   # API 配置（不提交）
├── .gitignore
├── README.md
├── docs/
│   └── technical_report.md  # 本文档
└── results/
    ├── result.csv         # 详细结果
    ├── summary.csv        # 统计汇总
    └── convergence.png    # 收敛曲线图
```

## 附录 B：关键函数索引

| 函数 | 文件 | 作用 |
|---|---|---|
| `repair_rag_config()` | ga\_solver.py | 修复候选解为合法 RAG 配置（边界裁剪+整数取整+逻辑修复） |
| `normalize_to_bbob_space()` | ga\_solver.py | 将 RAG 参数归一化到 \[-5.12, 5.12\] |
| `rastrigin()` | ga\_solver.py | 成本函数：模拟 RAG 综合损失 |
| `constraint_penalty()` | ga\_solver.py | 惩罚函数：对超出上下文预算的配置施加平方惩罚 |
| `evaluate_rag_config()` | ga\_solver.py | fit 函数：成本 + 惩罚 = 最终评价分数 |
| `tournament_selection()` | ga\_solver.py | 锦标赛选择算子 |
| `arithmetic_crossover()` | ga\_solver.py | 算术交叉算子 |
| `gaussian_mutation()` | ga\_solver.py | 高斯变异算子 |
| `elitist_update()` | ga\_solver.py | 精英保留策略 |
| `run_random_search()` | ga\_solver.py | 随机搜索算法 |
| `run_ga()` | ga\_solver.py | 标准遗传算法 |
| `run_llm_ga()` | ga\_solver.py | LLM 辅助的遗传算法 |
| `build_llm_prompt()` | llm\_module.py | 构造发给 LLM 的提示词 |
| `call_spark_generate()` | llm\_module.py | 通过 OpenAI 兼容接口调用 LLM API |
| `summarize_search_state()` | ga\_solver.py | 汇总种群搜索状态（适应度统计、收敛趋势、多样性、维度分布） |
| `format_search_state()` | llm\_module.py | 将搜索状态格式化为 prompt 可读文本 |
| `real_llm_generate()` | llm\_module.py | LLM 候选解生成的完整流程（含校验与重试） |
| `plot_convergence()` | utils.py | 绘制收敛曲线对比图 |

## 附录 C：核心公式汇总

**归一化映射：**

```
z_i = -5.12 + (x_i - lower_i) / (upper_i - lower_i) × 10.24
```

**Rastrigin 函数：**

```
f(z) = 10 × D + Σ(z_i² - 10 × cos(2π × z_i))
```

**惩罚函数：**

```
excess_ratio = max(0, (rerank_top_n × chunk_size - max_context_tokens) / max_context_tokens)
penalty = 100 × excess_ratio²
```

**fit 函数：**

```
fitness(x) = rastrigin(normalize(repair(x))) + constraint_penalty(repair(x))
```

**算术交叉：**

```
child = α × parent1 + (1 - α) × parent2,  α ~ U(0, 1)
```

**高斯变异：**

```
mutated[i] += N(0, (upper_i - lower_i) × 0.08),  以 20% 概率触发
```
