"""
项目参数集中管理文件。

所有可调参数都在这里定义，其他文件从这里导入。
修改实验配置只需改这一个文件，避免参数散落在各处导致不一致。
"""

import numpy as np

# 实验参数

DIM = 6                    # 优化问题维度：6 个 RAG 参数
MAX_FES = 300              # 最大函数评价次数（模拟昂贵优化的预算限制）
POP_SIZE = 20              # 种群大小：每一代有 20 个候选解
NUM_RUNS = 30              # 独立运行次数：多轮取平均，用于统计显著性检验
RANDOM_SEED = 42           # 随机种子起始值：每轮种子 = 42 + run_index
RESULT_DIR = "results"     # 结果输出目录

# GA 算子参数

TOURNAMENT_SIZE = 3        # 锦标赛选择的竞争者数量（从随机 3 个中选最优）
CROSSOVER_RATE = 0.9       # 交叉概率：90% 进行交叉，10% 直接复制父代
MUTATION_RATE = 0.2        # 变异概率：每个维度有 20% 概率被扰动
MUTATION_SIGMA_RATIO = 0.08  # 变异幅度比例：扰动标准差 = 变量范围 × 0.08

# LLM 辅助参数

LLM_INTERVAL = 3           # 每隔多少代调用一次 LLM（太频繁浪费预算，太少辅助不足）
LLM_NUM_CANDIDATES = 3     # LLM 每次生成的候选解数量
MAX_LLM_RETRIES = 3        # LLM 返回非法内容时的最大重试次数
RECENT_GENERATION_WINDOW = 5  # 传给 LLM 的近期收敛窗口：观察最近 5 代 best_f 是否还在下降

# RAG 参数空间（6 个变量的取值范围）

# 候选解向量 x 的含义：
#   x[0] = chunk_size            文本分片大小（字符数）
#   x[1] = chunk_overlap_ratio   相邻分片的重叠比例
#   x[2] = retrieval_top_k       第一步检索召回的候选文档数量
#   x[3] = similarity_threshold  相似度过滤阈值，低于此值的文档被丢弃
#   x[4] = rerank_top_n          经过精排后最终保留的文档数量
#   x[5] = max_context_tokens    喂给大模型的上下文总 token 数上限

RAG_BOUNDS = np.array(
    [
        [200, 1000],   # chunk_size: 分片太小丢失上下文，太大信息冗余
        [0.0, 0.3],    # chunk_overlap_ratio: 重叠太少断裂句意，太多浪费空间
        [3, 20],       # retrieval_top_k: 召回太少漏检，太多引入噪声
        [0.2, 0.8],    # similarity_threshold: 阈值太高召回少，太低噪声多
        [1, 8],        # rerank_top_n: 最终给大模型的文档数
        [1000, 4000],  # max_context_tokens: 上下文窗口预算
    ],
    dtype=float,
)

# 必须取整数的维度索引（chunk_size, retrieval_top_k, rerank_top_n, max_context_tokens）
# chunk_overlap_ratio（索引1）和 similarity_threshold（索引3）是连续浮点数
INTEGER_INDICES = (0, 2, 4, 5)

# 变量名称顺序必须与候选解向量 x 保持一致，用于结果输出和 LLM prompt 说明
RAG_VARIABLE_NAMES = (
    "chunk_size",
    "chunk_overlap_ratio",
    "retrieval_top_k",
    "similarity_threshold",
    "rerank_top_n",
    "max_context_tokens",
)
