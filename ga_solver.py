from typing import Callable

import numpy as np

from config import (
    INTEGER_INDICES,
    RAG_BOUNDS,
    RAG_VARIABLE_NAMES,
    RECENT_GENERATION_WINDOW,
)
from llm_module import real_llm_generate

# 类型别名：目标函数的签名，输入一个 6 维向量，返回一个标量 fitness 值
FitnessFunction = Callable[[np.ndarray], float]


def get_lower_bounds() -> np.ndarray:
    """返回所有变量的下界，即 RAG_BOUNDS 第 0 列。"""
    return RAG_BOUNDS[:, 0]


def get_upper_bounds() -> np.ndarray:
    """返回所有变量的上界，即 RAG_BOUNDS 第 1 列。"""
    return RAG_BOUNDS[:, 1]


def repair_rag_config(x: np.ndarray) -> np.ndarray:
    """
    将任意候选解修复成合法 RAG 配置（边界效应处理的核心函数）。

    修复步骤：
    1. np.clip 裁剪：保证每个维度不越过上下界，越界值被拉回到边界；
    2. 整数取整：对 chunk_size、retrieval_top_k、rerank_top_n、max_context_tokens 做 round()；
    3. 逻辑约束修复：rerank_top_n（精排保留数）不能大于 retrieval_top_k（召回数），
       否则精排数量比召回还多，逻辑不通，此时强制 rerank_top_n = retrieval_top_k。

    注意：边界值本身是合法解（如 chunk_size=200 或 1000），不会被额外惩罚。
    """
    repaired = np.asarray(x, dtype=float).copy()

    # 第一步：硬裁剪到合法范围
    repaired = np.clip(repaired, get_lower_bounds(), get_upper_bounds())

    # 第二步：整数维度做四舍五入
    for idx in INTEGER_INDICES:
        repaired[idx] = int(round(repaired[idx]))

    # 第三步：rerank_top_n（索引4）不能超过 retrieval_top_k（索引2）
    if repaired[4] > repaired[2]:
        repaired[4] = repaired[2]

    # 兜底：rerank_top_n 至少为 1
    if repaired[4] < 1:
        repaired[4] = 1

    return repaired


def decode_rag_config(x: np.ndarray) -> dict:
    """把 6 维向量转成带语义名称的字典，方便输出和保存。"""
    repaired = repair_rag_config(x)
    chunk_size = int(repaired[0])
    chunk_overlap_ratio = float(repaired[1])

    return {
        "chunk_size": chunk_size,
        "chunk_overlap_ratio": chunk_overlap_ratio,
        "chunk_overlap": int(round(chunk_size * chunk_overlap_ratio)),
        "retrieval_top_k": int(repaired[2]),
        "similarity_threshold": float(repaired[3]),
        "rerank_top_n": int(repaired[4]),
        "max_context_tokens": int(repaired[5]),
        "temperature": 0.2,
    }


def rastrigin(z: np.ndarray) -> float:
    """
    Rastrigin 测试函数（经典多峰黑箱函数）。

    公式：f(z) = 10*D + Σ(z_i² - 10*cos(2π*z_i))
    特点：大量局部最优，全局最优在原点 f(0)=0。
    用途：模拟真实 RAG 评价中"成本高、结构未知"的综合损失。
    """
    z = np.asarray(z, dtype=float)
    dim = len(z)
    return float(10 * dim + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z)))


def normalize_to_bbob_space(x: np.ndarray) -> np.ndarray:
    """
    将 RAG 参数线性映射到 [-5.12, 5.12]^D 统一量纲。

    因为 6 个 RAG 参数的取值范围差异很大（chunk_size 在 [200,1000]，
    overlap_ratio 在 [0,0.3]），需要归一化后才能交给同一个 Rastrigin 函数评价。
    映射公式：-5.12 + (x - lower) / (upper - lower) * 10.24
    """
    repaired = repair_rag_config(x)
    lower = get_lower_bounds()
    upper = get_upper_bounds()
    return -5.12 + (repaired - lower) / (upper - lower) * 10.24


def constraint_penalty(x: np.ndarray) -> float:
    """
    上下文预算惩罚函数（软约束）。

    约束条件：rerank_top_n × chunk_size ≤ max_context_tokens
    含义：最终给大模型的文本总量不应超过上下文窗口预算。

    惩罚策略：
    - 满足约束 → penalty = 0
    - 违反约束 → penalty = 100 × (超出比例)²
      用平方是为了让轻微超出的惩罚较小，严重超出时惩罚急剧增大。
    """
    cfg = decode_rag_config(x)
    estimated_context = cfg["rerank_top_n"] * cfg["chunk_size"]
    max_context = cfg["max_context_tokens"]

    if estimated_context <= max_context:
        return 0.0

    excess_ratio = (estimated_context - max_context) / max_context
    return float(100.0 * excess_ratio ** 2)


def evaluate_rag_config(x: np.ndarray) -> float:
    """
    最终 fitness 函数（越小越好）。

    fitness(x) = Rastrigin(归一化x) + constraint_penalty(x)
    - Rastrigin 项：模拟 RAG 配置的综合损失
    - penalty 项：惩罚超出上下文预算的配置
    """
    repaired = repair_rag_config(x)
    base_loss = rastrigin(normalize_to_bbob_space(repaired))
    penalty = constraint_penalty(repaired)
    return float(base_loss + penalty)


def build_result(best_x: np.ndarray, best_f: float, history: list[float]) -> dict:
    """统一封装算法返回值。history 长度 = 实际函数评价次数。"""
    return {
        "best_x": best_x,
        "best_f": float(best_f),
        "history": [float(value) for value in history],
        "evaluations": len(history),
    }


def initialize_population(
    pop_size: int,
    dim: int,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    """在变量边界内均匀随机初始化种群，并逐个修复为合法配置。"""
    population = np.random.uniform(lower_bound, upper_bound, size=(pop_size, dim))
    return np.array([repair_rag_config(individual) for individual in population])


def evaluate_population(
    population: np.ndarray,
    objective_func: FitnessFunction,
) -> np.ndarray:
    """逐个评价种群中每个个体的 fitness。每次调用计为一次函数评价（消耗预算）。"""
    return np.array([objective_func(individual) for individual in population])


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
) -> np.ndarray:
    """
    锦标赛选择：从种群中随机抽取 tournament_size 个个体，
    返回其中 fitness 最小（最优）的那个。
    tournament_size 越大，选择压力越大，收敛越快但多样性下降越快。
    """
    selected_indices = np.random.choice(
        len(population),
        size=tournament_size,
        replace=False,
    )
    best_index = selected_indices[np.argmin(fitness[selected_indices])]
    return population[best_index].copy()


def arithmetic_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float,
) -> np.ndarray:
    """
    算术交叉：以 crossover_rate 的概率在两个父代之间做线性插值。
    child = α × parent1 + (1-α) × parent2，α ∈ [0,1] 随机。
    如果不交叉（概率 1-crossover_rate），直接返回 parent1 的副本。
    """
    if np.random.rand() >= crossover_rate:
        return parent1.copy()

    alpha = np.random.rand()
    return alpha * parent1 + (1 - alpha) * parent2


def gaussian_mutation(
    child: np.ndarray,
    mutation_rate: float,
    mutation_sigma: np.ndarray,
) -> np.ndarray:
    """
    高斯变异：对每个维度，以 mutation_rate 的概率加入正态扰动 N(0, sigma_i)。
    sigma_i = 变量范围 × MUTATION_SIGMA_RATIO，因此每个维度的扰动幅度与其取值范围成比例。
    """
    mutated = child.copy()
    for idx in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[idx] += np.random.normal(0, mutation_sigma[idx])
    return mutated


def elitist_update(
    population: np.ndarray,
    fitness: np.ndarray,
    offspring: np.ndarray,
    offspring_fitness: np.ndarray,
    pop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    精英保留策略：将父代种群和子代合并，按 fitness 排序后只保留最优的 pop_size 个。
    保证每一代都不会比上一代差（不退化），但也可能压制多样性。
    """
    combined_population = np.vstack([population, offspring])
    combined_fitness = np.concatenate([fitness, offspring_fitness])
    sorted_indices = np.argsort(combined_fitness)
    elite_indices = sorted_indices[:pop_size]
    return combined_population[elite_indices], combined_fitness[elite_indices]


def get_top_solutions(
    population: np.ndarray,
    fitness: np.ndarray,
    max_count: int = 5,
) -> list[tuple[np.ndarray, float]]:
    """取当前种群中 fitness 最小的 max_count 个个体，作为 LLM prompt 的上下文信息。"""
    top_indices = np.argsort(fitness)[: min(max_count, len(population))]
    return [(population[idx].copy(), float(fitness[idx])) for idx in top_indices]


def summarize_search_state(
    population: np.ndarray,
    fitness: np.ndarray,
    history: list[float],
    generation: int,
    generation_best_history: list[float],
    recent_window: int = RECENT_GENERATION_WINDOW,
) -> dict:
    """
    汇总当前搜索状态，供 LLM 判断“继续局部开发”还是“增加探索”。

    这部分信息不参与 fitness 计算，只作为 prompt 上下文：
    - population fitness 的均值/标准差/最小值/最大值；
    - 最近 recent_window 代内 best_f 是否还在下降；
    - 每个变量在当前种群中的均值、标准差、最小值、最大值和覆盖比例。
    """
    population = np.asarray(population, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    lower = get_lower_bounds()
    upper = get_upper_bounds()
    span = upper - lower

    recent_values = (
        generation_best_history[-recent_window:]
        if len(generation_best_history) >= recent_window
        else generation_best_history[:]
    )
    if len(recent_values) >= 2:
        recent_start = float(recent_values[0])
        recent_end = float(recent_values[-1])
        recent_improvement = recent_start - recent_end  # 最小化问题：正数表示下降
    else:
        recent_start = float(history[0]) if history else None
        recent_end = float(history[-1]) if history else None
        recent_improvement = 0.0

    improvement_threshold = 1e-6
    trend = "improving" if recent_improvement > improvement_threshold else "stagnating"

    dimension_stats = []
    for idx, name in enumerate(RAG_VARIABLE_NAMES):
        values = population[:, idx]
        coverage_ratio = 0.0
        if span[idx] > 0:
            coverage_ratio = float((np.max(values) - np.min(values)) / span[idx])

        dimension_stats.append(
            {
                "name": name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "coverage_ratio": coverage_ratio,
            }
        )

    normalized_std = np.std(population, axis=0) / span

    return {
        "generation": generation,
        "evaluations": len(history),
        "fitness": {
            "best": float(np.min(fitness)),
            "mean": float(np.mean(fitness)),
            "std": float(np.std(fitness)),
            "worst": float(np.max(fitness)),
        },
        "recent_trend": {
            "window_generations": len(recent_values),
            "start_best": recent_start,
            "end_best": recent_end,
            "improvement": float(recent_improvement),
            "status": trend,
        },
        "diversity": {
            "mean_normalized_std": float(np.mean(normalized_std)),
            "min_normalized_std": float(np.min(normalized_std)),
            "max_normalized_std": float(np.max(normalized_std)),
        },
        "dimensions": dimension_stats,
    }


def run_random_search(
    objective_func: FitnessFunction,
    dim: int,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    max_fes: int,
) -> dict:
    """
    随机搜索基线：每次在边界内均匀随机采样一个配置，修复后评价。
    没有任何"学习"过程，纯粹靠运气，用于对比 GA/LLM-GA 是否真的有效。
    """
    best_x = None
    best_f = float("inf")
    history = []

    for _ in range(max_fes):
        candidate = np.random.uniform(lower_bound, upper_bound, size=dim)
        candidate = repair_rag_config(candidate)
        candidate_f = objective_func(candidate)

        if candidate_f < best_f:
            best_f = candidate_f
            best_x = candidate.copy()

        history.append(best_f)

    return build_result(best_x, best_f, history)


def run_ga(
    objective_func: FitnessFunction,
    dim: int,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    max_fes: int,
    pop_size: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_sigma_ratio: float,
) -> dict:
    """
    标准实数编码遗传算法。

    流程：初始化种群 → 评价 → 进入循环（选择→交叉→变异→修复→评价→精英保留）
    直到函数评价次数耗尽（达到 max_fes）。
    """
    if pop_size > max_fes:
        raise ValueError("pop_size must not be larger than max_fes.")

    # 初始化：随机生成 pop_size 个合法个体，计算各自的 fitness
    population = initialize_population(pop_size, dim, lower_bound, upper_bound)
    fitness = evaluate_population(population, objective_func)
    # 变异标准差 = 变量范围 × 比例系数，让扰动幅度与各维度的量纲匹配
    mutation_sigma = mutation_sigma_ratio * (upper_bound - lower_bound)

    fes = pop_size  # 已消耗的函数评价次数
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]
    history = [best_f] * fes  # 收敛曲线：每次评价后记录当前最优值

    while fes < max_fes:
        offspring = []
        offspring_fitness = []

        for _ in range(pop_size):
            if fes >= max_fes:
                break

            # GA 核心步骤：选择 → 交叉 → 变异 → 修复 → 评价
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)
            child = arithmetic_crossover(parent1, parent2, crossover_rate)
            child = gaussian_mutation(child, mutation_rate, mutation_sigma)
            child = repair_rag_config(child)  # 修复确保子代合法

            child_f = objective_func(child)
            fes += 1

            offspring.append(child)
            offspring_fitness.append(child_f)

            if child_f < best_f:
                best_f = child_f
                best_x = child.copy()

            history.append(best_f)

        # 精英保留：父代+子代合并，只留最优的 pop_size 个
        if offspring:
            population, fitness = elitist_update(
                population,
                fitness,
                np.array(offspring),
                np.array(offspring_fitness),
                pop_size,
            )

    return build_result(best_x, best_f, history)


def run_llm_ga(
    objective_func: FitnessFunction,
    dim: int,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    max_fes: int,
    pop_size: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_sigma_ratio: float,
    llm_interval: int,
    llm_num_candidates: int,
) -> dict:
    """
    LLM-GA：以 GA 为主搜索器，每隔 llm_interval 代调用真实 LLM 生成候选解。

    与标准 GA 的唯一区别：在每 llm_interval 代结束后，
    将当前最优解和 Top 解发给 LLM，让 LLM 提出新的候选配置。
    LLM 候选解仍需经过 objective_func 评价，计入 max_fes 预算。
    如果候选解优于种群中最差个体，则替换它。
    """
    if pop_size > max_fes:
        raise ValueError("pop_size must not be larger than max_fes.")

    population = initialize_population(pop_size, dim, lower_bound, upper_bound)
    fitness = evaluate_population(population, objective_func)
    mutation_sigma = mutation_sigma_ratio * (upper_bound - lower_bound)

    fes = pop_size
    generation = 0
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]
    history = [best_f] * fes
    generation_best_history = [float(best_f)]

    while fes < max_fes:
        generation += 1
        offspring = []
        offspring_fitness = []

        # --- GA 常规进化（与 run_ga 相同）---
        for _ in range(pop_size):
            if fes >= max_fes:
                break

            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)
            child = arithmetic_crossover(parent1, parent2, crossover_rate)
            child = gaussian_mutation(child, mutation_rate, mutation_sigma)
            child = repair_rag_config(child)

            child_f = objective_func(child)
            fes += 1

            offspring.append(child)
            offspring_fitness.append(child_f)

            if child_f < best_f:
                best_f = child_f
                best_x = child.copy()

            history.append(best_f)

        if offspring:
            population, fitness = elitist_update(
                population,
                fitness,
                np.array(offspring),
                np.array(offspring_fitness),
                pop_size,
            )

        generation_best_history.append(float(best_f))

        # --- LLM 辅助搜索（LLM-GA 的核心创新点）---
        # 每隔 llm_interval 代，用当前搜索状态"请教" LLM 生成新候选
        if generation % llm_interval == 0 and fes < max_fes:
            # 提取当前种群中最优的几个解，并补充整体搜索状态作为 LLM 上下文。
            # 如果 LLM 多次返回非法内容，real_llm_generate 会抛出清晰异常并终止实验。
            top_solutions = get_top_solutions(population, fitness)
            search_state = summarize_search_state(
                population=population,
                fitness=fitness,
                history=history,
                generation=generation,
                generation_best_history=generation_best_history,
            )
            try:
                candidates = real_llm_generate(
                    best_x=best_x,
                    best_f=float(best_f),
                    top_solutions=top_solutions,
                    num_candidates=llm_num_candidates,
                    search_state=search_state,
                )
            except Exception as exc:
                print(f"  [LLM skipped] gen {generation}: {exc}")
                candidates = np.empty((0, dim))

            # 对 LLM 返回的每个候选解：修复→评价→尝试替换种群中的最差个体
            for candidate in candidates:
                if fes >= max_fes:
                    break

                candidate = repair_rag_config(candidate)
                candidate_f = objective_func(candidate)
                fes += 1

                # 替换策略：如果 LLM 候选比当前最差个体好，就替换它
                worst_idx = np.argmax(fitness)
                if candidate_f < fitness[worst_idx]:
                    population[worst_idx] = candidate
                    fitness[worst_idx] = candidate_f

                # 同时更新全局最优
                if candidate_f < best_f:
                    best_f = candidate_f
                    best_x = candidate.copy()

                history.append(best_f)

    return build_result(best_x, best_f, history)
