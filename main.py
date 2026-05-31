import numpy as np

# 从 config.py 导入所有实验参数（参数集中管理，改参数只需改 config.py）
from config import (
    CROSSOVER_RATE,
    DIM,
    LLM_INTERVAL,
    LLM_NUM_CANDIDATES,
    MAX_FES,
    MUTATION_RATE,
    MUTATION_SIGMA_RATIO,
    NUM_RUNS,
    POP_SIZE,
    RANDOM_SEED,
    RESULT_DIR,
    TOURNAMENT_SIZE,
)
from ga_solver import (
    decode_rag_config,
    evaluate_rag_config,
    get_lower_bounds,
    get_upper_bounds,
    run_ga,
    run_llm_ga,
    run_random_search,
)
from utils import (
    ensure_dir,
    plot_convergence,
    save_results_csv,
    save_summary_csv,
    set_seed,
)


def build_result_row(run: int, seed: int, algorithm: str, result: dict) -> dict:
    """把单次算法运行结果整理成 CSV 的一行：包含运行编号、种子、算法名、最优值、RAG 配置。"""
    config = decode_rag_config(result["best_x"])
    row = {
        "run": run,
        "seed": seed,
        "algorithm": algorithm,
        "best_f": result["best_f"],        # 最优 fitness 值（越小越好）
        "evaluations": result["evaluations"],  # 实际消耗的函数评价次数
    }
    row.update(config)  # 展开 RAG 配置的 6 个参数
    return row


def main() -> None:
    """
    实验主流程：

    1. 创建结果输出目录 results/
    2. 对每个随机种子（共 NUM_RUNS 轮）分别运行三种算法：
       - Random Search（随机搜索基线）
       - GA（标准遗传算法）
       - LLM-GA（大模型辅助的遗传算法）
    3. 收集每轮的最优结果和收敛历史
    4. 输出三个文件：
       - result.csv：每轮每种算法的最优配置和最优值
       - summary.csv：三种算法的均值/标准差/最值汇总
       - convergence.png：三种算法的平均收敛曲线对比图
    """
    ensure_dir(RESULT_DIR)
    lower_bound = get_lower_bounds()
    upper_bound = get_upper_bounds()

    all_results = []           # 收集所有运行结果，用于生成 CSV
    random_histories = []      # Random Search 的收敛曲线集合
    ga_histories = []          # GA 的收敛曲线集合
    llm_ga_histories = []      # LLM-GA 的收敛曲线集合

    for run in range(NUM_RUNS):
        seed = RANDOM_SEED + run  # 每轮使用不同的种子，保证独立性
        set_seed(seed)
        print(f"Run {run + 1}/{NUM_RUNS}, seed = {seed}")

        # --- 运行随机搜索（基线）---
        random_result = run_random_search(
            objective_func=evaluate_rag_config,
            dim=DIM,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            max_fes=MAX_FES,
        )

        # --- 运行标准 GA ---
        ga_result = run_ga(
            objective_func=evaluate_rag_config,
            dim=DIM,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            max_fes=MAX_FES,
            pop_size=POP_SIZE,
            tournament_size=TOURNAMENT_SIZE,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            mutation_sigma_ratio=MUTATION_SIGMA_RATIO,
        )

        # --- 运行 LLM-GA（调用讯飞星火 API）---
        llm_ga_result = run_llm_ga(
            objective_func=evaluate_rag_config,
            dim=DIM,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            max_fes=MAX_FES,
            pop_size=POP_SIZE,
            tournament_size=TOURNAMENT_SIZE,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            mutation_sigma_ratio=MUTATION_SIGMA_RATIO,
            llm_interval=LLM_INTERVAL,
            llm_num_candidates=LLM_NUM_CANDIDATES,
        )

        # 收集收敛历史（用于画图）
        random_histories.append(random_result["history"])
        ga_histories.append(ga_result["history"])
        llm_ga_histories.append(llm_ga_result["history"])

        # 收集结果行
        all_results.append(build_result_row(run + 1, seed, "Random Search", random_result))
        all_results.append(build_result_row(run + 1, seed, "GA", ga_result))
        all_results.append(build_result_row(run + 1, seed, "LLM-GA", llm_ga_result))

        print(f"  Random Search best: {random_result['best_f']:.6f}")
        print(f"  GA best:            {ga_result['best_f']:.6f}")
        print(f"  LLM-GA best:        {llm_ga_result['best_f']:.6f}")
        print("-" * 50)

    # --- 保存结果 ---
    result_csv_path = f"{RESULT_DIR}/result.csv"
    summary_csv_path = f"{RESULT_DIR}/summary.csv"
    convergence_path = f"{RESULT_DIR}/convergence.png"

    save_results_csv(all_results, result_csv_path)
    save_summary_csv(all_results, summary_csv_path)

    # 画收敛曲线：对 NUM_RUNS 轮的 history 取平均，画三条对比曲线
    plot_convergence(
        histories=[
            np.mean(np.array(random_histories), axis=0),
            np.mean(np.array(ga_histories), axis=0),
            np.mean(np.array(llm_ga_histories), axis=0),
        ],
        labels=["Random Search", "GA", "LLM-GA"],
        save_path=convergence_path,
    )

    print("Experiment finished.")
    print(f"Results saved to: {result_csv_path}")
    print(f"Summary saved to: {summary_csv_path}")
    print(f"Convergence figure saved to: {convergence_path}")


if __name__ == "__main__":
    main()
