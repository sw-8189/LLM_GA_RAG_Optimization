import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    """
    固定随机种子，保证实验可复现。

    同时固定 Python 内置 random 和 NumPy 的随机状态，
    确保同一种子下 GA 的随机操作（选择、交叉、变异）完全一致。
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """如果目录不存在则自动创建，避免保存文件时报错。"""
    if not os.path.exists(path):
        os.makedirs(path)


def save_results_csv(results: list[dict], save_path: str) -> None:
    """
    保存详细结果到 CSV。每行 = 一次运行 + 一种算法的最优结果。
    包含：run、seed、algorithm、best_f、evaluations、以及 6 个 RAG 参数。
    """
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def save_summary_csv(results: list[dict], save_path: str) -> None:
    """
    按算法汇总统计量：best_f 的均值、标准差、最小值、最大值。
    多轮运行后，这张表用于判断三种算法的总体表现差异。
    """
    df = pd.DataFrame(results)
    summary = (
        df.groupby("algorithm")["best_f"]
        .agg(mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary.to_csv(save_path, index=False, encoding="utf-8-sig")


def plot_convergence(histories: list[np.ndarray], labels: list[str], save_path: str) -> None:
    """
    绘制收敛曲线对比图。

    横轴：函数评价次数（即消耗的"预算"）
    纵轴：当前找到的最优 fitness 值（越小越好）
    三条线分别代表 Random Search、GA、LLM-GA 的平均表现。
    曲线下降越快、最终值越低，说明算法效率越高。
    """
    plt.figure(figsize=(8, 5))

    for history, label in zip(histories, labels):
        plt.plot(history, label=label)

    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Objective Value")
    plt.title("Convergence Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
