import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from config import MAX_LLM_RETRIES


def format_rag_config(x: np.ndarray) -> dict:
    """把 6 维候选向量格式化为字典，用于嵌入 LLM 提示词中。"""
    values = np.asarray(x, dtype=float)
    return {
        "chunk_size": int(round(values[0])),
        "chunk_overlap_ratio": float(values[1]),
        "retrieval_top_k": int(round(values[2])),
        "similarity_threshold": float(values[3]),
        "rerank_top_n": int(round(values[4])),
        "max_context_tokens": int(round(values[5])),
        "temperature": 0.2,
    }


def build_llm_prompt(
    best_x: np.ndarray,
    best_f: float,
    top_solutions: list[tuple[np.ndarray, float]],
    num_candidates: int,
    search_state: dict | None = None,
    retry_note: str = "",
) -> str:
    """
    构造发给 LLM 的提示词（prompt）。

    提示词包含：
    - 6 个 RAG 变量的名称、类型、取值范围
    - 关键约束说明（rerank_top_n ≤ retrieval_top_k 等）
    - 当前最优解和它的 fitness 值
    - 当前种群中最好的几个解（给 LLM 参考趋势）
    - 如果是重试，附加"上次返回非法"的警告
    """
    best_cfg = format_rag_config(best_x)
    top_text = "\n".join(
        f"{idx}. config={format_rag_config(x)}, objective={fitness:.6f}"
        for idx, (x, fitness) in enumerate(top_solutions, start=1)
    )
    state_text = format_search_state(search_state)

    return f"""
You are assisting a genetic algorithm for expensive black-box optimization.

Application background:
We are tuning a simplified RAG question-answering system configuration.

Each candidate solution has 6 variables:

1. chunk_size
   - integer
   - range: [200, 1000]

2. chunk_overlap_ratio
   - float
   - range: [0.0, 0.3]

3. retrieval_top_k
   - integer
   - range: [3, 20]

4. similarity_threshold
   - float
   - range: [0.2, 0.8]

5. rerank_top_n
   - integer
   - range: [1, 8]

6. max_context_tokens
   - integer
   - range: [1000, 4000]

Important constraints:
- rerank_top_n must be <= retrieval_top_k.
- rerank_top_n * chunk_size should not be much larger than max_context_tokens.
- Balanced RAG configurations are preferred.
- Avoid extreme boundary values unless they are necessary.

Optimization objective:
Minimize the objective value.
Lower objective value is better.

Current best configuration:
{best_cfg}

Current best objective value:
{best_f:.6f}

Top solutions found so far:
{top_text}

Search state summary:
{state_text}

Generation strategy:
- If recent_trend.status is "improving", prefer local exploitation around good configurations.
- If recent_trend.status is "stagnating", generate more exploratory but still valid candidates.
- Use dimension statistics to avoid repeatedly sampling dimensions that have already collapsed too tightly.
- Keep candidates diverse from each other; do not simply copy the current best configuration.

{retry_note}

Please generate {num_candidates} new candidate configurations.

Requirements:
1. Return only valid JSON.
2. Do not explain.
3. Do not use markdown.
4. Each candidate must have exactly 6 values.
5. Keep candidates diverse but still close to good configurations.

Output format:
{{
  "candidates": [
    [chunk_size, chunk_overlap_ratio, retrieval_top_k, similarity_threshold, rerank_top_n, max_context_tokens]
  ]
}}
"""


def format_search_state(search_state: dict | None) -> str:
    """把 GA 搜索状态压缩成 prompt 中易读的文本。"""
    if not search_state:
        return "No population-level search state is available."

    fitness = search_state["fitness"]
    trend = search_state["recent_trend"]
    diversity = search_state["diversity"]

    lines = [
        f"- generation: {search_state['generation']}",
        f"- evaluations_so_far: {search_state['evaluations']}",
        (
            "- population_fitness: "
            f"best={fitness['best']:.6f}, mean={fitness['mean']:.6f}, "
            f"std={fitness['std']:.6f}, worst={fitness['worst']:.6f}"
        ),
        (
            "- recent_trend: "
            f"window_generations={trend['window_generations']}, "
            f"start_best={trend['start_best']:.6f}, "
            f"end_best={trend['end_best']:.6f}, "
            f"improvement={trend['improvement']:.6f}, status={trend['status']}"
        ),
        (
            "- population_diversity: "
            f"mean_normalized_std={diversity['mean_normalized_std']:.6f}, "
            f"min_normalized_std={diversity['min_normalized_std']:.6f}, "
            f"max_normalized_std={diversity['max_normalized_std']:.6f}"
        ),
        "- dimension_distribution:",
    ]

    for item in search_state["dimensions"]:
        lines.append(
            "  "
            f"{item['name']}: mean={item['mean']:.6f}, std={item['std']:.6f}, "
            f"min={item['min']:.6f}, max={item['max']:.6f}, "
            f"coverage_ratio={item['coverage_ratio']:.6f}"
        )

    return "\n".join(lines)


def extract_json_from_text(text: str) -> dict:
    """
    从 LLM 返回的文本中提取 JSON 对象。

    处理逻辑：去掉可能存在的 ```json 代码块标记，然后用正则提取最外层 {} 包裹的 JSON。
    解析失败会直接抛异常，不回退到 Mock 数据。
    """
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in LLM output.")

    return json.loads(match.group(0))


def call_spark_generate(prompt: str) -> str:
    """
    通过 OpenAI 兼容接口调用讯飞星火 API。

    从 .env 文件读取：SPARK_API_KEY、SPARK_BASE_URL、SPARK_MODEL。
    temperature=0.2：低随机性，让 LLM 输出更稳定、更倾向于利用已有信息。
    max_tokens=800：限制输出长度，避免 LLM 生成过多无关内容。
    """
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    api_key = os.getenv("SPARK_API_KEY")
    base_url = os.getenv("SPARK_BASE_URL", "https://spark-api-open.xf-yun.com/x2")
    model = os.getenv("SPARK_MODEL")

    if not api_key:
        raise ValueError("SPARK_API_KEY is not set in .env")

    if not model:
        raise ValueError("SPARK_MODEL is not set in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                # system 指令：限定 LLM 只输出 JSON，不要解释
                "content": "You are a JSON-only optimization assistant. Always output valid JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=800,
    )

    return response.choices[0].message.content


def real_llm_generate(
    best_x: np.ndarray,
    best_f: float,
    top_solutions: list[tuple[np.ndarray, float]],
    num_candidates: int,
    search_state: dict | None = None,
) -> np.ndarray:
    """
    调用真实 LLM 生成候选 RAG 配置的完整流程。

    步骤：
    1. 构造 prompt（包含当前最优解、Top 解、约束说明）
    2. 调用讯飞星火 API
    3. 解析返回的 JSON，提取 candidates 数组
    4. 严格校验每个候选解：必须是长度为 6 的数值列表
    5. 最多重试 MAX_LLM_RETRIES 次，全部失败则抛异常终止

    返回值：shape 为 (num_candidates, 6) 的 numpy 数组。
    """
    last_error = None
    last_text = ""

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        # 重试时在 prompt 中追加警告，提醒 LLM 上次输出不合法
        retry_note = ""
        if attempt > 1:
            retry_note = (
                "The previous response was invalid. Return only a JSON object "
                "with key \"candidates\" and no extra text."
            )

        prompt = build_llm_prompt(
            best_x=best_x,
            best_f=best_f,
            top_solutions=top_solutions,
            num_candidates=num_candidates,
            search_state=search_state,
            retry_note=retry_note,
        )

        try:
            last_text = call_spark_generate(prompt)
            data = extract_json_from_text(last_text)
            raw_candidates = data.get("candidates")

            # 校验：必须包含 "candidates" 字段且为列表
            if not isinstance(raw_candidates, list):
                raise ValueError(
                    "LLM output must contain a list field named 'candidates'."
                )

            # 校验：候选数量必须足够
            if len(raw_candidates) < num_candidates:
                raise ValueError(
                    f"LLM returned {len(raw_candidates)} candidates, "
                    f"but {num_candidates} are required."
                )

            # 校验：每个候选必须是 6 个数值组成的列表
            candidates = []
            for idx, item in enumerate(raw_candidates[:num_candidates], start=1):
                if not isinstance(item, list) or len(item) != 6:
                    raise ValueError(
                        f"Candidate {idx} must be a list with exactly 6 numeric values."
                    )

                try:
                    candidate = np.array(item, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Candidate {idx} contains non-numeric values."
                    ) from exc

                candidates.append(candidate)

            return np.array(candidates, dtype=float)

        except Exception as exc:
            last_error = exc
            if attempt == MAX_LLM_RETRIES:
                # 所有重试用尽，抛出详细错误信息便于调试
                snippet = (last_text or "").strip().replace("\n", " ")[:300]
                raise RuntimeError(
                    "Real LLM failed to return valid candidate JSON after "
                    f"{MAX_LLM_RETRIES} attempts. Last error: {last_error}. "
                    f"Last output snippet: {snippet}"
                ) from exc

            print(
                "[Warning] Real LLM returned invalid output; "
                f"retrying ({attempt}/{MAX_LLM_RETRIES}). Error: {exc}"
            )

    raise RuntimeError("Unexpected LLM generation failure.")
