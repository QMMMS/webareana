#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: LLM Rating Script (Random Baseline)
--------------------------------------------
功能描述:
    使用 LLM (GPT-5.1/4o) 对上一步生成的随机轨迹进行质量评估。
    评估维度包括：连贯性 (Coherence)、完成度 (Completion)、难度 (Difficulty)。

输入:
    - random_selected_trajs.json: 上一步生成的随机轨迹文件。
    - result_stage_1_explore_v2/add_local_intention_trajs: 原始轨迹详细数据目录。

输出:
    - rating_random_result.csv: 评分结果文件。
"""

import os
import csv
import json
import ast
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# LangChain 导入
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ChatMessage
except ImportError:
    print("请安装必要依赖: pip install langchain-openai langchain-core")
    exit(1)

# =============================================================================
# API & 环境配置 (严格保留原始内容)
# =============================================================================

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY'] = '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'
os.environ['LANGSMITH_PROJECT'] = 'pr-wooden-hedgehog-65'
os.environ['OPENAI_API_KEY'] = 'sk-g4MycviIFf8Lad0v5zVKPljDpueIDBLuqIC1nsDAiRnLZVKg'
os.environ['OPENAI_API_BASE'] = 'https://api2.aigcbest.top/v1'

# =============================================================================
# 文件路径配置
# =============================================================================

# 输入：上一步生成的随机轨迹文件
INPUT_JSON_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/advanced_selected_trajs_v2.json"

# 输入：原始轨迹文件夹 (用于获取页面详细描述)
ORIGIN_TRAJ_DIR = "/home/zjusst/qms/webarena/result_stage_1_explore_v2/add_local_intention_trajs"

# 输出：评分结果 CSV
OUTPUT_CSV_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/rating_advanced_result_v2.csv"

# =============================================================================
# Prompt 定义
# =============================================================================

SYSTEM_PROMPT = """You are an expert in evaluating GUI agent task trajectories. Your task is to assess the quality, effectiveness, and complexity of task trajectories for GUI manipulation tasks.

A trajectory consists of the following components:
1. High-level Instruction: Describes the user's overall intended task.
2. Interaction Sequence:
    - For all intermediate steps (Non-final turns):
        - Page Summary: A brief description of the webpage state.
        - Action: The specific operation executed (e.g., click, type).
        - Local Intent: The immediate purpose of that specific action.
    - For the final step (Last turn):
        - Full Accessibility Tree: The complete, simplified representation of the final webpage state.
        - Answer: The agent's final response (often in a Chain-of-Thought format) and the stop command indicating task completion.

When evaluating a trajectory, consider these key aspects:

### Evaluation Criteria:
1. Trajectory Coherence:
   - Do the low-level steps and corresponding actions follow a logical sequence toward the goal?
   - Are the actions clearly described and specific?
   - Are there redundant or unnecessary actions?

2. Task Completion:
   - Does the trajectory successfully achieve the instructed task?
   - Are all necessary interactions completed?
   - Are error cases handled appropriately?

3. Task Difficulty:
   - How many steps are required?
   - Does it involve cross-page navigation?
   - Is the instruction complex or abstract?
   - Did the agent have to perform error handling or self-correction?

### Scoring Guidelines:

#### 1. Trajectory Coherence (1-5)
- 5: The sequence is logically perfect and efficient. Every action clearly contributes to the goal with no redundancy.
- 4: The sequence is generally logical. There may be very minor inefficiencies or slightly unclear steps, but the overall flow is solid.
- 3: The sequence has some logical gaps or includes a few redundant actions (e.g., wandering briefly), but it eventually recovers or makes partial sense.
- 2: Significant inefficiencies or illogical detours. The agent wanders significantly before attempting the goal, or performs actions that contradict the intent.
- 1: The sequence is incoherent. It falls into an immediate deadlock, a repetitive loop, or actions are completely unrelated to the instruction.

#### 2. Task Completion (1-5)
- 5: The task is perfectly completed. All necessary interactions are done, and the final answer is correct and grounded in the evidence.
- 4: The task is mostly completed. The main goal is achieved, but there might be a minor detail missing in the final answer or a slight imperfection in the final state.
- 3: Partially completed. The agent performed some correct actions but failed to fully achieve the goal or the final answer is incorrect/incomplete.
- 2: Minimal progress. Only the initial setup or a few trivial actions were successful; the core task remains undone.
- 1: Complete failure. No meaningful progress was made toward the goal.

#### 3. Task Difficulty (1-5)
- 5 (Very Hard): Long horizon (>8 steps) requiring complex reasoning. Involves navigating multiple distinct pages, handling errors/popups, or synthesizing information from various sources.
- 4 (Hard): Moderate to long horizon (7-8 steps). Involves cross-page operations or a complex instruction that requires multi-step deduction.
- 3 (Medium): Moderate length (5-6 steps). Standard navigation and interaction tasks (e.g., searching and selecting an item) without significant obstacles.
- 2 (Easy): Short length (3-4 steps). Linear execution with minimal reasoning required (e.g., simple navigation).
- 1 (Very Easy): Trivial task (<3 steps). Completed on the same page or requires only a single direct action (e.g., "click the home button").

### Response Format:
You must output a JSON object with the following structure:
```json
{
  "reason": "<one line of simple reasoning process for the scores>",
  "score_coherence": <integer 1-5>,
  "score_completion": <integer 1-5>,
  "score_difficulty": <integer 1-5>
}"""

# =============================================================================
# 核心函数
# =============================================================================

def generate_from_openai_chat_completion_new(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    """
    使用 LangChain 的 ChatOpenAI 生成聊天回复，并支持 LangSmith 追踪。
    """
    # 1. 初始化 LangChain 的 ChatOpenAI 客户端
    llm = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE", "[https://api.openai.com/v1](https://api.openai.com/v1)"),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token] if stop_token else None,
        model_kwargs={},
        max_retries=3,
    )

    # 2. 将输入的字典列表转换为 LangChain 的消息对象
    langchain_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        name = msg.get("name")

        if role == "user":
            langchain_messages.append(HumanMessage(content=content, name=name))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content, name=name))
        elif role == "system":
            langchain_messages.append(SystemMessage(content=content, name=name))
        else:
            langchain_messages.append(ChatMessage(role=role, content=content))

    # 3. 调用 LLM 并获取结果
    try:
        response = llm.invoke(langchain_messages)
        return response.content
    except Exception as e:
        print(f"LLM 调用出错: {e}")
        return "{}"

def get_origin_traj_json(filename: str, index: int, folder_path: str = ORIGIN_TRAJ_DIR) -> Dict:
    """读取原始轨迹文件中的特定步骤数据"""
    full_path = os.path.join(folder_path, filename)
    
    # 路径容错处理
    if not os.path.exists(full_path):
        print(f"警告: 文件不存在 {full_path}")
        return {}

    with open(full_path, "r", encoding='utf-8') as f:
        origin_traj_json = json.load(f)
    return origin_traj_json[index]

def selected_trajectory_to_user_prompt(selected_dict: Dict) -> str:
    """
    构建发送给 LLM 的 User Prompt。
    适配逻辑：如果 random 轨迹中没有 Instruction/Answer，使用默认占位符。
    """
    # 适配随机轨迹可能没有指令和答案的情况
    instruction = selected_dict.get('Instruction', 'Random Exploration: Perform logical actions on the website.')
    answer = selected_dict.get('Answer', 'stop [Random exploration finished]')

    user_prompt = f"Instruction: {instruction}\n\n"

    selected_trajectory = selected_dict["selected_trajectory"]
    traj_len = len(selected_trajectory)
    
    # 处理中间步骤
    for i in range(traj_len - 1):
        try:
            # 兼容字符串形式的元组 "('filename', ...)" 或列表
            node_info = selected_trajectory[i][0]
            if isinstance(node_info, str):
                filename, index, _, _ = ast.literal_eval(node_info)
            else:
                filename, index, _, _ = node_info

            orgin_json = get_origin_traj_json(filename, index)
            if not orgin_json: continue

            page_description = orgin_json.get("web_desc_before", "")
            action = orgin_json.get("action_str", "")
            intention = orgin_json.get("intention", "")
            
            user_prompt += f"""Page{i+1}:
Page Description: {page_description}
Action: {action}
Local Intent: {intention}
        
"""
        except Exception as e:
            print(f"处理步骤 {i} 时出错: {e}")
            continue

    # 处理最后一步
    try:
        last_node_info = selected_trajectory[-1][0]
        if isinstance(last_node_info, str):
            filename, index, _, _ = ast.literal_eval(last_node_info)
        else:
            filename, index, _, _ = last_node_info
            
        orgin_json = get_origin_traj_json(filename, index)
        if orgin_json:
            final_a11y_tree = orgin_json.get("a11y_before", "")
            user_prompt += f"""Page{traj_len}:
Final Page Accessibility Tree: {final_a11y_tree}

Final Answer: {answer}
            """
    except Exception as e:
        print(f"处理最后一步时出错: {e}")

    user_prompt += """
Now, please generate your reason and score.
    """
    return user_prompt

def read_json_result(result: str) -> Dict:
    """解析 LLM 返回的 JSON 字符串"""
    clean_result = result.strip()
    if clean_result.startswith("```json"):
        clean_result = clean_result.split("```json")[1]
    if clean_result.endswith("```"):
        clean_result = clean_result.split("```")[0]
    
    try:
        return json.loads(clean_result.strip())
    except json.JSONDecodeError:
        print(f"JSON 解析失败，原始内容: {result}")
        return {
            "reason": "JSON Parse Error",
            "score_coherence": 0,
            "score_completion": 0,
            "score_difficulty": 0
        }

# =============================================================================
# 主程序
# =============================================================================

def main():
    # 1. 初始化 CSV 文件
    if not os.path.exists(OUTPUT_CSV_PATH):
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["traj_index", "traj_len", "reason", "score_coherence", "score_completion", "score_difficulty"])
    
    # 2. 读取输入文件
    print(f"正在读取随机轨迹文件: {INPUT_JSON_PATH}")
    if not os.path.exists(INPUT_JSON_PATH):
        print("错误：输入文件不存在。")
        return

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        random_selected_trajs = json.load(f)
    
    print(f"共加载 {len(random_selected_trajs)} 条轨迹。")

    # 3. 读取断点 (如果已存在部分结果)
    processed_indices = set()
    try:
        with open(OUTPUT_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed_indices.add(int(row[0]))
    except Exception:
        pass
    
    print(f"已处理 {len(processed_indices)} 条，继续处理剩余数据...")

    # 4. 循环打分
    for i in tqdm(range(len(random_selected_trajs)), desc="LLM Rating"):
        if i in processed_indices:
            continue
            
        selected_dict = random_selected_trajs[i]
        traj_len = len(selected_dict.get("selected_trajectory", []))

        # 构建 Prompt
        user_content = selected_trajectory_to_user_prompt(selected_dict)
        
        example_messages_with_name = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'name': 'user',
                'content': user_content
            }
        ]

        # 调用 LLM
        result = generate_from_openai_chat_completion_new(
            messages=example_messages_with_name,
            model="gpt-5.1", # 或者使用 "gpt-4o"
            temperature=0.7,
            max_tokens=2000,
            top_p=1.0,
            context_length=8192,
            stop_token=None
        )
        
        # 解析结果
        result_json = read_json_result(result)

        # 写入 CSV
        with open(OUTPUT_CSV_PATH, "a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                i, 
                traj_len,
                result_json.get("reason", ""), 
                result_json.get("score_coherence", 0), 
                result_json.get("score_completion", 0), 
                result_json.get("score_difficulty", 0)
            ])

    print(f"\n打分完成。结果已保存至: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()