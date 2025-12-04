#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: High-Quality Data Synthesis (Clean Version)
----------------------------------------------------
策略: Goal-Driven Semantic Beam Search
功能: 仅生成高质量的轨迹数据，不包含 Instruction 和 Answer 字段。

核心逻辑:
1. 目标锚定: 随机锁定一个深层意图作为导航目标。
2. 语义导航: 使用 Embedding 引导 Agent 逼近目标。
3. 动态终止: 允许在逻辑闭环点提前结束。

输入: flitered_triplets.csv
输出: human_like_trajs_clean.json
"""

import csv
import json
import random
import ast
import os
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from urllib.parse import urlparse

# =============================================================================
# 1. 环境与模型配置
# =============================================================================

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 本地模型路径 (如果您下载了)
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"

USE_EMBEDDING = False
try:
    from sentence_transformers import SentenceTransformer, util
    
    # 优先尝试加载本地模型，否则使用镜像在线加载
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"正在加载本地语义模型: {LOCAL_MODEL_PATH} ...")
        EMBEDDING_MODEL = SentenceTransformer(LOCAL_MODEL_PATH)
    else:
        print("本地模型未找到，正在从镜像在线加载模型 (all-MiniLM-L6-v2)...")
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        
    USE_EMBEDDING = True
    print("模型加载成功！启用语义导航。")
except ImportError:
    print("警告: 未安装 sentence_transformers。降级使用关键词匹配。")
except Exception as e:
    print(f"模型加载失败: {e}。降级使用关键词匹配。")

# =============================================================================
# 2. 配置参数
# =============================================================================

INPUT_CSV_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/flitered_triplets.csv"
OUTPUT_JSON_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/advanced_selected_trajs_v2.json"

# 确保覆盖 2-8 的长度
TARGET_LENGTHS = [2, 3, 4, 5, 6, 7, 8]
SAMPLES_PER_LENGTH = 25
BEAM_WIDTH = 10 

# 评分权重
W_SEMANTIC_GOAL = 5.0    # 目标对齐 (最重要)
W_SEMANTIC_PREV = 2.0    # 上下文连贯
W_ACTION_LOGIC = 2.5     # 动作逻辑 (Type->Click)
W_VISITED_PENALTY = -5.0 # 防环路

# =============================================================================
# 3. 辅助函数
# =============================================================================

def parse_node_str(node_str: str) -> str:
    try:
        return ast.literal_eval(node_str)[3] # URL
    except:
        return ""

def get_text_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2: return 0.0
    
    if USE_EMBEDDING:
        e1 = EMBEDDING_MODEL.encode(text1, convert_to_tensor=True)
        e2 = EMBEDDING_MODEL.encode(text2, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(e1, e2)[0][0])
    else:
        s1 = set(text1.lower().split())
        s2 = set(text2.lower().split())
        return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0

def is_strong_action(action: str) -> bool:
    """判断是否是强交互动作"""
    return any(x in action.lower() for x in ['click', 'type', 'press', 'select'])

def is_termination_action(action: str) -> bool:
    """判断是否是任务终结动作"""
    act = action.lower()
    keywords = ['search', 'submit', 'checkout', 'login', 'post', 'save', 'place order', 'add to']
    return any(k in act for k in keywords)

# =============================================================================
# 4. 评分引擎
# =============================================================================

def calculate_score(
    curr_node: str,
    next_node: str,
    edge_attr: Dict,
    history: List[List],
    virtual_goal: str,
    visited_urls: set
) -> float:
    
    score = 0.0
    curr_int = edge_attr.get('intention', '')
    curr_act = edge_attr.get('action', '')
    dst_url = edge_attr.get('dst_url', '')
    
    # 1. 目标吸引力
    if virtual_goal:
        sim = get_text_similarity(curr_int, virtual_goal)
        score += sim * W_SEMANTIC_GOAL
        
    # 2. 避免重复访问 (URL级别)
    if dst_url in visited_urls:
        score += W_VISITED_PENALTY
        
    # 3. 动作连贯性
    if history:
        prev_act = history[-1][1]
        prev_int = history[-1][2]
        
        # 语义平滑
        sim_prev = get_text_similarity(prev_int, curr_int)
        score += sim_prev * W_SEMANTIC_PREV
        
        # 动作逻辑 (Type -> Click)
        if "type" in prev_act.lower() and "click" in curr_act.lower():
            score += W_ACTION_LOGIC
            
    # 4. 优先强交互
    if not is_strong_action(curr_act):
        score -= 1.0 
        
    return score

# =============================================================================
# 5. 目标驱动搜索
# =============================================================================

def goal_driven_search(
    G: nx.MultiDiGraph, 
    start_node: str, 
    goal_intent: str, 
    max_steps: int
) -> List[List]:
    
    start_url = parse_node_str(start_node)
    beam = [(start_node, [], 0.0, {start_url})]
    
    best_finished_path = None
    best_finished_score = -float('inf')
    
    for step in range(max_steps + 1): # 多给一步余量用于判断
        candidates = []
        
        for curr, hist, cum_score, visited in beam:
            # --- 终止判定 ---
            if len(hist) > 0:
                last_act = hist[-1][1]
                last_intent = hist[-1][2]
                
                # 检查是否达成目标 (语义匹配 或 强终止动作)
                is_goal_reached = False
                bonus = 0.0
                
                # 情况A: 语义高度匹配目标
                sim_goal = get_text_similarity(last_intent, goal_intent)
                if sim_goal > 0.75:
                    is_goal_reached = True
                    bonus = 5.0
                
                # 情况B: 强终止动作 (Submit/Search) 且长度接近目标
                if is_termination_action(last_act) and len(hist) >= max_steps - 1:
                    is_goal_reached = True
                    bonus = 3.0
                
                if is_goal_reached or len(hist) == max_steps:
                    final_score = cum_score + bonus
                    if final_score > best_finished_score:
                        best_finished_score = final_score
                        best_finished_path = hist
            
            if len(hist) >= max_steps: continue
            if G.out_degree(curr) == 0: continue
            
            # --- 扩展 ---
            for neighbor in G.successors(curr):
                neigh_url = parse_node_str(neighbor)
                
                # 严格防环 (Node ID)
                if any(x[0] == neighbor for x in hist): continue
                
                # 多重边选择
                best_edge_s = -999
                best_edge_d = None
                
                for k, attr in G[curr][neighbor].items():
                    s = calculate_score(curr, neighbor, attr, hist, goal_intent, visited)
                    if s > best_edge_s:
                        best_edge_s = s
                        best_edge_d = attr
                
                if best_edge_d:
                    new_hist = hist + [[curr, best_edge_d['action'], best_edge_d['intention'], neighbor]]
                    new_visit = visited.copy()
                    new_visit.add(neigh_url)
                    
                    candidates.append((neighbor, new_hist, cum_score + best_edge_s, new_visit))
        
        if not candidates: break
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        beam = candidates[:BEAM_WIDTH]
        
    return best_finished_path

# =============================================================================
# 6. 主程序
# =============================================================================

def load_graph(filepath):
    print("构建图谱...")
    G = nx.MultiDiGraph()
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if len(r)<4: continue
            src, act, intt, dst = r
            s_url = parse_node_str(src)
            d_url = parse_node_str(dst)
            G.add_edge(src, dst, action=act, intention=intt, src_url=s_url, dst_url=d_url)
    return G

def main():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"错误: 找不到输入文件 {INPUT_CSV_PATH}")
        return

    G = load_graph(INPUT_CSV_PATH)
    
    # 1. 准备目标池 (Intents)
    print("提取潜在目标...")
    potential_goals = []
    for u, v, data in G.edges(data=True):
        intent = data.get('intention', '')
        act = data.get('action', '')
        
        # 筛选高质量目标：强交互动作 或 深层页面意图
        if (is_termination_action(act) or len(intent) > 15):
            potential_goals.append(intent)
            
    potential_goals = list(set(potential_goals))
    print(f"目标池大小: {len(potential_goals)}")
    
    # 2. 准备起点池 (Hubs)
    out_degrees = sorted(G.out_degree, key=lambda x: x[1], reverse=True)
    hubs = [n for n, d in out_degrees[:int(len(G)*0.3)]]
    
    final_output = []
    
    print("\n开始生成轨迹 (Goal-Driven)...")
    
    for target_len in TARGET_LENGTHS:
        collected = []
        pbar = tqdm(total=SAMPLES_PER_LENGTH, desc=f"Length {target_len}")
        
        attempts = 0
        while len(collected) < SAMPLES_PER_LENGTH and attempts < 3000:
            attempts += 1
            
            goal = random.choice(potential_goals)
            start = random.choice(hubs)
            
            # 搜索
            path = goal_driven_search(G, start, goal, max_steps=target_len)
            
            # 筛选条件：长度在目标附近 (±1)
            if path and abs(len(path) - target_len) <= 1:
                # 去重
                sig = str([x[1] for x in path])
                if not any(str([x[1] for x in c]) == sig for c in collected):
                    collected.append(path)
                    pbar.update(1)
        
        pbar.close()
        
        for p in collected:
            final_output.append({
                "length": len(p),
                "selected_trajectory": p
            })
            
    # 保存
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n生成完成！已保存至: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()