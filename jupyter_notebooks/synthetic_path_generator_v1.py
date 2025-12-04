#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: Advanced Path Generation (Optimization Strategy)
---------------------------------------------------------
功能描述:
    基于图算法生成高质量的合成轨迹，作为与随机 Baseline 对比的"最优"组。
    
    核心策略:
    1. 图构建: 从三元组 CSV 构建有向多重图。
    2. 指标计算: 计算 PageRank (节点重要性) 和 Edge Weight (意图语义丰富度)。
    3. 候选生成: 针对每个目标长度 (2-8)，使用"智能加权游走"生成大量候选路径。
       - 优先从高 PageRank 节点出发。
       - 优先选择带有丰富 Intention 描述的边。
       - 避免环路。
    4. 择优筛选: 对候选路径进行综合打分 (节点权重 + 意图密度)，选出每个长度下分数最高的 25 条。

输入:
    - flitered_triplets.csv: 图谱三元组数据。

输出:
    - advanced_selected_trajs_v1.json: 筛选出的高质量轨迹数据。
"""

import csv
import json
import random
import ast
import os
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set
from tqdm import tqdm

# =============================================================================
# 配置参数
# =============================================================================

INPUT_CSV_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/flitered_triplets.csv"
OUTPUT_JSON_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/advanced_selected_trajs_v1.json"

# 生成配置
TARGET_LENGTHS = [2, 3, 4, 5, 6, 7, 8]  # 目标轨迹长度
SAMPLES_PER_LENGTH = 25                 # 每个长度需要的样本数
CANDIDATE_POOL_SIZE = 500               # 每个长度生成的候选池大小 (从中选最优)

# 算法权重参数
ALPHA_PAGERANK = 0.85                   # PageRank 阻尼系数
WEIGHT_INTENTION = 2.0                  # 有效意图的额外权重奖励
WEIGHT_PAGERANK = 100.0                 # PageRank 在游走概率中的放大倍数

# =============================================================================
# 1. 图构建与数据加载
# =============================================================================

def load_graph_from_csv(filepath: str) -> nx.MultiDiGraph:
    """
    从 CSV 加载图数据，构建 NetworkX MultiDiGraph。
    """
    print(f"正在加载图谱数据: {filepath}")
    G = nx.MultiDiGraph()
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        
        count = 0
        for row in reader:
            if len(row) < 4: continue
            
            src_str, action, intention, dst_str = row
            
            # 计算边的"质量分数"
            # 如果 intention 长度 > 5 且不是 'none'，认为是有意义的边，给予高分
            quality_score = 1.0
            if intention and len(intention) > 5 and 'none' not in intention.lower():
                quality_score = 2.0
            
            G.add_edge(
                src_str, 
                dst_str, 
                action=action, 
                intention=intention,
                weight=quality_score # 用于后续算法参考
            )
            count += 1
            
    print(f"图构建完成: {len(G.nodes)} 节点, {len(G.edges)} 边")
    return G

# =============================================================================
# 2. 高级指标计算
# =============================================================================

def calculate_graph_metrics(G: nx.MultiDiGraph) -> Dict[str, float]:
    """
    计算节点 PageRank，作为节点重要性的全局指标。
    """
    print("正在计算 PageRank 中心度...")
    # 将 MultiDiGraph 转为 DiGraph 计算 PageRank (权重累加)
    G_simple = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if G_simple.has_edge(u, v):
            G_simple[u][v]['weight'] += data.get('weight', 1.0)
        else:
            G_simple.add_edge(u, v, weight=data.get('weight', 1.0))
            
    try:
        pagerank = nx.pagerank(G_simple, alpha=ALPHA_PAGERANK, weight='weight')
    except Exception as e:
        print(f"PageRank 计算失败 ({e})，使用度中心性代替。")
        pagerank = nx.degree_centrality(G_simple)
        
    # 归一化处理，方便后续计算
    max_pr = max(pagerank.values()) if pagerank else 1
    for k in pagerank:
        pagerank[k] /= max_pr
        
    return pagerank

def get_high_value_starters(pagerank: Dict[str, float], top_ratio=0.3) -> List[str]:
    """获取 PageRank 排名靠前的节点作为优选起点"""
    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    limit = int(len(sorted_nodes) * top_ratio)
    return [n for n, s in sorted_nodes[:limit]]

# =============================================================================
# 3. 智能路径生成算法
# =============================================================================

def smart_weighted_walk(
    G: nx.MultiDiGraph, 
    start_node: str, 
    target_length: int, 
    pagerank: Dict[str, float]
) -> List[Tuple]:
    """
    执行一次智能加权游走。
    策略：
    1. 下一步的概率 = 目标节点 PageRank * 边意图质量权重。
    2. 强力避免环路 (Visited Penalty)。
    """
    path = [] # 存储 (u, action, intention, v)
    curr = start_node
    visited = {start_node}
    
    for _ in range(target_length):
        if G.out_degree(curr) == 0:
            break
            
        neighbors = []
        weights = []
        edges_info = [] # (v, key, attr)
        
        # 遍历所有出边
        for v in G.successors(curr):
            # 获取两点间所有边 (MultiGraph 可能有多条)
            for key, attr in G[curr][v].items():
                
                # 基础分：目标节点重要性
                score = pagerank.get(v, 0.01) * WEIGHT_PAGERANK
                
                # 意图分：是否有清晰意图
                intention = attr.get('intention', '')
                if intention and len(intention) > 5:
                    score *= WEIGHT_INTENTION
                
                # 惩罚：如果已访问，分数极剧降低 (避免死循环)
                if v in visited:
                    score *= 0.001
                
                neighbors.append(v)
                weights.append(score)
                edges_info.append((v, attr))
        
        if not neighbors:
            break
            
        # 轮盘赌选择下一步 (依概率)
        try:
            chosen_idx = random.choices(range(len(neighbors)), weights=weights, k=1)[0]
        except ValueError:
            # 如果所有权重为0 (极端情况)，随机选
            chosen_idx = random.choice(range(len(neighbors)))
            
        next_node, edge_attr = edges_info[chosen_idx]
        
        path.append((
            curr,
            edge_attr.get('action', ''),
            edge_attr.get('intention', ''),
            next_node
        ))
        
        visited.add(next_node)
        curr = next_node
    
    return path

def calculate_path_score(path: List[Tuple], pagerank: Dict[str, float]) -> float:
    """
    综合评分函数：评价一条路径的质量。
    Score = Avg(Node PageRank) + Avg(Intention Quality)
    """
    if not path: return 0.0
    
    # 1. 节点重要性得分 (覆盖了多少核心页面)
    # 提取路径上的所有节点 (source + last target)
    nodes = [step[0] for step in path] + [path[-1][3]]
    unique_nodes = set(nodes)
    node_score = sum(pagerank.get(n, 0) for n in unique_nodes) / len(unique_nodes)
    
    # 2. 意图质量得分 (多少步骤有明确意图)
    meaningful_intentions = 0
    for _, _, intention, _ in path:
        if intention and len(intention) > 10 and 'none' not in intention.lower():
            meaningful_intentions += 1
    
    # 惩罚短意图，奖励长且清晰的意图
    intention_score = meaningful_intentions / len(path)
    
    # 3. 惩罚重复节点 (环路)
    loop_penalty = 1.0
    if len(nodes) != len(unique_nodes):
        loop_penalty = 0.5 # 包含环路的路径降权
        
    return (node_score * 0.4 + intention_score * 0.6) * loop_penalty

# =============================================================================
# 4. 主流程逻辑
# =============================================================================

def main():
    # 1. 加载图
    G = load_graph_from_csv(INPUT_CSV_PATH)
    if len(G) == 0:
        print("图为空，退出。")
        return

    # 2. 计算指标
    pagerank = calculate_graph_metrics(G)
    high_value_starters = get_high_value_starters(pagerank, top_ratio=0.2) # 取前20%作为种子
    print(f"识别出 {len(high_value_starters)} 个高价值起始节点。")

    final_results = []
    
    print(f"\n开始生成优化轨迹 (每个长度目标 {SAMPLES_PER_LENGTH} 条，从候选池优选)...")

    # 3. 针对每个目标长度进行生成和筛选
    for length in TARGET_LENGTHS:
        candidate_paths = []
        
        # 3.1 生成候选池 (Candidate Generation)
        # 尝试次数稍微多一点，保证能生成足够的有效路径
        attempts = 0
        pbar = tqdm(total=CANDIDATE_POOL_SIZE, desc=f"Length {length} Candidates", leave=False)
        
        while len(candidate_paths) < CANDIDATE_POOL_SIZE and attempts < CANDIDATE_POOL_SIZE * 10:
            attempts += 1
            
            # 随机选择起点，但在高价值节点中有更高概率
            if random.random() < 0.7 and high_value_starters:
                start_node = random.choice(high_value_starters)
            else:
                start_node = random.choice(list(G.nodes()))
                
            # 执行智能游走
            path = smart_weighted_walk(G, start_node, length, pagerank)
            
            # 只有长度严格匹配才收录
            if len(path) == length:
                score = calculate_path_score(path, pagerank)
                candidate_paths.append({
                    'path': path,
                    'score': score
                })
                pbar.update(1)
        
        pbar.close()
        
        # 3.2 择优筛选 (Selection)
        # 按分数从高到低排序
        candidate_paths.sort(key=lambda x: x['score'], reverse=True)
        
        # 简单的去重逻辑 (基于路径字符串)
        unique_top_paths = []
        seen_paths = set()
        
        for item in candidate_paths:
            # 序列化路径用于去重比较: (src, action, dst) 序列
            path_signature = tuple([(s[0], s[1], s[3]) for s in item['path']])
            if path_signature not in seen_paths:
                unique_top_paths.append(item['path'])
                seen_paths.add(path_signature)
            
            if len(unique_top_paths) >= SAMPLES_PER_LENGTH:
                break
        
        print(f"  Length {length}: 从 {len(candidate_paths)} 条候选中选出最优 {len(unique_top_paths)} 条。")
        
        # 3.3 格式化结果
        for path in unique_top_paths:
            # 转换成评测脚本需要的格式 (List of Lists)
            # 每个步骤: [node_str, action, intention, node_str]
            formatted_traj = []
            for step in path:
                formatted_traj.append(list(step))
            
            final_results.append({
                "length": length,
                "selected_trajectory": formatted_traj
            })
    
    # 4. 保存结果
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    print(f"\n生成完成。共收集 {len(final_results)} 条最优轨迹。")
    print(f"正在保存至: {OUTPUT_JSON_PATH}")
    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
        
    print("保存成功！")

if __name__ == "__main__":
    main()