#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: Centrality-Based Path Selection (Graph Analysis Strategy)
----------------------------------------------------------------
功能描述:
    基于节点中心性 (Centrality) 筛选高质量的关键路径。
    逻辑:
    1. 计算全图节点的出度中心性 (Out-degree) 和入度中心性 (In-degree)。
    2. 选取高出度节点作为"起点" (Hubs)，高入度节点作为"终点" (Authorities)。
    3. 寻找这些起止点之间的最短路径。
    4. 对路径进行综合评分并排序，按长度分组输出。

输入:
    - flitered_triplets.csv

输出:
    - advanced_selected_trajs.json
"""

import csv
import json
import os
import networkx as nx
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

# =============================================================================
# 配置参数
# =============================================================================

INPUT_CSV_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/flitered_triplets.csv"
OUTPUT_JSON_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/advanced_selected_trajs.json"

# 采样配置
TARGET_LENGTHS = [2, 3, 4, 5, 6, 7, 8]  # 目标路径长度
SAMPLES_PER_LENGTH = 25             # 每个长度采样的数量
TOP_K_NODES = 100                   # 筛选多少个高分节点作为候选起点/终点
MAX_PATHS_PER_PAIR = 10             # 每对起止点最多保留多少条路径

# =============================================================================
# 核心函数
# =============================================================================

def load_graph_from_csv(file_path: str) -> nx.DiGraph:
    """
    读取 CSV 并构建 NetworkX 有向图 (DiGraph)。
    """
    print(f"正在加载图谱数据: {file_path}")
    G = nx.DiGraph()
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # 跳过表头
            
            count = 0
            for row in reader:
                if len(row) < 4: continue
                source, action, intent, target = row
                # 将属性存储在边上
                G.add_edge(source, target, action=action, intent=intent)
                count += 1
                
        print(f"成功构建图谱: {len(G.nodes)} 节点, {len(G.edges)} 边")
        return G
    except Exception as e:
        print(f"加载失败: {e}")
        return nx.DiGraph()

def sample_centrality_paths(G: nx.DiGraph, max_length: int, top_k: int) -> List[Dict]:
    """
    执行中心性路径搜索算法。
    """
    if len(G) == 0: return []

    print("正在计算节点中心性...")
    # 1. 计算中心性
    out_centrality = nx.out_degree_centrality(G)
    in_centrality = nx.in_degree_centrality(G)

    # 2. 筛选 Top-K 起点和终点
    sorted_starts = sorted(out_centrality.items(), key=lambda x: x[1], reverse=True)
    sorted_ends = sorted(in_centrality.items(), key=lambda x: x[1], reverse=True)

    top_starts = [node for node, score in sorted_starts[:top_k]]
    top_ends = [node for node, score in sorted_ends[:top_k]]
    
    print(f"筛选出 Top-{top_k} 起点和终点。开始路径搜索...")

    candidate_paths = []
    
    # 3. 遍历起止点对，寻找路径
    total_pairs = len(top_starts) * len(top_ends)
    pbar = tqdm(total=total_pairs, desc="Searching Paths")

    for start_node in top_starts:
        for end_node in top_ends:
            pbar.update(1)
            
            if start_node == end_node:
                continue
            
            try:
                # 寻找简单路径 (无环)，按长度递增排序
                path_generator = nx.shortest_simple_paths(G, source=start_node, target=end_node)
                
                paths_found_count = 0
                
                for nodes_path in path_generator:
                    path_len = len(nodes_path) - 1 # 边数
                    
                    if path_len > max_length:
                        break 
                    
                    if path_len < 2:
                        continue

                    path_score = sum(in_centrality[n] + out_centrality[n] for n in nodes_path) / len(nodes_path)
                    
                    candidate_paths.append({
                        "score": path_score,
                        "nodes": nodes_path,
                        "length": path_len
                    })
                    
                    paths_found_count += 1
                    if paths_found_count >= MAX_PATHS_PER_PAIR:
                        break

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
                
    pbar.close()
    
    # 4. 全局排序
    print("正在对所有候选路径进行排序...")
    candidate_paths.sort(key=lambda x: x["score"], reverse=True)
    
    return candidate_paths

def reconstruct_trajectory(G: nx.DiGraph, nodes_path: List[str]) -> List[Tuple]:
    """
    将节点列表还原为完整的轨迹三元组列表。
    """
    trajectory = []
    for i in range(len(nodes_path) - 1):
        u = nodes_path[i]
        v = nodes_path[i+1]
        
        # 从图中获取边属性
        edge_data = G[u][v]
        
        step = (u, edge_data.get('action', ''), edge_data.get('intent', ''), v)
        trajectory.append(step)
        
    return trajectory

# =============================================================================
# 主程序
# =============================================================================

def main():
    # 1. 加载数据
    G = load_graph_from_csv(INPUT_CSV_PATH)
    if len(G) == 0: return

    # 2. 运行算法
    max_len = max(TARGET_LENGTHS)
    all_candidates = sample_centrality_paths(G, max_length=max_len, top_k=TOP_K_NODES)
    
    print(f"共找到 {len(all_candidates)} 条候选路径。正在按长度采样...")

    # 3. 按长度分组采样
    final_output = []
    count_dict = {l: 0 for l in TARGET_LENGTHS}
    
    # 按照分数高低顺序进行采样
    for item in all_candidates:
        length = item['length']
        
        if length in count_dict and count_dict[length] < SAMPLES_PER_LENGTH:
            # 还原轨迹详情
            traj = reconstruct_trajectory(G, item['nodes'])
            
            # 格式化输出: 转换为 list 格式以匹配 JSON 结构
            formatted_traj = [list(step) for step in traj]
            
            # --- 修改部分：仅保留核心字段 ---
            final_output.append({
                "length": length,
                "selected_trajectory": formatted_traj
            })
            
            count_dict[length] += 1
            
        # 检查是否所有长度都采满了
        if all(c >= SAMPLES_PER_LENGTH for c in count_dict.values()):
            break
            
    # 4. 输出统计与保存
    print("\n采样统计:")
    for length in TARGET_LENGTHS:
        print(f"  Length {length}: {count_dict[length]} 条")
        
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\n保存成功: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()