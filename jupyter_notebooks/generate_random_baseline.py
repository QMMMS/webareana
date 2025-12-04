#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: Random Path Selection Script
-------------------------------------
功能描述:
    基于 Stage 1/2 生成的图谱三元组数据，执行随机游走算法。
    生成不同长度（2, 3, 4, 5, 6, 7, 8）的随机轨迹，用于后续的数据合成或训练。
    支持从特定起点（Start Point）出发和完全随机起点两种模式。

输入:
    - flitered_triplets.csv: 包含图谱三元组 (source, action, intent, target)

输出:
    - random_selected_trajs.json: 选取的随机轨迹列表
"""

import csv
import json
import random
import ast
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional

# =============================================================================
# 配置参数
# =============================================================================

INPUT_CSV_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/flitered_triplets.csv"
OUTPUT_JSON_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/random_selected_trajs_v2.json"

# 采样配置
# 修改：添加了 6, 7, 8 长度
TARGET_PATH_LENGTHS = [2, 3, 4, 5, 6, 7, 8]  
SAMPLES_PER_LENGTH = 25             # 每个长度采样的数量
START_POINT_RATIO = 0.3             # 强制从"起始节点"开始采样的比例 (30%)

# =============================================================================
# 核心函数
# =============================================================================

def load_triplets(file_path: str) -> List[List[str]]:
    """
    读取 CSV 文件中的三元组数据。
    
    Args:
        file_path: CSV 文件路径
        
    Returns:
        List[List[str]]: 三元组列表，每行数据格式为 [node_str, action, intent, node_str]
    """
    print(f"正在加载图谱数据: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # 跳过表头
        next(reader, None)
        triplets = [row for row in reader]
    
    print(f"成功加载 {len(triplets)} 条三元组记录。")
    return triplets

def extract_start_nodes(triplets: List[List[str]]) -> List[str]:
    """
    从三元组中提取合法的起始节点。
    判断标准：解析节点字符串元组，index == 0 且 is_after == 0。
    
    Args:
        triplets: 原始三元组列表
        
    Returns:
        List[str]: 起始节点的原始字符串列表
    """
    start_point_set = set()
    
    for triplet in triplets:
        node_str = triplet[0]
        try:
            # 使用 ast.literal_eval 安全解析字符串形式的元组
            # 格式: (file_name, index, is_after, real_url)
            node_tuple = ast.literal_eval(node_str)
            index = node_tuple[1]
            is_after = node_tuple[2]
            
            if index == 0 and is_after == 0:
                start_point_set.add(node_str)
        except (ValueError, SyntaxError) as e:
            print(f"解析节点出错: {node_str}, 错误: {e}")
            continue

    start_point_list = list(start_point_set)
    print(f"提取到 {len(start_point_list)} 个唯一的起始节点 (Start Points)。")
    return start_point_list

def sample_random_path(graph_data: List[List[str]], 
                       path_length: int, 
                       start_node: Optional[str] = None) -> Optional[List[Tuple]]:
    """
    在图上进行随机游走，采样指定长度的路径。
    使用 MultiDiGraph 以支持节点间的多重边。

    Args:
        graph_data: 图数据列表，每项为 [source, action, intent, target]
        path_length: 期望的路径长度 (边数)
        start_node: 指定起始节点。如果为 None，则随机选择全图任意节点。

    Returns:
        List[Tuple] or None: 
            成功时返回路径列表，每步为 (current_node, action, intent, next_node)。
            失败（死胡同或起点无效）返回 None。
    """
    if not graph_data:
        print("错误：图数据为空。")
        return None

    # 1. 构建有向多重图
    G = nx.MultiDiGraph()
    for row in graph_data:
        # row: [source, action, intent, target]
        source, action, intent, target = row[0], row[1], row[2], row[3]
        G.add_edge(source, target, action=action, intent=intent)

    # 2. 确定起始节点
    if start_node is not None:
        if not G.has_node(start_node):
            # 如果指定的起点不在图中，无法开始
            return None
        current_node = start_node
    else:
        current_node = random.choice(list(G.nodes()))
    
    # 3. 执行随机游走
    trajectory = []
    # print(f"随机游走开始，起点: '{current_node}'，期望长度: {path_length}")

    for i in range(path_length):
        # 获取当前节点的所有出边 (source, target, key)
        outgoing_edges = list(G.out_edges(current_node, keys=True))
        
        # 检查是否遇到死胡同
        if not outgoing_edges:
            # print(f"警告：在第 {i+1} 步遇到死胡同。")
            break
        
        # 随机选择一条边
        source_node, next_node, edge_key = random.choice(outgoing_edges)
        
        # 获取边属性
        edge_data = G.get_edge_data(source_node, next_node, key=edge_key)
        
        # 记录步骤
        step_tuple = (current_node, edge_data['action'], edge_data['intent'], next_node)
        trajectory.append(step_tuple)
        
        # 移动到下一个节点
        current_node = next_node
        
    return trajectory

# =============================================================================
# 主程序逻辑
# =============================================================================

def main():
    # 1. 加载数据
    triplets = load_triplets(INPUT_CSV_PATH)
    
    # 2. 提取特定的起始节点集合
    start_point_list = extract_start_nodes(triplets)
    
    out_list = []
    total_iterations = len(TARGET_PATH_LENGTHS) * SAMPLES_PER_LENGTH
    pbar_counter = 0

    print(f"\n开始生成随机轨迹... (目标总数: {total_iterations})")

    # 3. 遍历不同的路径长度进行采样
    for traj_len in TARGET_PATH_LENGTHS:
        print(f"正在生成长度为 {traj_len} 的轨迹...")
        
        for i in range(SAMPLES_PER_LENGTH):
            while True:
                # 策略: 前 30% 的数据强制从 start_point_list 中随机选择起点
                # 后 70% 的数据在全图中随机选择起点
                if i < SAMPLES_PER_LENGTH * START_POINT_RATIO:
                    start_node = random.choice(start_point_list)
                    random_trajectory = sample_random_path(
                        triplets, 
                        path_length=traj_len, 
                        start_node=start_node
                    )
                else:
                    random_trajectory = sample_random_path(
                        triplets, 
                        path_length=traj_len, 
                        start_node=None
                    )

                # 只有当生成的轨迹长度严格等于期望长度时，才算成功（过滤掉死胡同数据）
                if random_trajectory and len(random_trajectory) == traj_len:
                    break
            
            # 封装结果
            temp_dict = {
                "length": traj_len,
                "selected_trajectory": random_trajectory,
            }
            out_list.append(temp_dict)
            pbar_counter += 1

    # 4. 保存结果
    print(f"\n生成完成。共收集 {len(out_list)} 条轨迹。")
    print(f"正在保存结果至: {OUTPUT_JSON_PATH}")
    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)
    
    print("保存成功！")

if __name__ == "__main__":
    main()