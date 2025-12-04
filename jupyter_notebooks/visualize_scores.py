#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: Visualization Script
-----------------------------
功能描述:
    读取 LLM 评分生成的 CSV 结果文件，进行统计分析并生成可视化图表。
    1. 评分分布图 (按轨迹长度分组)
    2. 平均分趋势图

输入:
    - rating_result.csv: 包含评分结果的 CSV 文件。
      格式: traj_index, traj_len, reason, score_coherence, score_completion, score_difficulty

输出:
    - score_distribution.png: 评分分布统计图
    - average_scores.png: 平均分统计图
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# =============================================================================
# 配置参数
# =============================================================================

# 输入文件路径 (请修改为您实际生成的 CSV 文件路径)
INPUT_CSV_PATH = "/home/zjusst/qms/webarena/result_stage_3_generation/rating_advanced_result_v2.csv"

# 输出图片保存目录
OUTPUT_DIR = "/home/zjusst/qms/webarena/result_stage_3_generation/plots"

# 可视化配置
TRAJ_LENS = [2, 3, 4, 5, 6, 7, 8, 9]  # 需要分析的轨迹长度列表
SCORE_TYPES = ['score_coherence', 'score_completion', 'score_difficulty']
SCORE_LABELS = ['Coherence', 'Completion', 'Difficulty']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色

# =============================================================================
# 数据处理函数
# =============================================================================

def load_rating_data(file_path: str) -> List[Dict[str, Any]]:
    """
    读取 CSV 评分数据。
    """
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return []

    print(f"正在加载数据: {file_path}")
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值类型
                try:
                    processed_row = {
                        'traj_index': int(row['traj_index']),
                        'traj_len': int(row['traj_len']),
                        'score_coherence': int(row['score_coherence']),
                        'score_completion': int(row['score_completion']),
                        'score_difficulty': int(row['score_difficulty'])
                    }
                    data.append(processed_row)
                except (ValueError, KeyError) as e:
                    print(f"警告: 跳过无效行 - {e}")
                    continue
    except Exception as e:
        print(f"读取文件出错: {e}")
        return []
    
    print(f"成功加载 {len(data)} 条评分记录。")
    return data

def organize_data_by_len(data: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[int]]]:
    """
    将数据按轨迹长度组织。
    结构: { len: { 'score_type': [scores...] } }
    """
    organized = {}
    for item in data:
        t_len = item['traj_len']
        if t_len not in organized:
            organized[t_len] = {k: [] for k in SCORE_TYPES}
        
        for s_type in SCORE_TYPES:
            organized[t_len][s_type].append(item[s_type])
            
    return organized

# =============================================================================
# 绘图函数
# =============================================================================

def plot_score_distribution(organized_data: Dict[int, Dict[str, List[int]]], save_path: str):
    """
    绘制每个长度下的评分分布图 (Subplots)。
    """
    # 筛选出存在的长度
    valid_lens = [l for l in TRAJ_LENS if l in organized_data]
    if not valid_lens:
        print("没有符合要求的轨迹长度数据，跳过分布图绘制。")
        return

    num_plots = len(valid_lens)
    fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=True)
    
    # 如果只有一个子图，axs 不是列表，需要包装
    if num_plots == 1:
        axs = [axs]

    x = np.arange(1, 6)  # 评分 1-5
    width = 0.25         # 柱状图宽度

    print("正在绘制评分分布图...")

    for idx, t_len in enumerate(valid_lens):
        ax = axs[idx]
        scores_map = organized_data[t_len]
        
        # 针对每种评分类型绘制柱状图
        max_count = 0
        for i, s_type in enumerate(SCORE_TYPES):
            scores = scores_map[s_type]
            # 统计 1-5 分的各有多少个
            counts = [scores.count(s) for s in range(1, 6)]
            max_count = max(max_count, max(counts) if counts else 0)
            
            # 计算偏移量以便并排显示
            offset = (i - 1) * width 
            bars = ax.bar(x + offset, counts, width, label=SCORE_LABELS[i], 
                          color=COLORS[i], edgecolor='black', alpha=0.8)
            
            # 标注数值
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                            str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(f'Trajectory Length {t_len}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Score (1-5)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(['1', '2', '3', '4', '5'])
        
        # 设置 Y 轴范围略高一点以便显示数字
        ax.set_ylim(0, max_count * 1.15 if max_count > 0 else 5)
        
        if idx == 0:
            ax.set_ylabel('Frequency', fontsize=12)
        
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Score Distribution by Trajectory Length', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"分布图已保存至: {save_path}")
    plt.close()

def plot_average_scores(organized_data: Dict[int, Dict[str, List[int]]], save_path: str):
    """
    绘制不同长度下的平均分对比图。
    """
    valid_lens = [l for l in TRAJ_LENS if l in organized_data]
    if not valid_lens:
        return

    # 计算平均分数据
    means_data = {label: [] for label in SCORE_LABELS}
    
    print("\n--- 平均分统计 ---")
    for t_len in valid_lens:
        print(f"Length {t_len}:")
        for i, s_type in enumerate(SCORE_TYPES):
            scores = organized_data[t_len][s_type]
            mean_val = np.mean(scores) if scores else 0
            means_data[SCORE_LABELS[i]].append(mean_val)
            print(f"  {SCORE_LABELS[i]}: {mean_val:.2f}")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(valid_lens))
    width = 0.25

    print("\n正在绘制平均分图...")

    for i, (label, color) in enumerate(zip(SCORE_LABELS, COLORS)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means_data[label], width, 
                      label=label, color=color, edgecolor='black', alpha=0.8)
        
        # 标注数值
        for bar, value in zip(bars, means_data[label]):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Trajectory Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Average Scores by Trajectory Length', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Len {tl}' for tl in valid_lens])
    ax.set_ylim(0, 5.5) # 分数上限为 5
    ax.set_yticks(np.arange(0, 5.5, 0.5))
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"平均分图已保存至: {save_path}")
    plt.close()

# =============================================================================
# 主程序
# =============================================================================

def main():
    # 1. 准备输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 加载数据
    data = load_rating_data(INPUT_CSV_PATH)
    if not data:
        return

    # 3. 组织数据
    organized_data = organize_data_by_len(data)

    # 4. 生成图表
    dist_plot_path = os.path.join(OUTPUT_DIR, "score_distribution.png")
    avg_plot_path = os.path.join(OUTPUT_DIR, "average_scores.png")

    plot_score_distribution(organized_data, dist_plot_path)
    plot_average_scores(organized_data, avg_plot_path)

    print("\n所有可视化任务完成！")

if __name__ == "__main__":
    main()