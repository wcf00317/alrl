# 文件名: wcf00317/alrl/alrl-reward_model/utils/k_center_greedy.py

import torch
import random
import numpy as np


def k_center_greedy(feature_vectors, k):
    """
    使用k-center贪心算法从一组特征向量中选择一个多样化的子集。

    Args:
        feature_vectors (torch.Tensor): 一个形状为 [N, D] 的张量，其中 N 是候选批次数量，D 是特征维度。
        k (int): 要选择的“中心”（即决赛选手）的数量。

    Returns:
        list[int]: 被选中的候选批次的索引列表。
    """
    if not isinstance(feature_vectors, torch.Tensor):
        feature_vectors = torch.tensor(np.array(feature_vectors), dtype=torch.float32)

    num_candidates = feature_vectors.shape[0]
    if k >= num_candidates:
        return list(range(num_candidates))

    # 存储到目前为止，每个非中心点到最近中心点的“最小距离”
    min_distances = torch.full((num_candidates,), float('inf'), dtype=torch.float32)

    selected_indices = []

    # 1. 随机选择第一个中心点
    first_index = random.randint(0, num_candidates - 1)
    selected_indices.append(first_index)

    # 2. 迭代选择剩下的 k-1 个中心点
    for _ in range(k - 1):
        # a. 更新所有点到新中心的距离
        last_selected_features = feature_vectors[selected_indices[-1]].unsqueeze(0)

        # 计算所有点到这个新中心的L1距离
        # L1范数（曼哈顿距离）对于高维稀疏特征通常更鲁棒
        dist_to_last = torch.sum(torch.abs(feature_vectors - last_selected_features), dim=1)

        # 更新每个点的“最小距离”
        min_distances = torch.minimum(min_distances, dist_to_last)

        # b. 选择具有最大“最小距离”的点作为下一个中心
        next_index = torch.argmax(min_distances).item()
        # 确保不会重复选择同一个索引
        min_distances[next_index] = -1.0  # 标记为已选
        selected_indices.append(next_index)

    return selected_indices