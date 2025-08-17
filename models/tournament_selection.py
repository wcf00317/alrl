# 文件名: wcf00317/alrl/alrl-reward_model/tournament_selection.py (V2 - Dynamic Strategy Pool)

import torch
import torch.nn.functional as F
import random
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.model_utils import add_labeled_videos
from run_rl_with_alrm import train_har_for_reward
from utils.k_center_greedy import k_center_greedy
from utils.feature_extractor import get_all_unlabeled_embeddings, get_all_labeled_embeddings


def get_strategy_pool(args):
    """
    (V2) 创建一个动态的、基于YAML配置的策略池。
    """
    strategy_pool = []

    # 1. 基础策略
    strategy_pool.append({'name': 'random'})

    # 2. 从args中动态添加基于特征的策略
    if args.use_statistical_features:
        strategy_pool.append({'name': 'entropy'})  # statistical_features的核心是不确定性
    if args.use_diversity_feature:
        strategy_pool.append({'name': 'diversity'})
    if args.use_representativeness_feature:
        strategy_pool.append({'name': 'representativeness'})
    if args.use_prediction_margin_feature:
        strategy_pool.append({'name': 'prediction_margin'})
    if args.use_labeled_distance_feature:
        strategy_pool.append({'name': 'labeled_distance'})
    if args.use_neighborhood_density_feature:
        strategy_pool.append({'name': 'neighborhood_density'})

    print(f"动态策略池已创建，共 {len(strategy_pool)} 个策略: {[s['name'] for s in strategy_pool]}")
    return strategy_pool


def calculate_scores_for_strategy(strategy, all_embeddings, all_probs, all_labeled_embeddings, unlabeled_centroid):
    """
    根据特定策略，为所有候选视频计算分数。
    """
    if strategy == 'entropy':
        scores = -torch.sum(all_probs * torch.log(all_probs + 1e-8), dim=1)
    elif strategy == 'prediction_margin':
        sorted_probs, _ = torch.sort(all_probs, dim=1, descending=True)
        scores = 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])  # 我们希望边际小，所以分数是 1 - margin
    elif strategy == 'representativeness':
        scores = F.cosine_similarity(all_embeddings, unlabeled_centroid)
    elif strategy == 'labeled_distance':
        if all_labeled_embeddings is None or len(all_labeled_embeddings) == 0:
            return torch.zeros(len(all_embeddings))
        dist_matrix = torch.cdist(all_embeddings, all_labeled_embeddings)
        scores, _ = torch.min(dist_matrix, dim=1)
    elif strategy == 'neighborhood_density':
        dist_matrix = torch.cdist(all_embeddings, all_embeddings)
        k = min(10, len(all_embeddings))
        knn_dists = torch.topk(dist_matrix, k, largest=False, dim=1).values
        mean_knn_dist = knn_dists.mean(dim=1)
        scores = 1.0 / (1.0 + mean_knn_dist)
    else:  # 默认为随机
        scores = torch.rand(len(all_embeddings))

    return scores


def generate_candidate_batches(args, strategy_pool, model, train_set, all_unlabeled_embeddings, all_labeled_embeddings):
    """
    (V2) 使用动态策略池，为每个策略选择最优批次，生成候选批次池。
    """
    print("正在为动态策略池生成候选批次...")
    candidate_batches = []

    # 预计算所有未标注视频的概率
    unlabeled_indices = train_set.get_candidates_video_ids()
    all_probs = []
    with torch.no_grad():
        for vid_idx in unlabeled_indices:
            video_tensor = train_set.get_video(vid_idx).unsqueeze(0).cuda()
            features = model.extract_feat(video_tensor)[0]
            logits = model.cls_head(features)
            probs = F.softmax(logits, dim=1).cpu()
            all_probs.append(probs)
    all_probs = torch.cat(all_probs, dim=0)

    unlabeled_centroid = all_unlabeled_embeddings.mean(dim=0,
                                                       keepdim=True) if all_unlabeled_embeddings is not None else None

    for strategy in tqdm(strategy_pool, desc="Generating Batches per Strategy"):
        scores = calculate_scores_for_strategy(
            strategy['name'], all_unlabeled_embeddings, all_probs, all_labeled_embeddings, unlabeled_centroid
        )

        # 选出得分最高的 k 个视频的索引
        _, top_k_indices = torch.topk(scores, k=args.num_each_iter)

        # 将这些在分数张量中的索引，映射回原始的视频ID
        batch_indices = [unlabeled_indices[i] for i in top_k_indices]
        candidate_batches.append(batch_indices)

    return candidate_batches


def run_tournament_round(args, net, train_set, val_loader, criterion, feature_extractor, past_val_acc):
    """
    (V2) 执行一轮完整的、基于动态策略池的锦标赛。
    """
    # --- 1. 策略锦标赛 - 生成海量候选 ---
    strategy_pool = get_strategy_pool(args)
    all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net, train_set)
    all_labeled_embeds = get_all_labeled_embeddings(args, net, train_set)

    candidate_batches = generate_candidate_batches(args, strategy_pool, net, train_set, all_unlabeled_embeds,
                                                   all_labeled_embeds)

    # --- 2. 批次特征化 ---
    print("为所有候选批次提取特征...")
    batch_feature_vectors = []
    for batch in tqdm(candidate_batches, desc="Characterizing Batches"):
        features = feature_extractor.extract(batch, net, train_set, all_unlabeled_embeds, all_labeled_embeds)
        batch_feature_vectors.append(features)

    # --- 3. 特征覆盖驱动 - 选拔决赛选手 ---
    print("使用k-center贪心算法选拔决赛选手...")
    num_finalists = min(4, len(candidate_batches))  # 决赛选手不能超过候选总数
    finalist_indices = k_center_greedy(torch.stack(batch_feature_vectors), k=num_finalists)
    finalist_batches = [candidate_batches[i] for i in finalist_indices]
    finalist_features = [batch_feature_vectors[i] for i in finalist_indices]
    print(f"已选出 {len(finalist_batches)} 个决赛批次进入最终评估。")
    finalist_strategies = [strategy_pool[i]['name'] for i in finalist_indices]
    print(f"决赛批次对应的策略是: {finalist_strategies}")

    # --- 4. 最终对决与丰富的偏好数据生成 ---
    finalist_rewards = []
    for i, batch in enumerate(tqdm(finalist_batches, desc="Evaluating Finalists")):
        net_copy = deepcopy(net)
        optimizer_temp = torch.optim.SGD(net_copy.parameters(), lr=args.lr)
        temp_set = deepcopy(train_set)
        add_labeled_videos(args, [], batch, temp_set, budget=args.budget_labels, n_ep=0)
        temp_loader = DataLoader(Subset(temp_set, list(temp_set.labeled_video_ids)),
                                 batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, acc = train_har_for_reward(net_copy, temp_loader, val_loader, optimizer_temp, criterion, args)
        finalist_rewards.append(acc - past_val_acc)

    # 生成排名和偏好对
    ranked_indices = np.argsort(finalist_rewards)[::-1]
    preference_pairs = []
    for i in range(len(ranked_indices)):
        for j in range(i + 1, len(ranked_indices)):
            winner_idx_in_finalists = ranked_indices[i]
            loser_idx_in_finalists = ranked_indices[j]

            # 确保奖励有显著差异，避免记录噪声
            if finalist_rewards[winner_idx_in_finalists] > finalist_rewards[loser_idx_in_finalists] + 1e-4:
                winner_features = finalist_features[winner_idx_in_finalists]
                loser_features = finalist_features[loser_idx_in_finalists]
                preference_pairs.append({'winner': winner_features, 'loser': loser_features})

    print(f"从本轮锦标赛中生成了 {len(preference_pairs)} 个偏好对。")

    # --- 5. 确定冠军批次 ---
    champion_batch_index_in_finalists = ranked_indices[0]
    champion_batch = finalist_batches[champion_batch_index_in_finalists]

    return preference_pairs, champion_batch