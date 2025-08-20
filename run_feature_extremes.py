# 文件名: wcf00317/alrl/alrl-reward_model/run_feature_extremes.py

import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple
import datetime
import torch.optim as optim
import torch
import torch.nn as nn
import yaml
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import pickle

# --- 模型和工具导入 ---
from models.model_utils import create_models, add_labeled_videos, optimize_model_conv, select_action_for_har, \
    compute_state_for_har
from utils.reward_model import KAN_ActiveLearningRewardModel, MLP_ActiveLearningRewardModel
from utils.feature_extractor import UnifiedFeatureExtractor, get_all_unlabeled_embeddings, get_all_labeled_embeddings
from torch.utils.data import Subset, DataLoader
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward
from train_alrm import train_reward_model
import utils.al_scoring as scoring  # 导入评分函数模块

cudnn.benchmark = False
cudnn.deterministic = True


def get_available_strategies(args):
    """
    根据配置文件(args)动态获取所有启用的策略及其对应的评分函数。
    """
    strategy_map = {
        'use_statistical_features': ('entropy', scoring.compute_entropy_score),
        'use_diversity_feature': ('diversity', scoring.compute_diversity_score),
        'use_representativeness_feature': ('representativeness', scoring.compute_representativeness_score),
        'use_prediction_margin_feature': ('prediction_margin', scoring.compute_prediction_margin_score),
        'use_labeled_distance_feature': ('labeled_distance', scoring.compute_labeled_distance_score),
        'use_neighborhood_density_feature': ('neighborhood_density', scoring.compute_neighborhood_density_score),
        'use_temporal_consistency_feature': ('temporal_consistency', scoring.compute_temporal_consistency_score)
    }

    # 始终包含 BALD 和 EGL 作为核心策略
    available_strategies = [
        ('bald', scoring.compute_bald_score),
        ('egl', scoring.compute_egl_adaptive_topk)  # 默认使用最高效的自适应K版本
    ]

    for arg_name, (score_name, score_func) in strategy_map.items():
        if getattr(args, arg_name, False):
            available_strategies.append((score_name, score_func))

    print(f"启用的策略共 {len(available_strategies)} 个: {[name for name, _ in available_strategies]}")
    return available_strategies


def main():
    # --- 1. 初始化和配置加载 ---
    args = parser.get_arguments()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    original_exp_name = args.exp_name
    args.exp_name = f"{original_exp_name}_feature_extremes_{timestamp}"
    exp_dir = os.path.join(args.ckpt_path, args.exp_name)
    check_mkdir(args.ckpt_path)
    check_mkdir(exp_dir)

    parser.save_arguments(args)
    shutil.copy(sys.argv[0], os.path.join(exp_dir, sys.argv[0].rsplit('/', 1)[-1]))
    if hasattr(args, 'config') and args.config and os.path.exists(args.config):
        shutil.copy(args.config, os.path.join(exp_dir, os.path.basename(args.config)))
    print(f"实验将保存在: {exp_dir}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    feature_extractor = UnifiedFeatureExtractor(args)

    # ===================================================================================
    #              第一阶段: 特征两端提取法进行偏好数据收集
    # ===================================================================================
    print("\n" + "=" * 50)
    print("           第一阶段: 特征两端提取法进行偏好数据收集")
    print("=" * 50)

    net, _, _ = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                              model_ckpt_path=args.model_ckpt_path, num_classes=args.num_classes,
                              use_policy=False)
    net.cuda()

    _, train_set, _, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        initial_labeled_ratio=args.initial_labeled_ratio
    )

    alrm_preference_data = []

    # --- 核心逻辑 ---
    unlabeled_indices = train_set.get_candidates_video_ids()
    if not unlabeled_indices:
        raise ValueError("未标注池为空，无法进行数据收集。")

    # 1. 预计算所有评分所需的数据
    precomputed_data = scoring.precompute_data_for_scoring(
        args, net, unlabeled_indices, train_set, batch_size=args.val_batch_size
    )

    # 2. 获取所有启用的策略
    strategies_to_run = get_available_strategies(args)

    # 3. 遍历每种策略，提取“最好”和“最差”的批次
    for strategy_name, scoring_function in tqdm(strategies_to_run, desc="为不同策略提取特征两端样本"):
        print(f"\n--- 处理策略: {strategy_name} ---")

        # a. 计算分数
        if strategy_name in ['bald', 'egl']:
            # 这些函数需要模型和数据作为输入
            scores = scoring_function(net, unlabeled_indices, train_set)
        else:
            # 其他函数直接使用预计算的数据
            scores = scoring_function(precomputed_data)

        if scores.numel() == 0:
            print(f"警告: 策略 {strategy_name} 未能计算出任何分数，跳过。")
            continue

        # b. 选出最好 (Winner) 和最差 (Loser) 的样本
        k = min(args.num_each_iter, len(unlabeled_indices))
        top_scores, top_indices = torch.topk(scores, k=k, largest=True)
        bottom_scores, bottom_indices = torch.topk(scores, k=k, largest=False)

        winner_batch_indices = [unlabeled_indices[i] for i in top_indices]
        loser_batch_indices = [unlabeled_indices[i] for i in bottom_indices]

        # c. 提取批次级特征
        all_unlabeled_embeds = precomputed_data.get('embeddings')
        all_labeled_embeds = precomputed_data.get('labeled_embeddings')

        winner_features = feature_extractor.extract(winner_batch_indices, net, train_set, all_unlabeled_embeds,
                                                    all_labeled_embeds)
        loser_features = feature_extractor.extract(loser_batch_indices, net, train_set, all_unlabeled_embeds,
                                                   all_labeled_embeds)

        # d. 存入偏好对
        alrm_preference_data.append({'winner': winner_features, 'loser': loser_features})
        # --- 新增的详细日志输出 ---
        print(f"  [好学生 - Winner Batch]")
        print(f"    - 样本索引: {winner_batch_indices}")
        print(f"    - 对应分数: {[f'{s:.4f}' for s in top_scores.tolist()]}")

        print(f"  [坏学生 - Loser Batch]")
        print(f"    - 样本索引: {loser_batch_indices}")
        print(f"    - 对应分数: {[f'{s:.4f}' for s in bottom_scores.tolist()]}")
        print(f"{'=' * 50}\n")
    # --- 逻辑结束 ---

    alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
    with open(alrm_data_path, 'wb') as f:
        pickle.dump(alrm_preference_data, f)
    print(f"\n第一阶段完成！偏好数据已保存至 {alrm_data_path}，共 {len(alrm_preference_data)} 对。")

    del net, train_set, precomputed_data
    torch.cuda.empty_cache()

    # ===================================================================================
    #                         第二阶段: 训练ALRM (基本复用)
    # ===================================================================================
    print("\n" + "=" * 50)
    print("           第二阶段: 训练主动学习奖励模型 (ALRM)")
    print("=" * 50)

    input_dim = feature_extractor.feature_dim
    if args.reward_model_type == 'kan':
        alrm_model = KAN_ActiveLearningRewardModel(input_dim=input_dim, hidden_layers=args.kan_hidden_layers).cuda()
    else:
        alrm_model = MLP_ActiveLearningRewardModel(input_dim=input_dim,
                                                   hidden_layers=[input_dim * 4, input_dim * 2]).cuda()

    optimizer_alrm = optim.Adam(alrm_model.parameters(), lr=1e-4)
    train_reward_model(alrm_model, alrm_preference_data, optimizer_alrm)

    alrm_save_path = os.path.join(exp_dir, f'{args.reward_model_type}_alrm_model.pth')
    torch.save(alrm_model.state_dict(), alrm_save_path)
    print(f"第二阶段完成！ALRM模型已保存至 {alrm_save_path}")

    # ===================================================================================
    #                  第三阶段: 使用ALRM训练RL智能体 (基本复用)
    # ===================================================================================
    print("\n" + "=" * 50)
    print("           第三阶段: 使用ALRM训练RL智能体")
    print("=" * 50)

    net, policy_net, target_net = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path,
                                                num_classes=args.num_classes,
                                                use_policy=True, embed_dim=args.embed_dim)

    # 为第三阶段重新加载一个干净的数据集
    _, train_set_rl, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        initial_labeled_ratio=args.initial_labeled_ratio
    )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_rl, optimizerP = create_and_load_optimizers(
        net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
        momentum=args.momentum, ckpt_path=args.ckpt_path, exp_name_toload=None,
        exp_name=args.exp_name, snapshot=None, checkpointer=False, load_opt=False,
        policy_net=policy_net, lr_dqn=args.lr_dqn
    )

    alrm_model.load_state_dict(torch.load(alrm_save_path))
    alrm_model.cuda().eval()

    Transition = namedtuple('Transition',
                            ('state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
    memory = ReplayMemory(args.rl_buffer)
    TARGET_UPDATE = 5
    steps_done = 0
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    scheduler = ExponentialLR(optimizer_rl, gamma=args.gamma)
    schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

    num_al_steps = (args.budget_labels - train_set_rl.get_num_labeled_videos()) // args.num_each_iter
    for i in range(num_al_steps):
        print(f'--- RL训练回合 {i + 1}/{num_al_steps} ---')

        all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net, train_set_rl)
        all_labeled_embeds = get_all_labeled_embeddings(args, net, train_set_rl)

        current_state, candidate_indices, _ = compute_state_for_har(args, net, train_set_rl,
                                                                    train_set_rl.get_candidates_video_ids(),
                                                                    list(train_set_rl.labeled_video_ids))
        action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
        actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

        batch_features = feature_extractor.extract(actual_video_ids_to_label, net, train_set_rl, all_unlabeled_embeds,
                                                   all_labeled_embeds).cuda()
        with torch.no_grad():
            predicted_reward = alrm_model(batch_features.unsqueeze(0)).item()

        add_labeled_videos(args, [], actual_video_ids_to_label, train_set_rl, budget=args.budget_labels, n_ep=i)

        current_labeled_indices = list(train_set_rl.labeled_video_ids)
        train_loader_rl = DataLoader(Subset(train_set_rl, current_labeled_indices),
                                     batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, _ = train_har_for_reward(net, train_loader_rl, val_loader, optimizer_rl, criterion, args)

        next_state = None
        if train_set_rl.get_num_labeled_videos() < args.budget_labels:
            next_state, _, _ = compute_state_for_har(args, net, train_set_rl,
                                                     train_set_rl.get_candidates_video_ids(),
                                                     list(train_set_rl.labeled_video_ids))

        memory.push(current_state, action, next_state,
                    torch.tensor([predicted_reward], dtype=torch.float, device='cuda'))
        if len(memory) >= args.dqn_bs:
            optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, GAMMA=args.dqn_gamma,
                                BATCH_SIZE=args.dqn_bs)
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("\n预算用尽，在所有已选数据上训练至收敛...")
    final_log_path = os.path.join(exp_dir, 'final_convergence_log.txt')
    logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None,
                                         log_name=os.path.basename(final_log_path))
    final_train_loader = DataLoader(Subset(train_set_rl, list(train_set_rl.labeled_video_ids)),
                                    batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)

    _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net, criterion, optimizer_rl,
                                            val_loader, best_record, logger, scheduler, schedulerP,
                                            final_train=True)
    logger.close()

    print(f"\n第三阶段完成！收敛后的最终验证集准确率: {final_val_acc:.4f}")
    torch.save(policy_net.state_dict(), os.path.join(exp_dir, 'policy_final.pth'))


if __name__ == '__main__':
    main()