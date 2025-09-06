# 文件名: run_ebm_workflow.py

import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import pickle

# --- 核心导入 ---
from models.model_utils import create_models, add_labeled_videos, compute_state_for_har, select_action_for_har, \
    optimize_model_conv
from utils.feature_extractor import UnifiedFeatureExtractor, get_all_unlabeled_embeddings, get_all_labeled_embeddings
from torch.utils.data import Subset, DataLoader
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward
import utils.al_scoring as scoring

# --- 导入我们新建的EBM训练模块 ---
from models.train_ebm import train_ebm_reward_model, load_ebm_scorer, predict_ebm_reward

cudnn.benchmark = False
cudnn.deterministic = True


# (VerboseLogger 类与 run_minimalist_tournament.py 中保持一致)
class VerboseLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __del__(self):
        try:
            self.log_file.close()
        except Exception:
            pass


def get_available_strategies(args):
    """根据配置文件动态获取所有启用的策略及其对应的评分函数。"""
    strategy_map = {
        'use_statistical_features': ('entropy', scoring.compute_entropy_score),
        'use_diversity_feature': ('diversity', scoring.compute_diversity_score),
        'use_representativeness_feature': ('representativeness', scoring.compute_representativeness_score),
        'use_prediction_margin_feature': ('prediction_margin', scoring.compute_prediction_margin_score),
        'use_labeled_distance_feature': ('labeled_distance', scoring.compute_labeled_distance_score),
        'use_neighborhood_density_feature': ('neighborhood_density', scoring.compute_neighborhood_density_score),
        'use_temporal_consistency_feature': ('temporal_consistency', scoring.compute_temporal_consistency_score),
        'use_cross_view_consistency_feature': ('cross_view_consistency', scoring.compute_cross_view_consistency_score)
    }
    available_strategies = [
        ('bald', scoring.compute_bald_score),
        ('egl', scoring.compute_egl_adaptive_topk)
    ]
    for arg_name, (score_name, score_func) in strategy_map.items():
        if getattr(args, arg_name, False):
            available_strategies.append((score_name, score_func))
    print(f"启用的策略共 {len(available_strategies)} 个: {[name for name, _ in available_strategies]}")
    return available_strategies


def main():
    # --- 0. 初始化和配置加载 (与 run_minimalist_tournament.py 一致) ---
    args = parser.get_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.exp_name = f"{args.exp_name}_{timestamp}"
    exp_dir = os.path.join(args.ckpt_path, args.exp_name)
    check_mkdir(args.ckpt_path)
    check_mkdir(exp_dir)

    sys.stdout = VerboseLogger(os.path.join(exp_dir, 'verbose_run_log.txt'))
    parser.save_arguments(args)
    if hasattr(args, 'config') and args.config and os.path.exists(args.config):
        shutil.copy(args.config, os.path.join(exp_dir, os.path.basename(args.config)))
    print(f"实验将保存在: {exp_dir}")

    # ===================================================================================
    #              第一阶段: 使用特征两端提取法收集偏好数据
    # ===================================================================================
    print("\n" + "=" * 25 + "  STAGE 1: PREFERENCE DATA COLLECTION  " + "=" * 25)

    feature_extractor = UnifiedFeatureExtractor(args)
    net_stage1, _, _ = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                     model_ckpt_path=args.model_ckpt_path, num_classes=args.num_classes,
                                     use_policy=False)
    net_stage1.cuda()
    aug_level = args.augment_level if getattr(args, 'use_cross_view_consistency_feature', False) else None

    _, train_set, _, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        augment_level=aug_level,initial_labeled_ratio=args.initial_labeled_ratio
    )

    alrm_preference_data = []
    unlabeled_indices = train_set.get_candidates_video_ids()
    if not unlabeled_indices:
        raise ValueError("未标注池为空，无法进行数据收集。")

    # precomputed_data = scoring.precompute_data_for_scoring(
    #     args, net_stage1, unlabeled_indices, train_set, batch_size=args.val_batch_size
    # )

    strategies_to_run = get_available_strategies(args)
    print("正在为所有策略预计算全局分数...")
    strategies_to_run = get_available_strategies(args)
    precomputed_data = scoring.precompute_data_for_scoring(
        args, net_stage1, unlabeled_indices, train_set, batch_size=args.val_batch_size
    )

    all_scores_map = {}
    for strategy_name, scoring_function in tqdm(strategies_to_run, desc="全局分数计算"):
        if strategy_name in ['bald', 'egl']:
            all_scores_map[strategy_name] = scoring_function(net_stage1, unlabeled_indices, train_set)
        else:
            all_scores_map[strategy_name] = scoring_function(precomputed_data)
    print("全局分数计算完成。")

    # --- 步骤 2: 保持您原有的For循环结构，用于筛选和特征化 ---
    print("\n开始遍历策略以提取两端批次...")
    for strategy_name, _ in tqdm(strategies_to_run, desc="为不同策略提取特征两端样本"):
        print(f"\n--- 处理策略: {strategy_name} ---")

        # a. 从预计算的全局分数中获取当前策略的分数
        scores = all_scores_map[strategy_name]

        if scores.numel() == 0:
            print(f"警告: 策略 {strategy_name} 未能计算出任何分数，跳过。")
            continue

        # b. 选出最好和最差样本的位置索引
        k = min(args.num_each_iter, len(unlabeled_indices))
        top_scores, top_pos_indices = torch.topk(scores, k=k, largest=True)
        bottom_scores, bottom_pos_indices = torch.topk(scores, k=k, largest=False)

        # c. 根据位置索引获取视频ID
        winner_batch_indices = [unlabeled_indices[i] for i in top_pos_indices]
        loser_batch_indices = [unlabeled_indices[i] for i in bottom_pos_indices]

        # d. (核心修正) 为选出的批次打包所有策略的分数
        winner_batch_scores = {s_name: all_scores_map[s_name][top_pos_indices] for s_name, _ in strategies_to_run}
        loser_batch_scores = {s_name: all_scores_map[s_name][bottom_pos_indices] for s_name, _ in strategies_to_run}

        # e. 使用修正后的、正确的参数调用 extract 函数
        winner_features = feature_extractor.extract(
            winner_batch_indices, net_stage1, train_set, batch_scores=winner_batch_scores
        )
        loser_features = feature_extractor.extract(
            loser_batch_indices, net_stage1, train_set, batch_scores=loser_batch_scores
        )

        alrm_preference_data.append({'winner': winner_features, 'loser': loser_features})
        # --- 新增的详细日志输出 ---
        print(f"  [好学生 - Winner Batch]")
        print(f"    - 样本索引: {winner_batch_indices}")
        print(f"    - 对应分数: {[f'{s:.4f}' for s in top_scores.tolist()]}")

        print(f"  [坏学生 - Loser Batch]")
        print(f"    - 样本索引: {loser_batch_indices}")
        print(f"    - 对应分数: {[f'{s:.4f}' for s in bottom_scores.tolist()]}")
        print(f"{'=' * 50}\n")

    alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
    with open(alrm_data_path, 'wb') as f:
        pickle.dump(alrm_preference_data, f)
    print(f"\n--- STAGE 1 COMPLETE --- 偏好数据已保存至 {alrm_data_path}，共 {len(alrm_preference_data)} 对。")
    del net_stage1, train_set, precomputed_data
    torch.cuda.empty_cache()

    # ===================================================================================
    #                         第二阶段: 训练 EBM 奖励模型
    # ===================================================================================
    print("\n" + "=" * 25 + "  STAGE 2: EBM REWARD MODEL TRAINING  " + "=" * 25)

    training_successful = train_ebm_reward_model(alrm_preference_data, exp_dir)

    if not training_successful:
        print("EBM奖励模型训练失败，工作流程终止。")
        return

    print(f"--- STAGE 2 COMPLETE ---")
    del alrm_preference_data
    torch.cuda.empty_cache()

    # ===================================================================================
    #                  第三阶段: 使用 EBM 奖励模型训练 RL 智能体
    # ===================================================================================
    print("\n" + "=" * 25 + "  STAGE 3: RL AGENT TRAINING WITH EBM  " + "=" * 25)

    net_stage3, policy_net, target_net = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                                       model_ckpt_path=args.model_ckpt_path,
                                                       num_classes=args.num_classes, use_policy=True,
                                                       embed_dim=args.embed_dim)

    _, train_set_rl, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        augment_level=aug_level,initial_labeled_ratio=args.initial_labeled_ratio
    )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_rl, optimizerP = create_and_load_optimizers(
        net=net_stage3, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
        momentum=args.momentum, policy_net=policy_net, lr_dqn=args.lr_dqn,
        ckpt_path=args.ckpt_path, exp_name=args.exp_name,
        exp_name_toload=args.exp_name_toload,  # 从args中获取
        snapshot=args.snapshot,  # 从args中获取
        checkpointer=args.checkpointer,  # 从args中获取
        load_opt=args.load_opt,  # 从args中获取
    )

    # --- 核心修改: 加载EBM计分器 ---
    ebm_scorer = load_ebm_scorer(exp_dir)

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
        print(f'\n--- RL训练回合 {i + 1}/{num_al_steps} ---')
        # # --- 新增RL阶段的预计算 ---
        # rl_unlabeled_indices = train_set_rl.get_candidates_video_ids()
        #
        # all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net_stage3, train_set_rl)
        # all_labeled_embeds = get_all_labeled_embeddings(args, net_stage3, train_set_rl)

        current_state, candidate_indices, _ = compute_state_for_har(args, net_stage3, train_set_rl,
                                                                    train_set_rl.get_candidates_video_ids(),
                                                                    list(train_set_rl.labeled_video_ids))
        action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
        actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

        # 3. **按需、仅为选中的批次计算所有必需的分数**
        batch_scores = {}

        # a. 预计算当前批次的所有数据
        #    注意：这里的 video_indices 参数只包含当前被选中的批次，非常高效
        precomputed_batch_data = scoring.precompute_data_for_scoring(
            args, net_stage3, actual_video_ids_to_label, train_set_rl, batch_size=args.train_batch_size
        )

        # b. 调用所有需要的评分函数
        #    get_available_strategies 帮助我们动态获取所有启用的评分函数
        strategies_to_run = get_available_strategies(args)
        for strategy_name, scoring_function in strategies_to_run:
            if strategy_name not in ['bald', 'egl']:  # BALD/EGL太慢，在RL循环中跳过
                batch_scores[strategy_name] = scoring_function(precomputed_batch_data)

        # 4. 调用 extract 函数，传入为该批次计算好的分数
        batch_features = feature_extractor.extract(
            actual_video_ids_to_label, net_stage3, train_set_rl, batch_scores=batch_scores
        ).cuda()

        # --- 核心修改: 使用 EBM 预测奖励 ---
        predicted_reward = predict_ebm_reward(ebm_scorer, batch_features)
        print(f"EBM 预测奖励 (P(x > μ)): {predicted_reward:.4f}")

        add_labeled_videos(args, [], actual_video_ids_to_label, train_set_rl, budget=args.budget_labels, n_ep=i)

        current_labeled_indices = list(train_set_rl.labeled_video_ids)
        train_loader_rl = DataLoader(Subset(train_set_rl, current_labeled_indices),
                                     batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, _ = train_har_for_reward(net_stage3, train_loader_rl, val_loader, optimizer_rl, criterion, args)

        next_state = None
        if train_set_rl.get_num_labeled_videos() < args.budget_labels:
            next_state, _, _ = compute_state_for_har(args, net_stage3, train_set_rl,
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

    _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net_stage3, criterion, optimizer_rl,
                                            val_loader, best_record, logger, scheduler, schedulerP,
                                            final_train=True)
    logger.close()

    print(f"\n--- STAGE 3 COMPLETE --- 收敛后的最终验证集准确率: {final_val_acc:.4f}")
    torch.save(policy_net.state_dict(), os.path.join(exp_dir, 'policy_final.pth'))


if __name__ == '__main__':
    main()