# 文件名: stage3_only.py

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
import yaml
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import pickle
import json
import argparse  # <-- 关键：确保导入argparse
from easydict import EasyDict as edict

# --- 模型和工具导入 ---
from models.model_utils import create_models, get_video_candidates, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv, load_models_for_har
from utils.reward_model import KAN_ActiveLearningRewardModel, MLP_ActiveLearningRewardModel, get_batch_features
from torch.utils.data import Subset, DataLoader
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser  # 我们仍然需要它的一些功能，比如 save_arguments
from run_rl_with_alrm import train_har_classifier, train_har_for_reward
from train_alrm import train_reward_model, check_model_weights

cudnn.benchmark = False
cudnn.deterministic = True


def main():
    # --- 1. 初始化和配置加载 ---
    # Bug修复: 创建一个临时的、独立的解析器，只用来获取 --exp_dir 参数。
    temp_parser = argparse.ArgumentParser(description="Stage 3 Runner: Load a trained ALRM and run RL.")
    temp_parser.add_argument('--exp_dir', type=str, required=True,
                             help='Path to the experiment directory from Stage 1 & 2 to resume Stage 3.')
    cli_args, _ = temp_parser.parse_known_args()

    if not os.path.isdir(cli_args.exp_dir):
        raise FileNotFoundError(f"错误: 提供的实验路径不存在: {cli_args.exp_dir}")

    # 从指定的实验文件夹加载args.json来恢复完整的实验配置
    args_path = os.path.join(cli_args.exp_dir, 'args.json')
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"错误: 在指定路径下找不到 'args.json': {args_path}")

    with open(args_path, 'r') as f:
        args_dict = json.load(f)

    args = edict(args_dict)
    # 确保exp_name与加载的文件夹名一致，以便日志和模型保存在正确的位置
    args.exp_name = os.path.basename(cli_args.exp_dir.rstrip('/\\'))
    print(f"成功从 {args_path} 加载实验配置。")
    print(f"将在此实验的基础上继续运行第三阶段: {args.exp_name}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ===================================================================================
    #                  第三阶段: 使用ALRM训练RL智能体
    # ===================================================================================
    print("\n" + "=" * 50)
    print("      第三阶段: 使用预训练的ALRM来训练RL智能体 (独立运行模式)")
    print("=" * 50)

    # --- 重新创建模型和数据 ---
    net, policy_net, target_net = create_models(dataset=args.dataset,
                                                model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path,
                                                num_classes=args.num_classes,
                                                use_policy=True,
                                                embed_dim=args.embed_dim)

    # 从头开始主动学习，所以使用一个带有初始标注的干净数据集
    train_loader, train_set, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        n_workers=args.workers, clip_len=args.clip_len, transform_type='c3d'
    )

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer, optimizerP = create_and_load_optimizers(net=net, opt_choice=args.optimizer, lr=args.lr,
                                                       wd=args.weight_decay,
                                                       momentum=args.momentum, ckpt_path=args.ckpt_path,
                                                       exp_name_toload=None, exp_name=args.exp_name,
                                                       snapshot=None, checkpointer=False,
                                                       load_opt=False, policy_net=policy_net,
                                                       lr_dqn=args.lr_dqn)

    # --- 加载训练好的ALRM ---
    alrm_save_path = os.path.join(cli_args.exp_dir, f'{args.reward_model_type}_alrm_model.pth')
    if not os.path.isfile(alrm_save_path):
        raise FileNotFoundError(f"错误: 找不到预训练的奖励模型: {alrm_save_path}")

    if args.reward_model_type == 'kan':
        alrm_model = KAN_ActiveLearningRewardModel(input_dim=4, hidden_layers=[8, 4])
        print("为RL Agent加载 KAN 奖励模型。")
    elif args.reward_model_type == 'mlp':
        alrm_model = MLP_ActiveLearningRewardModel(input_dim=4, hidden_layers=[16, 8])
        print("为RL Agent加载 MLP 奖励模型。")
    else:
        raise ValueError(f"未知的奖励模型类型: {args.reward_model_type}")

    alrm_model.load_state_dict(torch.load(alrm_save_path))
    alrm_model.cuda().eval()
    print("ALRM已加载，准备用于RL训练。")

    # --- RL训练循环 ---
    Transition = namedtuple('Transition',
                            ('state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
    memory = ReplayMemory(args.rl_buffer)
    TARGET_UPDATE = 5
    steps_done = 0
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

    num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter

    for i in range(num_al_steps):
        print(f'\n--- RL训练回合 {i + 1}/{num_al_steps} ---')

        current_state, candidate_indices, candidate_entropies = compute_state_for_har(
            args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
        )
        entropy_map = {idx: entropy for idx, entropy in zip(candidate_indices, candidate_entropies)}
        similarity_map = {idx: 0.5 for idx in candidate_indices}

        action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
        actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

        selected_entropies = [entropy_map.get(idx, 0) for idx in actual_video_ids_to_label]
        selected_similarities = [similarity_map.get(idx, 0) for idx in actual_video_ids_to_label]
        batch_features = get_batch_features(selected_entropies, selected_similarities).cuda()

        with torch.no_grad():
            predicted_reward = alrm_model(batch_features.unsqueeze(0)).item()
        print(f"ALRM 预测奖励: {predicted_reward:.4f}")

        add_labeled_videos(args, [], actual_video_ids_to_label, train_set, budget=args.budget_labels, n_ep=i)

        current_labeled_indices = list(train_set.labeled_video_ids)
        train_loader = DataLoader(Subset(train_set, current_labeled_indices),
                                  batch_size=args.train_batch_size, shuffle=True,
                                  num_workers=args.workers, drop_last=False)

        _, _ = train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args)

        next_state = None
        if train_set.get_num_labeled_videos() < args.budget_labels:
            next_state, _, _ = compute_state_for_har(
                args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
            )

        reward_tensor = torch.tensor([predicted_reward], dtype=torch.float, device='cuda')
        memory.push(current_state, action, next_state, reward_tensor)

        if len(memory) >= args.dqn_bs:
            optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, GAMMA=args.dqn_gamma,
                                BATCH_SIZE=args.dqn_bs)

        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # --- 最终收敛训练 ---
    print("\n预算用尽，在所有已选数据上训练至收敛...")
    final_log_path = os.path.join(args.ckpt_path, args.exp_name, 'final_convergence_log.txt')
    logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None,
                                         log_name=os.path.basename(final_log_path))

    final_train_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)),
                                    batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)

    _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net,
                                            criterion, optimizer, val_loader,
                                            best_record, logger, scheduler,
                                            schedulerP, final_train=True)
    logger.close()

    print(f"第三阶段完成！收敛后的最终验证集准确率: {final_val_acc:.4f}")
    torch.save(policy_net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, 'policy_final.pth'))


if __name__ == '__main__':
    main()