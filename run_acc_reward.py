# wcf00317/alrl/alrl-reward_model/run_rl_with_acc_reward.py

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
import yaml
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

# --- 核心复用: 导入您项目中已有的函数 ---
from models.model_utils import create_models, get_video_candidates, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward

cudnn.benchmark = False
cudnn.deterministic = True


def main(args):
    # --- 1. 初始化和配置加载 ---
    # (与原有脚本一致)
    if getattr(args, 'config', None):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    parser.save_arguments(args)
    shutil.copy(sys.argv[0], os.path.join(args.ckpt_path, args.exp_name, sys.argv[0].rsplit('/', 1)[-1]))

    # --- 2. 创建模型、数据、优化器 ---
    # (与原有脚本一致)
    net, policy_net, target_net = create_models(dataset=args.dataset,
                                                model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path,
                                                num_classes=args.num_classes,
                                                use_policy=True,
                                                embed_dim=args.embed_dim)

    _, train_set, val_loader, _ = get_data(
        data_path=args.data_path,
        tr_bs=args.train_batch_size,
        vl_bs=args.val_batch_size,
        dataset_name=args.dataset,
        model_type=args.model_type,
        n_workers=args.workers,
        clip_len=args.clip_len
    )
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer, optimizerP = create_and_load_optimizers(
        net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
        momentum=args.momentum, ckpt_path=args.ckpt_path,
        exp_name_toload=args.exp_name_toload, exp_name=args.exp_name,
        snapshot=args.snapshot, checkpointer=args.checkpointer,
        load_opt=args.load_opt, policy_net=policy_net, lr_dqn=args.lr_dqn
    )

    if args.train:
        print('--- 启动 RL 训练 (奖励信号: 真实△ACC) ---')

        # --- 3. RL 相关变量初始化 ---
        # (与原有脚本一致)
        Transition = namedtuple('Transition', (
        'state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
        memory = ReplayMemory(args.rl_buffer)
        TARGET_UPDATE = 5
        steps_done = 0
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

        # 初始化基准准确率
        print("正在计算初始模型的基准准确率...")
        initial_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)),
                                    batch_size=args.train_batch_size, shuffle=True,
                                    num_workers=args.workers)
        # 使用一个独立的优化器来评估初始准确率，避免影响主优化器状态
        initial_optimizer = create_and_load_optimizers(
            net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
            momentum=args.momentum,
            ckpt_path=args.ckpt_path,
            exp_name_toload=None,
            exp_name=args.exp_name,
            snapshot=args.snapshot,
            checkpointer=False,  # 临时计算，不加载 checkpoint
            load_opt=False  # 临时计算，不加载 optimizer 状态
        )[0]
        _, past_val_acc = train_har_for_reward(net, initial_loader, val_loader, initial_optimizer, criterion, args)
        print(f"初始基准准确率: {past_val_acc:.4f}")

        num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter

        # --- 4. 主动学习与 RL 训练循环 ---
        for i in range(num_al_steps):
            print(f'\\n--------------- RL (△ACC Reward) 回合 {i + 1}/{num_al_steps} ---------------')

            # a. 获取当前状态 (复用)
            current_state, candidate_indices, _ = compute_state_for_har(
                args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
            )

            # b. RL智能体选择动作 (复用)
            action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
            actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

            # --- c. 核心修改: 计算真实奖励 (△ACC) ---
            print("正在计算真实奖励 (△ACC)...")
            net_copy_for_reward = deepcopy(net)
            optimizer_for_reward = torch.optim.SGD(net_copy_for_reward.parameters(), lr=args.lr)  # 为副本创建独立优化器
            temp_set_for_reward = deepcopy(train_set)

            add_labeled_videos(args, [], actual_video_ids_to_label, temp_set_for_reward, budget=args.budget_labels,
                               n_ep=i)
            temp_loader_for_reward = DataLoader(
                Subset(temp_set_for_reward, list(temp_set_for_reward.labeled_video_ids)),
                batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)

            _, new_val_acc = train_har_for_reward(net_copy_for_reward, temp_loader_for_reward, val_loader,
                                                  optimizer_for_reward, criterion, args)

            real_reward = new_val_acc - past_val_acc
            print(f"真实奖励 (△ACC): {real_reward:.4f} (new: {new_val_acc:.4f} vs past: {past_val_acc:.4f})")

            # 释放副本占用的内存
            del net_copy_for_reward, optimizer_for_reward, temp_set_for_reward, temp_loader_for_reward
            torch.cuda.empty_cache()

            # d. 将选中的视频加入 *实际的* 已标注集合 (复用)
            add_labeled_videos(args, [], actual_video_ids_to_label, train_set,
                               budget=args.budget_labels, n_ep=i)

            # e. 重建 DataLoader (复用)
            current_labeled_indices = list(train_set.labeled_video_ids)
            train_loader = DataLoader(Subset(train_set, current_labeled_indices),
                                      batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=False)

            # f. 微调主模型，并更新 past_val_acc 以供下一轮使用
            print('使用新选择的视频更新主HAR网络...')
            _, past_val_acc = train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args)
            print(f"主模型已更新，新的基准准确率: {past_val_acc:.4f}")

            # g. 计算下一个状态 (复用)
            next_state = None
            if train_set.get_num_labeled_videos() < args.budget_labels:
                next_state, _, _ = compute_state_for_har(
                    args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
                )

            # h. 将经验存入Replay Buffer (使用计算出的真实奖励)
            reward_tensor = torch.tensor([real_reward], dtype=torch.float, device='cuda')
            memory.push(current_state, action, next_state, reward_tensor)

            # i. 优化策略网络 (复用)
            if len(memory) >= args.dqn_bs:
                optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, GAMMA=args.dqn_gamma,
                                    BATCH_SIZE=args.dqn_bs)

            # j. 更新目标网络 (复用)
            if i % TARGET_UPDATE == 0:
                print('更新目标网络...')
                target_net.load_state_dict(policy_net.state_dict())

        # --- 5. 最终收敛训练 ---
        # (与原有脚本一致)
        print("\\n预算已用尽。在所有已选数据上训练HAR模型至收敛...")
        logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None,
                                             log_name='final_convergence_log.txt')
        final_labeled_indices = list(train_set.labeled_video_ids)
        final_train_loader = DataLoader(Subset(train_set, final_labeled_indices), batch_size=args.train_batch_size,
                                        shuffle=True,
                                        num_workers=args.workers, drop_last=False)
        _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net,
                                                criterion, optimizer, val_loader,
                                                best_record, logger, scheduler,
                                                schedulerP, final_train=True)
        print(f"收敛后的最终验证集准确率: {final_val_acc:.4f}")
        torch.save(policy_net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, 'policy_final.pth'))

    # (测试部分与 run_rl_with_alrm.py 保持一致，这里省略以保持简洁)


if __name__ == '__main__':
    args = parser.get_arguments()
    main(args)