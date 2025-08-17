# 文件名: wcf00317/alrl/alrl-reward_model/run_tournament_workflow.py

import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy
import datetime
import torch.optim as optim
import torch
import torch.nn as nn
import yaml
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import pickle

# --- 模型和工具导入 ---
from models.model_utils import create_models, add_labeled_videos, optimize_model_conv, select_action_for_har, \
    compute_state_for_har
from utils.reward_model import KAN_ActiveLearningRewardModel, MLP_ActiveLearningRewardModel
from utils.feature_extractor import UnifiedFeatureExtractor, get_all_labeled_embeddings, get_all_unlabeled_embeddings
from torch.utils.data import Subset, DataLoader
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward
from train_alrm import train_reward_model

# --- 核心导入：导入我们新建的锦标赛模块 ---
from models.tournament_selection import run_tournament_round

cudnn.benchmark = False
cudnn.deterministic = True


def main():
    # --- 1. 初始化和配置加载 ---
    args = parser.get_arguments()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    original_exp_name = args.exp_name
    # 为新工作流创建一个独特的实验文件夹名称
    args.exp_name = f"{original_exp_name}_tournament_{timestamp}"
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

    # 实例化统一特征提取器
    feature_extractor = UnifiedFeatureExtractor(args)

    # ===================================================================================
    #                           第一阶段: 偏好数据收集 (锦标赛版本)
    # ===================================================================================
    print("\\n" + "=" * 50)
    print("                 第一阶段: 偏好数据收集 (锦标赛框架)")
    print("=" * 50)

    net, _, _ = create_models(dataset=args.dataset,
                              model_cfg_path=args.model_cfg_path,
                              model_ckpt_path=args.model_ckpt_path,
                              num_classes=args.num_classes,
                              use_policy=False)
    net.cuda()

    train_loader, train_set, val_loader, _ = get_data(
        data_path=args.data_path,
        tr_bs=args.train_batch_size,
        vl_bs=args.val_batch_size,
        dataset_name=args.dataset,
        n_workers=args.workers,
        clip_len=args.clip_len,
        initial_labeled_ratio=args.initial_labeled_ratio
    )
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_main = create_and_load_optimizers(net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
                                                momentum=args.momentum, ckpt_path=args.ckpt_path,
                                                exp_name_toload=args.exp_name_toload, exp_name=args.exp_name,
                                                snapshot=args.snapshot, checkpointer=args.checkpointer,
                                                load_opt=args.load_opt)[0]

    alrm_preference_data = []
    alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
    num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter
    past_val_acc = 0.0

    # 初始评估一次以获得基准准确率
    print("正在评估初始模型的基准准确率...")
    # 创建一个初始的DataLoader来进行评估
    initial_labeled_indices = list(train_set.labeled_video_ids)
    initial_train_loader = DataLoader(Subset(train_set, initial_labeled_indices),
                                      batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.workers)
    _, past_val_acc = train_har_for_reward(net, initial_train_loader, val_loader, optimizer_main, criterion, args)
    print(f"初始基准准确率: {past_val_acc:.4f}")

    for i in range(num_al_steps):
        print(f'\\n================ 锦标赛回合 {i + 1}/{num_al_steps} ===============')

        # --- 核心修改：调用锦标赛流程 ---
        preference_data_for_round, champion_batch = run_tournament_round(
            args, net, train_set, val_loader, criterion, feature_extractor, past_val_acc
        )

        alrm_preference_data.extend(preference_data_for_round)

        # --- 主模型更新 ---
        print(f"使用冠军批次更新主模型...")
        add_labeled_videos(args, [], champion_batch, train_set, budget=args.budget_labels, n_ep=i)
        main_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)), batch_size=args.train_batch_size,
                                 shuffle=True, num_workers=args.workers)

        _, past_val_acc = train_har_for_reward(net, main_loader, val_loader, optimizer_main, criterion, args)
        print(f"主模型已更新, 新的基准准确率: {past_val_acc:.4f}")

    with open(alrm_data_path, 'wb') as f:
        pickle.dump(alrm_preference_data, f)
    print(f"第一阶段完成！偏好数据已保存至 {alrm_data_path}，共 {len(alrm_preference_data)} 对。")

    del net, train_loader, train_set, val_loader, optimizer_main, main_loader
    torch.cuda.empty_cache()

    # ===================================================================================
    #                         第二阶段: 训练ALRM
    # ===================================================================================
    print("\\n" + "=" * 50)
    print("                 第二阶段: 训练主动学习奖励模型 (ALRM)")
    print("=" * 50)

    input_dim = feature_extractor.feature_dim

    if args.reward_model_type == 'kan':
        alrm_model = KAN_ActiveLearningRewardModel(
            input_dim=input_dim,
            grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order,
            hidden_layers=args.kan_hidden_layers
        ).cuda()
        print("使用 KAN 奖励模型进行训练。")
    elif args.reward_model_type == 'mlp':
        hidden_dim1 = max(16, input_dim * 4)
        hidden_dim2 = max(8, input_dim * 2)
        alrm_model = MLP_ActiveLearningRewardModel(input_dim=input_dim, hidden_layers=[hidden_dim1, hidden_dim2]).cuda()
        print(f"使用 MLP 奖励模型 (Baseline) 进行训练，结构: [{input_dim}, {hidden_dim1}, {hidden_dim2}, 1]")
    else:
        raise ValueError(f"未知的奖励模型类型: {args.reward_model_type}")

    optimizer_alrm = optim.Adam(alrm_model.parameters(), lr=1e-4)
    training_successful = train_reward_model(alrm_model, alrm_preference_data, optimizer_alrm)

    if not training_successful:
        print("ALRM训练失败，工作流程终止。")
        return

    alrm_save_path = os.path.join(exp_dir, f'{args.reward_model_type}_alrm_model.pth')
    torch.save(alrm_model.state_dict(), alrm_save_path)
    print(f"第二阶段完成！ALRM模型已保存至 {alrm_save_path}")

    del alrm_model, optimizer_alrm, alrm_preference_data
    torch.cuda.empty_cache()

    # ===================================================================================
    #                  第三阶段: 使用ALRM训练RL智能体
    # ===================================================================================
    print("\\n" + "=" * 50)
    print("                 第三阶段: 使用ALRM训练RL智能体")
    print("=" * 50)

    net, policy_net, target_net = create_models(dataset=args.dataset,
                                                model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path,
                                                num_classes=args.num_classes,
                                                use_policy=True,
                                                embed_dim=args.embed_dim)

    train_loader, train_set, val_loader, _ = get_data(
        data_path=args.data_path,
        tr_bs=args.train_batch_size,
        vl_bs=args.val_batch_size,
        dataset_name=args.dataset,
        n_workers=args.workers,
        clip_len=args.clip_len,
        initial_labeled_ratio=args.initial_labeled_ratio
    )
    optimizer, optimizerP = create_and_load_optimizers(net=net, opt_choice=args.optimizer, lr=args.lr,
                                                       wd=args.weight_decay,
                                                       momentum=args.momentum, ckpt_path=args.ckpt_path,
                                                       exp_name_toload=args.exp_name_toload, exp_name=args.exp_name,
                                                       snapshot=args.snapshot, checkpointer=args.checkpointer,
                                                       load_opt=args.load_opt, policy_net=policy_net,
                                                       lr_dqn=args.lr_dqn)

    if args.reward_model_type == 'kan':
        alrm_model = KAN_ActiveLearningRewardModel(
            input_dim=input_dim, grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order, hidden_layers=args.kan_hidden_layers
        )
    elif args.reward_model_type == 'mlp':
        hidden_dim1 = max(16, input_dim * 4);
        hidden_dim2 = max(8, input_dim * 2)
        alrm_model = MLP_ActiveLearningRewardModel(input_dim=input_dim, hidden_layers=[hidden_dim1, hidden_dim2])
    else:
        raise ValueError(f"未知的奖励模型类型: {args.reward_model_type}")

    alrm_model.load_state_dict(torch.load(alrm_save_path))
    alrm_model.cuda().eval()
    print("ALRM已加载，准备用于RL训练。")

    Transition = namedtuple('Transition',
                            ('state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
    memory = ReplayMemory(args.rl_buffer)
    TARGET_UPDATE = 5
    steps_done = 0
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

    for i in range(num_al_steps):
        print(f'\\n--- RL训练回合 {i + 1}/{num_al_steps} ---')

        all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net, train_set)
        all_labeled_embeds = get_all_labeled_embeddings(args, net, train_set)

        current_state, candidate_indices, _ = compute_state_for_har(
            args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
        )

        action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
        actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

        batch_features = feature_extractor.extract(
            actual_video_ids_to_label, net, train_set, all_unlabeled_embeds, all_labeled_embeds
        ).cuda()

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
    print("\\n预算用尽，在所有已选数据上训练至收敛...")
    final_log_path = os.path.join(exp_dir, 'final_convergence_log.txt')
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
    torch.save(policy_net.state_dict(), os.path.join(exp_dir, 'policy_final.pth'))


if __name__ == '__main__':
    main()