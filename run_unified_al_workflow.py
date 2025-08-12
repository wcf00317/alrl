# 文件名: run_unified_al_workflow.py

import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy
import datetime  # 导入datetime来创建时间戳
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
from models.model_utils import create_models, get_video_candidates, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv, load_models_for_har
from utils.reward_model import KAN_ActiveLearningRewardModel, get_batch_features, MLP_ActiveLearningRewardModel
from torch.utils.data import Subset, DataLoader
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward
from train_alrm import train_reward_model, check_model_weights

cudnn.benchmark = False
cudnn.deterministic = True


def main():
    # --- 1. 初始化和配置加载 ---
    args = parser.get_arguments()
    # --- 日志改进：创建带时间戳的唯一实验文件夹 ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    original_exp_name = args.exp_name
    args.exp_name = f"{original_exp_name}_{timestamp}"
    exp_dir = os.path.join(args.ckpt_path, args.exp_name)
    check_mkdir(args.ckpt_path)
    check_mkdir(exp_dir)

    # 保存参数和脚本快照
    parser.save_arguments(args)
    shutil.copy(sys.argv[0], os.path.join(exp_dir, sys.argv[0].rsplit('/', 1)[-1]))
    if args.config:
        shutil.copy(args.config, os.path.join(exp_dir, os.path.basename(args.config)))

    print(f"实验将保存在: {exp_dir}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ===================================================================================
    #                           第一阶段: 数据收集 (来自 run.py)
    # ===================================================================================
    print("\n" + "=" * 50)
    print("                 第一阶段: 偏好数据收集")
    print("=" * 50)

    # --- 为数据收集阶段准备模型和数据 ---
    net, _, _ = create_models(dataset=args.dataset,
                              model_cfg_path=args.model_cfg_path,
                              model_ckpt_path=args.model_ckpt_path,
                              num_classes=args.num_classes,
                              use_policy=False)  # 在这个阶段我们只需要一个HAR模型
    net.cuda()

    train_loader, train_set, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        n_workers=args.workers, clip_len=args.clip_len, transform_type='c3d'
    )
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_main = create_and_load_optimizers(net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
                                                momentum=args.momentum, ckpt_path=args.ckpt_path,
                                                exp_name_toload=args.exp_name_toload, exp_name=args.exp_name,
                                                snapshot=args.snapshot, checkpointer=args.checkpointer,
                                                load_opt=args.load_opt)[0]

    # --- 开始数据收集循环 ---
    alrm_preference_data = []
    alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
    num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter

    past_val_acc = 0.0  # 初始准确率
    # 可以在这里选择性地在初始标记数据上预训练一下模型来得到一个更准确的初始past_val_acc

    for i in range(num_al_steps):
        print(f'\n--- 数据收集回合 {i + 1}/{num_al_steps} ---')

        # 1. 获取当前状态和候选池信息
        current_state, candidate_indices, candidate_entropies = compute_state_for_har(
            args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
        )
        entropy_map = {idx: entropy for idx, entropy in zip(candidate_indices, candidate_entropies)}
        similarity_map = {idx: 0.5 for idx in candidate_indices}

        # 2. 生成两个候选批次 (熵 vs. 随机)
        temp_args = deepcopy(args)  # 避免修改原始args
        temp_args.al_algorithm = 'entropy'
        action_indices_A, _, _ = select_action_for_har(temp_args, None, current_state, 0, test=True)
        batch_A_indices = [candidate_indices[idx] for idx in action_indices_A.tolist()]

        temp_args.al_algorithm = 'random'
        action_indices_B, _, _ = select_action_for_har(temp_args, None, current_state, 0, test=True)
        batch_B_indices = [candidate_indices[idx] for idx in action_indices_B.tolist()]

        # 3. 评估两个批次的真实奖励
        print("评估批次 A (熵策略)...")
        net_copy_A = deepcopy(net)
        optimizer_A = torch.optim.SGD(net_copy_A.parameters(), lr=args.lr)
        temp_set_A = deepcopy(train_set)
        add_labeled_videos(args, [], batch_A_indices, temp_set_A, budget=args.budget_labels, n_ep=i)
        temp_loader_A = DataLoader(Subset(temp_set_A, list(temp_set_A.labeled_video_ids)),
                                   batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, acc_A = train_har_for_reward(net_copy_A, temp_loader_A, val_loader, optimizer_A, criterion, args)
        true_reward_A = acc_A - past_val_acc
        print(f"批次 A 奖励: {true_reward_A:.4f}")

        print("评估批次 B (随机策略)...")
        net_copy_B = deepcopy(net)
        optimizer_B = torch.optim.SGD(net_copy_B.parameters(), lr=args.lr)
        temp_set_B = deepcopy(train_set)
        add_labeled_videos(args, [], batch_B_indices, temp_set_B, budget=args.budget_labels, n_ep=i)
        temp_loader_B = DataLoader(Subset(temp_set_B, list(temp_set_B.labeled_video_ids)),
                                   batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, acc_B = train_har_for_reward(net_copy_B, temp_loader_B, val_loader, optimizer_B, criterion, args)
        true_reward_B = acc_B - past_val_acc
        print(f"批次 B 奖励: {true_reward_B:.4f}")

        # 4. 记录偏好数据
        features_A = get_batch_features([entropy_map.get(idx, 0) for idx in batch_A_indices],
                                        [similarity_map.get(idx, 0) for idx in batch_A_indices])
        features_B = get_batch_features([entropy_map.get(idx, 0) for idx in batch_B_indices],
                                        [similarity_map.get(idx, 0) for idx in batch_B_indices])

        if abs(true_reward_A - true_reward_B) > 0.001:
            alrm_preference_data.append(
                {'winner': features_A, 'loser': features_B} if true_reward_A > true_reward_B else {'winner': features_B,
                                                                                                   'loser': features_A})

        # 5. 更新主状态
        winner_batch = batch_A_indices if true_reward_A >= true_reward_B else batch_B_indices
        add_labeled_videos(args, [], winner_batch, train_set, budget=args.budget_labels, n_ep=i)
        main_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)), batch_size=args.train_batch_size,
                                 shuffle=True, num_workers=args.workers)
        _, past_val_acc = train_har_for_reward(net, main_loader, val_loader, optimizer_main, criterion, args)
        print(f"主模型已更新, 当前基准准确率: {past_val_acc:.4f}")

    # 保存最终的数据
    with open(alrm_data_path, 'wb') as f:
        pickle.dump(alrm_preference_data, f)
    print(f"第一阶段完成！偏好数据已保存至 {alrm_data_path}，共 {len(alrm_preference_data)} 对。")

    # 清理内存
    del net, train_loader, train_set, val_loader, optimizer_main, main_loader
    torch.cuda.empty_cache()

    # ===================================================================================
    #                         第二阶段: 训练ALRM (来自 train_alrm.py)
    # ===================================================================================
    print("\n" + "=" * 50)
    print("                 第二阶段: 训练主动学习奖励模型 (ALRM)")
    print("=" * 50)

    if args.reward_model_type == 'kan':
        alrm_model = KAN_ActiveLearningRewardModel(
            input_dim=4,
            grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order,
            hidden_layers=args.kan_hidden_layers
        ).cuda()
        print("使用 KAN 奖励模型进行训练。")
    elif args.reward_model_type == 'mlp':
        alrm_model = MLP_ActiveLearningRewardModel(input_dim=4, hidden_layers=[16, 8]).cuda()
        print("使用 MLP 奖励模型 (Baseline) 进行训练。")
    else:
        raise ValueError(f"未知的奖励模型类型: {args.reward_model_type}")


    optimizer_alrm = optim.Adam(alrm_model.parameters(), lr=1e-4)
    training_successful = train_reward_model(alrm_model, alrm_preference_data, optimizer_alrm)

    if not training_successful:
        print("ALRM训练失败，工作流程终止。")
        return

    # 保存ALRM
    alrm_save_path = os.path.join(exp_dir, f'{args.reward_model_type}_alrm_model.pth')
    torch.save(alrm_model.state_dict(), alrm_save_path)
    print(f"第二阶段完成！ALRM模型已保存至 {alrm_save_path}")

    del alrm_model, optimizer_alrm, alrm_preference_data
    torch.cuda.empty_cache()

    # ===================================================================================
    #                  第三阶段: 使用ALRM训练RL智能体 (来自 run_rl_with_alrm.py)
    # ===================================================================================
    print("\n" + "=" * 50)
    print("                 第三阶段: 使用ALRM训练RL智能体")
    print("=" * 50)

    # --- 重新创建模型和数据 ---
    net, policy_net, target_net = create_models(dataset=args.dataset,
                                                model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path,
                                                num_classes=args.num_classes,
                                                use_policy=True,
                                                embed_dim=args.embed_dim)

    train_loader, train_set, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        n_workers=args.workers, clip_len=args.clip_len, transform_type='c3d'
    )

    optimizer, optimizerP = create_and_load_optimizers(net=net, opt_choice=args.optimizer, lr=args.lr,
                                                       wd=args.weight_decay,
                                                       momentum=args.momentum, ckpt_path=args.ckpt_path,
                                                       exp_name_toload=args.exp_name_toload, exp_name=args.exp_name,
                                                       snapshot=args.snapshot, checkpointer=args.checkpointer,
                                                       load_opt=args.load_opt, policy_net=policy_net,
                                                       lr_dqn=args.lr_dqn)

    # --- 加载训练好的ALRM ---
    if args.reward_model_type == 'kan':
        alrm_model = KAN_ActiveLearningRewardModel(
            input_dim=4,
            grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order,
            hidden_layers=args.kan_hidden_layers
        )
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
    final_log_path = os.path.join(exp_dir, 'final_convergence_log.txt')
    logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None,
                                         log_name=os.path.basename(final_log_path))

    final_train_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)),
                                    batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)

    # --- 日志记录修正：在最终训练阶段传入logger ---
    _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net,
                                            criterion, optimizer, val_loader,
                                            best_record, logger, scheduler,  # 传入logger
                                            schedulerP, final_train=True)
    logger.close()  # 关闭日志文件

    print(f"第三阶段完成！收敛后的最终验证集准确率: {final_val_acc:.4f}")
    torch.save(policy_net.state_dict(), os.path.join(exp_dir, 'policy_final.pth'))


if __name__ == '__main__':
    main()