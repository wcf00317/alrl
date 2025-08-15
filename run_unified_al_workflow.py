# 文件名: wcf00317/alrl/alrl-reward_model/run_unified_al_workflow.py

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
from utils.feature_extractor import UnifiedFeatureExtractor, get_all_unlabeled_embeddings, get_all_labeled_embeddings
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import pickle

# --- 模型和工具导入 ---
from models.model_utils import create_models, get_video_candidates, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv, load_models_for_har
from utils.reward_model import KAN_ActiveLearningRewardModel, MLP_ActiveLearningRewardModel

# 1. 导入新的 UnifiedFeatureExtractor 和辅助函数，并移除旧的 get_batch_features
from utils.feature_extractor import UnifiedFeatureExtractor, get_all_unlabeled_embeddings

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
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    original_exp_name = args.exp_name
    args.exp_name = f"{original_exp_name}_{timestamp}"
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


    # 2. 在主流程开始时，根据配置实例化一个特征提取器
    feature_extractor = UnifiedFeatureExtractor(args)

    # ===================================================================================
    #                           第一阶段: 数据收集
    # ===================================================================================
    print("\n" + "=" * 50)
    print("                 第一阶段: 偏好数据收集")
    print("=" * 50)

    net, _, _ = create_models(dataset=args.dataset,
                              model_cfg_path=args.model_cfg_path,
                              model_ckpt_path=args.model_ckpt_path,
                              num_classes=args.num_classes,
                              use_policy=False)
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

    alrm_preference_data = []
    alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
    num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter
    past_val_acc = 0.0

    for i in range(num_al_steps):
        print(f'\n--- 数据收集回合 {i + 1}/{num_al_steps} ---')

        # 3. 如果需要计算高级特征，预先计算所有未标注视频的嵌入
        all_unlabeled_embeds = None
        all_labeled_embeds = None
        if 'representativeness' in feature_extractor.active_features or 'neighborhood_density' in feature_extractor.active_features:
            all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net, train_set)
        if 'labeled_distance' in feature_extractor.active_features:
            all_labeled_embeds = get_all_labeled_embeddings(args, net, train_set)


        current_state, candidate_indices, _ = compute_state_for_har(
            args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
        )

        temp_args = deepcopy(args)
        temp_args.al_algorithm = 'entropy'
        action_indices_A, _, _ = select_action_for_har(temp_args, None, current_state, 0, test=True)
        batch_A_indices = [candidate_indices[idx] for idx in action_indices_A.tolist()]

        temp_args.al_algorithm = 'random'
        action_indices_B, _, _ = select_action_for_har(temp_args, None, current_state, 0, test=True)
        batch_B_indices = [candidate_indices[idx] for idx in action_indices_B.tolist()]

        # ... (评估真实奖励的代码保持不变) ...
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

        # 4. 使用新的 feature_extractor 提取特征
        print("正在为候选批次提取特征...")
        features_A = feature_extractor.extract(batch_A_indices, net, train_set, all_unlabeled_embeds, all_labeled_embeds)
        features_B = feature_extractor.extract(batch_B_indices, net, train_set, all_unlabeled_embeds, all_labeled_embeds)

        if abs(true_reward_A - true_reward_B) > 0.001:
            alrm_preference_data.append(
                {'winner': features_A, 'loser': features_B} if true_reward_A > true_reward_B else {'winner': features_B,
                                                                                                   'loser': features_A})

        winner_batch = batch_A_indices if true_reward_A >= true_reward_B else batch_B_indices
        add_labeled_videos(args, [], winner_batch, train_set, budget=args.budget_labels, n_ep=i)
        main_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)), batch_size=args.train_batch_size,
                                 shuffle=True, num_workers=args.workers)
        _, past_val_acc = train_har_for_reward(net, main_loader, val_loader, optimizer_main, criterion, args)
        print(f"主模型已更新, 当前基准准确率: {past_val_acc:.4f}")

    with open(alrm_data_path, 'wb') as f:
        pickle.dump(alrm_preference_data, f)
    print(f"第一阶段完成！偏好数据已保存至 {alrm_data_path}，共 {len(alrm_preference_data)} 对。")

    del net, train_loader, train_set, val_loader, optimizer_main, main_loader
    torch.cuda.empty_cache()

    # ===================================================================================
    #                         第二阶段: 训练ALRM
    # ===================================================================================
    print("\n" + "=" * 50)
    print("                 第二阶段: 训练主动学习奖励模型 (ALRM)")
    print("=" * 50)

    # 5. 在创建奖励模型时，使用 feature_extractor 计算出的维度
    input_dim = feature_extractor.feature_dim

    if args.reward_model_type == 'kan':
        alrm_model = KAN_ActiveLearningRewardModel(
            input_dim=input_dim,  # 使用动态维度
            grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order,
            hidden_layers=args.kan_hidden_layers
        ).cuda()
        print("使用 KAN 奖励模型进行训练。")
    elif args.reward_model_type == 'mlp':
        # 让MLP的隐藏层大小也与输入维度相关联，更具适应性
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
    print("\n" + "=" * 50)
    print("                 第三阶段: 使用ALRM训练RL智能体")
    print("=" * 50)

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
        print(f'\n--- RL训练回合 {i + 1}/{num_al_steps} ---')

        # 6. 在RL训练循环中，同样使用新的特征提取流程
        all_unlabeled_embeds = None
        all_labeled_embeds = None
        if 'representativeness' in feature_extractor.active_features or 'neighborhood_density' in feature_extractor.active_features:
            all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net, train_set)
        if 'labeled_distance' in feature_extractor.active_features:
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
    print("\n预算用尽，在所有已选数据上训练至收敛...")
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