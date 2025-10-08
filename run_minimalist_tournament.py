# 文件名: wcf00317/alrl/alrl-reward_model/run_unified_alrl_workflow.py

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
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import pickle

# --- 核心导入 ---
from models.minimalist_tournament_selection import MinimalistTournamentSelector
from utils.feature_extractor import UnifiedFeatureExtractor, get_all_unlabeled_embeddings, get_all_labeled_embeddings
from models.model_utils import create_models, add_labeled_videos, optimize_model_conv, select_action_for_har, \
    compute_state_for_har
from utils.reward_model import KAN_ActiveLearningRewardModel, MLP_ActiveLearningRewardModel
from torch.utils.data import Subset, DataLoader
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward
from train_alrm import train_reward_model

# 设置随机种子
cudnn.benchmark = False
cudnn.deterministic = True


class VerboseLogger:
    """
    一个简单的日志类，用于将所有标准输出 (print) 实时重定向到文件。
    它会同时在控制台显示输出并在文件中记录。
    """

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


def main():
    # --- 0. 初始化和配置加载 ---
    args = parser.get_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 创建带时间戳的唯一实验文件夹
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.exp_name = f"{args.exp_name}_{timestamp}"
    exp_dir = os.path.join(args.ckpt_path, args.exp_name)
    check_mkdir(args.ckpt_path)
    check_mkdir(exp_dir)

    # 启用日志重定向
    verbose_log_path = os.path.join(exp_dir, 'verbose_run_log.txt')
    sys.stdout = VerboseLogger(verbose_log_path)

    parser.save_arguments(args)
    shutil.copy(sys.argv[0], os.path.join(exp_dir, sys.argv[0].rsplit('/', 1)[-1]))
    if hasattr(args, 'config') and args.config and os.path.exists(args.config):
        shutil.copy(args.config, os.path.join(exp_dir, os.path.basename(args.config)))
    print(f"实验将保存在: {exp_dir}")

    # ===================================================================================
    #              第一阶段: 使用锦标赛收集高质量的偏好数据
    # ===================================================================================
    print("\n" + "=" * 25 + "  STAGE 1: TOURNAMENT DATA COLLECTION  " + "=" * 25)

    net_stage1, _, _ = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                     model_ckpt_path=args.model_ckpt_path, num_classes=args.num_classes,
                                     use_policy=False)
    net_stage1.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    _, train_set, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        initial_labeled_ratio=args.initial_labeled_ratio
    )

    optimizer_main = create_and_load_optimizers(
        net=net_stage1,
        opt_choice=args.optimizer,
        lr=args.lr,
        wd=args.weight_decay,
        momentum=args.momentum,
        ckpt_path=args.ckpt_path,
        exp_name_toload=None,  # 在第一阶段，我们从头开始，不加载旧的优化器状态
        exp_name=args.exp_name,
        snapshot=None,
        checkpointer=False,
        load_opt=False
    )[0]

    print("--- 在初始已标注数据上进行训练以获得基线模型... ---")
    initial_labeled_indices = list(train_set.labeled_video_ids)
    initial_train_loader = DataLoader(Subset(train_set, initial_labeled_indices), batch_size=args.train_batch_size,
                                      shuffle=True, num_workers=args.workers)
    _, past_val_acc = train_har_for_reward(net_stage1, initial_train_loader, val_loader, optimizer_main, criterion,
                                           args)
    print(f"初始基线准确率: {past_val_acc:.4f}")

    alrm_preference_data = []
    num_al_steps = (args.budget_labels - len(initial_labeled_indices)) // args.num_each_iter
    feature_extractor = UnifiedFeatureExtractor(args)

    for i in range(num_al_steps):
        print(f'\n>>>>> 数据收集回合 {i + 1}/{num_al_steps} <<<<<')
        selector = MinimalistTournamentSelector(net_stage1, train_set, val_loader, criterion, args, past_val_acc)
        finalist_batches = selector.get_finalists(budget=args.num_each_iter)

        finalist_rewards = []
        finalist_features = []
        for j, batch in enumerate(tqdm(finalist_batches, desc="评估决赛选手批次")):
            net_copy = deepcopy(net_stage1)
            optimizer_temp = torch.optim.SGD(net_copy.parameters(), lr=args.lr)
            temp_set = deepcopy(train_set)
            add_labeled_videos(args, [], batch, temp_set, budget=args.budget_labels, n_ep=-1)
            temp_loader = DataLoader(Subset(temp_set, list(temp_set.labeled_video_ids)),
                                     batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
            _, acc = train_har_for_reward(net_copy, temp_loader, val_loader, optimizer_temp, criterion, args)
            finalist_rewards.append(acc - past_val_acc)
            finalist_features.append(
                feature_extractor.extract(batch, net_stage1, train_set, selector.precomputed_data['embeddings'],
                                          selector.precomputed_data['labeled_embeddings']))

        ranked_indices = np.argsort(finalist_rewards)[::-1]
        for idx1 in range(len(ranked_indices)):
            for idx2 in range(idx1 + 1, len(ranked_indices)):
                winner_idx, loser_idx = ranked_indices[idx1], ranked_indices[idx2]
                if finalist_rewards[winner_idx] > finalist_rewards[loser_idx] + 0.01:
                    alrm_preference_data.append(
                        {'winner': finalist_features[winner_idx], 'loser': finalist_features[loser_idx]})

        champion_batch = finalist_batches[ranked_indices[0]]
        add_labeled_videos(args, [], champion_batch, train_set, budget=args.budget_labels, n_ep=i)
        main_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)), batch_size=args.train_batch_size,
                                 shuffle=True, num_workers=args.workers)
        _, past_val_acc = train_har_for_reward(net_stage1, main_loader, val_loader, optimizer_main, criterion, args)
        print(f"主模型已更新, 新的基准准确率: {past_val_acc:.4f}")

    alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
    with open(alrm_data_path, 'wb') as f:
        pickle.dump(alrm_preference_data, f)
    print(f"--- STAGE 1 COMPLETE --- 偏好数据已保存至 {alrm_data_path}，共 {len(alrm_preference_data)} 对。")
    del net_stage1, train_set, val_loader, optimizer_main, main_loader, selector
    torch.cuda.empty_cache()

    # ===================================================================================
    #                         第二阶段: 训练ALRM
    # ===================================================================================
    print("\n" + "=" * 25 + "  STAGE 2: ALRM TRAINING  " + "=" * 25)

    input_dim = feature_extractor.feature_dim
    if args.reward_model_type == 'kan':
        alrm_model = KAN_ActiveLearningRewardModel(
            input_dim=input_dim,  # 使用动态维度
            grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order,
            hidden_layers=args.kan_hidden_layers
        ).cuda()
    else:
        hidden_dim1 = max(16, input_dim * 4)
        hidden_dim2 = max(8, input_dim * 2)
        alrm_model = MLP_ActiveLearningRewardModel(input_dim=input_dim,
                                                   hidden_layers=[hidden_dim1, hidden_dim2]).cuda()

    optimizer_alrm = optim.Adam(alrm_model.parameters(), lr=1e-4)
    train_reward_model(alrm_model, alrm_preference_data, optimizer_alrm)

    alrm_save_path = os.path.join(exp_dir, f'{args.reward_model_type}_alrm_model.pth')
    torch.save(alrm_model.state_dict(), alrm_save_path)
    print(f"--- STAGE 2 COMPLETE --- ALRM模型已保存至 {alrm_save_path}")

    # ===================================================================================
    #                  第三阶段: 使用ALRM训练RL智能体
    # ===================================================================================
    print("\n" + "=" * 25 + "  STAGE 3: RL AGENT TRAINING  " + "=" * 25)

    net_stage3, policy_net, target_net = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                                       model_ckpt_path=args.model_ckpt_path,
                                                       num_classes=args.num_classes,
                                                       use_policy=True, embed_dim=args.embed_dim)
    _, train_set_stage3, val_loader_stage3, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        initial_labeled_ratio=args.initial_labeled_ratio
    )
    optimizer_stage3, optimizerP = create_and_load_optimizers(
        net=net_stage3,
        opt_choice=args.optimizer,
        lr=args.lr,
        wd=args.weight_decay,
        momentum=args.momentum,
        ckpt_path=args.ckpt_path,
        exp_name_toload=args.exp_name_toload,  # 从args中获取
        exp_name=args.exp_name,
        snapshot=args.snapshot,  # 从args中获取
        checkpointer=args.checkpointer,  # 从args中获取
        load_opt=args.load_opt,  # 从args中获取
        policy_net=policy_net,
        lr_dqn=args.lr_dqn
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
    scheduler = ExponentialLR(optimizer_stage3, gamma=args.gamma)
    schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

    num_al_steps = (args.budget_labels - train_set_stage3.get_num_labeled_videos()) // args.num_each_iter
    for i in range(num_al_steps):
        print(f'\n--- RL训练回合 {i + 1}/{num_al_steps} ---')
        all_unlabeled_embeds = None
        all_labeled_embeds = None
        if 'representativeness' in feature_extractor.active_features or 'neighborhood_density' in feature_extractor.active_features:
            all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net_stage3, train_set_stage3)
        if 'labeled_distance' in feature_extractor.active_features:
            all_labeled_embeds = get_all_labeled_embeddings(args, net_stage3, train_set_stage3)

        current_state, candidate_indices, _ = compute_state_for_har(args, net_stage3, train_set_stage3,
                                                                    train_set_stage3.get_candidates_video_ids(),
                                                                    list(train_set_stage3.labeled_video_ids))
        action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
        actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

        batch_features = feature_extractor.extract(actual_video_ids_to_label, net_stage3, train_set_stage3, all_unlabeled_embeds, all_labeled_embeds).cuda()
        with torch.no_grad():
            predicted_reward = alrm_model(batch_features.unsqueeze(0)).item()

        add_labeled_videos(args, [], actual_video_ids_to_label, train_set_stage3, budget=args.budget_labels, n_ep=i)

        current_labeled_indices = list(train_set_stage3.labeled_video_ids)
        train_loader_stage3 = DataLoader(Subset(train_set_stage3, current_labeled_indices),
                                         batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, _ = train_har_for_reward(net_stage3, train_loader_stage3, val_loader_stage3, optimizer_stage3, criterion,
                                    args)

        next_state, _, _ = compute_state_for_har(args, net_stage3, train_set_stage3,
                                                 train_set_stage3.get_candidates_video_ids(), list(
                train_set_stage3.labeled_video_ids)) if train_set_stage3.get_num_labeled_videos() < args.budget_labels else (
        None, None, None)

        memory.push(current_state, action, next_state,
                    torch.tensor([predicted_reward], dtype=torch.float, device='cuda'))
        if len(memory) >= args.dqn_bs:
            optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, GAMMA=args.dqn_gamma,
                                BATCH_SIZE=args.dqn_bs)
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("\n预算用尽，在所有已选数据上训练至收敛...")
    final_log_path = os.path.join(exp_dir, 'final_convergence_log.txt')
    logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None, log_name=os.path.basename(final_log_path))
    final_train_loader = DataLoader(Subset(train_set_stage3, list(train_set_stage3.labeled_video_ids)),
                                    batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)

    _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net_stage3, criterion, optimizer_stage3,
                                            val_loader_stage3, best_record, logger, scheduler, schedulerP,
                                            final_train=True)
    logger.close()

    print(f"\n--- STAGE 3 COMPLETE --- 收敛后的最终验证集准确率: {final_val_acc:.4f}")
    torch.save(policy_net.state_dict(), os.path.join(exp_dir, 'policy_final.pth'))


if __name__ == '__main__':
    main()