# filename: run_unified_ebm_workflow.py

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

# --- 【核心】导入您的 EBM 模块 ---
# (假设 train_ebm.py 位于 models/ 目录下)
from models.train_ebm import train_ebm_reward_model, load_ebm_scorer, predict_ebm_reward

cudnn.benchmark = False
cudnn.deterministic = True


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


def main():
    # --- 0. 初始化和配置加载 (与 unified_al workflow 一致) ---
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

    feature_extractor = UnifiedFeatureExtractor(args)

    # ===================================================================================
    #              第一阶段: 偏好数据收集 (照抄 run_unified_al_workflow.py)
    # ===================================================================================
    print("\n" + "=" * 25 + "  STAGE 1: PREFERENCE DATA COLLECTION  " + "=" * 25)

    net_stage1, _, _ = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                     model_ckpt_path=args.model_ckpt_path, num_classes=args.num_classes,
                                     use_policy=False)
    net_stage1.cuda()

    _, train_set, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        initial_labeled_ratio=args.initial_labeled_ratio
    )
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_main = \
    create_and_load_optimizers(net=net_stage1, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
                               momentum=args.momentum, ckpt_path=args.ckpt_path,
                               exp_name_toload=args.exp_name_toload, exp_name=args.exp_name,
                               snapshot=args.snapshot, checkpointer=args.checkpointer,
                               load_opt=args.load_opt)[0]

    alrm_preference_data = []
    num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter
    past_val_acc = 0.0

    for i in range(num_al_steps):
        print(f'\n--- 数据收集回合 {i + 1}/{num_al_steps} ---')

        all_unlabeled_embeds, all_labeled_embeds = None, None
        if 'representativeness' in feature_extractor.active_features or 'neighborhood_density' in feature_extractor.active_features:
            all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net_stage1, train_set)
        if 'labeled_distance' in feature_extractor.active_features:
            all_labeled_embeds = get_all_labeled_embeddings(args, net_stage1, train_set)

        current_state, candidate_indices, _ = compute_state_for_har(
            args, net_stage1, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
        )

        temp_args = deepcopy(args)
        temp_args.al_algorithm = 'entropy'
        action_indices_A, _, _ = select_action_for_har(temp_args, None, current_state, 0, test=True)
        batch_A_indices = [candidate_indices[idx] for idx in action_indices_A.tolist()]

        temp_args.al_algorithm = 'random'
        action_indices_B, _, _ = select_action_for_har(temp_args, None, current_state, 0, test=True)
        batch_B_indices = [candidate_indices[idx] for idx in action_indices_B.tolist()]

        print("评估批次 A (熵策略)...")
        net_copy_A = deepcopy(net_stage1)
        optimizer_A = torch.optim.SGD(net_copy_A.parameters(), lr=args.lr)
        temp_set_A = deepcopy(train_set)
        add_labeled_videos(args, [], batch_A_indices, temp_set_A, budget=args.budget_labels, n_ep=i)
        temp_loader_A = DataLoader(Subset(temp_set_A, list(temp_set_A.labeled_video_ids)),
                                   batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, acc_A = train_har_for_reward(net_copy_A, temp_loader_A, val_loader, optimizer_A, criterion, args)
        true_reward_A = acc_A - past_val_acc

        print("评估批次 B (随机策略)...")
        net_copy_B = deepcopy(net_stage1)
        optimizer_B = torch.optim.SGD(net_copy_B.parameters(), lr=args.lr)
        temp_set_B = deepcopy(train_set)
        add_labeled_videos(args, [], batch_B_indices, temp_set_B, budget=args.budget_labels, n_ep=i)
        temp_loader_B = DataLoader(Subset(temp_set_B, list(temp_set_B.labeled_video_ids)),
                                   batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, acc_B = train_har_for_reward(net_copy_B, temp_loader_B, val_loader, optimizer_B, criterion, args)
        true_reward_B = acc_B - past_val_acc

        features_A = feature_extractor.extract(batch_A_indices, net_stage1, train_set, all_unlabeled_embeds,
                                               all_labeled_embeds)
        features_B = feature_extractor.extract(batch_B_indices, net_stage1, train_set, all_unlabeled_embeds,
                                               all_labeled_embeds)

        if abs(true_reward_A - true_reward_B) > 0.001:
            alrm_preference_data.append(
                {'winner': features_A, 'loser': features_B} if true_reward_A > true_reward_B else {'winner': features_B,
                                                                                                   'loser': features_A})

        winner_batch = batch_A_indices if true_reward_A >= true_reward_B else batch_B_indices
        add_labeled_videos(args, [], winner_batch, train_set, budget=args.budget_labels, n_ep=i)
        main_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)), batch_size=args.train_batch_size,
                                 shuffle=True, num_workers=args.workers)
        _, past_val_acc = train_har_for_reward(net_stage1, main_loader, val_loader, optimizer_main, criterion, args)
        print(f"主模型已更新, 当前基准准确率: {past_val_acc:.4f}")

    alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
    with open(alrm_data_path, 'wb') as f:
        pickle.dump(alrm_preference_data, f)
    print(f"\n--- STAGE 1 COMPLETE --- 偏好数据已保存至 {alrm_data_path}，共 {len(alrm_preference_data)} 对。")
    del net_stage1, train_set, val_loader, optimizer_main, main_loader
    torch.cuda.empty_cache()

    # ===================================================================================
    #                         第二阶段: 训练 EBM 奖励模型 (新)
    # ===================================================================================
    print("\n" + "=" * 25 + "  STAGE 2: EBM REWARD MODEL TRAINING  " + "=" * 25)

    # 【核心替换】调用我们导入的EBM训练函数
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

    _, train_set_rl, val_loader_rl, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        initial_labeled_ratio=args.initial_labeled_ratio
    )

    optimizer_stage3, optimizerP = create_and_load_optimizers(
        net=net_stage3, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
        momentum=args.momentum, policy_net=policy_net, lr_dqn=args.lr_dqn,
        ckpt_path=args.ckpt_path, exp_name=args.exp_name,
        exp_name_toload=args.exp_name_toload, snapshot=args.snapshot,
        checkpointer=args.checkpointer, load_opt=args.load_opt
    )

    # 【核心替换】加载EBM计分器
    ebm_scorer = load_ebm_scorer(exp_dir)

    Transition = namedtuple('Transition',
                            ('state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
    memory = ReplayMemory(args.rl_buffer)
    TARGET_UPDATE = 5
    steps_done = 0
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    scheduler = ExponentialLR(optimizer_stage3, gamma=args.gamma)
    schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

    num_al_steps_rl = (args.budget_labels - train_set_rl.get_num_labeled_videos()) // args.num_each_iter
    for i in range(num_al_steps_rl):
        print(f'\n--- RL训练回合 {i + 1}/{num_al_steps_rl} ---')

        all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net_stage3, train_set_rl)
        all_labeled_embeds = get_all_labeled_embeddings(args, net_stage3, train_set_rl)

        current_state, candidate_indices, _ = compute_state_for_har(args, net_stage3, train_set_rl,
                                                                    train_set_rl.get_candidates_video_ids(),
                                                                    list(train_set_rl.labeled_video_ids))
        action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
        actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

        batch_features = feature_extractor.extract(actual_video_ids_to_label, net_stage3, train_set_rl,
                                                   all_unlabeled_embeds,
                                                   all_labeled_embeds).cuda()

        # 【核心替换】使用 EBM 预测奖励
        predicted_reward = predict_ebm_reward(ebm_scorer, batch_features)
        print(f"EBM 预测奖励 (P(x > μ)): {predicted_reward:.4f}")

        add_labeled_videos(args, [], actual_video_ids_to_label, train_set_rl, budget=args.budget_labels, n_ep=i)

        current_labeled_indices = list(train_set_rl.labeled_video_ids)
        train_loader_rl = DataLoader(Subset(train_set_rl, current_labeled_indices),
                                     batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, _ = train_har_for_reward(net_stage3, train_loader_rl, val_loader_rl, optimizer_stage3, criterion, args)

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

    _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net_stage3, criterion, optimizer_stage3,
                                            val_loader_rl, best_record, logger, scheduler, schedulerP,
                                            final_train=True)
    logger.close()

    print(f"\n--- STAGE 3 COMPLETE --- 收敛后的最终验证集准确率: {final_val_acc:.4f}")
    torch.save(policy_net.state_dict(), os.path.join(exp_dir, 'policy_final.pth'))


if __name__ == '__main__':
    main()