# 文件名: run_rl_with_alrm.py

import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy

import torch
import torch.nn as nn
import yaml
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

# --- 核心修改：导入我们需要的模型和工具 ---
from models.model_utils import create_models, get_video_candidates, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv, load_models_for_har
from utils.reward_model import KAN_ActiveLearningRewardModel, get_batch_features
# --- 修改结束 ---

from torch.utils.data import Subset, DataLoader

from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from utils.final_utils import validate
import pickle

cudnn.benchmark = False
cudnn.deterministic = True


def train_har_classifier(args, curr_epoch, train_loader, net, criterion, optimizer,
                         val_loader, best_record, logger, scheduler, schedulerP,
                         final_train=False,epochs_to_run=None):
    # (此函数保持不变)
    best_val_acc = best_record.get('top1_acc', 0.0)
    patience_counter = 0
    if epochs_to_run is None:
        epochs_to_run = args.epoch_num
    for epoch in range(curr_epoch, epochs_to_run):
        print(f'\nEpoch {epoch + 1}/{epochs_to_run}')
        net.train()
        total_loss, correct, total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Training  ", unit="batch")
        i = 0
        for inputs, labels, idx in train_pbar:
            i += 1
            inputs, labels = inputs.cuda(), labels.cuda()
            #print("Inputs shape before model:", inputs.shape)
            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]

            optimizer.zero_grad()

            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)

            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=40.0)
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size * num_clips
            current_loss = total_loss / (train_pbar.n + 1) / batch_size
            current_acc = correct / total if total > 0 else 0
            train_pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")
        train_acc = correct / total
        avg_loss = total_loss / total
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        net.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        val_pbar = tqdm(val_loader, desc=f"Validating", unit="batch")
        with torch.no_grad():
            for inputs, labels, idx in val_pbar:
                inputs, labels = inputs.cuda(), labels.cuda()
                batch_size = inputs.shape[0]
                num_clips = inputs.shape[1]

                outputs = net(inputs, return_loss=False)
                outputs = net.cls_head(outputs)
                outputs_reshaped = outputs.view(batch_size, num_clips, -1)
                outputs = outputs_reshaped.mean(dim=1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += batch_size
                current_val_acc = val_correct / val_total if val_total > 0 else 0
                val_pbar.set_postfix(acc=f"{current_val_acc:.4f}")
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        scheduler.step()
        if schedulerP is not None:
            schedulerP.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, 'best_har_model.pth'))
            print("Validation accuracy improved, saving best model.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break
    return train_acc, best_val_acc


def train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args):
    # (此函数保持不变，但在此脚本中主要用于更新主模型)
    net.train()
    for epoch in range(args.al_train_epochs):
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            # inputs = inputs.unsqueeze(1)
            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]
            optimizer.zero_grad()
            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)
            labels_repeated = labels.repeat_interleave(num_clips)
            loss = criterion(outputs, labels_repeated)
            loss.backward()
            optimizer.step()
    net.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            # inputs = inputs.unsqueeze(1)
            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]
            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)
            outputs_reshaped = outputs.view(batch_size, num_clips, -1)
            final_outputs = outputs_reshaped.mean(dim=1)
            preds = final_outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += batch_size
    if val_total == 0:
        vl_acc = 0.0
    else:
        vl_acc = val_correct / val_total
    return 0.0, vl_acc


def main(args):
    # ... (所有初始化代码，包括加载配置、设置随机种子等，保持不变) ...
    if getattr(args, 'config', None):
        print(f"加载配置文件: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    arg_key = f"{key}_{sub_key}"
                    if not hasattr(args, arg_key) or getattr(args, arg_key) is None:
                        setattr(args, arg_key, sub_value)
            else:
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    parser.save_arguments(args)
    fn = sys.argv[0].rsplit('/', 1)[-1]
    shutil.copy(sys.argv[0], os.path.join(args.ckpt_path, args.exp_name, fn))

    net, policy_net, target_net = create_models(dataset=args.dataset,
                                                model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path,
                                                num_classes=args.num_classes,
                                                use_policy=True,
                                                embed_dim=args.embed_dim)

    # ... (加载模型权重、数据、优化器、损失函数的代码保持不变) ...
    train_loader, train_set, val_loader, candidate_set = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        n_workers=args.workers, clip_len=args.clip_len, transform_type='c3d', test=args.test
    )
    criterion = nn.CrossEntropyLoss().cuda()
    kwargs_load_opt = {"net": net, "opt_choice": args.optimizer, "lr": args.lr, "wd": args.weight_decay,
                       "momentum": args.momentum, "ckpt_path": args.ckpt_path,
                       "exp_name_toload": args.exp_name_toload, "exp_name": args.exp_name,
                       "snapshot": args.snapshot, "checkpointer": args.checkpointer,
                       "load_opt": args.load_opt, "policy_net": policy_net, "lr_dqn": args.lr_dqn,
                       "al_algorithm": args.al_algorithm}
    optimizer, optimizerP = create_and_load_optimizers(**kwargs_load_opt)

    # --- 核心修改 1: 加载训练好的KAN奖励模型 ---
    print("--- 启动强化学习模式 ---")
    alrm_model_path = os.path.join(args.ckpt_path, args.exp_name, 'kan_alrm_model.pth')
    try:
        input_dim = 4  # 特征维度
        alrm_model = KAN_ActiveLearningRewardModel(input_dim=input_dim, hidden_layers=[8, 4])
        alrm_model.load_state_dict(torch.load(alrm_model_path))
        alrm_model.cuda().eval()
        print("预训练的KAN奖励模型（ALRM）加载成功！")
    except FileNotFoundError:
        print(f"错误: 找不到训练好的奖励模型: {alrm_model_path}")
        print("请先成功运行数据收集脚本和 train_alrm.py 来生成此文件。")
        exit()

    if args.train:
        print('开始训练RL智能体...')

        # --- DQN/RL 相关变量初始化 ---
        Transition = namedtuple('Transition', (
        'state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
        memory = ReplayMemory(args.rl_buffer)
        TARGET_UPDATE = 5
        steps_done = 0
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

        past_val_acc = 0.0
        num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter

        for i in range(num_al_steps):
            print(f'\n--------------- RL 主动学习回合 {i + 1}/{num_al_steps} ---------------')

            # 1. 获取当前状态和候选池信息
            current_state, candidate_indices, candidate_entropies = compute_state_for_har(
                args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
            )
            entropy_map = {idx: entropy for idx, entropy in zip(candidate_indices, candidate_entropies)}
            similarity_map = {idx: 0.5 for idx in candidate_indices}  # 简化版相似度

            # 2. RL智能体根据状态选择一个批次
            action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
            actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

            # --- 核心修改 2: 使用ALRM预测奖励 ---
            selected_entropies = [entropy_map.get(idx, 0) for idx in actual_video_ids_to_label]
            selected_similarities = [similarity_map.get(idx, 0) for idx in actual_video_ids_to_label]
            batch_features = get_batch_features(selected_entropies, selected_similarities).cuda()

            with torch.no_grad():
                predicted_reward = alrm_model(batch_features.unsqueeze(0)).item()

            print(f"KAN-ALRM 预测奖励: {predicted_reward:.4f}")

            # 4. 将选中的视频加入已标注集合
            add_labeled_videos(args, [], actual_video_ids_to_label, train_set,
                               budget=args.budget_labels, n_ep=i)

            # 5. 重建 DataLoader
            current_labeled_indices = list(train_set.labeled_video_ids)
            train_loader = DataLoader(Subset(train_set, current_labeled_indices),
                                      batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=False)
            print(f"已重建 train_loader，包含 {len(current_labeled_indices)} 个已标注视频。")

            # 6. 微调主模型 (目的：更新状态计算的基础，不再用于计算奖励)
            print('使用新选择的视频更新HAR网络...')
            _, past_val_acc = train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args)
            print(f"主模型已更新，当前真实准确率: {past_val_acc:.4f}")

            # 7. 计算下一个状态
            next_state = None
            if train_set.get_num_labeled_videos() < args.budget_labels:
                next_candidates_video_ids = train_set.get_candidates_video_ids()
                num_videos_to_sample = args.num_each_iter * args.rl_pool
                next_video_candidates_for_state = get_video_candidates(next_candidates_video_ids, train_set,
                                                                       num_videos_to_sample=num_videos_to_sample)
                next_labeled_video_ids = list(train_set.labeled_video_ids)
                next_state, _, _ = compute_state_for_har(
                    args, net, train_set, next_video_candidates_for_state, labeled_video_indices=next_labeled_video_ids
                )

            # 8. 将经验存入Replay Buffer (使用预测的奖励)
            reward_tensor = torch.tensor([predicted_reward], dtype=torch.float, device='cuda')
            # 注意：这里的push逻辑可能需要根据您的ReplayBuffer实现进行调整
            memory.push(current_state, action, next_state, reward_tensor)

            # 9. 优化策略网络
            if len(memory) >= args.dqn_bs:
                optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, GAMMA=args.dqn_gamma,
                                    BATCH_SIZE=args.dqn_bs, dqn_epochs=args.dqn_epochs)

            # 10. 更新目标网络
            if i % TARGET_UPDATE == 0:
                print('更新目标网络...')
                target_net.load_state_dict(policy_net.state_dict())

        # --- 主动学习步骤结束后的最终收敛训练 ---
        print("\n预算已用尽。在所有已选数据上训练HAR模型至收敛...")
        logger, best_record, curr_epoch = get_logfile(args.ckpt_path, args.exp_name, args.checkpointer,
                                                      args.snapshot, log_name='final_convergence_log.txt')
        final_labeled_indices = list(train_set.labeled_video_ids)
        final_train_subset = Subset(train_set, final_labeled_indices)
        final_train_loader = DataLoader(final_train_subset, batch_size=args.train_batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=False)
        _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net,
                                                criterion, optimizer, val_loader,
                                                best_record, logger, scheduler,
                                                schedulerP, final_train=True)
        print(f"收敛后的最终验证集准确率: {final_val_acc:.4f}")
        torch.save({
            'policy_net': policy_net.cpu().state_dict(),
            'optimizerP': optimizerP.state_dict(),
        }, os.path.join(args.ckpt_path, args.exp_name, 'policy_final.pth'))
        policy_net.cuda()

    if args.test:
        print('--- 启动最终测试模式 ---')

        # 1. 加载最终训练好的策略网络
        policy_path = os.path.join(args.ckpt_path, args.exp_name, 'policy_final.pth')
        try:
            checkpoint = torch.load(policy_path)
            policy_net.load_state_dict(checkpoint['policy_net'])
            policy_net.cuda().eval()
            print("最终训练好的策略网络加载成功！")
        except FileNotFoundError:
            print(f"错误: 找不到最终的策略网络: {policy_path}")
            print("请先成功运行训练模式来生成此文件。")
            exit()

        # 2. 准备一个干净的、初始状态的主模型和数据集用于评估
        #    这确保了每次评估都是从同一个公平的起点开始
        initial_net, _, _ = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                          model_ckpt_path=args.model_ckpt_path, num_classes=args.num_classes,
                                          use_policy=False)

        # 重新加载数据，得到一个干净的、只有初始标注的 train_set
        _, test_train_set, _, _ = get_data(data_path=args.data_path, tr_bs=args.train_batch_size,
                                           vl_bs=args.val_batch_size, n_workers=args.workers, clip_len=args.clip_len,
                                           )

        # 3. 获取用于记录测试结果的日志文件
        test_logger, _, _ = get_logfile(args.ckpt_path, args.exp_name, checkpointer=False, snapshot=None,
                                        log_name='test_results_log.txt')
        test_logger.set_names(['Labeled_Count', 'Validation_Accuracy'])

        # 4. 主测试循环
        num_al_steps = (args.budget_labels - test_train_set.get_num_labeled_videos()) // args.num_each_iter
        for i in range(num_al_steps):
            num_labeled_before = test_train_set.get_num_labeled_videos()
            print(f'\n----- 测试步骤 {i + 1}/{num_al_steps}: 当前已标注 {num_labeled_before}/{args.budget_labels} -----')

            # a. 获取候选池并计算状态 (使用当前的net_to_train)
            candidates_video_ids = test_train_set.get_candidates_video_ids()
            num_videos_to_sample = args.num_each_iter * args.rl_pool
            video_candidates_for_state = get_video_candidates(candidates_video_ids, test_train_set,
                                                              num_videos_to_sample=num_videos_to_sample)
            labeled_video_ids_for_state = list(test_train_set.labeled_video_ids)

            # 使用初始的、未被污染的net来计算状态
            current_state, _, _ = compute_state_for_har(args, initial_net, test_train_set, video_candidates_for_state,
                                                        labeled_video_indices=labeled_video_ids_for_state)

            # b. 从策略网络贪婪地选择动作 (test=True)
            action, _, _ = select_action_for_har(args, policy_net, current_state, 0, test=True)
            actual_video_ids_to_label = [video_candidates_for_state[idx] for idx in action.tolist()]

            # c. 将新视频加入已标注集合
            add_labeled_videos(args, [], actual_video_ids_to_label, test_train_set, budget=args.budget_labels, n_ep=i)

            # d. 训练一个新的主模型直到收敛
            print('使用当前所有已选视频，从头训练HAR网络至收敛...')
            net_to_train = deepcopy(initial_net).cuda()  # 每次都从干净的初始模型开始
            optimizer_test = torch.optim.SGD(net_to_train.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                             momentum=args.momentum)
            scheduler_test = ExponentialLR(optimizer_test, gamma=args.gamma)

            test_labeled_indices = list(test_train_set.labeled_video_ids)
            test_train_subset = Subset(test_train_set, test_labeled_indices)
            test_train_loader = DataLoader(test_train_subset, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.workers, drop_last=False)

            # 使用重量级的训练函数
            step_best_record = {'top1_acc': 0.0}
            _, val_acc = train_har_classifier(args, 0, test_train_loader, net_to_train, criterion, optimizer_test,
                                              val_loader, step_best_record, logger=None, scheduler=scheduler_test,
                                              schedulerP=None, final_train=True)

            # e. 记录性能
            num_labeled_after = test_train_set.get_num_labeled_videos()
            print(f"在标注 {num_labeled_after} 个视频后, 验证集准确率达到: {val_acc:.4f}")
            test_logger.append([num_labeled_after, val_acc])

        print("--- 测试结束 ---")
        test_logger.close()


if __name__ == '__main__':
    args = parser.get_arguments()
    main(args)