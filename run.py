import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import yaml
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from models.model_utils import create_models, get_video_candidates, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv, load_models_for_har
from torch.utils.data import Subset, DataLoader

from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from utils.final_utils import validate

cudnn.benchmark = False
cudnn.deterministic = True


def train_har_classifier(args, curr_epoch, train_loader, net, criterion, optimizer,
                         val_loader, best_record, logger, scheduler, schedulerP,
                         final_train=False):
    """
    用于人体行为识别（HAR）分类模型（如 C3D）的训练逻辑。
    适配 MMAction2 标准分类接口，不依赖原始 RALIS 语义分割流程。
    """
    best_val_acc = best_record.get('top1_acc', 0.0)
    patience_counter = 0
    # 这里的 epoch_num 应该是您为收敛训练设置的总轮数，例如 100 或 200
    for epoch in range(curr_epoch, args.epoch_num):
        print(f'\nEpoch {epoch + 1}/{args.epoch_num}')
        net.train()
        total_loss, correct, total = 0.0, 0, 0

        # ==== 训练 ====
        train_pbar = tqdm(train_loader, desc=f"Training  ", unit="batch")
        for inputs, labels, idx in train_pbar:
            inputs, labels = inputs.cuda(), labels.cuda()
            # 动态获取批量大小和片段数
            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]

            optimizer.zero_grad()
            # print(inputs.shape)
            #    模型内部会自动处理 [N, num_clips, ...] -> [N * num_clips, ...] 的转换。
            outputs = net(inputs, return_loss=False)  # 输出形状: [N * num_clips, num_classes]
            outputs = net.cls_head(outputs)
            # print(net)
            # print(outputs.shape)
            import sys
            # sys.exit(0)
            # 2. 【损失计算】为损失函数准备重复的标签
            labels_repeated = labels.repeat_interleave(num_clips)  # 形状变为 [N * num_clips]

            loss = criterion(outputs, labels_repeated)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_repeated).sum().item()
            total += batch_size * num_clips  # 应该统计总clips数
            # --- 在进度条上显示实时损失和准确率 ---
            current_loss = total_loss / (train_pbar.n + 1) / batch_size
            current_acc = correct / total if total > 0 else 0
            train_pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

        train_acc = correct / total
        avg_loss = total_loss / total
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        # ==== 验证 ====
        # 在 final_train 模式下，或在每个 epoch 后都进行验证
        net.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        val_pbar = tqdm(val_loader, desc=f"Validating", unit="batch")

        with torch.no_grad():
            for inputs, labels, idx in val_pbar:
                inputs, labels = inputs.cuda(), labels.cuda()
                batch_size = inputs.shape[0]
                num_clips = inputs.shape[1]

                # 1. 直接将6D张量送入模型，模型内部会自动处理Reshape
                #    模型输出 outputs 的形状是 [N * num_clips, num_classes]
                outputs = net(inputs, return_loss=False)
                outputs = net.cls_head(outputs)
                # 2. 【验证/测试策略】平均化输出
                #    从 [N * num_clips, num_classes] -> [N, num_clips, num_classes]
                outputs_reshaped = outputs.view(batch_size, num_clips, -1)
                #    沿着 num_clips 维度求平均，得到每个视频的最终预测
                outputs = outputs_reshaped.mean(dim=1)  # 最终形状变为 [N, num_classes]

                # 3. 使用平均化后的最终结果计算损失和准确率
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += batch_size

                # --- 在进度条上显示实时验证准确率 ---
                current_val_acc = val_correct / val_total if val_total > 0 else 0
                val_pbar.set_postfix(acc=f"{current_val_acc:.4f}")

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ==== 学习率调度 + 早停 ====
        scheduler.step()
        if schedulerP is not None:
            schedulerP.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 这里可以加入保存最佳模型的逻辑
            torch.save(net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, 'best_har_model.pth'))
            print("Validation accuracy improved, saving best model.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:  # args.patience 是您设置的早停耐心值
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break  # 结束训练循环

        # ==== 日志记录 ====
        # 省略了原有的日志逻辑，您可以根据需要添加回来
        # log_info = [...]
        # logger.append(log_info)

    # 如果不是因为早停而正常结束，也需要返回最终的准确率
    return train_acc, best_val_acc  # 返回一个空的 best_record 字典


def main(args):
    if getattr(args, 'config', None):
        print(f"加载配置文件: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # 合并 YAML 参数（不会覆盖已有 argparse 参数）
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

    ####------ Create experiment folder  ------####
    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))

    ####------ Print and save arguments in experiment folder  ------####
    parser.save_arguments(args)
    ####------ Copy current config file to ckpt folder ------####
    fn = sys.argv[0].rsplit('/', 1)[-1]
    shutil.copy(sys.argv[0], os.path.join(args.ckpt_path, args.exp_name, fn))

    ####------ Create segmentation, query and target networks ------####

    net, policy_net, target_net = create_models(dataset=args.dataset,
                                                model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path,
                                                num_classes=args.num_classes,
                                                use_policy=True,
                                                embed_dim=args.embed_dim)
    # return: HAR recognition network, query network, target network (same construction as query network)

    ####------ Load weights if necessary and create log file ------####
    kwargs_load = {"net": net,
                   "load_weights": args.load_weights,
                   "exp_name_toload": args.exp_name_toload,
                   "snapshot": args.snapshot,
                   "exp_name": args.exp_name,
                   "ckpt_path": args.ckpt_path,
                   "checkpointer": args.checkpointer,
                   # "exp_name_toload_rl": args.exp_name_toload_rl,
                   "policy_net": policy_net,
                   "target_net": target_net,
                   "test": args.test,
                   "dataset": args.dataset,
                   "al_algorithm": args.al_algorithm}
    # logger_dummy, curr_epoch_dummy, best_record_dummy = load_models_for_har(
    #     model=net,  # Assuming 'net' from kwargs_load maps to 'model'
    #     load_weights=args.load_weights,
    #     exp_name_toload=args.exp_name_toload,
    #     snapshot=args.snapshot,
    #     exp_name=args.exp_name,
    #     ckpt_path=args.ckpt_path,
    #     checkpointer=args.checkpointer,
    #     policy_net=policy_net,
    #     target_net=target_net,
    #     test=args.test,
    #     dataset=args.dataset,
    #     num_classes=args.num_classes
    # )
    #
    # logger_dummy, curr_epoch_dummy, best_record_dummy = load_models_for_har(**kwargs_load)  # 使用新的函数名

    ####------ Load training and validation data ------####

    train_loader, train_set, val_loader, candidate_set = get_data(
        data_path=args.data_path,
        tr_bs=args.train_batch_size,
        vl_bs=args.val_batch_size,
        n_workers=4,  # 或者 args.n_workers，如果你支持这个参数
        clip_len=args.clip_len,
        transform_type='c3d',
        test=args.test
    )

    ####------ Create loss ------####
    # criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.ignore_label).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    ####------ Create optimizers (and load them if necessary) ------####
    kwargs_load_opt = {"net": net,
                       "opt_choice": args.optimizer,
                       "lr": args.lr,
                       "wd": args.weight_decay,
                       "momentum": args.momentum,
                       "ckpt_path": args.ckpt_path,
                       "exp_name_toload": args.exp_name_toload,
                       "exp_name": args.exp_name,
                       "snapshot": args.snapshot,
                       "checkpointer": args.checkpointer,
                       "load_opt": args.load_opt,
                       "policy_net": policy_net,
                       "lr_dqn": args.lr_dqn,
                       "al_algorithm": args.al_algorithm}

    optimizer, optimizerP = create_and_load_optimizers(**kwargs_load_opt)

    #####################################################################
    ####################### TRAIN ######################
    #####################################################################
    if args.train:
        print('开始训练...')

        # 创建学习率调度器
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)
        # --- 整个过程是单次、连续的训练 ---
        # HAR网络 (net) 是增量训练的，不应在每个Episode重置。
        net.train()

        # --- DQN 相关变量 ---
        Transition = namedtuple('Transition',
                                ('state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset',
                                 'reward'))
        memory = ReplayMemory(args.rl_buffer)
        TARGET_UPDATE = 5  # 每5个RL步骤更新一次目标网络
        steps_done = 0

        # 加载策略网络和目标网络的初始状态
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        # 在任何主动学习开始前，获取初始的验证集准确率
        # _, past_val_acc = validate(val_loader, net, criterion)
        # print(f"初始验证集准确率: {past_val_acc:.4f}")
        past_val_acc = 0
        # 主循环现在迭代主动学习的“步骤”，直到满足标注预算
        # "Episode"的概念现在更像是用于日志记录和保存的计数器
        # 计算总共需要进行多少次主动学习选择
        num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter
        # print(args.budget_labels, train_set.get_num_labeled_videos(), args.num_each_iter)
        for i in range(num_al_steps):
            print(f'\n--------------- 主动学习步骤 {i + 1}/{num_al_steps} ---------------')

            # 1. 从所有未标注视频中获取候选池
            num_videos_to_sample = args.num_each_iter * args.rl_pool
            candidates_video_ids = train_set.get_candidates_video_ids()
            video_candidates_for_state = get_video_candidates(candidates_video_ids, train_set,
                                                              num_videos_to_sample=num_videos_to_sample)

            # 2. 计算当前状态 (State)
            labeled_video_ids_for_state = list(train_set.labeled_video_ids)

            current_state, _ = compute_state_for_har(
                args,
                net,
                train_set,  # 第3个参数：传入完整的数据集对象
                video_candidates_for_state,  # 第4个参数：传入候选【视频ID列表】
                labeled_video_indices=labeled_video_ids_for_state  # 第5个参数：传入已标注【视频ID列表】
            )

            # labeled_video_ids_for_state = list(train_set.labeled_video_ids)
            # current_state, _ = compute_state_for_har(args, net, video_candidates_for_state,
            #                                  candidate_set,  # 这是 train_set 的一个别名
            #                                  labeled_video_indices=labeled_video_ids_for_state,
            #                                          train_set=train_set)

            # 3. 选择动作 (Action)，即决定标注哪些视频
            action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
            actual_video_ids_to_label = [video_candidates_for_state[idx] for idx in action.tolist()]

            # 4. 将选中的视频加入到已标注集合中
            add_labeled_videos(args, [], actual_video_ids_to_label, train_set,
                               budget=args.budget_labels, n_ep=i)

            # 5. 使用更新后的已标注数据集，重建 DataLoader
            current_labeled_indices = list(train_set.labeled_video_ids)
            train_subset = Subset(train_set, current_labeled_indices)
            train_loader = DataLoader(train_subset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      num_workers=args.workers,
                                      drop_last=False)
            print(f"已重建 train_loader，包含 {len(current_labeled_indices)} 个已标注视频。")

            # 6. 在新数据上训练HAR分类器，以获得奖励信号
            print('使用新选择的视频训练HAR网络...')
            temp_epoch_num = args.epoch_num
            args.epoch_num = args.al_train_epochs  # 训练少量epoch (例如 5-10)，这是个需要您新增的参数

            _, vl_acc = train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args)
            args.epoch_num = temp_epoch_num

            # 7. 计算奖励 (Reward)
            reward = Variable(torch.Tensor([[(vl_acc - past_val_acc) * 100]])).view(-1).cuda()
            print(f"验证集准确率从 {past_val_acc:.4f} 变为 {vl_acc:.4f}。奖励: {reward.item():.4f}")
            past_val_acc = vl_acc

            # 8. 计算下一个状态 (Next State)
            next_state = None
            if train_set.get_num_labeled_videos() < args.budget_labels:
                next_candidates_video_ids = train_set.get_candidates_video_ids()
                next_video_candidates_for_state = get_video_candidates(next_candidates_video_ids, train_set,
                                                                       num_videos_to_sample=num_videos_to_sample)
                next_labeled_video_ids = list(train_set.labeled_video_ids)
                next_state, _ = compute_state_for_har(
                    args,
                    net,
                    train_set,  # 第一个参数：完整的数据集对象
                    next_video_candidates_for_state,  # 第二个参数：候选视频ID列表
                    labeled_video_indices=next_labeled_video_ids  # 第三个参数：已标注视频ID列表
                )
            else:
                print("标注预算已用尽。")

            reward_per_action = reward / args.num_each_iter

            print(f"Pushing {args.num_each_iter} transitions to replay memory...")
            for j in range(args.num_each_iter):
                # 获取第 j 个动作
                single_action = action[j].view(1)  # 取出单个动作并确保其为1D张量

                # 使用包含所有独立字段的 push 调用
                if next_state is None:
                    memory.push(current_state,
                                single_action,
                                None,
                                reward_per_action)
                else:
                    memory.push(current_state,
                                single_action,
                                next_state,
                                reward_per_action)

            # 10. 优化策略网络 (DQN)
            optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, GAMMA=args.dqn_gamma,
                                BATCH_SIZE=args.dqn_bs)

            # 11. 更新目标网络
            if i % TARGET_UPDATE == 0:
                print('更新目标网络...')
                target_net.load_state_dict(policy_net.state_dict())

        # --- 主动学习步骤结束 ---
        print("\n预算已用尽。在所有已选数据上训练HAR模型至收敛...")
        logger, best_record, curr_epoch = get_logfile(args.ckpt_path, args.exp_name, args.checkpointer,
                                                      args.snapshot, log_name='final_convergence_log.txt')

        # 重建最终的dataloader
        final_labeled_indices = list(train_set.labeled_video_ids)
        final_train_subset = Subset(train_set, final_labeled_indices)
        final_train_loader = DataLoader(final_train_subset, batch_size=args.train_batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=False)

        # 在最终选出的数据集上训练到收敛
        _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net,
                                                criterion, optimizer, val_loader,
                                                best_record, logger, scheduler,
                                                schedulerP, final_train=True)

        print(f"收敛后的最终验证集准确率: {final_val_acc:.4f}")

        # 保存最终的策略网络
        torch.save({
            'policy_net': policy_net.cpu().state_dict(),
            'optimizerP': optimizerP.state_dict(),
        }, os.path.join(args.ckpt_path, args.exp_name, 'policy_final.pth'))
        policy_net.cuda()
    #####################################################################
    ################################ TEST ########################
    #####################################################################
    if args.test:
        print('开始测试...')
        # 测试时，策略网络是固定的，不需要优化器或学习率调度器
        policy_net.eval()

        # 但HAR模型仍然需要优化器和调度器来进行训练
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)

        # 在测试评估时，HAR模型是从头开始训练的
        net.train()

        # 获取用于记录测试结果的日志文件
        logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, args.checkpointer,
                                             args.snapshot, log_name='test_log.txt')

        # 主测试循环，与训练循环类似，但没有RL更新
        while train_set.get_num_labeled_videos() < args.budget_labels:
            num_labeled = train_set.get_num_labeled_videos()
            print(f'\n----- 测试步骤: 已标注视频数 {num_labeled}/{args.budget_labels} -----')

            # 1. 获取候选池并计算状态
            num_videos_to_sample = args.num_each_iter * args.rl_pool
            candidates_video_ids = train_set.get_candidates_video_ids()
            video_candidates_for_state = get_video_candidates(candidates_video_ids, train_set,
                                                              num_videos_to_sample=num_videos_to_sample)
            labeled_video_ids_for_state = list(train_set.labeled_video_ids)
            # current_state, _ = compute_state_for_har(args, net, video_candidates_for_state,
            #                                  candidate_set,
            #                                  labeled_video_indices=labeled_video_ids_for_state)
            current_state, _ = compute_state_for_har(
                args,
                net,
                train_set,  # 第3个参数：传入完整的数据集对象
                video_candidates_for_state,  # 第4个参数：传入候选【视频ID列表】
                labeled_video_indices=labeled_video_ids_for_state  # 第5个参数：传入已标注【视频ID列表】
            )

            # 2. 从策略网络贪婪地选择动作 (test=True 禁用了探索)
            action, _, _ = select_action_for_har(args, policy_net, current_state, 0, test=True)
            actual_video_ids_to_label = [video_candidates_for_state[idx] for idx in action.tolist()]

            # 3. 将新视频加入已标注集合
            add_labeled_videos(args, [], actual_video_ids_to_label, train_set, budget=args.budget_labels, n_ep=0)

            # 4. 重建 DataLoader
            current_labeled_indices = list(train_set.labeled_video_ids)
            train_subset = Subset(train_set, current_labeled_indices)
            train_loader = DataLoader(train_subset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      num_workers=args.workers,
                                      drop_last=False)
            print(f"已重建 train_loader，包含 {len(current_labeled_indices)} 个已标注视频。")

            # 5. 训练HAR分类器并记录性能
            # 测试时，我们在每一步都训练到收敛，以观察到目前为止所选数据的最佳性能。
            print('使用当前所选视频训练HAR网络至收敛...')

            # 为每一步的收敛训练重置 best_record
            step_best_record = {'top1_acc': 0.0}
            _, val_acc = train_har_classifier(args, 0, train_loader, net,
                                              criterion, optimizer, val_loader,
                                              step_best_record, logger, scheduler,
                                              None,  # 没有策略网络的调度器
                                              final_train=True)  # 训练到收敛

            print(f"当前步骤的验证集准确率: {val_acc:.4f}")

        print("测试结束。")
        logger.close()


def train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args):
    """
    一个简化的训练函数，仅运行几个epoch以获取用于计算奖励的验证分数。
    【已修改以正确处理多片段采样】
    """
    # ==================== 训练部分 ====================
    net.train()
    # args.al_train_epochs 是一个您需要新增的参数, 例如设置为 5
    for epoch in range(args.al_train_epochs):
        for inputs, labels, _ in train_loader:
            # inputs: [N, num_clips, C, T, H, W], labels: [N]
            inputs, labels = inputs.cuda(), labels.cuda()

            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]

            optimizer.zero_grad()

            # 模型会自动处理输入的reshape，直接传入6D张量即可
            # 模型输出 outputs 的形状是 [N * num_clips, num_classes]
            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)
            # 【训练策略】重复标签
            labels_repeated = labels.repeat_interleave(num_clips)

            loss = criterion(outputs, labels_repeated)
            loss.backward()
            optimizer.step()

    # ==================== 验证部分 ====================
    net.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            # inputs: [N, num_clips, C, T, H, W], labels: [N]
            inputs, labels = inputs.cuda(), labels.cuda()

            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]

            # 模型输出 outputs 的形状是 [N * num_clips, num_classes]
            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)
            # 【验证/测试策略】平均化输出
            # 从 [N * num_clips, num_classes] -> [N, num_clips, num_classes]
            outputs_reshaped = outputs.view(batch_size, num_clips, -1)
            # 沿着 num_clips 维度求平均
            final_outputs = outputs_reshaped.mean(dim=1)  # 最终形状变为 [N, num_classes]

            # 使用平均化后的结果计算验证指标
            preds = final_outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += batch_size

    # 确保 val_total 不为0，避免除零错误
    if val_total == 0:
        vl_acc = 0.0
    else:
        vl_acc = val_correct / val_total

    # 返回 (训练集准确率, 验证集准确率)
    # 因为我们只关心验证准确率，所以训练准确率可以返回一个占位符
    return 0.0, vl_acc


# def train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args):
#     """一个简化的训练函数，仅运行几个epoch以获取用于计算奖励的验证分数。"""
#     net.train()
#     # args.al_train_epochs 是一个您需要新增的参数, 例如设置为 5
#     for epoch in range(args.al_train_epochs):
#         for inputs, labels, _ in train_loader:
#             inputs, labels = inputs.cuda(), labels.cuda()
#             optimizer.zero_grad()
#             # inputs = inputs.unsqueeze(dim=1)
#             outputs = net(inputs, return_loss=False)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#     # 获取验证集准确率
#     net.eval()
#     val_correct, val_total = 0, 0
#     with torch.no_grad():
#         for inputs, labels, _ in val_loader:
#             inputs, labels = inputs.cuda(), labels.cuda()
#             # inputs = inputs.unsqueeze(dim=1)
#             outputs = net(inputs, return_loss=False)
#             preds = outputs.argmax(dim=1)
#             val_correct += (preds == labels).sum().item()
#             val_total += inputs.size(0)
#
#     # 返回 (训练集准确率, 验证集准确率)
#     return 0.0, val_correct / val_total

if __name__ == '__main__':
    ####------ Parse arguments from console  ------####
    args = parser.get_arguments()
    main(args)
