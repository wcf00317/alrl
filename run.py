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
import pickle
from copy import deepcopy
from torch.utils.data import Subset, DataLoader
from utils.reward_model import get_batch_features
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


def train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args):
    """
    一个简化的训练函数，仅运行几个epoch以获取用于计算奖励的验证分数。
    这个函数是专门为主动学习的奖励计算而设计的，追求速度而非模型的完全收敛。
    """
    # ==================== 训练部分 ====================
    net.train()
    # args.al_train_epochs 是您在配置文件中设置的微调周期数 (例如 10 或 15)
    for epoch in range(args.al_train_epochs):
        # 我们只做训练，可以省略tqdm来减少日志输出
        for inputs, labels, _ in train_loader:
            # inputs: [N, num_clips, C, T, H, W], labels: [N]
            inputs, labels = inputs.cuda(), labels.cuda()

            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]

            optimizer.zero_grad()

            # 模型会自动处理输入的reshape
            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)  # 获取分类头的输出

            # 为多片段输入重复标签
            labels_repeated = labels.repeat_interleave(num_clips)

            loss = criterion(outputs, labels_repeated)
            loss.backward()
            optimizer.step()

    # ==================== 验证部分 ====================
    net.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            # inputs: [N, num_clips, C, T, H, W], labels: [N]
            inputs, labels = inputs.cuda(), labels.cuda()

            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]

            # 模型输出 outputs 的形状是 [N * num_clips, num_classes]
            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)

            # 【验证策略】平均化输出
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
    # 因为我们只关心验证准确率作为奖励信号，所以训练准确率可以返回一个占位符 0.0
    return 0.0, vl_acc
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


    if args.train:
        print('--- 开始数据收集模式 ---')
        # 计算总共需要进行多少次主动学习选择
        # (总预算 - 初始已标注数) / 每轮选择数
        num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter
        print(f"总标注预算: {args.budget_labels}, 初始已标注: {train_set.get_num_labeled_videos()}, 每轮选择: {args.num_each_iter}个")
        print(f"计划执行 {num_al_steps} 个数据收集回合。")
        # --- 定义结束 ---
        # --- 新增：初始化用于存储偏好数据的列表 ---
        alrm_preference_data = []
        alrm_data_path = os.path.join(args.ckpt_path, args.exp_name, 'alrm_preference_data.pkl')

        # --- 奖励计算相关的微调模型，需要一个独立的优化器 ---
        optimizer_for_reward = create_and_load_optimizers(**kwargs_load_opt)[0]

        past_val_acc = 0.0
        for i in range(num_al_steps):
            print(f'\n--------------- 数据收集回合 {i + 1}/{num_al_steps} ---------------')

            # 1. 获取当前状态和候选池信息
            # 注意： compute_state_for_har 现在需要返回熵值
            current_state, candidate_indices, candidate_entropies = compute_state_for_har(
                args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
            )
            entropy_map = {idx: entropy for idx, entropy in zip(candidate_indices, candidate_entropies)}
            # 简化版相似度，后续可以替换为真实计算
            similarity_map = {idx: 0.5 for idx in candidate_indices}

            # 2. 生成两个候选批次：一个基于熵，一个随机
            # 批次A：熵策略
            original_algo = args.al_algorithm
            args.al_algorithm = 'entropy'
            action_indices_A, _, _ = select_action_for_har(args, policy_net, current_state, 0, test=True)
            batch_A_indices = [candidate_indices[idx] for idx in action_indices_A.tolist()]

            # 批次B：随机策略
            args.al_algorithm = 'random'
            action_indices_B, _, _ = select_action_for_har(args, policy_net, current_state, 0, test=True)
            batch_B_indices = [candidate_indices[idx] for idx in action_indices_B.tolist()]
            args.al_algorithm = original_algo  # 恢复原设置

            # 3. 分别评估两个批次的真实奖励
            print("评估批次 A (熵策略)...")
            net_copy_A = deepcopy(net)  # 使用网络副本，避免互相影响
            optimizer_A = torch.optim.SGD(net_copy_A.parameters(), lr=args.lr)  # 为副本创建独立优化器
            temp_set_A = deepcopy(train_set)
            add_labeled_videos(args, [], batch_A_indices, temp_set_A, budget=args.budget_labels, n_ep=i)
            temp_loader_A = DataLoader(Subset(temp_set_A, list(temp_set_A.labeled_video_ids)),
                                       batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
            _, acc_A = train_har_for_reward(net_copy_A, temp_loader_A, val_loader, optimizer_A, criterion, args)
            true_reward_A = acc_A - past_val_acc
            print(f"批次 A 奖励 (Acc_Gain): {true_reward_A:.4f}")

            print("评估批次 B (随机策略)...")
            net_copy_B = deepcopy(net)
            optimizer_B = torch.optim.SGD(net_copy_B.parameters(), lr=args.lr)
            temp_set_B = deepcopy(train_set)
            add_labeled_videos(args, [], batch_B_indices, temp_set_B, budget=args.budget_labels, n_ep=i)
            temp_loader_B = DataLoader(Subset(temp_set_B, list(temp_set_B.labeled_video_ids)),
                                       batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
            _, acc_B = train_har_for_reward(net_copy_B, temp_loader_B, val_loader, optimizer_B, criterion, args)
            true_reward_B = acc_B - past_val_acc
            print(f"批次 B 奖励 (Acc_Gain): {true_reward_B:.4f}")

            # 4. 记录偏好对
            features_A = get_batch_features([entropy_map.get(idx, 0) for idx in batch_A_indices],
                                            [similarity_map.get(idx, 0) for idx in batch_A_indices])
            features_B = get_batch_features([entropy_map.get(idx, 0) for idx in batch_B_indices],
                                            [similarity_map.get(idx, 0) for idx in batch_B_indices])

            if abs(true_reward_A - true_reward_B) > 0.001:  # 过滤噪声
                if true_reward_A > true_reward_B:
                    alrm_preference_data.append({'winner': features_A, 'loser': features_B})
                else:
                    alrm_preference_data.append({'winner': features_B, 'loser': features_A})

            # 5. 更新主状态
            winner_batch = batch_A_indices if true_reward_A >= true_reward_B else batch_B_indices
            add_labeled_videos(args, [], winner_batch, train_set, budget=args.budget_labels, n_ep=i)
            main_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)), batch_size=args.train_batch_size,
                                     shuffle=True, num_workers=args.workers)
            _, past_val_acc = train_har_for_reward(net, main_loader, val_loader, optimizer, criterion, args)
            print(f"主模型已更新, 当前基准准确率: {past_val_acc:.4f}")

            # 6. 定期保存数据
            if i % 5 == 0 or i == num_al_steps - 1:
                with open(alrm_data_path, 'wb') as f:
                    pickle.dump(alrm_preference_data, f)
                print(f"偏好数据已保存, 当前共 {len(alrm_preference_data)} 对。")

        print(f"--- 数据收集完成！请运行 train_alrm.py 来训练奖励模型。 ---")
if __name__ == '__main__':
    ####------ Parse arguments from console  ------####
    args = parser.get_arguments()
    main(args)
