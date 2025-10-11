# wcf00317/alrl/alrl-reward_model/run_bald.py

import os
import sys
import shutil
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from copy import deepcopy

# --- 复用您项目中的模块 ---
from models.model_utils import create_models, add_labeled_videos
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
import utils.parser as parser
from torch.utils.data import Subset, DataLoader
from run_rl_with_alrm import train_har_classifier

# --- 核心导入: 导入 BALD 评分函数 ---
from utils.al_scoring import compute_bald_score

cudnn.benchmark = False
cudnn.deterministic = True


def main(args):
    # --- 1. 初始化和配置加载 (与 run_random.py / run_entropy.py 一致) ---
    if getattr(args, 'config', None):
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

    # --- 2. 创建模型和加载数据 (一致) ---
    net, _, _ = create_models(dataset=args.dataset,
                              model_cfg_path=args.model_cfg_path,
                              model_ckpt_path=args.model_ckpt_path,
                              num_classes=args.num_classes,
                              use_policy=False,
                              embed_dim=args.embed_dim)
    net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    _, train_set, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        n_workers=args.workers, clip_len=args.clip_len, model_type=args.model_type, dataset_name=args.dataset
    )

    # --- 3. 设置日志 (一致) ---
    bald_logger, _, _ = get_logfile(args.ckpt_path, args.exp_name, checkpointer=False, snapshot=None,
                                    log_name='bald_baseline_log.txt')
    bald_logger.set_names(['Labeled_Count', 'Validation_Accuracy'])

    # --- 4. 初始训练阶段 (一致) ---
    initial_labeled_indices = list(train_set.labeled_video_ids)
    initial_train_subset = Subset(train_set, initial_labeled_indices)
    initial_train_loader = DataLoader(initial_train_subset, batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=False)
    print(f"--- 开始初始训练阶段, 当前样本数: {len(initial_train_subset)} ---")

    optimizer = create_and_load_optimizers(net=net, opt_choice=args.optimizer, lr=args.lr,
                                           wd=args.weight_decay, momentum=args.momentum, ckpt_path=args.ckpt_path,
                                           exp_name_toload=None, exp_name=args.exp_name, snapshot=None,
                                           checkpointer=False, load_opt=False)[0]
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    best_record = {'top1_acc': 0.0}

    _, initial_val_acc = train_har_classifier(args, 0, initial_train_loader, net,
                                              criterion, optimizer, val_loader, best_record,
                                              logger=None, scheduler=scheduler, schedulerP=None, final_train=True)

    num_initial_labeled = len(initial_labeled_indices)
    print(f"初始训练完成！已标注 {num_initial_labeled} 个样本, 验证集准确率: {initial_val_acc:.4f}")
    bald_logger.append([num_initial_labeled, initial_val_acc])

    # --- 5. 增量主动学习循环 ---
    num_al_steps = (args.budget_labels - num_initial_labeled) // args.num_each_iter
    print(f"\n--- 开始基于 BALD 的增量学习循环，共 {num_al_steps} 轮 ---")

    for i in range(num_al_steps):
        num_labeled_before = train_set.get_num_labeled_videos()
        print(f'\n----- BALD 选择第 {i + 1}/{num_al_steps} 轮: 当前已标注 {num_labeled_before}/{args.budget_labels} -----')

        unlabeled_indices = train_set.get_candidates_video_ids()

        # a. 核心修改: 调用 BALD 评分函数
        # 该函数内部会使用MC-Dropout进行多次前向传播来计算不确定性
        bald_scores = compute_bald_score(net, unlabeled_indices, train_set)

        # b. 选择分数最高的样本
        k = min(args.num_each_iter, len(unlabeled_indices))
        _, top_indices = torch.topk(bald_scores, k=k)
        selected_indices = [unlabeled_indices[idx] for idx in top_indices.tolist()]

        print(f"基于 BALD 选择了 {len(selected_indices)} 个视频。")
        add_labeled_videos(args, [], selected_indices, train_set, budget=args.budget_labels, n_ep=i)

        # c. 后续训练和日志记录 (与 run_random / run_entropy 一致)
        print('在扩充后的数据集上继续训练 HAR 网络...')
        current_labeled_indices = list(train_set.labeled_video_ids)
        train_subset = Subset(train_set, current_labeled_indices)
        current_train_loader = DataLoader(train_subset, batch_size=args.train_batch_size, shuffle=True,
                                          num_workers=args.workers, drop_last=False)

        _, val_acc = train_har_classifier(args, 0, current_train_loader, net,
                                          criterion, optimizer, val_loader, best_record,
                                          logger=None, scheduler=scheduler, schedulerP=None,
                                          final_train=True, epochs_to_run=args.al_train_epochs)

        num_labeled_after = train_set.get_num_labeled_videos()
        print(f"标注数量达到 {num_labeled_after} 后, 验证集准确率为: {val_acc:.4f}")
        bald_logger.append([num_labeled_after, val_acc])

    print("\n--- 基于 BALD 的基线测试结束 ---")
    bald_logger.close()


if __name__ == '__main__':
    args = parser.get_arguments()
    main(args)