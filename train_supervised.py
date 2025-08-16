# 文件名: train_full_data.py (Corrected)

import os
import sys
import shutil
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR,StepLR,MultiStepLR
from tqdm import tqdm
from torch.utils.data import DataLoader # <-- THIS LINE WAS MISSING

# 从您的项目中导入必要的模块
from models.model_utils import create_models
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier # 复用已有的训练函数

# 设置随机种子以保证结果可复现
cudnn.benchmark = False
cudnn.deterministic = True

def main(args):
    # --- 1. 初始化和配置加载 ---
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

    # --- 2. 创建模型 ---
    # use_policy=False 因为我们只是做有监督训练，不需要策略网络
    net, _, _ = create_models(dataset=args.dataset,
                              model_cfg_path=args.model_cfg_path,
                              model_ckpt_path=args.model_ckpt_path,
                              num_classes=args.num_classes,
                              use_policy=False)

    net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # --- 3. 加载全量数据 ---
    # 注意：这里我们不再需要candidate_set，并且train_set会包含所有训练视频
    # initial_labeled_ratio 设置为1.0来加载所有训练数据
    _, train_set, val_loader, _ = get_data(
        data_path=args.data_path,
        tr_bs=args.train_batch_size,
        vl_bs=args.val_batch_size,
        dataset_name=args.dataset,  # <-- 关键：传入 dataset_name
        n_workers=args.workers,
        clip_len=args.clip_len,
        initial_labeled_ratio=1.0  # 加载100%的训练数据
    )
    # 创建一个包含所有样本的DataLoader
    full_train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True,
                                   num_workers=args.workers, drop_last=False)

    # --- 4. 设置优化器和日志 ---
    optimizer = create_and_load_optimizers(net=net, opt_choice=args.optimizer, lr=args.lr,
                                           wd=args.weight_decay, momentum=args.momentum, ckpt_path=args.ckpt_path,
                                           exp_name_toload=None, exp_name=args.exp_name, snapshot=None,
                                           checkpointer=False, load_opt=False)[0]


    #scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    # 日志文件名可以自定义，以便区分
    log_name = 'full_data_training_log.txt'
    if 'scratch' in args.exp_name:
        log_name = 'full_data_scratch_log.txt'

    logger, best_record, curr_epoch = get_logfile(args.ckpt_path, args.exp_name,
                                                  checkpointer=False, snapshot=None,
                                                  log_name=log_name)

    # --- 5. 执行训练 ---
    if args.train:
        print(f"--- 开始在 {args.dataset.upper()} 全量数据集上进行有监督训练 ---")
        print(f"模型加载路径: {args.model_ckpt_path or 'None (Train from scratch)'}")
        print(f"日志将保存在: {os.path.join(args.ckpt_path, args.exp_name, log_name)}")

        # 调用复用的训练函数
        _, final_val_acc = train_har_classifier(
            args=args,
            curr_epoch=0,
            train_loader=full_train_loader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            val_loader=val_loader,
            best_record=best_record,
            logger=logger, # 传入logger
            scheduler=scheduler,
            schedulerP=None, # 没有策略网络
            final_train=True # 这是一个完整的训练过程
        )
        print(f"\n--- 训练结束 ---")
        print(f"在 {args.dataset.upper()} 全量数据上的最佳验证准确率为: {final_val_acc:.4f}")
        logger.close()

if __name__ == '__main__':
    args = parser.get_arguments()
    main(args)