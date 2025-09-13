# 文件名: wcf00317/alrl/alrl-reward-model/utils/parser.py

import json
import os
import argparse
import yaml
from easydict import EasyDict as edict


def get_arguments():
    """
    一个健壮的参数解析器，正确处理优先级：命令行 > YAML > 代码内默认值。
    """
    # 步骤 1: 创建主解析器，并定义程序所有已知的参数及其最低优先级的默认值
    parser = argparse.ArgumentParser(description="Reinforced active learning for HAR")

    # --- 这里是所有参数的“注册表”，确保解析器认识它们 ---
    parser.add_argument('--config', type=str, default=None, help='Path to the YAML configuration file.')

    # 基本信息
    parser.add_argument('--dataset', type=str, default='hmdb51')
    parser.add_argument('--data_path', type=str, default='../hmdb51')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints')
    parser.add_argument('--exp_name', type=str, default='hmdb_exp')
    # parser.add_argument('--al_algorithm', type=str, default='random')
    parser.add_argument('--al_algorithm', type=str, default='random',
                        choices=['random', 'dqn', 'minimalist_tournament', 'stage1_tournament'],
                        help='The active learning algorithm to use.')

    # 数据处理
    parser.add_argument('--input_size', type=int, default=112)
    parser.add_argument('--scale_size', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_each_iter', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)

    # 路径加载控制
    parser.add_argument('--load_weights', action='store_true')
    parser.add_argument('--load_opt', action='store_true')
    parser.add_argument('--exp_name_toload', type=str, default=None)
    parser.add_argument('--exp_name_toload_rl', type=str, default=None)
    parser.add_argument('--snapshot', type=str, default='0')
    parser.add_argument('--checkpointer', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--final_test', action='store_true')
    parser.add_argument('--only_last_labeled', action='store_true')

    # 训练设置
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_dqn', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--gamma_scheduler_dqn', type=float, default=0.99)
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--al_train_epochs', type=int, default=15)

    # Active Learning 参数
    parser.add_argument('--budget_labels', type=int, default=578)
    parser.add_argument('--initial_labeled_ratio', type=float, default=0.05,
                        help='The initial ratio of labeled data to start with.')
    parser.add_argument('--rl_pool', type=int, default=10)
    parser.add_argument('--rl_episodes', type=int, default=10)
    parser.add_argument('--rl_buffer', type=int, default=100)
    parser.add_argument('--dqn_bs', type=int, default=5)
    parser.add_argument('--dqn_gamma', type=float, default=0.99)
    parser.add_argument('--egl_strategy', type=str, default='adaptive_k',
                        choices=['adaptive_k', 'approx', 'standard'],
                        help='要使用的EGL计算策略: adaptive_k (自适应), approx (快速近似), standard (理论标准版).')

    parser.add_argument('--nomination_ratio_c', type=float, default=3.0,
                        help='Ratio to determine nomination pool size (c * budget).')
    parser.add_argument('--run_sanity_check', action='store_true',
                        help='If set, run the optional sanity check duel against a random batch.')
    parser.add_argument('--distance_threshold_alpha', type=float, default=0.9,
                        help='Scaling factor for the median distance threshold in pruning step (e.g., 0.9).')
    # MMACTION2 配置
    parser.add_argument('--model_type', type=str, default='c3d', choices=['c3d', 'timesformer', 'videomae'],
                        help='Specify the model architecture to use (e.g., c3d, timesformer, videomae).')
    parser.add_argument('--mmaction_config', type=str, default=None)
    parser.add_argument('--model_cfg_path', type=str, default=None)
    parser.add_argument('--model_ckpt_path', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=51)
    parser.add_argument('--embed_dim', type=int, default=4096)
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--num_clips', type=int, default=1)

    # ALRM 和 KAN 相关配置
    parser.add_argument('--reward_model_type', type=str, default='kan')
    parser.add_argument('--kan_grid_size', type=int, default=5)
    parser.add_argument('--kan_spline_order', type=int, default=3)
    parser.add_argument('--kan_hidden_layers', nargs='+', type=int, default=[8, 4])
    parser.add_argument('--exp_dir', type=str, default=None)

    #在这里定义所有新的特征提取器参数
    parser.add_argument('--use_statistical_features', action='store_true')
    parser.add_argument('--use_diversity_feature', action='store_true')
    parser.add_argument('--use_representativeness_feature', action='store_true')
    parser.add_argument('--use_prediction_margin_feature', action='store_true')
    parser.add_argument('--use_labeled_distance_feature', action='store_true')
    parser.add_argument('--use_neighborhood_density_feature', action='store_true')

    parser.add_argument('--use_temporal_consistency_feature', action='store_true',
                        help='(策略) 启用基于快慢速视频特征差异的策略。')
    parser.add_argument('--use_cross_view_consistency_feature', action='store_true',
                        help='(策略) 启用基于不同空间增强视角的特征一致性策略。')
    parser.add_argument('--augment_level', type=int, default=1, help='为Cross-view Consistency设置第二视角的增强等级 (整数，例如 1, 2, 3)。')
    # 步骤 2: 第一次解析，只获取 config 文件路径
    args, _ = parser.parse_known_args()

    # 步骤 3: 如果 config 文件存在，加载它并将其中的值设置为新的默认值
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # 将YAML中的值设为默认，这样命令行参数就可以覆盖它们
        parser.set_defaults(**yaml_config)

    # 步骤 4: 第二次（最终）解析。
    # 这次会应用所有优先级：命令行 > YAML > 代码默认值
    final_args = parser.parse_args()

    return edict(vars(final_args))


def save_arguments(args):
    """保存最终的配置参数到实验目录的args.json文件中。"""
    print_args = {}
    args_dict = dict(args)

    print("\n--- Final Configuration ---")
    for key in sorted(args_dict.keys()):
        value = args_dict[key]
        print(f'{key:<25}: {value}')
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            print_args[key] = value
    print("-------------------------\n")

    if args.get('ckpt_path') and args.get('exp_name'):
        exp_dir = os.path.join(args.ckpt_path, args.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, 'args.json')
        with open(path, 'w') as fp:
            json.dump(print_args, fp, indent=4)
        print(f'Args saved in {path}')