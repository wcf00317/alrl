import json
import os
# utils/parser.py
import argparse
import yaml
from easydict import EasyDict as edict


def get_arguments():
    # Step 0: Define default values
    default_config = {
        'ckpt_path': './checkpoints',
        'data_path': '../hmdb51',
        'exp_name': 'hmdb_exp',
        'train_batch_size': 16,
        'val_batch_size': 1,
        'epoch_num': 10,
        'lr': 0.0001,
        'gamma': 0.998,
        'gamma_scheduler_dqn': 0.97,
        'weight_decay': 1e-4,
        'input_size': [224, 224],
        'scale_size': 0,
        'momentum': 0.95,
        'patience': 60,
        'snapshot': 'last_jaccard_val.pth',
        'checkpointer': False,
        'load_weights': False,
        'load_opt': False,
        'optimizer': 'Adam',
        'train': True,
        'test': False,
        'final_test': False,
        'only_last_labeled': False,
        'rl_pool': 50,
        'al_algorithm': 'random',
        'dataset': 'hmdb51',
        'budget_labels': 100,
        'num_each_iter': 1,
        'region_size': [128, 128],
        'lr_dqn': 0.0001,
        'rl_buffer': 3200,
        'rl_episodes': 50,
        'dqn_bs': 16,
        'dqn_gamma': 0.999,
        'dqn_epochs': 1,
        'bald_iter': 20,
        'seed': 26,
        'full_res': False,
        'mmaction_config': None,
    }

    # Step 1: Pre-parse the --config argument
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    # Step 2: Load YAML config if provided
    yaml_config = {}
    if config_args.config is not None:
        with open(config_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)

    # Step 3: Define the full argument parser
    parser = argparse.ArgumentParser(description="Reinforced active learning for HAR")

    # ✨ THE FIX: Add the --config argument to the main parser too ✨
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')

    for key in default_config.keys():
        arg_type = type(default_config[key])
        if isinstance(default_config[key], bool):
            # This logic correctly handles boolean flags
            parser.add_argument(f'--{key}', action='store_true', default=None)
        elif isinstance(default_config[key], list):
            parser.add_argument(f'--{key}', nargs='+', type=type(default_config[key][0]))
        else:
            parser.add_argument(f'--{key}', type=arg_type)

    # Step 4: Parse command-line arguments
    cli_args = vars(parser.parse_args())

    # Step 5: Merge configurations with correct priority
    final_args = {}
    # Also include 'config' in the keys to iterate over
    all_keys = list(default_config.keys()) + ['config']
    for key in all_keys:
        # Priority 1: Command-line arguments (if they are not None)
        if key in cli_args and cli_args[key] is not None:
            final_args[key] = cli_args[key]
        # Priority 2: YAML file settings
        elif key in yaml_config:
            final_args[key] = yaml_config[key]
        # Priority 3: Default values
        elif key in default_config:
            final_args[key] = default_config[key]

    # Special handling for boolean flags where presence means True
    for key, value in default_config.items():
        if isinstance(value, bool):
            if cli_args.get(key) is None:  # Not specified on command line
                if key in yaml_config:
                    final_args[key] = yaml_config[key]
                else:
                    final_args[key] = default_config[key]
            else:  # Specified on command line
                final_args[key] = cli_args[key] if default_config[key] is False else not cli_args[key]

    return edict(final_args)

def save_arguments(args):
    print_args = {}
    param_names = [elem for elem in dir(args) if not elem.startswith('_')]

    for name in param_names:
        value = getattr(args, name)
        # 检查是否是可以被 JSON 序列化的基本类型
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            print_args[name] = value
            print(f'[{name}]   {value}')
        else:
            print(f'[Skipped] {name} (type: {type(value)})')

    if getattr(args, 'train', False):
        path = os.path.join(args.ckpt_path, args.exp_name, 'args.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(print_args, fp, indent=2)
        print('Args saved in ' + path)
