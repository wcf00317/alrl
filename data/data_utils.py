import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Lambda
import torch.nn.functional as F
import torchvision.transforms as T

# --- 动态导入所有数据集类 ---
from data.hmdb import HmdbDataset
from data.ucf import UcfDataset
import numpy as np
import random

def seed_worker(worker_id):
    """
    为 DataLoader 的 worker 设置随机种子，确保可复现性。
    """
    # torch.initial_seed() 返回一个由主进程为当前 worker 生成的唯一基础种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- 辅助函数 ---
def resize_short_side(clip, size, interpolation=T.InterpolationMode.BILINEAR):
    t, c, h, w = clip.shape
    new_h, new_w = (int(size * h / w), size) if h > w else (size, int(size * w / h))
    return torch.stack([T.functional.resize(frame, [new_h, new_w], interpolation=interpolation) for frame in clip])


def crop_clip(clip, crop_size, crop_type='center'):
    t, c, h, w = clip.shape
    th, tw = (crop_size, crop_size)
    if crop_type == 'center':
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
    elif crop_type == 'random':
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
    else:
        raise ValueError(f"Unknown crop_type: {crop_type}")
    return clip[:, :, i:i + th, j:j + tw]


def flip_clip(clip, flip_ratio=0.5):
    if torch.rand(1) < flip_ratio:
        return torch.stack([T.functional.hflip(frame) for frame in clip])
    return clip


def flip_channels_rgb_to_bgr(clip):
    """将 [T, C, H, W] 的 RGB 视频片段转换为 BGR"""
    return clip[:, [2, 1, 0], :, :]


# --- 主数据加载函数 ---
def get_data(
        data_path,
        tr_bs,
        vl_bs,
        dataset_name,  # <-- 关键：直接从 config 传入
        n_workers=4,
        clip_len=16,
        split_dir='.',
        video_dirname='videos',
        initial_labeled_ratio=0.05,
        seed=42
        # test=False  # test 参数不再需要，由 is_train_set 控制
):
    print(f'Loading data for dataset: {dataset_name}')

    dataset_map = {
        'hmdb': (HmdbDataset, 'train_videos.txt', 'val_videos.txt'),
        'ucf': (UcfDataset, 'train_videos.txt', 'val_videos.txt'),
        # 'ucf': (UcfDataset, 'train_videos.txt', 'val_videos.txt')
    }

    dataset_name = dataset_name.lower()
    if 'ucf' in dataset_name:
        dataset_name = 'ucf'
    elif 'hmdb' in dataset_name:
        dataset_name = 'hmdb'
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. Supported datasets are 'ucf' or 'hmdb'.")

    DatasetClass, train_ann_file, val_ann_file = dataset_map[dataset_name.lower()]

    train_list = os.path.join(data_path, split_dir, train_ann_file)
    val_list = os.path.join(data_path, split_dir, val_ann_file)
    video_dir = os.path.join(data_path, video_dirname)

    mean = torch.tensor([104.0, 117.0, 128.0]).view(3, 1, 1, 1)
    std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1, 1)

    train_transform = Compose([
        Lambda(lambda clip: resize_short_side(clip, size=128)),
        Lambda(lambda clip: crop_clip(clip, 112, 'random')),
        Lambda(lambda clip: flip_clip(clip, flip_ratio=0.5)),
        Lambda(flip_channels_rgb_to_bgr),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),
        Lambda(lambda clip: (clip - mean) / std)
    ])

    val_transform = Compose([
        Lambda(lambda clip: resize_short_side(clip, size=128)),
        Lambda(lambda clip: crop_clip(clip, 112, 'center')),
        Lambda(flip_channels_rgb_to_bgr),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),
        Lambda(lambda clip: (clip - mean) / std),
    ])

    if dataset_name == 'ucf':
        print("Using MMAction2 official data pipeline for UCF101.")
        # MMAction2's train_pipeline for C3D on UCF101
        train_transform = Compose([
            Lambda(lambda clip: resize_short_side(clip, size=128)),
            Lambda(lambda clip: crop_clip(clip, 112, 'random')),
            Lambda(lambda clip: flip_clip(clip, flip_ratio=0.5)),
            # MMAction2 C3D config doesn't flip channels, assuming model handles it or was trained on RGB
            # If accuracy is low, uncommenting the BGR flip is the first thing to test.
            # Lambda(flip_channels_rgb_to_bgr),
            Lambda(lambda x: x.permute(1, 0, 2, 3)), # To NCTHW format
            Lambda(lambda clip: (clip - mean) / std) # C3D-style normalization
        ])

        # MMAction2's val_pipeline for C3D on UCF101
        val_transform = Compose([
            Lambda(lambda clip: resize_short_side(clip, size=128)),
            Lambda(lambda clip: crop_clip(clip, 112, 'center')),
            # Lambda(flip_channels_rgb_to_bgr),
            Lambda(lambda x: x.permute(1, 0, 2, 3)), # To NCTHW format
            Lambda(lambda clip: (clip - mean) / std)
        ])

    train_full_dataset = DatasetClass(train_list, video_dir,
                                      transform=train_transform,
                                      clip_len=clip_len,
                                      is_train_set=True,
                                      eval_transform=val_transform,
                                      initial_labeled_ratio=initial_labeled_ratio)

    val_set = DatasetClass(val_list, video_dir,
                           transform=val_transform,
                           clip_len=clip_len,
                           is_train_set=False)

    current_labeled_indices = list(train_full_dataset.labeled_video_ids)
    train_subset = Subset(train_full_dataset, current_labeled_indices)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_subset,
                              batch_size=tr_bs,
                              shuffle=True,
                              num_workers=n_workers,
                              drop_last=False,
                              worker_init_fn=seed_worker, # <-- 应用 worker 初始化函数
                              generator=g)

    val_loader = DataLoader(val_set,
                            batch_size=vl_bs,
                            shuffle=False,
                            num_workers=n_workers)

    return train_loader, train_full_dataset, val_loader, train_full_dataset