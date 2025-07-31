import os, torch
from torch.utils.data import DataLoader, Subset  # 导入 Subset
import torchvision.transforms as standard_transforms
from data.hmdb import HmdbDataset  # 确保路径正确
from torchvision.transforms import Compose, Normalize, Lambda
import torch.nn.functional as F
import torchvision.transforms as T


def resize_and_crop(clip, scale_size=128, crop_size=112):
    """对 [T, C, H, W] 的 clip 逐帧 Resize 和 CenterCrop"""
    resized = F.interpolate(clip, size=(scale_size, scale_size), mode='bilinear', align_corners=False)
    h, w = resized.shape[-2:]
    th, tw = crop_size, crop_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return resized[:, :, i:i + th, j:j + tw]  # [T, C, H, W]


def apply_normalize_per_frame(normalize):
    def _apply(clip):
        if clip.shape[0] == 3:
            # 如果是 [C, T, H, W]，转为 [T, C, H, W]
            clip = clip.permute(1, 0, 2, 3)
        return torch.stack([normalize(frame) for frame in clip])
    return Lambda(_apply)

def resize_short_side(clip, size, interpolation=T.InterpolationMode.BILINEAR):
    """保持长宽比，将短边缩放到指定大小"""
    t, c, h, w = clip.shape
    if h > w:
        new_h, new_w = int(size * h / w), size
    else:
        new_h, new_w = size, int(size * w / h)
    return torch.stack([T.functional.resize(frame, [new_h, new_w], interpolation=interpolation) for frame in clip])

def crop_clip(clip, crop_size, crop_type='center'):
    """对视频片段进行裁剪"""
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
    return clip[:, :, i:i+th, j:j+tw]

def flip_clip(clip, flip_ratio=0.5):
    """以一定概率对视频片段进行水平翻转"""
    if torch.rand(1) < flip_ratio:
        return torch.stack([T.functional.hflip(frame) for frame in clip])
    return clip

def get_data(
        data_path,
        tr_bs,
        vl_bs,
        n_workers=4,
        clip_len=16,
        split_dir='.',
        video_dirname='videos',
        transform_type='imagenet',
        initial_labeled_ratio=0.05,
        test=False
):
    if 'hmdb' in data_path.lower():
        dataset_name = "HMDB51"
    elif 'ucf' in data_path.lower():
        dataset_name = "UCF101"
    else:
        raise ValueError(f"Unknown dataset in path: {data_path}")

    print(f'Loading video classification data for dataset: {dataset_name}')

    train_list = os.path.join(data_path, split_dir, 'train_videos.txt')
    val_list = os.path.join(data_path, split_dir, 'val_videos.txt')
    video_dir = os.path.join(data_path, video_dirname)

    scale_size, input_size = 128, 112

    # Normalization 配置
    if transform_type == 'basic':
        normalize = Normalize([0.5] * 3, [0.5] * 3)
    elif transform_type == 'imagenet':
        normalize = Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
    elif transform_type == 'c3d':
        mean = torch.tensor([104.0, 117.0, 128.0]).view(3, 1, 1, 1)
        std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1, 1)
    else:
        raise ValueError("Please use transform_type='c3d' for this model.")

    # ✅ 视频变换流程
    # input_transform = Compose([
    #     Lambda(lambda clip: resize_and_crop(clip)),                       # ✅ 新增：使用 torch.nn.functional.resize + crop
    #     apply_normalize_per_frame(normalize_per_frame),                  # ✅ 逐帧归一化
    #     Lambda(lambda x: x.permute(1, 0, 2, 3))                           # ✅ [T,C,H,W] → [C,T,H,W]
    # ])
    train_transform = Compose([
        Lambda(lambda clip: resize_short_side(clip, size=128)),
        Lambda(lambda clip: crop_clip(clip, 112, 'random')),
        Lambda(lambda clip: flip_clip(clip, flip_ratio=0.5)),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),              # 先换位 -> [C, T, H, W]
        Lambda(lambda clip: (clip - mean) / std)    ])

    # (2) 验证/测试变换流程 (确定性处理)
    val_transform = Compose([
        Lambda(lambda clip: resize_short_side(clip, size=128)),
        Lambda(lambda clip: crop_clip(clip, 112, 'center')),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),              # 先换位 -> [C, T, H, W]
        Lambda(lambda clip: (clip - mean) / std),
    ])

    train_full_dataset = HmdbDataset(train_list, video_dir,
                                     transform=train_transform,
                                     clip_len=clip_len,
                                     is_train_set=True,
                                     eval_transform=val_transform,
                                     initial_labeled_ratio=initial_labeled_ratio)

    val_set = HmdbDataset(val_list, video_dir,
                          transform=val_transform,
                          clip_len=clip_len,
                          is_train_set=False)

    current_labeled_indices = list(train_full_dataset.labeled_video_ids)
    train_subset = Subset(train_full_dataset, current_labeled_indices)

    train_loader = DataLoader(train_subset,
                              batch_size=tr_bs,
                              shuffle=True,
                              num_workers=n_workers,
                              drop_last=False)

    val_loader = DataLoader(val_set,
                            batch_size=vl_bs,
                            shuffle=False,
                            num_workers=n_workers)

    candidate_set_for_al = train_full_dataset

    return train_loader, train_full_dataset, val_loader, candidate_set_for_al