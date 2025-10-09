import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Lambda
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as F_vision

# --- 动态导入所有数据集类 ---
from data.hmdb import HmdbDataset
from data.ucf import UcfDataset
from data.sth import Sthv2Dataset
from data.kinetics import KineticsDataset
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
    # clip shape: (T, C, H, W)
    t, c, h, w = clip.shape
    if w < h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return torch.stack([F_vision.resize(frame, [new_h, new_w], interpolation=interpolation) for frame in clip])


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
        return torch.stack([F_vision.hflip(frame) for frame in clip])
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
        model_type='c3d', # <--- 新增model_type参数
        n_workers=4,
        clip_len=16,
        augment_level=None,
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
        'sthv2': (Sthv2Dataset, 'sthv2_train_list_videos.txt', 'sthv2_val_list_videos.txt'),
        'kinetics': (KineticsDataset, 'train_list.txt', 'val_list.txt')
    }

    dataset_name = dataset_name.lower()
    if 'ucf' in dataset_name:
        dataset_name = 'ucf'
    elif 'hmdb' in dataset_name:
        dataset_name = 'hmdb'
    elif 'sthv2' in dataset_name:
        dataset_name = 'sthv2'
    elif 'kinetics' in dataset_name:
        dataset_name = 'kinetics'
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. Supported datasets are 'ucf' or 'hmdb'.")

    if augment_level is not None:
        print(f"为第二视角启用数据增强，等级: {augment_level}")
        def apply_pil_transform_to_clip(clip, pil_transform):
            # clip: (T, C, H, W) tensor
            transformed_frames = []
            for frame_tensor in clip: # frame_tensor is (C, H, W)
                pil_image = F_vision.to_pil_image(frame_tensor.byte())
                transformed_pil = pil_transform(pil_image)
                transformed_tensor = F_vision.to_tensor(transformed_pil) * 255.0
                transformed_frames.append(transformed_tensor)
            return torch.stack(transformed_frames)

        def apply_frame_level_transform(clip, transform):
            # torchvision transforms expect (C, H, W)
            clip_chw = clip.permute(1, 0, 2, 3).squeeze(0)  # TCHW -> CTHW
            return torch.stack([transform(frame) for frame in clip_chw.permute(1, 0, 2, 3)]).permute(1, 0, 2, 3)

        if augment_level == 1: # 轻度
            aug_list = [Lambda(lambda clip: apply_pil_transform_to_clip(clip, T.ColorJitter(brightness=0.2, contrast=0.2)))]
        elif augment_level == 2: # 中度
            aug_list = [Lambda(lambda clip: apply_pil_transform_to_clip(clip, T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)))]
        elif augment_level == 3: # 强度
            aug_list = [
                Lambda(lambda clip: apply_pil_transform_to_clip(clip, T.Compose([
                    T.RandomResizedCrop(size=112, scale=(0.6, 1.0)),
                    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                ]))),
            ]
        else:
            raise ValueError(f"错误: 未知的 augment_level '{augment_level}'。")
        # if augment_level == 1:
        #     aug_list = [
        #         Lambda(lambda clip: apply_frame_level_transform(clip, T.ColorJitter(brightness=0.2, contrast=0.2)))]
        # elif augment_level == 2:
        #     aug_list = [
        #         Lambda(lambda clip: apply_frame_level_transform(clip, T.ColorJitter(brightness=0.4, contrast=0.4))),
        #         Lambda(
        #             lambda clip: apply_frame_level_transform(clip, T.RandomAffine(degrees=15, translate=(0.1, 0.1)))),
        #     ]
        # elif augment_level == 3:
        #     aug_list = [
        #         Lambda(lambda clip: apply_frame_level_transform(clip, T.Compose([
        #             T.ToPILImage(),
        #             T.RandomResizedCrop(size=112, scale=(0.6, 1.0)),
        #             T.ColorJitter(brightness=0.5, contrast=0.5),
        #             T.ToTensor()
        #         ]))),
        #     ]
        # else:
        #     raise ValueError(f"错误: 未知的 augment_level '{augment_level}'。")

        final_aug_list = [
                             Lambda(lambda clip: resize_short_side(clip, size=128)),
                             Lambda(lambda clip: crop_clip(clip, 112, 'random')),
                         ] + aug_list + [
                             Lambda(lambda clip: flip_clip(clip, flip_ratio=0.5)),
                             Lambda(flip_channels_rgb_to_bgr),
                             Lambda(lambda x: x.permute(1, 0, 2, 3)),
                             Lambda(lambda clip: (clip - mean) / std)
                         ]
        augment_transform = Compose(final_aug_list)
    else:
        augment_transform = None
    DatasetClass, train_ann_file, val_ann_file = dataset_map[dataset_name.lower()]

    train_list = os.path.join(data_path, split_dir, train_ann_file)
    val_list = os.path.join(data_path, split_dir, val_ann_file)
    video_dir = os.path.join(data_path, video_dirname)

    if model_type == 'timesformer':
        # Timesformer 使用更标准的ImageNet均值/标准差
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1) * 255.0
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1) * 255.0

        train_transform = Compose([
            Lambda(lambda clip: resize_short_side(clip, size=224)),
            Lambda(lambda clip: crop_clip(clip, 224, 'random')),
            Lambda(lambda clip: flip_clip(clip, flip_ratio=0.5)),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # TCHW -> CTHW
            Lambda(lambda clip: (clip - mean) / std)
        ])

        val_transform = Compose([
            Lambda(lambda clip: resize_short_side(clip, size=224)),
            Lambda(lambda clip: crop_clip(clip, 224, 'center')),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),
            Lambda(lambda clip: (clip - mean) / std),
        ])
    elif model_type == 'videomae':
        # VideoMAE 的数据处理流程
        print("为 VideoMAE 模型配置数据转换流程 (参照官方配置文件)...")

        # --- 核心修改：直接使用您提供的MMAction2官方配置中的均值和标准差 ---
        # mean=[123.675, 116.28, 103.53]
        # std=[58.395, 57.12, 57.375]
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1, 1)

        # 训练时的数据增强流程
        train_transform = Compose([
            Lambda(lambda clip: resize_short_side(clip, size=224)),
            Lambda(lambda clip: crop_clip(clip, 224, 'random')),
            Lambda(lambda clip: flip_clip(clip, flip_ratio=0.5)),
            # 注意：VideoMAE 和 C3D 不同，它不需要 BGR 转换，保持 RGB 即可
            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # TCHW -> CTHW
            Lambda(lambda clip: (clip - mean) / std)
        ])

        # 验证/测试时的数据处理流程
        val_transform = Compose([
            Lambda(lambda clip: resize_short_side(clip, size=224)),
            Lambda(lambda clip: crop_clip(clip, 224, 'center')),
            # 同样，不需要 BGR 转换
            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # TCHW -> CTHW
            Lambda(lambda clip: (clip - mean) / std),
        ])
    elif model_type == 'c3d':
    # mean = torch.tensor([104.0, 117.0, 128.0]).view(3, 1, 1, 1)
        mean = torch.tensor([124.0, 117.0, 104.0]).view(3, 1, 1, 1)  # 注意顺序是 R, G, B
        std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1, 1)

        train_transform = Compose([
            Lambda(lambda clip: resize_short_side(clip, size=128)),
            Lambda(lambda clip: crop_clip(clip, 112, 'center')),
            # Lambda(lambda clip: crop_clip(clip, 112, 'random')),
            # Lambda(lambda clip: flip_clip(clip, flip_ratio=0.5)),
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
    elif model_type == 'csn':
        print("为 CSN 模型配置数据转换流程 (已修正为 [0, 255] 范围归一化)...")

        # --- 核心修正 1: 使用官方的 [0, 255] 范围的 mean 和 std ---
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1, 1)

        # 训练时的数据增强流程
        train_transform = Compose([
            # --- 核心修正 2: 所有空间变换直接在 [0, 255] 范围的张量上操作 ---
            Lambda(lambda clip: resize_short_side(clip, size=256)),
            Lambda(lambda clip: crop_clip(clip, 224, 'random')),
            Lambda(lambda clip: flip_clip(clip, flip_ratio=0.5)),

            # 确保张量是浮点类型以进行后续计算，但 *不* 改变其数值范围
            Lambda(lambda x: x.float()),

            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # TCHW -> CTHW

            # --- 核心修正 3: 在 [0, 255] 范围的张量上直接应用 mean 和 std ---
            Lambda(lambda clip: (clip - mean) / std)
        ])

        # 验证/测试时的数据处理流程
        val_transform = Compose([
            Lambda(lambda clip: resize_short_side(clip, size=256)),
            Lambda(lambda clip: crop_clip(clip, 224, 'center')),
            Lambda(lambda x: x.float()),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # TCHW -> CTHW
            Lambda(lambda clip: (clip - mean) / std),
        ])
    else:
        # 如果不是支持的模型，则明确报错
        raise ValueError(f"不支持的模型类型 '{model_type}'。请在 data_utils.py 中为其添加数据处理流程。")

    train_full_dataset = DatasetClass(train_list, video_dir,
                                      transform=train_transform,
                                      eval_transform=val_transform,
                                      augment_transform=augment_transform,
                                      clip_len=clip_len,
                                      is_train_set=True,
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