# wcf00317/alrl/alrl-reward_model/data/base_dataset.py

import os
import random
import numpy as np
import decord
import torch
from torch.utils.data import Dataset


class ActiveLearningVideoDataset(Dataset):
    """
    一个通用的主动学习视频数据集基类。
    它封装了所有数据集共享的核心逻辑，包括：
    - 视频片段的加载与采样
    - 已标注集和未标注池的管理
    - 主动学习相关的接口方法
    子类需要实现 _load_annotations 方法来加载特定数据集的标注。
    """

    def __init__(self, annotation_file, video_dir, transform=None, eval_transform=None,augment_transform=None,
                 clip_len=16, sample_type='center_clip',
                 is_train_set=True, initial_labeled_ratio=0.01):
        self.video_dir = video_dir
        # self.video_list_info 存储 (video_name, label, original_index)
        self.video_list_info = self._load_annotations(annotation_file)
        self.transform = transform
        self.eval_transform = eval_transform if eval_transform is not None else transform
        self.augment_transform = augment_transform if augment_transform is not None else transform
        self.clip_len = clip_len
        self.num_clips = 1
        self.sample_type = sample_type

        self.all_video_ids = list(range(len(self.video_list_info)))

        # 主动学习相关状态
        self.is_train_set = is_train_set
        if self.is_train_set:
            self.labeled_video_ids = set()
            self.unlabeled_video_ids = set(self.all_video_ids)
            self._initialize_labeled_set(initial_labeled_ratio)
        else:
            self.labeled_video_ids = set(self.all_video_ids)
            self.unlabeled_video_ids = set()

        all_labels = [int(info[1]) for info in self.video_list_info]
        self.num_classes = max(all_labels) + 1 if all_labels else 0

    def _load_annotations(self, file):
        """
        子类必须实现此方法以加载特定格式的标注文件。
        """
        raise NotImplementedError("子类必须实现 _load_annotations 方法")

    def _initialize_labeled_set(self, ratio):
        if ratio > 0 and self.is_train_set:
            num_initial_labeled = max(1, int(len(self.all_video_ids) * ratio))
            initial_labeled_indices = random.sample(self.all_video_ids, num_initial_labeled)
            for vid_id in initial_labeled_indices:
                self.labeled_video_ids.add(vid_id)
                self.unlabeled_video_ids.remove(vid_id)

    def __getitem__(self, original_vid_idx):
        video_name, label_str, _ = self.video_list_info[original_vid_idx]
        full_path = os.path.join(self.video_dir, video_name)
        label = int(label_str)

        video_reader = decord.VideoReader(full_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)

        if self.is_train_set:
            # --- 训练模式：采样1个随机片段 ---
            start_frame = random.randint(0, max(0, total_frames - self.clip_len))
            indices = np.arange(start_frame, start_frame + self.clip_len)
            if total_frames < self.clip_len:
                indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)

            clip = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()

            if self.transform:
                clip = self.transform(clip)

            # --- 核心恢复：返回 5D 张量 [1, C, T, H, W] ---
            return clip.unsqueeze(0), label, original_vid_idx
        else:
            # --- 测试/验证模式：均匀采样 num_clips 个片段 ---
            clips_list = []
            tick = total_frames / self.num_clips
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_clips)])

            for start_frame in offsets:
                start_frame = max(min(start_frame, total_frames - self.clip_len), 0)
                indices = np.arange(start_frame, start_frame + self.clip_len)
                if total_frames < self.clip_len:
                    indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)

                raw_clip = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()

                transformed_clip = self.eval_transform(raw_clip) if self.eval_transform else raw_clip
                clips_list.append(transformed_clip)

            # --- 核心恢复：返回 5D 张量 [num_clips, C, T, H, W] ---
            final_clips = torch.stack(clips_list, dim=0)
            return final_clips, label, original_vid_idx

    def __len__(self):
        return len(self.video_list_info)

    # --- 主动学习相关方法 ---
    def get_candidates_video_ids(self):
        if not self.is_train_set:
            raise RuntimeError("只有训练集可以提供未标注候选。")
        return list(self.unlabeled_video_ids)

    def get_num_labeled_videos(self):
        return len(self.labeled_video_ids)

    def add_video_to_labeled(self, video_id):
        if not self.is_train_set:
            raise RuntimeError("不能向验证集添加已标注视频。")
        if video_id in self.unlabeled_video_ids:
            self.unlabeled_video_ids.remove(video_id)
            self.labeled_video_ids.add(video_id)
        else:
            print(f"警告: 视频 {video_id} 已被标注或不在未标注池中。")

    def reset(self, initial_labeled_ratio=0.01):
        if self.is_train_set:
            self.labeled_video_ids = set()
            self.unlabeled_video_ids = set(self.all_video_ids)
            self._initialize_labeled_set(initial_labeled_ratio)

    def get_video(self, vid_idx):
        video_name, _, _ = self.video_list_info[vid_idx]
        full_path = os.path.join(self.video_dir, video_name)
        video_reader = decord.VideoReader(full_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)

        start_frame = (total_frames - self.clip_len) // 2
        indices = np.arange(start_frame, start_frame + self.clip_len)
        if total_frames < self.clip_len:
            indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)

        clip = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()

        if self.eval_transform:
            transformed_clip = self.eval_transform(clip)
        else:
            raise RuntimeError("此数据集未设置 eval_transform！")

        return transformed_clip.unsqueeze(0)

    def get_video_multi_speed(self, vid_idx, fast_frames, slow_frames):
        """
        为同一个视频，以两种不同的帧率采样并返回两个片段。
        所有解码操作都在CPU上进行，与您仓库中的其他函数保持一致。
        """
        video_name, _, _ = self.video_list_info[vid_idx]
        full_path = os.path.join(self.video_dir, video_name)
        # 始终在 CPU 上解码
        video_reader = decord.VideoReader(full_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)

        clips = []
        for num_frames in [fast_frames, slow_frames]:
            # 为了简化，我们统一采用中心采样
            start_frame = (total_frames - num_frames) // 2
            indices = np.arange(start_frame, start_frame + num_frames)
            if total_frames < num_frames:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            # 从CPU上的解码结果创建Tensor
            clip = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()

            # 应用评估阶段的transform
            if self.eval_transform:
                transformed_clip = self.eval_transform(clip)
            else:
                raise RuntimeError("此数据集未设置 eval_transform！")

            # 确保返回的张量有批次维度
            if transformed_clip.dim() == 4:
                transformed_clip = transformed_clip.unsqueeze(0)

            clips.append(transformed_clip)

        return clips[0], clips[1]  # 返回 (fast_clip, slow_clip)

    def get_video_augmented_views(self, vid_idx):
        """
        为同一个视频，返回两个不同空间增强的视角。
        视角1: 使用标准的评估/验证 transform (通常是中心裁剪)。
        视角2: 使用更强的训练/增强 transform。
        """
        video_name, _, _ = self.video_list_info[vid_idx]
        full_path = os.path.join(self.video_dir, video_name)
        video_reader = decord.VideoReader(full_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)

        # --- 统一的采样逻辑 ---
        start_frame = (total_frames - self.clip_len) // 2
        indices = np.arange(start_frame, start_frame + self.clip_len)
        if total_frames < self.clip_len:
            indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)

        # raw_clip in shape (T, H, W, C)
        raw_clip_numpy = video_reader.get_batch(indices).asnumpy()
        # To (T, C, H, W)
        raw_clip_tensor = torch.from_numpy(raw_clip_numpy).permute(0, 3, 1, 2).float()

        # --- 生成两个视角 ---
        # 视角1: 标准视角 (使用 eval_transform)
        view1 = self.eval_transform(raw_clip_tensor.clone())

        # 视角2: 强增强视角 (使用 augment_transform)
        view2 = self.augment_transform(raw_clip_tensor.clone())

        return view1, view2.unsqueeze(0)