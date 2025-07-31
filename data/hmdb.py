# ralis/data/hmdb_dataset.py

import os
import numpy as np
import decord
import torch
from torch.utils.data import Dataset
import random # 用于随机采样帧


class HmdbDataset(Dataset):
    def __init__(self, annotation_file, video_dir, transform=None, eval_transform=None,
                 clip_len=16, sample_type='center_clip', # 添加 clip_len 和 sample_type
                 is_train_set=True, initial_labeled_ratio=0.01): # 用于主动学习
        self.video_dir = video_dir
        # self.video_list 存储 (video_name, label, original_index)
        # original_index 是在_load_annotations中加载时的原始序号
        self.video_list_info = self._load_annotations(annotation_file)
        self.transform = transform
        self.clip_len = clip_len
        self.eval_transform = eval_transform if eval_transform is not None else transform # ❤️

        self.num_clips = 1
        self.sample_type = sample_type # 'center_clip', 'random_clip', 'full_video_for_state'

        # 获取所有视频的原始ID（0到len(self.video_list_info)-1）
        self.all_video_ids = list(range(len(self.video_list_info)))
        
        # 主动学习相关状态
        self.is_train_set = is_train_set # 标记是否是训练集（只有训练集需要AL功能）
        if self.is_train_set:
            self.labeled_video_ids = set() # 存储已标记视频的ID集合
            self.unlabeled_video_ids = set(self.all_video_ids) # 存储未标记视频的ID集合
            self.current_unlabeled_candidates = [] # 当前供策略网络选择的候选池
            
            # 初始化：随机选择一部分视频作为初始已标记集
            self._initialize_labeled_set(initial_labeled_ratio)
            print(f"Initial labeled videos: {len(self.labeled_video_ids)}")
            print(f"Initial unlabeled videos: {len(self.unlabeled_video_ids)}")
        else:
            # 验证集不参与主动学习
            self.labeled_video_ids = set(self.all_video_ids) # 验证集所有视频都被认为是“已标记”
            self.unlabeled_video_ids = set()
            self.current_unlabeled_candidates = []

        # 获取类别数量（假设标签从0开始且连续）
        all_labels = [int(info[1]) for info in self.video_list_info]
        self.num_classes = max(all_labels) + 1 if all_labels else 0

    def _load_annotations(self, file):
        # 每行格式: video_path label
        video_info = []
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                video_name, label = line.strip().split()
                video_info.append((video_name, label, i)) # 保存原始索引
        return video_info

    def _initialize_labeled_set(self, ratio):
        if ratio > 0:
            num_initial_labeled = max(1, int(len(self.all_video_ids) * ratio))
            initial_labeled_indices = random.sample(self.all_video_ids, num_initial_labeled)
            for vid_id in initial_labeled_indices:
                self.labeled_video_ids.add(vid_id)
                self.unlabeled_video_ids.remove(vid_id)
        # 如果 ratio 为 0，则所有视频初始都为未标记

    def get_video_path(self, vid_idx):
        # 根据原始索引获取视频路径
        video_name, _, _ = self.video_list_info[vid_idx]
        return os.path.join(self.video_dir, video_name)

    def _load_video_clip(self, full_path):
        # 使用 decord 高效读取视频
        video_reader = decord.VideoReader(full_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)

        # 根据是训练还是测试，决定采样方式
        if self.is_train_set:
            # 随机采样
            start_frame = random.randint(0, max(0, total_frames - self.clip_len))
            indices = range(start_frame, start_frame + self.clip_len)
        else:
            # 中心采样
            start_frame = (total_frames - self.clip_len) // 2
            indices = range(start_frame, start_frame + self.clip_len)

        # 保证视频长度足够
        if total_frames < self.clip_len:
            indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)

        # ✅ 【核心修正】只读取原始帧，不进行任何变换和归一化
        # decord 读取后是 [T, H, W, C]，像素范围 [0, 255]
        # 我们将其转换为 [T, C, H, W] 的 float 张量
        clip = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()
        return clip
    # def _load_video_clip(self, full_path, sample_type='center_clip'):
    #     # 内部辅助函数，加载并采样视频片段
    #     video_reader = decord.VideoReader(full_path)
    #     total_frames = len(video_reader)
    #
    #     if total_frames < self.clip_len:
    #         # 如果视频太短，重复最后一帧或填充
    #         # 简化处理：暂时只取所有帧，后续 transform 可能处理大小不一致问题
    #         indices = list(range(total_frames))
    #         print(f"Warning: Video {full_path} is too short ({total_frames} frames) for clip_len {self.clip_len}. Using all frames.")
    #     else:
    #         if sample_type == 'center_clip':
    #             center_frame = total_frames // 2
    #             start_frame = max(0, center_frame - self.clip_len // 2)
    #             end_frame = min(total_frames, start_frame + self.clip_len)
    #             indices = list(range(start_frame, end_frame))
    #             # 确保 clip_len
    #             if len(indices) < self.clip_len:
    #                 indices += [indices[-1]] * (self.clip_len - len(indices)) # 简单填充
    #         elif sample_type == 'random_clip':
    #             start_frame = random.randint(0, max(0, total_frames - self.clip_len))
    #             indices = list(range(start_frame, start_frame + self.clip_len))
    #         elif sample_type == 'full_video_for_state':
    #             # 为 compute_state 准备，可能需要特殊处理，比如固定采样帧数或使用全部帧
    #             # 这里假设直接使用全部帧，或者在get_video方法中进行更复杂的采样
    #             indices = list(range(total_frames))
    #             # 如果 model 只接受固定长度输入，可能需要在这里进行补帧或截断
    #
    #         else:
    #             raise ValueError(f"Unknown sample_type: {sample_type}")
    #
    #     # 使用 decord 批量读取帧，并转换为 PyTorch Tensor
    #     # decord.VideoReader.get_batch() 返回的是 NumPy 数组 (T, H, W, C)
    #     # TODO: 修改
    #     clip = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()
    #
    #     # 如果长度不足 clip_len，需要补齐
    #     if clip.shape[0] < self.clip_len and sample_type != 'full_video_for_state':
    #          pad_needed = self.clip_len - clip.shape[0]
    #          # 简单地重复最后一帧
    #          padding = clip[-1:].repeat(pad_needed, 1, 1, 1)
    #          clip = torch.cat([clip, padding], dim=0)
    #
    #     if self.transform:
    #         clip = self.transform(clip)
    #     return clip
    def __getitem__(self, original_vid_idx):
        video_name, label_str, _ = self.video_list_info[original_vid_idx]
        full_path = os.path.join(self.video_dir, video_name)
        label = int(label_str)

        video_reader = decord.VideoReader(full_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)

        clips_list = []

        if self.is_train_set:
            # --- 训练模式：采样1个随机片段 ---
            start_frame = random.randint(0, max(0, total_frames - self.clip_len))
            indices = range(start_frame, start_frame + self.clip_len)

            # 保证视频长度足够
            if total_frames < self.clip_len:
                indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)

            clip = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()

            if self.transform:
                clip = self.transform(clip)

            # 对于训练，我们为了和测试保持输出维度一致，也增加一个维度
            # 注意：DataLoader的默认collate_fn会处理好batching
            return clip.unsqueeze(0), label, original_vid_idx

        else:
            # --- 测试模式：均匀采样 num_clips 个片段 ---
            # 计算每个片段的起始帧位置
            tick = total_frames / self.num_clips
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_clips)])

            for start_frame in offsets:
                start_frame = max(min(start_frame, total_frames - self.clip_len), 0)
                indices = range(start_frame, start_frame + self.clip_len)

                # 保证视频长度足够
                if total_frames < self.clip_len:
                    indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)

                raw_clip = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()

                if self.transform:
                    transformed_clip = self.transform(raw_clip)
                else:
                    transformed_clip = raw_clip

                clips_list.append(transformed_clip)

            # 将所有片段堆叠起来
            final_clips = torch.stack(clips_list, dim=0)  # [num_clips, C, T, H, W]
            return final_clips, label, original_vid_idx

    # def __getitem__(self, original_vid_idx):
    #     # 外部DataLoader访问时使用的数据索引是原始索引
    #     # 根据原始视频ID获取视频信息
    #     video_name, label_str, _ = self.video_list_info[original_vid_idx]
    #     full_path = os.path.join(self.video_dir, video_name)
    #     label = int(label_str)
    #
    #     # 核心：只有当视频被标记时，才通过 DataLoader 提供片段给 HAR 模型训练
    #     if self.is_train_set and original_vid_idx not in self.labeled_video_ids:
    #         # 如果是训练集但视频未标记，则跳过（DataLoader不会访问这些索引）
    #         # 这段代码实际上不会被 DataLoader 正常执行到，因为 DataLoader 应该只访问 labeled_dataset 的子集
    #         # 但作为安全检查，可以抛出错误或返回空数据（不推荐，应该在 DataLoader 层面过滤）
    #         raise IndexError(f"Attempting to access unlabeled video {original_vid_idx} via __getitem__ in train_set.")
    #     num_clips = 1 # TODO：以后不能设置死，现在为了跑通先设置的
    #
    #     clips_list = []
    #     for _ in range(num_clips):
    #         # 每次循环都加载一个随机片段
    #         # _load_video_clip 返回的片段形状通常是 (T, C, H, W)
    #         clip = self._load_video_clip(full_path, sample_type='random_clip' if self.is_train_set else 'center_clip')
    #
    #         # 对每个加载的片段应用变换。
    #         # 您的 self.transform 应该已经包含了 Resize, Crop, Normalize(per-frame), permute
    #         # 并且期望输入是 (T, C, H, W)，输出是 (C, T, H, W)。
    #         transformed_clip = self.transform(clip)  # transformed_clip now has shape (C, T, H, W)
    #         clips_list.append(transformed_clip)
    #
    #     # 将所有 num_clips 个片段堆叠起来，形成 (num_clips, C, T, H, W) 的形状。
    #     # 这是 MMACTION2 模型通常期望的多片段输入格式。
    #     final_clip_batch = torch.stack(clips_list, dim=0)
    #     # 返回 (视频片段, 标签, 原始视频ID)
    #     return final_clip_batch, label, original_vid_idx

    def __len__(self):
        # 对于训练集，__len__ 应该反映当前已标记视频的数量，以便 DataLoader 只遍历它们
        # 但这会导致 DataLoader 的索引不连续，更合理的方式是创建一个只包含已标记视频索引的子集 Dataset
        # 为了兼容性，这里暂时返回所有视频数量，但实际 DataLoader 会配合一个 Sampler 或 Subset
        # 待会儿会在 get_data 中处理这个
        return len(self.video_list_info) # 返回总视频数，但实际迭代时会通过Subset过滤

    # --- 主动学习相关方法 ---
    def get_candidates_video_ids(self):
        """
        返回所有当前未标记视频的原始ID列表。
        用于 `main` 函数中的 `get_video_candidates`。
        """
        if not self.is_train_set:
            raise RuntimeError("Only training set can provide unlabeled candidates.")
        return list(self.unlabeled_video_ids)

    def get_num_labeled_videos(self):
        """
        返回当前已标记视频的数量。
        """
        return len(self.labeled_video_ids)

    def add_video_to_labeled(self, video_id):
        """
        将指定视频ID标记为已标注。
        """
        if not self.is_train_set:
            raise RuntimeError("Cannot add labeled videos to validation set.")
        if video_id in self.unlabeled_video_ids:
            self.unlabeled_video_ids.remove(video_id)
            self.labeled_video_ids.add(video_id)
        else:
            print(f"Warning: Video {video_id} is already labeled or not in unlabeled pool.")

    def reset(self):
        """
        重置数据集状态，将所有视频恢复为未标记状态。
        用于每个RL Episode开始时。
        """
        if self.is_train_set:
            self.labeled_video_ids = set()
            self.unlabeled_video_ids = set(self.all_video_ids)
            self._initialize_labeled_set(0.01) # 重新初始化，或根据 args.initial_labeled_ratio 决定

    def get_video(self, vid_idx):
        """
        为策略网络获取一个视频经过确定性变换后的张量，用于计算状态。
        """
        video_name, _, _ = self.video_list_info[vid_idx]
        full_path = os.path.join(self.video_dir, video_name)

        video_reader = decord.VideoReader(full_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)

        # 1. 总是从视频中心采样一个片段，保证确定性
        start_frame = (total_frames - self.clip_len) // 2
        indices = np.arange(start_frame, start_frame + self.clip_len)
        if total_frames < self.clip_len:
            indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)

        # 2. 加载原始片段，形状 [T, C, H, W]
        clip = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()

        # 3. ❤️【关键】使用专为评估/状态计算准备的 eval_transform
        if self.eval_transform:
            transformed_clip = self.eval_transform(clip)
        else:
            raise RuntimeError("eval_transform is not set for this dataset!")

        # 4. 返回一个处理好的 [C, T, H, W] 张量
        return transformed_clip.unsqueeze(0)

    # 针对 replay buffer 中的 state_subset 可能会调用的方法
    def get_labeled_videos_info(self):
        """返回已标注视频的完整信息列表 (video_name, label, original_index)"""
        return [self.video_list_info[vid_id] for vid_id in self.labeled_video_ids]