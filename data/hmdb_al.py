import os
import random
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_video

class HmdbALDataset(Dataset):
    def __init__(self, video_list, video_root, clip_len=16, transform=None, candidates_option=False,
                 only_last_labeled=True, num_each_iter=1):
        """
        video_list: [(video_path, label), ...]
        video_root: HMDB51 video folder
        """
        self.video_root = video_root
        self.transform = transform
        self.clip_len = clip_len

        self.all_videos = video_list
        self.candidates_option = candidates_option
        self.only_last_labeled = only_last_labeled
        self.num_each_iter = num_each_iter
        #set用来加速
        self.labeled_set = []
        self.labeled_set_set = set()

        self.unlabeled_pool = list(range(len(self.all_videos)))
        self.unlabeled_pool_set = set(self.unlabeled_pool)

    def __len__(self):
        if self.candidates_option:
            return len(self.unlabeled_pool)
        elif self.only_last_labeled:
            return self.num_each_iter
        else:
            return len(self.labeled_set)

    def _load_clip(self, video_path):
        video_abs = os.path.join(self.video_root, video_path)
        video, _, _ = read_video(video_abs, pts_unit='sec')
        # Simple uniform sampling
        if video.shape[0] < self.clip_len:
            indices = list(range(video.shape[0])) + [video.shape[0] - 1] * (self.clip_len - video.shape[0])
        else:
            start = random.randint(0, video.shape[0] - self.clip_len)
            indices = list(range(start, start + self.clip_len))
        clip = video[indices].permute(0, 3, 1, 2).float() / 255.0  # [T,C,H,W]
        return clip

    def __getitem__(self, index):
        if self.candidates_option:
            idx = self.unlabeled_pool[index]
        elif self.only_last_labeled:
            idx = self.labeled_round_indices[index]
        else:
            idx = self.labeled_set[index]

        video_path, label = self.all_videos[idx]
        clip = self._load_clip(video_path)

        if self.transform:
            clip = self.transform(clip)

        return clip, label, idx

    def add_indices(self, indices):
        for idx in indices:
            if idx not in self.labeled_set_set:
                self.labeled_set.append(idx)
                self.labeled_set_set.add(idx)
                self.labeled_round_indices.append(idx)
            if idx in self.unlabeled_pool_set:
                self.unlabeled_pool.remove(idx)
                self.unlabeled_pool_set.remove(idx)
        print(f"Added {len(indices)} new clips, total labeled = {len(self.labeled_set)}")
    def reset_round_indices(self):
        self.labeled_round_indices = []

    def get_unlabeled_indices(self):
        return deepcopy(self.unlabeled_pool)

    def get_labeled_indices(self):
        return deepcopy(self.labeled_set)
