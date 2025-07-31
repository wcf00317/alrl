# ralis/data/hmdb_dataset.py
import os
import numpy as np
import decord
from torch.utils.data import Dataset

class HmdbDataset(Dataset):
    def __init__(self, annotation_file, video_dir, transform=None):
        self.video_dir = video_dir
        self.video_list = self._load_annotations(annotation_file)
        self.transform = transform

    def _load_annotations(self, file):
        # 每行格式: video_path label
        with open(file, 'r') as f:
            return [line.strip().split() for line in f]

    def __getitem__(self, index):
        video_name, label = self.video_list[index]
        full_path = os.path.join(self.video_dir, video_name)
        video = decord.VideoReader(full_path)
        # 简单处理：取中间的16帧
        total = len(video)
        center = total // 2
        indices = list(range(center - 8, center + 8))
        clip = video.get_batch(indices).permute(0, 3, 1, 2).float() / 255.0  # T,C,H,W

        if self.transform:
            clip = self.transform(clip)

        return clip, int(label)

    def __len__(self):
        return len(self.video_list)
