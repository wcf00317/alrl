# wcf00317/alrl/alrl-reward_model/data/ucf101.py

from data.base_dataset import ActiveLearningVideoDataset

class UcfDataset(ActiveLearningVideoDataset):
    """
    UCF101 数据集类。
    继承自 ActiveLearningVideoDataset，仅实现特定于 UCF101 的标注加载逻辑。
    """
    def _load_annotations(self, file):
        """
        加载 UCF101 的标注文件。
        官方文件格式通常是: video_path label (label从1开始)
        我们的处理逻辑使其与其他数据集保持一致（label从0开始）。
        """
        video_info = []
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                # 假设的标注文件格式是 'video_name label'
                video_name, label = line.strip().split()
                video_info.append((video_name, label, i))  # 保存原始索引
        return video_info