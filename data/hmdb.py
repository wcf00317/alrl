# wcf00317/alrl/alrl-reward_model/data/hmdb.py

from data.base_dataset import ActiveLearningVideoDataset

class HmdbDataset(ActiveLearningVideoDataset):
    """
    HMDB51 数据集类。
    继承自 ActiveLearningVideoDataset，仅实现特定于 HMDB51 的标注加载逻辑。
    """
    def _load_annotations(self, file):
        """
        加载 HMDB51 的标注文件。
        文件格式: video_path label
        """
        video_info = []
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                # 假设HMDB51的标注文件格式是 'video_name label'
                video_name, label = line.strip().split()
                video_info.append((video_name, label, i)) # 保存原始索引
        return video_info