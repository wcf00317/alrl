# wcf00317/alrl/alrl-reward_model/data/kinetics.py

from data.base_dataset import ActiveLearningVideoDataset

class KineticsDataset(ActiveLearningVideoDataset):
    """
    Kinetics 数据集类。
    继承自 ActiveLearningVideoDataset，仅实现特定于 Kinetics 的标注加载逻辑。
    """
    def _load_annotations(self, file):
        """
        加载 Kinetics 的标注文件。
        文件格式: video_path label
        """
        video_info = []
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                video_name, label = line.strip().split()
                video_info.append((video_name, label, i)) # 保存视频名、标签和原始索引
        return video_info