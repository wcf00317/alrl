from data.base_dataset import ActiveLearningVideoDataset

class Sthv2Dataset(ActiveLearningVideoDataset):
    """
    Something-Something V2 (sthv2) 数据集类。
    继承自 ActiveLearningVideoDataset，仅实现特定于 sthv2 的标注加载逻辑。
    """
    def _load_annotations(self, file):
        """
        加载 sthv2 的标注文件。
        文件格式: video_name label
        """
        video_info = []
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                # sthv2的标注文件格式是 'video_name label'
                video_name, label = line.strip().split()
                video_info.append((video_name, label, i)) # 保存视频名、标签和原始索引
        return video_info