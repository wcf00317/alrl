import torch
import torch.nn as nn
from kan import KAN  # 确保你已经 pip install pykan


class KAN_ActiveLearningRewardModel(nn.Module):
    """
    一个基于Kolmogorov-Arnold Network (KAN)的奖励模型。
    它将作为MLP版本的直接替代品，核心优势在于其可解释性。
    """

    def __init__(self, input_dim, hidden_layers=[10, 5], grid_size=5, spline_order=3):
        """
        初始化KAN奖励模型。

        :param input_dim: 输入特征向量的维度 (例如，批次级特征是4)。
        :param hidden_layers: 一个列表，定义了每个隐藏层的宽度（神经元数量）。
                              例如 [10, 5] 表示一个2层的隐藏网络。
        :param grid_size: 控制样条函数精度的网格点数量。数值越大，函数越平滑，但参数越多。
        :param spline_order: 样条函数的阶数（例如，3代表三次样条）。
        """
        super(KAN_ActiveLearningRewardModel, self).__init__()

        # 构建KAN的网络结构
        # 结构为：[输入层, 隐藏层1, 隐藏层2, ..., 输出层]
        layer_widths = [input_dim] + hidden_layers + [1]  # 输出是一个标量Q值

        self.kan_network = KAN(
            width=layer_widths,
            grid=grid_size,
            k=spline_order,
            seed=42
        )

    def forward(self, x):
        """
        前向传播，直接调用内部的KAN网络。
        """
        return self.kan_network(x)


# --- 新增MLP与KAN对比---
class MLP_ActiveLearningRewardModel(nn.Module):
    """
    一个基于MLP的奖励模型，作为KAN模型的Baseline。
    """

    def __init__(self, input_dim, hidden_layers=[16, 8]):
        """
        初始化MLP奖励模型。

        :param input_dim: 输入特征向量的维度 (例如，批次级特征是4)。
        :param hidden_layers: 一个列表，定义了每个隐藏层的宽度。
        """
        super(MLP_ActiveLearningRewardModel, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # 输出层，产生一个标量奖励值

        self.mlp_network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播，通过MLP网络。
        """
        return self.mlp_network(x)
def get_batch_features(batch_entropies, batch_similarities):
    """
    将一个批次的信息聚合成一个特征向量，作为ALRM的输入。
    这是一个简单的实现，后续可以扩展。

    :param batch_entropies: 一个包含批次内所有样本熵值的列表。
    :param batch_similarities: 一个包含批次内每个样本与已标注集相似度的列表。
    :return: 一个代表该批次特性的PyTorch张量。
    """
    if not batch_entropies:
        return torch.zeros(4)  # 返回一个零向量以防批次为空

    # 计算批次的统计特征
    mean_entropy = sum(batch_entropies) / len(batch_entropies)
    std_entropy = torch.std(torch.tensor(batch_entropies)).item() if len(batch_entropies) > 1 else 0

    mean_similarity = sum(batch_similarities) / len(batch_similarities)
    std_similarity = torch.std(torch.tensor(batch_similarities)).item() if len(batch_similarities) > 1 else 0

    # 拼接成最终的批次特征向量
    batch_feature_vector = torch.tensor([
        mean_entropy,
        std_entropy,
        mean_similarity,
        std_similarity
    ])

    return batch_feature_vector