# 在 models/query_network.py 文件中，可以新增以下这个改进版的策略网络
# 你可以在 create_models 函数中选择性地使用它来替代旧的 TransformerPolicyNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedTransformerPolicyNet(nn.Module):
    """
    一个使用交叉注意力机制来增强表达能力的策略网络。
    """

    def __init__(self, input_dim=4096, num_heads=8, dropout=0.1):
        super(AdvancedTransformerPolicyNet, self).__init__()
        self.input_dim = input_dim

        # 定义一个交叉注意力层
        # 候选视频的特征将是Query，已标注集的特征是Key和Value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 确保输入维度的顺序是 (batch, seq, feature)
        )

        # 后续处理模块，与原版类似
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim)
        )

        # 最终的Q值预测头
        self.q_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, subset):
        """
        x:       [B, D]             # 单个候选视频的特征
        subset:  [B, M, D]          # 已标注集视频的特征 (M是已标注数)
        """
        # 确保 x 的维度是 [B, 1, D]，以匹配MultiheadAttention的输入格式
        query = x.unsqueeze(1)

        # 使用交叉注意力
        # query会“询问”subset，得到一个考虑了subset相关信息的、增强版的上下文向量
        attn_output, _ = self.cross_attention(query=query, key=subset, value=subset)

        # 残差连接与层归一化 (标准的Transformer模块)
        x_enhanced = self.layer_norm1(query + attn_output)

        # 前馈网络
        ffn_output = self.ffn(x_enhanced)

        # 第二个残差连接与层归一化
        x_final = self.layer_norm2(x_enhanced + ffn_output)

        # 预测Q值
        q_value = self.q_predictor(x_final)

        return q_value.squeeze(1)  # 返回 [B, 1] 的Q值