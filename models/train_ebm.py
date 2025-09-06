# 文件名: train_ebm.py

import os
import pickle
import torch
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier


def train_ebm_reward_model(preference_data, exp_dir):
    """
    使用差分特征训练EBM奖励模型。

    :param preference_data: 从第一阶段收集的偏好数据列表。
    :param exp_dir: 实验目录，用于保存模型。
    :return: 训练是否成功 (bool)。
    """
    print("\n--- 正在准备EBM训练数据 ---")
    if not preference_data:
        print("警告: 偏好数据为空，无法训练EBM模型。")
        return False

    # 1. 提取所有特征并计算全局均值
    all_features = []
    for pair in preference_data:
        all_features.append(pair['winner'])
        all_features.append(pair['loser'])

    all_features_tensor = torch.stack(all_features)
    global_mean_feature = torch.mean(all_features_tensor, dim=0)
    print(f"从 {len(all_features)} 个特征向量中计算出全局均值特征。")

    # 2. 创建差分特征和标签
    diff_features = []
    labels = []
    for pair in preference_data:
        # 正样本: Δx = x_best - x_worst, y=1
        diff_pos = pair['winner'] - pair['loser']
        diff_features.append(diff_pos.numpy())
        labels.append(1)

        # 负样本: Δx = x_worst - x_best, y=0
        diff_neg = pair['loser'] - pair['winner']
        diff_features.append(diff_neg.numpy())
        labels.append(0)

    X_train = np.array(diff_features)
    y_train = np.array(labels)
    print(f"已创建差分特征数据集，包含 {len(X_train)} 个样本。")

    # 3. 训练EBM分类器
    print("--- 开始训练EBM分类器 ---")
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X_train, y_train)
    print("--- EBM模型训练完成 ---")

    # 4. 保存模型和均值特征
    ebm_scorer = {
        'model': ebm,
        'mean_feature': global_mean_feature
    }
    scorer_path = os.path.join(exp_dir, 'ebm_scorer.pkl')
    with open(scorer_path, 'wb') as f:
        pickle.dump(ebm_scorer, f)
    print(f"EBM计分器已保存至: {scorer_path}")

    return True


def load_ebm_scorer(exp_dir):
    """加载EBM计分器 (模型 + 均值特征)。"""
    scorer_path = os.path.join(exp_dir, 'ebm_scorer.pkl')
    if not os.path.exists(scorer_path):
        raise FileNotFoundError(f"找不到EBM计分器文件: {scorer_path}")
    with open(scorer_path, 'rb') as f:
        ebm_scorer = pickle.load(f)
    print("EBM计分器加载成功。")
    return ebm_scorer


def predict_ebm_reward(ebm_scorer, batch_features):
    """
    使用训练好的EBM模型，通过与全局均值比较来预测奖励。

    :param ebm_scorer: 包含 'model' 和 'mean_feature' 的字典。
    :param batch_features: 单个批次的特征向量 (torch.Tensor)。
    :return: 代理奖励分数 (float, 0到1之间)。
    """
    ebm_model = ebm_scorer['model']
    mean_feature = ebm_scorer['mean_feature']

    # 确保在同一设备上
    if batch_features.device != mean_feature.device:
        mean_feature = mean_feature.to(batch_features.device)

    # 计算差分特征 Δx = x - μ
    diff_feature = batch_features - mean_feature

    # EBM需要numpy输入
    diff_feature_np = diff_feature.cpu().numpy().reshape(1, -1)

    # 预测 P(x 胜 μ)，即正类的概率
    proba = ebm_model.predict_proba(diff_feature_np)
    reward = proba[0, 1]  # 获取标签为1的概率

    return reward