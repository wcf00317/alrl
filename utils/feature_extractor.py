# 文件名: wcf00317/alrl/alrl-reward_model/utils/feature_extractor.py

import torch
import torch.nn as nn  # --- NEW --- (为了使用 nn.Module)
import torch.nn.functional as F
from tqdm import tqdm


class UnifiedFeatureExtractor:
    """
    一个统一的、可配置的特征提取器。
    通过args参数，可以灵活地开启或关闭任何一种特征的计算，方便进行消融实验。
    """

    def __init__(self, args):
        self.args = args
        self.active_features = []
        self.feature_dim = 0

        print("\n--- Initializing Unified Feature Extractor ---")

        # --- 在这里，您可以像开关一样控制所有特征 ---
        # --- 在您的 .yaml 配置文件中设置这些参数为 true 或 false ---

        # 建议的Baseline特征
        if getattr(args, 'use_statistical_features', True):
            self.active_features.append('statistical')
            self.feature_dim += 4
            print("  - Statistical Features (熵/相似度统计): ENABLED")

        # 批内多样性
        if getattr(args, 'use_diversity_feature', False):
            self.active_features.append('diversity')
            self.feature_dim += 1
            print("  - Intra-Batch Diversity Feature: ENABLED")

        # 代表性
        if getattr(args, 'use_representativeness_feature', False):
            self.active_features.append('representativeness')
            self.feature_dim += 1
            print("  - Representativeness Feature: ENABLED")


        # 邻域密度
        if getattr(args, 'use_neighborhood_density_feature', False):
            self.active_features.append('neighborhood_density')
            self.feature_dim += 1
            print("  - Neighborhood Density Feature: ENABLED")

        # --- NEW: 为新特征增加开关 ---
        if getattr(args, 'use_prediction_margin_feature', False):
            self.active_features.append('prediction_margin')
            self.feature_dim += 2  # 均值和标准差
            print("  - Prediction Margin Feature: ENABLED")

        if getattr(args, 'use_labeled_distance_feature', False):
            self.active_features.append('labeled_distance')
            self.feature_dim += 2  # 均值和标准差
            print("  - Distance to Labeled Set Feature: ENABLED")
        # --- NEW: 结束 ---

        if not self.active_features:
            raise ValueError("错误：至少需要启用一种特征！请检查您的配置文件。")

        print(f"Total feature dimension: {self.feature_dim}")
        print("------------------------------------------\n")

    def extract(self, batch_video_indices, model, train_set, all_unlabeled_embeddings=None,
                all_labeled_embeddings=None):  # --- NEW: 增加 all_labeled_embeddings 参数 ---
        if not batch_video_indices:
            return torch.zeros(self.feature_dim)

        # --- MODIFIED: 修改函数调用以获取 probs ---
        batch_embeddings, batch_probs = self.get_embeddings_and_probs(batch_video_indices, model, train_set)

        feature_tensors = []

        # --- 按需计算和拼接特征 ---

        if 'statistical' in self.active_features:
            # --- MODIFIED: 从 probs 计算熵 ---
            entropies = (-torch.sum(batch_probs * torch.log(batch_probs + 1e-8), dim=1)).tolist()
            batch_similarities = [0.5] * len(batch_video_indices)
            mean_entropy = sum(entropies) / len(entropies)
            std_entropy = torch.std(torch.tensor(entropies)).item() if len(entropies) > 1 else 0
            mean_similarity = sum(batch_similarities) / len(batch_similarities)
            std_similarity = torch.std(torch.tensor(batch_similarities)).item() if len(batch_similarities) > 1 else 0
            feature_tensors.append(torch.tensor([mean_entropy, std_entropy, mean_similarity, std_similarity]))

        if 'diversity' in self.active_features:
            diversity_score = 0.0
            if len(batch_video_indices) > 1:
                normed_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                cosine_sim_matrix = torch.matmul(normed_embeddings, normed_embeddings.t())
                upper_tri_indices = torch.triu_indices(len(batch_video_indices), len(batch_video_indices), offset=1)
                mean_cosine_sim = cosine_sim_matrix[upper_tri_indices[0], upper_tri_indices[1]].mean().item()
                diversity_score = 1.0 - mean_cosine_sim
            feature_tensors.append(torch.tensor([diversity_score]))

        if 'representativeness' in self.active_features:
            representativeness_score = 0.0
            if all_unlabeled_embeddings is not None and len(all_unlabeled_embeddings) > 0:
                unlabeled_centroid = all_unlabeled_embeddings.mean(dim=0, keepdim=True).to(batch_embeddings.device)
                batch_centroid = batch_embeddings.mean(dim=0, keepdim=True)
                representativeness_score = F.cosine_similarity(unlabeled_centroid, batch_centroid).item()
            feature_tensors.append(torch.tensor([representativeness_score]))

        if 'neighborhood_density' in self.active_features:
            density_score = 0.0
            if all_unlabeled_embeddings is not None and len(all_unlabeled_embeddings) > 1:
                dist_matrix = torch.cdist(batch_embeddings.cpu(), all_unlabeled_embeddings.cpu())
                k = min(10, len(all_unlabeled_embeddings))
                knn_dists = torch.topk(dist_matrix, k, largest=False, dim=1).values
                mean_knn_dist = knn_dists.mean().item()
                density_score = 1.0 / (1.0 + mean_knn_dist)
            feature_tensors.append(torch.tensor([density_score]))

        # --- NEW: 增加新特征的计算逻辑 ---
        if 'prediction_margin' in self.active_features:
            # 对每个样本的概率分布进行排序
            sorted_probs, _ = torch.sort(batch_probs, dim=1, descending=True)
            # 边际 = (第一大概率) - (第二大概率)
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            mean_margin = margins.mean().item()
            std_margin = margins.std().item() if len(margins) > 1 else 0.0
            # 我们希望边际越小越好（模型越纠结），所以用 1.0 - margin 来表示不确定性的大小
            feature_tensors.append(torch.tensor([1.0 - mean_margin, std_margin]))

        if 'labeled_distance' in self.active_features:
            mean_dist = 0.0
            std_dist = 0.0
            if all_labeled_embeddings is not None and len(all_labeled_embeddings) > 0:
                # 计算批内每个样本到已标注集的最短距离
                dist_matrix = torch.cdist(batch_embeddings.cpu(), all_labeled_embeddings.cpu())
                min_dists, _ = torch.min(dist_matrix, dim=1)
                mean_dist = min_dists.mean().item()
                std_dist = min_dists.std().item() if len(min_dists) > 1 else 0.0
            feature_tensors.append(torch.tensor([mean_dist, std_dist]))
        # --- NEW: 结束 ---

        return torch.cat(feature_tensors)

    # --- MODIFIED: 重命名函数并返回 probs ---
    def get_embeddings_and_probs(self, video_indices, model, train_set):
        model.eval()
        batch_embeddings = []
        batch_probs = []  # --- MODIFIED ---

        with torch.no_grad():
            for vid_idx in video_indices:
                video_tensor = train_set.get_video(vid_idx).cuda()
                video_tensor = video_tensor.unsqueeze(0).cuda()
                features = model.extract_feat(video_tensor)[0]
                if features.shape[0] > 1: features = features.mean(dim=0, keepdim=True)
                batch_embeddings.append(features)
                logits = model.cls_head(features)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs)  # --- MODIFIED: 存储 probs 而不是 entropy ---

        return torch.cat(batch_embeddings, dim=0), torch.cat(batch_probs, dim=0)


def get_all_unlabeled_embeddings(args, model, train_set):
    # ... (此函数保持不变) ...
    unlabeled_indices = train_set.get_candidates_video_ids()
    print(f"正在为 {len(unlabeled_indices)} 个未标注视频预计算特征嵌入...")
    all_embeddings = []
    batch_size = args.val_batch_size
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(unlabeled_indices), batch_size), desc="Pre-computing embeddings"):
            batch_indices = unlabeled_indices[i:i + batch_size]
            videos = [train_set.get_video(idx) for idx in batch_indices]
            video_batch_tensor = torch.cat(videos, dim=0).cuda()
            video_batch_tensor = video_batch_tensor.unsqueeze(0).cuda()
            features = model.extract_feat(video_batch_tensor)[0]
            if features.dim() > 2: features = features.mean(dim=0)
            all_embeddings.append(features.cpu())
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, args.embed_dim)


# --- NEW: 增加一个辅助函数来获取已标注样本的嵌入 ---
def get_all_labeled_embeddings(args, model, train_set):
    """
    一个新增的辅助函数，用于计算所有已标注视频的特征嵌入。
    """
    labeled_indices = list(train_set.labeled_video_ids)
    if not labeled_indices:
        return None

    print(f"正在为 {len(labeled_indices)} 个已标注视频预计算特征嵌入...")

    all_embeddings = []
    batch_size = args.val_batch_size

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(labeled_indices), batch_size), desc="Pre-computing labeled embeddings"):
            batch_indices = labeled_indices[i:i + batch_size]
            # 假设 get_video 可以接受 is_eval 参数来使用评估时的数据增强
            videos = [train_set.get_video(idx) for idx in batch_indices]
            video_batch_tensor = torch.cat(videos, dim=0).cuda()
            video_batch_tensor = video_batch_tensor.unsqueeze(0).cuda()

            features = model.extract_feat(video_batch_tensor)[0]
            if features.dim() > 2: features = features.mean(dim=0)

            all_embeddings.append(features.cpu())

    return torch.cat(all_embeddings, dim=0) if all_embeddings else None
# --- NEW: 结束 ---