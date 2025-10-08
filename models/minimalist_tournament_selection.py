# 文件: wcf00317/alrl/alrl-reward_model/models/minimalist_tournament_selection.py

import torch
import torch.nn.functional as F
import random
import numpy as np
from copy import deepcopy

import utils.al_scoring as scoring


class MinimalistTournamentSelector:
    """
    极简锦标赛实现（**不含 k-center**）：
    并集提名 -> BALD+EGL Borda 排序 -> 中位数阈贪心去冗余 -> 顺序切分为若干决赛批次
    """
    def __init__(self, model, train_set, val_loader, criterion, args, base_val_acc):
        # ✅ 增加一个 base_val_acc 参数
        self.model = model
        self.train_set = train_set
        self.val_loader = val_loader
        self.criterion = criterion
        self.args = args
        self.base_val_acc = base_val_acc
        self.unlabeled_indices = self.train_set.get_candidates_video_ids()
        self.scores = {}
        self.precomputed_data = {}
        self.idx_to_pos = {}

        if self.unlabeled_indices:
            self._pre_calculate_and_score()
        else:
            print("警告: 未标注池为空，跳过所有计算。")

    # ---------- 工具函数 ----------
    def _to_tensor(self, x):
        """确保评分为 torch.Tensor（按长度为未标注池长度排列）"""
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, dtype=torch.float)

    # ---------- 主要步骤 ----------
    def _pre_calculate_and_score(self):
        """预计算并根据 args 动态计算启发式分数，以及 bald/egl"""
        # 预计算（包含 embeddings、模型预测等）
        self.precomputed_data = scoring.precompute_data_for_scoring(
            self.args, self.model, self.unlabeled_indices, self.train_set, getattr(self.args, 'val_batch_size', 32)
        )

        print("根据配置计算启发式分数...")
        feature_map = {
            'use_statistical_features': ('entropy', scoring.compute_entropy_score),
            'use_diversity_feature': ('diversity', scoring.compute_diversity_score),
            'use_representativeness_feature': ('representativeness', scoring.compute_representativeness_score),
            'use_prediction_margin_feature': ('prediction_margin', scoring.compute_prediction_margin_score),
            'use_labeled_distance_feature': ('labeled_distance', scoring.compute_labeled_distance_score),
            'use_neighborhood_density_feature': ('neighborhood_density', scoring.compute_neighborhood_density_score),
            'use_temporal_consistency_feature': ('temporal_consistency', scoring.compute_temporal_consistency_score)
        }

        for arg_name, (score_name, score_func) in feature_map.items():
            if getattr(self.args, arg_name, False):
                print(f"  - 计算 {score_name}")
                self.scores[score_name] = self._to_tensor(score_func(self.precomputed_data))

        # BALD / EGL 独立计算（通常更精确）
        print("  - 计算 bald / egl ...")
        self.scores['bald'] = self._to_tensor(scoring.compute_bald_score(self.model, self.unlabeled_indices, self.train_set))
        #self.scores['egl'] = self._to_tensor(scoring.compute_egl_score(self.model, self.unlabeled_indices, self.train_set))
        egl_strat = getattr(self.args, 'egl_strategy', 'adaptive_k')
        print(f"  - 使用 '{egl_strat}' 策略计算 EGL 分数...")
        # scoring.compute_egl_score可替换为scoring.compute_egl_score_aprox的近似版本，会更快;
        # compute_egl_adaptive_topk是最佳的权衡

        if egl_strat == 'adaptive_k':
            batch_size = getattr(self.args, 'val_batch_size', 16)
            self.scores['egl'] = self._to_tensor(scoring.compute_egl_adaptive_topk(self.model, self.unlabeled_indices, self.train_set,
                                                                                   prob_threshold=0.95, batch_size=batch_size))
        elif egl_strat == 'approx':
            self.scores['egl'] = self._to_tensor(
                scoring.compute_egl_score_approx(self.model, self.unlabeled_indices, self.train_set))
        elif egl_strat == 'standard':
            # 假设您的理论标准版本已命名为 compute_egl_score
            self.scores['egl'] = self._to_tensor(
                scoring.compute_egl_score(self.model, self.unlabeled_indices, self.train_set)
            )

        # 索引映射
        self.idx_to_pos = {vid_idx: pos for pos, vid_idx in enumerate(self.unlabeled_indices)}

    def _nominate_candidates(self, total_budget):
        """
        并集提名：对每个启发式取 Top-(c * total_budget) 并取并集。
        total_budget = B * num_finalists（由外部调用者传入）
        """
        print("Step 1: 并集提名...")
        c = getattr(self.args, 'nomination_ratio_c', 3)
        nomination_size = max(1, int(c * total_budget))

        candidate_union = set()
        for metric_name, scores in self.scores.items():
            print(f"  - 来自 {metric_name} 的 top-{nomination_size} 提名")
            k = min(nomination_size, scores.numel())
            if k <= 0:
                continue
            _, topk_idx = torch.topk(scores, k)
            topk_idx = topk_idx.tolist()
            topk_vids = [self.unlabeled_indices[i] for i in topk_idx]
            candidate_union.update(topk_vids)

        candidate_list = list(candidate_union)

        # 如果并集过大则用 bald+egl 的 Borda 截断到 6*total_budget（更公平）
        cap = max(6 * total_budget, 1)
        if len(candidate_list) > cap:
            print(f"  - 候选池过大 ({len(candidate_list)})，使用 bald+egl Borda 截断到 {cap}")
            ranks = []
            for metric in ['bald', 'egl']:
                if metric not in self.scores:
                    continue
                all_scores = self.scores[metric]
                sorted_idx = torch.argsort(all_scores, descending=True).tolist()
                rank_map = {self.unlabeled_indices[idx]: len(sorted_idx) - rank for rank, idx in enumerate(sorted_idx)}
                ranks.append([rank_map.get(vid, 0) for vid in candidate_list])
            if not ranks:
                # 极端回退
                return candidate_list[:cap]
            borda = np.sum(np.array(ranks), axis=0)
            sorted_by_borda = [vid for _, vid in sorted(zip(borda, candidate_list), key=lambda x: x[0], reverse=True)]
            return sorted_by_borda[:cap]

        return candidate_list

    def _rank_and_shortlist(self, candidate_list, total_budget):
        """
        Step 2: 只用 bald + egl 做 Borda 排序，取前 2 * total_budget 作为短名单
        """
        print("Step 2: BALD+EGL Borda 排序并截取短名单...")
        if not candidate_list:
            return []

        ranks = []
        for metric in ['bald', 'egl']:
            if metric not in self.scores:
                continue
            all_scores = self.scores[metric]
            sorted_idx = torch.argsort(all_scores, descending=True).tolist()
            rank_map = {self.unlabeled_indices[idx]: len(sorted_idx) - rank for rank, idx in enumerate(sorted_idx)}
            ranks.append([rank_map.get(vid, 0) for vid in candidate_list])

        if not ranks:
            # 回退到随机
            random.shuffle(candidate_list)
            return candidate_list[:min(len(candidate_list), 2 * total_budget)]

        borda = np.sum(np.array(ranks), axis=0)
        sorted_candidates = [vid for _, vid in sorted(zip(borda, candidate_list), key=lambda x: x[0], reverse=True)]
        shortlist_size = min(len(sorted_candidates), max(2 * total_budget, 1))
        return sorted_candidates[:shortlist_size]

    def _prune_with_median_distance(self, shortlist, total_needed):
        """
        Step 3: 在短名单上做中位数距离的贪心去冗余，直到得到 total_needed 个样本。
        如果不足则按短名单顺序补齐。
        """
        print("Step 3: 中位数阈贪心去冗余...")
        if not shortlist:
            return []

        # 获取 embeddings
        embeddings = self.precomputed_data.get('embeddings', None)
        if embeddings is None:
            # 无嵌入则直接截取
            return shortlist[:total_needed]

        positions = [self.idx_to_pos[vid] for vid in shortlist]
        emb = embeddings[positions]  # [len(shortlist), D]
        emb = F.normalize(torch.tensor(emb, dtype=torch.float), p=2, dim=1) if not torch.is_tensor(emb) else F.normalize(emb, p=2, dim=1)

        # 计算距离矩阵并取正的中位数
        dist = torch.cdist(emb, emb, p=2)
        if dist.numel() <= 1:
            median_dist = 0.0
        else:
            # 排除 0 距离（自己对自己的）
            non_zero = dist[dist > 0]
            median_dist = non_zero.median().item() if non_zero.numel() > 0 else 0.0

        alpha = getattr(self.args, 'distance_threshold_alpha', 0.9)
        threshold = alpha * median_dist if median_dist > 0 else 0.0
        print(f"  - median_dist={median_dist:.4f}, alpha={alpha}, threshold={threshold:.4f}")

        selected = []
        for i, vid in enumerate(shortlist):
            if len(selected) >= total_needed:
                break
            if not selected:
                selected.append(vid)
                continue
            cur_pos = self.idx_to_pos[vid]
            cur_emb = emb[i].unsqueeze(0)  # shape [1,D]
            selected_pos = [self.idx_to_pos[v] for v in selected]
            sel_idx = [shortlist.index(v) for v in selected]  # positions within shortlist
            sel_emb = emb[sel_idx]
            min_dist = torch.cdist(cur_emb, sel_emb, p=2).min().item()
            if min_dist >= threshold:
                selected.append(vid)

        # 若未选满，从短名单顺序补齐
        if len(selected) < total_needed:
            remaining = [v for v in shortlist if v not in selected]
            to_add = remaining[:(total_needed - len(selected))]
            selected.extend(to_add)
        return selected[:total_needed]

    def _run_sanity_check(self, batch_vids, total_batch_size):
        """
        可选的理智检查（开销较大，按 args.run_sanity_check 控制）：
        将给定的 batch 与等大小随机 batch 做短跑训练比较，若表现差则温和替换部分样本。
        注意：需要外部提供 train_har_for_reward, add_labeled_videos 等工具（保持你原有的实现）。
        """
        if not getattr(self.args, 'run_sanity_check', False):
            return batch_vids

        print("Step 4: 运行可选理智检查（比较与随机 basline）...")
        # 这里保留接口结构，实际实现依赖外部 train_har_for_reward 与 add_labeled_videos
        try:
            from run_rl_with_alrm import train_har_for_reward
            from models.model_utils import add_labeled_videos
        except Exception as e:
            print("  - 无法导入微调工具，跳过理智检查。", e)
            return batch_vids

        # baseline acc（建议在外部计算一次并传入，这里为保险性重新计算）
        model_copy = deepcopy(self.model)
        # note: 这里为简洁起见假设 train_har_for_reward 接受 (model, train_loader, val_loader, optimizer, criterion, args)
        # 实际工程中可以把 base_val_acc 缓存到 selector，避免重复计算
        _, base_val_acc = train_har_for_reward(model_copy,
                                              None,  # 期望外部传入或使用已缓存的训练集 loader
                                              self.val_loader, None, self.criterion, self.args)

        # 我方微调（短跑）
        net_copy = deepcopy(self.model)
        temp_set = deepcopy(self.train_set)
        add_labeled_videos(self.args, [], batch_vids, temp_set, budget=getattr(self.args, 'budget_labels', 0), n_ep=-1)
        temp_loader = None  # 外部提供数据加载器创建函数以替换
        _, acc_our = train_har_for_reward(net_copy, temp_loader, self.val_loader, None, self.criterion, self.args)

        # 随机微调
        rand_batch = random.sample(self.unlabeled_indices, total_batch_size)
        net_copy2 = deepcopy(self.model)
        temp_set2 = deepcopy(self.train_set)
        add_labeled_videos(self.args, [], rand_batch, temp_set2, budget=getattr(self.args, 'budget_labels', 0), n_ep=-1)
        temp_loader2 = None
        _, acc_rand = train_har_for_reward(net_copy2, temp_loader2, self.val_loader, None, self.criterion, self.args)

        delta_our = acc_our - base_val_acc
        delta_rand = acc_rand - base_val_acc
        print(f"  - Δ our={delta_our:.4f}, Δ rand={delta_rand:.4f}")
        if delta_our < delta_rand - 0.005:
            # 替换 10%
            k = max(1, int(0.1 * total_batch_size))
            print(f"  - 我方落后，替换 {k} 个样本为随机样本")
            new_batch = batch_vids[:-k] + rand_batch[:k]
            return new_batch
        return batch_vids

    # ---------- 对外接口 ----------
    def get_finalists(self, budget, num_finalists=4):
        """
        返回 num_finalists 个候选批次，每个批次大小为 budget。
        """
        if not self.unlabeled_indices or len(self.unlabeled_indices) < budget:
            return [self.unlabeled_indices]

        total_budget = budget * max(1, num_finalists)
        # Step1: 并集提名
        candidate_pool = self._nominate_candidates(total_budget)
        # Step2: 排序并 shortlist
        shortlist = self._rank_and_shortlist(candidate_pool, total_budget)
        # Step3: 去冗余直到 total_budget
        selected = self._prune_with_median_distance(shortlist, total_budget)
        # 可选 Step4: sanity check (对整体 selected 进行一次性检查)
        if getattr(self.args, 'run_sanity_check', False):
            selected = self._run_sanity_check(selected, total_budget)

        # 切分为 num_finalists 个批次（按顺序），每个大小为 budget
        batches = [selected[i * budget:(i + 1) * budget] for i in range(num_finalists)]
        # 如果最后一个批次不足 size，会被截断；去掉空批次
        batches = [b for b in batches if len(b) > 0]
        return batches
