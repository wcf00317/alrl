# 文件名: train_alrm.py (最终稳定版 + 官方正则化技巧)

import os
import torch
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.reward_model import KAN_ActiveLearningRewardModel
import subprocess


def check_model_weights(model, stage=""):
    """一个辅助函数，用于检查模型参数中是否存在nan或inf。"""
    for name, param in model.named_parameters():
        if not torch.all(torch.isfinite(param)):
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!! 关键错误: 在 '{stage}' 阶段，权重变为无效 !!!")
            print(f"!!! 损坏的参数层: '{name}'")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            return False
    return True


def train_reward_model(alrm_model, alrm_dataset, optimizer, num_epochs=30, batch_size=8, margin=0.1, lamb=0.01):
    if len(alrm_dataset) < batch_size:
        print(f"数据量不足 ({len(alrm_dataset)}对)，无法训练。")
        return False

    print(f"开始训练奖励模型（ALRM），共 {len(alrm_dataset)} 个偏好对...")
    alrm_model.train()
    data_loader = DataLoader(alrm_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_epoch_loss = 0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            winner_features = batch['winner'].cuda()
            loser_features = batch['loser'].cuda()

            # --- 核心修正 1: 正确处理KAN的前向传播和正则化损失 ---
            all_features = torch.cat([winner_features, loser_features], dim=0)

            # 执行前向传播
            forward_result = alrm_model(all_features)

            # 检查返回类型，以兼容不同版本的pykan
            if isinstance(forward_result, dict):
                scores = forward_result['outputs']
                reg_loss = forward_result.get('reg', 0.0)  # 如果没有'reg'键，则正则损失为0
            else:
                scores = forward_result
                reg_loss = 0.0  # 如果只返回张量，我们暂时不使用正则化

            score_winner, score_loser = torch.split(scores, [len(winner_features), len(loser_features)])

            # 计算主要的排序损失
            ranking_loss = torch.clamp(margin - (score_winner - score_loser), min=0).mean()

            # 最终的总损失
            total_batch_loss = ranking_loss + lamb * reg_loss

            if torch.isnan(total_batch_loss):
                print(f"\n!!! BUG定位: Epoch {epoch + 1}, Batch {i + 1}，总损失变为 nan !!!")
                return False

            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(alrm_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_epoch_loss += total_batch_loss.item()

        print(f"ALRM 训练 Epoch {epoch + 1}/{num_epochs}, 平均总损失: {total_epoch_loss / len(data_loader):.4f}")

    print("\n训练完成。")
    return True


if __name__ == '__main__':
    # ... (配置和数据加载部分保持不变) ...
    exp_name = 'ucf101_exp'
    ckpt_path = './checkpoints'
    input_dim = 4
    alrm_data_path = os.path.join(ckpt_path, exp_name, 'alrm_preference_data.pkl')
    with open(alrm_data_path, 'rb') as f:
        alrm_dataset = pickle.load(f)

    alrm_model = KAN_ActiveLearningRewardModel(input_dim=input_dim, hidden_layers=[8, 4]).cuda()
    optimizer = optim.Adam(alrm_model.parameters(), lr=1e-4)  # 使用较保守的学习率

    training_successful = train_reward_model(alrm_model, alrm_dataset, optimizer, lamb=0.01)

    if training_successful:
        if not check_model_weights(alrm_model, stage="训练后"):
            print("由于模型权重无效，跳过保存和可视化。")
            exit()

        save_path = os.path.join(ckpt_path, exp_name, 'kan_alrm_model.pth')
        torch.save(alrm_model.state_dict(), save_path)
        print(f"训练好的KAN奖励模型已保存至: {save_path}")

        # --- 可视化 ---
        print("正在生成KAN奖励模型的可视化图...")
        viz_folder = os.path.join(ckpt_path, exp_name, "kan_visualization")
        try:
            # 必须先执行一次前向传播
            dummy_input = torch.randn(1, input_dim).cuda()
            alrm_model(dummy_input)

            # .plot() 函数
            gv_source = alrm_model.kan_network.plot(
                folder=viz_folder,
                in_vars=['Mean Entropy', 'Std Entropy', 'Mean Similarity', 'Std Similarity'],
                out_vars=['Predicted Reward'],
                beta=100
            )

            # 手动调用dot命令进行渲染
            gv_path = os.path.join(viz_folder, "kan_model.gv")
            with open(gv_path, "w") as f:
                f.write(gv_source)
            pdf_path = os.path.join(viz_folder, "kan_model.pdf")
            command = ["dot", "-Tpdf", gv_path, "-o", pdf_path]
            print(f"正在执行渲染命令: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"可视化PDF已成功生成: {pdf_path}")
            else:
                print("\n" + "=" * 30);
                print("!!! PDF 渲染失败 !!!");
                print("`dot` 命令执行出错:");
                print("--- STDOUT ---");
                print(result.stdout);
                print("--- STDERR ---");
                print(result.stderr);
                print("=" * 30 + "\n")
        except Exception as e:
            print("\n" + "=" * 30);
            print("!!! KAN 可视化过程出现Python异常 !!!");
            print(f"错误类型: {type(e).__name__}");
            print(f"错误信息: {e}");
            print("=" * 30 + "\n")
    else:
        print("由于训练失败或数据不足，跳过模型保存和可视化。")