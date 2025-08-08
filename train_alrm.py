# 文件名: train_alrm.py (最终可视化Debug版)

import os
import torch
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.reward_model import KAN_ActiveLearningRewardModel
import subprocess


# (train_reward_model 和 check_model_weights 函数保持不变)
def check_model_weights(model):
    """
    An auxiliary function to check for nan or inf values in model parameters.
    """
    for name, param in model.named_parameters():
        if not torch.all(torch.isfinite(param)):
            print(f"!!! Invalid weights detected: {name} !!!")
            return False
    return True


def train_reward_model(alrm_model, alrm_dataset, optimizer, num_epochs=30, batch_size=8, margin=0.1, lamb=0.01):
    """
    Train the reward model using a ranking loss and KAN's regularization loss.
    """
    if len(alrm_dataset) < batch_size:
        print(f"Insufficient data ({len(alrm_dataset)} pairs) to start training.")
        return False

    print(f"Starting to train ALRM with {len(alrm_dataset)} preference pairs...")
    alrm_model.train()
    data_loader = DataLoader(alrm_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_epoch_loss = 0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            winner_features = batch['winner'].cuda()
            loser_features = batch['loser'].cuda()
            all_features = torch.cat([winner_features, loser_features], dim=0)
            forward_result = alrm_model(all_features)

            if isinstance(forward_result, dict):
                scores = forward_result['outputs']
                reg_loss = forward_result.get('reg', 0.0)
            else:
                scores = forward_result
                reg_loss = 0.0

            score_winner, score_loser = torch.split(scores, [len(winner_features), len(loser_features)])
            ranking_loss = torch.clamp(margin - (score_winner - score_loser), min=0).mean()
            total_batch_loss = ranking_loss + lamb * reg_loss

            if torch.isnan(total_batch_loss):
                print(f"\n!!! BUG DETECTED: Total loss became nan at Epoch {epoch + 1}, Batch {i + 1} !!!")
                return False

            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(alrm_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_epoch_loss += total_batch_loss.item()

        print(
            f"ALRM Training Epoch {epoch + 1}/{num_epochs}, Average Total Loss: {total_epoch_loss / len(data_loader):.4f}")

    print("\nTraining finished.")
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
    optimizer = optim.Adam(alrm_model.parameters(), lr=1e-4)
    training_successful = train_reward_model(alrm_model, alrm_dataset, optimizer, lamb=0.01)

    if training_successful:
        if not check_model_weights(alrm_model):
            print("Skipping due to invalid model weights.")
            exit()

        save_path = os.path.join(ckpt_path, exp_name, 'kan_alrm_model.pth')
        torch.save(alrm_model.state_dict(), save_path)
        print(f"Trained KAN reward model saved to: {save_path}")

        print("Generating visualization for the KAN reward model...")
        viz_folder = os.path.join(ckpt_path, exp_name, "kan_visualization")
        if not os.path.exists(viz_folder): os.makedirs(viz_folder)

        try:
            # 步骤 1: 调用 .plot() 并捕获其返回的 "蓝图" 字符串
            vis_dataloader = DataLoader(alrm_dataset, batch_size=min(16, len(alrm_dataset)), shuffle=True)
            vis_batch = next(iter(vis_dataloader))
            vis_features = torch.cat([vis_batch['winner'], vis_batch['loser']], dim=0).cuda()
            alrm_model(vis_features)

            gv_source = alrm_model.kan_network.plot(
                folder=viz_folder,
                in_vars=['Mean Entropy', 'Std Entropy', 'Mean Similarity', 'Std Similarity'],
                out_vars=['Predicted Reward']
            )
            print(gv_source)
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            # 步骤 2: 检查 "蓝图" 是否有效
            if gv_source is None or not isinstance(gv_source, str) or len(gv_source) < 10:
                raise RuntimeError(
                    f"KAN's .plot() function returned an invalid or empty source string. Visualization cannot proceed.")

            # 步骤 3: 手动保存 .gv 文件并调用 dot 命令渲染
            gv_path = os.path.join(viz_folder, "kan_model.gv")
            with open(gv_path, "w") as f:
                f.write(gv_source)
            print(f"Graphviz source file saved to: {gv_path}")

            pdf_path = os.path.join(viz_folder, "kan_model_rendered.pdf")
            command = ["dot", "-Tpdf", gv_path, "-o", pdf_path]

            print(f"Executing rendering command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            if result.returncode == 0 and os.path.exists(pdf_path):
                print(f"Visualization PDF successfully generated: {pdf_path}")
            else:
                print("\n" + "=" * 30)
                print("!!! PDF Rendering Failed !!!")
                print("The `dot` command encountered an error:")
                print("--- STDOUT (Standard Output) ---")
                print(result.stdout if result.stdout else "No standard output.")
                print("--- STDERR (Standard Error) ---")
                print(result.stderr if result.stderr else "No standard error.")
                print("=" * 30 + "\n")

        except Exception as e:
            print("\n" + "=" * 30)
            print("!!! KAN visualization process encountered a Python exception !!!")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print("=" * 30 + "\n")
    else:
        print("Skipping due to training failure or insufficient data.")