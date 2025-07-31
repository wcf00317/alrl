
import torch,yaml
from data.data_utils import get_data
from models.model_utils import create_models
import utils.parser as parser
def main(args):
    if getattr(args, 'config', None):
        print(f"加载配置文件: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # 合并 YAML 参数（不会覆盖已有 argparse 参数）
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    arg_key = f"{key}_{sub_key}"
                    if not hasattr(args, arg_key) or getattr(args, arg_key) is None:
                        setattr(args, arg_key, sub_value)
            else:
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
    net, policy_net, target_net = create_models(dataset=args.dataset,

                                                model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path,
                                                num_classes=args.num_classes,
                                                use_policy=True,
                                                embed_dim=args.embed_dim)
    train_loader, train_set, val_loader, candidate_set = get_data(
        data_path=args.data_path,
        tr_bs=args.train_batch_size,
        vl_bs=args.val_batch_size,
        n_workers=4,  # 或者 args.n_workers，如果你支持这个参数
        clip_len=args.clip_len,
        transform_type='c3d',
        test=args.test
    )
    net.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            # inputs: [N, num_clips, C, T, H, W], labels: [N]
            inputs, labels = inputs.cuda(), labels.cuda()

            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]

            # 模型输出 outputs 的形状是 [N * num_clips, num_classes]
            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)
            # 【测试策略】平均化输出
            # 从 [N * num_clips, num_classes] -> [N, num_clips, num_classes]
            outputs_reshaped = outputs.view(batch_size, num_clips, -1)
            # 沿着 num_clips 维度求平均
            final_outputs = outputs_reshaped.mean(dim=1)  # 最终形状变为 [N, num_classes]

            # 使用平均化后的结果计算测试指标
            preds = final_outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += batch_size

    # 确保 test_total 不为0，避免除零错误
    if test_total == 0:
        test_acc = 0.0
    else:
        test_acc = test_correct / test_total
    print("验证集准确率为："+str(test_acc))
    return test_acc
if __name__ == '__main__':
    ####------ Parse arguments from console  ------####
    args = parser.get_arguments()
    main(args)