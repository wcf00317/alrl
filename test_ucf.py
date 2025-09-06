# filename: test.py

import torch
import yaml
from tqdm import tqdm

# Import necessary modules from your project
from data.data_utils import get_data
from models.model_utils import create_models
import utils.parser as parser

def main(args):
    """
    Main testing function.
    """
    # --- 1. Create Model ---
    # use_policy=False because we are only testing the classifier
    print("Creating the C3D model...")
    net, _, _ = create_models(
        dataset=args.dataset,
        model_cfg_path=args.model_cfg_path,
        model_ckpt_path=args.model_ckpt_path,
        num_classes=args.num_classes,
        use_policy=False
    )
    net.cuda()
    net.eval()  # Set the model to evaluation mode

    # --- 2. Load Data ---
    # We only need the validation loader for testing
    print(f"Loading data for dataset: {args.dataset}")
    _, _, val_loader, _ = get_data(
        data_path=args.data_path,
        tr_bs=args.train_batch_size,
        vl_bs=args.val_batch_size,
        dataset_name=args.dataset,
        n_workers=args.workers,
        clip_len=args.clip_len
    )

    # --- 3. Run Inference ---
    print("\n--- Starting evaluation on the validation set ---")
    test_correct = 0
    test_total = 0

    pbar = tqdm(val_loader, desc="Testing", unit="batch")

    with torch.no_grad():
        for inputs, labels, _ in pbar:
            # Move data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()

            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]

            # The model expects a batch of clips and returns predictions for each
            # Output shape: [N * num_clips, num_classes]
            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)

            # Average the predictions across clips for each video
            # Reshape to [N, num_clips, num_classes]
            outputs_reshaped = outputs.view(batch_size, num_clips, -1)
            # Average along the num_clips dimension
            final_outputs = outputs_reshaped.mean(dim=1)  # Final shape: [N, num_classes]

            # Get the final prediction and update accuracy
            preds = final_outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += batch_size

            # Update progress bar with live accuracy
            current_acc = test_correct / test_total if test_total > 0 else 0
            pbar.set_postfix(accuracy=f"{current_acc:.4f}")

    # --- 4. Display Results ---
    final_accuracy = test_correct / test_total if test_total > 0 else 0.0

    print("\n--- Test Complete ---")
    print(f"Checkpoint Path: {args.model_ckpt_path}")
    print(f"Total Videos / Correct Predictions: {test_total} / {test_correct}")
    print(f"Validation Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")

    return final_accuracy

if __name__ == '__main__':
    # --- Corrected Argument and Config Loading ---
    # 1. Create a parser instance
    args_parser = parser.get_arguments()

    # 2. Manually set the config file path. This makes the script dedicated to this test.
    args_parser.config = './configs/ucf_har_config.yaml'
    print(f"Loading configuration from: {args_parser.config}")

    # 3. Load settings from the YAML file
    with open(args_parser.config, 'r') as f:
        config = yaml.safe_load(f)

    # 4. Update the args object with the config values. This will correctly
    #    set `args.dataset` to 'ucf101' before it's used.
    for key, value in config.items():
        setattr(args_parser, key, value)

    # 5. Run the main testing function
    main(args_parser)