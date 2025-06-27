import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import time
import datetime
import matplotlib.pyplot as plt
import csv
import argparse
from tqdm import tqdm

from sequence_transformer import ASDTransformer
from dino_sequence_dataset import DinoSequenceDataset
from utils import clip_gradients

def my_collate_fn(batch):
    # 可以按你已有逻辑写，我先写个简单示例
    return {
        "student_seq": [item["student_seq"] for item in batch],
        "student_mask": [item["student_mask"] for item in batch],
        "target_d": [item["target_d"] for item in batch],
        "target_a": [item["target_a"] for item in batch],
    }

def train_finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DinoSequenceDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=my_collate_fn)

    student = ASDTransformer(mode="finetune").to(device)
    if args.pretrained_weights:
        state_dict = torch.load(args.pretrained_weights, map_location=device)
        student.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained weights from {args.pretrained_weights}")

    # === 模式：linear probing / full finetune ===
    if args.train_mode == "linear":
        print("🔒 Linear Probing: Only training prediction heads...")
        for name, param in student.named_parameters():
            if "decision_head" in name or "action_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.train_mode == "finetune":
        print("🔓 Full Fine-tuning: Training entire model...")
        for param in student.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown train_mode: {args.train_mode}")

    params_groups = [
        {"params": [p for n, p in student.named_parameters() if p.requires_grad and ("bias" not in n) and ("norm" not in n)], "weight_decay": 0.05},
        {"params": [p for n, p in student.named_parameters() if p.requires_grad and ("bias" in n or "norm" in n)], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(params_groups, lr=args.lr)
    scaler = GradScaler() if args.use_fp16 else None

    # === 路径和日志 ===
    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    finetune_dir = Path(args.output_dir) / f"finetune_{args.train_mode}_{time_tag}"
    weights_dir = finetune_dir / "weights"
    figures_dir = finetune_dir / "figures"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = finetune_dir / f"finetune_metrics.csv"

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "decision_loss", "action_loss"])

    loss_curve = []
    decision_loss_curve = []
    action_loss_curve = []

    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    start_time = time.time()
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        total_decision_loss = 0.0
        total_action_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            student_seq = batch['student_seq']
            student_mask = torch.stack(batch['student_mask']).to(device).bool()

            student_seq = [
                [
                    {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in token.items()}
                    for token in sample
                ]
                for sample in student_seq
            ]

            target_d = torch.tensor(batch['target_d'], dtype=torch.long).to(device)
            target_a = torch.tensor(batch['target_a'], dtype=torch.float32).to(device).view(-1, 1)

            optimizer.zero_grad()
            if scaler:
                with autocast():
                    d_out, a_out = student(student_seq, mask=student_mask)
                    d_loss = ce_loss_fn(d_out, target_d)
                    a_loss = mse_loss_fn(a_out, target_a)
                    loss = d_loss + a_loss
                scaler.scale(loss).backward()
                if args.clip_grad:
                    scaler.unscale_(optimizer)
                    clip_gradients(student, args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                d_out, a_out = student(student_seq, mask=student_mask)
                d_loss = ce_loss_fn(d_out, target_d)
                a_loss = mse_loss_fn(a_out, target_a)
                loss = d_loss + a_loss
                loss.backward()
                if args.clip_grad:
                    clip_gradients(student, args.clip_grad)
                optimizer.step()

            total_loss += loss.item()
            total_decision_loss += d_loss.item()
            total_action_loss += a_loss.item()
            pbar.set_postfix(loss=loss.item(), d_loss=d_loss.item(), a_loss=a_loss.item())

        avg_loss = total_loss / len(dataloader)
        avg_d_loss = total_decision_loss / len(dataloader)
        avg_a_loss = total_action_loss / len(dataloader)
        loss_curve.append(avg_loss)
        decision_loss_curve.append(avg_d_loss)
        action_loss_curve.append(avg_a_loss)

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, avg_d_loss, avg_a_loss])

        # 保存图像
        plt.figure()
        plt.plot(loss_curve, label="Total Loss")
        plt.plot(decision_loss_curve, label="Decision Loss")
        plt.plot(action_loss_curve, label="Action Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Finetuning Loss Curve up to Epoch {}".format(epoch + 1))
        plt.legend()
        plt.grid(True)
        plt.savefig(figures_dir / f"finetune_loss_curve_epoch{epoch+1}.png")
        plt.close()

        # 保存权重
        ckpt_path = weights_dir / f"student_finetune_epoch{epoch+1}.pth"
        torch.save(student.state_dict(), ckpt_path)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\nFinetuning complete in: {total_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to finetune .pt data file")
    parser.add_argument('--pretrained_weights', type=str, default=None, help="Path to pretrained weights")
    parser.add_argument('--output_dir', type=str, help="Directory to save outputs")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--train_mode', type=str, choices=['linear', 'finetune'], default='finetune',
                        help="Training mode: linear (linear probing) or finetune (full fine-tuning)")

    args = parser.parse_args()
    args.data_path = "../dino_data/dino_sequence_data/finetune.pt"
    args.output_dir = "../dino_data/output_dino"
    train_finetune(args)
