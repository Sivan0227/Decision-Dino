import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

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

    # 数据加载
    train_dataset = DinoSequenceDataset(args.train_data_path)
    val_dataset = DinoSequenceDataset(args.val_data_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=my_collate_fn)

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
        writer.writerow(["epoch", "loss", "decision_loss", "action_loss", "accuracy", "mae", "precision", "recall", "f1"])

    history = {"train": {"loss":[], "d_loss":[], "a_loss":[]}, "val": {"loss":[], "d_loss":[], "a_loss":[], "acc":[], "mae":[]}}


    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    start_time = time.time()
    for epoch in range(args.epochs):
        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            student.train() if phase == "train" else student.eval()

            total_loss = total_d_loss = total_a_loss = 0.0
            total_correct = total_samples = total_mae = 0.0
            all_targets, all_preds = [], []

            pbar = tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                student_seq = batch['student_seq']
                student_mask = torch.stack(batch['student_mask']).to(device).bool()
                student_seq = [[{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in token.items()} for token in sample] for sample in student_seq]
                target_d = torch.tensor(batch['target_d'], dtype=torch.long).to(device)
                target_a = torch.tensor(batch['target_a'], dtype=torch.float32).to(device).view(-1,1)

                with torch.set_grad_enabled(phase == "train"):
                    if scaler and phase == "train":
                        with autocast():
                            d_out, a_out = student(student_seq, mask=student_mask)
                            d_loss = ce_loss_fn(d_out, target_d)
                            a_loss = mse_loss_fn(a_out, target_a)
                            loss = d_loss + a_loss
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        d_out, a_out = student(student_seq, mask=student_mask)
                        d_loss = ce_loss_fn(d_out, target_d)
                        a_loss = mse_loss_fn(a_out, target_a)
                        loss = d_loss + a_loss
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    if phase == "train":
                        optimizer.zero_grad()

                total_loss += loss.item()
                total_d_loss += d_loss.item()
                total_a_loss += a_loss.item()
                pred_class = torch.argmax(d_out, dim=1)
                correct = (pred_class == target_d).sum().item()
                total_correct += correct
                total_samples += target_d.size(0)
                mae = torch.mean(torch.abs(a_out.squeeze() - target_a.squeeze())).item()
                total_mae += mae * target_d.size(0)
                all_targets.extend(target_d.cpu().numpy())
                all_preds.extend(pred_class.cpu().numpy())
                pbar.set_postfix(loss=loss.item(), acc=correct/target_d.size(0), mae=mae)

            avg_loss = total_loss / len(loader)
            avg_d_loss = total_d_loss / len(loader)
            avg_a_loss = total_a_loss / len(loader)
            history[phase]["loss"].append(avg_loss)
            history[phase]["d_loss"].append(avg_d_loss)
            history[phase]["a_loss"].append(avg_a_loss)

            if phase == "val":
                avg_acc = total_correct / total_samples
                avg_mae = total_mae / total_samples
                history[phase]["acc"].append(avg_acc)
                history[phase]["mae"].append(avg_mae)
                cm = confusion_matrix(all_targets, all_preds)
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Confusion Matrix")
                plt.savefig(figures_dir / f"confusion_matrix_epoch{epoch+1}.png")
                plt.close()
                ckpt_path = weights_dir / f"student_finetune_epoch{epoch+1}.pth"
                torch.save(student.state_dict(), ckpt_path)

            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro")
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, phase, avg_loss, avg_d_loss, avg_a_loss, avg_acc if phase=="val" else "", avg_mae if phase=="val" else "", precision, recall, f1])

        # 主 loss 曲线
        plt.figure()
        plt.plot(history['train']['loss'], label='Train Loss')
        plt.plot(history['val']['loss'], label='Val Loss')
        plt.plot(history['train']['d_loss'], '--', label='Train D Loss')
        plt.plot(history['val']['d_loss'], '--', label='Val D Loss')
        plt.plot(history['train']['a_loss'], ':', label='Train A Loss')
        plt.plot(history['val']['a_loss'], ':', label='Val A Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title("Train-Val Loss Curve")
        plt.savefig(figures_dir / f"loss_curve_epoch{epoch+1}.png")
        plt.close()

        # 两两画图
        def save_pair_plot(train_list, val_list, name):
            plt.figure()
            plt.plot(train_list, label=f'Train {name}')
            plt.plot(val_list, label=f'Val {name}')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.legend()
            plt.grid(True)
            plt.title(f'Train-Val {name} Curve')
            plt.savefig(figures_dir / f"{name.lower()}_curve_epoch{epoch+1}.png")
            plt.close()
        save_pair_plot(history['train']['loss'], history['val']['loss'], 'Loss')
        save_pair_plot(history['train']['d_loss'], history['val']['d_loss'], 'Decision Loss')
        save_pair_plot(history['train']['a_loss'], history['val']['a_loss'], 'Action Loss')

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\nFinetuning complete in: {total_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to finetune train dataset')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to finetune validation dataset')
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
    args.train_data_path = "../dino_data/dino_sequence_data/finetune_train.pt"
    args.val_data_path = "../dino_data/dino_sequence_data/finetune_val.pt"
    args.pretrained_weights = f"../dino_data/output_dino/{'pretrain_'}/weights/student_pretrain_epoch{50}.pth"
    args.output_dir = "../dino_data/output_dino"
    train_finetune(args)
