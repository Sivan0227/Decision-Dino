import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from dino_sequence_dataset import DinoSequenceDataset  
from sequence_transformer import ASDTransformer  
from dino_finetune_main import my_collate_fn 

def test_model(model, test_loader, device, criterion_decision, criterion_action):
    model.eval()
    test_loss_total = 0
    test_decision_correct = 0
    test_decision_total = 0
    test_action_error = 0
    all_decisions = []
    all_decision_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            student_seq = batch['student_seq']
            student_mask = torch.stack(batch['student_mask']).to(device).bool()

            student_seq = [
                [
                    {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in token.items()}
                    for token in sample
                ]
                for sample in student_seq
            ]

            decision_logits, action_pred = model(student_seq, mask=student_mask)
            d_loss = criterion_decision(decision_logits, batch['target_d'].to(device))
            a_loss = criterion_action(action_pred.squeeze(), batch['target_a'].to(device).float())
            loss = d_loss + a_loss

            test_loss_total += loss.item()

            _, decision_pred = torch.max(decision_logits, 1)
            test_decision_correct += (decision_pred == batch['target_d'].to(device)).sum().item()
            test_decision_total += batch['target_d'].size(0)

            test_action_error += torch.abs(action_pred.squeeze() - batch['target_a'].to(device).float()).sum().item()

            all_decisions.extend(batch['target_d'].cpu().numpy())
            all_decision_preds.extend(decision_pred.cpu().numpy())

    avg_loss = test_loss_total / len(test_loader)
    acc = test_decision_correct / test_decision_total
    mae = test_action_error / test_decision_total

    print(f"Test Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}, Test MAE: {mae:.4f}")
    return avg_loss, acc, mae, all_decisions, all_decision_preds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASDTransformer(mode="finetune").to(device)
    model.load_state_dict(torch.load(args.weight_path, map_location=device))

    test_dataset = DinoSequenceDataset(args.test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=my_collate_fn)

    criterion_decision = nn.CrossEntropyLoss()
    criterion_action = nn.L1Loss()

    test_loss, test_acc, test_mae, y_true, y_pred = test_model(model, test_loader, device, criterion_decision, criterion_action)

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / f"figures_{time_tag}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame({
        'test_loss': [test_loss],
        'test_acc': [test_acc],
        'test_mae': [test_mae]
    })
    metrics_df.to_csv(output_dir / f"test_metrics_{time_tag}.csv", index=False)

    plt.figure()
    plt.bar(['Loss', 'Acc', 'MAE'], [test_loss, test_acc, test_mae])
    plt.title("Final Test Metrics")
    plt.savefig(figures_dir / "test_bar_metrics.png")
    plt.close()

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Test Confusion Matrix")
    plt.savefig(figures_dir / "test_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--weight_path', type=str, required=True, help='Path to finetuned model weights')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    main(args)
