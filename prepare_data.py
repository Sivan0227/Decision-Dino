import json
import torch
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==== 可调整参数 ====
EMBED_DIM = 256
STUDENT_TOKEN_LIMIT = 30 * 3
TEACHER_TOKEN_LIMIT = 50 * 3
IGN_LEN = int(TEACHER_TOKEN_LIMIT / 3 - STUDENT_TOKEN_LIMIT / 3)
VAL_IDS = {'1', '25', '29', '32'}

# ==== 输入输出路径 ====
INPUT_PATH = Path("../processed_data/paper2/transformer_input.jsonl")
OUTPUT_DIR = Path("../dino_data/dino_sequence_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== 读取 JSONL 数据 ====
data = []
with INPUT_PATH.open("r") as f:
    for line in f:
        sample = json.loads(line)
        data.append(sample)

# ==== 构造序列数据（不 padding，延迟到 Dataset 内部处理）====
sequences = []
print(len(data), "条数据")
for sample in data:
    pid = sample['person_id']
    exp_id = sample['exp_type']
    ts_seq = [i for i in range(len(sample['ts_seq']))]  # 将时间戳序列从 0 开始调整为从 1 开始
    dis_seq = sample['dis_state_seq']
    v_seq = sample['v_state_seq']
    d_seq = sample['decision_seq']
    a_seq = sample['action_seq']

    min_len = min(len(dis_seq), len(v_seq), len(d_seq), len(a_seq))
    if min_len < IGN_LEN:
        # print(len(dis_seq), f"跳过 {pid} 的数据，长度不足{IGN_LEN}")
        continue

    merged_seq = []
    for i in range(min(min_len, int(TEACHER_TOKEN_LIMIT / 3))):
        merged_seq.append({"type": "A", "value": a_seq[i]})
        merged_seq.append({"type": "S", "value": {"dis": dis_seq[i], "v": v_seq[i]}})
        merged_seq.append({"type": "D", "value": d_seq[i]})

    sequences.append({
        "person_id": pid,
        "exp_id": exp_id,
        "ts_seq": ts_seq[:len(merged_seq)],  # 确保时间戳序列与 merged_seq 长度一致
        "seq": merged_seq
    })

# ==== 划分数据集 ====
val_set = [s for s in sequences if s['person_id'] in VAL_IDS]
trainable_set = [s for s in sequences if s['person_id'] not in VAL_IDS]
random.shuffle(trainable_set)
pretrain_set, finetune_set = train_test_split(trainable_set, test_size=0.2, random_state=42)

# ==== 保存为 .pt 文件 ====
torch.save(pretrain_set, OUTPUT_DIR / "pretrain.pt")
torch.save(finetune_set, OUTPUT_DIR / "finetune.pt")
torch.save(val_set, OUTPUT_DIR / "val.pt")

print("✅ 数据处理完成")
print(f"预训练集样本数: {len(pretrain_set)}")
print(f"微调集样本数: {len(finetune_set)}")
print(f"验证集样本数: {len(val_set)}")
