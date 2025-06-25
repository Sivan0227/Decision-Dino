import json
import torch
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from parameter import IGN_LEN, TEACHER_TOKEN_LIMIT, VAL_IDS

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
sequences_finetune_2 = []
print(len(data), "条数据")
for i, sample in tqdm(enumerate(data), desc="Processing samples", total=len(data)):
    pid = sample['person_id']
    exp_id = sample['exp_type']
    ts = sample['ts']
    dis_seq = sample['dis_state_seq']
    v_seq = sample['v_state_seq']
    d_seq = sample['decision_seq']
    a_seq = sample['action_seq']

    min_len = min(len(dis_seq), len(v_seq), len(d_seq), len(a_seq))
    # if min_len < IGN_LEN:
    #     # print(len(dis_seq), f"跳过 {pid} 的数据，长度不足{IGN_LEN}")
    #     continue

    merged_seq = []
    for i in range(min(min_len, int(TEACHER_TOKEN_LIMIT / 3))):
        merged_seq.append({"type": "A", "value": a_seq[i]})
        merged_seq.append({"type": "S", "value": {"dis": dis_seq[i], "v": v_seq[i]}})
        merged_seq.append({"type": "D", "value": d_seq[i]})

    sample_dict = {
    "person_id": pid,
    "exp_id": exp_id,
    "ts": ts,
    "seq": merged_seq
    }

    if min_len < IGN_LEN:
        sequences_finetune_2.append(sample_dict)
    else:
        sequences.append(sample_dict)

# ==== 划分数据集 ====
val_set = [s for s in sequences if s['person_id'] in VAL_IDS]
trainable_set = [s for s in sequences if s['person_id'] not in VAL_IDS]
random.shuffle(trainable_set)
pretrain_set, finetune_set = train_test_split(trainable_set, test_size=0.2, random_state=42)
finetune_set = finetune_set + sequences_finetune_2  # 添加 finetune_2 数据
# ==== 保存为 .pt 文件 ====
torch.save(pretrain_set, OUTPUT_DIR / "pretrain.pt")
torch.save(finetune_set, OUTPUT_DIR / "finetune.pt")
torch.save(val_set, OUTPUT_DIR / "val.pt")

print("✅ 数据处理完成")
print(f"预训练集样本数: {len(pretrain_set)}")
print(f"samples in pretrain_set: {pretrain_set[:2]}")  # 打印前两个样本以验证
print(f"微调集样本数: {len(finetune_set)}")
print(f"验证集样本数: {len(val_set)}")
