import torch
from torch.utils.data import Dataset

class DinoSequenceDataset(Dataset):
    def __init__(self, data_path, max_teacher_tokens=150, max_student_tokens=89):
        self.data = torch.load(data_path)
        self.max_teacher_tokens = max_teacher_tokens
        self.max_student_tokens = max_student_tokens

    def __len__(self):
        return len(self.data)

    def pad_right(self, seq, target_len):
        pad_token = {"type": "PAD", "value": None}
        mask = [1] * len(seq) + [0] * (target_len - len(seq))
        padded = seq + [pad_token] * (target_len - len(seq))
        return padded[:target_len], mask[:target_len]

    def pad_left(self, seq, target_len):
        pad_token = {"type": "PAD", "value": None}
        pad_len = target_len - len(seq)
        mask = [0] * pad_len + [1] * len(seq)
        padded = [pad_token] * pad_len + seq
        return padded[-target_len:], mask[-target_len:]

    def __getitem__(self, idx):
        item = self.data[idx]
        print("item",item.keys())
        token_seq = item['seq']
        person_id = item['person_id']
        exp_id = item['exp_id']
        ts_seq = item['ts_seq']

        # teacher: left-aligned full sequence
        teacher_seq, teacher_mask = self.pad_left(token_seq, self.max_teacher_tokens)

        # student: right-aligned partial sequence (trailing part)
        student_raw = token_seq[-self.max_student_tokens:]
        student_seq, student_mask = self.pad_left(student_raw, self.max_student_tokens)

        # 预测目标：student 的下一个 D（第 90 个）和 A（第 91 个）
        d_index = -self.max_student_tokens + self.max_student_tokens  # = 0
        target_d = token_seq[d_index] if len(token_seq) > self.max_student_tokens and token_seq[d_index]['type'] == 'D' else None

        a_index = d_index + 1
        target_a = token_seq[a_index] if len(token_seq) > self.max_student_tokens + 1 and token_seq[a_index]['type'] == 'A' else None

        return {
            "student_seq": student_seq,
            "student_mask": student_mask,
            "teacher_seq": teacher_seq,
            "teacher_mask": teacher_mask,
            "target_d": target_d,
            "target_a": target_a,
            "person_id": person_id,
            "exp_id": exp_id,
            "ts_seq": ts_seq
        }

# 用法示例：
dataset = DinoSequenceDataset("../dino_data/dino_sequence_data/pretrain.pt")
sample = dataset[0]
print("len",len(sample['teacher_seq']),len(sample['student_seq']))
print("teacher_seq:", sample['teacher_seq'])
print("student_seq:", sample['student_seq'])
print("target_d:", sample['target_d'])
print("target_a:", sample['target_a'])
print("person_id:", sample['person_id'])
print("exp_id:", sample['exp_id'])
print("ts_seq:", sample['ts_seq'])

