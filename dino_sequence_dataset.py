import torch
from torch.utils.data import Dataset
from parameter import TEACHER_TOKEN_LIMIT, STUDENT_TOKEN_LIMIT

class DinoSequenceDataset(Dataset):
    def __init__(self, data_path, mode='pretrain', 
                 max_teacher_tokens=TEACHER_TOKEN_LIMIT, 
                 max_student_tokens=STUDENT_TOKEN_LIMIT - 1):
        self.data = torch.load(data_path)
        assert mode in ['pretrain', 'finetune', 'val']
        self.mode = mode
        self.max_teacher_tokens = max_teacher_tokens
        self.max_student_tokens = max_student_tokens

    def __len__(self):
        return len(self.data)

    def pad_left(self, seq, target_len):
        pad_len = target_len - len(seq)
        # if pad_len > 0:
        #     print(f"[Padding] Added {pad_len} PAD tokens")
        pad_token = [{'type': 'A', 'value': 0.0},
                     {'type': 'S', 'value': {'dis': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'v': [0, 0, 0]}},
                     {'type': 'D', 'value': 0}]
        padded = pad_token * int(pad_len/3) + seq
        mask = [0] * pad_len + [1] * len(seq)
        return padded[-target_len:], mask[-target_len:]

    def __getitem__(self, idx):
        item = self.data[idx]
        # print(f"[Processing] Sample {idx}")
        token_seq = item['seq']
        person_id = item['person_id']
        exp_id = item['exp_id']
        ts = item['ts']

        if self.mode == 'finetune':
            max_tokens = self.max_student_tokens
        else:
            max_tokens = self.max_teacher_tokens

        raw_seq = token_seq[-max_tokens:]
        input_len = max_tokens

        padded_seq, mask = self.pad_left(raw_seq, input_len)

        # student 是 teacher 的前段子序列
        if self.mode == 'finetune':
            student_seq, student_mask = padded_seq, mask
            teacher_seq = teacher_mask = None
        else:
            student_seq, student_mask = padded_seq[:self.max_student_tokens], mask[:self.max_student_tokens]
            teacher_seq, teacher_mask = padded_seq, mask

        # 提取目标位置：student 最后位置的 D 和下一个的 A
        d_index = self.max_student_tokens
        a_index = d_index + 1
        full_seq = padded_seq if self.mode != 'finetune' else token_seq
        target_d = full_seq[d_index] if d_index < len(full_seq) and full_seq[d_index]['type'] == 'D' else None
        target_a = full_seq[a_index] if a_index < len(full_seq) and full_seq[a_index]['type'] == 'A' else None

        output = {
            "student_seq": student_seq,
            "student_mask": student_mask,
            "target_d": target_d,
            "target_a": target_a,
            "person_id": person_id,
            "exp_id": exp_id,
            "ts": ts
        }

        if self.mode == 'pretrain':
            output.update({
                "teacher_seq": teacher_seq,
                "teacher_mask": teacher_mask,
            })

        return output


if __name__ == "__main__":
    # 用法示例：
    # 加载 pretrain 数据

    pretrain_dataset = DinoSequenceDataset("../dino_data/dino_sequence_data/pretrain.pt", mode='pretrain')
    for i in range(100):
        sample = pretrain_dataset[i]
    sample = pretrain_dataset[7]
    print("--------------len",len(sample['teacher_seq']),len(sample['student_seq']))
    print("--------------teacher_seq:", sample['teacher_seq'])
    print("----------------student_seq:", sample['student_seq'])
    print("---------------target_d:", sample['target_d'])
    print("----------------target_a:", sample['target_a'])
    print("----------------person_id:", sample['person_id'])
    print("----------------exp_id:", sample['exp_id'])
    print("----------------ts:", sample['ts'])

    # 加载 finetune 数据

    # finetune_dataset = DinoSequenceDataset("../dino_data/dino_sequence_data/finetune.pt", mode='finetune')
    # for i in range(100):
    #     sample = finetune_dataset[i]
    # sample = finetune_dataset[51]
    # print("--------------len",len(sample['student_seq']))
    # print("----------------student_seq:", sample['student_seq'])
    # print("---------------target_d:", sample['target_d'])
    # print("----------------target_a:", sample['target_a'])
    # print("----------------person_id:", sample['person_id'])
    # print("----------------exp_id:", sample['exp_id'])
    # print("----------------ts:", sample['ts'])


