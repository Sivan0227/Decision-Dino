import torch
import torch.nn as nn
import math

class ASDTransformer(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 num_heads=8, 
                 depth=6, 
                 mlp_ratio=4.0, 
                 dropout=0.1,
                 num_decision_classes=4,
                 max_seq_len=150,
                 mode='pretrain'  # 'pretrain' or 'finetune'
                ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mode = mode

        # === Embedding layers ===
        self.action_embed = nn.Linear(1, embed_dim)
        self.dis_embed = nn.Linear(10, embed_dim)
        self.v_embed = nn.Linear(3, embed_dim)
        self.decision_embed = nn.Embedding(num_embeddings=num_decision_classes, embedding_dim=embed_dim)

        # === Positional embedding ===
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))  # +1 for cls_token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # === CLS token ===
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # === Transformer encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # === Output heads ===
        self.decision_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_decision_classes)  # 分类
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)  # 回归
        )

    def forward(self, input_tokens,mask=None):
        """
        input_tokens: [B, T] of list of dicts with keys: type, value
        """
        B, T = len(input_tokens), len(input_tokens[0])
        device = self.cls_token.device

        # === Token embedding ===
        embeds = []
        for batch in input_tokens:
            embed_seq = []
            for token in batch:
                if token['type'][0] == 'A':
                    val = torch.tensor(token['value'], dtype=torch.float32, device=device).view(1, 1)
                    embed = self.action_embed(val)
                elif token['type'][0] == 'S':
                    dis = torch.tensor(token['value']['dis'], dtype=torch.float32, device=device).view(1, -1)
                    v = torch.tensor(token['value']['v'], dtype=torch.float32, device=device).view(1, -1)
                    embed = self.dis_embed(dis) + self.v_embed(v)
                elif token['type'][0] == 'D':
                    val = torch.tensor(token['value'], dtype=torch.long, device=device).view(1)
                    embed = self.decision_embed(val)
                else:  # PAD or unknown
                    embed = torch.zeros(1, self.embed_dim, device=device)
                embed_seq.append(embed)
            embed_seq = torch.cat(embed_seq, dim=0)  # [T, D]
            embeds.append(embed_seq)

        x = torch.stack(embeds, dim=0)  # [B, T, D]

        # === Positional embedding ===
        x = x + self.pos_embed[:, :x.size(1), :]

        # === CLS token prepend ===
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, D]

        if mask is not None:
            # mask: batch x seq_len → 加一列 False 给 cls
            cls_mask = torch.zeros(mask.size(0), 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat((cls_mask, mask == 0), dim=1)  # True 表示 padding
        else:
            full_mask = None

        # x shape: (B, S, D)
        x = x.transpose(0, 1)  # → (S, B, D)

        x = self.transformer(x, src_key_padding_mask=full_mask)

        x = x.transpose(0, 1)  # → 回到 (B, S, D)


        # === Output ===
        cls_output = x[:, 0, :]  # [B, D]


        if self.mode == "pretrain":
            return cls_output

        else:  # "finetune"
            decision_logits = self.decision_head(cls_output)
            action_pred = self.action_head(cls_output)
            return {
                "decision_logits": decision_logits,
                "action_pred": action_pred
            }
