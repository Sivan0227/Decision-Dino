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
        self.decision_head_combined = nn.Sequential(
            nn.LayerNorm(embed_dim*3),
            nn.Linear(embed_dim*3, num_decision_classes)  # 分类
        )

        self.action_head_combined = nn.Sequential(
            nn.LayerNorm(embed_dim*3),
            nn.Linear(embed_dim*3, 1)  # 回归
        )

    def forward(self, a_tensor,s_tensor, d_tensor, a_idx, s_idx, d_idx, mask, finetune_use_combined=False):
        """
        Args:
            a_tensor: (total_A, 1)
            s_tensor: (total_S, 13)
            d_tensor: (total_D, 1)
            a_idx: (total_A, 2) [batch_idx, token_idx]
            s_idx: (total_S, 2)
            d_idx: (total_D, 2)
            mask: (B, T) bool
            finetune_use_combined: bool

        Returns:
            output: (B, D) if not finetune_use_combined else (B, T+1, D)
        """
        B, T = mask.shape
        device = a_tensor.device

        x = torch.zeros(B, T, self.embed_dim, device=device)

        if a_idx.numel() > 0:
            A_embed = self.action_embed(a_tensor)
            _, T_a, _ = A_embed.shape
            batch_idx = torch.arange(B, device=A_embed.device).view(B, 1).expand(B, T_a)  # [32, 30]

            # 放入 x
            x[batch_idx, a_idx] = A_embed

        if s_idx.numel() > 0:
            S_dis = s_tensor[:, :, :10]
            S_v = s_tensor[:, :, 10:]
            S_embed = self.dis_embed(S_dis) + self.v_embed(S_v)
            _, T_s, _ = S_embed.shape
            batch_idx = torch.arange(B, device=S_embed.device).view(B, 1).expand(B, T_s)  # [32, 30]
            # 放入 x
            x[batch_idx, s_idx] = S_embed


        if d_idx.numel() > 0:
            D_embed = self.decision_embed(d_tensor)
            _, T_d, _ = D_embed.shape
            batch_idx = torch.arange(B, device=D_embed.device).view(B, 1).expand(B, T_d)  # [32, 30]
            # 放入 x
            x[batch_idx, d_idx] = D_embed

        # 拼接 cls token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)

        # 拼接 mask
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat((cls_mask, ~mask), dim=1)

        # 编码器
        x = x.transpose(0, 1)  # (T+1, B, D)
        x = self.transformer(x, src_key_padding_mask=full_mask)
        x = x.transpose(0, 1)  # (B, T+1, D)

        # === Output ===
        cls_output = x[:, 0, :]  # [B, T+1]
        last_a = x[:, -2, :]  # (B, D)
        last_s = x[:, -1, :]  # (B, D)


        if self.mode == "pretrain":
            return cls_output

        else:  # "finetune"
            if finetune_use_combined:
                """策略1"""
                new_output = torch.cat([cls_output, last_a, last_s], dim=-1)
                decision_logits = self.decision_head_combined(new_output)
                action_pred = self.action_head_combined(new_output)
                return {
                    "decision_logits": decision_logits,
                    "action_pred": action_pred
                }
            else:
                """策略2"""
                decision_logits = self.decision_head(cls_output)
                action_pred = self.action_head(cls_output)
                return decision_logits, action_pred
