import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenType:
    ACTION = "A"
    STATE = "S"
    DECISION = "D"
    PAD = "PAD"


class ASDTransformer(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 num_layers=6, 
                 num_heads=8,
                 dropout=0.1, 
                 max_len=150,
                 num_decision_classes=4,
                 action_output_dim=1,
                 output_mode='dino'):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_len = max_len
        self.output_mode = output_mode

        # === Embedding ===
        self.action_embedding = nn.Embedding(100, embed_dim)   # 100 是假设动作种类数，可调整
        self.decision_embedding = nn.Embedding(num_decision_classes, embed_dim)
        self.dis_embedding = nn.Embedding(50, embed_dim)       # 距离状态类别数，可调整
        self.v_embedding = nn.Embedding(50, embed_dim)         # 速度状态类别数，可调整

        # === Positional Embedding ===
        self.pos_embedding = nn.Parameter(torch.randn(max_len, embed_dim))

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Heads ===
        if output_mode == 'finetune' or output_mode == 'both':
            self.decision_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_decision_classes)
            )
            self.action_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, action_output_dim)
            )

    def forward(self, tokens):
        # === Token Embedding ===
        embeddings = []
        for token in tokens:
            t_type = token["type"]
            val = token["value"]
            if t_type == TokenType.ACTION:
                embeddings.append(self.action_embedding(torch.tensor(val)))
            elif t_type == TokenType.DECISION:
                embeddings.append(self.decision_embedding(torch.tensor(val)))
            elif t_type == TokenType.STATE:
                dis = val["dis"]
                v = val["v"]
                dis_emb = self.dis_embedding(torch.tensor(dis))
                v_emb = self.v_embedding(torch.tensor(v))
                embeddings.append(dis_emb + v_emb)
            elif t_type == TokenType.PAD:
                embeddings.append(torch.zeros(self.embed_dim))

        x = torch.stack(embeddings).unsqueeze(0)  # shape: [1, seq_len, embed_dim]
        x = x + self.pos_embedding[:x.size(1)]
        x_encoded = self.encoder(x)

        # 取最后一个 D/A 位置的输出作为预测输入
        outputs = {"representation": x_encoded}  # 用于对比学习

        if self.output_mode in ['finetune', 'both']:
            d_feat = x_encoded[0, -2]  # 倒数第二位是 D
            a_feat = x_encoded[0, -1]  # 倒数第一位是 A
            outputs["decision_logits"] = self.decision_head(d_feat)
            outputs["action_pred"] = self.action_head(a_feat)

        return outputs
