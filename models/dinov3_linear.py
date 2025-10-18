import torch
import torch.nn as nn
from transformers import AutoModel

class AttentionAggregator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, tokens):
        # tokens: [batch_size, seq_len, hidden_size]
        attention_weights = torch.softmax(self.attention(tokens), dim=1)  # [batch_size, seq_len, 1]
        weighted_tokens = tokens * attention_weights  # [batch_size, seq_len, hidden_size]
        aggregated = weighted_tokens.sum(dim=1)  # [batch_size, hidden_size]
        return aggregated



class DinoV3Linear(nn.Module):
    def __init__(self, backbone: AutoModel, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("backbone.config must have a 'hidden_size' attribute")
        
        self.aggregator = AttentionAggregator(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes)
        )
        self.head_flatten = nn.Sequential(
            nn.Linear(201 * 384, 1024),  # 展平所有tokens
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state # [B, 201, 384]

        # 1.只使用 [CLS] token
        cls = last_hidden[:, 0, :] # [B, 384]
        logits = self.head(cls)

        # # 2.使用 [CLS] token 和 4个 register tokens
        # selected_tokens = last_hidden[:, :5, :]  # [B, 5, 384]
        # aggregated = selected_tokens.mean(dim=1)  # [B, 384]
        # logits = self.head(aggregated)

        # # 3.使用 注意力机制聚合
        # aggregated = self.aggregator(last_hidden)  # [batch_size, 384]
        # logits = self.head(aggregated)

        # # 4.分类头扩展（多层感知机处理所有tokens）
        # flattened = last_hidden.view(last_hidden.size(0), -1)  # [batch_size, 201*384]
        # logits = self.head_flatten(flattened)

        return logits