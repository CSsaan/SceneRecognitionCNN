import os
import torch
import torch.nn as nn
from transformers import AutoModel

class AttentionAggregator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, tokens):
        # tokens: [batch_size, seq_len, hidden_size]
        attention_weights = torch.softmax(self.attention(tokens), dim=1)  # [batch_size, seq_len, hidden_size]
        weighted_tokens = tokens * attention_weights  # [batch_size, seq_len, hidden_size]
        aggregated = weighted_tokens.sum(dim=1)  # [batch_size, hidden_size]
        return aggregated



class FastVitLinear(nn.Module):
    def __init__(self, backbone: AutoModel, num_classes: int, freeze_backbone: bool = True, pretrained_weights_path: str = None):
        super().__init__()
        self.backbone = backbone

        # load pretrained weights(if provided)
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            print(f"Loading pretrained weights from {pretrained_weights_path}")
            checkpoint = torch.load(pretrained_weights_path)
            # remove 'module.' (DataParallel)
            if 'state_dict' in checkpoint:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            else:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}   
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded successfully")
        elif pretrained_weights_path:
            raise FileNotFoundError(f"Pretrained weights file not found at {pretrained_weights_path}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

            # train last N players
            for p in self.backbone.head.parameters():
                p.requires_grad = True
            # for p in self.backbone.conv_exp.parameters():
            #     p.requires_grad = True
            # for layer in self.backbone.network[-1:]:
            #     for p in layer.parameters():
            #         p.requires_grad = True
        
        # replace the last layer
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        return outputs