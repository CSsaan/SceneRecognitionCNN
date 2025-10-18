import torch
import torch.nn as nn
from transformers import AutoModel

from .swin_transformer import load_swin_pretrained

class SwinLinear(nn.Module):
    def __init__(self, backbone: AutoModel, num_classes: int, freeze_backbone: bool = True, pretrained_weights_path: str = None):
        super().__init__()

        self.backbone = backbone
        load_swin_pretrained(pretrained_weights_path, self.backbone)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # unfreeze the last layer
        for p in self.backbone.head.parameters():
            p.requires_grad = True

        # 解冻指定的后几层
        num_layers_to_freeze = 1
        for i in range(1, num_layers_to_freeze + 1):
            if i <= len(self.backbone.layers):
                layer = self.backbone.layers[-i]  # 从后往前获取层
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"解冻了第 {len(self.backbone.layers) - i + 1} 层")
        
        
        self.head = nn.Sequential(
            nn.Linear(num_classes, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes)
        )
        # self.head_flatten = nn.Sequential(
        #     nn.Linear(201 * 384, 1024),  # 展平所有tokens
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(512, num_classes)
        # )
    def forward(self, pixel_values):
        cls = self.backbone(pixel_values)

        # 1.只使用 [CLS] token
        logits = cls # self.head(cls)

        return logits
    