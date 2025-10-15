import torch.nn as nn
from transformers import AutoModel

class DinoV3Linear(nn.Module):
    def __init__(self, backbone: AutoModel, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        
        hidden_size = getattr(backbone.config, "hidden_size", None)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state
        cls = last_hidden[:, 0]
        logits = self.head(cls)
        return logits