import os
import torch
import torch.nn as nn

class ResNetLinear(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, freeze_backbone: bool = True, pretrained_weights_path: str = None):
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
        
        # freeze the backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # self.backbone.eval()

        # unfreeze the last layer
        for p in self.backbone.fc.parameters():
            p.requires_grad = True
        
        # replace the last layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        outputs = self.backbone(x)
        return outputs