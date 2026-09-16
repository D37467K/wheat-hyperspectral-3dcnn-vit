import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class PretrainedViT(nn.Module):
    """ImageNet-pretrained ViT-B/16 feature extractor."""

    def __init__(self, fine_tune_blocks=1):
        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()

        for param in self.vit.parameters():
            param.requires_grad = False

        for block in list(self.vit.encoder.layers.children())[-fine_tune_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        for param in self.vit.encoder.ln.parameters():
            param.requires_grad = True

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        x = torch.sigmoid(x)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.vit(x)
