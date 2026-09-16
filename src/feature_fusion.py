import torch
import torch.nn as nn

class MetadataHead(nn.Module):
    """Four tabular variables mapped to a 16-dimensional representation."""

    def __init__(self):
        super().__init__()
        self.meta_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
        )

    def forward(self, meta):
        return self.meta_head(meta)

class FeatureFusionClassifier(nn.Module):
    """Concatenate ViT features and metadata features before classification."""

    def __init__(self, vit_dim=768, meta_dim=16, num_classes=4):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(vit_dim + meta_dim, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, vit_features, meta_features):
        fused = torch.cat([vit_features, meta_features], dim=1)
        return self.classifier(fused)
