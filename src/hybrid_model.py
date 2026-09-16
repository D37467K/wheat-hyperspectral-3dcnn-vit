import torch.nn as nn

from model_3dcnn import SpectralCNN
from vision_transformer import PretrainedViT
from feature_fusion import MetadataHead, FeatureFusionClassifier

class HybridSpectralCNNViT(nn.Module):
    """
    Hybrid model used by the notebook:
    hyperspectral cube -> 3D-CNN -> 3-channel representation ->
    pretrained ViT, with metadata fused before final classification.
    """

    def __init__(self, fine_tune_blocks=1):
        super().__init__()

        self.cnn_branch = SpectralCNN()
        self.vit_branch = PretrainedViT(fine_tune_blocks=fine_tune_blocks)
        self.meta_head = MetadataHead()

        self.fusion_classifier = FeatureFusionClassifier(
            vit_dim=768,
            meta_dim=16,
            num_classes=4,
        )

    def forward(self, image, meta):
        cnn_features = self.cnn_branch(image)
        vit_features = self.vit_branch(cnn_features)
        meta_features = self.meta_head(meta)

        return self.fusion_classifier(
            vit_features,
            meta_features,
        )
