import torch.nn as nn

class SpectralCNN(nn.Module):
    """3D-CNN branch for local spectral-spatial feature extraction."""

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )

        self.spectral_to_rgb = nn.Conv3d(64, 3, kernel_size=1)

    def forward(self, image):
        x = self.cnn(image)
        x = self.spectral_to_rgb(x)
        x = x.mean(dim=2)
        return x
