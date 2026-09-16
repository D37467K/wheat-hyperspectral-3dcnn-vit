import numpy as np
import torch
from pathlib import Path
from torch import nn
from tifffile import imread

IMAGE_SIZE = 96

def image_lookup(images_dir):
    result = {}
    for path in Path(images_dir).glob("*.tiff"):
        try:
            result[int(path.stem)] = path
        except ValueError:
            pass
    return result

def preprocess_cube(path, image_size=IMAGE_SIZE):
    cube = imread(path).astype(np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3-D hyperspectral cube, got {cube.shape}: {path}")

    low = np.percentile(cube, 1, axis=(0, 1), keepdims=True)
    high = np.percentile(cube, 99, axis=(0, 1), keepdims=True)
    cube = np.clip((cube - low) / (high - low + 1e-6), 0, 1)

    bands = torch.from_numpy(cube).permute(2, 0, 1).unsqueeze(0)
    bands = nn.functional.interpolate(
        bands,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return bands.squeeze(0).permute(1, 2, 0).numpy().astype(np.float16)
