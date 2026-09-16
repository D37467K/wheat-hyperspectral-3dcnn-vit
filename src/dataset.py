import numpy as np
import torch
from torch.utils.data import Dataset

META_COLUMNS = ["GrainWeight", "Gsw", "PhiPS2", "Fertilizer"]

class HyperLeafDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        image = (
            torch.from_numpy(np.load(row.cache_path).astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)

        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-2)

        meta = torch.tensor(
            row[META_COLUMNS].to_numpy(dtype=np.float32)
        )

        return (
            image,
            meta,
            torch.tensor(int(row.label), dtype=torch.long),
        )
