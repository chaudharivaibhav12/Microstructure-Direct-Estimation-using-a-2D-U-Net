import numpy as np
import torch
from torch.utils.data import Dataset

class dMRIDataset(Dataset):
    def __init__(self, sparse_data, FA, MD, mask, augment=False):
        self.sparse_data = sparse_data  # (81, 106, 76, 22)
        self.FA          = FA           # (81, 106, 76)
        self.MD          = MD           # (81, 106, 76)
        self.mask        = mask         # (81, 106, 76)
        self.augment     = augment
        self.n_slices    = sparse_data.shape[2]

        # Compute global stats for normalization (only brain voxels)
        brain_data = sparse_data[mask > 0]
        self.global_mean = brain_data.mean(axis=0).astype(np.float32)
        self.global_std  = (brain_data.std(axis=0) + 1e-8).astype(np.float32)

    def __len__(self):
        return self.n_slices

    def __getitem__(self, idx):
        x    = self.sparse_data[:, :, idx, :].astype(np.float32)  # (81,106,22)
        fa   = self.FA[:, :, idx].astype(np.float32)
        md   = self.MD[:, :, idx].astype(np.float32)
        mask = self.mask[:, :, idx].astype(np.float32)

        # Global normalization (more stable than per-slice)
        x = (x - self.global_mean) / self.global_std.astype(np.float32)

        # Clip targets
        fa = np.clip(fa, 0, 1)
        md = np.clip(md / 0.005, 0, 1)

        # Augmentation — random horizontal/vertical flip
        if self.augment and np.random.rand() > 0.5:
            x    = np.flip(x,    axis=0).copy()
            fa   = np.flip(fa,   axis=0).copy()
            md   = np.flip(md,   axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if self.augment and np.random.rand() > 0.5:
            x    = np.flip(x,    axis=1).copy()
            fa   = np.flip(fa,   axis=1).copy()
            md   = np.flip(md,   axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # To tensors
        x      = torch.tensor(x).permute(2, 0, 1)          # (22, 81, 106)
        target = torch.stack([torch.tensor(fa),
                               torch.tensor(md)], dim=0)    # (2, 81, 106)
        mask   = torch.tensor(mask)                         # (81, 106)

        return x, target, mask