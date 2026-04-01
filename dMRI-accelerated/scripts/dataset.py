import numpy as np
import torch
from torch.utils.data import Dataset

class dMRIDataset(Dataset):
    def __init__(self, sparse_data, FA, MD, mask):
        """
        sparse_data : (X, Y, Z, C)  - undersampled dMRI
        FA          : (X, Y, Z)     - ground truth FA
        MD          : (X, Y, Z)     - ground truth MD
        mask        : (X, Y, Z)     - brain mask
        """
        self.sparse_data = sparse_data  # (81, 106, 76, 22)
        self.FA = FA                    # (81, 106, 76)
        self.MD = MD                    # (81, 106, 76)
        self.mask = mask                # (81, 106, 76)
        self.n_slices = sparse_data.shape[2]  # 76 axial slices

    def __len__(self):
        return self.n_slices

    def __getitem__(self, idx):
        # Get one axial slice
        x = self.sparse_data[:, :, idx, :]     # (81, 106, 22)
        fa = self.FA[:, :, idx]                 # (81, 106)
        md = self.MD[:, :, idx]                 # (81, 106)
        mask = self.mask[:, :, idx]             # (81, 106)

        # Normalize input: per-channel zero mean, unit std
        x = x.astype(np.float32)
        for c in range(x.shape[-1]):
            ch = x[:, :, c]
            std = ch.std()
            if std > 0:
                x[:, :, c] = (ch - ch.mean()) / std

        # Normalize targets to [0, 1]
        fa = np.clip(fa.astype(np.float32), 0, 1)
        md = np.clip(md.astype(np.float32), 0, 0.005)
        md = md / 0.005  # scale to [0,1]

        # Convert to tensors — channels first for PyTorch
        x = torch.tensor(x).permute(2, 0, 1)          # (22, 81, 106)
        target = torch.stack([
            torch.tensor(fa),
            torch.tensor(md)
        ], dim=0)                                       # (2, 81, 106)
        mask = torch.tensor(mask.astype(np.float32))   # (81, 106)

        return x, target, mask