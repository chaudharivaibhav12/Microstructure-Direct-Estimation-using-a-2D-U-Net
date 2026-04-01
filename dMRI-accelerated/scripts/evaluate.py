import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(__file__))

from unet import UNet
from dataset import dMRIDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
sparse_data = np.load("data/processed/sparse_data.npy")
FA          = np.load("data/processed/FA_ground_truth.npy")
MD          = np.load("data/processed/MD_ground_truth.npy")
mask        = np.load("data/processed/mask.npy")

# ── Load model ─────────────────────────────────────────────────────────────
model = UNet(in_channels=22, out_channels=2).to(DEVICE)
model.load_state_dict(torch.load("models/best_unet.pth", map_location=DEVICE))
model.eval()
print("Model loaded!")

# ── Run inference on all slices ────────────────────────────────────────────
dataset = dMRIDataset(sparse_data, FA, MD, mask)
loader  = DataLoader(dataset, batch_size=4, shuffle=False)

FA_pred_all = []
MD_pred_all = []
FA_true_all = []
MD_true_all = []

with torch.no_grad():
    for x, target, mask_batch in loader:
        x    = x.to(DEVICE)
        pred = model(x).cpu().numpy()
        FA_pred_all.append(pred[:, 0, :, :])
        MD_pred_all.append(pred[:, 1, :, :])
        FA_true_all.append(target[:, 0, :, :].numpy())
        MD_true_all.append(target[:, 1, :, :].numpy())

FA_pred = np.concatenate(FA_pred_all, axis=0)   # (76, 81, 106)
MD_pred = np.concatenate(MD_pred_all, axis=0)
FA_true = np.concatenate(FA_true_all, axis=0)
MD_true = np.concatenate(MD_true_all, axis=0)

# ── Compute metrics ────────────────────────────────────────────────────────
mask_flat = mask.transpose(2, 0, 1).astype(bool)  # (76, 81, 106)

fa_mse  = np.mean((FA_pred[mask_flat] - FA_true[mask_flat])**2)
md_mse  = np.mean((MD_pred[mask_flat] - MD_true[mask_flat])**2)
fa_mae  = np.mean(np.abs(FA_pred[mask_flat] - FA_true[mask_flat]))
md_mae  = np.mean(np.abs(MD_pred[mask_flat] - MD_true[mask_flat]))

# Correlation
fa_corr = np.corrcoef(FA_pred[mask_flat], FA_true[mask_flat])[0,1]
md_corr = np.corrcoef(MD_pred[mask_flat], MD_true[mask_flat])[0,1]

print("\n── Metrics (brain voxels only) ───────────────────")
print(f"FA  MSE:  {fa_mse:.6f}")
print(f"FA  MAE:  {fa_mae:.6f}")
print(f"FA  Corr: {fa_corr:.4f}  (1.0 = perfect)")
print(f"MD  MSE:  {md_mse:.6f}")
print(f"MD  MAE:  {md_mae:.6f}")
print(f"MD  Corr: {md_corr:.4f}  (1.0 = perfect)")

# ── Visualize best slice ───────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

# Pick the middle slice
slice_idx = 38

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("U-Net: 12 directions → FA & MD maps\nvs Ground Truth (160 directions)",
             fontsize=14, fontweight='bold')

# FA maps
axes[0,0].imshow(FA_true[slice_idx].T, cmap='hot', origin='lower', vmin=0, vmax=1)
axes[0,0].set_title("FA — Ground Truth\n(160 directions)", fontsize=11)
axes[0,0].axis('off')

axes[0,1].imshow(FA_pred[slice_idx].T, cmap='hot', origin='lower', vmin=0, vmax=1)
axes[0,1].set_title("FA — U-Net Prediction\n(12 directions input)", fontsize=11)
axes[0,1].axis('off')

diff_fa = np.abs(FA_pred[slice_idx] - FA_true[slice_idx])
im = axes[0,2].imshow(diff_fa.T, cmap='hot', origin='lower', vmin=0, vmax=0.2)
axes[0,2].set_title(f"FA — Absolute Error\nMAE={fa_mae:.4f}", fontsize=11)
axes[0,2].axis('off')
plt.colorbar(im, ax=axes[0,2], fraction=0.046)

# MD maps
axes[1,0].imshow(MD_true[slice_idx].T, cmap='inferno', origin='lower', vmin=0, vmax=1)
axes[1,0].set_title("MD — Ground Truth\n(160 directions)", fontsize=11)
axes[1,0].axis('off')

axes[1,1].imshow(MD_pred[slice_idx].T, cmap='inferno', origin='lower', vmin=0, vmax=1)
axes[1,1].set_title("MD — U-Net Prediction\n(12 directions input)", fontsize=11)
axes[1,1].axis('off')

diff_md = np.abs(MD_pred[slice_idx] - MD_true[slice_idx])
im2 = axes[1,2].imshow(diff_md.T, cmap='inferno', origin='lower', vmin=0, vmax=0.2)
axes[1,2].set_title(f"MD — Absolute Error\nMAE={md_mae:.4f}", fontsize=11)
axes[1,2].axis('off')
plt.colorbar(im2, ax=axes[1,2], fraction=0.046)

plt.tight_layout()
plt.savefig("results/fa_md_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Figure saved to results/fa_md_comparison.png")

# ── Loss curve ────────────────────────────────────────────────────────────
history = np.load("models/history.npy", allow_pickle=True).item()
plt.figure(figsize=(8, 4))
plt.plot(history["train"], label="Train Loss", linewidth=2)
plt.plot(history["val"],   label="Val Loss",   linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=150)
plt.show()
print("✅ Loss curve saved to results/loss_curve.png")