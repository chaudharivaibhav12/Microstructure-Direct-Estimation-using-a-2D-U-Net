import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu
import os

# ── Load data ──────────────────────────────────────────────────────────────
fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

data = img.get_fdata()
print(f"Full data shape: {data.shape}")  # (81, 106, 76, 160)

# ── Brain mask ─────────────────────────────────────────────────────────────
print("Computing brain mask...")
b0_mask, mask = median_otsu(data, vol_idx=range(0, 10), numpass=1)
print(f"Mask shape: {mask.shape}")

# ── Simulate undersampling (keep only 12 directions) ───────────────────────
print("Simulating undersampling...")

# Get indices of non-b0 volumes (actual diffusion directions)
dwi_indices = np.where(gtab.bvals > 0)[0]

# Randomly pick 12 directions (fix seed for reproducibility)
np.random.seed(42)
selected_indices = np.sort(np.random.choice(dwi_indices, 12, replace=False))

# Also keep the b0 volumes (needed for signal normalization)
b0_indices = np.where(gtab.bvals == 0)[0]
sparse_indices = np.concatenate([b0_indices, selected_indices])
sparse_indices = np.sort(sparse_indices)

# Create sparse data and gradient table
sparse_data = data[..., sparse_indices]
sparse_bvals = gtab.bvals[sparse_indices]
sparse_bvecs = gtab.bvecs[sparse_indices]
sparse_gtab = gradient_table(sparse_bvals, sparse_bvecs)

print(f"Sparse data shape: {sparse_data.shape}")  # should be (81,106,76, ~22)

# ── Fit DTI on FULL data → ground truth ────────────────────────────────────
print("Fitting DTI on full data (ground truth)...")
tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data, mask=mask)

FA_full = tenfit.fa
MD_full = tenfit.md

print(f"FA map shape: {FA_full.shape}")
print(f"FA range: {FA_full.min():.3f} – {FA_full.max():.3f}")
print(f"MD range: {MD_full.min():.6f} – {MD_full.max():.6f}")

# ── Save everything ────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/sparse_data.npy", sparse_data)
np.save("data/processed/sparse_bvals.npy", sparse_bvals)
np.save("data/processed/sparse_bvecs.npy", sparse_bvecs)
np.save("data/processed/full_data.npy", data)
np.save("data/processed/FA_ground_truth.npy", FA_full)
np.save("data/processed/MD_ground_truth.npy", MD_full)
np.save("data/processed/mask.npy", mask)

print("\n✅ All files saved to data/processed/")
print("Files saved:")
for f in os.listdir("data/processed"):
    size = os.path.getsize(f"data/processed/{f}") / 1e6
    print(f"  {f:35s} {size:.1f} MB")