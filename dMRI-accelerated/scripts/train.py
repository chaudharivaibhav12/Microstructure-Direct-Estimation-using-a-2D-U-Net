import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys, os
sys.path.append(os.path.dirname(__file__))

from unet import UNet
from dataset import dMRIDataset

# ── Config ─────────────────────────────────────────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 4
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
sparse_data = np.load("data/processed/sparse_data.npy")
FA          = np.load("data/processed/FA_ground_truth.npy")
MD          = np.load("data/processed/MD_ground_truth.npy")
mask        = np.load("data/processed/mask.npy")

# ── Dataset & splits ───────────────────────────────────────────────────────
dataset = dMRIDataset(sparse_data, FA, MD, mask)

train_size = int(0.8 * len(dataset))  # 60 slices for training
val_size   = len(dataset) - train_size  # 16 slices for validation

train_ds, val_ds = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train slices: {train_size} | Val slices: {val_size}")

# ── Model, optimizer, loss ─────────────────────────────────────────────────
model     = UNet(in_channels=22, out_channels=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)
criterion = nn.MSELoss()

# ── Training loop ──────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
best_val_loss = float("inf")
history = {"train": [], "val": []}

for epoch in range(1, EPOCHS + 1):

    # — Train —
    model.train()
    train_loss = 0.0
    for x, target, mask_batch in train_loader:
        x      = x.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # — Validate —
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, target, mask_batch in val_loader:
            x      = x.to(DEVICE)
            target = target.to(DEVICE)
            pred   = model(x)
            val_loss += criterion(pred, target).item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    history["train"].append(train_loss)
    history["val"].append(val_loss)

    # — Save best model —
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_unet.pth")
        saved = "  ✓ saved"
    else:
        saved = ""

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f}{saved}")

# ── Save training history ──────────────────────────────────────────────────
np.save("models/history.npy", history)
print(f"\n✅ Training complete! Best val loss: {best_val_loss:.6f}")
print("Model saved to models/best_unet.pth")