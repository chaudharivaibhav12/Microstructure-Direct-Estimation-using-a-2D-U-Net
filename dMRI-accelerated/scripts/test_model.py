import torch
from unet import UNet

model = UNet(in_channels=22, out_channels=2)
model.eval()

# Dummy input: batch of 2 slices, 22 channels, 81x106
x = torch.randn(2, 22, 81, 106)
with torch.no_grad():
    out = model(x)

print(f"Input shape:  {x.shape}")    # (2, 22, 81, 106)
print(f"Output shape: {out.shape}")  # (2, 2, 81, 106)
print(f"Output range: {out.min():.3f} – {out.max():.3f}")  # should be 0–1
print("✅ Model working correctly!")