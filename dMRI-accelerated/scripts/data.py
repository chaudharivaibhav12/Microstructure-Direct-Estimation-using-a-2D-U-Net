from dipy.data import fetch_stanford_hardi, read_stanford_hardi

# Downloads ~80MB automatically
fetch_stanford_hardi()

# Load it
img, gtab = read_stanford_hardi()

print("Data shape:", img.shape)        # (81, 106, 76, 160) — 160 directions
print("b-values:", gtab.bvals[:10])    # shows b-values
print("b-vectors shape:", gtab.bvecs.shape)