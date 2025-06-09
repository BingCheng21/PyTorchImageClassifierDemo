# 🧪 Fly Wing Vein Section Analyzer
# This notebook segments veins and calculates the area of enclosed sections

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label
import collections
import cv2

# === Step 1: Load or simulate a fly wing image ===
# Replace this with actual image loading if you have real data
# image = np.zeros((256, 256), dtype=np.uint8)

# # Simulate veins using lines
# cv2.line(image, (50, 0), (50, 255), 255, 3)
# cv2.line(image, (0, 50), (255, 50), 255, 3)
# cv2.line(image, (100, 0), (200, 255), 255, 2)
# cv2.line(image, (0, 100), (255, 200), 255, 2)

image = Image.open("D:\\temp\\very-closeup-view-parts-flys-wing-was-taken-with-digital-microscope_1111864-77.jpg")

# Show the simulated image
plt.imshow(image, cmap="gray")
plt.title("Simulated Fly Wing Veins")
plt.show()

# === Step 2: Simulate a model output mask ===
# In practice, use your PyTorch model output here
vein_mask = torch.from_numpy(image > 127).float().unsqueeze(0)  # (1, H, W)

# === Step 3: Invert mask to highlight enclosed areas ===
non_vein_mask = 1 - vein_mask

# === Step 4: Label each enclosed section ===
labeled_array, num_sections = label(non_vein_mask.squeeze().numpy())
print(f"🧩 Number of enclosed sections: {num_sections}")

# === Step 5: Calculate the area of each region ===
area_counts = collections.Counter(labeled_array.flatten())
if 0 in area_counts:
    del area_counts[0]

# Convert area to mm² (optional)
PIXEL_TO_MM2 = 0.01 ** 2  # 0.01 mm per pixel
real_area_mm2 = {
    region: pixels * PIXEL_TO_MM2 for region, pixels in area_counts.items()
}

# === Step 6: Visualize everything ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original (Simulated) Vein Image")

plt.subplot(1, 3, 2)
plt.imshow(non_vein_mask.squeeze(), cmap="gray")
plt.title("Inverted Mask (Enclosed Regions)")

plt.subplot(1, 3, 3)
plt.imshow(labeled_array, cmap="nipy_spectral")
plt.title(f"Labeled Sections")
plt.colorbar()
plt.tight_layout()
plt.show()

# === Step 7: Print area results ===
print("📏 Section Area Report:")
for region_id, pixel_area in area_counts.items():
    print(f"  - Region {region_id:2}: {pixel_area:4} pixels = {real_area_mm2[region_id]:.4f} mm²")
