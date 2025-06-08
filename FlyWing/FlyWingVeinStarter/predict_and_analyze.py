from unet_model import UNet
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label
import collections

model = UNet()
model.load_state_dict(torch.load("unet_fly.pth", map_location="cpu"))
model.eval()

img = Image.open("dataset/images/wing1.png").convert("L")
input_tensor = torch.tensor(np.array(img) / 255.0).float().unsqueeze(0).unsqueeze(0)
output = model(input_tensor).squeeze().detach().numpy()

vein_mask = (output > 0.5).astype(np.uint8)
non_vein = 1 - vein_mask
labeled, num = label(non_vein)
counts = collections.Counter(labeled.flatten())
if 0 in counts:
    del counts[0]

plt.imshow(labeled, cmap="nipy_spectral")
plt.title(f"Detected {num} Sections")
plt.show()

for k, v in counts.items():
    print(f"Region {k}: {v} pixels")
