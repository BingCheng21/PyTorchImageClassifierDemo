Here’s a complete **starter project** for segmenting fly wing veins using **U-Net in PyTorch**. It includes:

* ✅ Dataset loader (for your wing images and masks)
* ✅ U-Net model (lightweight version)
* ✅ Training loop
* ✅ Inference + visualization
* ✅ Integration point for area calculation (like you’re already doing)

---

## 📁 Folder Structure

```
fly_wing_vein_project/
├── dataset/
│   ├── images/
│   │   └── wing1.png
│   └── masks/
│       └── wing1_mask.png
├── unet_model.py
├── train.py
├── predict_and_analyze.py
└── requirements.txt
```

---

## 📦 1. `requirements.txt`

```txt
torch
torchvision
numpy
matplotlib
Pillow
scipy
opencv-python
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🧠 2. `unet_model.py` — U-Net Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.middle = CBR(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.middle(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))
```

---

## 🏋️ 3. `train.py` — Training the Model

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from unet_model import UNet

class WingVeinDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, self.images[idx])).convert("L")

        return self.transform(image), self.transform(mask)

# === Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = WingVeinDataset("dataset/images", "dataset/masks")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()

for epoch in range(10):
    model.train()
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        output = model(img)
        loss = criterion(output, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "unet_fly.pth")
```

---

## 🔍 4. `predict_and_analyze.py` — Inference & Vein Section Area

This is where you integrate the vein mask → region labeling → area analysis — you already have this in your working code.

```python
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

# Threshold and analyze
vein_mask = (output > 0.5).astype(np.uint8)
non_vein = 1 - vein_mask
labeled, num = label(non_vein)
counts = collections.Counter(labeled.flatten())
if 0 in counts:
    del counts[0]

# Show result
plt.imshow(labeled, cmap="nipy_spectral")
plt.title(f"Detected {num} Sections")
plt.show()

for k, v in counts.items():
    print(f"Region {k}: {v} pixels")
```

---

Would you like me to zip this starter project and send you a download link? Or paste this into a GitHub repo for you?
