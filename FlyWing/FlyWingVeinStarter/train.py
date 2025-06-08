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
