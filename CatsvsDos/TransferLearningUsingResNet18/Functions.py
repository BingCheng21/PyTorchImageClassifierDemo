import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

# --- CONFIG ---
# Path to the folder when the model is stored
MODEL_FOLDER = "models"
# Model file name
MODEL_FILE_NAME = "model.pth"
# Class file anme
CLASS_FILE_NAME = 'classname.json'
# Image size
IMAGE_SIZE = 224
# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Path to the folder where the data files are stroed
DATA_FOLDER = "../data"
BATCH_SIZE = 16
NUM_CLASSES = 2


# --- DATA LOADING ---
def get_data_loaders():
    transform = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Required by ResNet
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Required by ResNet
        ])
    }



    datasets_ = {
        x: datasets.ImageFolder(os.path.join(DATA_FOLDER, x), transform=transform[x])
        for x in ['train', 'val']
    }

    loaders = {
        x: torch.utils.data.DataLoader(datasets_[x], batch_size=BATCH_SIZE, shuffle=True)
        for x in ['train', 'val']
    }

    return loaders, datasets_['train'].classes


# --- TRAIN FUNCTION ---
def train_model(model, loaders, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(DEVICE)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loaders['train']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")


# --- EVALUATE FUNCTION ---
def evaluate_model(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc


# --- SAVE / LOAD ---
def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)


def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

