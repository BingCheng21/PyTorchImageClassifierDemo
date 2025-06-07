import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import os
import json

# --- CONFIG ---
# Path to the folder where the model is stored
MODEL_FOLDER = "models"
# Model file name
MODEL_FILE_NAME = "model.pth"
# Class file anme
CLASS_FILE_NAME = 'classname.json'
# Image size
IMAGE_SIZE = 128
# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Path to the folder where the data files are stroed
DATA_FOLDER = "../data"
BATCH_SIZE = 16
NUM_CLASSES = 2


# --- MODEL DEFINITION ---
# Create a new SimpleCNN class inheriting the nn.Module class
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --- DATA LOADING ---
def get_data_loaders():
    transform = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    }

    # Relative path doesn't work well when debugging the code in VS Code 
    # Take the absolute path of the script file and use it as the base path.
    BASE_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

    datasets_ = {
        x: datasets.ImageFolder(os.path.join(BASE_FOLDER_PATH, DATA_FOLDER, x), transform=transform[x])
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
def save_model(model, class_names):

    # Relative path doesn't work well when debugging the code in VS Code 
    # Take the absolute path of the script file and use it as the base path.
    BASE_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

    # Save the model statistics
    # This saves just the model parameters, not the entire architecture.
    torch.save(model.state_dict(), os.path.join(BASE_FOLDER_PATH, MODEL_FOLDER, MODEL_FILE_NAME))

    # Save the class name
    # While training the model, the folders are 
    # data/train/cats/
    # data/train/dogs/
    # Then ImageFolder will assign: 'cats' → class 0, 'dogs' → class 1
    # And class_names = ['cats', 'dogs'].
    # So this line in predict_image: print(f"Prediction: {class_names[pred.item()]}"), it will show cats, instead of 0
    # In order to use the model later and be able to show cats and dogs, we are saving the class names
    with open( os.path.join(BASE_FOLDER_PATH, MODEL_FOLDER, CLASS_FILE_NAME), "w") as f:
        json.dump(class_names, f)


def load_model():

    BASE_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

    # Load the model
    model = SimpleCNN()
    model.load_state_dict(torch.load(os.path.join(BASE_FOLDER_PATH, MODEL_FOLDER, MODEL_FILE_NAME), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load the class names
    with open("class_names.json") as f:
        class_names = json.load(f)

    return model, class_names

