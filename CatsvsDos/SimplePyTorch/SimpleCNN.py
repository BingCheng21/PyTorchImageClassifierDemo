import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import os
import json

# SimpleCNN: A basic Convolutional Neural Network for image classification (cats vs dogs)
class SimpleCNN(nn.Module):
    # Model folder Name
    __model_folder = "models"
    # Model file name
    __model_file_name = "model.pth"
    # Class file name
    __class_name_file_name = 'classname.json'
    # Image size (input images will be resized to this size)
    __image_size = 128
    # Device (GPU if available, else CPU)
    __device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Batch size for training and validation
    __batch_size = 16
    # Number of output classes (cats, dogs)
    __num_classes = 2
    # List of class names (populated after loading data)
    __class_names = None

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Feature extractor: 2 convolutional layers with ReLU and MaxPool
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Input: 3 channels (RGB), 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 16 input channels, 32 output channels
            nn.ReLU(),
            nn.MaxPool2d(2)   # Downsample by 2 again
        )
        # Classifier: Flatten, Linear, ReLU, Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # The input size is 32 * (image_size // 4) * (image_size // 4) after two MaxPool2d(2)
            nn.Linear(32 * (self.__image_size // 4) * (self.__image_size // 4), 64),
            nn.ReLU(),
            nn.Linear(64, self.__num_classes)  # Output: 2 classes
        )

    def forward(self, x):
        # Forward pass: feature extraction then classification
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_model(self, data_folder, epochs=3):
        # Train the model on the dataset in data_folder for a given number of epochs
        loaders, class_names = self.__get_data_loaders(data_folder)
        self.__class_names = class_names
        criterion = nn.CrossEntropyLoss()  # Loss function for classification
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer
        self.to(self.__device)  # Move model to device
        self.train()  # Set model to training mode
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in loaders['train']:
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
        # Evaluate model after training
        self.__evaluate_model(loaders)

    def __get_data_loaders(self, data_folder):
        # Prepare data loaders for training and validation
        transform = {
            'train': transforms.Compose([
                transforms.Resize((self.__image_size, self.__image_size)),  # Resize images
                transforms.RandomHorizontalFlip(),  # Data augmentation
                transforms.ToTensor(),  # Convert to tensor
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.__image_size, self.__image_size)),
                transforms.ToTensor(),
            ])
        }
        # Load datasets from folders (expects 'train' and 'val' subfolders)
        datasets_ = {
            x: datasets.ImageFolder(os.path.join(data_folder, x), transform=transform[x])
            for x in ['train', 'val']
        }
        # Create data loaders for batch processing
        loaders = {
            x: torch.utils.data.DataLoader(datasets_[x], batch_size=self.__batch_size, shuffle=True)
            for x in ['train', 'val']
        }
        return loaders, datasets_['train'].classes  # Return loaders and class names

    def __evaluate_model(self, loaders):
        # Evaluate the model on the validation set
        self.eval()  # Set model to evaluation mode
        correct = total = 0
        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in loaders['val']:
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        return acc

    def save_model(self):
        # Save the model parameters and class names to disk
        BASE_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
        # Save model weights (not the full model)
        torch.save(self.state_dict(), os.path.join(BASE_FOLDER_PATH, self.__model_folder, self.__model_file_name))
        # Save class names for later use (for prediction)
        with open( os.path.join(BASE_FOLDER_PATH, self.__model_folder, self.__class_name_file_name), "w") as f:
            json.dump(self.__class_names, f)

    def load_model(self):
        # Load the model parameters and class names from disk
        BASE_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
        # Load model weights
        self.load_state_dict(torch.load(os.path.join(BASE_FOLDER_PATH, self.__model_folder, self.__model_file_name), map_location=self.__device))
        self.to(self.__device)
        self.eval()
        # Load class names
        with open(os.path.join(BASE_FOLDER_PATH, self.__model_folder, self.__class_name_file_name)) as f:
            self.__class_names = json.load(f)

    def predict_image(self, image_path):
        # Predict the class of a single image file
        transform = transforms.Compose([
                transforms.Resize((self.__image_size, self.__image_size)),
                transforms.ToTensor(),
            ])
        image = Image.open(image_path).convert('RGB')  # Open and convert image to RGB
        image = transform(image).unsqueeze(0).to(self.__device)  # Preprocess and add batch dimension
        output = self(image)
        _, predicted = torch.max(output, 1)
        return self.__class_names[predicted.item()]  # Return class name

# Only run the following block if this file is executed directly (not imported)
if __name__ == "__main__":
    model = SimpleCNN()
    # Get the data folder (absolute path)
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    model.train_model(data_folder)
    # Save the model after training
    model.save_model()