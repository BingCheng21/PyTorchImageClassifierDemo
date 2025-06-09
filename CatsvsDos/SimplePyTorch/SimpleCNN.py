import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import os
import json



# --- MODEL DEFINITION ---
# Create a new SimpleCNN class inheriting the nn.Module class
class SimpleCNN(nn.Module):

    # Model folder Name
    __model_folder = "models"

    # Model file name
    __model_file_name = "model.pth"
    # Class file anme
    __class_name_file_name = 'classname.json'
    # Image size
    __image_size = 128    # Device
    __device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    __batch_size = 16
    __num_classes = 2

    __class_names = None

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
            nn.Linear(32 * (self.__image_size // 4) * (self.__image_size // 4), 64),
            nn.ReLU(),
            nn.Linear(64, self.__num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


    def train_model(self, data_folder, epochs=3):
        
        loaders, class_names = self.__get_data_loaders(data_folder)

        self.__class_names = class_names

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.to(self.__device)
        self.train()

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


    # Prefixing the function name with __ to make this function private
    def __get_data_loaders(self, data_folder):

        transform = {
            'train': transforms.Compose([
                transforms.Resize((self.__image_size, self.__image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.__image_size, self.__image_size)),
                transforms.ToTensor(),
            ])
        }

        # # Relative path doesn't work well when debugging the code in VS Code 
        # # Take the absolute path of the script file and use it as the base path.
        # BASE_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

        datasets_ = {
            x: datasets.ImageFolder(os.path.join(data_folder, x), transform=transform[x])
            for x in ['train', 'val']
        }

        loaders = {
            x: torch.utils.data.DataLoader(datasets_[x], batch_size=self.__batch_size, shuffle=True)
            for x in ['train', 'val']
        }

        return loaders, datasets_['train'].classes


    # --- EVALUATE FUNCTION ---
    def __evaluate_model(self, loaders):

        self.eval()
        correct = total = 0

        with torch.no_grad():
            for inputs, labels in loaders:
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        return acc


    # --- SAVE / LOAD ---
    def save_model(self):

        # # Relative path doesn't work well when debugging the code in VS Code 
        # # Take the absolute path of the script file and use it as the base path.
        BASE_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

        # Save the model statistics
        # This saves just the model parameters, not the entire architecture.
        torch.save(self.state_dict(), os.path.join(BASE_FOLDER_PATH, self.__model_folder, self.__model_file_name))

        # Save the class name
        # While training the model, the folders are 
        # data/train/cats/
        # data/train/dogs/
        # Then ImageFolder will assign: 'cats' → class 0, 'dogs' → class 1
        # And class_names = ['cats', 'dogs'].
        # So this line in predict_image: print(f"Prediction: {class_names[pred.item()]}"), it will show cats, instead of 0
        # In order to use the model later and be able to show cats and dogs, we are saving the class names
        with open( os.path.join(BASE_FOLDER_PATH, self.__model_folder, self.__class_name_file_name), "w") as f:
            json.dump(self.__class_names, f)


    def load_model(self):

        BASE_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

        # Load the model
        self.load_state_dict(torch.load(os.path.join(BASE_FOLDER_PATH, self.__model_folder, self.__model_file_name), map_location=self.__device))
        self.to(self.__device)
        self.eval()

        # Load the class names
        with open(os.path.join(BASE_FOLDER_PATH, self.__model_folder, self.__class_name_file_name)) as f:
            self.__class_names = json.load(f)

    def predict_image(self, image_path):

        transform = transforms.Compose([
                transforms.Resize((self.__image_size, self.__image_size)),
                transforms.ToTensor(),
            ])

        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.__device)
        output = self(image)
        _, predicted = torch.max(output, 1)
        return self.__class_names[predicted.item()]


# Every Python file has a built-in variable called __name__.
# When you run a file directly, Python sets __name__ to "__main__".
# When you import that file as a module, __name__ is set to the module’s name instead.

# Only run the following block of code if this file is being run directly, not if it’s being imported as a module into another file.

if __name__ == "__main__":
    
    model = SimpleCNN()

    # Get the data folder
    # # Relative path doesn't work well when debugging the code in VS Code 
    # # Take the absolute path of the script file and use it as the base path.
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

    model.train_model(data_folder)

    # Save the mode
    model.save_model()