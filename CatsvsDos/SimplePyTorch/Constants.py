# Define containts

import torch

# Path to the folder when the model is stored
MODEL_FOLDER = "models/cnn"
# Model file name
MODEL_FILE_NAME = "model.pth"
# Class file anme
CLASS_FILE_NAME = 'classname.json'
# Image size
IMAGE_SIZE = 128
# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
