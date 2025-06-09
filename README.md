# PyTorch Image Classifier Demo

This is a sample project to use PyTorch to train a model to classify dog vs. cat photos using a small dataset. This is a minimal working example that can be run locally and build on.

## Getting the training images

I downloaded the traing images [here](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/code)

## Set up the Python virtual environment

The venv module supports creating lightweight “virtual environments”, each with their own independent set of Python packages installed in their site directories. A virtual environment is created on top of an existing Python installation, known as the virtual environment’s “base” Python, and may optionally be isolated from the packages in the base environment, so only those explicitly installed in the virtual environment are available.

In my Git work folder, run this command to create the virtual environment for this repo

```python
python -m venv .venv
```

This will add a .venv folder in my git repo. Note that .gitignore file should have excluded .venv folder.

To activate the virtual environment, run commmand

```powershell
.venv/scripts/activate.ps1
```

## Install the required libraries

Run this command in the python virtual environment to install the required libraries

```python
pip install torch torchvision matplotlib ipykernel
```

## Upgrade pip in the virtual environment

Run this command

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
```

## Support import functions from one notebook to another

First, you need the import-ipynb library. Make suer that you are running the virtual python enviornment. Then run this command

```powershell
pip install import-ipynb
```

To import the notebook, add these lines

```python
import import-ipynb
from SimplCNN import SimpleCNN
```

## List all required Python libraries and install them with pip

1. Create a plain text file named requirements.txt and list each package you need ona  new line. Here is an example:

```txt
torch
torchvision
numpy
matplotlib
opencv-python
scikit-learn
```

You can also specify versions if needed:

```txt
torch==2.2.0
opencv-python>=4.8.0
```

If you already installed your packages and want to save your environment, run this command

```python
pip freeze > requirement.txt
```

2. Run this command in the same directory as requirements.txt

```python
pip install -r requirements.txt
``` 