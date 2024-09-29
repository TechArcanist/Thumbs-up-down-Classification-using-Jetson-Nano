# Thumbs-up-down Classification on Jetson Nano and Windows 👍👎

<h1 align="center">
    <img src="https://readme-typing-svg.herokuapp.com/?font=Righteous&size=35&center=true&vCenter=true&width=700&height=100&duration=4000&lines=Thumbs-up+down+Classification!+🤖;" />
</h1>

## Overview

Want to create your own deep learning model for recognizing gestures like thumbs up and thumbs down? You can make modifications to the code in the future to accommodate various other hand gestures. Additionally, you can use these hand gestures to connect with hardware components, enabling gesture-controlled robotics, and expanding the system for more complex and interactive robotic control.

## Follow the Instructions Below

### Step 1: Install Required Libraries

1. Open Jupyter Lab notebook on your desktop.
2. Paste the following code in the notebook cell to install all the libraries:

```python
import sys
import subprocess
packages = [
    "opencv-python",
    "torch",
    "torchvision",
    "ipywidgets",
    "Pillow",  
    "numpy",
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

print("All packages are installed and imported successfully!")
```

3. Paste the code below in another cell to generate all the directories:

```python
import os
import json
# this function will create all directories, files, and folders
def create_project_structure(base_path='project_root'):
    project_structure = {
        "src": {
            "dataset.py": None,
            "utils.py": None,
        },
        "thumbs_A": {
            "thumbs_up": {},
            "thumbs_down": {},
        },
        "thumbs_B": {
            "thumbs_up": {},
            "thumbs_down": {},
        },
        "training.ipynb": None,
        "testing.ipynb": None,
    }

    # function to create files and directories
    def create(path, structure):
        if isinstance(structure, dict):
            os.makedirs(path, exist_ok=True)
            for key, value in structure.items():
                create(os.path.join(path, key), value)
        elif structure is None:
            # create an empty file
            with open(path, 'w') as f:
                # creating a basic JSON structure for a Jupyter notebook
                notebook_content = {
                    "cells": [],
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3"
                        },
                        "language_info": {
                            "codemirror_mode": {
                                "name": "ipython",
                                "version": 3
                            },
                            "file_extension": ".py",
                            "mimetype": "text/x-python",
                            "name": "python",
                            "nbconvert_exporter": "python",
                            "pygments_lexer": "ipython3",
                            "version": "3.8.5"
                        }
                    },
                    "nbformat": 4,
                    "nbformat_minor": 4
                }
                json.dump(notebook_content, f)

    # creating the project structure
    create(base_path, project_structure)

    # saving the code to dataset.py
    dataset_code = """
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class ImageClassificationDataset(Dataset):
    def __init__(self, dataset_name, categories, transform=None):
        self.dataset_name = dataset_name
        self.categories = categories
        self.transform = transform

        for category in self.categories:
            category_dir = os.path.join(self.dataset_name, category)
            os.makedirs(category_dir, exist_ok=True)

        self.category_to_idx = {category: idx for idx, category in enumerate(self.categories)}
        self.idx_to_category = {idx: category for category, idx in self.category_to_idx.items()}

    def get_count(self, category):
        category_dir = os.path.join(self.dataset_name, category)
        return len([f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def save_entry(self, image, category):
        if category not in self.categories:
            raise ValueError(f"Category '{category}' is not defined in the dataset categories.")

        category_dir = os.path.join(self.dataset_name, category)
        image_count = self.get_count(category)
        image_filename = f'{image_count:04d}.jpg'
        image_path = os.path.join(category_dir, image_filename)

        image_rgb = image[:, :, ::-1]
        image_pil = Image.fromarray(image_rgb)
        image_pil.save(image_path)
        print(f'Saved image: {image_path}')

    def __len__(self):
        total = 0
        for category in self.categories:
            total += self.get_count(category)
        return total

    def __getitem__(self, idx):
        cumulative = 0
        for category in self.categories:
            count = self.get_count(category)
            if idx < cumulative + count:
                image_idx = idx - cumulative
                category_dir = os.path.join(self.dataset_name, category)
                image_filename = f'{image_idx:04d}.jpg'
                image_path = os.path.join(category_dir, image_filename)
                label = self.category_to_idx[category]
                break
            cumulative += count
        else:
            raise IndexError("Index out of range")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    """
    with open(os.path.join(base_path, 'src', 'dataset.py'), 'w') as f:
        f.write(dataset_code.strip())

    # save the code to utils.py
    utils_code = """
import cv2
import torchvision.transforms as transforms

def preprocess(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(image).unsqueeze(0)
    """
    with open(os.path.join(base_path, 'src', 'utils.py'), 'w') as f:
        f.write(utils_code.strip())

create_project_structure()
```

### Step 2: Navigate to Project Folder

1. Close this notebook.
2. Now, on Jupyter Home, go to your desktop and open the folder that was created named `project_root`.
3. Open the training notebook.

![image](https://github.com/user-attachments/assets/20122d65-1d4d-4b1e-9538-3836b729f89b)

Now click the link below to open my notebook on this repo, copy the code of each cell, and paste it in the same order in your training notebook. Execute them one by one and follow all the instructions written in that notebook to create the dataset and train the model.

**Link:** [[Training Notebook](https://github.com/TechArcanist/Thumbs-up-down-Classification-using-Jetson-Nano/blob/main/training.ipynb)]

### Step 3: Test Your Model

Now that your model is saved, open your testing notebook and click on the link below to access my notebook. Paste the code in your testing notebook and congratulations, your model is ready to be tested!

**Link:** [Testing Notebook](#)

---

## Developed by

**Lavitra Sahu**  
Feel free to reach out on linkedin or instagram if you have any questions or need assistance!

---

<h1 align="center">
    <img src="https://readme-typing-svg.herokuapp.com/?font=Righteous&size=35&center=true&vCenter=true&width=500&height=70&duration=4000&lines=Thanks+for+Visiting!+👋;" />
</h1> 
