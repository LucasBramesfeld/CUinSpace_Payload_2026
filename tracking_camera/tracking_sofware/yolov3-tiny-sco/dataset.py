import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd

class Dataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_size=640, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load labels
        label_file = os.path.join(self.labels_dir, self.image_files[idx].replace(".jpg", ".txt"))
        box = None
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                line = f.readline()
                cls, x, y, w, h = map(float, line.strip().split()[0:5])
                box = [x, y, w, h]
        box = torch.tensor(box) if box else torch.zeros(4)
        
        return image, box
