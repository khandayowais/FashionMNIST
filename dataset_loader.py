import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import load_mnist


class MNISTDATASET(Dataset):
    
    def __init__(self, PATH, kind, transform=None):
        self.images, self.labels = load_mnist(PATH, kind)        
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index): 
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        
        return image, label
            
        