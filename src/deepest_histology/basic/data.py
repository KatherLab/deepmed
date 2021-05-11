import torch
import numpy as np
from torchvision import transforms
from PIL import Image


class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, transform = None, target_patch_size = -1):
        'Initialization'
        self.labels = labels
        self.imgs = imgs
        self.target_patch_size = target_patch_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = Image.open(self.imgs[index])
        y = self.labels[index]
        if self.target_patch_size is  not None:
            X = X.resize((self.target_patch_size, self.target_patch_size))
            X = np.array(X)
        if self.transform is not None:
            X = self.transform(X)
        return X, y
