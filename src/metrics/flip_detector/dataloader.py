import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


class FlippedMNIST(Dataset):
    def __init__(self, root="./data", train=True):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, _ = self.mnist[idx]

        flipped = idx % 2

        if flipped:
            image = TF.hflip(image)

        label = torch.tensor(
            float(flipped),
            dtype=torch.float32
        )

        return image, label