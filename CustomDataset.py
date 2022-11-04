import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from helpers import toTensor

# A dataset class that can be used by PyTorch data loaders
class CustomDataset(Dataset):

    def __init__(self, X, Y, num_classes=1, aug_transform=None):

        self.X = X
        # Convert labels to a tensor for training
        # self.Y = nn.functional.one_hot(toTensor(Y).long(), num_classes=num_classes)
        self.Y = toTensor(Y).long()

        X, Y = None, None

        self.xi_preprocessing_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                          ])
        self.yi_preprocessing_transform = None
        self.aug_transform = aug_transform
        if self.aug_transform is not None:
            print("[WARNING]: Dataset loaded with augmentations set, make sure this is training only")

        self.nitems = self.X.shape[0]

        self.xi = None
        self.yi = None

    def __getitem__(self, index):

        X, Y = None, None # Avoiding accidently using X and Y instead of self.X

        xi = self.X[index]
        yi = self.Y[index]

        xi_final = xi
        yi_final = yi

        seed = random.randrange(sys.maxsize) #get a random seed so that we can reproducibly do the transformations

        # First augment the data, if we have some augmentations to apply
        if self.aug_transform is not None:
            random.seed(seed)

            trans_dict = self.aug_transform(image=xi)
            # Apply to the images before doing standard transforms
            xi = trans_dict["image"]

        # Convert examples to a tensor
        if self.xi_preprocessing_transform is not None:
            random.seed(seed) # apply this seed to img transforms
            xi_final = self.xi_preprocessing_transform(xi)

        return xi_final,  yi_final

    def __len__(self):
        return self.nitems
